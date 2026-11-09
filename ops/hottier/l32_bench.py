"""Hot-expert tier experiment, step 2 (Spark, native image). KERNEL MICROBENCHMARK + correctness, one MoE layer.

GPU tier = the N hottest experts of model layer 32 (hotness from traced tokens disjoint from the test tokens), held
in the control checkpoint's own EXL3 trellis format (byte re-layout only), run by b12x's projection-tiered MCG W4A16
kernel with GLM's clamped SiLU. Cold routes get weight 0 and are aliased to the token's top hot expert (b12x's
mixed path has no expert-map support), so their CPU contribution is accounted separately.
Measures: correctness of the hot partial sum vs ExLlamaV3's own dequantisation and vs BF16 (per route), prepared
GPU bytes, per-call time over a token sweep with CUDA graphs, effective code bandwidth.
Usage: python3 l32_bench.py DATA_DIR OUT_JSON
"""
import inspect, json, os, sys, textwrap, time
import torch
from safetensors.torch import load_file
from b12x.moe import fused_moe as fm
from b12x.moe.fused_moe import _impl, _compat
from b12x.moe.fused_moe.config import TrellisConfig
from b12x.moe.fused_moe.weights import ScaleFactors, TrellisWeights
from b12x.moe._shared.kernels.w4a16 import mixed_trellis as _mt

D, OUTJ = sys.argv[1], sys.argv[2]
E, H, I, TOPK, SWIGLU = 288, 4096, 2048, 8, 10.0
NS = [int(v) for v in os.environ.get("HOT_NS", "256,216,160,128,96,64").split(",")]
SWEEP = [int(v) for v in os.environ.get("SWEEP", "1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192").split(",")]
dev = torch.device("cuda:0")
# build_projection_tiered_maps validates its slot namespace against 256 even for subsets; tier-local indices stay < 256.
_src = textwrap.dedent(inspect.getsource(_mt.build_projection_tiered_maps)).replace("value > 256", "value > 1024")
exec(compile(_src, _mt.__file__, "exec"), _mt.__dict__)

pk = load_file(f"{D}/l32_b12x.safetensors")
tk = load_file(f"{D}/l32_tokens.safetensors")
rc = json.load(open(f"{D}/l32_route_counts.json"))
K = (pk["rate"].long() & 0xF)                                     # [E,3]
sec = (H // 16) * 64 * K                                          # bytes per (expert, proj) per slot row
offs = torch.cat([torch.zeros(1, dtype=torch.long), sec.reshape(-1).cumsum(0)[:-1]]).reshape(E, 3)
total = sum(torch.tensor(v) for v in rc["counts"].values())
hot_order = torch.argsort(total, descending=True)
CFG = TrellisConfig.from_dict({
    "version": 2, "codebook": "mcg", "rate": {"granularity": "per_expert_projection"},
    "scale": {k: {"vectors": "per_expert", "gains": "none"} for k in ("input_scales", "intermediate_scales", "output_scales")},
    "transform": {"projection": {"kind": "scaled_hadamard", "block_size": 128}, "expert": {"kind": "none"}},
})

x = tk["x"].to(dev).bfloat16(); ids = tk["ids"].long(); w = tk["w"].float()
route_x, route_b = tk["route_exl3"].float(), tk["route_bf16"].float()   # weighted per-route contributions [T,8,H]
dom = tk["dom"].long(); T0 = x.shape[0]
rel = lambda a, b: float((a - b).norm() / b.norm())
res = {"label": "kernel microbenchmark (one MoE layer, b12x on GB10)", "hot_sets": {}}


def build_tier(hot):
    n = len(hot)
    rows = []
    for e in hot.tolist():
        for p in range(3):
            rows.append(pk["codes"][:, offs[e, p]:offs[e, p] + sec[e, p]])
    codes = torch.cat(rows, 1).to(dev).contiguous()
    weights = TrellisWeights(codes=codes, rate=pk["rate"][hot].to(dev).contiguous(),
                             input_scales=ScaleFactors(pk["in_s"][hot].to(dev).contiguous()),
                             intermediate_scales=ScaleFactors(pk["mid_s"][hot].to(dev).contiguous()),
                             output_scales=ScaleFactors(pk["out_s"][hot].to(dev).contiguous()))
    plan = fm.plan_weights(source=fm.TrellisSource(config=CFG),
                           activation=fm.ActivationSpec(mode="a16", nonlinearity="silu", io_dtype=torch.bfloat16,
                                                        swiglu_limit=SWIGLU),
                           geometry=fm.MoEGeometry(num_experts=n, hidden_size=H, intermediate_size=I))
    torch.cuda.synchronize(); base = torch.cuda.memory_allocated()
    prep = fm.prepare_weights(plan=plan, weights=weights, device=dev)
    torch.cuda.synchronize()
    code_bytes = codes.numel()
    del codes, weights; torch.cuda.empty_cache(); torch.cuda.synchronize()
    resident = torch.cuda.memory_allocated() - base + code_bytes   # prepared state after the source copy is freed
    return plan, prep, int(code_bytes), int(resident)


class Runner:
    def __init__(self, plan, prep):
        self.plan, self.prep, self.cache = plan, prep, {}

    def get(self, T):
        if T not in self.cache:
            caps = _compat.Caps(max_tokens=T, num_topk=TOPK, device=dev, weight_plan=self.plan._impl,
                                quant_mode="w4a16", decode_config=None, swiglu_limit=SWIGLU)
            sp = _impl.plan_tp_moe_scratch(caps)
            scratch = [torch.empty(s.shape, dtype=s.dtype, device=dev) for s in sp.scratch_specs()]
            self.cache[T] = (sp, scratch, torch.empty((T, H), dtype=torch.bfloat16, device=dev))
        return self.cache[T]

    def __call__(self, a, lid, lw):
        sp, scratch, out = self.get(a.shape[0])
        b = sp.bind(scratch=scratch, a=a, experts=self.prep._impl, topk_weights=lw, topk_ids=lid, output=out)
        fm.run(binding=b)
        return out


def local_routes(hot, gids, gw, distinct=False):
    """Cold routes -> weight 0. duplicate alias: reuse one of the token's hot experts (no extra GPU weight traffic,
    models GPU work of a hot/cold split; the small-M kernel mis-merges duplicate ids, so values are not used).
    distinct alias: an unused hot expert per cold slot (exact values, costs extra expert loads)."""
    loc = torch.full((E,), -1, dtype=torch.long); loc[hot] = torch.arange(len(hot))
    lid = loc[gids]; is_hot = lid >= 0
    if distinct:
        lid = lid.clone()
        for t in range(lid.shape[0]):
            used = set(lid[t][is_hot[t]].tolist()); spare = (j for j in range(len(hot)) if j not in used)
            for k in range(TOPK):
                if not is_hot[t, k]: lid[t, k] = next(spare)
    else:
        first = torch.where(is_hot, lid, torch.full_like(lid, 1 << 30)).min(1).values
        first = torch.where(first == (1 << 30), torch.zeros_like(first), first)
        lid = torch.where(is_hot, lid, first.unsqueeze(1).expand_as(lid))
    lw = torch.where(is_hot, gw, torch.zeros_like(gw))
    return lid.int().to(dev).contiguous(), lw.to(dev).contiguous(), is_hot


def tile(t, T):
    return t[:T] if T <= t.shape[0] else t.repeat((T + t.shape[0] - 1) // t.shape[0], *([1] * (t.ndim - 1)))[:T]


def time_call(run, a, lid, lw, iters):
    run(a, lid, lw); torch.cuda.synchronize()
    graph = None
    try:
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            run(a, lid, lw)
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run(a, lid, lw)
    except Exception as exc:                                 # eager fallback is reported
        graph = None; res.setdefault("graph_errors", []).append(repr(exc)[:200])
    fn = graph.replay if graph is not None else (lambda: run(a, lid, lw))
    for _ in range(3): fn()
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(True), torch.cuda.Event(True)
    st.record()
    for _ in range(iters): fn()
    en.record(); torch.cuda.synchronize()
    return st.elapsed_time(en) / iters, graph is not None


for N in NS:
    hot = hot_order[:N].sort().values
    t0 = time.time()
    plan, prep, code_bytes, resident = build_tier(hot)
    run = Runner(plan, prep)
    lid, lw, is_hot = local_routes(hot, ids, w)
    did, dw, _ = local_routes(hot, ids, w, distinct=True)
    r = {"N": N, "build_s": round(time.time() - t0, 1), "code_bytes": code_bytes, "resident_bytes": resident,
         "test_route_coverage": round(float(is_hot.float().mean()), 4),
         "test_route_coverage_by_domain": {d: round(float(is_hot[dom == i].float().mean()), 4)
                                           for i, d in enumerate(["knowledge_qa", "code", "math", "instruction_following", "multilingual"])},
         "tokens_with_no_cold_route": round(float(is_hot.all(1).float().mean()), 4),
         "cold_routes_per_token_mean": round(float((~is_hot).sum(1).float().mean()), 3)}
    # correctness on all test tokens, in chunks of 512 (prefill-sized calls) and token-by-token for 32 (decode path)
    y = torch.cat([run(x[i:i + 512], did[i:i + 512], dw[i:i + 512]).float().cpu() for i in range(0, T0, 512)])
    m = is_hot.unsqueeze(-1).float()
    hx, hb = (route_x * m).sum(1), (route_b * m).sum(1)
    full_b = route_b.sum(1)
    r["hot_partial_vs_exllamav3_rel"] = round(rel(y, hx), 5)
    r["hot_partial_vs_bf16_rel"] = round(rel(y, hb), 5)
    r["exllamav3_vs_bf16_hot_partial_rel"] = round(rel(hx, hb), 5)
    r["hot_partial_plus_exact_cold_vs_full_bf16_rel"] = round(rel(y + (route_b * (1 - m)).sum(1), full_b), 5)
    y1 = torch.cat([run(x[i:i + 1], did[i:i + 1], dw[i:i + 1]).float().cpu() for i in range(64)])
    r["T1_vs_exllamav3_rel"] = round(rel(y1, hx[:64]), 5)
    yd = torch.cat([run(x[i:i + 1], lid[i:i + 1], lw[i:i + 1]).float().cpu() for i in range(64)])
    r["T1_duplicate_alias_vs_exllamav3_rel"] = round(rel(yd, hx[:64]), 5)
    if os.environ.get("DIAG"):
        pt = lambda a, b: [round(float((a[i] - b[i]).norm() / b[i].norm()), 4) for i in range(a.shape[0])]
        r["diag_T1_per_token"] = pt(y1[:32], hx[:32])
        r["diag_chunk_per_token"] = pt(y[:32], hx[:32])
        r["diag_absmax_x"] = [round(float(x[i].float().abs().max()), 1) for i in range(32)]
        for Tt in (2, 4, 8, 16, 32, 64, 128, 256):
            yt = torch.cat([run(x[i:i + Tt], lid[i:i + Tt], lw[i:i + Tt]).float().cpu() for i in range(0, 512, Tt)])
            r[f"diag_T{Tt}_rel"] = round(rel(yt, hx[:512]), 5)
        os.environ["B12X_DIAG_DONE"] = "1"
    print(json.dumps(r), flush=True)
    sweep = []
    for T in SWEEP:
        a, li, lwt = tile(x, T), tile(lid, T).contiguous(), tile(lw, T).contiguous()
        ms, graphed = time_call(run, a, li, lwt, iters=max(5, min(200, 4000 // T)))
        uniq = torch.unique(li).cpu()
        need = int(sec[hot[uniq]].sum() * (I // 32))
        sweep.append({"T": T, "ms": round(ms, 4), "graph": graphed, "distinct_hot_experts": int(uniq.numel()),
                      "code_MB": round(need / 1e6, 1), "eff_GBps": round(need / ms / 1e6, 1),
                      "tok_per_s_layer": round(T / ms * 1e3, 1)})
        print(json.dumps(sweep[-1]), flush=True)
    r["sweep"] = sweep
    res["hot_sets"][N] = r
    del run, prep, plan; torch.cuda.empty_cache()
    json.dump(res, open(OUTJ, "w"), indent=1)
print("BENCH_DONE", flush=True)
