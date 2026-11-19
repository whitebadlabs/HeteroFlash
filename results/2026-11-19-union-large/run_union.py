# Union-aware routing eval on large MoE via vLLM (teacher-forced prompt-only forwards, full-vocab KL).
import argparse, json, os, time
import ulhook
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--tp", type=int, default=1); ap.add_argument("--ks", required=True)
    ap.add_argument("--ch-per-dom", type=int, default=2); ap.add_argument("--gpu-mem", type=float, default=0.80)
    ap.add_argument("--expect-layers", type=int, default=0)
    ap.add_argument("--ctx", type=int, default=2048); ap.add_argument("--max-chunks", type=int, default=0)
    a = ap.parse_args()
    W = "/srv/cache/heteroflash/union-large"
    CTX = a.ctx
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    man = json.load(open(f"{W}/data/q_eval_balanced_v1.manifest.json"))
    glm = Tokenizer.from_file(f"{W}/data/glm_tokenizer/tokenizer.json")
    gch = json.load(open(f"{W}/data/q_eval_balanced_v1.tokens.json"))["chunks"]
    raw = open(f"{W}/data/q_eval_balanced_v1.txt", newline="").read()
    starts = []
    for d in man["domains"]:
        pre = glm.decode(gch[d["chunks"][0]][:40]); pos = raw.find(pre); assert pos >= 0; starts.append(pos)
    starts.append(len(raw)); assert starts == sorted(starts)
    chunks = []
    for di, d in enumerate(man["domains"]):
        txt = raw[starts[di]:starts[di + 1]]
        if os.environ.get("HARMONY"):
            import re
            txt = re.sub(r"(?m)^User: ?", "<|end|><|start|>user<|message|>", txt)
            txt = re.sub(r"(?m)^Assistant: ?", "<|end|><|start|>assistant<|channel|>final<|message|>", txt)
            txt = re.sub(r"\s*<\|end\|>", "<|end|>", txt)
        q = tok(txt, add_special_tokens=False)["input_ids"]
        n = len(q) // CTX
        picks = [round(i * (n - 1) / max(1, a.ch_per_dom - 1)) for i in range(a.ch_per_dom)] if n >= a.ch_per_dom else list(range(n))
        for p_ in picks:
            chunks.append((d["domain"], p_, [int(x) for x in q[p_ * CTX:(p_ + 1) * CTX]]))
        print(d["domain"], "tokens", len(q), "avail", n, "picked", picks, flush=True)
    if a.max_chunks:
        chunks = chunks[: a.max_chunks]

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    cfg = json.load(open(os.path.join(a.model, "config.json")))
    extra = {"limit_mm_per_prompt": {"image": 0, "video": 0}} if "vision_config" in cfg else {}
    if os.environ.get("ATTN"): extra["attention_backend"] = os.environ["ATTN"]
    if os.environ.get("MOEB"): extra["moe_backend"] = os.environ["MOEB"]
    print("extra", extra, flush=True)
    t0 = time.time()
    llm = LLM(model=a.model, tensor_parallel_size=a.tp, enforce_eager=True, compilation_config={"mode": 0},
              enable_prefix_caching=False, max_model_len=CTX + 64, max_num_batched_tokens=CTX + 64, max_num_seqs=1,
              gpu_memory_utilization=a.gpu_mem, seed=0, **extra)
    print("load s", time.time() - t0, flush=True)
    info = llm.collective_rpc(ulhook.install)
    print("install", info, flush=True)
    tc = cfg.get("text_config", cfg)
    TOPK = int(tc.get("num_experts_per_tok") or tc.get("experts_per_token"))
    print("TOPK from config", TOPK, flush=True)
    sp = SamplingParams(max_tokens=1, temperature=0.0)

    def rpc_set(**kw): llm.collective_rpc(ulhook.set_state, kwargs=kw)
    def run(ids, capture, route, K=None, union=False):
        rpc_set(T=len(ids), capture=capture, route=route, K=K, union=union, reset_stats=True)
        llm.generate([TokensPrompt(prompt_token_ids=ids)], sp, use_tqdm=False)
        outs = [o for o in llm.collective_rpc(ulhook.get_out) if o is not None]
        rpc_set(route=False, capture=None)
        drv = [o for o in outs if o["kind"] == "ref" and o["n"] > 0] or [o for o in outs if o.get("kl") is not None]
        assert drv, outs
        return drv[0]

    variants = [("hookref", TOPK, False)]
    for K in [int(k) for k in a.ks.split(",")]:
        variants += [(f"k{K}", K, False), (f"k{K}u", K, True)]
    res = {v[0]: {"kl_sum": 0.0, "top1": 0, "n": 0, "dom": {}, "per_chunk": [], "stats": [0.0] * 8, "calls": 0} for v in variants}
    checks = {}
    t0 = time.time()
    for ci, (dom, pidx, ids) in enumerate(chunks):
        if os.environ.get("REFONLY"):
            rpc_set(ids=ids)
            r = run(ids, "ref", False)
            print(f"REF {ci} {dom}#{pidx} nll={r.get(chr(110)+chr(108)+chr(108))} ent={r.get(chr(101)+chr(110)+chr(116))}", flush=True)
            continue
        rpc_set(ids=ids)
        r = run(ids, "ref", False)
        print(f"REF {ci} {dom}#{pidx} nll={r.get(chr(110)+chr(108)+chr(108))} ent={r.get(chr(101)+chr(110)+chr(116))}", flush=True)
        checks.setdefault("ref_nll", []).append([ci, dom, r.get("nll"), r.get("ent")])
        if ci == 0:
            o = run(ids, "var", False)  # unhooked rerun vs ref: determinism floor
            checks["unhooked_rerun"] = dict(meankl=sum(o["kl"]) / len(o["kl"]), maxd=o["maxd"], agree=o["agree"] / len(ids))
            print("CHECK unhooked rerun", checks["unhooked_rerun"], flush=True)
            for j in range(int(os.environ.get("DET", "0"))):
                o = run(ids, "var", False)
                print("DET rerun", j, sum(o["kl"]) / len(o["kl"]), o["maxd"], flush=True)
            if os.environ.get("DET"):
                r = run(ids, "ref", False); o = run(ids, "var", False)
                print("DET after re-ref", sum(o["kl"]) / len(o["kl"]), o["maxd"], flush=True)
        for name, K, un in variants:
            o = run(ids, "var", True, K, un)
            kl = o["kl"]; rr = res[name]
            rr["kl_sum"] += sum(kl); rr["top1"] += o["agree"]; rr["n"] += len(kl); rr["calls"] += o["calls"]
            dd = rr["dom"].setdefault(dom, [0.0, 0]); dd[0] += sum(kl); dd[1] += len(kl)
            rr["per_chunk"].append([ci, dom, pidx, sum(kl) / len(kl), o["agree"] / len(kl)])
            rr["stats"] = [x + y for x, y in zip(rr["stats"], o["stats"])]
            NL = o["calls"]
            assert NL > 0 and (a.expect_layers == 0 or NL == a.expect_layers), ("router calls", NL)
            kpt = o["stats"][7] / (len(kl) * NL)
            assert K - 1e-6 <= kpt <= TOPK + 1e-6, ("kept picks/tok", kpt)
            if name == "hookref":
                checks.setdefault("hookref", []).append(dict(chunk=ci, meankl=sum(kl) / len(kl), maxd=o["maxd"], calls=o["calls"]))
                if ci == 0:
                    print("CHECK hookref (K=topk through hook)", checks["hookref"][-1], flush=True)
        print(f"chunk {ci} {dom}#{pidx} {time.time()-t0:.0f}s " + " ".join(f"{n}={res[n]['per_chunk'][-1][3]:.4f}" for n, *_ in variants), flush=True)
        json.dump({"partial_after": ci, "res": res, "checks": checks}, open(f"{W}/{a.name}_partial.json", "w"))

    if os.environ.get("REFONLY"):
        return
    out = {}
    for name, K, un in variants:
        rr = res[name]; s = rr["stats"]
        out[name] = dict(K=K, union=un, kl=rr["kl_sum"] / rr["n"], top1=rr["top1"] / rr["n"], ntok=rr["n"],
                         dom={d: v[0] / v[1] for d, v in rr["dom"].items()},
                         tail_recovered_frac=(s[1] / s[0]) if s[0] else None,
                         reads_per_window_kept=s[2] / s[3], reads_per_window_full=s[4] / s[3],
                         kept_weight_frac=s[5] / s[6], kept_picks_per_tok=s[7] / (rr["n"] * rr["calls"] / len(chunks)) if rr["calls"] else None,
                         router_calls_per_chunk=rr["calls"] / len(chunks), per_chunk=rr["per_chunk"])
        print(name, {k: v for k, v in out[name].items() if k != "per_chunk"}, flush=True)
    json.dump({"model": a.model, "tp": a.tp, "top_k": TOPK, "install": info, "checks": checks,
               "chunks": [(d, p, len(i)) for d, p, i in chunks], "results": out}, open(f"{W}/{a.name}_results.json", "w"), indent=1)
    print("ALL DONE", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
