"""Hot-expert tier experiment, step 1 (runs on the workstation, control image, one GPU).

For one representative MoE layer (model layer 32 = MoE index 29, a fully traced layer):
  1. Re-lays the control checkpoint's ExLlamaV3 trellis experts into b12x's projection-tiered MCG layout.
     Pure byte re-layout: no code word or scale is changed.
  2. Builds a real-token test set from captured layer inputs (hidden + router logits), routes it exactly as
     GLM does (sigmoid + correction bias for selection, normalised weights x 2.5).
  3. Computes two reference outputs of the routed-expert sum on the GPU:
       ref_bf16 : BF16 teacher experts
       ref_exl3 : ExLlamaV3's own dequantisation of the same control experts (gold reference for the converter)
     Activation is GLM's clamped SiLU: silu(min(g, 10)) * clamp(u, -10, 10).
Usage: python3 l32_extract.py EXL3_DIR BF16_DIR TRACE_DIR OUT_DIR [tokens_per_domain]
"""
import glob, json, os, sys, time
import numpy as np, torch
from safetensors import safe_open
from safetensors.torch import save_file

EXL3, BF16, TRACES, OUT = sys.argv[1:5]
PER_DOM = int(sys.argv[5]) if len(sys.argv) > 5 else 512
LAYER, MOE_IDX, E, H, I, TOPK, LIMIT, SCALE = 32, 29, 288, 4096, 2048, 8, 10.0, 2.5
os.makedirs(OUT, exist_ok=True)
dev = torch.device("cuda:0")
P = f"model.language_model.layers.{LAYER}.mlp."
PROJ = ("gate_proj", "up_proj", "down_proj")


def opener(root):
    idx = json.load(open(f"{root}/model.safetensors.index.json"))["weight_map"]
    handles = {}
    def get(name):
        f = idx[name]
        if f not in handles:
            handles[f] = safe_open(f"{root}/{f}", "pt")
        return handles[f].get_tensor(name)
    return get

gx = opener(EXL3)
t0 = time.time()
# ---- 1. re-layout ------------------------------------------------------------------------------
tr, suh, svh, K = {}, {}, {}, np.zeros((E, 3), np.int64)
for e in range(E):
    for p, name in enumerate(PROJ):
        k = f"{P}experts.{e}.{name}."
        t = gx(k + "trellis"); tr[e, p] = t; K[e, p] = t.shape[-1] // 16
        suh[e, p] = gx(k + "suh"); svh[e, p] = gx(k + "svh")
        mcg = int(gx(k + "mcg").item()) & 0xFFFFFFFF
        assert mcg == 0xCBAC1FED, (e, name, hex(mcg))

assert set(np.unique(K).tolist()) <= {3, 4, 5}
slots = I // 32
sec = lambda k: (H // 16) * 64 * int(k)                  # bytes of one matrix section in one slot row
payload = int(sum(sec(K[e, p]) for e in range(E) for p in range(3)))
codes = torch.zeros((slots, payload), dtype=torch.uint8)
for s in range(slots):
    cur = 0
    for e in range(E):
        for p in range(3):
            t = tr[e, p]                                   # gate/up [H/16, I/16, 16K]; down [I/16, H/16, 16K]
            planes = t[:, 2 * s:2 * s + 2, :].permute(1, 0, 2) if p < 2 else t[2 * s:2 * s + 2]
            raw = planes.contiguous().view(torch.uint8).reshape(-1)
            assert raw.numel() == sec(K[e, p])
            codes[s, cur:cur + raw.numel()] = raw; cur += raw.numel()
    assert cur == payload
rate = torch.tensor(K * 0x11, dtype=torch.uint8)
in_s = torch.stack([torch.stack([suh[e, 0], suh[e, 1]]) for e in range(E)]).half()        # [E,2,H]
mid_s = torch.stack([torch.stack([svh[e, 0], svh[e, 1], suh[e, 2]]) for e in range(E)]).half()  # [E,3,I]
out_s = torch.stack([svh[e, 2] for e in range(E)]).half()                                    # [E,H]
save_file({"codes": codes, "rate": rate, "in_s": in_s, "mid_s": mid_s, "out_s": out_s}, f"{OUT}/l32_b12x.safetensors")
kinds = {}
for e in range(E):
    kinds[str(tuple(K[e].tolist()))] = kinds.get(str(tuple(K[e].tolist())), 0) + 1
bpw = sum(16 * K[e, p] * 256 for e in range(E) for p in range(3)) * 2 * 8 / (E * 3 * H * I)
print(json.dumps({"relayout_s": round(time.time() - t0, 1), "payload_bytes_per_slot": payload,
                  "code_bytes_total": slots * payload, "trellis_bpw": round(float(bpw), 4), "K_patterns": kinds}), flush=True)

# ---- 2. real-token test set ----------------------------------------------------------------------
bias = gx(f"{P}gate.e_score_correction_bias").float()
doms = ["knowledge_qa", "code", "math", "instruction_following", "multilingual"]
hid, lg, lab = [], [], []
for d in doms:
    got = 0
    for f in sorted(glob.glob(f"{TRACES}/inputs_{d}_*.npz")):
        z = np.load(f); L, n = z["layer"], z["ntok"]; off = np.concatenate([[0], np.cumsum(n)])
        for j in np.nonzero(L == MOE_IDX)[0]:
            a, b = off[j], off[j + 1]; take = min(b - a, PER_DOM - got)
            hid.append(z["hidden"][a:a + take]); lg.append(z["logits"][a:a + take]); lab += [d] * take; got += take
            if got >= PER_DOM: break
        if got >= PER_DOM: break
cnt = {d: np.zeros(E, np.int64) for d in doms}; ntr = {d: 0 for d in doms}   # ntr kept as python ints
bias_np = bias.numpy()
for d in doms:
    skip = PER_DOM
    for f in sorted(glob.glob(f"{TRACES}/inputs_{d}_*.npz")):
        z = np.load(f); L, n = z["layer"], z["ntok"]; off = np.concatenate([[0], np.cumsum(n)])
        sel = np.nonzero(L == MOE_IDX)[0]
        if not len(sel): continue
        lgt = z["logits"]
        for j in sel:
            a, b = off[j], off[j + 1]
            if skip >= b - a: skip -= b - a; continue
            a += skip; skip = 0
            sc = 1 / (1 + np.exp(-lgt[a:b].astype(np.float32)))
            top = np.argpartition(-(sc + bias_np), TOPK, axis=1)[:, :TOPK]
            np.add.at(cnt[d], top.reshape(-1), 1); ntr[d] += int(b - a)
json.dump({"tokens": ntr, "counts": {d: cnt[d].tolist() for d in doms}}, open(f"{OUT}/l32_route_counts.json", "w"))
print(json.dumps({"count_tokens": ntr}), flush=True)
x = torch.from_numpy(np.concatenate(hid)).to(dev)
logits = torch.from_numpy(np.concatenate(lg)).to(dev).float()
scores = logits.sigmoid()
ids = torch.topk(scores + bias.to(dev), TOPK, dim=-1).indices
w = scores.gather(1, ids); w = w / w.sum(-1, keepdim=True) * SCALE
T = x.shape[0]
print(json.dumps({"tokens": T, "per_domain": {d: lab.count(d) for d in doms}}), flush=True)

# ---- 3. references ---------------------------------------------------------------------------
sys.path.insert(0, "/opt/exllamav3")
import exllamav3_ext as ext                       # ExLlamaV3's compiled kernels (python package not importable here)
HAD_DIR = "/opt/exllamav3-python/exllamav3/util/hadamard_data"
_had = {}
for f in glob.glob(f"{HAD_DIR}/hadamard_*.txt"):
    rows = [r.strip() for r in open(f) if r.strip()]
    _had[len(rows)] = torch.tensor([[1. if c == "+" else -1. for c in r] for r in rows])
def get_had(n):                                   # same resolution order as exllamav3.util.hadamard.get_hadamard
    if n in _had: return _had[n]
    assert n % 2 == 0, n
    s_ = get_had(n // 2); return torch.cat([torch.cat([s_, s_], 1), torch.cat([s_, -s_], 1)], 0)
HAD = (get_had(128) / 128 ** 0.5).to(dev)
def exl3_weight(trellis, su, sv, fin, fout):      # = LinearEXL3.get_weight_tensor, W is [in, out]
    K_ = trellis.shape[-1] // 16
    wi = torch.empty((fin, fout), dtype=torch.half, device=dev)
    ext.reconstruct(wi, trellis, K_, True, False)  # mcg codebook, no mul1
    W = (HAD @ wi.float().view(-1, 128, fout)).view(fin, fout) * su.float().unsqueeze(1)
    W = (W.view(fin, -1, 128) @ HAD).view(fin, fout) * sv.float().unsqueeze(0)
    return W

def expert_out(xs, Wg, Wu, Wd):                            # all [in, out], fp32 math
    g = xs @ Wg; u = xs @ Wu
    h = torch.nn.functional.silu(g.clamp(max=LIMIT)) * u.clamp(-LIMIT, LIMIT)
    return h @ Wd

gb = opener(BF16)
ref_b = torch.zeros((T, H), dtype=torch.float32, device=dev)
ref_x = torch.zeros_like(ref_b)
route_b = torch.zeros((T, TOPK, H), dtype=torch.float16, device=dev)   # weighted per-route contributions
route_x = torch.zeros_like(route_b)
werr = []
xf = x.float()
for e in range(E):
    rows, slot = (ids == e).nonzero(as_tuple=True)
    if rows.numel() == 0:
        continue
    xs = xf[rows]; we = w[rows, slot].unsqueeze(1)
    Wb = [gb(f"{P}experts.{e}.{n}.weight").to(dev).float().t() for n in PROJ]     # [in,out]
    Wx = []
    for p, n in enumerate(PROJ):
        t = tr[e, p].to(dev)
        fin, fout = (H, I) if p < 2 else (I, H)
        Wx.append(exl3_weight(t, suh[e, p].to(dev), svh[e, p].to(dev), fin, fout))
    cb = we * expert_out(xs, *Wb); cx = we * expert_out(xs, *Wx)
    ref_b.index_add_(0, rows, cb); ref_x.index_add_(0, rows, cx)
    route_b[rows, slot] = cb.half(); route_x[rows, slot] = cx.half()
rel = float((ref_x - ref_b).norm() / ref_b.norm())
print(json.dumps({"exl3_vs_bf16_layer_rel_err": round(rel, 5),
                  "experts_touched": int(torch.unique(ids).numel())}), flush=True)
save_file({"x": x.half().cpu(), "ids": ids.int().cpu(), "w": w.float().cpu(), "ref_bf16": ref_b.cpu(),
           "ref_exl3": ref_x.cpu(), "route_bf16": route_b.cpu(), "route_exl3": route_x.cpu(), "dom": torch.tensor([doms.index(d) for d in lab], dtype=torch.int8)},
          f"{OUT}/l32_tokens.safetensors")
print("EXTRACT_DONE", round(time.time() - t0, 1), "s", flush=True)
