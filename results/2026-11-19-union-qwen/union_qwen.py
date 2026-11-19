# Union-aware routing + mean-output stand-in on Qwen3-30B-A3B (teacher-forced, HF).
import json, time, sys, os, math
import torch, torch.nn.functional as F
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3_moe import modeling_qwen3_moe as M

W = "/srv/cache/heteroflash/union-qwen"
MP = "/srv/models/Qwen3-30B-A3B"
CTX = 2048; WIN = 4
CH_PER_DOM = int(os.environ.get("CH_PER_DOM", "2"))
KS = [int(k) for k in os.environ.get("KS", "6,5,4,3").split(",")]
dev = "cuda:0"
torch.manual_seed(0)

tok = AutoTokenizer.from_pretrained(MP)
# ---- eval chunks: decode each domain's GLM tokens -> domain text -> Qwen tokens
man = json.load(open(f"{W}/data/q_eval_balanced_v1.manifest.json"))
glm = Tokenizer.from_file(f"{W}/data/glm_tokenizer/tokenizer.json")
gch = json.load(open(f"{W}/data/q_eval_balanced_v1.tokens.json"))["chunks"]
chunks = []  # (domain, ids)
raw = open(f"{W}/data/q_eval_balanced_v1.txt", newline="").read()
starts = []
for d in man["domains"]:
    a = d["chunks"][0]
    pre = glm.decode(gch[a][:40])
    pos = raw.find(pre); assert pos >= 0, d["domain"]
    starts.append(pos)
starts.append(len(raw))
assert starts == sorted(starts), starts
for di, d in enumerate(man["domains"]):
    q = tok(raw[starts[di]:starts[di + 1]], add_special_tokens=False)["input_ids"]
    n = len(q) // CTX
    picks = [round(i * (n - 1) / max(1, CH_PER_DOM - 1)) for i in range(CH_PER_DOM)] if n >= CH_PER_DOM else list(range(n))
    for p_ in picks:
        chunks.append((d["domain"], q[p_ * CTX:(p_ + 1) * CTX]))
    print(d["domain"], "chars", starts[di], starts[di + 1], "qwen tokens", len(q), "chunks avail", n, "picked", picks, flush=True)
calib_ids = tok(open(f"{W}/data/q_imatrix_calib.txt", newline="").read()[:200000], add_special_tokens=False)["input_ids"][:5 * CTX]
print("calib tokens", len(calib_ids), flush=True)


def masks(ids, w, K, union, E):
    """core/inunion/keep for (N,8) ids sorted by weight desc; windows of WIN rows."""
    TOPK_ = ids.shape[1]; N = ids.shape[0]
    core = torch.zeros_like(ids, dtype=torch.bool); core[:, :K] = True
    nw = N // WIN
    idw = ids[: nw * WIN].reshape(nw, WIN * TOPK_)
    corew = core[: nw * WIN].reshape(nw, WIN * TOPK_)
    pres = torch.zeros(nw, E + 1, dtype=torch.bool, device=ids.device)
    pres.scatter_(1, torch.where(corew, idw, torch.full_like(idw, E)), True)
    inunion = torch.zeros_like(ids, dtype=torch.bool)
    inunion[: nw * WIN] = pres[:, :E].gather(1, idw).view(nw * WIN, TOPK_)
    keep = (core | inunion) if union else core
    return core, inunion, keep, idw, nw

def literal_port(ids, w, K, union):  # python port of drop_win_kernel (tau=0, rn=0), per window
    N, k8 = ids.shape; keep = [[False] * k8 for _ in range(N)]
    for s0 in range(0, N - N % WIN, WIN):
        rows = range(s0, s0 + WIN); present = set(); core = {}
        for t in rows:
            wv = w[t].tolist()
            for i in range(k8):
                r = sum((wv[j] > wv[i]) or (wv[j] == wv[i] and j < i) for j in range(k8))
                core[t, i] = r < K
                if core[t, i]: present.add(int(ids[t, i]))
        for t in rows:
            for i in range(k8):
                keep[t][i] = core[t, i] or (union and int(ids[t, i]) in present)
    for t in range(N - N % WIN, N):
        wv = w[t].tolist()
        for i in range(k8):
            keep[t][i] = sum((wv[j] > wv[i]) or (wv[j] == wv[i] and j < i) for j in range(k8)) < K
    return torch.tensor(keep)

g = torch.Generator().manual_seed(1)
for trial in range(20):
    N = 4 * 13 + (trial % 4)
    lg = torch.randn(N, 16, generator=g) * 2
    pr = lg.softmax(-1); wv, iv = torch.topk(pr, 8, dim=-1)
    for K in (3, 4, 5, 6, 8):
        for un in (False, True):
            _, _, kp, _, _ = masks(iv, wv, K, un, 16)
            assert torch.equal(kp, literal_port(iv, wv, K, un)), (trial, K, un)
print("mask unit test vs literal kernel port: PASS", flush=True)

t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(MP, dtype=torch.bfloat16, device_map=dev)
model.eval()
cfg = model.config
E, TOPK, H = cfg.num_experts, cfg.num_experts_per_tok, cfg.hidden_size
print("loaded in", time.time() - t0, "s; norm_topk_prob", cfg.norm_topk_prob, "experts impl", getattr(cfg, "_experts_implementation", None), flush=True)
blocks = [l.mlp for l in model.model.layers]
for i, b in enumerate(blocks):
    b._lid = i
NL = len(blocks)

# ---- state
S = {"mode": "orig", "K": 8, "union": False, "stand": False}
mean_sum = torch.zeros(NL, E, H, dtype=torch.float32, device=dev)
mean_cnt = torch.zeros(NL, E, dtype=torch.float32, device=dev)
MEAN = None
stats = {}
orig_forward = M.Qwen3MoeSparseMoeBlock.forward

def expert_outputs_accumulate(blk, x, ids):
    ex = blk.experts
    for e in ids.unique().tolist():
        tk, _ = torch.where(ids == e)
        xs = x[tk]
        g, u = F.linear(xs, ex.gate_up_proj[e]).chunk(2, dim=-1)
        o = F.linear(ex.act_fn(g) * u, ex.down_proj[e])
        mean_sum[blk._lid, e] += o.float().sum(0)
        mean_cnt[blk._lid, e] += tk.numel()

def hooked_forward(self, hidden_states):
    if S["mode"] == "orig":
        return orig_forward(self, hidden_states)
    B, T, Hd = hidden_states.shape
    x = hidden_states.view(-1, Hd)
    _, w, ids = self.gate(x)              # w: bf16, sorted descending (topk sorted)
    L = self._lid
    if S["mode"] == "calib":
        expert_outputs_accumulate(self, x, ids)
        return self.experts(x, ids, w).reshape(B, T, Hd)
    # mode == "var"
    K = S["K"]
    core, inunion, keep, idw, nw = masks(ids, w, K, S["union"], E)
    # stats
    st = stats.setdefault(L, torch.zeros(6, dtype=torch.float64, device=x.device))
    tail = ~core
    st[0] += tail.sum(); st[1] += (tail & inunion).sum()
    # reads per window: distinct experts among kept picks per window (full windows)
    kw = keep[: nw * WIN].view(nw, WIN * TOPK)
    pk = torch.zeros(nw, E + 1, dtype=torch.bool, device=x.device)
    pk.scatter_(1, torch.where(kw, idw, torch.full_like(idw, E)), True)
    st[2] += pk[:, :E].sum(); st[3] += nw
    pa = torch.zeros(nw, E + 1, dtype=torch.bool, device=x.device); pa.scatter_(1, idw, True)
    st[4] += pa[:, :E].sum()
    # kept weight mass
    st[5] += (w.float() * keep).sum() / max(1, 1)
    w2 = torch.where(keep, w, torch.zeros_like(w))
    y = self.experts(x, ids, w2)
    if S["stand"] and K < TOPK:
        drop = ~keep
        dw = torch.where(drop, w.float(), torch.zeros_like(w, dtype=torch.float32))  # (N,8)
        mu = MEAN[L][ids]  # (N,8,H) fp32
        y = (y.float() + (dw.unsqueeze(-1) * mu).sum(1)).to(y.dtype)
    return y.reshape(B, T, Hd)

M.Qwen3MoeSparseMoeBlock.forward = hooked_forward

@torch.no_grad()
def logits_of(ids):
    t = torch.tensor([ids], device=dev)
    return model(input_ids=t, use_cache=False).logits[0].float()

# ---- calibration means
t0 = time.time()
S["mode"] = "calib"
for i in range(0, len(calib_ids), CTX):
    logits_of(calib_ids[i:i + CTX])
MEAN = (mean_sum / mean_cnt.clamp(min=1).unsqueeze(-1)).to(torch.bfloat16).float()
print("mean_cnt<10:", int((mean_cnt < 10).sum()), "median cnt", mean_cnt.median().item(), flush=True)
print("calib done", time.time() - t0, "s; experts never hit:", int((mean_cnt == 0).sum()), "of", NL * E, flush=True)
torch.save({"mean": MEAN.to(torch.bfloat16).cpu(), "cnt": mean_cnt.cpu()}, f"{W}/qwen_mean_outputs.pt")
del mean_sum

# ---- eval
variants = []
for K in KS:
    variants += [(f"k{K}", K, False, False), (f"k{K}u", K, True, False), (f"k{K}s", K, False, True), (f"k{K}us", K, True, True)]
variants = [("k8hook", 8, False, False)] + variants
res = {v[0]: {"kl_sum": 0.0, "top1": 0, "n": 0, "dom": {}} for v in variants}
vstats = {v[0]: None for v in variants}
t0 = time.time()
for ci, (dom, ids) in enumerate(chunks):
    S["mode"] = "orig"
    ref = logits_of(ids)
    lref = F.log_softmax(ref, -1); pref = lref.exp(); aref = ref.argmax(-1)
    del ref
    for name, K, un, sd in variants:
        S.update(mode="var", K=K, union=un, stand=sd)
        stats.clear()
        lv = F.log_softmax(logits_of(ids), -1)
        kl = (pref * (lref - lv)).sum(-1)  # per token
        if name == "k8hook" and ci == 0:
            md = (lv - lref).abs().max().item()
            print("K=8 hook check: max|dlogp|", md, "mean KL", kl.mean().item(), flush=True)
            assert kl.mean().item() < 1e-6 and md < 1e-3, "K=8 hook does not reproduce reference"
        r = res[name]
        r["kl_sum"] += kl.sum().item(); r["top1"] += (lv.argmax(-1) == aref).sum().item(); r["n"] += kl.numel()
        dd = r["dom"].setdefault(dom, [0.0, 0])
        dd[0] += kl.sum().item(); dd[1] += kl.numel()
        r.setdefault("per_chunk", []).append([ci, dom, kl.mean().item()])
        tot = torch.stack([stats[l] for l in sorted(stats)]).sum(0).cpu()
        vstats[name] = tot if vstats[name] is None else vstats[name] + tot
        del lv, kl
    del lref, pref
    print(f"chunk {ci} {dom} done {time.time()-t0:.0f}s", flush=True)
    json.dump({"partial_after_chunk": ci, "res": res, "stats": {k: (v.tolist() if v is not None else None) for k, v in vstats.items()}}, open(f"{W}/partial.json", "w"))

out = {}
for name, *_ in variants:
    r = res[name]; s = vstats[name]
    out[name] = {
        "kl": r["kl_sum"] / r["n"], "top1": r["top1"] / r["n"], "ntok": r["n"],
        "dom": {d: v[0] / v[1] for d, v in r["dom"].items()},
        "tail_recovered_frac": (s[1] / s[0]).item() if s[0] > 0 else None,
        "reads_per_window_kept": (s[2] / s[3]).item(), "reads_per_window_full8": (s[4] / s[3]).item(),
        "kept_weight_mass_per_tok": (s[5] / (r["n"] * NL)).item(),
        "per_chunk": r["per_chunk"],
    }
    print(name, json.dumps(out[name]), flush=True)
json.dump({"chunks": [(d, len(i)) for d, i in chunks], "results": out}, open(f"{W}/results.json", "w"), indent=1)
print("ALL DONE", flush=True)
