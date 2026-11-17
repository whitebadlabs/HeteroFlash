#!/usr/bin/env python3
"""Bet 1 step 4: head-split exactness of one real KDA layer and one real MLA layer (GLM-5.3-Flash, FP8 build).
  extract  --ckpt DIR --out W.pt --kda 4 --mla 3 --tokens 36      (Spark; real weights + real embedding rows as inputs)
  run      --w W.pt --kind kda|mla --heads 0:64 --mode ref|rt --backend torch|c --device cuda|cpu --out P.pt
  compare  --dir D                                                  (prints rel. L2 errors as JSON)
Modes: ref = fp32 activations, fp32 matmul with dequantized exact weights (the reference).
       rt  = runtime-like: bf16 activations at every op boundary, fp32 accumulation, fp32 recurrence/softmax.
Backends for rt linears: torch (GPU) or c (workstation CPU: the libhfgemv.so FP8/BF16 kernels on the checkpoint's own bytes).
A partial = this side's heads' contribution to the layer output [T, 4096] (o_proj restricted to its heads' columns).
Inputs: RMS-normalised embedding rows of fixed pseudo-random token ids (not real layer-4 residuals) [A]."""
import argparse, ctypes, json, os, struct, sys
import torch
F8 = torch.float8_e4m3fn
H, D, LB, EPS = 64, 128, -5.0, 1e-5

def deq(t):  # dict(w=raw, s8|sf) -> fp32 weight
    w = t["w"]
    if w.dtype == torch.uint8:
        f = w.view(F8).float()
        if "s8" in t: return f * torch.exp2(t["s8"].float() - 127).repeat_interleave(32, dim=1)
        sf = t["sf"]; return f * sf.repeat_interleave(128, 0).repeat_interleave(128, 1)[: f.shape[0], : f.shape[1]]
    return w.float()

def rows(t, a, b):
    o = {"w": t["w"][a:b].contiguous()}
    if "s8" in t: o["s8"] = t["s8"][a:b].contiguous()
    if "sf" in t: assert a % 128 == 0 and b % 128 == 0; o["sf"] = t["sf"][a // 128: b // 128].contiguous()
    return o
def cols(t, a, b):
    o = {"w": t["w"][:, a:b].contiguous()}
    if "s8" in t: o["s8"] = t["s8"][:, a // 32: b // 32].contiguous()
    if "sf" in t: o["sf"] = t["sf"][:, a // 128: b // 128].contiguous()
    return o

class Lin:
    def __init__(s, mode, backend, dev):
        s.mode, s.backend, s.dev = mode, backend, dev
        if backend == "c":
            s.lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "libhfgemv.so"))
    def act(s, x): return x if s.mode == "ref" else x.to(torch.bfloat16).float()
    def __call__(s, x, t):
        x = s.act(x)
        if s.backend == "torch":
            return s.act(x.to(s.dev) @ deq({k: v.to(s.dev) for k, v in t.items()}).t())
        w = t["w"]; N, K = w.shape; T = x.shape[0]
        f = 1 if w.dtype == torch.bfloat16 else (0 if "s8" in t else 3)
        xb = x.to(torch.bfloat16).contiguous(); y = torch.empty(T, N, dtype=torch.float32)
        wv = w.view(torch.int16) if f == 1 else w
        s8 = t.get("s8"); sf = t.get("sf")
        sfc = sf.reshape(sf.shape[0], -1).contiguous().float() if sf is not None else None
        r = s.lib.hf_gemv(f, ctypes.c_void_p(wv.data_ptr()), ctypes.c_void_p(s8.data_ptr() if s8 is not None else 0),
                          ctypes.c_void_p(sfc.data_ptr() if sfc is not None else 0), N, K,
                          ctypes.c_void_p(xb.view(torch.int16).data_ptr()), T, ctypes.c_void_p(y.data_ptr()))
        assert r == 0, (N, K, f)
        return s.act(y)

def rms(x, w, eps=EPS): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w.float()

def kda(W, x, h0, h1, lin):
    T = x.shape[0]; n = h1 - h0; a, b = h0 * D, h1 * D; dev = x.device
    q, k, v = (lin(x, rows(W[p], a, b)) for p in ("q_proj", "k_proj", "v_proj"))
    beta = lin(x, rows(W["b_proj"], h0, h1)); fa = lin(x, W["f_a_proj"]); ga = lin(x, W["g_a_proj"])
    g1 = lin(fa, rows(W["f_b_proj"], a, b)); g2 = lin(ga, rows(W["g_b_proj"], a, b))
    def conv(u, wname):  # causal depthwise conv, kernel 4, zero initial state, then SiLU
        w = W[wname]["w"][a:b, 0, :].float().to(dev); up = torch.cat([torch.zeros(3, u.shape[1], device=dev), u], 0)
        return lin.act(torch.nn.functional.silu(sum(up[j: j + T] * w[:, j] for j in range(4))))
    q, k, v = conv(q, "q_conv1d"), conv(k, "k_conv1d"), conv(v, "v_conv1d")
    q, k, v, g1, g2 = (z.reshape(T, n, D) for z in (q, k, v, g1, g2))
    q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6) * D ** -0.5
    k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    A = torch.exp(W["A_log"]["w"].float()[h0:h1].to(dev)).view(n, 1); dtb = W["dt_bias"]["w"].float()[a:b].view(n, D).to(dev)
    gk = LB / (1 + torch.exp(-(A * (g1 + dtb))))           # [T, n, D], bounded log-gate
    bt = torch.sigmoid(beta)                                 # [T, n]
    S = torch.zeros(n, D, D, device=dev); o = torch.empty(T, n, D, device=dev)
    for t in range(T):
        S = S * torch.exp(gk[t])[:, None, :]
        vv = (v[t] - torch.einsum("hvk,hk->hv", S, k[t])) * bt[t][:, None]
        S = S + vv[:, :, None] * k[t][:, None, :]
        o[t] = torch.einsum("hvk,hk->hv", S, q[t])
    o = lin.act(o)
    o = lin.act(rms(o, W["o_norm"]["w"].to(dev)) * torch.sigmoid(g2))
    return lin(o.reshape(T, n * D), cols(W["o_proj"], a, b))

def mla(W, x, h0, h1, lin):
    T = x.shape[0]; n = h1 - h0; dev = x.device; dq, dv = 256, 256
    qa = lin.act(rms(lin(x, W["q_a_proj"]), W["q_a_layernorm"]["w"].to(dev)))          # replicated on both sides
    ckv = lin.act(rms(lin(x, W["kv_a_proj_with_mqa"]), W["kv_a_layernorm"]["w"].to(dev)))
    q = lin(qa, rows(W["q_b_proj"], h0 * dq, h1 * dq)).reshape(T, n, dq)
    kv = lin(ckv, rows(W["kv_b_proj"], h0 * (dq + dv), h1 * (dq + dv))).reshape(T, n, dq + dv)
    kk, vv = kv[..., :dq], kv[..., dq:]
    s = torch.einsum("thd,shd->hts", q, kk) * dq ** -0.5
    s = s.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool, device=dev), 1), float("-inf"))
    o = lin.act(torch.einsum("hts,shd->thd", torch.softmax(s, -1), vv))   # context <= 2048: DSA top-k selects all
    return lin(o.reshape(T, n * dv), cols(W["o_proj"], h0 * dv, h1 * dv))

def extract(a):
    from safetensors import safe_open
    idx = {}
    for fn in sorted(os.listdir(a.ckpt)):
        if fn.endswith(".safetensors"):
            with safe_open(os.path.join(a.ckpt, fn), "pt") as f:
                for k in f.keys(): idx[k] = fn
    def get(k):
        with safe_open(os.path.join(a.ckpt, idx[k]), "pt") as f: return f.get_tensor(k)
    def layer(L, names):
        pre = [k for k in idx if k.endswith(f"layers.{L}.self_attn.{names[0]}.weight")][0].rsplit(f"{names[0]}.weight", 1)[0]
        out = {}
        for nm in names:
            key = pre + nm + (".weight" if nm not in ("A_log", "dt_bias") else "")
            w = get(key); d = {"w": w.view(torch.uint8) if w.dtype == F8 else w}
            if key + "_scale" in idx:
                sc = get(key + "_scale")
                if sc.dtype == torch.uint8: d["s8"] = sc
                else: d["sf"] = sc.reshape(sc.shape[0], sc.shape[2]).float()
            out[nm] = d
        return out, pre
    kda_n = ["q_proj", "k_proj", "v_proj", "b_proj", "f_a_proj", "g_a_proj", "f_b_proj", "g_b_proj", "q_conv1d", "k_conv1d",
             "v_conv1d", "A_log", "dt_bias", "o_norm", "o_proj"]
    mla_n = ["q_a_proj", "q_a_layernorm", "q_b_proj", "kv_a_proj_with_mqa", "kv_a_layernorm", "kv_b_proj", "o_proj"]
    Wk, pk = layer(a.kda, kda_n); Wm, pm = layer(a.mla, mla_n)
    ek = [k for k in idx if k.endswith("embed_tokens.weight")][0]
    with safe_open(os.path.join(a.ckpt, idx[ek]), "pt") as f: emb = f.get_slice(ek)
    g = torch.Generator().manual_seed(1234); ids = torch.randint(1000, 120000, (a.tokens,), generator=g)
    x = torch.stack([emb[int(i): int(i) + 1][0] for i in ids]).float()
    xk = rms(x, get(pk.replace("self_attn.", "input_layernorm.weight")))
    xm = rms(x, get(pm.replace("self_attn.", "input_layernorm.weight")))
    torch.save({"kda": Wk, "mla": Wm, "x_kda": xk, "x_mla": xm, "ids": ids, "prefix": [pk, pm]}, a.out)
    print("saved", a.out, pk, pm, {n: tuple(Wk[n]["w"].shape) for n in kda_n}, {n: tuple(Wm[n]["w"].shape) for n in mla_n})

def run(a):
    torch.backends.cuda.matmul.allow_tf32 = False
    B = torch.load(a.w); h0, h1 = map(int, a.heads.split(":"))
    lin = Lin(a.mode, a.backend, a.device); x = B["x_" + a.kind].to(a.device if a.backend == "torch" else "cpu")
    import time; t0 = time.time()
    with torch.no_grad(): p = (kda if a.kind == "kda" else mla)(B[a.kind], x, h0, h1, lin)
    torch.save(p.float().cpu(), a.out); print(a.out, tuple(p.shape), f"{time.time() - t0:.2f}s", file=sys.stderr)

def compare(a):
    L = lambda n: torch.load(os.path.join(a.dir, n))
    rel = lambda y, r: float((y - r).norm() / r.norm())
    res = {}
    for k in ("kda", "mla"):
        ref, gpu = L(f"{k}_ref_all.pt"), L(f"{k}_rt_gpu_all.pt")
        pg, pc = L(f"{k}_rt_gpu_lo.pt"), L(f"{k}_rt_c_hi.pt")
        rlo, rhi = L(f"{k}_ref_lo.pt"), L(f"{k}_ref_hi.pt")
        bf = lambda z: z.to(torch.bfloat16).float()
        split32 = pg + pc; split16 = bf(bf(pg) + bf(pc))
        res[k] = {"ref_split_vs_ref_unsplit (decomposition, fp32)": rel(rlo + rhi, ref),
                  "gpu_only_rt_vs_ref (runtime's own error)": rel(gpu, ref),
                  "split_rt_fp32_exchange_vs_ref": rel(split32, ref),
                  "split_rt_bf16_exchange_vs_ref": rel(split16, ref),
                  "split_rt_fp32_exchange_vs_gpu_only": rel(split32, gpu),
                  "split_rt_bf16_exchange_vs_gpu_only": rel(split16, gpu),
                  "cpu_half_c_vs_ref_half": rel(pc, rhi), "gpu_half_vs_ref_half": rel(pg, rlo),
                  "last4_split_bf16_vs_gpu_only": rel(split16[-4:], gpu[-4:]),
                  "output_rms": float(ref.pow(2).mean().sqrt())}
    print(json.dumps(res, indent=1))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd")
    for o in ("--ckpt", "--out", "--w", "--kind", "--heads", "--mode", "--backend", "--device", "--dir"): ap.add_argument(o)
    ap.add_argument("--kda", type=int, default=4); ap.add_argument("--mla", type=int, default=3); ap.add_argument("--tokens", type=int, default=36)
    a = ap.parse_args(); {"extract": extract, "run": run, "compare": compare}[a.cmd](a)
