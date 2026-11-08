#!/usr/bin/env python3
"""Bounded QAT pilot: IQ3_XXS quantization-aware training of one MoE layer's routed experts (docs/QAT_PILOT_PROTOCOL.md).

IQ3_XXS super-block (256 weights along the input dim, 98 bytes): fp16 d; 64 grid indices (each -> 4 magnitudes from a
256x4 grid of {4,12,...,62}); 8 uint32 words, one per 32 weights: bits 28-31 sub-scale s (0..15), bits 0-27 four 7-bit
sign codes (8 signs each, even parity, bit k set = value k negative). value = d*(0.5+s)*0.5 * grid * sign.

Training (per expert, independent given the original top-8 routing):
  latent W (fp32, init BF16), learned sub-scale sigma per 32 (init from the llama.cpp IQ3_XXS PTQ of the same layer);
  forward projects onto the exact IQ3_XXS grid (fp16 d, integer s, nearest grid row, parity-fixed signs) with a
  straight-through estimator; loss = gate-weighted ||E_q(x) - E_bf16(x)||^2 on the expert's routed tokens.
Schedule: P passes, each a fixed share of the time budget; after every pass the full routed layer output
  y = sum_k w_k E_k(x) is evaluated on HELD-OUT-DOMAIN tokens for: BF16 (reference), Q4_K PTQ, IQ3_XXS PTQ (llama.cpp),
  IQ3_XXS step-0 projection, IQ3_XXS trained. Plateau stop: < 2% improvement over the last pass (= 20% of budget).
Commands: selftest | train. Run inside the vLLM image on a Max-Q GPU (torch + CUDA); gguf-py mounted.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

QK = 256
BLOCK_BYTES = 2 + QK // 4 + QK // 8      # 98


# ---------------------------------------------------------------- format: grid, decode, encode
def load_tables(gguf_py):
    sys.path.insert(0, gguf_py)
    from gguf.quants import IQ2_XXS, IQ3_XXS
    IQ3_XXS.init_grid()
    grid = np.asarray(IQ3_XXS.grid, dtype=np.float32).reshape(256, 4)
    ksigns = np.frombuffer(IQ2_XXS.ksigns, dtype=np.uint8)            # 128 8-bit patterns (even parity)
    sign_to_code = {int(p): i for i, p in enumerate(ksigns)}          # 8-bit pattern -> 7-bit code
    return grid, ksigns, sign_to_code


def unpack_blocks(raw, grid, ksigns):
    """raw uint8 [n_blocks, 98] -> d [nb], s [nb,8], idx [nb,64], neg [nb,256] (bool), values [nb,256]."""
    nb = raw.shape[0]
    d = raw[:, :2].copy().view(np.float16).astype(np.float32)[:, 0]
    idx = raw[:, 2:66].astype(np.int64)
    words = raw[:, 66:98].copy().view(np.uint32)                      # [nb, 8]
    s = (words >> 28).astype(np.float32)
    codes = np.stack([(words >> (7 * g)) & 0x7F for g in range(4)], axis=-1)   # [nb, 8, 4]
    pats = ksigns[codes]                                              # [nb, 8, 4] uint8
    neg = ((pats[..., None] >> np.arange(8, dtype=np.uint8)) & 1).astype(bool).reshape(nb, 256)
    mags = grid[idx].reshape(nb, 256)
    db = (d[:, None] * (0.5 + s) * 0.5)                               # [nb, 8]
    vals = np.repeat(db, 32, axis=1) * mags * np.where(neg, -1.0, 1.0)
    return d, s, idx, neg, vals


def pack_blocks(d16, s_int, idx, neg, sign_to_code):
    """Inverse of unpack: d16 float16 [nb], s_int [nb,8] 0..15, idx [nb,64] 0..255, neg [nb,256] bool (even parity per 8)."""
    nb = d16.shape[0]
    out = np.zeros((nb, BLOCK_BYTES), dtype=np.uint8)
    out[:, :2] = d16.astype(np.float16).view(np.uint8).reshape(nb, 2)
    out[:, 2:66] = idx.astype(np.uint8)
    bits = neg.reshape(nb, 8, 4, 8).astype(np.uint8)
    pats = (bits << np.arange(8, dtype=np.uint8)).sum(-1).astype(np.uint8)   # [nb, 8, 4]
    lut = np.full(256, 255, dtype=np.int64)
    for p, c in sign_to_code.items():
        lut[p] = c
    codes = lut[pats]
    assert (codes != 255).all(), "odd-parity sign pattern"
    words = (s_int.astype(np.uint32) << 28)
    for g in range(4):
        words |= (codes[..., g].astype(np.uint32) << (7 * g))
    out[:, 66:98] = words.astype(np.uint32).view(np.uint8).reshape(nb, 32)
    return out


# ---------------------------------------------------------------- projection (torch, exact grid)
class Projector:
    def __init__(self, grid, dev):
        self.g = torch.tensor(grid, device=dev)                       # [256, 4]
        self.g2 = (self.g ** 2).sum(1)                                # [256]

    def nearest(self, a):                                             # a [N,4] -> idx [N]
        out = torch.empty(a.shape[0], dtype=torch.long, device=a.device)
        for i in range(0, a.shape[0], 1 << 19):
            aa = a[i:i + (1 << 19)]
            out[i:i + aa.shape[0]] = torch.argmax(2 * aa @ self.g.T - self.g2, dim=1)
        return out

    def project(self, W, d, logsig, ste=True):
        """W [rows, cols] fp32; d [rows, cols/256] fp16-exact super-scales (fixed, from PTQ); logsig [rows, cols/32]
        learned sub-scales, snapped to the 4-bit grid s in 0..15 relative to d. Returns (W_eff, parts)."""
        R, Cn = W.shape
        sig = logsig.exp()
        sb = sig.view(R, Cn // QK, 8)
        d = d.view(R, Cn // QK)
        s = torch.clamp(torch.round(sb / (d.unsqueeze(-1) * 0.5) - 0.5), 0, 15)
        sq = d.unsqueeze(-1) * (0.5 + s) * 0.5                                    # [R, nb, 8]
        sq_ste = sb + (sq - sb).detach() if ste else sq
        Wb = W.view(R, Cn // QK, 8, 32)
        a = (Wb.abs() / sq.unsqueeze(-1)).detach().reshape(-1, 4)
        idx = self.nearest(a)
        mags = self.g[idx].view(R, Cn // QK, 8, 32)
        neg = (Wb < 0).detach()
        # even parity per 8: flip the smallest-|w| element of odd groups
        n8 = neg.view(R, Cn // QK, 8, 4, 8)
        odd = (n8.sum(-1) % 2 == 1)
        absw = Wb.detach().abs().view(R, Cn // QK, 8, 4, 8)
        jmin = absw.argmin(-1, keepdim=True)
        flip = torch.zeros_like(n8).scatter_(-1, jmin, True) & odd.unsqueeze(-1)
        n8 = n8 ^ flip
        sign = torch.where(n8.view_as(neg), -1.0, 1.0)
        q = (mags * sign).detach()
        Wq = (sq_ste.unsqueeze(-1) * q).view(R, Cn)
        W_eff = Wq + (W - W.detach()) if ste else Wq
        parts = {"d": d, "s": s, "idx": idx.view(R, Cn // QK, 64), "neg": n8.reshape(R, Cn // QK, 256)}
        return W_eff, parts


def parts_to_bytes(parts, sign_to_code):
    R, nb = parts["d"].shape
    return pack_blocks(parts["d"].cpu().numpy().astype(np.float16).reshape(-1),
                       parts["s"].cpu().numpy().astype(np.int64).reshape(-1, 8),
                       parts["idx"].cpu().numpy().reshape(-1, 64),
                       parts["neg"].cpu().numpy().reshape(-1, 256), sign_to_code).reshape(R, nb * BLOCK_BYTES)


# ---------------------------------------------------------------- data
def silu(v):
    return v * torch.sigmoid(v)


def ffn(x, g, u, dn):
    return (silu(x @ g.T) * (x @ u.T)) @ dn.T


def load_layer_captures(glob_pat, moe_idx):
    """Tagged pilot captures inputs_<domain>_<pid>_<n>.npz -> hidden [N,4096] f16, logits [N,288] f16, domain [N]."""
    H, L, D = [], [], []
    for f in sorted(glob.glob(glob_pat)):
        base = os.path.basename(f)[len("inputs_"):-len(".npz")]
        parts = base.split("_")
        dom = "_".join(parts[:-2])
        with np.load(f) as z:
            layer, ntok, hid, lg = z["layer"], z["ntok"], z["hidden"], z["logits"]
        off = np.concatenate([[0], np.cumsum(ntok)])
        for i in np.where(layer == moe_idx)[0]:
            H.append(hid[off[i]:off[i + 1]]); L.append(lg[off[i]:off[i + 1]]); D.extend([dom] * int(ntok[i]))
    return np.concatenate(H), np.concatenate(L), np.array(D)


class GGUFLayer:
    def __init__(self, path, gguf_py, layer):
        sys.path.insert(0, gguf_py)
        import gguf
        self.gguf = gguf
        r = gguf.GGUFReader(path)
        self.t = {m: next(t for t in r.tensors if t.name == f"blk.{layer}.ffn_{m}_exps.weight") for m in ("gate", "up", "down")}

    def raw(self, m, e):
        return np.asarray(self.t[m].data[e])

    def deq(self, m, e):
        t = self.t[m]
        return np.asarray(self.gguf.quants.dequantize(np.asarray(t.data[e]), t.tensor_type), dtype=np.float32)


# ---------------------------------------------------------------- commands
def cmd_selftest(a):
    grid, ksigns, s2c = load_tables(a.gguf_py)
    g = GGUFLayer(a.gguf_iq3, a.gguf_py, a.layer)
    raw = g.raw("gate", 0)                                # [rows, bytes]
    R = raw.shape[0]
    rb = raw.reshape(-1, BLOCK_BYTES)
    d, s, idx, neg, vals = unpack_blocks(rb, grid, ksigns)
    ref = g.deq("gate", 0).reshape(-1, 256)
    e1 = float(np.abs(vals - ref).max())
    back = pack_blocks(d.astype(np.float16), s.astype(np.int64), idx, neg, s2c)
    e2 = int((back != rb).sum())
    # projection of the dequantized PTQ weights with the PTQ scales must reproduce them exactly
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    P = Projector(grid, dev)
    Wd = torch.tensor(ref.reshape(R, -1), device=dev)
    db = (d[:, None] * (0.5 + s) * 0.5).reshape(R, -1)
    W2, parts = P.project(Wd, torch.tensor(d.reshape(R, -1), device=dev), torch.tensor(np.log(np.maximum(db, 1e-12)), device=dev), ste=False)
    e3 = float((W2 - Wd).abs().max())
    b2 = parts_to_bytes(parts, s2c)
    e4 = float(np.abs(unpack_blocks(b2.reshape(-1, BLOCK_BYTES), grid, ksigns)[4].reshape(R, -1) - W2.cpu().numpy()).max())
    print(json.dumps({"decode_vs_ggufpy_maxabs": e1, "repack_byte_mismatch": e2,
                      "project_ptq_maxabs": e3, "pack_roundtrip_maxabs": e4}))


def cmd_train(a):
    import quantize_experts_int4 as qe
    from compensation_probe import layer_bias
    t_start = time.time()
    dev = "cuda"
    grid, ksigns, s2c = load_tables(a.gguf_py)
    P = Projector(grid, dev)
    ck = qe.Checkpoint(a.model)
    gi3 = GGUFLayer(a.gguf_iq3, a.gguf_py, a.layer)
    gq4 = GGUFLayer(a.gguf_q4k, a.gguf_py, a.layer)
    X, LG, DOM = load_layer_captures(a.captures, a.layer - 3)
    sgm = 1 / (1 + np.exp(-LG.astype(np.float32)))
    ids = np.argsort(-(sgm + layer_bias(ck, a.layer)), axis=1)[:, :8]
    w8 = np.take_along_axis(sgm, ids, 1); w8 /= w8.sum(1, keepdims=True)
    rng = np.random.default_rng(0)
    held = (rng.random(DOM.size) < 0.2) if a.holdout == "random" else (DOM == a.holdout)   # "random": smoke tests only
    ev = np.where(held)[0]; ev = np.sort(rng.choice(ev, min(a.eval_tokens, ev.size), replace=False))
    tr_mask = ~held
    doms = {d: int((DOM == d).sum()) for d in sorted(set(DOM.tolist()))}
    Xev = torch.tensor(X[ev].astype(np.float32), device=dev)
    # per-expert train token lists (non-held-out domains)
    tok_e = {e: [] for e in range(288)}
    for t in np.where(tr_mask)[0]:
        for j in range(8):
            tok_e[int(ids[t, j])].append((t, w8[t, j]))
    # teacher experts on GPU (bf16), state on host (fp32 latent + logsig)
    def bf16(L, e):
        g, u, dn = ck.expert(L, e)
        return [torch.tensor(np.asarray(m, np.float32), device=dev).to(torch.bfloat16) for m in (g, u, dn)]
    teach = [bf16(a.layer, e) for e in range(288)]
    lat, lsig, dfix = {}, {}, {}
    for e in range(288):
        lat[e], lsig[e], dfix[e] = [], [], []
        for m, Wt in zip(("gate", "up", "down"), teach[e]):
            raw = gi3.raw(m, e); R = raw.shape[0]
            d, s, _, _, _ = unpack_blocks(raw.reshape(-1, BLOCK_BYTES), grid, ksigns)
            db = (d[:, None] * (0.5 + s) * 0.5).reshape(R, -1)
            # latent init: the llama.cpp IQ3_XXS PTQ solution (importance-weighted grid search), so step 0 == PTQ;
            # --init bf16 starts from the original weights instead
            w0 = torch.tensor(gi3.deq(m, e)) if a.init == "ptq" else Wt.float().cpu()
            lat[e].append(w0); lsig[e].append(torch.tensor(np.log(np.maximum(db, 1e-12))))
            dfix[e].append(torch.tensor(d.reshape(R, -1)))

    def layer_out(get_w, Xs, idx_ev):
        y = torch.zeros_like(Xs)
        for e in range(288):
            sel = np.where((ids[idx_ev] == e).any(1))[0]
            if sel.size == 0:
                continue
            wj = torch.tensor((w8[idx_ev[sel]] * (ids[idx_ev[sel]] == e)).sum(1), device=dev, dtype=torch.float32)
            g, u, dn = get_w(e)
            y[sel] += wj[:, None] * ffn(Xs[sel], g, u, dn)
        return y

    with torch.no_grad():
        y_ref = layer_out(lambda e: [m.float() for m in teach[e]], Xev, ev)
        nref = y_ref.norm()

        def rel(y):
            return float((y - y_ref).norm() / nref)
        q4 = rel(layer_out(lambda e: [torch.tensor(gq4.deq(m, e), device=dev) for m in ("gate", "up", "down")], Xev, ev))
        ptq = rel(layer_out(lambda e: [torch.tensor(gi3.deq(m, e), device=dev) for m in ("gate", "up", "down")], Xev, ev))

        def cur_w(e):
            return [P.project(lat[e][k].to(dev), dfix[e][k].to(dev), lsig[e][k].to(dev), ste=False)[0] for k in range(3)]
        step0 = rel(layer_out(cur_w, Xev, ev))
    log = {"layer": a.layer, "holdout": a.holdout, "init": a.init, "lr_w": a.lr_w, "lr_s": a.lr_s, "batch": a.batch, "tokens_by_domain": doms, "eval_tokens": int(ev.size),
           "train_assignments": int(sum(len(v) for v in tok_e.values())), "budget_s": a.budget_s, "passes": a.passes,
           "rel_err_heldout": {"q4k_ptq": q4, "iq3xxs_ptq_llamacpp": ptq, "iq3xxs_step0": step0}, "curve": []}
    print(json.dumps(log), flush=True)
    t_train0 = time.time()
    per_pass = a.budget_s / a.passes
    prev = step0
    for p in range(a.passes):
        t_pass = time.time()
        order = list(range(288))
        per_e = per_pass / 288
        steps_done = 0
        for e in order:
            toks = tok_e[e]
            if not toks:
                continue
            tt = np.array([t for t, _ in toks]); ww = np.array([w for _, w in toks], np.float32)
            Ws = [lat[e][k].to(dev).requires_grad_(True) for k in range(3)]
            Ds = [dfix[e][k].to(dev) for k in range(3)]
            Ls = [lsig[e][k].to(dev).requires_grad_(True) for k in range(3)]
            opt = torch.optim.Adam([{"params": Ws, "lr": a.lr_w}, {"params": Ls, "lr": a.lr_s}])
            te = time.time()
            while time.time() - te < per_e:
                b = rng.choice(tt.size, min(a.batch, tt.size), replace=False)
                x = torch.tensor(X[tt[b]].astype(np.float32), device=dev)
                wv = torch.tensor(ww[b], device=dev)
                with torch.no_grad():
                    yt = ffn(x, *[m.float() for m in teach[e]])
                Wq = [P.project(Ws[k], Ds[k], Ls[k], ste=True)[0] for k in range(3)]
                ys = ffn(x, *Wq)
                loss = (wv[:, None] * (ys - yt) ** 2).sum() / (wv[:, None] * yt ** 2).sum().clamp_min(1e-12)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                steps_done += 1
            lat[e] = [w.detach().cpu() for w in Ws]
            lsig[e] = [l.detach().cpu() for l in Ls]
        with torch.no_grad():
            cur = rel(layer_out(cur_w, Xev, ev))
        rec = {"pass": p + 1, "budget_frac": round((p + 1) / a.passes, 2), "rel_err_heldout": cur,
               "gap_recovered": round((ptq - cur) / max(ptq - q4, 1e-9), 3), "steps": steps_done,
               "pass_s": round(time.time() - t_pass, 1)}
        log["curve"].append(rec)
        print(json.dumps(rec), flush=True)
        if prev - cur < 0.02 * prev:
            log["stopped"] = f"plateau after pass {p + 1}: improvement {prev - cur:.5f} < 2% of {prev:.5f}"
            break
        prev = cur
    log["gpu_seconds_train"] = round(time.time() - t_train0, 1)
    log["gpu_seconds_total"] = round(time.time() - t_start, 1)
    os.makedirs(a.out, exist_ok=True)
    # export packed IQ3_XXS bytes for the GGUF patch step, with a round-trip check
    exp = {}
    maxerr = 0.0
    for e in range(288):
        for k, m in enumerate(("gate", "up", "down")):
            Wq, parts = P.project(lat[e][k].to(dev), dfix[e][k].to(dev), lsig[e][k].to(dev), ste=False)
            b = parts_to_bytes(parts, s2c)
            if e < 4:
                rt = unpack_blocks(b.reshape(-1, BLOCK_BYTES), grid, ksigns)[4].reshape(Wq.shape)
                maxerr = max(maxerr, float(np.abs(rt - Wq.cpu().numpy()).max()))
            exp[f"{m}_{e}"] = b
    np.savez(os.path.join(a.out, f"iq3xxs_layer{a.layer}_hold-{a.holdout}.npz"), **exp)
    log["export_roundtrip_maxabs"] = maxerr
    json.dump(log, open(os.path.join(a.out, f"qat_layer{a.layer}_hold-{a.holdout}.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in log.items() if k != "curve"}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["selftest", "train"])
    ap.add_argument("--gguf-py", required=True)
    ap.add_argument("--gguf-iq3", required=True)
    ap.add_argument("--gguf-q4k")
    ap.add_argument("--model")
    ap.add_argument("--captures")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--holdout", default="multilingual")
    ap.add_argument("--budget-s", type=float, default=3600)
    ap.add_argument("--passes", type=int, default=5)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr-w", type=float, default=1e-5)
    ap.add_argument("--lr-s", type=float, default=1e-3)
    ap.add_argument("--eval-tokens", type=int, default=2048)
    ap.add_argument("--init", default="ptq", choices=["ptq", "bf16"])
    ap.add_argument("--out", default="qat_out")
    a = ap.parse_args()
    {"selftest": cmd_selftest, "train": cmd_train}[a.cmd](a)


if __name__ == "__main__":
    main()
