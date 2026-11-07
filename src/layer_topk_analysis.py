#!/usr/bin/env python3
"""G2 analysis: decompose the top-k reduction loss at the MoE-layer level (numpy, CPU).

For captured MoE inputs x (hook v4 `capture_inputs`: hidden + router logits per layer) and the
official BF16 expert weights, compute the routed-expert output of the original router (top-8,
normalised, x routed_scaling_factor) and compare reconstructions:
  k8           reference (exact original routing)
  k4_active    top-4 by (sigmoid + bias), weights renormalised over the 4 (router default; what M2 measured)
  k4_ref8      top-4, weights normalised over the original top-8 (missing mass dropped, not redistributed)
  k4_oracle_ls top-4 experts, per-token least-squares optimal weights (upper bound for any re-weighting)
  k4_contrib   the 4 of the original 8 with the largest ||w_i E_i(x)|| (oracle selection, original weights,
               renormalised): how much a better *choice* of 4 could recover
  k6_active    top-6 renormalised
Error: relative L2 of the routed output, mean over tokens, per layer. The shared expert is identical
in all variants and is omitted (it would only shrink relative errors).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quantize_experts_int4 as qe  # noqa: E402  (Checkpoint: bf16 reader)


def silu(v):
    return v / (1 + np.exp(-v))


def expert_outputs(ck, layer, X, ids):
    """outs[t][e] for every (token t, expert e in ids[t]); experts loaded one at a time (bounded memory)."""
    need = {}
    for t in range(ids.shape[0]):
        for e in ids[t]:
            need.setdefault(int(e), []).append(t)
    outs = [dict() for _ in range(ids.shape[0])]
    for e, toks in need.items():
        g, u, d = ck.expert(layer, e)
        xs = X[toks]
        y = (silu(xs @ g.T) * (xs @ u.T)) @ d.T
        for j, t in enumerate(toks):
            outs[t][e] = y[j]
        del g, u, d
    return outs, len(need)


def analyse_layer(ck, layer, X, logits, bias, scale=2.5, max_tokens=512):
    X = X[:max_tokens].astype(np.float32)
    lg = logits[:max_tokens].astype(np.float32)
    s = 1 / (1 + np.exp(-lg))
    order = np.argsort(-(s + bias), axis=1)
    outs, nexp = expert_outputs(ck, layer, X, order[:, :8])
    errs = {k: [] for k in ("k4_active", "k4_ref8", "k4_oracle_ls", "k4_contrib", "k6_active",
                            "k4_contrib_ref8", "k6_ref8", "k5_ref8", "k3_ref8", "k2_ref8")}
    for t in range(X.shape[0]):
        o = outs[t]
        ids8 = order[t, :8]
        w8 = s[t, ids8] / s[t, ids8].sum() * scale
        ref = sum(w8[i] * o[int(e)] for i, e in enumerate(ids8))
        nref = np.linalg.norm(ref) + 1e-12

        def comb(ids, w):
            return sum(w[i] * o[int(e)] for i, e in enumerate(ids))
        ids4 = ids8[:4]
        w4 = s[t, ids4] / s[t, ids4].sum() * scale
        errs["k4_active"].append(np.linalg.norm(comb(ids4, w4) - ref) / nref)
        errs["k4_ref8"].append(np.linalg.norm(comb(ids4, w8[:4]) - ref) / nref)
        A = np.stack([o[int(e)] for e in ids4], 1)
        coef, *_ = np.linalg.lstsq(A, ref, rcond=None)
        errs["k4_oracle_ls"].append(np.linalg.norm(A @ coef - ref) / nref)
        contrib = np.array([np.linalg.norm(w8[i] * o[int(e)]) for i, e in enumerate(ids8)])
        idsc = ids8[np.argsort(-contrib)[:4]]
        wc = s[t, idsc] / s[t, idsc].sum() * scale
        errs["k4_contrib"].append(np.linalg.norm(comb(idsc, wc) - ref) / nref)
        ids6 = ids8[:6]
        w6 = s[t, ids6] / s[t, ids6].sum() * scale
        errs["k6_active"].append(np.linalg.norm(comb(ids6, w6) - ref) / nref)
        top_c = np.argsort(-contrib)[:4]
        errs["k4_contrib_ref8"].append(np.linalg.norm(comb(ids8[top_c], w8[top_c]) - ref) / nref)
        for k in (6, 5, 3, 2):
            errs[f"k{k}_ref8"].append(np.linalg.norm(comb(ids8[:k], w8[:k]) - ref) / nref)
    return {k: float(np.mean(v)) for k, v in errs.items()}, nexp


def load_captures(cap_dir, first_moe=3):
    """Group captured inputs by MoE layer index (0..41) -> (hidden, logits)."""
    hid, lgt = {}, {}
    pattern = cap_dir if "*" in cap_dir else os.path.join(cap_dir, "inputs_*.npz")   # dir or explicit glob
    for f in sorted(glob.glob(pattern)):
        with np.load(f) as d:                      # npz re-reads a whole array on every d[key]: load once
            layer, ntok, hidden, logits = d["layer"], d["ntok"], d["hidden"], d["logits"]
        off = np.concatenate([[0], np.cumsum(ntok)])
        for i in range(len(layer)):
            li = int(layer[i])
            hid.setdefault(li, []).append(hidden[off[i]:off[i + 1]].copy())
            lgt.setdefault(li, []).append(logits[off[i]:off[i + 1]].copy())
        del hidden, logits
    return {li: (np.concatenate(hid[li]), np.concatenate(lgt[li])) for li in hid}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--captures", help="dir with inputs_*.npz (hook v4); omit with --synthetic")
    ap.add_argument("--synthetic", action="store_true", help="random inputs + real router weights (pipeline test only)")
    ap.add_argument("--layers", type=int, nargs="+", default=[3, 12, 20, 32, 44])
    ap.add_argument("--all-layers", action="store_true", help="analyse all 42 MoE layers (3..44)")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--out")
    a = ap.parse_args()
    ck = qe.Checkpoint(a.model)
    caps = None if a.synthetic else load_captures(a.captures)
    res = {}
    if a.all_layers:
        a.layers = list(range(3, 45))
    for layer in a.layers:
        name = f"model.language_model.layers.{layer}.mlp.gate.e_score_correction_bias"
        fn = ck.map[name]
        h, base = ck._header(fn)
        meta = h[name]
        with open(os.path.join(ck.d, fn), "rb") as f:
            f.seek(base + meta["data_offsets"][0])
            raw = f.read(meta["data_offsets"][1] - meta["data_offsets"][0])
        bias = np.frombuffer(raw, dtype={"F32": np.float32, "BF16": np.uint16}[meta["dtype"]])
        if meta["dtype"] == "BF16":
            bias = (bias.astype(np.uint32) << 16).view(np.float32)
        if a.synthetic:
            W = ck.bf16(f"model.language_model.layers.{layer}.mlp.gate.weight")
            X = np.random.default_rng(layer).standard_normal((a.max_tokens, 4096)).astype(np.float32) * 0.5
            logits = X @ W.T
        else:
            X, logits = caps[layer - 3]
        r, nexp = analyse_layer(ck, layer, X, logits, bias, max_tokens=a.max_tokens)
        r["experts_loaded"] = nexp
        res[layer] = r
        print(layer, json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}), flush=True)
    if a.out:
        json.dump({"source": "synthetic" if a.synthetic else "captured", "layers": res}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
