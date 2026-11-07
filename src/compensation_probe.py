#!/usr/bin/env python3
"""G2 probe: can a cheap learned term recover the dropped experts' contribution?

Per MoE layer, with captured inputs X and the official BF16 experts:
  y8   = original routed output (top-8, ref normalisation)
  yk   = top-k routed output with original-top-8 normalisation (k_ref = 8)
  R    = y8 - yk                       (the dropped experts' contribution)
Fit on a train split, evaluate on held-out tokens:
  linear compensation   yk + X W          (ridge, full rank)
  low-rank compensation yk + X A B        (reduced-rank ridge, rank r)
Report held-out relative error ||y8 - y_hat|| / ||y8|| for: yk alone, + rank-r, + full-rank.
Parameter cost of rank r: r * (4096 + 4096) values per layer.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quantize_experts_int4 as qe  # noqa: E402
from layer_topk_analysis import expert_outputs, load_captures  # noqa: E402


def layer_bias(ck, layer):
    name = f"model.language_model.layers.{layer}.mlp.gate.e_score_correction_bias"
    fn = ck.map[name]
    h, base = ck._header(fn)
    meta = h[name]
    with open(os.path.join(ck.d, fn), "rb") as f:
        f.seek(base + meta["data_offsets"][0])
        raw = f.read(meta["data_offsets"][1] - meta["data_offsets"][0])
    b = np.frombuffer(raw, dtype={"F32": np.float32, "BF16": np.uint16}[meta["dtype"]])
    return (b.astype(np.uint32) << 16).view(np.float32) if meta["dtype"] == "BF16" else b.astype(np.float32)


def routed(outs, ids, w):
    return np.stack([sum(w[t, i] * outs[t][int(e)] for i, e in enumerate(ids[t])) for t in range(ids.shape[0])])


def rel(a, b):
    return float(np.mean(np.linalg.norm(a - b, axis=1) / (np.linalg.norm(b, axis=1) + 1e-12)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--captures", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[12, 20, 32])
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--ranks", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    ap.add_argument("--ridges", type=float, nargs="+", default=[1.0, 100.0, 1e3, 1e4, 1e5])
    ap.add_argument("--max-tokens", type=int, default=12000)
    ap.add_argument("--ridge", type=float, default=1.0)
    ap.add_argument("--out")
    a = ap.parse_args()
    ck = qe.Checkpoint(a.model)
    caps = load_captures(a.captures)
    res = {}
    for layer in a.layers:
        X, lg = caps[layer - 3]
        X = X[:a.max_tokens].astype(np.float32)
        lg = lg[:a.max_tokens].astype(np.float32)
        s = 1 / (1 + np.exp(-lg))
        order = np.argsort(-(s + layer_bias(ck, layer)), axis=1)[:, :8]
        outs, _ = expert_outputs(ck, layer, X, order)
        w8 = np.take_along_axis(s, order, 1)
        w8 = w8 / w8.sum(1, keepdims=True) * 2.5
        y8 = routed(outs, order, w8)
        yk = routed(outs, order[:, :a.k], w8[:, :a.k])
        R = y8 - yk
        n = X.shape[0]
        r = {f"k{a.k}_ref8_all": rel(yk, y8)}
        rng = np.random.default_rng(0)
        splits = {"random": rng.permutation(n) < int(0.75 * n), "tail": np.arange(n) < int(0.75 * n)}
        for sname, tr in splits.items():
            te = ~tr
            Xtr, Rtr = X[tr], R[tr]
            mu_x, mu_r = Xtr.mean(0), Rtr.mean(0)
            Xc, Rc = Xtr - mu_x, Rtr - mu_r
            XtX, XtR = Xc.T @ Xc, Xc.T @ Rc
            base = rel(yk[te], y8[te])
            r[f"{sname}:k{a.k}_ref8"] = base
            best = (base, "none")
            for lam in a.ridges:
                W = np.linalg.solve(XtX + lam * np.eye(X.shape[1], dtype=np.float32), XtR)
                fit = Xc @ W
                _, _, Vt = np.linalg.svd(fit, full_matrices=False)
                for rk in a.ranks:
                    P = Vt[:rk].T @ Vt[:rk]
                    e = rel(yk[te] + (X[te] - mu_x) @ (W @ P) + mu_r, y8[te])
                    r[f"{sname}:lam{lam:g}:rank{rk}"] = e
                    if e < best[0]:
                        best = (e, f"lam{lam:g}:rank{rk}")
            r[f"{sname}:best"] = {"err": best[0], "cfg": best[1], "recovered_frac": (base - best[0]) / base}
        res[layer] = r
        print(layer, json.dumps({k: v for k, v in r.items() if "best" in k or k.endswith("_ref8") or k.endswith("_all")}), flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
