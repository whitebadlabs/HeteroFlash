#!/usr/bin/env python3
"""G2 probe: non-linear distilled compensation for dropped experts (torch CPU).

Per MoE layer: R = y8 - yk (routed output of the original top-8 minus top-k with original-top-8
normalisation), from captured inputs + official BF16 experts. Train g(x) = W2 silu(W1 x + b1) + b2
(hidden h) on the first 75% of prompts (internal 10% random validation for early stopping), and
evaluate on the last 25% of prompts (unseen prompts / mostly unseen domains) — the split on which
linear compensation recovered only 1-11%. Report the fraction of the k_ref8 error recovered.
Parameter cost: h * (2 * 4096) per layer.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quantize_experts_int4 as qe  # noqa: E402
from compensation_probe import layer_bias, rel, routed  # noqa: E402
from layer_topk_analysis import expert_outputs, load_captures  # noqa: E402


def train(Xtr, Rtr, Xva, Rva, h, epochs, lr, wd, seed=0):
    torch.manual_seed(seed)
    d = Xtr.shape[1]
    net = torch.nn.Sequential(torch.nn.Linear(d, h), torch.nn.SiLU(), torch.nn.Linear(h, d))
    torch.nn.init.zeros_(net[2].weight); torch.nn.init.zeros_(net[2].bias)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    xs, rs = torch.from_numpy(Xtr), torch.from_numpy(Rtr)
    xv, rv = torch.from_numpy(Xva), torch.from_numpy(Rva)
    best, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        perm = torch.randperm(xs.shape[0])
        for i in range(0, xs.shape[0], 256):
            b = perm[i:i + 256]
            loss = ((net(xs[b]) - rs[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            v = ((net(xv) - rv) ** 2).mean().item()
        if v < best - 1e-9:
            best, best_state, bad = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 5:
                break
    net.load_state_dict(best_state)
    return net, ep + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--captures", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[12, 20, 32])
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--hidden", type=int, nargs="+", default=[256, 1024])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--max-tokens", type=int, default=12000)
    ap.add_argument("--out")
    a = ap.parse_args()
    torch.set_num_threads(24)
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
        w8 = np.take_along_axis(s, order, 1); w8 = w8 / w8.sum(1, keepdims=True) * 2.5
        y8 = routed(outs, order, w8).astype(np.float32)
        yk = routed(outs, order[:, :a.k], w8[:, :a.k]).astype(np.float32)
        R = y8 - yk
        n = X.shape[0]
        cut = int(0.75 * n)
        rng = np.random.default_rng(0)
        idx = rng.permutation(cut)
        va, tr = idx[: cut // 10], idx[cut // 10:]
        te = np.arange(cut, n)
        base = rel(yk[te], y8[te])
        r = {"tokens_train": len(tr), "tokens_test": len(te), f"tail:k{a.k}_ref8": base}
        for h in a.hidden:
            net, eps = train(X[tr], R[tr], X[va], R[va], h, a.epochs, a.lr, a.wd)
            with torch.no_grad():
                pred = net(torch.from_numpy(X[te])).numpy()
            e = rel(yk[te] + pred, y8[te])
            r[f"tail:mlp{h}"] = {"err": e, "recovered_frac": (base - e) / base, "epochs": eps,
                                 "params_M": round(h * 2 * X.shape[1] / 1e6, 2)}
        res[layer] = r
        print(layer, json.dumps(r), flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
