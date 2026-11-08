#!/usr/bin/env python3
"""Per-layer quantization sensitivity of the routed experts (pilot layer selection, remote-layer choice).

For each MoE layer: captured inputs X (src/trace_hook capture_inputs), GLM top-8 routing recomputed from the captured
router logits, routed output y = sum_e w_e E_e(x) with the official BF16 experts vs the same with experts dequantized
from GGUF files (e.g. Q4_K reference and IQ3_XXS). Reports ||y_q - y|| / ||y|| (token-weighted over all tokens) per
layer and format. Experts are dequantized one at a time with gguf-py (bounded memory).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quantize_experts_int4 as qe  # noqa: E402
from compensation_probe import layer_bias  # noqa: E402
from layer_topk_analysis import load_captures, silu  # noqa: E402


class GGUFExperts:
    def __init__(self, path, gguf_py):
        sys.path.insert(0, gguf_py)
        import gguf  # noqa: E402
        self.gguf = gguf
        self.r = gguf.GGUFReader(path)
        self.t = {t.name: t for t in self.r.tensors if "_exps.weight" in t.name}

    def expert(self, layer, e):
        out = []
        for m in ("gate", "up", "down"):
            t = self.t[f"blk.{layer}.ffn_{m}_exps.weight"]
            w = self.gguf.quants.dequantize(np.asarray(t.data[e]), t.tensor_type)
            out.append(np.asarray(w, dtype=np.float32))
        return out  # gate [2048,4096], up [2048,4096], down [4096,2048]


def routed(get_expert, layer, X, ids, w):
    y = np.zeros((X.shape[0], X.shape[1]), np.float32)
    need = {}
    for t in range(ids.shape[0]):
        for j, e in enumerate(ids[t]):
            need.setdefault(int(e), []).append((t, j))
    for e, tj in need.items():
        g, u, d = get_expert(layer, e)
        toks = [t for t, _ in tj]
        xs = X[toks]
        o = (silu(xs @ g.T) * (xs @ u.T)) @ d.T
        for k, (t, j) in enumerate(tj):
            y[t] += w[t, j] * o[k]
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures", required=True)
    ap.add_argument("--model", required=True, help="official BF16 safetensors dir")
    ap.add_argument("--gguf", nargs="+", required=True, help="label=path pairs, e.g. q4k=/x.gguf iq3xxs=/y.gguf")
    ap.add_argument("--gguf-py", required=True)
    ap.add_argument("--layers", type=int, nargs="*", default=None)
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ck = qe.Checkpoint(a.model)
    caps = load_captures(a.captures)
    ggufs = {kv.split("=", 1)[0]: GGUFExperts(kv.split("=", 1)[1], a.gguf_py) for kv in a.gguf}
    layers = a.layers or [li + 3 for li in sorted(caps)]
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    rng = np.random.default_rng(0)
    for layer in layers:
        if str(layer) in res:
            continue
        X, lg = caps[layer - 3]
        if X.shape[0] > a.max_tokens:
            sel = np.sort(rng.choice(X.shape[0], a.max_tokens, replace=False))
            X, lg = X[sel], lg[sel]
        X = X.astype(np.float32)
        s = 1 / (1 + np.exp(-lg.astype(np.float32)))
        ids = np.argsort(-(s + layer_bias(ck, layer)), axis=1)[:, :8]
        w = np.take_along_axis(s, ids, 1)
        w = w / w.sum(1, keepdims=True)
        y = routed(lambda L, e: ck.expert(L, e), layer, X, ids, w)
        ny = np.linalg.norm(y)
        row = {"tokens": int(X.shape[0])}
        for lab, gx in ggufs.items():
            yq = routed(gx.expert, layer, X, ids, w)
            row[lab] = round(float(np.linalg.norm(yq - y) / ny), 5)
        res[str(layer)] = row
        json.dump(res, open(a.out, "w"), indent=1)
        print(layer, json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
