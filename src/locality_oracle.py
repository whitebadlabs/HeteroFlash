#!/usr/bin/env python3
"""Locality oracle: how much can the verify-window expert union shrink if routing may prefer experts already used
in the window, and how much routing mass does that move? Upper-bounds what locality-regularised router distillation
could buy without training.

Input: raw router-logit captures from src/trace_hook (r0_*.npz with fwd, layer, ntok, t, logits[ntok_total, 288]).
Decode steps are forwards with ntok == 1; consecutive decode steps of one request form a stream.
Selection is GLM's: s = sigmoid(logits); choice = s + bias; ids = top-8(choice); w = s[ids] / sum(s[ids]).

For each stream, layer and window of W consecutive decode tokens:
  baseline union = |U| over the original top-8 sets.
  greedy locality: walk the window; for each token, an original expert e not yet in U is replaced by the best candidate
  c already in U (by choice score) when choice[c] >= choice[e] - delta. The routing mass moved is the original
  normalised weight of e (the replacement receives its own sigmoid score in the renormalised gate).
Reports union size and the mean moved mass per token for several delta values.
"""
import argparse
import glob
import json
import os

import numpy as np

E, K = 288, 8


def load_bias(model_dir):
    from safetensors import safe_open
    idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    out = {}
    for key, fn in idx.items():
        if key.endswith("mlp.gate.e_score_correction_bias"):
            layer = int(key.split("layers.")[1].split(".")[0])
            with safe_open(os.path.join(model_dir, fn), "numpy") as f:
                out[layer] = f.get_tensor(key).astype(np.float32)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--windows", type=int, nargs="+", default=[4, 7])
    ap.add_argument("--deltas", type=float, nargs="+", default=[0.0, 0.01, 0.02, 0.05, 0.1])
    ap.add_argument("--max-files", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    bias = load_bias(a.model)
    layers_sorted = sorted(bias)
    # decode rows per layer, in time order: collect (t, fwd, layer, logits_row)
    dec = {}
    files = sorted(glob.glob(os.path.join(a.raw, "r0_*.npz")))
    if a.max_files:
        files = files[:a.max_files]
    for path in files:
        d = np.load(path)
        fwd, layer, ntok, t, logits = d["fwd"], d["layer"], d["ntok"], d["t"], d["logits"]
        off = np.concatenate([[0], np.cumsum(ntok)])
        for i in range(len(fwd)):
            if ntok[i] != 1:
                continue
            dec.setdefault(int(layer[i]), []).append((float(t[i]), int(fwd[i]), logits[off[i]].astype(np.float32)))
    # map capture layer index -> model layer index (captures are numbered over MoE layers in order)
    cap_layers = sorted(dec)
    lmap = {c: layers_sorted[j] for j, c in enumerate(cap_layers)} if len(cap_layers) == len(layers_sorted) else {c: c for c in cap_layers}
    res = {"files": len(files), "layers": len(cap_layers), "windows": {}}
    for W in a.windows:
        acc = {d: {"union": [], "moved": []} for d in a.deltas}
        base_union = []
        for c in cap_layers:
            rows = sorted(dec[c], key=lambda r: r[0])
            b = bias.get(lmap[c], np.zeros(E, np.float32))
            S = 1 / (1 + np.exp(-np.stack([r[2] for r in rows])))
            C = S + b
            fw = np.array([r[1] for r in rows])
            # streams: break where forward ids jump (new request) - consecutive decode forwards differ by 1
            brk = np.where(np.diff(fw) != 1)[0] + 1
            for seg in np.split(np.arange(len(rows)), brk):
                for st in range(0, len(seg) - W + 1, W):
                    idx = seg[st:st + W]
                    tops = [np.argpartition(-C[i], K)[:K] for i in idx]
                    base_union.append(len(set(np.concatenate(tops).tolist())))
                    for dl in a.deltas:
                        U, moved = set(), []
                        for i, top in zip(idx, tops):
                            w = S[i][top] / S[i][top].sum()
                            chosen, m = [], 0.0
                            cand = sorted((u for u in U if u not in top), key=lambda u: -C[i][u])
                            ci = 0
                            for e, we in sorted(zip(top.tolist(), w.tolist()), key=lambda x: C[i][x[0]]):
                                if e in U or ci >= len(cand) or C[i][cand[ci]] < C[i][e] - dl:
                                    chosen.append(e)
                                else:
                                    chosen.append(cand[ci]); ci += 1; m += we
                            U.update(chosen); moved.append(m)
                        acc[dl]["union"].append(len(U)); acc[dl]["moved"].extend(moved)
        res["windows"][W] = {"windows_n": len(base_union), "baseline_union": round(float(np.mean(base_union)), 2),
                             "by_delta": {str(dl): {"union": round(float(np.mean(v["union"])), 2),
                                                     "moved_mass_mean": round(float(np.mean(v["moved"])), 4),
                                                     "moved_mass_p90": round(float(np.percentile(v["moved"], 90)), 4)}
                                          for dl, v in acc.items()}}
        print(json.dumps({W: res["windows"][W]}), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
