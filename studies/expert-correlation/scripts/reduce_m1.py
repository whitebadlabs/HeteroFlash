#!/usr/bin/env python3
"""Reduce raw GLM-5.3-Flash router-logit captures (HeteroFlash M1 run) to top-8 expert ids.

Input (not in this repo; ~6.4 GB):
  RAW_DIR/r0_<pid>_<chunk>.npz   one row block per (forward, MoE layer):
        fwd[i]    forward-pass counter
        layer[i]  MoE layer index 0..41 (model layer 3 + i)
        ntok[i]   tokens in that forward (1 = decode step, >1 = prefill chunk)
        t[i]      host timestamp (s)
        logits    float16 [sum(ntok), 288]  raw gate (router) output, before bias
  CLIENT_LOG     client_log.jsonl: one line per request with t_send / t_done / domain / split
  BIAS_JSON      e_score_correction_bias per model layer (float32 [288]) from the same checkpoint

Selection is recomputed exactly as GLM's noaux_tc router with n_group = 1:
  s = sigmoid(logits); choice = s + bias; ids = top-8(choice)
(the same rule and the same np.argsort call as the HeteroFlash M1 analyzer).

Forward rows are assigned to requests by timestamp (server ran max-num-seqs 1, client sequential).
Rows outside any request window are startup / probe traffic and are counted as stray.

Output: OUT.npz with, per kept request r and phase p in {prefill, decode}, an array
  ids_<r>_<p>  uint16 [T, 42, 8]   token-major, layer, top-8 ids sorted by choice score (descending)
and OUT.json with request metadata and integrity counters.
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

E, K, L, FIRST = 288, 8, 42, 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--client", required=True)
    ap.add_argument("--bias", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    bj = json.load(open(a.bias))
    bias = np.zeros((L, E), np.float32)
    for k, v in bj.items():
        layer = int(k.split("layers.")[1].split(".")[0])
        if FIRST <= layer < FIRST + L:          # layer 45 is the MTP layer: not in the trace
            bias[layer - FIRST] = np.asarray(v["values"], np.float32)
    assert np.abs(bias).sum(1).min() > 0, "missing bias for some layer"

    reqs = [json.loads(x) for x in open(a.client) if '"type": "request"' in x]
    reqs = [r for r in reqs if not r.get("error")]
    reqs.sort(key=lambda r: r["t_send"])
    t_send = np.array([r["t_send"] for r in reqs])
    t_done = np.array([r["t_done"] for r in reqs])

    per = defaultdict(lambda: [[] for _ in range(L)])   # (req, phase) -> per-layer list of [n,8]
    stray_rows = anomalies = 0
    fwd_layers = defaultdict(set)                        # integrity: each forward should cover all 42 layers
    files = sorted(glob.glob(os.path.join(a.raw, "r0_*.npz")),
                   key=lambda p: (int(p.split("_")[-2]), int(p.split("_")[-1].split(".")[0])))
    for fi, path in enumerate(files):
        d = np.load(path)
        fwd, layer, ntok, t, logits = d["fwd"], d["layer"], d["ntok"], d["t"], d["logits"]
        off = np.concatenate([[0], np.cumsum(ntok)])
        for i in range(len(fwd)):
            li = int(layer[i])
            if li >= L:
                anomalies += 1
                continue
            j = np.searchsorted(t_send, t[i], side="right") - 1
            if j < 0 or t[i] > t_done[j]:
                stray_rows += int(ntok[i])
                continue
            lg = logits[off[i]:off[i + 1]].astype(np.float32)
            c = 1.0 / (1.0 + np.exp(-lg)) + bias[li]
            ids = np.argsort(-c, axis=1)[:, :K].astype(np.uint16)
            ph = "prefill" if ntok[i] > 1 else "decode"
            per[(j, ph)][li].append(ids)
            fwd_layers[(int(fwd[i]), j, ph)].add(li)
        if fi % 25 == 0:
            print(f"{fi}/{len(files)} {os.path.basename(path)}", flush=True)

    incomplete_fwd = sum(1 for v in fwd_layers.values() if len(v) != L)
    arrays, meta, tokens, misaligned = {}, [], {"prefill": 0, "decode": 0}, 0
    for (j, ph), layers in sorted(per.items()):
        lens = [sum(x.shape[0] for x in lst) for lst in layers]
        if len(set(lens)) != 1:
            misaligned += 1
            continue
        arr = np.stack([np.concatenate(lst, 0) for lst in layers], axis=1)   # [T, 42, 8]
        arrays[f"ids_{j}_{ph}"] = arr
        tokens[ph] += arr.shape[0]
    kept = sorted({int(j) for (j, _ph) in per})
    for j in kept:
        r = reqs[j]
        meta.append({"req": int(j), "prompt_id": r["prompt_id"], "domain": r["domain"], "split": r["split"],
                     "prompt_tokens": r.get("prompt_tokens"), "completion_tokens": r.get("completion_tokens"),
                     "prefill_rows": int(arrays[f"ids_{j}_prefill"].shape[0]) if f"ids_{j}_prefill" in arrays else 0,
                     "decode_rows": int(arrays[f"ids_{j}_decode"].shape[0]) if f"ids_{j}_decode" in arrays else 0})
    np.savez(a.out + ".npz", **arrays)
    json.dump({"requests": meta, "tokens": tokens, "stray_rows_outside_requests": stray_rows,
               "layer_index_anomalies": anomalies, "forwards_missing_layers": incomplete_fwd,
               "request_phases_with_unequal_layer_lengths": misaligned, "num_files": len(files),
               "num_experts": E, "top_k": K, "moe_layers": list(range(FIRST, FIRST + L))},
              open(a.out + ".json", "w"), indent=1)
    print(json.dumps({"tokens": tokens, "stray": stray_rows, "anomalies": anomalies,
                      "incomplete_fwd": incomplete_fwd, "misaligned": misaligned, "requests": len(meta)}))


if __name__ == "__main__":
    main()
