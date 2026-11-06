#!/usr/bin/env python3
"""Convert raw router-logit captures (trace_hook) into routing statistics and a
schema-v1 trace (router_trace.py). Requires numpy + safetensors.

Inputs
  --raw DIR        r0_<pid>_<chunk>.npz from src/trace_hook/sitecustomize.py
  --client LOG     client_log.jsonl from src/run_trace_client.py (t_send/t_done per request)
  --model DIR      checkpoint dir (reads model.safetensors.index.json + gate e_score_correction_bias)
Outputs
  --summary JSON   statistics (see keys below); --trace OUT.jsonl.gz optional full schema-v1 trace

Selection is recomputed exactly as Glm5NextTextTopkRouter (n_group = 1):
  s = sigmoid(logits); choice = s + bias; ids = top-k(choice); w = s[ids] / sum(s[ids]) * scaling.
Forward passes are assigned to requests by timestamp (server runs max-num-seqs 1; client is sequential).
Forwards with ntok > 1 are prefill chunks, ntok == 1 are decode steps.
"""
import argparse
import glob
import hashlib
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_bias(model_dir, layers):
    from safetensors import safe_open
    idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    out = {}
    for l in layers:
        key = next(k for k in idx if k.endswith(f"layers.{l}.mlp.gate.e_score_correction_bias"))
        with safe_open(os.path.join(model_dir, idx[key]), "numpy") as f:
            out[l] = f.get_tensor(key).astype(np.float32)
    return out


def model_revision(model_dir):
    h = hashlib.sha256()
    for name in ("config.json", "materialization-receipt.json", "model.safetensors.index.json"):
        p = os.path.join(model_dir, name)
        if os.path.exists(p):
            h.update(open(p, "rb").read())
    return "sha256:" + h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--client", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--trace")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--topn", type=int, default=16)
    ap.add_argument("--scaling", type=float, default=2.5)
    ap.add_argument("--first-moe-layer", type=int, default=3)
    a = ap.parse_args()

    cfg = json.load(open(os.path.join(a.model, "config.json")))
    tc = cfg.get("text_config", cfg)
    E, K = tc["n_routed_experts"], a.top_k
    first = a.first_moe_layer
    moe_layers = list(range(first, tc["num_hidden_layers"]))
    L = len(moe_layers)
    bias = load_bias(a.model, moe_layers)

    reqs = [json.loads(x) for x in open(a.client) if '"type": "request"' in x]
    reqs = [r for r in reqs if not r.get("error")]
    reqs.sort(key=lambda r: r["t_send"])
    t_send = np.array([r["t_send"] for r in reqs])
    t_done = np.array([r["t_done"] for r in reqs])

    # gather per-forward data: fwd -> list of (layer_idx, ntok, t, logits)
    per_req = defaultdict(lambda: {"prefill": [[] for _ in range(L)], "decode": [[] for _ in range(L)]})
    stray_rows = 0
    anomalies = 0
    for path in sorted(glob.glob(os.path.join(a.raw, "r0_*.npz")),
                       key=lambda p: (int(p.split("_")[-2]), int(p.split("_")[-1].split(".")[0]))):
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
                stray_rows += int(ntok[i])        # startup profiling / probe / health traffic
                continue
            ph = "prefill" if ntok[i] > 1 else "decode"
            per_req[j][ph][li].append(logits[off[i]:off[i + 1]])

    writer = None
    if a.trace:
        from router_trace import TraceMeta, TraceWriter
        writer = TraceWriter(a.trace, TraceMeta(
            model_id="GLM-5.3-Flash EXL3 TR3 3.5bpw (mixed K3/K4)", model_revision=model_revision(a.model),
            engine="vllm-glm53f-exl3-sm120-turbo:r6 eager, no MTP, no prefix cache, TP2",
            source="real", num_experts=E, top_k=K, moe_layers=moe_layers, routed_scaling_factor=a.scaling,
            norm_topk_prob=True, topn_recorded=a.topn,
            notes="router logits captured by trace_hook; selection recomputed with checkpoint e_score_correction_bias"))

    counts = {ph: np.zeros((L, E), np.int64) for ph in ("prefill", "decode")}
    dom_counts = defaultdict(lambda: np.zeros((L, E), np.int64))
    split_counts = defaultdict(lambda: np.zeros((L, E), np.int64))
    dec_ids = defaultdict(dict)            # req -> layer -> (T x K) decode ids in order
    top4_mass = []
    tokens = {"prefill": 0, "decode": 0}
    for j, phases in per_req.items():
        r = reqs[j]
        for ph in ("prefill", "decode"):
            for li in range(L):
                if not phases[ph][li]:
                    continue
                lg = np.concatenate(phases[ph][li], axis=0).astype(np.float32)
                s = 1.0 / (1.0 + np.exp(-lg))
                c = s + bias[moe_layers[li]]
                topn = np.argsort(-c, axis=1)[:, :a.topn]
                ids = topn[:, :K]
                w = np.take_along_axis(s, ids, 1)
                w = w / w.sum(1, keepdims=True) * a.scaling
                np.add.at(counts[ph][li], ids.ravel(), 1)
                np.add.at(dom_counts[r["domain"]][li], ids.ravel(), 1)
                np.add.at(split_counts[r["split"]][li], ids.ravel(), 1)
                ws = np.sort(w, axis=1)[:, ::-1]
                top4_mass.append((ws[:, :4].sum(1) / ws.sum(1)))
                if li == 0:
                    tokens[ph] += lg.shape[0]
                if ph == "decode":
                    dec_ids[j][li] = ids
                if writer:
                    writer.record(request_id=r["prompt_id"], domain=r["domain"], phase=ph,
                                  layer=moe_layers[li], token_start=0, expert_ids=ids.tolist(),
                                  gate_weights=w.tolist(), topn_ids=topn.tolist(),
                                  topn_scores=np.take_along_axis(s, topn, 1).tolist())
    if writer:
        writer.close()

    def cov_curve(cnt, fracs):
        out = {}
        for f in fracs:
            m = round(f * E)
            srt = -np.sort(-cnt, axis=1)
            per_layer = srt[:, :m].sum(1) / np.maximum(cnt.sum(1), 1)
            out[str(f)] = {"mean": round(float(per_layer.mean()), 4), "min": round(float(per_layer.min()), 4),
                           "max": round(float(per_layer.max()), 4)}
        return out

    fracs = (0.1, 0.25, 0.35, 0.5, 0.62, 0.75, 0.9)
    dec = counts["decode"]
    summary = {
        "status": "MEASURED (real router logits; selection recomputed with checkpoint bias)",
        "model_revision": model_revision(a.model), "requests": len(per_req), "tokens": tokens,
        "stray_rows_outside_requests": stray_rows, "layer_index_anomalies": anomalies,
        "coverage_decode": cov_curve(dec, fracs), "coverage_prefill": cov_curve(counts["prefill"], fracs),
        "per_layer_coverage_decode_at_0.5": [round(float(x), 4) for x in
                                             (-np.sort(-dec, 1))[:, :E // 2].sum(1) / np.maximum(dec.sum(1), 1)],
        "experts_never_selected_decode_per_layer": [int(x) for x in (dec == 0).sum(1)],
    }
    # all-hot probability per resident fraction (hot set from decode counts)
    allhot = {}
    for f in (0.35, 0.5, 0.62, 0.75):
        m = round(f * E)
        hot = np.zeros((L, E), bool)
        for li in range(L):
            hot[li, np.argsort(-dec[li])[:m]] = True
        hits = tot = 0
        for j, lay in dec_ids.items():
            for li, ids in lay.items():
                hits += int(hot[li][ids].all(1).sum())
                tot += ids.shape[0]
        allhot[str(f)] = round(hits / max(tot, 1), 4)
    summary["p_all8_hot_decode"] = allhot
    # speculative window union (decode, consecutive tokens)
    unions = {}
    for wsz in (1, 2, 4, 6):
        vals = []
        for j, lay in dec_ids.items():
            for li, ids in lay.items():
                for s0 in range(0, ids.shape[0] - wsz + 1, wsz):
                    vals.append(len(np.unique(ids[s0:s0 + wsz])))
        unions[str(wsz)] = round(float(np.mean(vals)), 3) if vals else None
    summary["decode_window_union_mean_distinct_experts_per_layer"] = unions
    # calib -> holdout transfer: hot set from calib, coverage on holdout
    if "calib" in split_counts and "holdout" in split_counts:
        tr = {}
        for f in (0.35, 0.5, 0.62):
            m = round(f * E)
            cov = []
            for li in range(L):
                hot = np.argsort(-split_counts["calib"][li])[:m]
                h = split_counts["holdout"][li]
                cov.append(h[hot].sum() / max(h.sum(), 1))
            tr[str(f)] = round(float(np.mean(cov)), 4)
        summary["holdout_coverage_with_calib_hot_set"] = tr
    # domain hot-set overlap (top 25% per layer), mean Jaccard over layers
    doms = sorted(dom_counts)
    m = round(0.25 * E)
    jac = {}
    for x in range(len(doms)):
        for y in range(x + 1, len(doms)):
            vals = []
            for li in range(L):
                A = set(np.argsort(-dom_counts[doms[x]][li])[:m].tolist())
                B = set(np.argsort(-dom_counts[doms[y]][li])[:m].tolist())
                vals.append(len(A & B) / len(A | B))
            jac[f"{doms[x]}|{doms[y]}"] = round(float(np.mean(vals)), 3)
    summary["domain_hotset_jaccard_top25pct"] = jac
    if top4_mass:
        tm = np.concatenate(top4_mass)
        summary["top4_share_of_top8_gate_mass"] = {"mean": round(float(tm.mean()), 4),
                                                  "p10": round(float(np.percentile(tm, 10)), 4),
                                                  "p90": round(float(np.percentile(tm, 90)), 4)}
    json.dump(summary, open(a.summary, "w"), indent=1)
    # per-request decode routing sequences for trace-driven simulation: [T, L, K] int16
    seqs, meta = {}, []
    for j, lay in dec_ids.items():
        if len(lay) != L:
            continue
        T = min(v.shape[0] for v in lay.values())
        seqs[f"r{j}"] = np.stack([lay[li][:T] for li in range(L)], axis=1).astype(np.int16)
        meta.append({"key": f"r{j}", "prompt_id": reqs[j]["prompt_id"], "domain": reqs[j]["domain"], "split": reqs[j]["split"], "T": int(T)})
    np.savez_compressed(os.path.splitext(a.summary)[0] + "_decode_ids.npz", **seqs)
    json.dump(meta, open(os.path.splitext(a.summary)[0] + "_decode_ids.json", "w"))
    np.savez_compressed(os.path.splitext(a.summary)[0] + "_counts.npz", decode=dec, prefill=counts["prefill"],
                        **{f"domain_{k}": v for k, v in dom_counts.items()})
    print(json.dumps({k: summary[k] for k in ("requests", "tokens", "coverage_decode", "p_all8_hot_decode",
                                              "decode_window_union_mean_distinct_experts_per_layer")}, indent=1))


if __name__ == "__main__":
    main()
