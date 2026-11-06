#!/usr/bin/env python3
"""Trace-driven decode-step simulator for Spark GPU + remote CPU expert placement (numpy).

Replays REAL decode routing (from analyze_raw_trace.py *_decode_ids.npz) of the
held-out requests against a placement chosen on the calibration requests, and
estimates speculative-verify step time from MEASURED component costs. This is a
model built on measured inputs, not an end-to-end measurement.

Per MoE layer, for a verify window of w consecutive tokens:
  U_hot, U_cold = distinct experts of the window on the GPU / CPU tier
  t_gpu  = (fixed_GB / L_fixed + U_hot * expert_GB_gpu) / gpu_GBps
  t_cpu  = U_cold * cpu_us_per_expert + rtt + 2 * w * act_KB / link_GBps     (0 if U_cold == 0)
  t_layer = max(t_gpu, t_cpu)           (tiers run concurrently; no deferral)
Fixed weights of the 3 dense layers + lm_head are spread over the MoE layers
(they are serial GPU work either way). Step time = sum(t_layer) + overhead.
Decode speed = tau / step, where tau = accepted tokens per verify step
(tau <= w; an input, since acceptance depends on the drafter).
"""
import argparse
import itertools
import json

import numpy as np


def load(prefix):
    meta = json.load(open(prefix + "_decode_ids.json"))
    z = np.load(prefix + "_decode_ids.npz")
    return meta, {m["key"]: z[m["key"]] for m in meta}


def hot_sets(meta, seqs, frac, E):
    calib = [seqs[m["key"]] for m in meta if m["split"] == "calib"]
    L = calib[0].shape[1]
    cnt = np.zeros((L, E), np.int64)
    for s in calib:
        for li in range(L):
            np.add.at(cnt[li], s[:, li, :].ravel(), 1)
    m = round(frac * E)
    hot = np.zeros((L, E), bool)
    for li in range(L):
        hot[li, np.argsort(-cnt[li])[:m]] = True
    return hot


def simulate(meta, seqs, hot, w, p, k_active=8, deferral=False):
    """deferral=False: per-layer barrier, t_layer = max(t_gpu, t_cpu).
    deferral=True: cold-expert results are consumed one layer late (CPU and GPU pipelines overlap):
    step = max(sum t_gpu, sum t_cpu) + max single-layer time (pipeline fill). Quality impact NOT modelled."""
    L = hot.shape[0]
    steps = []
    for m in meta:
        if m["split"] != "holdout":
            continue
        s = seqs[m["key"]]
        for t0 in range(0, s.shape[0] - w + 1, w):
            win = s[t0:t0 + w, :, :k_active]         # [w, L, k]; ids are ordered by router preference
            tot = p["overhead_ms"] / 1e3
            cold_layers = 0
            sg = sc = mx = 0.0
            for li in range(L):
                u = np.unique(win[:, li, :])
                h = hot[li, u]
                uh, uc = int(h.sum()), int((~h).sum())
                tg = (p["fixed_GB"] / L + uh * p["expert_GB_gpu"]) / p["gpu_GBps"]
                tc = 0.0
                if uc:
                    cold_layers += 1
                    tc = uc * p["cpu_us"] / 1e6 + p["rtt_ms"] / 1e3 + 2 * w * p["act_KB"] / 1e6 / p["link_GBps"]
                sg += tg; sc += tc; mx = max(mx, tg, tc)
                tot += max(tg, tc)
            if deferral:
                tot = p["overhead_ms"] / 1e3 + max(sg, sc) + mx
            steps.append((tot, cold_layers / L))
    st = np.array([x[0] for x in steps])
    return {"step_ms_mean": round(float(st.mean()) * 1e3, 2), "step_ms_p95": round(float(np.percentile(st, 95)) * 1e3, 2),
            "frac_layers_with_remote_work": round(float(np.mean([x[1] for x in steps])), 3), "steps": len(steps)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True, help="analysis prefix, e.g. .../summary (reads _decode_ids.npz/.json)")
    ap.add_argument("--experts", type=int, default=288)
    ap.add_argument("--fixed-gb", type=float, nargs="*", default=[16.6, 8.3, 4.7], help="fixed bytes per step: BF16 / FP8 / NVFP4")
    ap.add_argument("--hot-frac", type=float, nargs="*", default=[0.5, 0.62, 0.75, 1.0])
    ap.add_argument("--windows", type=int, nargs="*", default=[1, 4, 6])
    ap.add_argument("--tau", type=float, nargs="*", default=[1.0, 2.8, 3.5, 4.4], help="accepted tokens per step (<= window)")
    ap.add_argument("--expert-gb-gpu", type=float, default=0.0110, help="EXL3 ~3.5 bpw expert = 11.0 MB")
    ap.add_argument("--gpu-gbps", type=float, default=222.0)
    ap.add_argument("--cpu-us", type=float, default=73.5)
    ap.add_argument("--rtt-ms", type=float, nargs="*", default=[0.211, 0.05, 0.01], help="measured 1GbE tuned / 10G est / RDMA est")
    ap.add_argument("--link-gbps", type=float, default=0.125, help="GB/s (1 GbE = 0.125)")
    ap.add_argument("--act-kb", type=float, default=8.0)
    ap.add_argument("--overhead-ms", type=float, default=2.0)
    ap.add_argument("--k-active", type=int, nargs="*", default=[8, 6, 4])
    ap.add_argument("--deferral", type=int, nargs="*", default=[0, 1])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    meta, seqs = load(a.prefix)
    rows = []
    for frac in a.hot_frac:
        hot = hot_sets(meta, seqs, frac, a.experts)
        for fixed, w, rtt, ka, dfr in itertools.product(a.fixed_gb, a.windows, a.rtt_ms, a.k_active, a.deferral):
            p = dict(fixed_GB=fixed, expert_GB_gpu=a.expert_gb_gpu, gpu_GBps=a.gpu_gbps, cpu_us=a.cpu_us, rtt_ms=rtt,
                     link_GBps=a.link_gbps if rtt > 0.1 else 1.25, act_KB=a.act_kb, overhead_ms=a.overhead_ms)
            r = simulate(meta, seqs, hot, w, p, ka, bool(dfr))
            spark_gb = frac * a.experts * 42 * a.expert_gb_gpu + fixed
            for tau in a.tau:
                if tau > w:
                    continue
                rows.append({"hot_frac": frac, "spark_resident_GB": round(spark_gb, 1), "fits_spark": spark_gb <= 105, "fixed_GB": fixed, "window": w, "k_active": ka, "deferral": bool(dfr),
                             "rtt_ms": rtt, "tau": tau, **r, "tok_s": round(tau / (r["step_ms_mean"] / 1e3), 1),
                             "tau_needed_for_125": round(125 * r["step_ms_mean"] / 1e3, 2)})
    json.dump({"status": "SIMULATION on measured component costs + real routing traces (not an end-to-end measurement)",
               "params": vars(a), "rows": rows}, open(a.out, "w"), indent=1)
    best = sorted(rows, key=lambda r: -r["tok_s"])[:12]
    for r in best:
        print(r)


if __name__ == "__main__":
    main()
