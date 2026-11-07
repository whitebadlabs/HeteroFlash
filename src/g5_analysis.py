#!/usr/bin/env python3
"""G5 go/no-go analysis: existing MTP drafter vs draft length (no constant-cycle assumption).

Inputs: /srv/cache/heteroflash/g5/<variant>/decode_{prose,mixed}.jsonl from ops/g5_bench.sh
  serveN : owner's serving config (CUDA graphs) with N draft tokens  -> measured cycle cost vs N on the Control hardware
  mtpN   : eager + hook, target treatment (top-6, original-top-8 normalisation, FP8 fixed weights) -> acceptance vs N
Spark projection: verify-step time for a window of N+1 tokens from decode_sim (real routing + measured costs) plus
N draft passes of the MTP layer on the Spark (draft bytes per pass is an input, estimated from the checkpoint).
Throughput = committed tokens per cycle / (verify(N+1) + N * draft).
"""
import argparse
import json
import os

import numpy as np


def summary(path):
    if not os.path.exists(path):
        return None
    for line in open(path):
        r = json.loads(line)
        if r.get("type") == "summary":
            return r
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g5", default="/srv/cache/heteroflash/g5")
    ap.add_argument("--sim", default="/srv/cache/heteroflash/analysis/sim_g5.json",
                    help="decode_sim output with windows 2,4,6,8 for the target config")
    ap.add_argument("--draft-gb", type=float, default=0.45, help="Spark bytes read per MTP draft pass (GB)")
    ap.add_argument("--spark-gbps", type=float, default=222.0)
    ap.add_argument("--hot-frac", type=float, default=0.5)
    ap.add_argument("--fixed-gb", type=float, default=8.3)
    ap.add_argument("--rtt-ms", type=float, default=0.05)
    ap.add_argument("--out", default="/srv/cache/heteroflash/analysis/g5_go_nogo.json")
    a = ap.parse_args()
    res = {"control_serving": {}, "target_acceptance": {}, "spark_projection": {}}
    xs, ys = [], []
    for n in (1, 3, 5, 7):
        for kind in ("prose", "mixed"):
            s = summary(f"{a.g5}/serve{n}/decode_{kind}.jsonl")
            if s and s.get("spec", {}).get("committed_tokens_per_verify"):
                c = s["spec"]["committed_tokens_per_verify"]
                step = s["tpot_ms_p50"] * c
                res["control_serving"][f"N{n}_{kind}"] = {"tpot_ms_p50": s["tpot_ms_p50"], "committed": round(c, 3),
                                                         "acceptance": round(s["spec"]["draft_acceptance_rate"], 3),
                                                         "cycle_ms": round(step, 2), "tok_s": s["decode_tok_s_mean"]}
                if kind == "prose":
                    xs.append(n); ys.append(step)
            t = summary(f"{a.g5}/mtp{n}/decode_{kind}.jsonl")
            if t and t.get("spec", {}).get("committed_tokens_per_verify"):
                res["target_acceptance"][f"N{n}_{kind}"] = {"committed": round(t["spec"]["committed_tokens_per_verify"], 3),
                                                           "acceptance": round(t["spec"]["draft_acceptance_rate"], 3)}
    if len(xs) >= 2:
        b, c0 = np.polyfit(xs, ys, 1)
        res["control_cycle_fit"] = {"cycle_ms = a + b*N": {"a_ms": round(c0, 2), "b_ms_per_draft_token": round(b, 2)},
                                    "note": "Control (2x Max-Q, graphs): a ~ verify, b ~ marginal draft+verify cost per extra draft token"}
    sim = json.load(open(a.sim))["rows"] if os.path.exists(a.sim) else []
    draft_ms = a.draft_gb / a.spark_gbps * 1e3
    for n in (1, 3, 5, 7):
        w = n + 1
        rows = [r for r in sim if r["window"] == w and r["k_active"] == 6 and abs(r["fixed_GB"] - a.fixed_gb) < 0.05
                and abs(r["hot_frac"] - a.hot_frac) < 1e-6 and abs(r["rtt_ms"] - a.rtt_ms) < 1e-6]
        acc = res["target_acceptance"].get(f"N{n}_prose")
        if not rows or not acc:
            continue
        verify = rows[0]["step_ms_mean"]
        cycle = verify + n * draft_ms
        committed = acc["committed"]
        res["spark_projection"][f"N{n}"] = {"verify_ms": verify, "draft_ms_total": round(n * draft_ms, 2),
                                            "cycle_ms": round(cycle, 2), "committed_prose": committed,
                                            "tok_s_prose": round(committed / cycle * 1e3, 1),
                                            "committed_needed_for_125": round(0.125 * cycle, 2)}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
