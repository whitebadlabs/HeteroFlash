#!/usr/bin/env python3
"""Headline numbers from analyze.py results -> markdown. Usage: summarize.py NAME=RESULTS.json [...] > numbers.md"""
import json
import sys

import numpy as np


def m(x):
    return float(np.mean(x))


for kv in sys.argv[1:]:
    name, path = kv.split("=", 1)
    R = json.load(open(path))
    D, K, E = R["domains"], R["K"], R["E"]
    cp, nl, runs = R["coselect_partner"], R["next_layer_recall"], R["sim"]["runs"]
    print(f"## {name} ({E} experts, top-{K}, {R['layers']} MoE layers, task types: {', '.join(D)})\n")
    t = R["tokens"]
    print("Tokens used (calib fit / holdout test): " + "; ".join(
        f"{d} {t[d]['train_used']}/{t[d]['test_used']}" + (f" ({t[d]['requests_holdout']} holdout requests)"
                                                         if t[d]['requests_holdout'] is not None else "")
        for d in D) + "\n")
    ls = R["load_skew"]["mix"]
    print(f"- Load skew (mixed tasks, mean over layers): busiest expert {ls['max_over_mean']:.2f}x the mean; "
          f"top 10% of experts get {100 * ls['top10pct_share']:.1f}% of picks; CV {ls['cv']:.2f}")
    diag = [cp[f"{d}->{d}"] for d in D]
    off = [cp[f"{a}->{b}"] for a in D for b in D if a != b]
    print(f"- Within-layer co-selection, top-7 partners per expert, measured on holdout: "
          f"P(partner | expert) {m([v['p_partner_given_i'] for v in diag]):.3f} vs P(partner) "
          f"{m([v['p_partner_marginal'] for v in diag]):.3f}; lift same task {m([v['lift'] for v in diag]):.2f}, "
          f"partners from another task {m([v['lift'] for v in off]):.2f}, mixed->mixed {cp['mix->mix']['lift']:.2f}")
    lpl = np.array([cp[f'{d}->{d}']['lift_per_layer'] for d in D]).mean(0)
    ml = R.get("moe_layers") or list(range(R["layers"]))
    print(f"  - lift per layer (same task, mean over tasks): min {lpl.min():.2f} (layer {ml[int(lpl.argmin())]}), "
          f"max {lpl.max():.2f} (layer {ml[int(lpl.argmax())]}), first {lpl[0]:.2f}, last {lpl[-1]:.2f}")
    sd = [nl[f"{d}->{d}"] for d in D]
    xd = [nl[f"{a}->{b}"] for a in D for b in D if a != b]
    print(f"- Layer N -> N+1 recall@{K}: same task {m([v['transition'] for v in sd]):.3f}, other task "
          f"{m([v['transition'] for v in xd]):.3f}, mixed {nl['mix->mix']['transition']:.3f}; popularity baseline "
          f"{m([v['popularity'] for v in sd]):.3f}; random {K / E:.3f}")
    ph = R["next_layer_recall_by_phase"]
    print(f"  - mixed fit by phase: prefill {ph['prefill']['transition']:.3f} ({ph['prefill']['tokens']} tok), "
          f"decode {ph['decode']['transition']:.3f} ({ph['decode']['tokens']} tok)")
    rec = np.array([nl[f'{d}->{d}']['transition_per_layer'] for d in D]).mean(0)
    print(f"  - per layer (same task): min {rec.min():.3f}, max {rec.max():.3f}")
    print(f"- Per-task recall@{K} (fit row -> test column):\n")
    print("| fit \\ test | " + " | ".join(D) + " |")
    print("|---" * (len(D) + 1) + "|")
    for a in D:
        print(f"| {a} | " + " | ".join(f"{nl[f'{a}->{b}']['transition']:.3f}" for b in D) + " |")
    print()
    Gs = sorted({int(k.split("|")[0][1:]) for k in runs})
    print("Simulation, holdout replay, mean over task types. sends = cross-GPU sends per token per layer; "
          "imb = max/mean GPU load over the whole holdout set; imb_b = same per 2048-token batch.\n")
    print("| GPUs | placement | sends | vs Default | distinct GPUs | imb | imb_b |")
    print("|---|---|---|---|---|---|---|")

    def row(G, fit, key, per=False):
        vals = [runs.get(f"G{G}|{d if per else fit}|{key}|{d}") for d in D]
        vals = [v for v in vals if v]
        if not vals:
            return None
        return {k: m([v[k] for v in vals if v.get(k) is not None]) for k in ("sends", "distinct", "imbalance", "imbalance_batch")}

    for G in Gs:
        base = row(G, "mix", "default")
        for lbl, fit, key, per in (("random equal-size", "mix", "random_0", False),
                                   ("DefaultEplbPolicy, mixed load", "mix", "default", False),
                                   ("DefaultEplbPolicy, per-task load", None, "default", True),
                                   ("correlated, mixed fit, cap 1.0x", "mix", "correlated_s1.0", False),
                                   ("correlated, mixed fit, cap 1.1x", "mix", "correlated_s1.1", False),
                                   ("correlated, per-task fit, cap 1.0x", None, "correlated_s1.0", True)):
            r = row(G, fit, key, per)
            if r:
                print(f"| {G} | {lbl} | {r['sends']:.3f} | {100 * (r['sends'] / base['sends'] - 1):+.1f}% | "
                      f"{r['distinct']:.3f} | {r['imbalance']:.3f} | {r['imbalance_batch']:.3f} |")
        o1, o2 = runs.get(f"G{G}|holdout-in-sample|oracle_default|mix"), runs.get(f"G{G}|holdout-in-sample|oracle_correlated_s1.0|mix")
        if o1 and o2:
            print(f"| {G} | oracle (fit on the test tokens, mixed set only): Default -> correlated | "
                  f"{o1['sends']:.3f} -> {o2['sends']:.3f} | {100 * (o2['sends'] / o1['sends'] - 1):+.1f}% | | "
                  f"{o1['imbalance']:.3f} -> {o2['imbalance']:.3f} | |")
    sw = [x for k, v in runs.items() if "swaps_per_layer" in v and k.endswith("|mix") for x in v["swaps_per_layer"]]
    if sw:
        print(f"\nCorrelated search: swaps per layer max {max(sw)}, mean {m(sw):.0f} (cap 5000; "
              f"{sum(x >= 5000 for x in sw)} layers hit the cap)")
    print(f"\nAnalytic distinct GPUs for a random placement: " +
          ", ".join(f"{G}: {v:.3f}" for G, v in R["sim"]["random_analytic_distinct"].items()) + "\n")
