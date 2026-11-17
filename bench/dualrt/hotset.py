#!/usr/bin/env python3
"""Hot set per MoE layer from a HF_SPLIT_COUNT dump (decode-window route counts): experts ordered hottest first.
  hotset.py COUNTS.json OUT.json   (also prints route coverage of the top-N for a few N)"""
import json, sys
c = json.load(open(sys.argv[1])); hot = {}; cov = {}
for L, v in c.items():
    order = sorted(range(len(v)), key=lambda e: -v[e]); hot[L] = order; tot = sum(v) or 1
    for n in (64, 96, 120, 144, 160):
        cov.setdefault(n, []).append(sum(v[e] for e in order[:n]) / tot)
json.dump(hot, open(sys.argv[2], "w"))
print(json.dumps({"routes_counted": sum(sum(v) for v in c.values()), "mean_coverage_topN": {n: round(sum(x) / len(x), 3) for n, x in cov.items()}}))
