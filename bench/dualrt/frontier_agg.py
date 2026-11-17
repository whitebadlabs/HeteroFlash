#!/usr/bin/env python3
"""Pool frontier speed points over rounds: frontier_agg.py DIR TAG -> per point, per workload group: tok/s (pooled),
per-round tok/s, committed tokens per cycle, ms per cycle; plus all 15 requests pooled."""
import glob, json, re, sys, collections
D, TAG = sys.argv[1], sys.argv[2]
pts = collections.defaultdict(list)
for f in sorted(glob.glob(f"{D}/{TAG}_*_r[0-9]_client.jsonl")):
    m = re.match(rf".*/{TAG}_(.+)_r(\d)_client.jsonl", f); pts[m.group(1)].append((int(m.group(2)), [json.loads(l) for l in open(f) if l.strip()]))
out = {}
for p, runs in pts.items():
    o = {"rounds": sorted(r for r, _ in runs)}
    allrows = [x for _, rs in runs for x in rs]
    for g in ("essay", "chat4", "hermes", "all"):
        def grp(r):
            k = r["workload"].split(":")[0]; k = "chat4" if k in ("prose", "code", "chat", "reasoning") else ("essay" if k.startswith("essay") else k)
            return g == "all" or k == g
        rs = [r for r in allrows if grp(r)]
        if not rs: continue
        tok = sum(r["tokens"] for r in rs); s = sum(r["s"] for r in rs); d = sum(r["drafts"] for r in rs)
        per = [round(sum(r["tokens"] for r in x if grp(r)) / max(sum(r["s"] for r in x if grp(r)), 1e-9), 2) for _, x in sorted(runs)]
        o[g] = {"tok_s": round(tok / s, 2), "per_round": per, "commit": round(tok / d, 3), "ms_cycle": round(1e3 * s / d, 1)}
    out[p] = o
for p, o in sorted(out.items(), key=lambda kv: -kv[1]["all"]["tok_s"]):
    rd, gs = o["rounds"], [g for g in ("essay", "chat4", "hermes", "all") if g in o]
    print(f"{p:28s} r{rd}  " + "  ".join("%s %5.2f (%.2f/%5.1fms)" % (g, o[g]["tok_s"], o[g]["commit"], o[g]["ms_cycle"]) for g in gs) + "  per-round all %s" % o["all"]["per_round"])
json.dump(out, open(f"{D}/{TAG}_frontier_speed.json", "w"), indent=1)
