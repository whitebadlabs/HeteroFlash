#!/usr/bin/env python3
"""Pool the deferral speed points per name (all 15 requests, tokens / seconds over the rounds given), with wait accounting.
  defer_agg.py DIR TAG [ROUNDS e.g. 2,3]  -> JSON lines per point"""
import collections, glob, json, os, sys
d, tag = sys.argv[1], sys.argv[2]; keep = set(sys.argv[3].split(",")) if len(sys.argv) > 3 else None
agg = collections.defaultdict(lambda: {"tok": 0, "s": 0, "cyc": 0, "blk": [], "rt": [], "srv": [], "rounds": []})
for f in sorted(glob.glob(os.path.join(d, f"{tag}_*_r*_waits.json"))):
    base = os.path.basename(f)[len(tag) + 1:-len("_waits.json")]; name, r = base.rsplit("_r", 1)
    if keep and r not in keep: continue
    rows = [json.loads(l) for l in open(f.replace("_waits.json", "_client.jsonl")) if l.strip()]
    w = json.load(open(f)); a = agg[name]
    a["tok"] += sum(x["tokens"] for x in rows); a["s"] += sum(x["s"] for x in rows); a["cyc"] += sum(x["drafts"] for x in rows)
    a["blk"].append(w["moe_blocked_ms_per_window"]); a["rt"].append(w["decode_round_trip_ms"]); a["srv"].append(w["decode_server_ms"])
    a["rounds"].append((r, w["tok_s"]))
for n, a in agg.items():
    m = lambda v: round(sum(v) / len(v), 3)
    print(json.dumps({"point": n, "tok_s": round(a["tok"] / a["s"], 2), "ms_per_cycle": round(1e3 * a["s"] / a["cyc"], 1),
                      "committed_per_cycle": round(a["tok"] / a["cyc"], 3), "moe_blocked_ms_per_window": m(a["blk"]),
                      "round_trip_ms": m(a["rt"]), "server_ms": m(a["srv"]), "per_round_tok_s": a["rounds"]}))
