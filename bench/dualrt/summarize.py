#!/usr/bin/env python3
"""Summarize bench/dualrt client runs: per workload group (essay, chat, hermes) mean tok/s, committed tokens per
MTP step and ms per cycle, over all reps.   summarize.py RUN_client.jsonl [...]"""
import json, sys, statistics as S
for f in sys.argv[1:]:
    rows = [json.loads(l) for l in open(f) if l.strip()]
    g = {}
    for r in rows:
        k = r["workload"].split(":")[0]; k = "chat4" if k in ("prose", "code", "chat", "reasoning") else k
        g.setdefault(k, []).append(r)
    out = {"run": f.split("/")[-1].replace("_client.jsonl", "")}
    for k, rs in g.items():
        tok = sum(r["tokens"] for r in rs); s = sum(r["s"] for r in rs); d = sum(r["drafts"] for r in rs)
        reps = sorted({r["rep"] for r in rs})
        per_rep = [round(sum(r["tokens"] for r in rs if r["rep"] == p) / sum(r["s"] for r in rs if r["rep"] == p), 2) for p in reps]
        out[k] = {"tok_s": round(tok / s, 2), "per_rep_tok_s": per_rep, "committed_per_step": round(tok / d, 3) if d else None,
                  "ms_per_cycle": round(1e3 * s / d, 1) if d else None, "requests": len(rs)}
    print(json.dumps(out))
