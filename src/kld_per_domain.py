#!/usr/bin/env python3
"""Per-domain KLD / dNLL / same-top-token from a llama-perplexity --kl-divergence log and the balanced-eval manifest.
The log's per-chunk rows are running means over equal-size chunks, so a block mean is
(k2 * m[k2] - k1 * m[k1]) / (k2 - k1) with k counted in chunks (1-based, inclusive end)."""
import json
import re
import sys

ROW = re.compile(r"^\s*(\d+)\s+([\d.]+)\s+±\s+[\d.]+\s+(-?[\d.]+)\s+±\s+[\d.]+\s+([\d.]+)\s+±\s+[\d.]+\s+([\d.]+)\s+±\s+[\d.]+\s+%\s+([\d.]+)\s+±")


def main(log, manifest):
    m = json.load(open(manifest))
    run = {}
    for line in open(log, errors="replace"):
        g = ROW.match(line)
        if g:
            k = int(g.group(1))
            run[k] = {"dnll": float(g.group(3)), "kld": float(g.group(4)), "top": float(g.group(6))}
    out = {}
    for d in m["domains"]:
        a, b = d["chunks"][0], d["chunks"][1] + 1          # chunks are 0-based in the manifest; rows are 1-based
        if b not in run:
            continue
        prev = run.get(a, {"dnll": 0.0, "kld": 0.0, "top": 0.0}) if a > 0 else {"dnll": 0.0, "kld": 0.0, "top": 0.0}
        out[d["domain"]] = {k: round((b * run[b][k] - a * prev[k]) / (b - a), 5) for k in ("dnll", "kld", "top")}
    last = max(run) if run else 0
    out["_overall"] = run.get(last, {}) | {"chunks": last}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
