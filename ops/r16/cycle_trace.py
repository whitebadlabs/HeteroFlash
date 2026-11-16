#!/usr/bin/env python3
"""Per-cycle GPU time by kernel category from a torch-profiler trace of a decode-only window.
  cycle_trace.py TRACE.json.gz CYCLES   (CYCLES = spec-decode drafts counted during the profiled request)
Categories reuse ops/audit/trace_breakdown.py. Prints span, busy, idle, and ms per cycle per category as JSON."""
import gzip, json, sys, collections, importlib.util, os
spec = importlib.util.spec_from_file_location("tb", os.path.join(os.path.dirname(__file__), "..", "audit", "trace_breakdown.py"))
tb = importlib.util.module_from_spec(spec); spec.loader.exec_module(tb)
tr, cyc = sys.argv[1], float(sys.argv[2])
d = json.load(gzip.open(tr, "rt")); ev = d["traceEvents"] if isinstance(d, dict) else d
k = sorted((e["ts"], e.get("dur", 0), e.get("name", "")) for e in ev
           if e.get("ph") == "X" and str(e.get("cat", "")).lower() in ("kernel", "gpu_memcpy", "gpu_memset"))
span = (max(a + b for a, b, _ in k) - k[0][0]) / 1e3; busy = tb.union_len([(a, a + b) for a, b, _ in k]) / 1e3
by = collections.defaultdict(float); top = collections.defaultdict(lambda: collections.defaultdict(float))
for a, b, n in k: c = tb.cat(n); by[c] += b / 1e3; top[c][n[:60]] += b / 1e3
out = {"trace": os.path.basename(tr), "cycles": cyc, "span_ms": round(span, 1), "gpu_busy_ms": round(busy, 1),
       "busy_frac": round(busy / span, 3), "ms_per_cycle_wall": round(span / cyc, 2), "ms_per_cycle_busy": round(busy / cyc, 2),
       "ms_per_cycle_by_cat": {c: round(v / cyc, 2) for c, v in sorted(by.items(), key=lambda x: -x[1])},
       "top_kernels": {c: [[n, round(t / cyc, 3)] for n, t in sorted(top[c].items(), key=lambda x: -x[1])[:4]] for c in by}}
print(json.dumps(out, indent=1))
