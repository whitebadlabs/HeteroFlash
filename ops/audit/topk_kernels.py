import gzip, json, sys, collections
d = json.load(gzip.open(sys.argv[1], "rt")); ev = d["traceEvents"]
k = collections.defaultdict(lambda: [0.0, 0]); 
for e in ev:
    if e.get("ph") == "X" and str(e.get("cat", "")).lower() == "kernel":
        n = e.get("name", "")[:110]; k[n][0] += e.get("dur", 0); k[n][1] += 1
tot = sum(v[0] for v in k.values())
for n, (t, c) in sorted(k.items(), key=lambda x: -x[1][0])[:18]:
    print(f"{t/1e3:8.1f} ms {100*t/tot:5.1f}% n={c:6d} avg={t/c:7.1f}us  {n}")
