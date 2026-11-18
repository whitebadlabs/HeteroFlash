# Prefix-cache-robust speed: per request (workload, index), take the fastest of the rounds, then pool tokens/s.
import json, glob, re, sys, collections
D = "/srv/vllm-research/work/HeteroFlash/results/2026-11-17-frontier"
out = {}
for tag in ("f1", "f2"):
    pts = collections.defaultdict(dict)
    for f in sorted(glob.glob(f"{D}/{tag}_*_r[0-9]_client.jsonl")):
        m = re.match(rf".*/{tag}_(.+)_r(\d)_client.jsonl", f)
        pts[m.group(1)][int(m.group(2))] = [json.loads(l) for l in open(f) if l.strip()]
    for p, runs in pts.items():
        if len(runs) < 2 or not any(k in p for k in ("kdaoff",)) or p.startswith(("coal", "combo", "frs", "alt15")):
            continue
        n = min(len(r) for r in runs.values())
        groups = collections.defaultdict(lambda: [0, 0.0])
        for i in range(n):
            best = min((runs[r][i] for r in runs), key=lambda x: x["s"] / max(x["tokens"], 1))
            k = best["workload"].split(":")[0]; g = "essay" if k.startswith("essay") else ("hermes" if k.startswith("hermes") else "chat4")
            for gg in (g, "all"):
                groups[gg][0] += best["tokens"]; groups[gg][1] += best["s"]
        out[f"{tag}:{p}"] = {"rounds": sorted(runs), **{g: round(t / s, 2) for g, (t, s) in groups.items()}}
for k, v in sorted(out.items(), key=lambda kv: (kv[0].split(":")[1], kv[0])):
    print(k, v)
json.dump(out, open(f"{D}/f2_bestround_speed.json", "w"), indent=1)
