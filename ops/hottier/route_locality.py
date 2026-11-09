"""Hot-expert tier experiment, item 5/6: routing locality and blocking fractions for layer 32 (MoE 29).

Hot sets are chosen ONLY from the M1 eval traces (l32_route_counts.json, disjoint from everything scored here) and
evaluated on (a) the M1 held-out test tokens, (b) Hermes-suite first turns (prefill and decode separately),
(c) long-context prompts (20-43K tokens). Reports per workload and N: route coverage, fraction of layer executions
that block on the CPU (any cold route) for T=1 decode and T=4 verify windows, cold routes per token, and temporal
locality (share of decode routes whose expert was used in the previous 16 / 64 decode tokens of the same request).
Usage: python3 route_locality.py TRACE_DIR COUNTS_JSON CAPTURE_LOG OUT_JSON
"""
import glob, json, sys
import numpy as np

TR, COUNTS, LOG, OUT = sys.argv[1:5]
E, TOPK, MOE = 288, 8, 29
NS = (288, 256, 232, 216, 200, 184, 160, 128, 96, 64)
rc = json.load(open(COUNTS))
total = np.sum([np.array(v) for v in rc["counts"].values()], 0)
order = np.argsort(-total)
bias = None
items = [json.loads(l) for l in open(LOG)]


def routes(tag):
    """-> list of (ntok, topk ids [ntok,8]) per forward, in order."""
    out = []
    for f in sorted(glob.glob(f"{TR}/inputs_{tag}_*.npz"), key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0])):
        z = np.load(f); L, n = z["layer"], z["ntok"]; off = np.concatenate([[0], np.cumsum(n)])
        lg = z["logits"]
        for j in np.nonzero(L == MOE)[0]:
            s = 1 / (1 + np.exp(-lg[off[j]:off[j + 1]].astype(np.float32)))
            out.append((int(n[j]), s))
    return out


# the router bias is not in the trace; recover it from the M1 count file's companion if present, else from the model
bias = np.array(json.load(open(COUNTS.replace("l32_route_counts.json", "l32_bias.json"))))


def topk(s):
    return np.argpartition(-(s + bias), TOPK, axis=1)[:, :TOPK]


work = {}
for it in items:
    if it.get("error"): continue
    fw = routes(it["tag"])
    pre = [topk(s) for n, s in fw if n > 1]
    dec = [topk(s) for n, s in fw if n == 1]
    key = "long_context" if it["category"] == "long_context" else "hermes"
    w = work.setdefault(key, {"prefill": [], "decode": [], "decode_seqs": [], "items": 0, "prompt_tokens": 0})
    w["items"] += 1; w["prompt_tokens"] += it.get("prompt_tokens") or 0
    if pre: w["prefill"].append(np.concatenate(pre))
    if dec:
        d = np.concatenate(dec); w["decode"].append(d); w["decode_seqs"].append(d)
    w.setdefault("by_category", {}).setdefault(it["category"], []).append(np.concatenate(dec) if dec else None)


def metrics(ids_list, hotmask, seqs=False):
    if not ids_list: return None
    ids = np.concatenate(ids_list)
    hot = hotmask[ids]
    r = {"tokens": int(ids.shape[0]), "coverage": round(float(hot.mean()), 4),
         "cold_routes_per_token": round(float((~hot).sum(1).mean()), 3),
         "block_T1": round(float((~hot).any(1).mean()), 4)}
    if seqs:   # T=4 verify windows within each request
        b4 = []
        for d in ids_list:
            h = hotmask[d]
            for s in range(0, d.shape[0] - 3, 4): b4.append((~h[s:s + 4]).any())
        r["block_T4"] = round(float(np.mean(b4)), 4) if b4 else None
    return r


res = {"label": "routing trace analysis, layer 32, hot sets chosen on disjoint M1 traces",
       "workloads": {k: {"items": v["items"], "prompt_tokens": v["prompt_tokens"],
                         "prefill_tokens": int(sum(a.shape[0] for a in v["prefill"])),
                         "decode_tokens": int(sum(a.shape[0] for a in v["decode"]))} for k, v in work.items()},
       "by_N": {}}
for N in NS:
    hm = np.zeros(E, bool); hm[order[:N]] = True
    row = {}
    for k, v in work.items():
        row[k] = {"prefill": metrics(v["prefill"], hm), "decode": metrics(v["decode_seqs"], hm, seqs=True)}
        row[k]["decode_by_category"] = {c: metrics([a for a in arr if a is not None], hm, seqs=True)
                                        for c, arr in v.get("by_category", {}).items()}
    res["by_N"][N] = row
# temporal locality of decode routes (independent of N): reuse within the last W decode tokens of the same request
for k, v in work.items():
    loc = {}
    for W in (4, 16, 64):
        hit, tot = 0, 0
        for d in v["decode_seqs"]:
            for t in range(W, d.shape[0]):
                recent = set(d[t - W:t].reshape(-1).tolist())
                hit += sum(int(e) in recent for e in d[t]); tot += TOPK
        loc[f"reuse_last_{W}"] = round(hit / tot, 4) if tot else None
    # global popularity agreement: rank correlation of this workload's expert frequencies with the M1 counts
    cnt = np.bincount(np.concatenate(v["prefill"] + v["decode"]).reshape(-1), minlength=E)
    rk = lambda a: np.argsort(np.argsort(-a))
    loc["spearman_vs_M1_popularity"] = round(float(np.corrcoef(rk(cnt), rk(total))[0, 1]), 4)
    res["workloads"][k]["decode_temporal_locality"] = loc
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res["workloads"], indent=1))
for N in (256, 216, 200, 160):
    print(N, json.dumps({k: {"pre_cov": v["prefill"]["coverage"] if v["prefill"] else None,
                             "dec_cov": v["decode"]["coverage"] if v["decode"] else None,
                             "block_T1": v["decode"]["block_T1"] if v["decode"] else None,
                             "block_T4": v["decode"]["block_T4"] if v["decode"] else None}
                         for k, v in res["by_N"][N].items()}))
print("LOCALITY_DONE")
