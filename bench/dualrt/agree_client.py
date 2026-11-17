#!/usr/bin/env python3
"""Teacher-forced agreement probe (Bet 1 runtime, 2026-11-17). For fixed texts (prompt + a reference continuation),
the server scores every position (prompt_logprobs = 5). Run against an `agree`-mode server (no MTP, eager,
max-num-batched-tokens 16, so every position goes through the decode-size path). Writes one JSON per run:
  agree_client.py TEXTS.json OUT.json      TEXTS.json = [{"id":..., "text":...}, ...]
Compare two outputs with: agree_client.py --compare A.json B.json"""
import json, math, sys, urllib.request
U = "http://127.0.0.1:8095"
def post(path, body):
    r = urllib.request.Request(U + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=7200))
if sys.argv[1] == "--compare":
    A, B = json.load(open(sys.argv[2])), json.load(open(sys.argv[3]))
    n = same = 0; dlp = []; flips = []
    for a, b in zip(A, B):
        assert a["ids"] == b["ids"]
        for i, (pa, pb) in enumerate(zip(a["pos"], b["pos"])):
            if pa is None or pb is None: continue
            n += 1; same += pa["top1"] == pb["top1"]; dlp.append(abs(pa["lp"] - pb["lp"]))
            if pa["top1"] != pb["top1"] and len(flips) < 10: flips.append((a["id"], i, pa["top1_lp"], pb["top1_lp"]))
    dlp.sort()
    print(json.dumps({"positions": n, "top1_agree": round(same / n, 5), "mean_abs_dlogprob_ref_token": round(sum(dlp) / n, 6),
                      "p99_abs_dlogprob": round(dlp[int(0.99 * (n - 1))], 5), "max_abs_dlogprob": round(dlp[-1], 5), "first_flips": flips}))
    raise SystemExit(0)
texts = json.load(open(sys.argv[1])); out = []
for t in texts:
    ids = post("/tokenize", {"model": "GLM-5.3-Flash", "prompt": t["text"]})["tokens"]
    o = post("/v1/completions", {"model": "GLM-5.3-Flash", "prompt": ids, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 5})
    pl = o["choices"][0]["prompt_logprobs"]; pos = []
    for i, d in enumerate(pl):
        if not d: pos.append(None); continue
        top = sorted(((int(k), v["logprob"]) for k, v in d.items()), key=lambda kv: -kv[1])
        ref = d.get(str(ids[i]))
        pos.append({"top1": top[0][0], "top1_lp": round(top[0][1], 6), "lp": ref["logprob"] if ref else float("nan")})
    out.append({"id": t["id"], "ids": ids, "pos": pos}); print(t["id"], len(ids), flush=True)
json.dump(out, open(sys.argv[2], "w"))
