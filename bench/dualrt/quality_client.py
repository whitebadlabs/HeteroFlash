#!/usr/bin/env python3
"""Balanced-eval client for small-batch split runs (2026-11-17): like src/control_kld.py client, but writes the request
index to CAPDIR/capture.cur before each chunk so the split hook's capture (HF_SPLIT_CAPTURE) groups logits rows.
  quality_client.py TOKENS.json URL CAPDIR"""
import json, os, sys, time, urllib.request
toks = json.load(open(sys.argv[1])); url = sys.argv[2]; cap = sys.argv[3]
chunks = toks["chunks"] if isinstance(toks, dict) and "chunks" in toks else toks
t0 = time.time()
for i, ids in enumerate(chunks):
    open(os.path.join(cap, "capture.cur"), "w").write(str(i))
    body = {"model": "GLM-5.3-Flash", "prompt": ids, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 1}
    r = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    urllib.request.urlopen(r, timeout=7200).read()
    print(f"chunk {i + 1}/{len(chunks)} {time.time() - t0:.0f} s", flush=True)
open(os.path.join(cap, "capture.cur"), "w").write("stop")
time.sleep(3)
