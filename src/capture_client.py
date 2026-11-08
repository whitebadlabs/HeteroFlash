#!/usr/bin/env python3
"""Bounded MoE-input capture for the QAT pilot (stdlib only; runs on the workstation next to the trace server).

For each domain: write the hook control file with capture_inputs = {max_rows, layers, tag=<domain>}, then generate long
completions for that domain's prompts with up to --concurrency requests in flight until --tokens-per-domain generated
tokens are reached (prefill rows count too, so this is a lower bound on captured rows). Files land as
inputs_<domain>_<pid>_<n>.npz in the trace directory. Afterwards capture is switched off (periodic flush writes the rest).
"""
import argparse
import concurrent.futures as cf
import json
import os
import time
import urllib.request


def ctl(path, ci):
    c = json.load(open(path))
    c.update({"capture": False, "capture_inputs": ci})
    tmp = path + ".tmp"
    json.dump(c, open(tmp, "w"))
    os.replace(tmp, path)


def one(url, model, messages, max_tokens, timeout):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 1.0, "ignore_eos": True}
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        u = json.load(r).get("usage", {})
    return u.get("prompt_tokens", 0), u.get("completion_tokens", 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--control", required=True)
    ap.add_argument("--layers", type=int, nargs="+", required=True, help="MoE layer indices 0..41 (model layer - 3)")
    ap.add_argument("--tokens-per-domain", type=int, default=170000)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--url", default="http://127.0.0.1:8001/v1/chat/completions")
    ap.add_argument("--model", default="GLM-5.3-Flash")
    ap.add_argument("--log", required=True)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.prompts)]
    domains = sorted({r["domain"] for r in rows if r["domain"] != "long_context"})
    log = open(a.log, "a")
    t_all = time.time()
    for d in domains:
        ps = [r for r in rows if r["domain"] == d]
        ctl(a.control, {"max_rows": 10 ** 9, "layers": a.layers, "tag": d})
        time.sleep(2.5)                                   # hook re-reads the control file at most once per second
        gen = pin = 0
        t0 = time.time()
        i = 0
        with cf.ThreadPoolExecutor(a.concurrency) as ex:
            futs = set()
            while gen < a.tokens_per_domain or futs:
                while gen + len(futs) * a.max_tokens < a.tokens_per_domain and len(futs) < a.concurrency:
                    futs.add(ex.submit(one, a.url, a.model, ps[i % len(ps)]["messages"], a.max_tokens, 3600))
                    i += 1
                done, futs = cf.wait(futs, return_when=cf.FIRST_COMPLETED)
                for f in done:
                    try:
                        p, c = f.result()
                        pin += p; gen += c
                    except Exception as e:  # noqa: BLE001
                        log.write(json.dumps({"domain": d, "error": repr(e)[:200]}) + "\n")
        rec = {"domain": d, "requests": i, "prompt_tokens": pin, "completion_tokens": gen, "s": round(time.time() - t0, 1)}
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(json.dumps(rec), flush=True)
    ctl(a.control, None)
    time.sleep(45)                                        # periodic flush (30 s) writes the last buffer
    print(json.dumps({"done": True, "s": round(time.time() - t_all, 1)}), flush=True)


if __name__ == "__main__":
    main()
