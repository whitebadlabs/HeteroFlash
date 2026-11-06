#!/usr/bin/env python3
"""Drive the M1 trace server sequentially (stdlib only).

Sends each prompt of a calibration JSONL to an OpenAI-compatible endpoint one
at a time (the server runs with max-num-seqs 1, so every forward pass belongs
to exactly one request), and appends a log line per request:
  {"prompt_id","domain","split","t_send","t_done","prompt_tokens","completion_tokens","finish_reason","error"}
Resumable: prompt_ids already in the log are skipped. Generation settings are
the server defaults (temperature 1, top_p 0.95, reasoning_effort high) unless
overridden; they are recorded in the log header.
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8001/v1/chat/completions")
    ap.add_argument("--model", default="GLM-5.3-Flash")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=3600)
    a = ap.parse_args()
    log = Path(a.log)
    done = set()
    if log.exists():
        for line in log.read_text().splitlines():
            r = json.loads(line)
            if r.get("type") == "request" and not r.get("error"):
                done.add(r["prompt_id"])
    prompts = [json.loads(l) for l in Path(a.prompts).read_text().splitlines()]
    todo = [p for p in prompts if p["prompt_id"] not in done]
    if a.limit:
        todo = todo[:a.limit]
    with log.open("a") as f:
        f.write(json.dumps({"type": "header", "t": time.time(), "url": a.url, "model": a.model,
                            "prompts_file": a.prompts, "todo": len(todo), "generation": "server defaults"}) + "\n")
        f.flush()
        for i, p in enumerate(todo):
            body = json.dumps({"model": a.model, "messages": p["messages"], "max_tokens": p["max_tokens"]}).encode()
            req = urllib.request.Request(a.url, data=body, headers={"Content-Type": "application/json"})
            rec = {"type": "request", "prompt_id": p["prompt_id"], "domain": p["domain"], "split": p["split"],
                   "t_send": time.time()}
            try:
                with urllib.request.urlopen(req, timeout=a.timeout) as r:
                    resp = json.load(r)
                u = resp.get("usage", {})
                rec.update(prompt_tokens=u.get("prompt_tokens"), completion_tokens=u.get("completion_tokens"),
                           finish_reason=resp["choices"][0].get("finish_reason"), error=None)
            except Exception as e:  # noqa: BLE001
                rec["error"] = repr(e)[:300]
            rec["t_done"] = time.time()
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"{i + 1}/{len(todo)} {p['domain']} {rec.get('prompt_tokens')}+{rec.get('completion_tokens')} "
                  f"{rec['t_done'] - rec['t_send']:.1f}s {rec['error'] or ''}", flush=True)
            time.sleep(0.5)  # gap so forward passes of consecutive requests are separable by time


if __name__ == "__main__":
    main()
