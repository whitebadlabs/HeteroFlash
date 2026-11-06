#!/usr/bin/env python3
"""End-to-end TTFT benchmark for the HeteroFlash SLAs (stdlib only).

Measures client-observed time from sending the HTTP request to receiving the
first streamed token (SSE), for exact token counts, and verifies cache state
from the server's own accounting on every trial:

  cold    : N-token prompt whose first tokens are a per-trial nonce, so no cached
            prefix can match; the model process is warm (protocol: "model resident,
            prefix not cached"). Requires cached_tokens == 0 (or reports a violation).
  cached  : a fixed N-token prefix is sent once (warm-up, not timed), then
            prefix + S new tokens is timed. Requires cached_tokens >= N - block_size.
  Suffixes S default to 32 (primary SLA) and 16/64/128/256 (reported separately).

Prompts are token-id lists sent to /v1/completions (no chat template), built
from a text corpus via /tokenize, so the prompt length is exact. Trials of
different modes are interleaved to spread drift across conditions. Output:
one JSONL record per trial + a summary with p50/p95/p99 and bootstrap CIs.
"""
import argparse
import json
import random
import statistics
import time
import urllib.request


def http_json(base, path, body, timeout=600):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def stream_completion(base, model, prompt_ids, max_tokens=4, timeout=900):
    """Return dict(t_first, t_done, prompt_tokens, cached_tokens, n_chunks); times relative to send."""
    body = {"model": model, "prompt": prompt_ids, "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(base + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    t_first = None
    usage = {}
    chunks = 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                if ch.get("text") and t_first is None:
                    t_first = time.perf_counter() - t0
                    chunks += 1
                elif ch.get("text"):
                    chunks += 1
    t_done = time.perf_counter() - t0
    details = usage.get("prompt_tokens_details") or {}
    return {"t_first": t_first, "t_done": t_done, "prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": details.get("cached_tokens", 0), "n_chunks": chunks}


def corpus_tokens(base, model, corpus_path, need):
    text = open(corpus_path, encoding="utf-8").read()
    toks = []
    for start in range(0, len(text), 200000):
        toks += http_json(base, "/tokenize", {"model": model, "prompt": text[start:start + 200000],
                                                "add_special_tokens": False})["tokens"]
        if len(toks) >= need:
            break
    if len(toks) < need:
        raise SystemExit(f"corpus too short: {len(toks)} < {need} tokens")
    return toks


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def boot_ci(xs, q, reps=2000, seed=1):
    rng = random.Random(seed)
    vals = sorted(pct([rng.choice(xs) for _ in xs], q) for _ in range(reps))
    return [round(vals[int(0.025 * reps)], 4), round(vals[int(0.975 * reps)], 4)]


def summarize(trials, sla):
    out = {}
    for key in sorted({t["cond"] for t in trials}):
        xs = [t["t_first"] for t in trials if t["cond"] == key and t["t_first"] is not None and not t["violation"]]
        if not xs:
            continue
        s = {"n": len(xs), "p50": round(pct(xs, .5), 4), "p95": round(pct(xs, .95), 4), "p99": round(pct(xs, .99), 4),
             "mean": round(statistics.fmean(xs), 4), "p50_ci95": boot_ci(xs, .5), "p95_ci95": boot_ci(xs, .95),
             "violations_excluded": sum(1 for t in trials if t["cond"] == key and t["violation"])}
        if key in sla:
            s["sla_s"] = sla[key]
            s["pass_p95_upper_ci"] = s["p95_ci95"][1] <= sla[key]
        out[key] = s
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8001")
    ap.add_argument("--model", default="GLM-5.3-Flash")
    ap.add_argument("--corpus", required=True, help="UTF-8 text, long enough for N + suffix tokens")
    ap.add_argument("--n", type=int, default=45000)
    ap.add_argument("--suffixes", type=int, nargs="*", default=[32, 16, 64, 128, 256])
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--cold-sla", type=float, default=10.8)
    ap.add_argument("--cached-sla", type=float, default=0.2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    toks = corpus_tokens(a.base, a.model, a.corpus, a.n + max(a.suffixes) + 64)
    prefix = toks[:a.n]
    vocab_hi = max(toks)
    plan = [("cold", None)] * a.repeats + [("cached", s) for s in a.suffixes for _ in range(a.repeats)]
    rng.shuffle(plan)
    trials = []
    with open(a.out, "w") as f:
        f.write(json.dumps({"type": "header", "t": time.time(), "base": a.base, "model": a.model, "n": a.n,
                            "suffixes": a.suffixes, "repeats": a.repeats, "block_size": a.block_size}) + "\n")
        for i, (mode, s) in enumerate(plan):
            if mode == "cold":
                nonce = [rng.randrange(1000, vocab_hi) for _ in range(16)]
                ids = nonce + prefix[16:]
                r = stream_completion(a.base, a.model, ids)
                cond = f"cold_{a.n}"
                viol = (r["cached_tokens"] or 0) > a.block_size or r["prompt_tokens"] != a.n
            else:
                start = rng.randrange(0, 64)
                suffix = toks[a.n + start:a.n + start + s]
                stream_completion(a.base, a.model, prefix, max_tokens=1)          # warm the exact prefix (untimed)
                r = stream_completion(a.base, a.model, prefix + suffix)
                cond = f"cached_{a.n}+{s}"
                viol = (r["cached_tokens"] or 0) < a.n - a.block_size or r["prompt_tokens"] != a.n + s
            rec = {"type": "trial", "i": i, "cond": cond, **r, "violation": bool(viol)}
            trials.append(rec)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"{i + 1}/{len(plan)} {cond} ttft={r['t_first']} cached={r['cached_tokens']} "
                  f"prompt={r['prompt_tokens']}{' VIOLATION' if viol else ''}", flush=True)
        sla = {f"cold_{a.n}": a.cold_sla, f"cached_{a.n}+32": a.cached_sla}
        summ = summarize(trials, sla)
        f.write(json.dumps({"type": "summary", **summ}) + "\n")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
