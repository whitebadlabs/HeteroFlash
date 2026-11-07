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


ENGINE = {"name": "vllm"}   # "vllm" or "llamacpp" (llama-server): tokenization / templating / cache-accounting differ


def tokenize_text(base, model, text):
    if ENGINE["name"] == "llamacpp":
        return http_json(base, "/tokenize", {"content": text, "add_special": False, "parse_special": True})["tokens"]
    return http_json(base, "/tokenize", {"model": model, "prompt": text, "add_special_tokens": False})["tokens"]


def tokenize_chat(base, model, messages):
    if ENGINE["name"] == "llamacpp":
        prompt = http_json(base, "/apply-template", {"messages": messages})["prompt"]
        return tokenize_text(base, model, prompt)
    return http_json(base, "/tokenize", {"model": model, "messages": messages, "add_generation_prompt": True})["tokens"]


def http_json(base, path, body, timeout=600):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def stream_completion(base, model, prompt_ids, max_tokens=4, timeout=900, temperature=0.0, keep_times=False,
                      ignore_eos=False):
    """Return dict(t_first, t_done, prompt_tokens, cached_tokens, completion_tokens, n_chunks[, chunk_times]);
    times are seconds relative to sending the request. With speculative decoding one SSE chunk can carry
    several tokens, so per-chunk gaps are inter-chunk latencies; per-token TPOT comes from token counts."""
    body = {"model": model, "prompt": prompt_ids, "max_tokens": max_tokens, "temperature": temperature, "stream": True,
            "stream_options": {"include_usage": True, "continuous_usage_stats": keep_times}, "ignore_eos": ignore_eos}
    req = urllib.request.Request(base + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    t_first = None
    usage = {}
    chunks = 0
    times = []          # (t, cumulative completion tokens) per text chunk
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
            if ev.get("timings"):                              # llama-server: cache_n = prompt tokens reused from cache
                usage.setdefault("prompt_tokens_details", {})
                usage["prompt_tokens_details"] = {"cached_tokens": ev["timings"].get("cache_n", 0)}
                if "prompt_tokens" not in usage and "prompt_n" in ev["timings"]:
                    usage["prompt_tokens"] = ev["timings"]["prompt_n"] + ev["timings"].get("cache_n", 0)
                for k in ("draft_n", "draft_n_accepted"):              # llama-server speculative counters
                    if k in ev["timings"]:
                        usage[k] = ev["timings"][k]
            for ch in ev.get("choices", []):
                if ch.get("text"):
                    now = time.perf_counter() - t0
                    if t_first is None:
                        t_first = now
                    chunks += 1
                    if keep_times:
                        times.append((now, (ev.get("usage") or {}).get("completion_tokens")))
    t_done = time.perf_counter() - t0
    details = usage.get("prompt_tokens_details") or {}
    out = {"t_first": t_first, "t_done": t_done, "prompt_tokens": usage.get("prompt_tokens"),
           "cached_tokens": details.get("cached_tokens", 0), "completion_tokens": usage.get("completion_tokens"),
           "n_chunks": chunks, "draft_n": usage.get("draft_n"), "draft_n_accepted": usage.get("draft_n_accepted")}
    if keep_times:
        out["chunk_times"] = times
    return out


SPEC_KEYS = ("vllm:spec_decode_num_drafts", "vllm:spec_decode_num_draft_tokens", "vllm:spec_decode_num_accepted_tokens")


def scrape_spec(base):
    """Sum vLLM speculative-decoding Prometheus counters (all label sets). Missing -> {}."""
    try:
        with urllib.request.urlopen(base + "/metrics", timeout=30) as r:
            text = r.read().decode()
    except Exception:  # noqa: BLE001
        return {}
    out = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        for k in SPEC_KEYS:
            if line.startswith(k + "_total") or line.startswith(k + "{") or line.startswith(k + " "):
                try:
                    out[k] = out.get(k, 0.0) + float(line.rsplit(" ", 1)[1])
                except ValueError:
                    pass
    return out


def decode_metrics(r):
    """TPOT and ITL from one streamed request with keep_times=True."""
    ct = r.get("chunk_times") or []
    n = r.get("completion_tokens") or 0
    res = {"completion_tokens": n, "ttft": r["t_first"]}
    if n >= 2 and r["t_first"] is not None:
        res["tpot"] = (r["t_done"] - r["t_first"]) / (n - 1)        # seconds per output token after the first
        res["decode_tok_s"] = 1.0 / res["tpot"]
    gaps = [b[0] - a[0] for a, b in zip(ct, ct[1:])]
    if gaps:
        res["itl_chunk_p50"] = pct(gaps, .5)
        res["itl_chunk_p95"] = pct(gaps, .95)
        res["itl_chunk_p99"] = pct(gaps, .99)
        res["itl_chunk_max"] = max(gaps)
        toks = [t for _, t in ct if t is not None]
        if len(toks) == len(ct) and toks:
            res["tokens_per_chunk_mean"] = toks[-1] / len(ct)
    return res


def corpus_tokens(base, model, corpus_path, need):
    text = open(corpus_path, encoding="utf-8").read()
    toks = []
    for start in range(0, len(text), 200000):
        toks += tokenize_text(base, model, text[start:start + 200000])
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


def decode_main(argv=None):
    """TPOT / ITL / speculative acceptance on prose prompts (chat-templated via /tokenize)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8001")
    ap.add_argument("--model", default="GLM-5.3-Flash")
    ap.add_argument("--prompts", required=True, help="JSONL with messages (e.g. M1 calibration file)")
    ap.add_argument("--domains", nargs="*", default=["writing_chat"])
    ap.add_argument("--split", default="holdout")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--target-tok-s", type=float, default=125.0)
    ap.add_argument("--engine", default="vllm", choices=["vllm", "llamacpp"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    ENGINE["name"] = a.engine
    rows = [json.loads(l) for l in open(a.prompts)]
    rows = [r for r in rows if r["split"] == a.split and r["domain"] in a.domains][:a.limit]
    before = scrape_spec(a.base)
    per = []
    with open(a.out, "w") as f:
        f.write(json.dumps({"type": "header", "t": time.time(), "base": a.base, "max_tokens": a.max_tokens,
                            "temperature": a.temperature, "domains": a.domains, "n": len(rows)}) + "\n")
        for i, p in enumerate(rows):
            ids = tokenize_chat(a.base, a.model, p["messages"])
            r = stream_completion(a.base, a.model, ids, max_tokens=a.max_tokens, temperature=a.temperature,
                                  keep_times=True, ignore_eos=True)
            m = {"type": "trial", "prompt_id": p["prompt_id"], "domain": p["domain"], "prompt_tokens": r["prompt_tokens"],
                 **decode_metrics(r)}
            if r.get("draft_n") is not None:
                m["draft_n"], m["draft_n_accepted"] = r["draft_n"], r["draft_n_accepted"]
                m["completion_tokens"] = r["completion_tokens"]
            per.append(m)
            f.write(json.dumps(m) + "\n")
            f.flush()
            print(f"{i + 1}/{len(rows)} tpot={m.get('tpot')} itl95={m.get('itl_chunk_p95')}", flush=True)
        after = scrape_spec(a.base)
        spec = {k: after.get(k, 0) - before.get(k, 0) for k in SPEC_KEYS if k in after}
        if spec.get("vllm:spec_decode_num_drafts"):
            d = spec["vllm:spec_decode_num_drafts"]
            spec["mean_accepted_draft_tokens_per_verify"] = spec.get("vllm:spec_decode_num_accepted_tokens", 0) / d
            spec["committed_tokens_per_verify"] = spec["mean_accepted_draft_tokens_per_verify"] + 1
            spec["draft_acceptance_rate"] = (spec.get("vllm:spec_decode_num_accepted_tokens", 0)
                                             / max(spec.get("vllm:spec_decode_num_draft_tokens", 1), 1))
        dr = [m for m in per if m.get("draft_n")]
        if dr and not spec:                                      # llama.cpp: steps = generated - accepted drafts
            gen = sum(m["completion_tokens"] or 0 for m in dr); acc = sum(m["draft_n_accepted"] for m in dr)
            spec = {"draft_tokens": sum(m["draft_n"] for m in dr), "accepted_draft_tokens": acc,
                    "draft_acceptance_rate": acc / max(sum(m["draft_n"] for m in dr), 1),
                    "committed_tokens_per_verify": gen / max(gen - acc, 1)}
        tp = [m["tpot"] for m in per if m.get("tpot")]
        summ = {"type": "summary", "n": len(per),
                "tpot_ms_p50": round(pct(tp, .5) * 1e3, 3) if tp else None,
                "tpot_ms_p95": round(pct(tp, .95) * 1e3, 3) if tp else None,
                "decode_tok_s_mean": round(statistics.fmean(1 / x for x in tp), 2) if tp else None,
                "itl_chunk_ms_p95_median_over_requests": round(pct([m["itl_chunk_p95"] for m in per if "itl_chunk_p95" in m], .5) * 1e3, 3)
                if any("itl_chunk_p95" in m for m in per) else None,
                "spec": spec}
        if tp and spec.get("committed_tokens_per_verify"):
            tau = spec["committed_tokens_per_verify"]
            step = pct(tp, .5) * tau
            summ["implied_verify_step_ms_p50"] = round(step * 1e3, 2)
            summ["committed_tokens_needed_at_this_step_for_target"] = round(a.target_tok_s * step, 2)
        f.write(json.dumps(summ) + "\n")
    print(json.dumps(summ, indent=1))


def main():
    import sys as _sys
    if len(_sys.argv) > 1 and _sys.argv[1] == "decode":
        return decode_main(_sys.argv[2:])
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
    ap.add_argument("--engine", default="vllm", choices=["vllm", "llamacpp"])
    a = ap.parse_args()
    ENGINE["name"] = a.engine
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
