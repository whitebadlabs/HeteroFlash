#!/usr/bin/env python3
"""Build the step-1 prompt set (JSONL) from Hugging Face dataset parquet files. CPU only.

Tasks (the study plan asks for starcoderdata, lmsys-chat-1m, gsm8k):
  math  openai/gsm8k (main/test)                       chat prompt: the question
  code  bigcode/starcoderdata  [gated: needs an HF token whose account accepted the terms]
        fallback codeparrot/github-code-clean (ungated raw GitHub files, several languages)
        completion prompt: the first PROMPT_CHARS characters of a source file
  chat  lmsys/lmsys-chat-1m    [gated]
        fallback allenai/WildChat-1M (ungated real user chats)    chat prompt: first user turn
Each line: {prompt_id, task, source, kind: "chat"|"completion", text}
Usage: python3 prepare_prompts.py --out prompts.jsonl --per-task 400 [--hf-token-file F] [--gated]
"""
import argparse
import hashlib
import io
import json
import random
import urllib.request

import pyarrow.parquet as pq

API = "https://huggingface.co/api/datasets/{repo}/parquet"


def parquet_urls(repo, config, split, token):
    req = urllib.request.Request(API.format(repo=repo), headers=_hdr(token))
    d = json.load(urllib.request.urlopen(req, timeout=60))
    return d[config][split]


def _hdr(token):
    return {"Authorization": f"Bearer {token}"} if token else {}


def read_parquet(url, token, columns):
    req = urllib.request.Request(url, headers=_hdr(token))
    data = urllib.request.urlopen(req, timeout=600).read()
    return pq.read_table(io.BytesIO(data), columns=columns).to_pylist()


def pid(task, text):
    return f"{task}:{hashlib.sha256(text.encode()).hexdigest()[:12]}"


def gsm8k(n, rng, token):
    rows = read_parquet(parquet_urls("openai/gsm8k", "main", "test", token)[0], token, ["question"])
    rng.shuffle(rows)
    return [dict(prompt_id=pid("math", r["question"]), task="math", source="openai/gsm8k:main/test",
                 kind="chat", text=r["question"]) for r in rows[:n]]


def code(n, rng, token, gated, chars):
    out = []
    if gated:
        urls = parquet_urls("bigcode/starcoderdata", "default", "train", token)
        rows = read_parquet(rng.choice(urls), token, ["content"])
        src = "bigcode/starcoderdata"
    else:
        rows, src = [], "codeparrot/github-code-clean"
        for cfg in ("Python-all", "JavaScript-all", "Java-all"):
            urls = parquet_urls("codeparrot/github-code-clean", cfg, "train", token)
            rows += [dict(content=r["code"], cfg=cfg) for r in read_parquet(urls[0], token, ["code"])[:4000]]
    rng.shuffle(rows)
    for r in rows:
        t = r["content"]
        if len(t) < 400:
            continue
        t = t[:chars]
        out.append(dict(prompt_id=pid("code", t), task="code", source=src + (":" + r["cfg"] if "cfg" in r else ""),
                        kind="completion", text=t))
        if len(out) == n:
            break
    return out


def chat(n, rng, token, gated):
    if gated:
        repo, col = "lmsys/lmsys-chat-1m", "conversation"
    else:
        repo, col = "allenai/WildChat-1M", "conversation"
    urls = parquet_urls(repo, "default", "train", token)
    rows = read_parquet(urls[0], token, [col, "language"])
    rng.shuffle(rows)
    out = []
    for r in rows:
        if r.get("language") not in ("English", "english"):
            continue
        turns = r[col]
        if not turns or turns[0].get("role") != "user":
            continue
        t = turns[0]["content"]
        if not (20 <= len(t) <= 6000):
            continue
        out.append(dict(prompt_id=pid("chat", t), task="chat", source=repo, kind="chat", text=t))
        if len(out) == n:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-task", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--code-chars", type=int, default=4000)
    ap.add_argument("--hf-token-file")
    ap.add_argument("--gated", action="store_true", help="use starcoderdata + lmsys-chat-1m (needs token)")
    a = ap.parse_args()
    token = open(a.hf_token_file).read().strip() if a.hf_token_file else None
    rng = random.Random(a.seed)
    rows = gsm8k(a.per_task, rng, token) + code(a.per_task, rng, token, a.gated, a.code_chars) + \
        chat(a.per_task, rng, token, a.gated)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    from collections import Counter
    print(Counter((r["task"], r["source"].split(":")[0]) for r in rows))


if __name__ == "__main__":
    main()
