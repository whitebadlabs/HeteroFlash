#!/usr/bin/env python3
"""Build the M1 general-purpose routing-calibration prompt set (stdlib only).

Pulls rows from public Hugging Face datasets through the datasets-server rows
API, at spread-out offsets, and writes JSONL:
  {"prompt_id", "domain", "source", "split", "messages", "max_tokens"}
split = "holdout" for ~25% of prompts (sha256(prompt_id) % 4 == 0), else "calib".
Long-context prompts are built from Wikipedia articles concatenated to a target
character length, followed by a question.

No Hermes or other private data is used.
"""
import argparse
import hashlib
import json
import random
import time
import urllib.parse
import urllib.request

API = "https://datasets-server.huggingface.co/rows"
UA = {"User-Agent": "heteroflash-calibration/1"}


def rows(dataset, config, split, offset, length=100):
    q = urllib.parse.urlencode(dict(dataset=dataset, config=config, split=split, offset=offset, length=length))
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(f"{API}?{q}", headers=UA), timeout=60) as r:
                return [x["row"] for x in json.load(r)["rows"]]
        except Exception:  # noqa: BLE001
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"failed: {dataset} {offset}")


def num_rows(dataset, config, split):
    q = urllib.parse.urlencode(dict(dataset=dataset))
    with urllib.request.urlopen(urllib.request.Request(f"https://datasets-server.huggingface.co/size?{q}", headers=UA), timeout=60) as r:
        for s in json.load(r)["size"]["splits"]:
            if s["config"] == config and s["split"] == split:
                return s["num_rows"]
    raise RuntimeError("split not found")


def sample(dataset, config, split, n, rng, chunks=3):
    total = num_rows(dataset, config, split)
    per = -(-n // chunks)
    out = []
    for off in sorted(rng.sample(range(max(1, total - per)), chunks)):
        out += rows(dataset, config, split, off, per)
    return out[:n]


SOURCES = {
    "writing_chat": ("HuggingFaceH4/ultrachat_200k", "default", "test_sft",
                     lambda r: r["prompt"]),
    "knowledge_qa": ("TIGER-Lab/MMLU-Pro", "default", "test",
                     lambda r: r["question"] + "\n" + "\n".join(f"{chr(65 + i)}. {o}" for i, o in enumerate(r["options"]))
                     + "\nAnswer with the letter and a short justification."),
    "math": ("openai/gsm8k", "main", "test", lambda r: r["question"]),
    "code": ("openai/openai_humaneval", "openai_humaneval", "test",
             lambda r: "Complete the following Python function. Return only code.\n\n" + r["prompt"]),
    "multilingual": ("CohereLabs/aya_dataset", "default", "test",
                     lambda r: r["inputs"] if r.get("language") != "English" else None),
    "instruction_following": ("google/IFEval", "default", "train", lambda r: r["prompt"]),
}


def split_of(pid):
    return "holdout" if int(hashlib.sha256(pid.encode()).hexdigest(), 16) % 4 == 0 else "calib"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-domain", type=int, default=60)
    ap.add_argument("--long", type=int, default=4)
    ap.add_argument("--long-chars", type=int, default=170000)  # ~40-45K tokens of English
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=20261106)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    out = []
    for domain, (ds, cfg, sp, fmt) in SOURCES.items():
        got = []
        for r in sample(ds, cfg, sp, a.per_domain * (3 if domain == "multilingual" else 1), rng):
            text = fmt(r)
            if text and len(text) < 12000:
                got.append(text)
        for i, text in enumerate(got[:a.per_domain]):
            pid = f"{domain}:{hashlib.sha256(text.encode()).hexdigest()[:12]}"
            out.append({"prompt_id": pid, "domain": domain, "source": ds, "split": split_of(pid),
                        "messages": [{"role": "user", "content": text}], "max_tokens": a.max_tokens})
        print(domain, len(got[:a.per_domain]), flush=True)
    wiki = sample("wikimedia/wikipedia", "20231101.en", "train", a.long * 60, rng, chunks=a.long * 2)
    rng.shuffle(wiki)
    for j in range(a.long):
        buf, k = [], 0
        while sum(len(x) for x in buf) < a.long_chars and wiki:
            w = wiki.pop()
            buf.append(f"## {w['title']}\n{w['text']}")
        doc = "\n\n".join(buf)[:a.long_chars]
        q = ("You are given several encyclopedia articles above. Write a careful, well-organized essay that "
             "identifies connections between at least three of them, citing specific details from each.")
        pid = f"long_context:{hashlib.sha256(doc.encode()).hexdigest()[:12]}"
        out.append({"prompt_id": pid, "domain": "long_context", "source": "wikimedia/wikipedia 20231101.en",
                    "split": split_of(pid), "messages": [{"role": "user", "content": doc + "\n\n" + q}],
                    "max_tokens": a.max_tokens})
    print("long_context", a.long, flush=True)
    with open(a.out, "w") as f:
        for x in out:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(json.dumps({"total": len(out), "holdout": sum(x["split"] == "holdout" for x in out)}))


if __name__ == "__main__":
    main()
