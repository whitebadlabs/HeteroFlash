#!/usr/bin/env python3
"""Balanced per-domain evaluation text for KLD/NLL (replaces the Wikipedia-dominated q_kld_holdout.txt).

Seven domains, each exactly --chunks-per-domain x --ctx tokens (GLM tokenizer), concatenated in a fixed order so that
llama-perplexity's chunks align with domain blocks; per-domain metrics are recovered from its per-chunk running means.
Each sample is a user prompt plus a reference answer ("User: ...\\n\\nAssistant: ...").
Sources are public Hugging Face datasets, chosen to be disjoint from the calibration/imatrix prompt set
(data/m1_calib_v1.jsonl): different splits or datasets, plus a prefix-overlap filter against every calibration prompt.
Writes the text file and a manifest (domain order, chunk ranges, sources, sample counts, token counts).
"""
import argparse
import json
import random
import time
import urllib.parse
import urllib.request

API = "https://datasets-server.huggingface.co"
UA = {"User-Agent": "heteroflash-eval/1"}


def get(path, **q):
    url = f"{API}/{path}?{urllib.parse.urlencode(q)}"
    for attempt in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return json.load(r)
        except Exception:  # noqa: BLE001
            time.sleep(2 + 3 * attempt)
    raise RuntimeError(url)


def n_rows(ds, cfg, sp):
    for s in get("size", dataset=ds)["size"]["splits"]:
        if s["config"] == cfg and s["split"] == sp:
            return s["num_rows"]
    raise RuntimeError(f"no split {ds} {cfg} {sp}")


def fetch(ds, cfg, sp, want, rng):
    total = n_rows(ds, cfg, sp)
    offs = sorted(rng.sample(range(0, max(total - 100, 1)), k=min(8, max(total // 100, 1)))) if total > 100 else [0]
    out = []
    for off in offs:
        out += [x["row"] for x in get("rows", dataset=ds, config=cfg, split=sp, offset=off, length=100)["rows"]]
        if len(out) >= want:
            break
    rng.shuffle(out)
    return out


def chat(msgs):
    parts = []
    for m in msgs:
        role = {"user": "User", "assistant": "Assistant"}.get(m.get("role"))
        if role:
            parts.append(f"{role}: {m['content'].strip()}")
    return "\n\n".join(parts)


SOURCES = [  # (domain, dataset, config, split, formatter) - disjoint from the calibration splits
    ("writing_chat", "HuggingFaceH4/ultrachat_200k", "default", "test_gen", lambda r: chat(r["messages"])),
    ("knowledge_qa", "TIGER-Lab/MMLU-Pro", "default", "validation",
     lambda r: "User: " + r["question"] + "\n" + "\n".join(f"{chr(65 + i)}. {o}" for i, o in enumerate(r["options"]))
     + "\n\nAssistant: " + (r.get("cot_content") or "").replace("A: Let's think step by step.", "Let's think step by step.").strip()
     + f"\nThe answer is ({r['answer']})." if r.get("cot_content") else None),
    ("math", "openai/gsm8k", "main", "train", lambda r: "User: " + r["question"] + "\n\nAssistant: " + r["answer"]),
    ("code", "google-research-datasets/mbpp", "full", "test",
     lambda r: "User: " + r["text"] + "\nYour code should pass: " + r["test_list"][0] + "\n\nAssistant:\n```python\n" + r["code"] + "\n```"),
    ("multilingual", "CohereLabs/aya_dataset", "default", "train",
     lambda r: ("User: " + r["inputs"] + "\n\nAssistant: " + r["targets"]) if r.get("language") != "English" else None),
    ("instruction_following", "HuggingFaceH4/no_robots", "default", "test", lambda r: chat(r["messages"])),
    ("long_context", "wikimedia/wikipedia", "20231101.en", "train", lambda r: r["title"] + "\n\n" + r["text"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--calib", required=True, help="calibration prompts (excluded by prefix overlap)")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--chunks-per-domain", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20261108)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", required=True)
    a = ap.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(a.tokenizer)
    rng = random.Random(a.seed)
    calib = [json.loads(l) for l in open(a.calib)]
    prefixes = set()
    for r in calib:
        for m in r["messages"]:
            c = m["content"] if isinstance(m["content"], str) else json.dumps(m["content"])
            prefixes.add(" ".join(c.split())[:120])
    target = a.ctx * a.chunks_per_domain
    blocks, man = [], {"ctx": a.ctx, "chunks_per_domain": a.chunks_per_domain, "domains": []}
    chunk0 = 0
    for dom, ds, cfg, sp, fmt in SOURCES:
        rows = fetch(ds, cfg, sp, 400, rng)
        texts, skipped = [], 0
        ntok = 0
        for r in rows:
            try:
                t = fmt(r)
            except Exception:  # noqa: BLE001
                t = None
            if not t:
                continue
            if any(p and p in " ".join(t.split()) for p in prefixes if len(p) >= 60):
                skipped += 1
                continue
            texts.append(t)
            ntok += len(tok.encode(t).ids)
            if ntok > target * 1.15:
                break
        body = "\n\n".join(texts)
        ids = tok.encode(body).ids
        if len(ids) < target:
            raise SystemExit(f"{dom}: only {len(ids)} tokens < {target}; add rows")
        cut = tok.decode(ids[:target])
        n2 = len(tok.encode(cut).ids)
        while n2 > target:            # decode/encode round trip can merge across the cut; trim characters
            cut = cut[:-1]; n2 = len(tok.encode(cut).ids)
        while n2 < target:            # pad with spaces/newlines to the exact block length
            cut += "\n"; n2 = len(tok.encode(cut).ids)
        blocks.append(cut)
        man["domains"].append({"domain": dom, "source": f"{ds} [{cfg}/{sp}]", "samples": len(texts),
                               "skipped_calib_overlap": skipped, "tokens": n2,
                               "chunks": [chunk0, chunk0 + a.chunks_per_domain - 1]})
        chunk0 += a.chunks_per_domain
        print(json.dumps(man["domains"][-1]), flush=True)
    text = "".join(blocks)
    open(a.out, "w").write(text)
    man["total_tokens_concat"] = len(tok.encode(text).ids)
    man["total_chunks"] = chunk0
    json.dump(man, open(a.manifest, "w"), indent=1)
    print(json.dumps({"total_tokens_concat": man["total_tokens_concat"], "expected": chunk0 * a.ctx}))


if __name__ == "__main__":
    main()
