#!/usr/bin/env python3
"""FR-Spec draft vocabulary: the K most frequent token ids in our public calibration text (m1_calib_v1 messages +
longctx_corpus), plus every special/added token. Output: JSON {"K":..,"ids":[sorted],"coverage":..}.
  build_ids.py TOKENIZER_DIR DATA_DIR K OUT"""
import collections, json, sys
from tokenizers import Tokenizer
tok_dir, data, K, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
tk = Tokenizer.from_file(f"{tok_dir}/tokenizer.json")
texts = []
for line in open(f"{data}/m1_calib_v1.jsonl"):
    r = json.loads(line)
    texts += [m["content"] for m in r.get("messages", []) if isinstance(m.get("content"), str)]
texts.append(open(f"{data}/longctx_corpus.txt").read())
cnt = collections.Counter()
for t in texts:
    cnt.update(tk.encode(t, add_special_tokens=False).ids)
total = sum(cnt.values())
special = set()
cfg = json.load(open(f"{tok_dir}/tokenizer_config.json"))
for k, v in (cfg.get("added_tokens_decoder") or {}).items():
    special.add(int(k))
for a in json.load(open(f"{tok_dir}/tokenizer.json")).get("added_tokens", []):
    special.add(int(a["id"]))
top = [i for i, _ in cnt.most_common(K)]
ids = sorted(set(top) | special)
cov = sum(cnt[i] for i in ids) / total
json.dump({"K": K, "n_ids": len(ids), "n_special": len(special), "calib_tokens": total, "distinct_seen": len(cnt),
           "coverage_on_calib": round(cov, 5), "ids": ids}, open(out, "w"))
print(json.dumps({"K": K, "n_ids": len(ids), "calib_tokens": total, "distinct_seen": len(cnt), "coverage_on_calib": round(cov, 5)}))
