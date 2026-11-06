#!/usr/bin/env python3
"""Byte breakdown of a safetensors checkpoint by tensor category (reads headers only)."""
import json, os, re, struct, sys
from collections import defaultdict
d = sys.argv[1]
idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
files = sorted(set(idx.values()))
cat = defaultdict(int); layer_cat = defaultdict(int)
def classify(k):
    if "visual" in k or "vision" in k or "mm_projector" in k or "merger" in k: return "vision"
    if re.search(r"layers\.(\d+)\.", k) and int(re.search(r"layers\.(\d+)\.", k).group(1)) >= 45: return "mtp_layer"
    if ".mlp.experts." in k: return "routed_experts"
    if "shared_expert" in k: return "shared_experts"
    if ".mlp.gate." in k: return "router"
    if ".mlp." in k: return "dense_mlp"
    if "indexer" in k: return "dsa_indexer"
    if "self_attn" in k or "linear_attn" in k or "attention" in k: return "attention"
    if "embed_tokens" in k: return "embeddings"
    if "lm_head" in k: return "lm_head"
    if "norm" in k or "hc_" in k or "hyper" in k: return "norms_hyperconn"
    return "other"
other = defaultdict(int)
for fn in files:
    with open(os.path.join(d, fn), "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(n))
    for k, v in h.items():
        if k == "__metadata__": continue
        b = v["data_offsets"][1] - v["data_offsets"][0]
        c = classify(k); cat[c] += b
        if c == "other": other[k.rsplit(".", 1)[0][:80]] += b
        if c == "attention":
            layer_cat["linear_attn" if "linear_attn" in k else "mla_attn"] += b
tot = sum(cat.values())
print(json.dumps({"total_GB": round(tot / 1e9, 3), "by_category_GB": {k: round(v / 1e9, 3) for k, v in sorted(cat.items(), key=lambda kv: -kv[1])},
                  "attention_split_GB": {k: round(v / 1e9, 3) for k, v in layer_cat.items()},
                  "other_examples": dict(sorted(other.items(), key=lambda kv: -kv[1])[:8])}, indent=1))
