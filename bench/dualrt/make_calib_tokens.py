#!/usr/bin/env python3
"""Calibration token file for the expert mean-output table (2026-11-18): first N 2048-token chunks of data/q_imatrix_calib.txt
(the imatrix calibration text, not the balanced eval set), same format as q_eval_balanced_v1.tokens.json."""
import json, sys
from transformers import AutoTokenizer
H = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 4
tok = AutoTokenizer.from_pretrained(f"{H}/data/glm_tokenizer")
ids = tok(open(f"{H}/data/q_imatrix_calib.txt").read(), add_special_tokens=False)["input_ids"]
ev = json.load(open(f"{H}/data/q_eval_balanced_v1.sub7.tokens.json"))
chunks = [ids[i * 2048:(i + 1) * 2048] for i in range(N)]
assert all(len(c) == 2048 for c in chunks)
out = {"n_ctx": 2048, "n_vocab": ev["n_vocab"], "n_chunk": N, "chunk_ids": list(range(N)), "chunks": chunks, "source": "data/q_imatrix_calib.txt"}
json.dump(out, open(f"{H}/data/calib_imx{N}.tokens.json", "w"))
print("tokens", len(ids), "chunks", N)
