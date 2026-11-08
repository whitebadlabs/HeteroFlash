#!/usr/bin/env python3
"""Route 3: build our own ModelOpt MIXED_PRECISION checkpoint of GLM-5.3-Flash from the official BF16 weights.

Per-layer routed-expert formats come from llama.cpp GGUFs produced from the same BF16 (Q8_0, IQ2_XS, IQ2_XXS are
served by vLLM's ModelOpt IQ2 path with packing "ggml": each expert projection is a uint8 tensor
(out_rows, in/block, block_bytes), i.e. the GGUF rows unchanged). Non-expert tensors are copied from BF16
(unquantized in this first version). The plan file maps MoE layer -> {"codec": "Q8_0"|"IQ2_XS"|"IQ2_XXS", "gguf": path}.

  build_mixed_checkpoint.py --bf16 DIR --plan plan.json --out DIR [--max-layers N]
--max-layers N keeps decoder layers < N only (truncated load tests; use with hf_overrides num_hidden_layers=N).
"""
import argparse
import glob
import json
import os
import re
import shutil
import sys

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

CODECS = {"Q8_0": (32, 34, 8), "IQ2_XS": (256, 74, 17), "IQ2_XXS": (256, 66, 16)}   # block elems, bytes, ggml type
LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


def gguf_expert_reader(path, gguf_py):
    sys.path.insert(0, gguf_py)
    import gguf
    r = gguf.GGUFReader(path)
    return {t.name: t for t in r.tensors if "_exps.weight" in t.name}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gguf-py", required=True)
    ap.add_argument("--max-layers", type=int, default=0)
    ap.add_argument("--shard-gb", type=float, default=4.0)
    a = ap.parse_args()
    plan = {int(k): v for k, v in json.load(open(a.plan)).items()}
    os.makedirs(a.out, exist_ok=True)
    idx = json.load(open(os.path.join(a.bf16, "model.safetensors.index.json")))["weight_map"]

    def keep(name):
        m = LAYER_RE.search(name)
        return not (a.max_layers and m and int(m.group(1)) >= a.max_layers)

    readers = {}
    shard, shard_bytes, shard_no, weight_map = {}, 0, 0, {}

    def flush():
        nonlocal shard, shard_bytes, shard_no
        if not shard:
            return
        fn = f"model-{shard_no:05d}.safetensors"
        save_file(shard, os.path.join(a.out, fn), metadata={"format": "pt"})
        for k in shard:
            weight_map[k] = fn
        shard, shard_bytes, shard_no = {}, 0, shard_no + 1

    def add(name, tensor):
        nonlocal shard_bytes
        shard[name] = tensor.contiguous()
        shard_bytes += tensor.numel() * tensor.element_size()
        if shard_bytes >= a.shard_gb * 1e9:
            flush()

    # 1) non-expert tensors from BF16, plus BF16 experts for layers not in the plan
    by_file = {}
    for name, fn in idx.items():
        if keep(name):
            by_file.setdefault(fn, []).append(name)
    expert_re = re.compile(r"\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight$")
    for fn, names in sorted(by_file.items()):
        with safe_open(os.path.join(a.bf16, fn), "pt") as f:
            for name in names:
                m = expert_re.search(name)
                if m and int(m.group(1)) in plan:
                    continue                                   # replaced by GGUF blocks below
                add(name, f.get_tensor(name))
    # 2) planned experts from GGUF blocks
    quantized_layers = {}
    proj = {"gate": "ffn_gate_exps", "up": "ffn_up_exps", "down": "ffn_down_exps"}
    for L in sorted(plan):
        if a.max_layers and L >= a.max_layers:
            continue
        codec = plan[L]["codec"].upper()
        blk, nbytes, gtype = CODECS[codec]
        g = readers.setdefault(plan[L]["gguf"], gguf_expert_reader(plan[L]["gguf"], a.gguf_py))
        for p, gname in proj.items():
            t = g[f"blk.{L}.{gname}.weight"]
            assert int(t.tensor_type) == gtype, (L, p, int(t.tensor_type), codec)
            data = np.asarray(t.data)                          # (n_experts, rows, row_bytes)
            n_e, rows = data.shape[0], data.shape[1]
            per = data.reshape(n_e, rows, -1, nbytes)
            for e in range(n_e):
                add(f"model.language_model.layers.{L}.mlp.experts.{e}.{p}_proj.weight", torch.from_numpy(np.ascontiguousarray(per[e])))
        quantized_layers[f"model.language_model.layers.{L}.mlp.experts"] = {
            "quant_algo": codec, "group_size": blk, "block_payload_bytes": nbytes, "packing": "ggml"}
    flush()
    json.dump({"metadata": {}, "weight_map": weight_map}, open(os.path.join(a.out, "model.safetensors.index.json"), "w"), indent=1)
    cfg = json.load(open(os.path.join(a.bf16, "config.json")))
    cfg["quantization_config"] = {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION", "exclude_modules": [],
                                  "quantized_layers": quantized_layers}
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=1)
    for f in glob.glob(os.path.join(a.bf16, "*")):
        b = os.path.basename(f)
        if b.endswith((".json", ".jinja", ".txt", ".model")) and b not in ("config.json", "model.safetensors.index.json"):
            shutil.copy(f, os.path.join(a.out, b))
    print(json.dumps({"tensors": len(weight_map), "shards": shard_no, "quantized_layers": len(quantized_layers)}))


if __name__ == "__main__":
    main()
