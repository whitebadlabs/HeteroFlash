#!/usr/bin/env python3
"""Route 3: build our own ModelOpt MIXED_PRECISION checkpoint of GLM-5.3-Flash from the official BF16 weights.

Per-layer routed-expert formats come from llama.cpp GGUFs produced from the same BF16 (Q8_0, IQ2_XS, IQ2_XXS are
served by vLLM's ModelOpt IQ2 path with packing "ggml": each expert projection is a uint8 tensor
(out_rows, in/block, block_bytes), i.e. the GGUF rows unchanged). Non-expert tensors are copied from BF16
(unquantized in this first version). The plan file maps MoE layer -> {"codec": "Q8_0"|"IQ2_XS"|"IQ2_XXS", and either
"gguf": path (copy that GGUF's rows) or "quantize": true (quantize the BF16 experts here with llama.cpp's quantizer via
ggml_quant.py, using --imatrix for codecs that need one; byte-identical to llama-quantize, see ggml_quant.selftest)}.

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

from concurrent.futures import ThreadPoolExecutor

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


_imatrix = None


def quantize_layer(a, idx, L, codec, nbytes, add):
    """Quantize layer L's BF16 routed experts to `codec` (llama.cpp quantizer, per-expert imatrix if given)."""
    global _imatrix
    import time
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import ggml_quant
    if a.imatrix and _imatrix is None:
        _imatrix = ggml_quant.Imatrix(a.imatrix, a.gguf_py)
    gname = {"gate": "ffn_gate_exps", "up": "ffn_up_exps", "down": "ffn_down_exps"}
    names = [(e, p) for p in ("gate", "up", "down") for e in range(288)
             if f"model.language_model.layers.{L}.mlp.experts.{e}.{p}_proj.weight" in idx]
    t0 = time.time()

    def one(ep):
        e, p = ep
        name = f"model.language_model.layers.{L}.mlp.experts.{e}.{p}_proj.weight"
        with safe_open(os.path.join(a.bf16, idx[name]), "pt") as f:
            w = f.get_tensor(name).float().numpy()
        im = _imatrix.get(f"blk.{L}.{gname[p]}.weight", e) if _imatrix else None
        if im is None and _imatrix is not None:
            im = np.ones(w.shape[1], dtype=np.float32)        # tensor absent from the imatrix (e.g. MTP layer 45)
        q = ggml_quant.quantize(codec, w, im)
        return name, q.reshape(q.shape[0], -1, nbytes)

    with ThreadPoolExecutor(a.threads) as ex:
        for name, q in ex.map(one, names):
            add(name, torch.from_numpy(q))
    print(json.dumps({"layer": L, "codec": codec, "matrices": len(names), "s": round(time.time() - t0, 1)}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gguf-py", required=True)
    ap.add_argument("--max-layers", type=int, default=0)
    ap.add_argument("--shard-gb", type=float, default=4.0)
    ap.add_argument("--imatrix", default="")
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--fixed", default="bf16", choices=["bf16", "fp8", "fp8_nvfp4attn"],
                    help="fp8: non-expert weights as in the NVIDIA reference recipe (MXFP8 1x32 for KDA q/k/v/o, dense MLP, "
                         "lm_head; FP8_PB_WO 128x128 for MLA projections and shared experts; small projections stay BF16)")
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

    mla_layers = {int(m.group(1)) for m in (re.search(r"\.layers\.(\d+)\.self_attn\.q_a_proj\.weight$", k) for k in idx) if m}
    quantized_layers = {}

    def fixed_recipe(name):
        if a.fixed not in ("fp8", "fp8_nvfp4attn") or not name.endswith(".weight"):
            return None
        if a.fixed == "fp8_nvfp4attn":
            m0 = LAYER_RE.search(name)
            if m0 and re.search(r"\.self_attn\.(q_proj|k_proj|v_proj|o_proj|q_a_proj|q_b_proj|kv_a_proj_with_mqa)\.weight$", name):
                return "W4A16_NVFP4"
        mod = name[:-len(".weight")]
        if mod == "lm_head":
            return "MXFP8"
        m = LAYER_RE.search(name)
        if not m:
            return None
        L = int(m.group(1))
        if re.search(r"\.mlp\.(gate|up|down)_proj$", mod):
            return "MXFP8"                                   # dense MLP layers 0-2
        if re.search(r"\.mlp\.shared_experts\.(gate|up|down)_proj$", mod):
            return "FP8_PB_WO"
        if L in mla_layers and re.search(r"\.self_attn\.(q_a_proj|q_b_proj|kv_a_proj_with_mqa|o_proj)$", mod):
            return "FP8_PB_WO"
        if L not in mla_layers and re.search(r"\.self_attn\.(q|k|v|o)_proj$", mod):
            return "MXFP8"                                   # KDA projections
        return None

    E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

    def fused_group(mod):                                    # modules vLLM fuses must share weight_scale_2
        m1 = re.match(r"(.*\.self_attn)\.(q_proj|k_proj|v_proj)$", mod)
        if m1:
            return m1.group(1) + ".qkv"
        m2 = re.match(r"(.*\.self_attn)\.(q_a_proj|kv_a_proj_with_mqa)$", mod)
        if m2:
            return m2.group(1) + ".qkv_a"
        return mod

    group_amax = {}
    if a.fixed == "fp8_nvfp4attn":                           # pre-pass: per fused-group amax for the global scale
        for name, fn in idx.items():
            if keep(name) and fixed_recipe(name) == "W4A16_NVFP4":
                with safe_open(os.path.join(a.bf16, fn), "pt") as f:
                    amax = float(f.get_tensor(name).float().abs().max())
                g = fused_group(name[:-len(".weight")])
                group_amax[g] = max(group_amax.get(g, 0.0), amax)

    def add_fixed(name, w, recipe):
        mod = name[:-len(".weight")]
        w = w.float()
        if recipe == "W4A16_NVFP4":                          # ModelOpt NVFP4: 16-blocks, E4M3 block scales, fp32 global
            O, I = w.shape
            assert I % 16 == 0, (name, w.shape)
            gs = group_amax[fused_group(mod)] / (448.0 * 6.0)            # weight_scale_2 = amax / 2688
            blk = w.view(O, I // 16, 16)
            bs = (blk.abs().amax(-1) / (6.0 * gs)).clamp(max=448.0).to(torch.float8_e4m3fn)
            scale = bs.float().clamp_min(1e-12) * gs
            y = (blk / scale.unsqueeze(-1)).clamp(-6.0, 6.0)
            code = (y.abs().unsqueeze(-1) - E2M1).abs().argmin(-1).to(torch.uint8) | ((y < 0).to(torch.uint8) << 3)
            code = code.view(O, I)
            packed = (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous()
            add(name, packed); add(mod + ".weight_scale", bs.contiguous())
            add(mod + ".weight_scale_2", torch.tensor([gs], dtype=torch.float32))
            quantized_layers[mod] = {"quant_algo": "W4A16_NVFP4", "group_size": 16}
            return
        if recipe == "MXFP8":                                # vLLM mxfp8_utils._mxfp8_e4m3_quantize_torch semantics
            O, I = w.shape
            blk = w.view(O, I // 32, 32)
            amax = blk.abs().amax(-1).clamp(min=torch.finfo(torch.float32).tiny)
            sb = (torch.ceil(torch.log2(amax / 448.0)) + 127.0).clamp(0, 254)
            q = (blk / torch.exp2(sb - 127.0).unsqueeze(-1)).view(O, I).to(torch.float8_e4m3fn)
            add(name, q); add(mod + ".weight_scale", sb.to(torch.uint8))
            quantized_layers[mod] = {"quant_algo": "MXFP8", "weight_block_size": [1, 32]}
        else:                                                # FP8_PB_WO: fp32 scale per 128x128 block, dequant = q * scale
            O, I = w.shape
            assert O % 128 == 0 and I % 128 == 0, (name, w.shape)
            blk = w.view(O // 128, 128, I // 128, 128)
            sc = (blk.abs().amax(dim=(1, 3), keepdim=True) / 448.0).clamp(min=torch.finfo(torch.float32).tiny)
            q = (blk / sc).clamp(-448, 448).view(O, I).to(torch.float8_e4m3fn)
            add(name, q); add(mod + ".weight_scale", sc.view(O // 128, 1, I // 128, 1).contiguous())
            quantized_layers[mod] = {"quant_algo": "FP8_PB_WO", "weight_block_size": [128, 128]}

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
                r = fixed_recipe(name)
                if r:
                    add_fixed(name, f.get_tensor(name), r)
                else:
                    add(name, f.get_tensor(name))
    # 2) planned experts from GGUF blocks
    proj = {"gate": "ffn_gate_exps", "up": "ffn_up_exps", "down": "ffn_down_exps"}
    for L in sorted(plan):
        if a.max_layers and L >= a.max_layers:
            continue
        codec = plan[L]["codec"].upper()
        blk, nbytes, gtype = CODECS[codec]
        if plan[L].get("quantize"):
            quantize_layer(a, idx, L, codec, nbytes, add)
        else:
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
