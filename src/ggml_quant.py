#!/usr/bin/env python3
"""Quantize BF16 weight rows to any ggml block codec by calling llama.cpp's own quantizer (libggml-base) via ctypes.

Used by build_mixed_checkpoint.py so each MoE layer's routed experts can be emitted in the codec the placement
plan chooses, without re-running llama-quantize over the whole 340 GB model. Output rows are byte-identical to
what llama-quantize writes (checked by `selftest`, which compares against an existing GGUF).
"""
import ctypes
import os
import sys

import numpy as np

TYPES = {"Q8_0": 8, "Q4_K": 12, "Q6_K": 14, "IQ2_XXS": 16, "IQ2_XS": 17, "IQ3_XXS": 18, "IQ3_S": 21, "IQ2_S": 22, "IQ4_XS": 23}
_lib = None


def lib(path=None):
    global _lib
    if _lib is None:
        path = path or os.environ.get("GGML_BASE_LIB", "/srv/tools/llama.cpp/build/bin/libggml-base.so")
        _lib = ctypes.CDLL(path)
        _lib.ggml_quantize_chunk.restype = ctypes.c_size_t
        _lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
                                             ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
        _lib.ggml_row_size.restype = ctypes.c_size_t
        _lib.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
        _lib.ggml_quantize_requires_imatrix.restype = ctypes.c_bool
        _lib.ggml_quantize_requires_imatrix.argtypes = [ctypes.c_int]
    return _lib


def quantize(codec, w, imatrix=None):
    """w: (rows, n_per_row) float array -> uint8 (rows, row_bytes)."""
    L, t = lib(), TYPES[codec.upper()]
    w = np.ascontiguousarray(w, dtype=np.float32)
    rows, n = w.shape
    if imatrix is None and L.ggml_quantize_requires_imatrix(t):
        raise ValueError(f"{codec} requires an importance matrix")
    rb = L.ggml_row_size(t, n)
    out = np.empty((rows, rb), dtype=np.uint8)
    im = None if imatrix is None else np.ascontiguousarray(imatrix, dtype=np.float32)
    got = L.ggml_quantize_chunk(t, w.ctypes.data, out.ctypes.data, 0, rows, n, None if im is None else im.ctypes.data)
    assert got == rows * rb, (got, rows * rb)
    return out


class Imatrix:
    """llama.cpp imatrix GGUF: <tensor>.in_sum2 (n_experts, n_per_row) and .counts (n_experts,)."""

    def __init__(self, path, gguf_py):
        sys.path.insert(0, gguf_py)
        import gguf
        self.t = {t.name: t for t in gguf.GGUFReader(path).tensors}

    def get(self, tensor, expert):
        if tensor + ".in_sum2" not in self.t:
            return None                                       # caller falls back to a uniform importance
        s2 = np.asarray(self.t[tensor + ".in_sum2"].data, dtype=np.float32)
        c = np.asarray(self.t[tensor + ".counts"].data, dtype=np.float32).reshape(-1)
        s2 = s2.reshape(c.shape[0], -1)
        if c[expert] <= 0:
            return np.ones(s2.shape[1], dtype=np.float32)
        return s2[expert] / c[expert]


def selftest(bf16_dir, gguf_path, gguf_py, layer=3):
    """Byte-compare our Q8_0 of the BF16 experts against llama-quantize's Q8_0 GGUF rows."""
    import json
    from safetensors import safe_open
    sys.path.insert(0, gguf_py)
    import gguf
    g = {t.name: t for t in gguf.GGUFReader(gguf_path).tensors if f"blk.{layer}.ffn_" in t.name}
    idx = json.load(open(os.path.join(bf16_dir, "model.safetensors.index.json")))["weight_map"]
    res = {}
    for p, gn in (("gate", "ffn_gate_exps"), ("up", "ffn_up_exps"), ("down", "ffn_down_exps")):
        ref = np.asarray(g[f"blk.{layer}.{gn}.weight"].data)
        for e in (0, 1, ref.shape[0] - 1):
            name = f"model.language_model.layers.{layer}.mlp.experts.{e}.{p}_proj.weight"
            with safe_open(os.path.join(bf16_dir, idx[name]), "pt") as f:
                w = f.get_tensor(name).float().numpy()
            q = quantize("Q8_0", w)
            r = ref[e].reshape(q.shape)
            res[f"{p}.{e}"] = {"shape": list(w.shape), "bytes_equal": bool(np.array_equal(q, r)),
                               "frac_bytes_diff": float((q != r).mean())}
    return res


if __name__ == "__main__":
    import json
    print(json.dumps(selftest(*sys.argv[1:4]), indent=1))
