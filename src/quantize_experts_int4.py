#!/usr/bin/env python3
"""Quantize GLM-5.3-Flash routed experts (official BF16) to the CPU-tier int4 g128 format.

Reads safetensors byte ranges directly (numpy has no bf16). Kernel layout per expert
(runtime/expert_server.c): gate|up matrix [2*2048 x 4096] and down [4096 x 2048];
per row and 128-wide group: 64 bytes of nibbles (low = elements 0..63, high = 64..127,
stored as w+8), fp32 scale s, and folded constant c = s * 128 * sum(w) (the 128*8 per-element offset is subtracted exactly inside the kernel).

  measure  : quantization error report for sampled layers/experts (weights + FFN output on random inputs)
  export   : write layer files <out>/layer<L>.hfx with all (or selected) experts
File format: magic 'HFQ1', u32 layer, u32 n_experts, u32 ids[n]; then per expert, for gu then dn:
  q (rows*cols/2 bytes), s (rows*cols/128 f32), c (rows*cols/128 f32).
"""
import argparse
import json
import os
import struct
import sys
import time

import numpy as np

H, I, G = 4096, 2048, 128


class Checkpoint:
    def __init__(self, d):
        self.d = d
        self.map = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
        self.hdr = {}

    def _header(self, fn):
        if fn not in self.hdr:
            with open(os.path.join(self.d, fn), "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                self.hdr[fn] = (json.loads(f.read(n)), 8 + n)
        return self.hdr[fn]

    def bf16(self, name):
        fn = self.map[name]
        h, base = self._header(fn)
        meta = h[name]
        assert meta["dtype"] == "BF16", meta["dtype"]
        a, b = meta["data_offsets"]
        with open(os.path.join(self.d, fn), "rb") as f:
            f.seek(base + a)
            raw = np.frombuffer(f.read(b - a), dtype=np.uint16)
        return (raw.astype(np.uint32) << 16).view(np.float32).reshape(meta["shape"])

    def expert(self, layer, e):
        p = f"model.language_model.layers.{layer}.mlp.experts.{e}."
        return self.bf16(p + "gate_proj.weight"), self.bf16(p + "up_proj.weight"), self.bf16(p + "down_proj.weight")


def quant_int4_g128(w):
    """RTN symmetric int4 per (row, 128-group). Returns q int8 in [-8,7], s f32 [rows, groups]."""
    r, c = w.shape
    g = w.reshape(r, c // G, G)
    s = np.abs(g).max(-1) / 7.0
    s[s == 0] = 1e-12
    q = np.clip(np.rint(g / s[..., None]), -8, 7).astype(np.int8)
    return q.reshape(r, c), s.astype(np.float32)


def quant_int4_g128_mse(w, ratios=np.linspace(0.6, 1.0, 17)):
    """Per-group clipping search: choose s = ratio*amax/7 minimising group MSE (RTN rounding)."""
    r, c = w.shape
    g = w.reshape(r, c // G, G)
    amax = np.abs(g).max(-1)
    best_err = np.full(amax.shape, np.inf, np.float32)
    best_s = amax / 7.0
    for ratio in ratios:
        s = np.maximum(ratio * amax / 7.0, 1e-12)
        q = np.clip(np.rint(g / s[..., None]), -8, 7)
        err = ((q * s[..., None] - g) ** 2).sum(-1)
        m = err < best_err
        best_err[m] = err[m]
        best_s[m] = s[m]
    best_s = np.maximum(best_s, 1e-12).astype(np.float32)
    q = np.clip(np.rint(g / best_s[..., None]), -8, 7).astype(np.int8)
    return q.reshape(r, c), best_s


def quant_int8_g128(w):
    r, c = w.shape
    g = w.reshape(r, c // G, G)
    s = np.maximum(np.abs(g).max(-1) / 127.0, 1e-12).astype(np.float32)
    q = np.clip(np.rint(g / s[..., None]), -127, 127).astype(np.int8)
    return q.reshape(r, c), s


QUANT = {"int4_rtn": quant_int4_g128, "int4_mse": quant_int4_g128_mse, "int8": quant_int8_g128}


def dequant(q, s):
    r, c = q.shape
    return (q.reshape(r, c // G, G).astype(np.float32) * s[..., None]).reshape(r, c)


def pack(q, s):
    r, c = q.shape
    qq = (q.reshape(r, c // G, G) + 8).astype(np.uint8)
    packed = (qq[..., :64] | (qq[..., 64:] << 4)).reshape(r, c // 2)
    sw = q.reshape(r, c // G, G).astype(np.int32).sum(-1)
    cc = (s * (128.0 * sw)).astype(np.float32)          # the 128*8 per-element offset is removed exactly in-kernel
    return packed, s, cc


def ffn(x, gate, up, down):
    gt = x @ gate.T
    return ((gt / (1 + np.exp(-gt))) * (x @ up.T)) @ down.T


def cmd_measure(a):
    ck = Checkpoint(a.model)
    rng = np.random.default_rng(0)
    out = []
    for layer in a.layers:
        for e in a.experts:
            t0 = time.time()
            gate, up, down = ck.expert(layer, e)
            rec = {"layer": layer, "expert": e}
            x = rng.standard_normal((16, H)).astype(np.float32)
            ref = ffn(x, gate, up, down)
            for meth in a.methods:
                deq = {}
                for name, w in (("gate", gate), ("up", up), ("down", down)):
                    q, s = QUANT[meth](w)
                    deq[name] = dequant(q, s)
                    rec[f"{meth}_{name}_rel_err"] = float(np.linalg.norm(deq[name] - w) / np.linalg.norm(w))
                got = ffn(x, deq["gate"], deq["up"], deq["down"])
                rec[f"{meth}_ffn_out_rel_err"] = float(np.linalg.norm(got - ref) / np.linalg.norm(ref))
            rec["sec"] = round(time.time() - t0, 2)
            out.append(rec)
            print(json.dumps(rec), flush=True)
    keys = [k for k in out[0] if k.endswith("_err")]
    summ = {k: float(np.mean([r[k] for r in out])) for k in keys}
    print(json.dumps({"summary_mean": summ, "n": len(out)}))
    if a.out:
        json.dump({"records": out, "summary_mean": summ}, open(a.out, "w"), indent=1)


def pack8(q, s):
    r, c = q.shape
    sw = q.reshape(r, c // G, G).astype(np.int32).sum(-1)
    return q.astype(np.int8).view(np.uint8), s, (s * 128.0 * sw).astype(np.float32)


def cmd_export(a):
    ck = Checkpoint(a.model)
    os.makedirs(a.out, exist_ok=True)
    for layer in a.layers:
        ids = a.experts if a.experts else list(range(288))
        path = os.path.join(a.out, f"layer{layer}.hfx")
        t0 = time.time()
        with open(path + ".tmp", "wb") as f:
            f.write((b"HFQ8" if a.bits == 8 else b"HFQ1") + struct.pack("<II", layer, len(ids)) + struct.pack(f"<{len(ids)}I", *ids))
            for e in ids:
                gate, up, down = ck.expert(layer, e)
                for w in (np.concatenate([gate, up], 0), down):
                    p, s, c = pack8(*quant_int8_g128(w)) if a.bits == 8 else pack(*QUANT[a.int4_method](w))
                    f.write(p.tobytes()); f.write(s.tobytes()); f.write(c.tobytes())
        os.replace(path + ".tmp", path)
        print(f"layer {layer}: {len(ids)} experts, {os.path.getsize(path) / 1e9:.2f} GB, {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["measure", "export"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--layers", type=int, nargs="+", required=True)
    ap.add_argument("--experts", type=int, nargs="*", default=[])
    ap.add_argument("--methods", nargs="*", default=["int4_rtn", "int4_mse", "int8"])
    ap.add_argument("--bits", type=int, default=4, choices=[4, 8])
    ap.add_argument("--int4-method", default="int4_rtn", choices=["int4_rtn", "int4_mse"])
    ap.add_argument("--out")
    a = ap.parse_args()
    {"measure": cmd_measure, "export": cmd_export}[a.cmd](a)


if __name__ == "__main__":
    main()
