#!/usr/bin/env python3
"""Which 4-bit format should the workstation CPU hold? Routed-output error on REAL captured MoE inputs (CPU only).

For a MoE layer, using the bounded-pilot captures (hidden states + router logits of the original model), compute the
gate-weighted routed output sum_j w_j E_j(x) for 2048 sampled tokens (all domains) with each expert format, and report
||y_fmt - y_bf16|| / ||y_bf16||:
  int4_rtn   : what runtime-v1's expert_server holds now (quantize_experts_int4.quant_int4_g128)
  int4_mse   : same layout, per-group scale chosen by MSE search (quant_int4_g128_mse) -> drop-in for expert_server
  q4_k       : llama.cpp Q4_K with the imatrix (the 4-bit format that scored +0.0045 dNLL end to end)
  iq2_xxs    : the Spark GPU format (llama.cpp quantizer via ggml_quant, same imatrix)
  q8_0       : near-lossless reference
  cpu_srv    : int4_rtn plus the server's per-token u8 activation quantization (to bound that confound)
  check_compare(a, b) relations then say whether a better CPU quantizer is worth a re-export.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gguf-py", required=True)
    ap.add_argument("--gguf-q4k", required=True)
    ap.add_argument("--imatrix", required=True)
    ap.add_argument("--captures", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--tokens", type=int, default=2048)
    ap.add_argument("--formats", default="int4_rtn,int4_mse,q4_k,iq2_xxs,q8_0,cpu_srv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pypath", default="", help="extra import dirs (colon-separated), e.g. the tools dir with quantize_experts_int4.py")
    a = ap.parse_args()
    for p in filter(None, a.pypath.split(":")):
        sys.path.insert(0, p)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "32")))
    import quantize_experts_int4 as qe
    from compensation_probe import layer_bias
    from qat_iq3xxs import GGUFLayer, ffn, load_layer_captures
    import ggml_quant
    sys.path.insert(0, a.gguf_py)
    import gguf

    t0 = time.time()
    ck = qe.Checkpoint(a.model)
    X, LG, DOM = load_layer_captures(a.captures, a.layer - 3)
    sgm = 1 / (1 + np.exp(-LG.astype(np.float32)))
    ids = np.argsort(-(sgm + layer_bias(ck, a.layer)), axis=1)[:, :8]
    w8 = np.take_along_axis(sgm, ids, 1); w8 /= w8.sum(1, keepdims=True)
    rng = np.random.default_rng(0)
    ev = np.sort(rng.choice(DOM.size, min(a.tokens, DOM.size), replace=False))
    Xev = torch.tensor(X[ev].astype(np.float32))
    I_ = ids[ev]; W_ = w8[ev]
    fmts = a.formats.split(",")
    gq4 = GGUFLayer(a.gguf_q4k, a.gguf_py, a.layer) if "q4_k" in fmts else None
    im = ggml_quant.Imatrix(a.imatrix, a.gguf_py)
    gname = {"gate": "ffn_gate_exps", "up": "ffn_up_exps", "down": "ffn_down_exps"}

    def ggml_rt(codec, w, m, e):
        q = ggml_quant.quantize(codec, w, im.get(f"blk.{a.layer}.{gname[m]}.weight", e))
        t = {"IQ2_XXS": gguf.GGMLQuantizationType.IQ2_XXS, "Q8_0": gguf.GGMLQuantizationType.Q8_0}[codec]
        return np.asarray(gguf.quants.dequantize(q, t), dtype=np.float32).reshape(w.shape)

    def int4(fn, w):
        return np.asarray(qe.dequant(*fn(w)), dtype=np.float32).reshape(w.shape)

    def u8_act(x):     # expert_server.quant_act: per 128-group symmetric scale, u8 with +128 offset (int8 range)
        g = x.view(x.shape[0], -1, 128)
        s = g.abs().amax(-1, keepdim=True).clamp_min(1e-30) / 127.0
        return (torch.round(g / s).clamp(-127, 127) * s).view_as(x)

    ys = {f: torch.zeros_like(Xev) for f in ["bf16"] + fmts}
    for e in range(288):
        sel = np.where((I_ == e).any(1))[0]
        if sel.size == 0:
            continue
        wj = torch.tensor((W_[sel] * (I_[sel] == e)).sum(1), dtype=torch.float32)[:, None]
        g, u, dn = [np.asarray(m, np.float32) for m in ck.expert(a.layer, e)]
        xs = Xev[sel]
        ys["bf16"][sel] += wj * ffn(xs, *[torch.tensor(m) for m in (g, u, dn)])
        for f in fmts:
            if f == "int4_rtn" or f == "cpu_srv":
                gu = int4(qe.quant_int4_g128, np.concatenate([g, u], 0)); W3 = [gu[:2048], gu[2048:], int4(qe.quant_int4_g128, dn)]
            elif f == "int4_mse":
                gu = int4(qe.quant_int4_g128_mse, np.concatenate([g, u], 0)); W3 = [gu[:2048], gu[2048:], int4(qe.quant_int4_g128_mse, dn)]
            elif f == "q4_k":
                W3 = [gq4.deq(m, e).reshape(s.shape) for m, s in (("gate", g), ("up", u), ("down", dn))]
            elif f == "iq2_xxs":
                W3 = [ggml_rt("IQ2_XXS", w, m, e) for m, w in (("gate", g), ("up", u), ("down", dn))]
            elif f == "q8_0":
                W3 = [ggml_rt("Q8_0", w, m, e) for m, w in (("gate", g), ("up", u), ("down", dn))]
            Gt, Ut, Dt = [torch.tensor(m) for m in W3]
            if f == "cpu_srv":
                h = torch.nn.functional.silu(u8_act(xs) @ Gt.T) * (u8_act(xs) @ Ut.T)
                y = u8_act(h) @ Dt.T
            else:
                y = ffn(xs, Gt, Ut, Dt)
            ys[f][sel] += wj * y
        if e % 48 == 0:
            print(json.dumps({"layer": a.layer, "expert": e, "s": round(time.time() - t0)}), flush=True)
    nref = ys["bf16"].norm()
    res = {"layer": a.layer, "tokens": int(ev.size), "domains": sorted(set(DOM[ev].tolist())),
           "rel_err": {f: round(float((ys[f] - ys["bf16"]).norm() / nref), 5) for f in fmts}, "s": round(time.time() - t0)}
    print(json.dumps(res))
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
