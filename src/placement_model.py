#!/usr/bin/env python3
"""Hot/cold expert placement and per-device format model (the project lead's founding ideas), using real per-expert routing
counts (M1 traces) and MEASURED device numbers. Outputs are MODELED [P]: decode tok/s, prefill CPU/link/GPU times,
Spark memory fit.

Inputs [M]: decode/prefill routing counts per (layer, expert); Spark expert GEMV ~170 GB/s, Spark fixed-weight GEMV
~218 GB/s (FP8), CPU expert read ~182 GB/s, CPU int8 GEMM ~15 TOPS sustained, 10GbE ~1.2 GB/s per direction, link
exchange 0.06 ms (decode) / 0.17 ms (8-token verify) per layer, Spark 121 GiB with 16.6 GiB overheads.
Assumptions [P]: tokens choose experts independently within a verify window (calibrated against the measured union);
GPU prefill for 45K tokens ~6.5 s when every expert it needs is resident; activations sent int8 (1 byte/elem).
"""
import argparse
import json

import numpy as np

H, I = 4096, 2048
FLOP_PER_TOK_EXPERT = 2 * 3 * H * I
SIZE = {"q8_0": 26.74e6, "q4k": 14.16e6, "iq3xxs": 9.74e6}   # bytes per expert (3 matrices), from the GGUF tables
GiB = 1024 ** 3
SPARK_ROOM = (121 - 16.6) * GiB          # after OS, CUDA ctx, KV, drafter, buffers (pre-check B)
FIXED_FP8 = 8.7e9                        # fixed weights read per step
BW_SPARK_EXP, BW_SPARK_FIX, BW_CPU = 170e9, 218e9, 182e9
CPU_OPS = 15e12
LINK = 1.2e9
GPU_PREFILL_45K = 6.5                    # [P] if all needed experts are resident on the Spark
T45 = 45000


UNION_MEASURED = {1: 8.0, 4: 22.9, 8: 37.0}   # [M] for 1 and 4 (M1 traces; 6 -> 31.0), 8 extrapolated [P]
_SCALE = {}


def union_prob(q, w):
    p = 1 - (1 - q) ** w
    return p * _SCALE.get(w, 1.0)


def decode_step(q, res, fmt_spark, fmt_cpu, w, balance, rtt):
    """q[l,e] per-token selection prob; res[l,e] bool resident on Spark; returns step seconds (sum of per-layer max)."""
    L = q.shape[0]
    fix_l = FIXED_FP8 / 45                # fixed bytes per layer (45 layers incl. dense)
    total = 0.0
    for l in range(L):
        pin = union_prob(q[l], w)
        r_bytes = float((pin * res[l]).sum()) * SIZE[fmt_spark]
        n_bytes = float((pin * ~res[l]).sum()) * SIZE[fmt_cpu]
        t_fix = fix_l / BW_SPARK_FIX
        if balance:                       # CPU holds a replica of every expert (fmt_cpu): move resident work to the CPU
            r_cpu_equiv = r_bytes / SIZE[fmt_spark] * SIZE[fmt_cpu]
            # choose fraction a of resident union computed on CPU to equalise times
            best = None
            for a in np.linspace(0, 1, 41):
                ts = t_fix + (1 - a) * r_bytes / BW_SPARK_EXP
                tc = (n_bytes + a * r_cpu_equiv) / BW_CPU + (rtt if (n_bytes + a * r_cpu_equiv) > 0 else 0)
                m = max(ts, tc)
                best = m if best is None or m < best else best
            total += best
        else:
            ts = t_fix + r_bytes / BW_SPARK_EXP
            tc = n_bytes / BW_CPU + (rtt if n_bytes > 0 else 0)
            total += max(ts, tc)
    total += 3 * fix_l / BW_SPARK_FIX      # 3 dense layers + head approximated
    return total


def prefill(qp, res):
    """Prefill: CPU computes non-resident experts; link carries int8 activations both ways for every layer with any
    non-resident expert selected by a token (fraction of tokens needing the CPU)."""
    cpu_flops = 0.0
    link_bytes = 0.0
    for l in range(qp.shape[0]):
        cold_mass = float((qp[l] * ~res[l]).sum())          # expected cold selections per token (of 8)
        cpu_flops += T45 * cold_mass * FLOP_PER_TOK_EXPERT
        p_token_needs_cpu = 1 - np.prod((1 - qp[l][~res[l]]))  # P(token picks >= 1 cold expert)
        link_bytes += 2 * T45 * H * p_token_needs_cpu
    t_cpu = cpu_flops / CPU_OPS
    t_link = link_bytes / LINK
    return t_cpu, t_link


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = np.load(a.counts)
    dec, pre = d["decode"].astype(np.float64), d["prefill"].astype(np.float64)
    # per-token selection probability: 8 picks per token => q = 8 * share
    q = 8 * dec / dec.sum(1, keepdims=True)
    qp = 8 * pre / pre.sum(1, keepdims=True)
    q, qp = np.clip(q, 0, 1), np.clip(qp, 0, 1)
    L, E = q.shape
    calib = float(np.mean([union_prob(q[l], 4).sum() for l in range(L)]))
    for w, u in UNION_MEASURED.items():          # calibrate the independence model to the measured union
        m = float(np.mean([(1 - (1 - q[l]) ** w).sum() for l in range(L)]))
        _SCALE[w] = u / m
    order = np.argsort(-dec, axis=1)                         # hottest first (decode counts)
    configs = []

    def mk_res(per_layer_hot):
        res = np.zeros((L, E), bool)
        for l in range(L):
            res[l, order[l, :per_layer_hot[l]]] = True
        return res

    def mem(res, fmt):
        return float(res.sum()) * SIZE[fmt]

    def add(name, res, fs, fc, note):
        for fc in ((fc,) if fs == "q4k" else ("q4k", "iq3xxs")):
            _add(name + f" | CPU replica {fc}", res, fs, fc, note)

    def _add(name, res, fs, fc, note):
        row = {"config": name, "note": note, "spark_expert_GiB": round(mem(res, fs) / GiB, 1),
               "fits": mem(res, fs) <= SPARK_ROOM, "resident_frac": round(float(res.mean()), 3),
               "resident_mass_decode": round(float((q * res).sum() / q.sum()), 3)}
        for w, c, rtt, lab in ((1, 1.0, 0.06e-3, "no_spec"), (4, 2.8, 0.12e-3, "mtp_c2.8"), (8, 3.3, 0.17e-3, "dflash_c3.3")):
            for bal in (False, True):
                t = decode_step(q, res, fs, fc, w, bal, rtt)
                row[f"decode_{lab}{'_bal' if bal else ''}_tok_s"] = round(c / t, 1)
        t_cpu, t_link = prefill(qp, res)
        row["prefill_cpu_s"] = round(t_cpu, 1)
        row["prefill_link_s"] = round(t_link, 1)
        row["cold45k_ttft_s_model"] = round(max(GPU_PREFILL_45K, t_cpu, t_link), 1)
        configs.append(row)

    # A. whole-layer split (llama.cpp style): first k MoE layers resident at fmt
    for fmt, k in (("q4k", 25), ("iq3xxs", 36)):
        res = np.zeros((L, E), bool); res[:k] = True
        add(f"whole-layer {fmt} {k} local / {L - k} remote", res, fmt, "q4k", "llama.cpp-compatible placement")
    # B. founding idea: per-expert hot/cold in every layer, as many hot as fit
    for fmt in ("q4k", "iq3xxs"):
        n = int(SPARK_ROOM // SIZE[fmt]) // L
        n = min(n, E)
        res = mk_res([n] * L)
        add(f"per-expert hot {n}/288 per layer at {fmt}, cold on CPU q4k", res, fmt, "q4k", "hot/cold in every layer")
    # C. hybrid: per-expert hot/cold but cold experts concentrated in few layers (keeps prefill link small)
    for fmt in ("q4k", "iq3xxs"):
        per = SIZE[fmt] * E
        k_full = int(SPARK_ROOM // per)
        rem = SPARK_ROOM - k_full * per
        hot = [E] * k_full + [0] * (L - k_full)
        # spend leftover memory on hottest experts of the remaining layers
        extra = int(rem // SIZE[fmt])
        j = k_full
        while extra > 0 and j < L:
            take = min(extra, E); hot[j] = take; extra -= take; j += 1
        res = mk_res(hot)
        add(f"hybrid {fmt}: {k_full} full layers + hot experts of the rest", res, fmt, "q4k", "cold confined to few layers")
    json.dump({"union_w4_model": round(calib, 1), "union_w4_measured": 22.9, "configs": configs}, open(a.out, "w"), indent=1)
    print(json.dumps({"union_w4_model": round(calib, 1), "union_w4_measured": 22.9}))
    for c in configs:
        print(json.dumps(c))


if __name__ == "__main__":
    main()
