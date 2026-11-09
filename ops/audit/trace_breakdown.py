#!/usr/bin/env python3
"""Aggregate a torch-profiler chrome trace (vLLM --profiler-config torch) into a GPU time breakdown.
  trace_breakdown.py TRACE.json[.gz] [--split-gap-ms 5]
Reports: wall span of GPU activity, GPU busy time (union of kernel intervals), idle fraction, and kernel time per
category (name-pattern based, printed with the top kernel names so the categories can be audited). With --split-gap-ms
the kernel timeline is cut into segments at idle gaps larger than the threshold (e.g. separates prefill from decode
steps), and each segment is reported."""
import argparse, collections, gzip, json, re, sys

CATS = [
    ("moe_experts", r"moe|expert|fused_moe|w4a16|iq2|route_pack|topk_sum|grouped|b12x.*moe"),
    ("attn_mla_dsa", r"mla|flash|attn|attention|indexer|sparse|paged|dsa|topk(?!_sum)"),
    ("kda_linear_attn", r"kda|gdn|delta|chunk_|recurrent|conv1d|ssm|mamba|gated"),
    ("dense_gemm", r"gemm|gemv|mxfp8|fp8|cutlass|sm9|sm10|sm12|nvjet|cublas|matmul|linear|xmma|wgmma"),
    ("norm_act_elem", r"norm|rms|silu|act|elementwise|vectorized|reduce|softmax|copy|cat|fill|index|scatter|gather|add|mul"),
    ("sample_logits", r"sample|argmax|logit|topp|topk_topp"),
]


def cat(name):
    n = name.lower()
    for c, pat in CATS:
        if re.search(pat, n):
            return c
    return "other"


def union_len(iv):
    iv.sort(); tot = 0; cs = ce = None
    for s, e in iv:
        if cs is None or s > ce:
            if cs is not None: tot += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    if cs is not None: tot += ce - cs
    return tot


def report(kern, label):
    if not kern:
        print(f"== {label}: no kernels"); return
    t0 = min(k[0] for k in kern); t1 = max(k[0] + k[1] for k in kern)
    busy = union_len([(k[0], k[0] + k[1]) for k in kern])
    by = collections.defaultdict(float); names = collections.defaultdict(lambda: collections.defaultdict(float))
    for ts, dur, name in kern:
        c = cat(name); by[c] += dur; names[c][name[:70]] += dur
    span = t1 - t0
    print(f"== {label}: span {span/1e3:.2f} ms, GPU busy {busy/1e3:.2f} ms ({100*busy/span:.1f}%), kernels {len(kern)}")
    tot = sum(by.values())
    for c, v in sorted(by.items(), key=lambda x: -x[1]):
        top = sorted(names[c].items(), key=lambda x: -x[1])[:3]
        print(f"   {c:16s} {v/1e3:9.2f} ms {100*v/tot:5.1f}%   top: " + "; ".join(f"{n} {d/1e3:.1f}" for n, d in top))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("trace"); ap.add_argument("--split-gap-ms", type=float, default=0)
    a = ap.parse_args()
    op = gzip.open if a.trace.endswith(".gz") else open
    with op(a.trace, "rt") as f:
        d = json.load(f)
    ev = d["traceEvents"] if isinstance(d, dict) else d
    kern = [(e["ts"], e.get("dur", 0), e.get("name", "")) for e in ev
            if e.get("ph") == "X" and str(e.get("cat", "")).lower() in ("kernel", "gpu_memcpy", "gpu_memset")]
    kern.sort()
    report(kern, "whole trace")
    if a.split_gap_ms:
        segs, cur, last_end = [], [], None
        for k in kern:
            if last_end is not None and k[0] - last_end > a.split_gap_ms * 1e3:
                segs.append(cur); cur = []
            cur.append(k); last_end = max(last_end or 0, k[0] + k[1])
        segs.append(cur)
        print(f"-- {len(segs)} segments at gaps > {a.split_gap_ms} ms")
        for i, s in enumerate(segs[:12]):
            report(s, f"segment {i}")


if __name__ == "__main__":
    main()
