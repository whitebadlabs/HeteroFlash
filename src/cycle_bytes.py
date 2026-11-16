#!/usr/bin/env python3
"""Bytes read per MTP-k decode cycle, from safetensors headers only (no weights loaded).
  cycle_bytes.py CKPT_DIR [--k 3] [--union 22.7] [--out out.json]
Classifies every tensor (main layers 0..44, MTP layer 45, head/embeddings) into components and reports, per component,
stored bytes and bytes read per verify pass and per draft pass. Model of what is read (stated, not measured):
  verify pass  = all non-routed weights of layers 0..44 + lm_head + U distinct routed experts in each MoE layer
  draft pass   = MTP-layer non-routed weights + 8 routed experts of the MTP layer + lm_head (shared head)
  cycle        = 1 verify + k draft passes. KV/state reads and activations are excluded (small at short context).
Embedding rows are gathered (not streamed) and the vision tower is unused for text."""
import argparse, glob, json, os, re, struct
from collections import defaultdict

def comp(k):
    m = re.search(r"layers\.(\d+)\.", k)
    L = int(m.group(1)) if m else None
    if "visual" in k or "vision" in k or "mm_projector" in k or "merger" in k: return L, "vision"
    if L is None:
        if "embed_tokens" in k: return L, "embeddings"
        if "lm_head" in k: return L, "lm_head"
        return L, "final_norm_other"
    if ".mlp.experts." in k: return L, "routed_experts"
    if "shared_expert" in k: return L, "shared_experts"
    if ".mlp.gate." in k: return L, "router"
    if ".mlp." in k: return L, "dense_mlp"
    if "indexer" in k: return L, "dsa_indexer"
    if re.search(r"self_attn\.(q_a|q_b|kv_a|kv_b)", k) or "kv_a_layernorm" in k or "q_a_layernorm" in k: return L, "mla_attn"
    if "self_attn" in k: return L, "attn_o_or_kda"
    if "hc_" in k: return L, "mhc"
    if "norm" in k: return L, "norms"
    if "shared_head" in k: return L, "mtp_head"
    return L, "other"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("ckpt"); ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--union", type=float, default=22.7); ap.add_argument("--out"); a = ap.parse_args()
    stored = defaultdict(int); per_layer = defaultdict(lambda: defaultdict(int)); dtypes = defaultdict(set)
    mla_layers = set(); n_experts = defaultdict(int)
    for fn in sorted(glob.glob(os.path.join(a.ckpt, "*.safetensors"))):
        with open(fn, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(n))
        for k, v in h.items():
            if k == "__metadata__": continue
            b = v["data_offsets"][1] - v["data_offsets"][0]
            L, c = comp(k)
            if c == "mla_attn" and L is not None: mla_layers.add(L)
            if c == "routed_experts":
                m = re.search(r"experts\.(\d+)\.", k)
                if m: n_experts[L] = max(n_experts[L], int(m.group(1)) + 1)
            key = ("mtp:" if (L is not None and L >= 45) else "") + c
            stored[key] += b; dtypes[key].add(v["dtype"])
            if L is not None: per_layer[L][c] += b
    # attn_o_or_kda: split into KDA layers vs MLA o_proj
    kda = sum(per_layer[L]["attn_o_or_kda"] for L in per_layer if L < 45 and L not in mla_layers)
    mla_o = sum(per_layer[L]["attn_o_or_kda"] for L in per_layer if L < 45 and L in mla_layers)
    moe_layers = sorted(L for L in per_layer if L < 45 and per_layer[L]["routed_experts"] > 0)
    exp_bytes = {L: per_layer[L]["routed_experts"] / n_experts[L] for L in moe_layers}
    mean_exp = sum(exp_bytes.values()) / len(exp_bytes)
    mtp = [L for L in per_layer if L >= 45]
    mtp_nonexp = sum(v for L in mtp for c, v in per_layer[L].items() if c != "routed_experts")
    mtp_exp1 = sum(per_layer[L]["routed_experts"] / max(n_experts[L], 1) for L in mtp)
    main_fixed = {c: stored[c] for c in ("dense_mlp", "shared_experts", "router", "dsa_indexer", "mhc", "norms", "other")}
    main_fixed["kda_attn"] = kda; main_fixed["mla_attn"] = stored["mla_attn"] + mla_o
    main_fixed["lm_head"] = stored["lm_head"]
    verify_fixed = sum(main_fixed.values())
    verify_exp = a.union * sum(exp_bytes.values())
    draft = mtp_nonexp + 8 * mtp_exp1 + stored["lm_head"]
    cyc = verify_fixed + verify_exp + a.k * draft
    G = 1e9
    res = {
        "ckpt": os.path.basename(a.ckpt.rstrip("/")), "k_draft": a.k, "union_per_layer": a.union,
        "label": "bytes from safetensors headers [M]; read model per pass is stated in the docstring [model]",
        "mla_layers": sorted(mla_layers), "n_moe_layers": len(moe_layers), "mean_routed_expert_MB": round(mean_exp / 1e6, 3),
        "stored_GB": {k: round(v / G, 3) for k, v in sorted(stored.items())},
        "dtypes": {k: sorted(v) for k, v in sorted(dtypes.items())},
        "verify_fixed_GB_by_component": {k: round(v / G, 3) for k, v in sorted(main_fixed.items(), key=lambda x: -x[1])},
        "verify_fixed_GB": round(verify_fixed / G, 3), "verify_experts_GB": round(verify_exp / G, 3),
        "draft_pass_GB": {"mtp_nonexpert": round(mtp_nonexp / G, 3), "mtp_8_experts": round(8 * mtp_exp1 / G, 3), "lm_head": round(stored["lm_head"] / G, 3), "total": round(draft / G, 3)},
        "cycle_GB": round(cyc / G, 3),
        "cycle_ms_at_GBps": {str(bw): round(cyc / G / bw * 1e3, 1) for bw in (200, 222, 243, 273)},
    }
    s = json.dumps(res, indent=1); print(s)
    if a.out: open(a.out, "w").write(s + "\n")

if __name__ == "__main__":
    main()
