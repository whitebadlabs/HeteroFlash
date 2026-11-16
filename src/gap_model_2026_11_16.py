#!/usr/bin/env python3
"""Decode / prefill gap model for the 2026-11-16 research plan. Pure arithmetic; every input is labelled
[M] measured on our hardware, [D] derived from measurements, [V] vendor/third-party claimed, [A] assumption.
Outputs are MODELED, never measurements.  python3 src/gap_model_2026_11_16.py [--bwg 0.208] [--json out.json]"""
import argparse, json
L = 42
IN = {  # value, label, source
 "F_verify_fp8_GB": (8.936, "M", "results/2026-11-16-cycle/bytes_full_iq2xxs_fp8_k3.json (safetensors headers)"),
 "draft_pass_GB": (0.950, "M", "same; MTP non-expert 0.244 + 8 MTP experts 0.052 + lm_head 0.654"),
 "lm_head_GB": (0.654, "M", "same"),
 "kda_0_21_GB": (2.47, "D", "half of KDA attention bytes 4.946 (17 of 34 KDA layers are in 0-21)"),
 "q4k_expert_MB": (14.16, "M", "GGUF tensor table"),
 "exl3_expert_MB": (11.3, "M", "hottier README (control format)"),
 "U4": (22.7, "M", "MTP-3 verify-window expert union, 31,416 windows"),
 "U8": (38.0, "M", "8-token window union (committed windows; real verify windows may be larger)"),
 "A_mtp3_prose": (2.217, "M", "audit mtp3_harness, prose"),
 "A_mtp3_hermes": (2.532, "M", "audit Hermes slice"),
 "A_block_drafter": (3.5, "A", "block drafter on prose; DFlash2 self-reports 3.68 with thinking on [V]"),
 "BWg": (0.208, "D", "TB/s effective; 17.97 GB per V1 MTP-3 cycle / 86.4 ms (25.67 tok/s at A=2.217)"),
 "BWc_verify": (0.170, "M", "TB/s; Q4_K expert server on 4-token windows, 165-175 GB/s"),
 "BWc_upgrade": (0.250, "A", "TB/s; hypothetical 8-CCD Threadripper PRO; no measured STREAM found, spec peak 410 [V]"),
 "net_exchange_10g_ms": (0.110, "M", "busy-poll mailbox network p50 at T=4 over 10 GbE"),
 "net_exchange_rdma_ms": (0.015, "A", "RoCE small message; no Spark ib_write_lat found, one unverified 1.7 us report [V]"),
 "fr_spec_draft_head_frac": (0.25, "V", "FR-Spec: draft lm_head compute -75%, verification uses full vocab (arXiv 2502.14856)"),
 "dflash_drafter_GB": (1.7, "A", "FP8 block drafter read once per cycle"),
}

def build(bwg):
    v = {k: x[0] for k, x in IN.items()}; v["BWg"] = bwg
    def cycle(F, D, E, A, mode, bwc, net, n_sync=90):
        if mode == "cpu_experts": t = (F + D) / bwg + E / bwc + L * net
        elif mode == "split_experts": t = (F + D) / bwg + E / (bwg + bwc) + L * net
        else: t = D / bwg + (F + E) / (bwg + bwc) + n_sync * net   # dual_memory: exact split of fixed + experts
        return t, 1e3 * A / t   # t in ms
    F, d, lm = v["F_verify_fp8_GB"], v["draft_pass_GB"], v["lm_head_GB"]
    Eq4 = v["U4"] * L * v["q4k_expert_MB"] / 1e3
    Emix4 = v["U4"] * L * (v["q4k_expert_MB"] + v["exl3_expert_MB"]) / 2e3
    Emix8 = v["U8"] * L * (v["q4k_expert_MB"] + v["exl3_expert_MB"]) / 2e3
    r = 1 - 4.5 / 8.25   # FP8 with block scales ~8.25 bpw -> NVFP4 ~4.5 bpw
    fp4_save = (v["kda_0_21_GB"] + lm) * r
    d_diet = (d - lm) + lm * v["fr_spec_draft_head_frac"]
    A3, A3h, Ab = v["A_mtp3_prose"], v["A_mtp3_hermes"], v["A_block_drafter"]
    rows = []
    def add(name, F_, D_, E_, A_, mode, bwc, net, quality):
        t, tps = cycle(F_, D_, E_, A_, mode, bwc, net)
        rows.append({"rung": name, "fixed_GB": round(F_, 2), "draft_GB": round(D_, 2), "experts_GB": round(E_, 2), "A": round(A_, 3),
                     "mode": mode, "cycle_ms": round(t, 1), "tok_s": round(tps, 1), "quality": quality})
    q = "config B (+0.0069 [M])"; n10, nr = v["net_exchange_10g_ms"], v["net_exchange_rdma_ms"]; c, cu = v["BWc_verify"], v["BWc_upgrade"]
    add("R0 config B: fixed on GPU, all experts on CPU (Q4_K), MTP-3, 10G [measured 15.3-19.5]", F, 3*d, Eq4, A3, "cpu_experts", c, n10, q)
    add("R1 balanced expert split GPU/CPU, fixed path GPU only", F, 3*d, Eq4, A3, "split_experts", c, n10, q)
    add("R2 Bet1 dual-memory exact split, 10G", F, 3*d, Eq4, A3, "dual_memory", c, n10, q)
    add("R3 Bet1 dual-memory, RDMA link", F, 3*d, Eq4, A3, "dual_memory", c, nr, q)
    add("R4 + draft diet (FR-Spec draft head)", F, 3*d_diet, Eq4, A3*0.98, "dual_memory", c, nr, q + ", lossless")
    add("R5 + training-free FP4 mix (KDA 0-21 + lm_head) + half the experts in control format", F-fp4_save, 3*d_diet, Emix4, A3*0.98,
        "dual_memory", c, nr, "+0.0069 +0.0045 = +0.0114 if additive [D]; bar +0.0148")
    add("R5h same, Hermes-slice acceptance", F-fp4_save, 3*d_diet, Emix4, A3h*0.98, "dual_memory", c, nr, "as R5")
    add("R6 + block drafter A=3.5 (8-token window, U=38)", F-fp4_save, v["dflash_drafter_GB"], Emix8, Ab, "dual_memory", c, nr, "as R5; lossless")
    add("R7 + workstation CPU with 250 GB/s (hypothetical upgrade)", F-fp4_save, v["dflash_drafter_GB"], Emix8, Ab, "dual_memory", cu, nr, "as R5")
    F4 = 1.5 + (F - 1.5) * (1 - r)
    add("R8 + QAD: fixed weights NVFP4 except ~1.5 GB (needs Max-Q training)", F4, v["dflash_drafter_GB"], Emix8, Ab, "dual_memory", cu, nr,
        "unknown; FP4-attention PTQ +0.0173 [M] needs 25-50% recovery")
    for u4 in (16.0, 12.0):   # union-aware routing (Bet 4): hypothetical verify-window unions, MTP-3 windows
        add(f"S  R5h with verify union U4={u4} (union-aware routing, hypothetical)", F-fp4_save, 3*d_diet, u4 * L * (v["q4k_expert_MB"] + v["exl3_expert_MB"]) / 2e3,
            A3h*0.98, "dual_memory", c, nr, "unknown; must be measured")
        add(f"S  R8 stack with U8 scaled by {u4/22.7:.2f}", F4, v["dflash_drafter_GB"], Emix8 * u4 / 22.7, Ab, "dual_memory", cu, nr, "unknown")
    budget = {f"BWc={b}": {"aggregate_GBps": round((bwg + b) * 1e3), "GB_per_committed_token_at_80": round((bwg + b) * 1e3 / 80, 2),
                           "GB_per_committed_token_at_125": round((bwg + b) * 1e3 / 125, 2)} for b in (c, cu)}
    return rows, budget, v

def prefill(v):
    q4k_cold = 88 * v["q4k_expert_MB"] / 1e3   # 200 hot experts/layer resident (memory budget [D]); 88 cold streamed as Q4_K
    links = {"10GbE [M] 1.15 GB/s": 1.15, "1x100G RoCE [V ~109 Gbit/s] ~11 GB/s": 11.0, "2x100G both rails [V 185-190 Gbit/s] ~22 GB/s": 22.0}
    compute = {"today's BF16-math kernels [D]": (18, 20), "best published FP8/FP4 kernels [D from V]": (11, 13), "+ NVFP4 non-expert [D from V]": (11, 11.5)}
    rows = [{"link": ln, "compute": cn, "stream_s": round(L * q4k_cold / g, 1), "ttft_s": [round(max(lo, L * q4k_cold / g), 1), round(max(hi, L * q4k_cold / g), 1)]}
            for ln, g in links.items() for cn, (lo, hi) in compute.items()]
    return {"cold_set_GB_per_prefill_q4k": round(L * q4k_cold, 1), "rows": rows}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--bwg", type=float, default=0.208); ap.add_argument("--json"); a = ap.parse_args()
    rows, budget, v = build(a.bwg)
    res = {"inputs": {k: {"value": x[0], "label": x[1], "source": x[2]} for k, x in IN.items()}, "BWg_used": a.bwg,
           "decode_rungs_MODELED": rows, "bytes_budget_for_targets": budget, "cold_prefill_MODELED": prefill(v)}
    for r in rows:
        print(f'{r["tok_s"]:6.1f} tok/s {r["cycle_ms"]:6.1f} ms  F{r["fixed_GB"]:5.2f} D{r["draft_GB"]:5.2f} E{r["experts_GB"]:5.2f} A{r["A"]:.2f}  {r["rung"]}')
    print(json.dumps(budget))
    for r in res["cold_prefill_MODELED"]["rows"]: print(r)
    if a.json: open(a.json, "w").write(json.dumps(res, indent=1) + "\n")
