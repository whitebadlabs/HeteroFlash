#!/usr/bin/env python3
"""Bet 1 dual-memory decode: tok/s composed from MEASURED components (results/2026-11-16-dualmem/).
The inputs are measured [M] or derived [D] unless marked [A]. The output is MODELED: measured parts, modeled composition.
Each phase is split so both sides finish together (per-phase balance), then a swap of T x 4096 bf16 partials.
  python3 src/dualmem_compose_2026_11_16.py [--json out.json]"""
import argparse, json
IN = {
 "BWg_fixed": (0.201, "M", "GB10 dense FP8 GEMMs in the real V2 runtime, cycle anatomy 2026-11-16 (58.6 ms / 11.8 GB)"),
 "BWg_expert": (0.195, "M/A", "GB10 routed-expert kernels (IQ2_XXS) in the same profile; the GPU Q4_K kernel rate is assumed equal [A]"),
 "BWc_fixed": (0.215, "M", "workstation FP8 (E4M3+E8M0, exact) GEMV, T=4, 32 cores, measured inside the prototype while swapping (212-218); 229-246 standalone"),
 "BWc_expert": (0.170, "M", "workstation Q4_K expert server, 4-token windows, 165-175 GB/s (2026-11-09 audit)"),
 "swap_us": {"EEE off + RX coalescing off": (60, "M", "prototype: 50 us swap + 10 us sync/D2H per phase"),
             "EEE off": (72, "M", "prototype: 62 + 10"),
             "NIC defaults (EEE on)": (205, "M", "prototype: 195 + 10")},
 "A_mtp3_prose": (2.217, "M", "committed tokens per MTP-3 cycle, prose"),
 "kda_layer_MB": (4946 / 34, "M", "headers: KDA attention bytes / 34 layers"),
 "mla_layer_MB": ((1477 + 164) / 11, "M", "headers: MLA attention + DSA indexer / 11 layers"),
 "replicated_MB_per_layer": ((99 + 71) / 42, "M", "router + mHC, read by both sides (not split)"),
 "dense_mlp_layer_MB": (467 / 3, "M", "headers"),
 "shared_expert_MB": (1057 / 42, "M", "headers"),
 "q4k_expert_MB": (14.16, "M", "GGUF tensor table"),
 "U4": (22.7, "M", "verify-window expert union per layer, MTP-3"),
 "lm_head_MB": (654, "M", "headers, FP8"),
 "draft_pass_MB": (950, "M", "MTP layer + 8 experts + lm_head"),
}
v = {k: (x if k == "swap_us" else x[0]) for k, x in IN.items()}
def split_ms(fixed_MB, expert_MB, repl_MB=0.0):
    """Per-phase balanced split: GPU and CPU finish together. Fixed bytes split at the fixed rates, expert bytes at the
    expert rates; replicated bytes are read by both. Returns ms."""
    gF, cF, gE, cE = v["BWg_fixed"], v["BWc_fixed"], v["BWg_expert"], v["BWc_expert"]
    # time t with gF*tf + cF*tf = fixed, gE*te + cE*te = expert; t = tf + te (both sides busy whole phase)
    return (fixed_MB / (gF + cF) + expert_MB / (gE + cE)) / 1e3 + repl_MB / gF / 1e3
def cycle(swap_us, drafts_split, imbalance=0.0, extra_us_per_phase=0.0):
    t = 0.0; n = 0
    for L in range(45):
        attn = v["mla_layer_MB"] if L in (3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43) else v["kda_layer_MB"]
        t += split_ms(attn, 0, v["replicated_MB_per_layer"] if L >= 3 else 0); n += 1
        if L < 3: t += split_ms(v["dense_mlp_layer_MB"], 0)
        else: t += split_ms(v["shared_expert_MB"], v["U4"] * v["q4k_expert_MB"])
        n += 1
    t += split_ms(v["lm_head_MB"], 0); n += 1
    t *= (1 + imbalance)
    if drafts_split: d = 3 * (split_ms(v["draft_pass_MB"], 0) + 3 * (swap_us + extra_us_per_phase) / 1e3); n_d = 9
    else: d = 3 * v["draft_pass_MB"] / v["BWg_fixed"] / 1e3; n_d = 0
    sync = n * (swap_us + extra_us_per_phase) / 1e3
    tot = t + sync + d
    return {"verify_stream_ms": round(t, 1), "sync_ms": round(sync, 1), "drafts_ms": round(d, 1), "phases": n + n_d,
            "cycle_ms": round(tot, 1), "tok_s": round(1e3 * v["A_mtp3_prose"] / tot, 1)}
def config_b():   # same inputs, config B structure: fixed + drafts on GPU, all experts on CPU, 42 exchanges
    F = 8936; E = 42 * v["U4"] * v["q4k_expert_MB"]; D = 3 * v["draft_pass_MB"]
    t = (F + D) / v["BWg_fixed"] / 1e3 + E / v["BWc_expert"] / 1e3 + 42 * 0.110
    return {"cycle_ms": round(t, 1), "tok_s": round(1e3 * v["A_mtp3_prose"] / t, 1), "measured_tok_s": "15.3-19.5 [M]"}
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--json"); a = ap.parse_args()
    rows = []
    for nic, (us, _, _) in v["swap_us"].items():
        for ds in (False, True):
            for imb, extra in ((0.0, 0.0), (0.10, 0.0), (0.10, 10.0)):
                r = cycle(us, ds, imb, extra); r.update({"nic": nic, "drafts": "split" if ds else "GPU only",
                     "imbalance": imb, "extra_us_per_phase_graph_relaunch_A": extra}); rows.append(r)
    for r in rows: print(f'{r["tok_s"]:5.1f} tok/s  {r["cycle_ms"]:5.1f} ms (stream {r["verify_stream_ms"]}, sync {r["sync_ms"]}, drafts {r["drafts_ms"]})  {r["nic"]:28s} drafts {r["drafts"]:8s} imb {r["imbalance"]:.2f} +{r["extra_us_per_phase_graph_relaunch_A"]:.0f}us')
    cb = config_b(); print("config B, same inputs:", cb)
    if a.json:
        json.dump({"label": "MODELED composition of MEASURED components; not a measured decode speed",
                   "inputs": {k: (x if k != "swap_us" else {n: list(t) for n, t in x.items()}) for k, x in IN.items()},
                   "rows": rows, "config_B_same_inputs": cb}, open(a.json, "w"), indent=1)
