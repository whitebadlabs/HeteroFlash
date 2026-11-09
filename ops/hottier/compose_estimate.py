"""Hot-expert tier experiment: compose ONE-LAYER measurements into whole-model estimates.

Every output here is a SPECULATIVE ARCHITECTURE ESTIMATE (or a bound where stated): it combines
  * complete-model measurements of the GPU-only Spark deployment (FP8 fixed path, 2-bit experts, MTP-3),
  * kernel microbenchmarks of one MoE layer (b12x on the control's EXL3 experts; Q4_K CPU server),
  * routing traces of layer 32 on Hermes-style and long-context workloads,
under the assumptions listed in ASSUMPTIONS. All 42 MoE layers are assumed to behave like layer 32, which is the
largest unverified assumption (per-layer coverage on M1 traces ranges 81-100% at 216 hot experts).
Usage: python3 compose_estimate.py RESULTS_DIR > estimate.json
"""
import json, sys

R = sys.argv[1]
gpu = json.load(open(f"{R}/bench_full.json"))["hot_sets"]
cpu = json.load(open(f"{R}/cpu_bench.json"))["N"]
loc = json.load(open(f"{R}/locality.json"))["by_N"]

M = {  # complete-model measurements, GPU-only Spark: FP8 fixed path, 2-bit experts, MTP-3 (results/2026-11-09-audit/mtp3_harness)
    "step_ms_prose": 2.217 / 25.02 * 1e3, "accept_prose": 2.217, "accept_hermes": 2.532,
    "moe_share_decode": 0.34,             # nsys decode profile (GPU 94% busy; dense ~61%, MoE ~34%)
    "cold45k_s": 37.92, "cached45k_s": 0.36,
    "moe_prefill_2bit_s": 45000 * 8 * 42 * 50.33e6 / 42e12,   # 2-bit MoE prefill at the profiled ~42 TFLOPS
}
M["nonmoe_step_ms"] = M["step_ms_prose"] * (1 - M["moe_share_decode"])
M["nonmoe_prefill_s"] = M["cold45k_s"] - M["moe_prefill_2bit_s"]
ASSUMPTIONS = {
    "all_layers_like_layer32": True,
    "link_net_us_busy_poll": 110,          # C relay verify4 net p50 (results/2026-11-09-audit/q4k_layer_exchange.txt)
    "sync_us_per_blocking_layer": [0, 50, 200],
    "link_bulk_GBps": 1.15,                # 10GbE; 64 KB ping-pong gives 0.87 GB/s incl. latency
    "cpu_us_per_pair_prefill": {"current_server_T16": None, "llamacpp_3TOPS": 50.33e6 / 3e12 * 1e6,
                                "torch_bf16_7.2TFLOPS": 50.33e6 / 7.2e12 * 1e6},
    "two_bit_expert_MB": 3 * 4096 * 2048 * 2.06 / 8 / 1e6,
}


def interp(table, N, key):
    Ns = sorted(int(k) for k in table)
    if N in Ns: return key(table[str(N)] if str(N) in table else table[N])
    lo = max(n for n in Ns if n < N); hi = min(n for n in Ns if n > N)
    a, b = key(table[str(lo)]), key(table[str(hi)])
    return a + (b - a) * (N - lo) / (hi - lo)


def sweep_ms(N, T):
    return interp(gpu, N, lambda h: next(s["ms"] for s in h["sweep"] if s["T"] == T))


def cpu_server_us(N, T):
    return interp(cpu, N, lambda h: h[f"T{T}"]["server_us"]["p50"] if h[f"T{T}"]["server_us"] else 0.0)


out = {"label": "speculative architecture estimate from measured components (see ASSUMPTIONS)",
       "measured_inputs": M, "assumptions": ASSUMPTIONS, "by_N": {}}
for N in (256, 216, 200, 160, 128):
    L = loc[str(N)] if str(N) in loc else None
    hermes, lc = (L or {}).get("hermes", {}), (L or {}).get("long_context", {})
    b4 = (hermes.get("decode") or {}).get("block_T4")
    cov_pre_lc = (lc.get("prefill") or {}).get("coverage")
    tok_block_pre = None
    r = {"hermes_decode_block_T4": b4, "long_context_prefill_coverage": cov_pre_lc}
    # ---- decode, per MTP-3 verify step (4 tokens) -------------------------------------------------------------
    g4 = sweep_ms(N, 4); c4 = (cpu_server_us(N, 4) + ASSUMPTIONS["link_net_us_busy_poll"]) / 1e3
    dec = {}
    for s_us in ASSUMPTIONS["sync_us_per_blocking_layer"]:
        s = s_us / 1e3
        overlap = M["nonmoe_step_ms"] + 42 * (max(g4, c4 * (b4 or 1)) + (b4 or 1) * s)
        serial = M["nonmoe_step_ms"] + 42 * (g4 + (b4 or 1) * (c4 + s))
        dec[f"sync{s_us}us"] = {"step_ms_overlap": round(overlap, 1), "step_ms_serial": round(serial, 1),
                                "prose_tok_s": [round(M["accept_prose"] / serial * 1e3, 1), round(M["accept_prose"] / overlap * 1e3, 1)],
                                "hermes_tok_s": [round(M["accept_hermes"] / serial * 1e3, 1), round(M["accept_hermes"] / overlap * 1e3, 1)]}
    r["decode"] = {"gpu_hot_ms_per_layer_T4": round(g4, 3), "cpu_cold_ms_per_layer_T4": round(c4, 3), **dec}
    # ---- cold 45K prefill ------------------------------------------------------------------------------------
    if cov_pre_lc is not None:
        gpu_hot_s = sweep_ms(N, 8192) / 1e3 * cov_pre_lc * (45000 / 8192) * 42       # rows scale with coverage
        pairs = 45000 * 8 * (1 - cov_pre_lc) * 42
        cur = interp(cpu, N, lambda h: (h["T16"]["server_us"]["p50"] / (16 * max(h["cold_routes_per_token"], 1e-9))) if h["T16"]["server_us"] else 0.0)
        tokfrac = (lc.get("prefill") or {}).get("block_T1") or 1.0
        link_s = 45000 * tokfrac * 4096 * 2 * 2 * 42 / (ASSUMPTIONS["link_bulk_GBps"] * 1e9)
        pre = {"gpu_hot_experts_s": round(gpu_hot_s, 1), "cold_pairs": int(pairs), "link_activation_traffic_s": round(link_s, 1),
               "nonmoe_s": round(M["nonmoe_prefill_s"], 1)}
        for name, us in (("current_server_T16", cur), ("llamacpp_3TOPS", ASSUMPTIONS["cpu_us_per_pair_prefill"]["llamacpp_3TOPS"]),
                         ("torch_bf16_7.2TFLOPS", ASSUMPTIONS["cpu_us_per_pair_prefill"]["torch_bf16_7.2TFLOPS"])):
            cpu_s = pairs * us / 1e6
            pre[name] = {"cpu_cold_s": round(cpu_s, 1),
                         "ttft_overlapped_s": round(M["nonmoe_prefill_s"] + max(gpu_hot_s, cpu_s, link_s), 1),
                         "ttft_serial_s": round(M["nonmoe_prefill_s"] + gpu_hot_s + cpu_s + link_s, 1)}
        pre["all_experts_on_gpu_if_they_fit_s"] = round(M["nonmoe_prefill_s"] + sweep_ms(256, 8192) / 1e3 * (45000 / 8192) * 42, 1)
        r["cold45k"] = pre
    # ---- cached 45K + 32 -------------------------------------------------------------------------------------
    g32 = sweep_ms(N, 32)
    union32 = interp(gpu, N, lambda h: next(s["distinct_hot_experts"] for s in h["sweep"] if s["T"] == 32))
    two_bit_ms = union32 * ASSUMPTIONS["two_bit_expert_MB"] * 1e6 / 210e9 * 1e3   # same experts at 2-bit, 210 GB/s
    c32 = 2 * (cpu_server_us(N, 16) + ASSUMPTIONS["link_net_us_busy_poll"]) / 1e3
    r["cached45k_plus32"] = {"gpu_hot_ms_T32": round(g32, 2), "two_bit_equiv_ms": round(two_bit_ms, 2), "cpu_ms_T32": round(c32, 2),
                             "s_overlap": round(M["cached45k_s"] + 42 * (max(g32, c32) - two_bit_ms) / 1e3, 3),
                             "s_serial": round(M["cached45k_s"] + 42 * (g32 + c32 - two_bit_ms) / 1e3, 3)}
    out["by_N"][N] = r
print(json.dumps(out, indent=1))
