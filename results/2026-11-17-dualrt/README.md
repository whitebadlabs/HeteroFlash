# Bet 1, step 5: the dual-memory split inside the real runtime (2026-11-17)

Labels: [M] measured here, [D] derived from measured numbers, [A] assumption. Every tok/s below is measured end to end
on the served model (vLLM V2 runner, CUDA graphs, MTP-3, batch 1, greedy), NIC defaults (EEE on) unless stated.
Plan and design: `docs/RESEARCH_PLAN_2026-11-16.md` section E4. Code: `bench/dualrt/`.

## 0. What runs where (as built)

- **Runtime:** the config-B deployment (FP8 fixed path `full_iq2xxs_fp8`, every routed expert as Q4_K + imatrix on the
  workstation CPU, graph-capturable mailbox). The split is a new hook (`bench/dualrt/hook/sitecustomize.py`) plus a CUDA
  extension (`hook/hf_drt.cu`) and an extended server (`runtime/expert_server_gguf.c`: fp32 replies, HFD1 dense rows).
- **Routed experts (layers 3-44), by expert, re-balanced every step.** The GPU frees its unused IQ2 copies after startup
  and loads a resident hot set of the *same Q4_K bytes* (from the Spark's copy of the GGUF; hot set = decode-route counts
  on a disjoint calibration slice of the Hermes suite). Per layer and step an in-graph assign kernel takes the window's
  expert union U; cap = round(A·U + B) resident experts run on the GPU (own Q4_K kernel), the rest on the workstation.
- **lm_head (verify and the 3 draft passes), by vocabulary:** rows 0-84,991 on the GPU (own MXFP8 GEMV), rows
  84,992-154,879 on the workstation; exact bf16 logits come back (no top-k approximation).
- **KDA q/k/v projections (34 layers), by head:** the last 32 of 64 heads' q, k and v rows on the workstation, bf16
  rows back; conv, recurrence, gating and o_proj stay on the GPU with vLLM's own state (spec-decode rollback, prefix
  cache and prefill unchanged).
- **Partial sums:** bf16 by default (Carmen 2026-11-17: "is fine. Try to optimize for more tok/s"); fp32 is an option.
- **Prefill windows (> 16 tokens):** every route to the workstation, as config B.

## 1. Same-session baseline: config B on the original hook [M] (`cfgB_base_r1_client.jsonl`, 3 reps)

- Essay benchmark (serve_eval prompt, 256 tokens, decode only): **16.58 tok/s** (16.48 / 16.62 / 16.63), 2.46 committed per cycle, 148.5 ms per cycle.
- 4 chat prompts (prose/code/chat/reasoning, 400 tokens, incl. prefill): **15.43 tok/s**, 2.33 per cycle, 150.9 ms.
- 10 Hermes-suite items (2 each of tool_calling, multi_turn, coding, prose, reasoning; 300 tokens; incl. prefill): **18.00 tok/s**, 2.97 per cycle, 164.9 ms.
- Same hook as 11-08 (all routes on the workstation, bf16 partials, mailbox + CUDA graphs). Earlier runs: 15.3-19.5.
- The new mailbox in mode 0 (every route on the workstation, `m1_mode0_calib_client.jsonl`, 1 rep) gives the same speed: 15.8 / 15.4 / 15.2 tok/s, 150 ms per essay cycle.

## 2. Kernels [M] (`unit_q4k_L3.json`, `unit_dense_lmhead.json`)

- GPU Q4_K routed experts (own kernel, compact resident pool): exact against an fp32 dequantized reference (rel. L2 3-6e-7 at T = 1, 2, 4, 7). The workstation server on the same routes differs by 1.7-1.9% from that reference: its activations are quantized to q8_K, the GPU's are fp32.
  - Bandwidth: T = 1 176-184 GB/s, T = 4 155-162 GB/s (first version: 85 GB/s at T = 4, bank conflicts and one block per SM).
- GPU MXFP8 row GEMV (lm_head rows, KDA q/k/v rows): 228-231 GB/s at T = 1-4; rel. L2 1.66e-3 vs fp32 = bf16 output rounding. The workstation's lm_head rows: same error, 0.97 ms for 69,888 rows.

## 3. End-to-end decode with the split [M]

Hot set: 112-120 resident experts per layer from decode-route counts on Hermes-suite items 2-7 (disjoint from the measured items 0-1); in-sample route coverage of the top 120 is 66%, top 144 73% (`hotset_calib.json`).

| run (boot) | essay tok/s | ms/cycle | chat4 tok/s | ms/cycle | Hermes tok/s | note |
|---|---|---|---|---|---|---|
| config B, original hook | 16.58 | 148.5 | 15.43 | 150.9 | 18.00 | 3 reps |
| M1: experts split only, 120/layer, A 0.5 (`m1_split_n120`) | 19.38 | 117.9 | 18.77 | 124.1 | 18.92 | 1 rep |
| same boot `m4`: experts only (switches 0 0) | 20.07 | 119.2 | 19.74 | 127.2 | | 1 rep, keep-alive + busy-poll relay |
| m4: + lm_head split | 20.85 | 114.7 | 21.89 | 116.4 | | |
| m4: + KDA q/k/v split | 21.57 | 118.7 | 22.89 | 118.1 | | |
| m4: + KDA, A 0.7 | 22.97 | 119.8 | 22.14 | 118.3 | | |
| m4: + KDA, A 0.4 | 20.34 | 124.6 | 21.53 | 124.9 | | |
| m4: + KDA, A 0.7, **EEE off (temporary, restored)** | 24.14 | 114.0 | 23.14 | 113.7 | | |
| **m4 final: experts + lm_head + KDA, A 0.7, NIC defaults** | **22.82** | 120.6 | **22.05** | 119.9 | **23.03** | **3 reps** (22.6/23.1/22.8; 21.7/22.1/22.4; 19.6/25.3/25.2) |

- **Measured speedup at config-B weights, NIC defaults: +38% essay, +43% chat, +28% Hermes** (same session, same prompts, 3 reps each).
- The lm_head and KDA splits change acceptance as well as cycle time: committed tokens per cycle rose from 2.39 to 2.56 (essay) and 2.51 to 2.70 (chat) in the same boot. The GPU and workstation halves use bf16 activations (W8A16), while the stock b12x MXFP8 path quantizes activations (`VLLM_GLM53_KDA_MXFP8_ACTIVATION_MODE=auto`). This is a plausible cause, not yet isolated; the quality runs (section 4) measure the effect on the output distribution.
- **Cycle time hardly moved with the fixed-path splits (114.7-120.6 ms).** The exchange overhead eats the bandwidth gain at the current link cost:
  - expert exchange: server compute 0.91 ms, round trip seen by the relay 1.23 ms per layer (decode windows, final run);
  - dense rows (KDA, lm_head): server 0.34 ms, round trip 0.70 ms;
  - so ~0.32-0.36 ms per exchange is outside compute, x ~80 exchanges per cycle ≈ 26 ms [D]. EEE off removes ~0.1 ms of it (round trips 1.12 / 0.60 ms), worth ~5 ms per cycle (114 vs 120 ms).
  - the remaining ~0.2 ms per exchange is the request/reply design (two sequential transfers of 32-100 KB plus relay hops), against 0.05-0.06 ms for the prototype's symmetric swap.
- Per-step rebalancing (final run): window union U 23.3 experts, 12.5 of them resident, 12.1 computed on the GPU per layer and step; the rest (11) on the workstation. Raising A above 0.7 cannot help: the GPU already takes almost every resident expert in the union. More resident experts (memory) or a better hot set would.

RESULTS_PLACEHOLDER
