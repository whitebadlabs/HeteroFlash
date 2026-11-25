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
- **Partial sums:** bf16 by default (the project lead 2026-11-17: "is fine. Try to optimize for more tok/s"); fp32 is an option.
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
| same boot `m4`: experts only (switches 0 0; lm_head and KDA rows all on the GPU with the new kernels) | 20.07 | 119.2 | 19.74 | 127.2 | | 1 rep, keep-alive + busy-poll relay |
| m4: + lm_head split | 20.85 | 114.7 | 21.89 | 116.4 | | |
| m4: + KDA q/k/v split | 21.57 | 118.7 | 22.89 | 118.1 | | |
| m4: + KDA, A 0.7 | 22.97 | 119.8 | 22.14 | 118.3 | | |
| m4: + KDA, A 0.4 | 20.34 | 124.6 | 21.53 | 124.9 | | |
| m4: + KDA, A 0.7, **EEE off (temporary, restored)** | 24.14 | 114.0 | 23.14 | 113.7 | | |
| **m4 final: experts + lm_head + KDA, A 0.7, NIC defaults** | **22.82** | 120.6 | **22.05** | 119.9 | **23.03** | **3 reps** (22.6/23.1/22.8; 21.7/22.1/22.4; 19.6/25.3/25.2) |

- **Cycle time (the hardware effect): 148.5 -> 120.6 ms essay, 150.9 -> 119.9 ms chat, 164.9 -> 135.9 ms Hermes, 1.21-1.26x** [M, 3 reps each, same session, same prompts].
- tok/s: +38% essay, +43% chat, +28% Hermes. The part beyond the cycle-time ratio is acceptance (committed tokens per cycle 2.46 -> 2.75, 2.33 -> 2.64, 2.97 -> 3.13, i.e. 1.05-1.14x), which is **not attributed**:
  - With the run-time switches at 0 the GPU still computes the lm_head and KDA rows with the new W8A16 kernels (that is the loopback arm), so g00 / g10 / g11 share one arithmetic. Their acceptance still ranged 2.39-2.75 (essay) and 2.51-2.70 (chat), so differences between those rows are trajectory noise, not an effect of the splits.
  - Against config B the arithmetic differs (stock b12x MXFP8 path vs the new kernels); whether that moves acceptance is open.
- **The plan's gate (measured >= 25 tok/s on prose at config-B quality) is not met at NIC defaults**: 22.8 essay / 22.1 chat. The final run is "NIC defaults + software keep-alive" (a 0x00 byte from the Spark relay after 50 us of idle; EEE itself unchanged). The temporary EEE-off point reached 24.1 / 23.1 (1 rep).
- The KDA split did not lower cycle time on the same boot (114.7 -> 118.7 ms essay, 116.4 -> 118.1 ms chat with lm_head on); it was kept in the final configuration on acceptance, which the point above shows is noise. A repeated "1 0" vs "1 1" comparison is the next measurement.
- **Cycle time hardly moved with the fixed-path splits (114.7-120.6 ms).** The exchange overhead eats the bandwidth gain at the current link cost:
  - expert exchange: server compute 0.91 ms, round trip seen by the relay 1.23 ms per layer (decode windows, final run);
  - dense rows (KDA, lm_head): server 0.34 ms, round trip 0.70 ms;
  - so ~0.32-0.36 ms per exchange is outside compute, x ~80 exchanges per cycle ≈ 26 ms [D]. EEE off removes ~0.1 ms of it (round trips 1.12 / 0.60 ms), worth ~5 ms per cycle (114 vs 120 ms).
  - the remaining ~0.2 ms per exchange is the request/reply design (two sequential transfers of 32-100 KB plus relay hops), against 0.05-0.06 ms for the prototype's symmetric swap.
- Per-step rebalancing (final run): window union U 23.3 experts, 12.5 of them resident, 12.1 computed on the GPU per layer and step; the rest (11) on the workstation. Raising A above 0.7 cannot help: the GPU already takes almost every resident expert in the union. More resident experts (memory) or a better hot set would.

## 4. Quality: balanced eval at decode-size batches [M] (`run_quality.sh`, `compare_q8.py`, `paired_quality.py`)

Every position runs through the split path: eager mode, max-num-batched-tokens 16 (the 2048-token chunks prefill 16
tokens per step), logits captured by the split hook, scored against the Q8_0 BF16-proxy reference exactly as
`src/control_kld.py compare` does (rows 1024-2046 of each chunk). Split runs: 112 resident experts per layer, A 0.7,
lm_head + KDA q/k/v split; exchange counters asserted nonzero in the stats files.

| run | chunks | dNLL vs Q8 proxy (SE) | KLD mean / p99 | same top-1 |
|---|---|---|---|---|
| config B, 2048-token batches (11-08) | 41 | +0.0069 (0.0014) | 0.030 / 0.42 | 94.8% |
| unsplit (config B on this mailbox), 16-token batches | 40* | +0.0037 (0.0014) | 0.028 / 0.39 | 94.9% |
| **split, bf16 partial exchange (default)** | 41 | **+0.0044 (0.0014)** | **0.027 / 0.37** | **95.1%** |
| split, fp32 partial exchange | 41 | +0.0051 (0.0014) | 0.026 / 0.37 | 95.0% |
| control: EXL3 3.5 bpw on 2x Max-Q (11-08) | 41 | +0.0148 (0.0017) | 0.042 | 94.0% |

\* The last chunk of the unsplit run was not flushed by the first capture version (fixed for the split runs; paired
comparisons use the chunks both runs have). The 16-token-batch unsplit run agrees with the 11-08 config-B number
within ~2 SE, which validates the new capture path.

**Paired, same tokens (`paired_*.json`; dNLL = B - A per token, SE over 40,920 tokens):**
- unsplit -> split bf16: dNLL **+0.0007 (SE 0.0008)**, direct KL(unsplit || split) 0.010 (p99 0.13), same top-1 **96.9%**.
- unsplit -> split fp32: dNLL +0.0011 (SE 0.0008), direct KL 0.010 (p99 0.14), same top-1 96.8%.
- split fp32 -> split bf16: dNLL **-0.0007 (SE 0.0008)**, direct KL 0.009 (p99 0.12), same top-1 97.0%. **The bf16 expert-partial exchange costs no measurable quality**, which supports the bf16 default.
- For scale: the same-top-1 of 96.8-97.0% between any two of these runs equals the run-to-run disagreement caused by summation order alone. Every pair has a direct KLD of ~0.01, far below each run's 0.026-0.028 KLD to the Q8 proxy.
- These paired numbers are the full-model teacher-forced agreement check. The 1-layer and N-layer agreement stages of the plan were not run separately; correctness was established by the kernel unit tests (section 2), the same-boot switch arms, and this full-model comparison, which shows no significant shift.
- What the split arm changes, all at once: GPU vs workstation Q4_K arithmetic for ~half the routed experts (fp32 vs q8_K activations), bf16 vs fp32 expert partials (fp32 vs bf16 touches the expert partial sums only; the dense rows are bf16 outputs either way), and the new W8A16 kernels for lm_head and KDA q/k/v rows instead of the stock b12x MXFP8 path. The paired delta is not significant, so no attribution run was made.

## 5. NIC state during these runs

- Every speed number is at NIC defaults: EEE on, Spark Tx LPI 115 us, workstation Tx LPI 0 us. Runs from boot m4 on also use the relay's software keep-alive (a 0x00 byte after 50 us of idle). That changes traffic, not NIC settings.
- One temporary EEE-off point (m4 `g11a7_eeeoff`) was taken with `ethtool --set-eee <if> eee off` on both hosts, then restored.
  - Restoring needed explicit timers on both hosts: `eee on tx-lpi on tx-timer 115` on the Spark, `tx-timer 0` on the workstation.
  - The workstation then stayed "enabled - inactive" for more than 8 minutes, until `ethtool -r enp209s0f1np1` (link renegotiation).
  - Both hosts were verified "enabled - active" with the original timers before the final 3-rep run and again at the end.

## 6. Not done / next

- **Exchange cost** is now the limit: ~0.2 ms per exchange beyond wire time and compute, with EEE off. Next steps:
  - Timestamp the relay hops to locate it.
  - `SO_BUSY_POLL` on the relay socket: written in `hook/hf_drt.cu`, not yet booted.
  - Then fuse per-layer requests, or move to the prototype's symmetric swap.
- **KDA split on vs off** (switches "1 0" vs "1 1") needs a repeated, alternating comparison on ms per cycle. The single pair showed no cycle-time gain.
- **Per-component GPU timing** (torch profiler on a 64-token decode) was not taken. Section 3 gives the exchange-side breakdown from the counters.
- **Not built:** MLA q-row split, shared-expert / dense-MLP column split, split MTP draft layer.
  - The draft passes split only through the lm_head.
  - The draft layer's experts stay on the GPU (IQ2, as in config B).
- **Expert side:** the GPU takes 12.1 of the 12.5 resident experts per window. More resident memory or a better hot set would raise its share; the hot set is calibrated on Hermes items only.
- **Prefill** (cold / cached TTFT) is unchanged from config B (all routes on the workstation for windows over 16 tokens); not measured here.
