# Path to the hard targets without new hardware (2026-11-07)

This plan has three inputs:

- **Deep research:** 5 search angles; 102 agents fetched sources and checked claims with three adversarial votes each.
- **Locality oracle:** run on the M1 decode traces.
- **E2E measurements:** decode, prefill and kernel rooflines from 2026-11-07.

The hardware is fixed: one DGX Spark, the workstation's CPU and RAM, and the existing 10 GbE direct link. Max-Q GPUs are used only for offline training and evaluation.

## What the literature says (verified claims)

| Technique | Best verified result | Quality | Fit here |
|---|---|---|---|
| EcoSpec (cost-aware draft-tree selection) | Unique experts per step 23.7 -> 20.5 (-13.5%) on Qwen3-235B | lossless | yes, drops into the verifier |
| MoE-Spec (per-layer expert budget at verify) | +10-30% over EAGLE-3 | lossy when aggressive (Mixtral 78% quality at max) | moderate budgets only |
| DraftExpert (self-speculation with a resident draft expert) | 1.45x end to end, acceptance 84-87% per token | lossless | 16B-class evidence only |
| DFlash block-diffusion drafter | tau 6.5-7.9 on math/code; **tau 4.2-4.4 on MT-Bench prose** | lossless | needs training on GLM hidden states; MoE speedup per accepted token is ~0.4 vs 0.64 on dense models |
| NVFP4 quantization-aware distillation (QAD) | Near-BF16 on Nemotron hybrids with 0.3-6B tokens | ~lossless for MoE/Mamba weights | **attention was kept BF16 in every hybrid model; NVFP4 attention is unproven** |
| INT4 QAT of routed experts | Production (Kimi K2 Thinking, g32) | vendor says lossless | yes |
| ~3-bit trellis quantization (QTIP / EXL3) | +0.044 nats on Llama 2 70B | small but measurable | yes (Q1 tests the PTQ version) |
| Aggressive MoE compression (pruning + ~2 bit) | DeepSeek-V3 at 130 GB roughly matches Q2_K_L at 230 GB | no BF16 comparison | no (non-inferiority not shown) |

No source reaches 7-12 accepted tokens on general prose. No published DGX Spark result exists for a 300B-class MoE.

## Locality oracle (M1 decode traces, no training)

The test lets a token swap one of its experts for one already used in the same verify window, but only when the replacement scores within delta of the original.

| window | baseline union | delta 0.05 | delta 0.1 |
|---|---|---|---|
| 4 tokens | 18.6 | 16.9 (4.2% routing mass moved) | 15.5 (7.9%) |
| 7 tokens | 26.4 | 23.3 (5.0%) | 21.0 (9.3%) |

Routing is sharp, so after-the-fact locality is not a free lever. A locality-regularised router distillation would have to beat this oracle substantially to matter, and that remains a research bet.

## Decode model for the full stack

The model is: t_step = max(Spark bytes / 241 GB/s, CPU bytes / 182 GB/s), with the Spark/CPU expert split chosen to balance the two, plus a sync term.

Inputs:

- **Fixed weights:** NVFP4 after QAD, 4.6 GB.
- **Experts:** 3-bit at 9.4 MB each, or 4-bit at 12.6 MB each.
- **Verify window:** 7 tokens, with an EcoSpec-controlled union of about 23 experts per layer.
- **Drafter:** tau 4.5, about the measured DFlash prose level.

| experts | sync per step | step | tok/s |
|---|---|---|---|
| 4-bit | 8 ms (42 x RTT, kernel TCP) | 47.7 ms | 94 |
| 4-bit | 3 ms (busy-poll, overlapped) | 42.7 ms | 105 |
| 3-bit | 8 ms | 40.4 ms | 111 |
| 3-bit | 3 ms | 35.4 ms | **127** |

The sum of the bets reaches the target band only if all of these pay off together:

1. NVFP4 attention survives QAD at non-inferior quality.
2. ~3-bit experts pass, through PTQ (Q1) or QAT.
3. A trained prose drafter reaches tau >= 4.5 at window 7.
4. EcoSpec keeps the union at about 23 experts per layer.
5. A custom runtime overlaps Spark and CPU expert reads, with <= 3 ms of sync per step.
6. Kernels run near the bandwidth roofline: 241 GB/s on the Spark, and the CPU kernel reaches ~180 GB/s on scattered experts.

## Cold 45K TTFT and cached turn under the same stack

- **Cold 45K.** With ~3-bit experts, 37 MoE layers stay on the Spark: about 100 GB of experts plus 4.6 GB of fixed weights plus about 3 GB of KV/state. The other 5 layers run on the workstation CPU with int8 GEMM (about 1.0-1.2 s per layer per 45K tokens, overlapped). The GPU runs large-chunk NVFP4 MoE prefill (about 5 s). Estimate: 7-9 s, within the <= 10.8 s target.
- **Cached 45K + 32.** A 32-token step touches about 120 experts per layer, about 0.15-0.2 s. That is borderline against the 0.2 s target.

## Experiment order (risk first; all on existing hardware)

1. **Q1, PTQ 3-bit: FAILED (2026-11-07).** Routed experts were quantized to IQ3_XXS (~3.06 bpw, imatrix from the calibration split), against the Q4_K-expert reference on the 6-domain holdout (20 x 2048 tokens).

   | metric | IQ3_XXS vs Q4_K |
   |---|---|
   | dNLL | +0.068 +/- 0.004 nats |
   | mean KLD | 0.138 (median 0.031, p99 1.69) |
   | same top token | 87.0% |

   - That is about 10x the FP8 budget, so post-training 3-bit quantization is not non-inferior.
   - Bet 1 now requires quantization-aware training of the experts (QAT/QAD on the Max-Q rig).
   - The IQ3_S (3.44 bpw) and IQ2_S (2.5 bpw) curve points are queued.
2. **NVFP4 QAD of attention and shared experts** on the Max-Q rig. This is the largest quality risk. Gate on KLD and the M2 nats margin.
3. **Link latency: PASSED (2026-11-07).** `runtime/link_pingpong.c` was run Spark <-> workstation over the 10G direct link with TCP_NODELAY, no sudo, and the workstation at load 32.

   | message (each way) | default sockets, p50 | SO_BUSY_POLL + spin recv, p50 | busy p99 |
   |---|---|---|---|
   | 8 KB | 317 us | 50 us | 57 us |
   | 64 KB | 425 us | 151 us | 498 us |

   - Single-token decode sync over 42 layers falls to about 2 ms per token.
   - A 7-token verify exchange costs about 0.1-0.15 ms per layer, which per-layer overlap with Spark-side work can hide.
4. **Prose drafter.** Train a DFlash-style block drafter on GLM-5.3-Flash hidden states using the Max-Q rig. Measure prose tau and the expert union with EcoSpec selection.
5. **Runtime.** Extend the HeteroFlash expert server: a per-layer split of the expert set between Spark and CPU, one message per layer, batched int8/int4 GEMM for prefill, overlapped with GPU work.

Decision rule: each gate is measured, and failures are reported as negative results. If gates 1-2 fail, the decode ceiling at non-inferior quality falls back to about 60-80 tok/s.
