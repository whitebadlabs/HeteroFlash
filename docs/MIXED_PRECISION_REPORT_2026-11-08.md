# Mixed-precision report (2026-11-08): six items before any Max-Q GPU pilot

This report uses only the Spark GPU and the workstation CPU. No Max-Q GPU time has been used.

Tags: **[M]** measured; **[P]** modeled.

The hard targets and quality criteria are unchanged.

## Summary

- The mixed-precision hypothesis is **mostly falsified at model level**. Keeping Q4_K experts on the "sensitive" early layers 3-15 and IQ3_XXS on 16-44 costs **+0.058 nats against the BF16 proxy**. All-IQ3_XXS costs +0.074, so the mix recovers only about 20% of the 3-bit loss. All-Q4_K costs +0.006 [M].
- The per-layer "sensitivity" pattern was largely an artefact of the metric.
  - Relative routed-output error is high exactly where a layer's output is small: layers 3-15, 32, 35, 37 and 39 have output RMS 0.03-0.08.
  - In absolute terms, the largest 3-bit errors are in late, large-output layers: 42, 43, 29, 44 and 17 (plus layer 3).
- End to end, mixed precision gives small measured gains over the Q4_K baseline: decode up to 12.5 vs 10.5 tok/s, cold 45K 198 vs 223 s, cached +32 0.59 vs 0.68 s [M]. **All targets remain far away.** The llama.cpp RPC runtime, which is serial, dominates the latency.

## 1. Per-layer absolute error and relative sensitivity [M]

**Method** (`src/expert_quant_sensitivity.py`):

- Inputs: the captured MoE-layer inputs on the control model.
- Routing: GLM top-8, recomputed from the captured router logits.
- Comparison: the routed output `sum_e w_e E_e(x)` with the official BF16 experts, against the same output with experts dequantized from each GGUF (gguf-py).
- Metrics: `rel = ||y_q - y|| / ||y||` and `abs = ||y_q - y|| / sqrt(numel)`.

**Sample:**

- 768 tokens per layer, 192 per domain.
- Domains: **code, knowledge_qa, math, writing_chat only**. The existing captures stopped at their row cap before the multilingual and instruction_following requests, so those two domains are not covered here.
- All 42 MoE layers.

**Formats:**

- Q6_K: no imatrix.
- Q4_K: no imatrix, the E2E reference.
- IQ3_XXS: imatrix from the calibration split.

The imatrix confound remains: some of the IQ3 vs Q4_K comparison reflects the imatrix, not only the format.

| layer | output RMS | Q6_K rel | Q4_K rel | IQ3_XXS rel | IQ3/Q4_K | Q4_K abs | IQ3_XXS abs | IQ3 abs increase | IQ3 rel range across domains |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 0.567 | 0.016 | 0.065 | 0.174 | 2.66 | 0.0370 | 0.0986 | +0.0615 | 0.163-0.179 |
| 4 | 0.031 | 0.023 | 0.095 | 0.232 | 2.44 | 0.0029 | 0.0071 | +0.0042 | 0.225-0.239 |
| 5 | 0.030 | 0.025 | 0.104 | 0.261 | 2.50 | 0.0032 | 0.0079 | +0.0048 | 0.255-0.267 |
| 6 | 0.108 | 0.017 | 0.072 | 0.195 | 2.70 | 0.0078 | 0.0210 | +0.0132 | 0.191-0.267 |
| 7 | 0.037 | 0.021 | 0.088 | 0.207 | 2.35 | 0.0032 | 0.0076 | +0.0044 | 0.190-0.258 |
| 8 | 0.049 | 0.019 | 0.077 | 0.160 | 2.07 | 0.0038 | 0.0079 | +0.0041 | 0.142-0.184 |
| 9 | 0.036 | 0.021 | 0.088 | 0.198 | 2.25 | 0.0031 | 0.0071 | +0.0039 | 0.162-0.254 |
| 10 | 0.105 | 0.017 | 0.073 | 0.084 | 1.16 | 0.0076 | 0.0089 | +0.0012 | 0.056-0.238 |
| 11 | 0.033 | 0.021 | 0.087 | 0.210 | 2.41 | 0.0029 | 0.0069 | +0.0041 | 0.200-0.220 |
| 12 | 0.029 | 0.020 | 0.085 | 0.209 | 2.47 | 0.0025 | 0.0061 | +0.0036 | 0.205-0.222 |
| 13 | 0.046 | 0.020 | 0.085 | 0.208 | 2.46 | 0.0039 | 0.0096 | +0.0057 | 0.205-0.212 |
| 14 | 0.152 | 0.016 | 0.069 | 0.087 | 1.26 | 0.0105 | 0.0132 | +0.0027 | 0.079-0.101 |
| 15 | 0.075 | 0.018 | 0.074 | 0.146 | 1.96 | 0.0056 | 0.0110 | +0.0054 | 0.127-0.201 |
| 16 | 0.163 | 0.015 | 0.062 | 0.072 | 1.15 | 0.0101 | 0.0117 | +0.0016 | 0.061-0.090 |
| 17 | 2.041 | 0.004 | 0.016 | 0.027 | 1.62 | 0.0336 | 0.0545 | +0.0209 | 0.027-0.077 |
| 18 | 0.460 | 0.016 | 0.066 | 0.049 | 0.74 | 0.0303 | 0.0224 | -0.0079 | 0.046-0.082 |
| 19 | 0.519 | 0.008 | 0.035 | 0.032 | 0.92 | 0.0179 | 0.0165 | -0.0014 | 0.029-0.237 |
| 20 | 0.185 | 0.012 | 0.053 | 0.065 | 1.23 | 0.0097 | 0.0120 | +0.0023 | 0.056-0.108 |
| 21 | 1.228 | 0.005 | 0.021 | 0.024 | 1.15 | 0.0260 | 0.0299 | +0.0039 | 0.024-0.209 |
| 22 | 0.962 | 0.005 | 0.022 | 0.026 | 1.20 | 0.0207 | 0.0250 | +0.0042 | 0.025-0.239 |
| 23 | 1.185 | 0.005 | 0.024 | 0.027 | 1.14 | 0.0284 | 0.0325 | +0.0041 | 0.027-0.234 |
| 24 | 0.689 | 0.006 | 0.026 | 0.028 | 1.07 | 0.0178 | 0.0191 | +0.0012 | 0.026-0.238 |
| 25 | 0.969 | 0.005 | 0.021 | 0.038 | 1.84 | 0.0199 | 0.0366 | +0.0167 | 0.037-0.255 |
| 26 | 0.241 | 0.013 | 0.053 | 0.064 | 1.21 | 0.0127 | 0.0153 | +0.0026 | 0.054-0.258 |
| 27 | 0.140 | 0.018 | 0.072 | 0.092 | 1.27 | 0.0102 | 0.0129 | +0.0027 | 0.081-0.100 |
| 28 | 0.459 | 0.013 | 0.054 | 0.041 | 0.76 | 0.0248 | 0.0188 | -0.0060 | 0.039-0.045 |
| 29 | 2.434 | 0.006 | 0.027 | 0.026 | 0.95 | 0.0657 | 0.0625 | -0.0031 | 0.025-0.093 |
| 30 | 0.436 | 0.008 | 0.036 | 0.046 | 1.27 | 0.0158 | 0.0200 | +0.0042 | 0.037-0.269 |
| 31 | 0.906 | 0.008 | 0.033 | 0.036 | 1.11 | 0.0295 | 0.0327 | +0.0033 | 0.033-0.266 |
| 32 | 0.061 | 0.026 | 0.109 | 0.276 | 2.54 | 0.0067 | 0.0170 | +0.0103 | 0.268-0.285 |
| 33 | 1.577 | 0.006 | 0.024 | 0.028 | 1.18 | 0.0376 | 0.0443 | +0.0067 | 0.026-0.252 |
| 34 | 0.801 | 0.010 | 0.041 | 0.040 | 0.98 | 0.0327 | 0.0319 | -0.0008 | 0.038-0.129 |
| 35 | 0.070 | 0.026 | 0.106 | 0.248 | 2.35 | 0.0074 | 0.0173 | +0.0099 | 0.215-0.274 |
| 36 | 0.754 | 0.010 | 0.044 | 0.041 | 0.93 | 0.0333 | 0.0309 | -0.0025 | 0.038-0.274 |
| 37 | 0.084 | 0.023 | 0.097 | 0.235 | 2.41 | 0.0082 | 0.0198 | +0.0116 | 0.210-0.269 |
| 38 | 0.811 | 0.009 | 0.040 | 0.049 | 1.22 | 0.0322 | 0.0395 | +0.0072 | 0.045-0.263 |
| 39 | 0.073 | 0.026 | 0.107 | 0.271 | 2.52 | 0.0079 | 0.0198 | +0.0120 | 0.267-0.278 |
| 40 | 1.163 | 0.010 | 0.040 | 0.046 | 1.14 | 0.0471 | 0.0537 | +0.0067 | 0.044-0.275 |
| 41 | 1.008 | 0.011 | 0.045 | 0.047 | 1.04 | 0.0454 | 0.0474 | +0.0020 | 0.043-0.093 |
| 42 | 2.968 | 0.007 | 0.032 | 0.037 | 1.15 | 0.0947 | 0.1092 | +0.0144 | 0.036-0.038 |
| 43 | 1.635 | 0.007 | 0.035 | 0.052 | 1.47 | 0.0574 | 0.0843 | +0.0269 | 0.049-0.206 |
| 44 | 0.955 | 0.008 | 0.038 | 0.065 | 1.73 | 0.0359 | 0.0621 | +0.0262 | 0.060-0.174 |

**Observations:**

- **The ranking by relative error is driven by output magnitude.**
  - The IQ3/Q4_K ratio is 2.0-2.7 on every layer with output RMS < 0.08, and 0.7-1.8 on layers with large outputs.
  - Absolute error ranks the late, large-output layers highest. Top by IQ3 absolute error: 42, 3, 43, 29, 44, 17, 40, 41, 33, 38.
  - Top by the IQ3-minus-Q4_K absolute increase: 3, 43, 44, 17, 25, 42, 6, 39, 37, 32.
- **Domain spread.** On 20 layers, one domain's IQ3 relative error exceeds twice the lowest (mostly writing_chat or knowledge_qa).
  - The same layers show 2-5x domain spread for Q4_K as well.
  - This is a property of small-norm token subsets under a relative metric, not a 3-bit-specific domain effect.
- **Neither per-layer metric is validated as a predictor of end-to-end loss.** Item 2 shows the relative ranking was not (see the mixed-model result).

## 2. Model-level quality [M]

**Method:**

- Data: 6-domain held-out text (`data/q_kld_holdout.txt`, holdout split, 83 prompts). 20 chunks x 2048 tokens; llama.cpp scores the second half of each chunk.
- Metrics: paired KLD and dNLL against reference logits from the same run setup.
- Fixed weights are Q8_0 in every model.

**The BF16 proxy.** The full BF16 model (640 GB) does not fit the Spark plus workstation, so the reference is **routed experts at Q8_0** (341 GB). Q6_K against this proxy gives KLD 0.012 and dNLL -0.001.

| experts | size | dNLL vs Q8_0 proxy | KLD vs Q8_0 | p99 KLD | same top token | dNLL vs Q6_K | KLD vs Q6_K |
|---|---|---|---|---|---|---|---|
| Q6_K (6.56 bpw) | 266 GB | -0.001 +/- 0.001 | 0.012 | 0.14 | 96.3% | (ref) | (ref) |
| Q4_K (4.5) | 186 GB | +0.006 +/- 0.002 | 0.038 | 0.45 | 93.1% | +0.007 | 0.038 |
| **Mixed: Q4_K L3-15 + IQ3_XXS L16-44** | **148 GB** | **+0.058 +/- 0.004** | **0.102** | 1.27 | 89.1% | +0.059 | 0.101 |
| IQ3_S (3.44) | 145 GB | n/a | n/a | n/a | n/a | +0.059 | 0.099 |
| IQ3_XXS (3.06) | 131 GB | +0.074 +/- 0.004 | 0.126 | 1.61 | 87.7% | +0.075 | 0.126 |

What this shows:

- The mixed model is about 9x the FP8 fixed-weight budget (+0.006).
- It is no better than uniform IQ3_S at the same size.
- Protecting the low-output early layers was the wrong choice.

**Not yet measured:**

- the actual BF16 control on the Max-Q rig;
- task benchmarks (only NLL/KLD so far).

## 3. Checkpoint size, Spark residency, headroom, remote layers [M]

The mixed GGUF is 147.9 GB on disk. Spark memory in use after load (`free -g`) is measured with no drafter and Qwen stopped:

| placement | Spark-resident experts | remote experts (workstation CPU) | Spark in use | headroom (121 GiB total) |
|---|---|---|---|---|
| Q4_K baseline (E2E-v1b) | L3-23 (Q4_K) | L24-44, 21 layers | 98 GiB | 23 GiB |
| mixed P1: sensitive-early local | L3-15 (Q4_K) + L16-32 (IQ3) | L33-44, 12 layers (IQ3) | 111 GiB | 10 GiB |
| mixed P2: sensitive-early remote | L16-44 (IQ3) | L3-15, 13 layers (Q4_K) | 93 GiB | 28 GiB |

- The memory figures include the llama.cpp KV for a 49K context and its compute buffers.
- A drafter adds +3 GiB (DFlash2 bf16, [P]) or +6 GiB (llama.cpp MTP context, [M]).
- Neither mixed placement reaches the <= 6 remote layers that the cold-TTFT budget needs. That needs about 3 bpw everywhere (pre-check B).

## 4. Spark kernel throughput for these formats [M]

Source: pre-check A, llama.cpp `MUL_MAT_ID` on GB10, GLM shapes (288 experts, top-8, gate/up matrix 2048x4096).

| format | 1 token | 4 tokens | 8 tokens | 8 tokens vs Q4_K | effective GB/s at 8 tokens |
|---|---|---|---|---|---|
| Q4_K | 178 us | 834 us | 1681 us | 1.00 | ~160 |
| IQ3_XXS | 73 us | 525 us | 1049 us | 0.62 | ~174 |

The single-token numbers are inflated by cache reuse in the test.

In the E2E decode runs, Spark GPU utilization was 72-76% with mixed experts, against 66% for the Q4_K baseline. The workstation rpc-server ran at about 11-12 cores, against 16 for the baseline.

**Not measured:**

- an nsys trace of the mixed run itself;
- per-layer timing inside the E2E run.

## 5. End-to-end effect [M]

All runs use llama.cpp RPC: Spark GPU plus workstation-CPU experts over the 10G link.

- Decode: 10 prose prompts x 256 tokens at temperature 1, no drafter.
- TTFT: 45K tokens, 2 repeats.
- The CPU analysis container was paused during the mixed runs.

| run | decode TPOT p50 | decode tok/s | cold 45K TTFT p50 | cached 45K+32 p50 | cached 45K+128 p50 |
|---|---|---|---|---|---|
| Q4_K baseline (v1b) | 94.7 ms | 10.5 | 223.1 s | 0.682 s | 1.247 s |
| mixed P1 (12 remote) | 82.7 ms | 12.1 | 214.1 s | 0.757 s | 1.308 s |
| mixed P2 (13 remote, early) | 80.2 ms | 12.5 | 197.8 s | 0.588 s | 1.113 s |
| target | 7.7-8.3 ms | 120-130 | <= 10.8 s | <= 0.2 s | n/a |

- The improvements are real but small: decode -13 to -15%, cold prefill -4 to -11%.
- The cached-turn result is mixed: P2 is -14%, while P1 is +11%.
- P2 (remote early layers) is faster than P1. The cause is **not attributed**: P2 holds less data on the Spark, and its remote layers differ in format and position. An nsys trace of both runs is needed before drawing a placement conclusion.
- The dominant cost is unchanged: serial per-layer RPC in llama.cpp. Prefill still runs at about 200-230 tok/s.
- The v1b baseline was run earlier on 2026-11-07 with the same script. P1 and P2 were run consecutively on 2026-11-08.

## 6. Which layers still need QAT

The measured answer is: **3-bit PTQ fails at model level, and per-layer error does not tell us which layers carry the loss.** No layer can yet be declared safe at IQ3 PTQ on per-layer evidence.

### Proposed next step (CPU/Spark only, no Max-Q GPU)

Measure the end-to-end loss attribution directly with a few group-substitution models: IQ3_XXS on one layer group, Q4_K elsewhere, KLD against the Q8_0 proxy. Each costs about 50 min of workstation-CPU quantize plus 25 min of Spark eval.

- **G-late:** IQ3 on L33-44 (12 layers; contains the large absolute errors at 42/43/44).
- **G-mid:** IQ3 on L16-32 (17 layers).
- **G-small:** IQ3 on the 14 low-output layers (3-9, 11-13, 15, 32, 35, 37, 39). This is the inverse of what the mixed model tested.
- **Additivity check:** compare the sum of the group costs with the measured all-IQ3 +0.074.

That identifies where 3-bit PTQ is cheapest per GiB saved, and which groups must be trained or kept at 4-bit.

### Proposed bounded QAT pilot budget (for approval; not started)

- **Capture:** an all-6-domain capture on the control model.
  - Hook layer filter and domain tags are already implemented.
  - About 170K tokens per domain, 2 pilot layers.
  - **<= 2 GPU-hours on both Max-Q GPUs.**
- **Training:** IQ3_XXS QAT (STE onto the IQ3_XXS grid, learned scales) on **two layers chosen from the group-substitution result**: the costliest group, and the cheapest group whose cost is still above budget.
  - **<= 2 GPU-hours per layer on one GPU.**
- **Total: <= 6-8 GPU-hours.**
- Pre-registered criterion: `docs/QAT_PILOT_PROTOCOL.md`.
  - Go requires trained IQ3 <= Q4_K PTQ error on held-out domains for both layers.
  - Plateau stop: less than 2% improvement over the last 20% of the budget.
  - E2E resolution limits are reported with the result.
- **Still needed from Carmen:** the overall quality margin for criterion 3 (for example dNLL <= +0.02 vs the BF16 proxy), or a benchmark-based criterion.

## Caveats

- **Domains:** the per-layer scan covers 4 of 6 domains. The model-level KLD covers all 6.
- **Reference:** Q8_0 experts are a proxy for BF16. Fixed weights are Q8_0 in every model.
- **imatrix:** the IQ3 formats use an imatrix; the Q4_K/Q6_K references do not.
- **Mixed-model latency:** each placement is a single E2E configuration with 2 TTFT repeats.
