# M2 round 1: training-free top-k and fixed-weight precision on the Control model (2026-11-07)

Method: `src/m2_router_surgery.py` against the trace server (hook v3, eager, TP2, gpu util 0.90). 81 held-out prompts (6 domains; the two ~40K long-context prompts are excluded and measured separately), greedy top-8 baseline generations of up to 256 tokens, then teacher-forced prompt+generation under each configuration with top-64 prompt logprobs. Metrics vs the `k8_ref` reference: mean NLL increase on the actual tokens (exact), top-1 agreement, and a top-64 forward-KL lower bound. `k8_repeat` measures run-to-run noise (exactly 0 here).
Top-k surgery renormalises over the kept experts (router default). Fixed-weight precision is simulated in place (FP8 128x128 blocks; NVFP4 16-blocks with e4m3 scales) on attention projections (incl. MLA kv_b), dense MLPs and lm_head (~14.4 GB of 16.6 GB). In round 1 the shared experts were **not** included (pattern bug); round 2 covers them.

| Config | tokens | NLL increase (nats/token) | top-1 agreement | KL lower bound mean | KL p95 |
|---|---|---|---|---|---|
| k8_ref | 14936 | 0.0000 | 1.0000 | 0.0000 | 0.000 |
| k8_repeat | 14936 | 0.0000 | 1.0000 | 0.0000 | 0.000 |
| k6_active | 14936 | 0.0633 | 0.9401 | 0.0641 | 0.341 |
| k4_active | 14936 | 0.2570 | 0.8719 | 0.2318 | 1.230 |
| k3_active | 14936 | 0.4892 | 0.8140 | 0.4398 | 2.224 |
| k2_active | 14936 | 1.1802 | 0.6778 | 1.0684 | 4.398 |
| fixed_fp8 | 14936 | 0.0055 | 0.9768 | 0.0106 | 0.052 |
| fixed_nvfp4 | 14936 | 0.0326 | 0.9545 | 0.0385 | 0.202 |
| k4_fixed_nvfp4 | 14936 | 0.2835 | 0.8655 | 0.2577 | 1.412 |

## NLL increase by domain
| Config | writing_chat | knowledge_qa | math | code | multilingual | instruction_following |
|---|---|---|---|---|---|---|
| k6_active | 0.0822 | 0.0394 | 0.0156 | 0.0090 | 0.0909 | 0.0746 |
| k4_active | 0.3389 | 0.1969 | 0.0839 | 0.0249 | 0.3438 | 0.2973 |
| k3_active | 0.5752 | 0.3900 | 0.2446 | 0.0854 | 0.6464 | 0.5850 |
| fixed_fp8 | 0.0025 | 0.0047 | 0.0028 | 0.0011 | 0.0114 | 0.0063 |
| fixed_nvfp4 | 0.0454 | 0.0282 | 0.0094 | 0.0044 | 0.0440 | 0.0327 |
| k4_fixed_nvfp4 | 0.3655 | 0.2080 | 0.1127 | 0.0328 | 0.3810 | 0.3290 |

## Findings
- FP8 fixed weights are near-lossless (0.0055 nats, KL >= 0.011): the fixed per-step read can drop from 16.6 GB to ~9 GB without training.
- NVFP4 fixed weights cost 0.033 nats (prose 0.045, multilingual 0.044): a candidate for quantization-aware distillation.
- Training-free top-k reduction is cheap at k=6 (0.063 nats) but large at k=4 (0.257). The loss is strongly domain-dependent: code 0.025, math 0.084, knowledge 0.197, instruction following 0.297, prose 0.339, multilingual 0.344. Prose and multilingual are the domains to protect; router recovery (G2) is required before k=4 is usable.
- The costs add roughly: k4 + NVFP4 = 0.284 vs 0.257 + 0.033.

## Decode simulation with these levers (`src/decode_sim.py`, real routing, measured costs, ~0.05 ms link; MODEL)
| Config | Step | tok/s @ 2.43 committed (Control MTP) | @ 4 | committed needed for 125 |
|---|---|---|---|---|
| k8 + FP8 fixed (~9 GB) | 75.5 ms (w4) | 32 | 53 | 9.4 |
| k6 + FP8 fixed | 67.9 ms (w4) | 36 | 59 | 8.5 |
| k6 + NVFP4 fixed (4.7 GB) | 48.7 ms (w4) | 50 | 82 | 6.1 |
| k4 + NVFP4 fixed (needs recovery) | 40.7 ms (w4) | 60 | 98 | 5.1 |

The remaining gap is now mostly committed tokens per verify (drafter quality on the modified model) and step time; the quality data show which reductions need recovery training.
