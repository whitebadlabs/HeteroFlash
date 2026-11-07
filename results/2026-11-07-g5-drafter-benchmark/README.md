# G5 step 1: existing MTP drafter vs draft length, Control and target treatment (2026-11-07)

Setup: `ops/g5_bench.sh` (sudo-less, lok rootless Docker). serveN = owner's serving config (CUDA graphs, prefix cache) with N MTP draft tokens; mtpN = eager trace server with the target treatment (top-6, original-top-8 normalisation, FP8 on all fixed weights) and N draft tokens. Decode: 14 prose (writing_chat holdout) and 36 mixed-domain holdout prompts, 256 tokens, temperature 1, ignore_eos; committed tokens per verify from vLLM spec-decode counters. Spark projection: `src/g5_analysis.py`, using the decode_sim verify step for a window of N+1 tokens (real routing + measured Spark/CPU costs, k6, FP8 fixed 8.3 GB, 50% of experts on the Spark, 0.05 ms link) plus N MTP draft passes (~0.45 GB each, 2.0 ms on the Spark).

| N | Control prose: committed / accept / cycle / tok/s | Control mixed: committed / tok/s | Target prose: committed / accept | Target mixed: committed | Spark cycle (verify + drafts) | Spark tok/s (prose) | committed needed for 125 |
|---|---|---|---|---|---|---|---|
| 1 | 1.751 / 75% / 14.14 ms / 124.1 | 1.788 / 125.49 | 1.751 / 75% | 1.792 | 54.6 + 2.03 = 56.63 ms | 30.9 | 7.08 |
| 3 | 2.429 / 48% / 16.07 ms / 154.88 | 2.618 / 160.56 | 2.442 / 48% | 2.619 | 64.77 + 6.08 = 70.85 ms | 34.5 | 8.86 |
| 5 | 2.589 / 32% / 19.87 ms / 140.34 | 2.913 / 151.16 | 2.653 / 33% | 2.93 | 73.56 + 10.14 = 83.7 ms | 31.7 | 10.46 |
| 7 | 2.705 / 24% / 21.74 ms / 130.29 | 2.889 / 132.14 | 2.698 / 24% | 2.959 | 81.32 + 14.19 = 95.51 ms | 28.2 | 11.94 |

Control cycle fit on prose: cycle = 12.64 ms + 1.33 ms x N (not constant).

## Go/no-go
- The target treatment does not reduce drafter acceptance: committed tokens per verify match the Control within noise at every N.
- The existing MTP drafter saturates at ~2.7 committed tokens on prose (per-position acceptance 75% -> 48% -> 33% -> 24% at N = 1/3/5/7). Longer drafts add cycle cost on both machines; the Control's optimum is N = 3 (155 tok/s prose).
- On the Spark deployment the verify step grows with the window (54.6 -> 81.3 ms for 2 -> 8 tokens). At every N, 125 tok/s needs 7-12 committed tokens per cycle, 3-4x what this drafter delivers and above the best published chat acceptance (~5-6 with large drafters). A compute-capped drafter pilot (+0.3-1.0 committed tokens per the literature) would leave the projection at ~40-45 tok/s.
- **Decision proposal: NO-GO on expensive drafter training as the primary lever.** The gap must close mainly in the verify step (~55-80 ms simulated vs ~17-20 ms needed): fixed-weight bytes (QAD toward NVFP4), expert bytes and union, and a faster link. A small pilot remains useful for the paper but is not on the critical path.
