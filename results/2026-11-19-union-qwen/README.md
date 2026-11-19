# Union-aware routing and the mean-output stand-in on a second model: Qwen3-30B-A3B (2026-11-19)

Teacher-forced bf16 forward (HF transformers) on one Max-Q GPU; 14 chunks x 2048 Qwen tokens from the balanced 7-domain
eval text (2 per domain); KL = mean full-vocabulary KL(top-8 reference || variant). Windows = 4 consecutive positions
(speculative-decoding verify window). Same definitions as the GLM runtime (drop_win_kernel / mean_sub_kernel): no
renormalisation after drops. K=8 through the hook reproduces the reference exactly.

| K | plain kK | union kKu (ratio) | mean kKs | union+mean kKus | expert reads / window (kK = kKu) |
|---|---|---|---|---|---|
| 6 | 0.0398 | 0.0226 (0.57) | 0.0396 | 0.0227 | 14.4 |
| 5 | 0.0707 | 0.0432 (0.61) | 0.0767 | 0.0431 | 12.3 |
| 4 | 0.1260 | 0.0766 (0.61) | 0.1454 | 0.0818 | 10.1 |
| 3 | 0.2239 | 0.1465 (0.65) | 0.2960 | 0.1703 | 7.8 |

- Union routing transfers: KL x0.57-0.65 of plain top-K, better on 14/14 chunks at every K, at the same expert reads
  (GLM subset ratios: 0.76 / 0.76 / 0.69 at K = 6 / 5 / 4). k5u reaches k6-level quality at k5-level reads.
- The mean-output stand-in does NOT transfer: worse than plain drops at K <= 5, no gain over union; oracle (in-sample) and
  layer-selected means are neutral. It helped strongly on GLM-5.3-Flash (k5 0.085 -> k5us 0.045). Inferred: model-specific.
- Caveats: prefill-style forward, not real decode with KV cache; bf16; per-domain samples are 4,096 tokens.
Files: union_qwen.py, results.json (per chunk), followup_mean.{py,json}, diag_mean.py, run.log.
