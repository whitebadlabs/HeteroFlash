# Pre-checks A and C: kernel baselines on the Spark GB10 (2026-11-07, all [M])

## A. Routed-expert MUL_MAT_ID by quant type

Tool: llama.cpp `test-backend-ops perf`, with an `HF_GLM_PERF` case set for GLM-5.3-Flash.

- **Shape:** 288 experts, top-8, one 2048x4096 matrix (gate/up) or one 4096x2048 matrix (down).
- **Ids:** random per token, so distinct experts grow with tokens. That matches a verify batch.
- **Raw output:** `mmid_raw.txt`.

| type (bpw) | n=1 | n=4 | n=8 | n=8 vs Q4_K | n=512 |
|---|---|---|---|---|---|
| Q8_0 (8.5) | 300 us | 1339 us | 2697 us | 1.60 | 18.0 ms |
| Q4_K (4.5) | 178 us | 834 us | 1681 us | 1.00 | 11.7 ms |
| Q3_K (3.44) | 116 us | 767 us | 1558 us | 0.93 | 11.0 ms |
| IQ3_S (3.44) | 116 us | 642 us | 1296 us | 0.77 | 11.7 ms |
| IQ3_XXS (3.06) | 73 us | 525 us | 1049 us | 0.62 | n/a (run timed out) |

All times are for the gate/up matrix.

- At n=4-8, IQ3_XXS converts its byte reduction (0.68x) fully into time (0.62x), at about 174 GB/s effective. Q3_K barely helps.
- Effective bandwidth of the existing MoE GEMV kernels at verify sizes is about 132-174 GB/s, not the 241 GB/s peak used in the plan model.
- The n=1 timings are inflated by cache reuse (the same 8 experts every run) and are not representative.

## C. Attention-projection GEMM: BF16 vs FP8 vs NVFP4

Tool: vLLM v0.32.0, `torch._scaled_mm` (FP8) and `cutlass_scaled_fp4_mm` (NVFP4). Raw output: `attn_fp8_nvfp4.json`.

| shape K x N | M | FP8 | NVFP4 | NVFP4 speedup |
|---|---|---|---|---|
| 4096 x 12288 | 1 / 4 / 8 | 231 / 234 / 229 us (~218 GB/s) | 121 / 113 / 103 us (235-274 GB/s) | 1.9-2.2x |
| 12288 x 4096 | 1 / 4 / 8 | 230 / 226 / 232 us | 134 / 136 / 126 us | 1.7-1.8x |
| 4096 x 12288 | 4096 | 2276 us (181 TF) | 1564 us (264 TF) | 1.45x |
| 12288 x 4096 | 4096 | 2317 us (178 TF) | 2482 us (166 TF) | 0.93x |

- The 4096x4096 and 4096x2048 shapes fit in cache and show artefactual >800 GB/s. They are excluded from conclusions.
- Quality baselines come from M2 on the Max-Q rig: FP8 fixed weights +0.006 nats; NVFP4 fixed weights +0.040 nats, of which attention is +0.030.
