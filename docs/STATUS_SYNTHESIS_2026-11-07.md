# Where HeteroFlash stands (2026-11-07, after G1-G3 first passes)

## Measured on the Control (original GLM-5.3-Flash, dual Max-Q)
- Cold 45K TTFT p50 11.10 s (p95 11.30 s); cached 45K+32 p95 161 ms; prose TPOT 6.70 ms (~151 tok/s); MTP commits 2.43 tokens per verify on prose (2.73 mixed); verify step ~16 ms.

## Quality cost of training-free treatments (M2, NLL increase in nats/token vs Control)
| Treatment | NLL | Verdict |
|---|---|---|
| FP8 all fixed weights (attention, shared, dense, lm_head) | 0.006 | adopt: fixed reads 16.6 -> ~8.3 GB/step |
| NVFP4 all fixed | 0.040 (attention 0.030 of it) | needs QAD for attention; NVFP4 lm_head/shared are cheap |
| k6, original-top-8 normalisation | 0.060 | quality-safe operating point |
| k5 / k4 / k3, original-top-8 normalisation | 0.117 / 0.205 / 0.398 | too lossy on prose/multilingual |
| linear or small non-linear dropped-expert compensation | recovers 1-11% / <=4% on unseen domains | does not generalize (negative) |

## Deployment-side measurements
- Spark GPU 241 GB/s read, ~222 GB/s expert streaming, NVFP4 GEMM ~300 TFLOPS. The Spark Arm tier is starved by GPU traffic (~18 GB/s).
- Workstation CPU expert kernel: 73.5 us/expert int4; int8 path 2% error vs 21% for int4. Remote calls over 1 GbE: 1.0-1.5 ms/layer network at 4-8 tokens (link upgrade mandatory).

## Implication
With the quality-safe treatments (k6 + original-top-8 normalisation + FP8 fixed), the simulated Spark+CPU verify step is ~68 ms at a 4-token window. At the Control's MTP acceptance (2.43) that is ~36 tok/s; 125 tok/s would need ~8.5 committed tokens per verify. The levers left, in order of expected size:
1. **Drafter (G5):** committed tokens per verify must rise far above 2.43, since every configuration needs ~6-9. Retrain or extend the MTP drafter (multi-step / block drafting) on general-purpose data against the modified model.
2. **Fixed weights to ~4.7-7 GB:** QAD of attention toward NVFP4 (M2: training-free NVFP4 attention costs 0.030).
3. **Expert bytes and union:** lower-bit GPU experts and a hardware-aware router that reuses experts across the verify window (union regularisation), evaluated at k = 6.
4. **Link:** 10G/ConnectX to make the CPU tier usable for multi-token windows. Prefill (G4) needs it for expert staging.
5. Reduced-k (k <= 5) only with large-scale distillation; small compensators failed.
