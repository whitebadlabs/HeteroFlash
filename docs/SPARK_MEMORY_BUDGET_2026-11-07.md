# Spark memory budget (pre-check B, 2026-11-07)

All figures are GiB, with 121 GiB total as reported by `free -g` on the Spark.

Tags: **[M]** measured; **[L]** from a published model card; **[P]** projected.

## Overheads

| item | GiB | basis |
|---|---|---|
| OS + base services (Qwen stopped) | 5.0 | [M] `free -g` before model load |
| CUDA context + llama.cpp KV for a 49K context + compute buffers | 3.6 | [M] 98 GiB in use after load (E2E-v1), minus 5 base, minus 89.4 of weights on the Spark |
| one 45K-token sequence state (KV + KDA recurrent state) | 1.0 | [M] llama-server prompt-cache state size: 1,043,339,916 bytes |
| drafter: DFlash2 bf16 weights + buffers | 3.0 | [L] 2.34 GB of weights (incoai config) + [P] ~0.5 GiB buffers. The llama.cpp MTP draft context measured +6 GiB [M]: 104 vs 98 GiB |
| large-chunk prefill activations (8-16K-token chunks) | 3.0 | [P] |
| runtime staging buffers for the CPU-expert exchange | 1.0 | [P] |
| **total overhead** | **16.6** | of which 9.6 measured |

## Experts that fit on the Spark

Expert sizes:

- 4-bit (Q4_K): 14.16 MB per expert [M], from the GGUF tensor table.
- 3-bit (IQ3_XXS): 9.74 MB per expert [M], from the 131 GB IQ3_XXS GGUF. That is 2.61 GiB per layer, against 3.80 GiB at 4-bit.

| fixed weights | experts | room for experts | MoE layers on Spark / remote |
|---|---|---|---|
| Q8_0 (current, 9.6 GiB) | 4-bit | 94.8 GiB | 24 / 18 |
| FP8 (8.8 GiB) | 4-bit | 95.6 GiB | 25 / 17 |
| NVFP4 after QAD (4.6 GiB) | 4-bit | 99.8 GiB | 26 / 16 (1.1 GiB left) |
| Q8_0 | 3-bit | 94.8 GiB | 36 / 6 (0.8 GiB left) |
| **FP8** | **3-bit** | **95.6 GiB** | **36 / 6 (1.6 GiB left)** |
| NVFP4 after QAD | 3-bit | 99.8 GiB | 38 / 4 (0.5 GiB left) |

## Consequences

- **Cold 45K TTFT** needs <= 5-6 remote layers [P, from the measured CPU int8 GEMM roofline]. That requires 3-bit experts.
  - FP8 fixed weights give 6 remote layers, the edge of the budget.
  - NVFP4 fixed weights give 4 remote layers.
  - 4-bit experts leave 16-18 layers remote, which is incompatible with 10.8 s.
- **The drafter choice matters for memory.**
  - DFlash2 (~3 GiB) is cheaper than llama.cpp's MTP context as implemented today (+6 GiB measured).
  - A custom runtime could hold the MTP head in about 0.3 GiB.
- **Margin.** Only 0.5-1.6 GiB of headroom is left in the 3-bit rows, and 7 GiB of the overhead is [P]. The budget is tight: the [P] items (prefill activations, staging buffers) must be measured in the runtime before the placement is final. Dropping one more layer to the CPU frees 2.6 GiB.
