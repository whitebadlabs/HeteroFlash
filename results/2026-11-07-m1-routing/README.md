# M1: routing of the Control model on the general-purpose calibration set (2026-11-07)

Source: the owner's served GLM-5.3-Flash (EXL3 TR3 3.5bpw, model_revision sha256:33e9930574884035) on the Max-Q rig, eager vLLM, no MTP, no prefix cache. Full 288-way router logits for all 42 MoE layers; selection recomputed with the checkpoint's e_score_correction_bias. 364 prompts (6 domains x 60 + 4 long-context, 25% holdout), 0 errors.
Tokens: prefill 202,910, decode 72,120 (generation: server defaults, temperature 1, top_p 0.95, reasoning high, max 256).

## Decode coverage by resident fraction (mean over 42 layers; min-max across layers)
| Resident experts | Coverage | Layer min | Layer max | P(all 8 resident) |
|---|---|---|---|---|
| 10% | 22.8% | 15.3% | 29.1% | — |
| 25% | 44.5% | 33.4% | 52.5% | — |
| 35% | 56.5% | 44.4% | 64.6% | 2.6% |
| 50% | 71.7% | 59.6% | 78.9% | 11.3% |
| 62% | 82.1% | 71.0% | 87.9% | 26.7% |
| 75% | 91.1% | 82.2% | 95.2% | 52.9% |
| 90% | 98.1% | 93.7% | 99.3% | — |

- Holdout coverage with the hot set chosen on the calibration split: 35% -> 62.5%, 50% -> 74.8%, 62% -> 83.3%.
- Decode expert union per speculative window (distinct experts per layer): 1 -> 8, 2 -> 13.569, 4 -> 22.941, 6 -> 31.036.
- Early layers are flattest: coverage at 50% residency rises from ~60% in the first MoE layers to ~75-79% in the middle layers.
- Every expert in every layer was selected at least once in decode. There are no dead experts on this set.
- Domain hot sets (top 25%) overlap weakly: mean Jaccard 0.11-0.51 (lowest: long_context vs knowledge_qa; highest: writing_chat vs instruction_following). A single static placement is a poor fit for a general-purpose model.
- Gate mass: the top 4 of the 8 selected experts carry 65.4% of the routed weight on average (p10 53.9%, p90 77.8%).

## Trace-driven decode simulation (`src/decode_sim.py`; real holdout routing + measured component costs; MODEL, not an end-to-end measurement)
Spark-resident configurations (<= 105 GB). tau = committed tokens per verify step (an input until measured).

| Experts on Spark | Fixed weights | k_active | Window | RTT ms | tau | Step ms | tok/s | tau needed for 125 |
|---|---|---|---|---|---|---|---|---|
| 50% | 4.7 GB | 4 | 6 | 0.01 | 4.4 | 46.99 | 93.6 | 5.87 |
| 62% | 4.7 GB | 4 | 6 | 0.01 | 4.4 | 50.6 | 87.0 | 6.33 |
| 75% | 4.7 GB | 4 | 6 | 0.01 | 4.4 | 53.75 | 81.9 | 6.72 |
| 75% | 4.7 GB | 4 | 6 | 0.211 | 4.4 | 54.88 | 80.2 | 6.86 |
| 62% | 4.7 GB | 4 | 6 | 0.211 | 4.4 | 55.35 | 79.5 | 6.92 |
| 50% | 4.7 GB | 4 | 6 | 0.211 | 4.4 | 58.67 | 75.0 | 7.33 |
| 50% | 16.6 GB | 8 | 6 | 0.01 | 2.8 | 120.77 | 23.2 | 15.1 |
| 62% | 16.6 GB | 8 | 6 | 0.01 | 2.8 | 128.17 | 21.8 | 16.02 |
| 75% | 16.6 GB | 8 | 6 | 0.01 | 2.8 | 134.68 | 20.8 | 16.84 |

- With the unmodified model (top-8, BF16 fixed weights) the heterogeneous Spark+CPU split gives ~21-23 tok/s at tau 2.8. Top-4 routing plus NVFP4 fixed weights brings the modelled step to ~47 ms (~94 tok/s at tau 4.4). Reaching 125 tok/s still needs ~5.9 committed tokens per cycle at that step, or a shorter step.
- Unmodelled so far: dynamic per-domain placement, quality effects (M2), CPU kernels below int4, GPU expert formats below 3.5 bpw, attention-time beyond weight reads.
