# Building a Spark-deployable checkpoint without losing capability

Published single-Spark GLM-5.3-Flash copies benchmark badly for structural reasons. HeteroFlash avoids each of them by design, and admits a candidate only if it passes quality gates measured against the original model.

## Why existing low-memory copies lose quality, and what HeteroFlash does instead
| Failure mode in existing copies | HeteroFlash design |
|---|---|
| **Everything must fit in ~110 GB of one Spark**, forcing ~2 bpw uniform experts (KLD ~0.12, broken outputs reported) | The Spark is not the only store: hot experts live on the Spark, the rest in ~230 GB workstation RAM, served by the measured CPU kernel. The full checkpoint can stay at ~3.5-4.5 bpw. Low bits are used only where measured sensitivity allows. |
| **Expert pruning (REAP-style) to fit**, calibrated on code/agentic data; prose, knowledge and multilingual degrade most | **No expert deletion by default.** M1 shows every expert is used and domain hot sets overlap weakly (Jaccard 0.11-0.5). Cold experts move to CPU RAM instead of being deleted. Pruning is an ablation only, with general-purpose gates. |
| **Uniform bit width**, one recipe for all layers | Per-layer / per-projection / per-expert bit allocation from measured sensitivity (layer-wise KL probes; down_proj and early MoE layers get more bits). Router, DSA indexer, norms, mHC mixing and kv_b stay 16-bit. |
| **Quantized from an already-quantized or FP8 export**, calibrated on clean BF16 activations | Quantize from the official BF16 release, with **causal calibration** (each layer calibrated on activations from the already-quantized preceding layers; down_proj on quantized gate/up outputs), as Hecate reports. |
| **No recovery training** | Recovery on the Max-Q rig: router knowledge distillation first, then LoRA on sensitive projections, then quantization-aware distillation (QAD) for the fixed weights against the BF16 teacher's logits. |
| **Attention/fixed weights either left BF16 (too slow) or crushed blindly** | Fixed weights (16.6 GB at BF16, ~75 ms/step on the Spark) get FP8 or NVFP4 only where the M2 sensitivity sweep shows they are safe, and QAD where they are not. |
| **Judged by perplexity or a speed number** | Pre-registered gates vs the original: full-vocabulary KL on held-out general text; task suites (MMLU-Pro, GSM8K, HumanEval, IFEval, multilingual, 45K long-context recall, tool-call validity); generation-level blind comparison; paired non-inferiority bounds per domain. |
| **Speculative drafter not matched to the modified model** | The drafter (MTP head) is retrained/validated against the final modified model, on general-purpose data. |

## Candidate pipeline (each stage gated before the next)
1. **Sensitivity map** (Max-Q, original BF16 / Control): per-layer, per-projection, per-expert KL impact of FP8/NVFP4/int4/int3 and of top-k reduction (M2).
2. **Budget solve:** choose bits per tensor and expert residency (Spark vs CPU) to minimise predicted KL subject to Spark memory, workstation RAM and the decode-step budget from the simulator.
3. **Quantize causally** from BF16 into Spark-native formats (NVFP4/FP8 for GPU tensor cores) and CPU formats (int4 g128 / int8 for the AVX-512 kernel).
4. **Recover:** router KD, LoRA on the most sensitive projections, QAD on fixed weights. All runs on the two Max-Q GPUs, against BF16 teacher logits, on general-purpose data only.
5. **Gate:** KL plus task suites vs the original; candidates that fail a domain are rejected or given more bits.
6. **Deploy and measure** on the Spark + workstation: TTFT/TPOT/ITL/acceptance with the same harness as the Control.

## Honest tension
The latency targets push toward fewer active experts and fewer bits; the quality gates decide how far that can go. If no candidate passes both, the result is a measured Pareto frontier, reported as such.
