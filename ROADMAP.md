# HeteroFlash research roadmap — draft v0.1 (2026-11-06)

## Research title
**HeteroFlash: Joint Routing, Mixed-Precision Expert Placement, and Cross-Processor Scheduling for Memory-Constrained Mixture-of-Experts Inference**

## Hard boundary and SLAs
Final runtime uses ONE DGX Spark GB10 GPU, its integrated Arm CPU, and optional x86 CPU-only assistance via ~230GB RAM on a separate workstation. **No Max-Q VRAM/compute during deployed inference**. The two Max-Q GPUs can be used during research for teacher evaluation, compression/fine-tuning, and loss measurement. Model remains broadly general-purpose.

- Decode: sustained 120–130 tok/s on representative single-user prose; specify temperature, thinking setting, prompt length, decoding length, p50/p95, and accepted speculation length.
- Cold prefill: 45K *tokenized* uncached text to first emitted token in <=10.8 s.
- Cached: fully reused 45K prefix + 32/64/128-token additions; primary <=0.2 s defined at +32 tokens, with other suffixes reported separately. No claim of an unrestricted <200ms next turn.
- Preserve quality: preregister noninferiority against teacher on independent held-out general-purpose evaluations; KL is diagnostic, not sole safety criterion.

## Research hypotheses
H1. Separately calibrated expert usage and quality sensitivity enables nonuniform pruning + mixed precision to improve quality/latency tradeoffs over frequency-only pruning.
H2. Router reference-normalization, top-k surgery and parameter-efficient router/expert recovery reduce active weight traffic with bounded KL/noninferiority loss.
H3. Processor-specific precision and overlapped expert compute across Spark GPU, Spark Arm and *remote workstation CPU* beat GPU-only inference for some routing regimes; if not, identify break-even conditions.
H4. Combining H1–H3 with speculation/prefill kernels yields multiplicative (not blindly additive) gains on end-to-end metrics.
H5. Structural student distillation or reduced depth becomes necessary if the intact 45-layer backbone cannot satisfy target budgets.

## Tracks and stages

### M0 — Reproduce environment and baselines
**Deliverables:** hardware/links inventory, reproducible model hash and runtime versions, raw S0 Spark inference metrics, teacher T measured on dual GPUs; measured network latency, Spark shared bandwidth, remote CPU throughput; prefill/decode timeline/profile per model operation. **Gate:** all numbers reproducible across restarts; trace exact token count, prefix-hit rate. Status: probes and protocol prepared; no hardware access in this session.

### M1 — Router traces and saliency (independent of system changes)
Build hooks for 42 MoE layers; record top-k, pre-gate scores, correction bias, gate weights, token position, activation norms and teacher outputs. Diverse calibration data: writing, QA, multilingual, code, math, tools, long context. Compute entropy, concentration, variation by layer/domain; importance metric inspired by REAP and interaction-aware HOPE; measure stability under 45K prompt. **Gate:** trace schema validated, no train/test leakage, observed placement cost replaces synthetic assumptions.

### M2 — Training-free routing ablation
Vary active top-k {8,6,4,3,2} and normalizer reference k_r {k_a,8,12,16}; preserve sigmoid correction-bias selection and shared expert; test normalization scaling and entropy. Evaluate immediate teacher-forced KL and layer-output errors before expensive recovery. **Gate:** Pareto candidates improve traffic at bounded quality loss; otherwise no premature aggressive pruning.

### M3 — Nonuniform structural expert pruning
Per layer retain {288,256,216,192,160,144,115}; compare frequency-only, REAP, HOPE-style interactions, loss-aware/token-critical criteria, pruning vs merging; cap memory and protect critical layers. Record quality and byte savings separately from speed. **Gate:** candidate where recovery seems plausible; maintain exact reproducible mapping of expert IDs.

### M4 — Targeted recovery training on the two Max-Q GPUs
Freeze teacher and most student weights. First train routers and correction biases via local MoE-output reconstruction / router KD; then selectively train expert LoRA (especially down projections) and optionally sparse shared/attention adapters. Distill against reference logits on frozen held-out prefixes; use CE and per-layer reconstruction; track full-vocab KL on sampled windows and long-context task outcomes. For 320B full teacher, stage/stream or use a 4-bit teacher and quantify its own reference error. **Gate:** domain noninferiority or explicit candidate rejection. Complete distillation of a 100B+ model is NOT assumed feasible from scratch.

### M5 — Layerwise mixed precision and hardware-specific weight packing
Use calibrated quality sensitivity and measured per-processor kernel table. Quant candidates GPU {EXL3~2bpw,int4,NVFP4,FP8}, Spark Arm {INT4,INT8}, remote CPU {INT4,INT8}, selectively high-precision router/shared/DSA. Precompute packed replicas offline; calibration flows sequentially through preceding quantized layers. In Spark, reserve weights+KV+activation+OS buffers <128GB physically. **Gate:** measured quant latency and genuine quality; reject slow low-bit kernels.

### M6 — Heterogeneous execution and scheduling
Implement activation-only remote CPU RPC and local Arm worker; no Max-Q GPUs. Schedule selected bands in parallel; merge weighted outputs, track exposed stalls and p95/p99. Compare static hot/warm/cold, dynamic per-layer epoch planner, per-phase placements and replicas; migration cost in optimizer; fallback exact GPU compute on delayed network. Benchmark real NIC and CPU memory topology. **Gate:** compare S1 all-Spark versus S2 heterogeneous same checkpoint, precision and top-k; must show true net gain.

### M7 — Decode + long context optimizations
Train/retune speculative drafter to final checkpoint; test MTP/DFlash2 according to license and compatibility; graphed decode pipeline, fused router/combiner, KDA/DSA prefill optimized grouped GEMM, exact prefix-cache state restoration. **Gate:** all three measured SLAs on same hardware/config and model, not cherry-picked separate best configurations.

### M8 — If needed: structural distillation track (optional, high risk)
If Amdahl's law and measured per-layer fixed costs preclude >=120 tok/s or <=10.8s cold prefill, ablate 45 -> 41 -> 37 -> 33 decoder blocks, preserve attention/mHC state semantics carefully, local block recovery, QAT/KD and downstream tests. The main line keeps the 45-layer backbone until this gate is triggered.

### M9 — Confirmatory study & paper release
Freeze implementation/checkpoints before final test. Publish preregistered endpoint semantics, quality/latency CIs, raw traces (if license/privacy allows), reproducible config scripts, analysis code, negative results, long context accuracy, and CPU/network hardware measurements. Draft complete paper and review licenses/ethical data use before arXiv submission.

## Timeline (relative milestones; no promised dates)
| Milestone | Earliest dependency | Output |
|---|---|---|
| M0 + M1 | hardware access | baseline + router trace + calibrated tables |
| M2 + M3 | trace | routing/pruning Pareto front |
| M4 + M5 | promising candidate | recovered mixed precision checkpoints |
| M6 | benchmarked kernels | true three-tier scheduler vs controls |
| M7 | stable runtime | end-to-end target search |
| M8 conditional | measured fixed-work floor | compressed-depth student |
| M9 | confirmed checkpoint | publishable paper and artifacts |

## Stopping / pivot rules
- If remote CPU band completion makes p95 token time worse than all-Spark, disable remote execution: report negative result and keep remote RAM only for development/staging.
- If top-k/pruning recovery misses preregistered general-purpose quality margin, reduce compression or branch to depth/student distillation; no 'quality is fine' based on KL alone.
- If even idealized active-weight and measured kernel lower bounds exceed 8ms/token at target draft acceptance, run a structural student track rather than polishing offloading heuristics.
- If the three SLAs cannot simultaneously hold, publish the Pareto frontier and explain the limiting bound transparently.

## Potential paper contribution (to establish by experiments)
A topology-aware mathematical placement formulation coupling expert usage, quality sensitivity, memory footprint, processor kernel efficiency, interconnect delays, and critical-path overlap; a GLM-5.3-Flash router/compression protocol; an open-source benchmark and ablation study on one GB10 plus CPU-only auxiliary memory. Claim novelty only after a formal related-work comparison with HetRoute, Hecate, REAP, HOPE, MoE-Infinity, PreScope and SPICE.
