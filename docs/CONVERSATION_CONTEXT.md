# Research conversation context for Codex

**HeteroFlash / user-to-Codex handoff**  
Context consolidated from the project-specific discussion through **2026-11-06**. This is a faithful *curated research-thread handoff*, **not a verbatim export of every assistant sentence**. User quotations below are exact or lightly excerpted from the discussion. The decisions and rejected alternatives matter more than surface wording.

## Read this first: what the user wants

The user wants a **novel, reproducible systems-and-model-compression research project**, targeting a **generally capable** GLM-5.3-Flash-derived MoE served on **ONE NVIDIA DGX Spark** with access to ~230 GB **remote CPU RAM** from a separate dual-Max-Q workstation, eventually documented in a formal experimental paper suitable to submit to arXiv. The research should **actually begin in code**, not stop at a feasibility objection or high-level brainstorm.

The user's overriding instruction, stated forcefully after an earlier misunderstanding:
> "I need to save my 2 max q VRAM for other research work. that is the whole fucking point"

Do **not** tell the user to run final inference on the two Max-Q GPUs. The earlier assistant mistakenly recommended that, despite their impressive bandwidth, and the user explicitly rejected the idea. The GPUs are **temporarily usable for training, compression and evaluation only**, as clarified later by the user.

### The three hard deployment targets

1. Single-user **prose** decode: **120-130 committed tokens/second** (not draft tokens), with the project targeting 125 tok/s as central design point.
2. **45,000-token cold prompt**: <= **10.8 s** end-to-end time to first streamed token (approximately 4,167 input tok/s if the first-token overhead is neglected). Specify whether the model process is already warm; initial protocol assumes model resident/warm but **prefix not cached**.
3. **Cached next turn** with **45K exact prefix reused**: <= **0.2 s** end-to-end first streamed token latency. Freeze suffix length in protocol; project proposes **+32 new tokens** as the primary test, with +16/+64/+128 reported too. This suffix-length convention is provisional, **not a user-supplied limit**.

For a publishable claim, define tail-latency endpoints and confidence bounds in advance. These targets are **not yet demonstrated**.

### Model quality constraint

The model must remain broadly **general-purpose**; the user explicitly said:
> "I don't want it to be hermes specific you know. I might use iT FOR LATER STUFF>"

Hermes is the immediate integration use case, and prose is the primary **speed** benchmark, but do not distill exclusively on Hermes transcripts. Include factuality, writing, reasoning, coding, multilingual, instruction following, tool calling, and 45K long-context behavior. A strict requirement is not stated for exact teacher-distribution equality; a predeclared **quality noninferiority** criterion plus KL diagnostics is the research proposal.

## Actual hardware and permission boundaries

| Hardware | Role | May be used in deployed inference? |
|---|---|---|
| 1 x DGX Spark GB10 | Sole GPU inference device, plus integrated Arm CPU | **YES** |
| Spark LPDDR5X, 128 GB physical unified | Must contain GPU/Arm allocations, model, caches, buffers and system | **YES**, one shared pool, not two |
| Separate workstation, about 230 GB **available** DDR5 ECC RDIMM | Remote CPU expert execution and/or staging, accessed over real network | **YES**, CPU only |
| Workstation's 2 x RTX PRO 6000 Blackwell Max-Q | Teacher/experimental training, LoRA, calibration, evaluation during development | **NO for deployed serving**, **YES temporarily during offline R&D** |

Workstation CPU model, NUMA, actual installed memory, NIC/link topology, bandwidth, and network RTT are **not established**. Do not assume 200Gb/s end-to-end or eight memory channels. The workstation RAM is **not** a local extension of Spark VRAM. Spark's Arm CPU and GPU contend for the **same physical 128 GB RAM and memory bandwidth**.

## How the project evolved: chronological decisions and corrections

### 1. Initial question: local Spark + remote CPU RAM and heterogeneous quantization

The user originally asked to run GLM-5.3-Flash with Spark-resident weights, less-used experts offloaded into the separate workstation's system RAM, and processor-specific quantization, coordinated so the system maintains 120-130 tok/s, 10.8 s cold 45K prompt and 0.2 s cached follow-up. This was framed as a **deep research topic**.

Early discussion examined GLM's MoE architecture and the difference between decode (small-batch, effective active-weight bandwidth and speculation) and prefill (grouped GEMM and activation precision). Treat numeric benchmarks quoted in earlier assistant answers as **reported external results or order-of-magnitude estimates, not verified local results**.

GLM-5.3-Flash was described as approximately 320B total, 18B activated, 45 decoder layers, 42 MoE layers, 288 routed experts/layer, top-8 routing. Pin the actual model revision and check these values before modifying model code.

### 2. Critical assistant mistake: using dual-Max-Q GPUs for inference

After learning about two Max-Q GPUs and remote memory, the assistant suggested serving directly from the 2 x 96 GB Max-Q VRAM (faster published prose throughput). **The user explicitly rejected this** and insisted the Max-Q VRAM remain available for their separate research. They provided the exact constraints: one Spark, ~230 GB CPU RAM on the Max-Q workstation, and the three performance targets. Do not repeat that suggestion.

The assistant then described existing single-Spark results (~30 prose tok/s in one published configuration and lower-than-target prefill), network bottlenecks and expert-cache diversity. The user rejected treating published baselines as immutable impossibility proofs:
> "you know you could also adjust GLM 5.3 flash weights or sth. I am not satisfied with this research result. I need these hard constraint to be met. this is the research questiON WE ARE EXPLORing"

**Decision:** The model **and** serving engine are modifiable. Hard targets are research endpoints; be ambitious in exploring methods without claiming that success is guaranteed.

### 3. Expand the design space to weight surgery and distillation

The assistant proposed changing active expert budget (top-k 8 -> 6/4/2), nonuniform pruning and/or expert merging, router retraining/KD, selective expert LoRA, layer-sensitive mixed precision, low-bit Blackwell kernels, retraining an aligned speculative drafter, and optional shorter-depth student distillation if the fixed backbone cannot fit latency budget. The user found the combination appealing:
> "top-k reduction + nonuniform expert pruning + router distillation + mixed-precision re-quantization. --> this sounds fun can we do that."

They were concerned that a fully distilled smaller MoE might damage quality, and requested KL monitoring:
> "D. Distill GLM into a smaller, purpose-built MoE --> could be risky need to track with KLD actually i am not sure if that is ok"

**Decision:** Full student distillation is a **high-risk conditional track**, not the default. First perform training-free router ablations, calibrated pruning, router KD, expert-local reconstruction, mixed-precision tuning. Use KL **plus** downstream quality; KL alone can hide task degradation.

### 4. Training GPUs are permitted for R&D

The user clarified:
> "I guess we could train it using 2 max q rig and offload the distilled model to spark later."

**Decision:** The two Max-Q cards can run teacher inference, train routers/adapters or compressed candidates, and evaluate candidates **offline**. The final deployed inference must still be Spark GPU + allowed CPUs **only**. Do not assert full 320B end-to-end training is cheap or fits; use frozen teacher, layerwise reconstruction and parameter-efficient training until measured otherwise.

### 5. Proper A/B experimentation and significance

The user:
> "AB testing is interesting, we should run experimentatION With 2maxq's GLM flash setup and then run the optimzied versiON ON SPARk, run x thousand times and compare if the metrics are statistically significantly better (my old Data science brain)."

**Decision:** A/B science needs *three roles*: (T) high-quality GLM teacher on temporary 2-Max-Q research rig for quality; (S0) unchanged GLM low-bit baseline **on Spark** for performance; (S1) compressed GPU-only candidate on **the same Spark**; and (S2/S3) same checkpoint heterogeneous variants to isolate placement/scheduling gains. **Cross-hardware runtime comparisons are confounded**. Use paired held-out prompts, prompt-clustered bootstrap, one-sided noninferiority confidence bounds, independent domain slices, multiple restarts, thermal controls, p95/p99, exact cache token accounting. Thousands of repetitions of one prompt are *not* thousands of independent examples.

### 6. Heterogeneous execution was re-emphasized

The latest project brief from the user:
> "on top of the training, don't forget the heterogeneous moE inference system which spread experit by usage across processorys, and different quant types each fitting the best bandwidth / compute load for that subprocessor, coordinating that so nothing installs is interesting, so you shuffle around experts and quants so that iT Fits the runtime. some takeaway from https://github.com/third-party/hecate. packAGE EVERYTHing in a roadmap and start it. end result is a formal research paper that I could publish to arxiv idk."

**Decision:** This is **not** a compression-only project. Optimize **jointly**: which experts to retain/merge, which to activate, which quantized representation they use, and on which processor they execute at runtime. Include time-varying access traces, placement epochs, migration costs, concurrent execution, latency hiding, and distinct prefill/decode plans.

The Hecate-inspired pattern: let different processors execute **different selected expert subsets of the same layer concurrently**, then combine their weighted outputs. For remote CPU execution, send **activations**, not whole weights, when possible. Prepack quantized expert variants offline for the target processor, rather than requantize on every token. The central deadline is the **slowest concurrent processor band's completion plus communication/synchronization**, not aggregate nominal FLOPs.

Caution: Hecate's own system targets **GLM-5.3 (not GLM-5.3-Flash)** on **different high-end GB300/Grace + RTX class hardware**. The exact Hecate repository **was not cloned** in the earlier artifact-creation environment due to GitHub DNS failure. Its principles came from public README/docs, **not a full audited source build**. Read `docs/HECATE_TRANSFER.md` and pin an upstream commit before claiming direct code transfer.

### 7. User now wants Codex to inherit this conversation

The immediate ask is to **package the conversation with the project** so Codex can continue without repeating mistakes or losing decisions. This document, top-level `AGENTS.md`, and `docs/CODEX_KICKOFF.md` satisfy the handoff. They include distilled context, not all prose from the chat transcript.

## Technical ideas in scope

### Router algebra / top-k surgery

The router reportedly selects experts using sigmoid scores plus learned correction bias, but uses uncorrected sigmoid scores for selected gates, normalizes and scales by a routed factor. For active top-k `k_a` and reference-normalization top-k `k_r`, experiment with decoupling activated indices from normalization mass. **Verify exact source and checkpoint version**. With `k_a = k_r = original top-8`, the patched router should have exact-equivalence tests before changing K or pruning. Layerwise/top-k budgets can be heterogeneous.

### Nonuniform pruning and expert sensitivity

Do not assume a small fraction of experts sees all traffic; under a purely uniform null model, many experts are touched within long prefixes. **Measure** expert selection distribution by layer, context and workload on real data. Compare frequency, REAP/saliency, interaction-aware methods, random controls, and loss-critical token weighting. Distinguish fewer **stored** experts from fewer **activated** experts. Rebuild compact indices, not just zero weights.

### Recovery training and KL

Compute teacher-forced forward KL on identical token prefixes and logits processing, plus full-vocab or validated top-logit-tail approximations. Track layerwise reconstruction errors, NLL, answer correctness and generation drift. Try router-only KD first, selected down-projection/expert LoRA second, optional QAT, and full structural student distillation only on a justified evidence gate. A four-bit teacher can differ from the FP8 checkpoint; check reference bias.

### Per-processor mixed precision

Spark GPU: native FP4/INT4 (plus low-bit EXL3 as a baseline), possible FP8 in sensitive operations. Spark Arm and remote CPU: kernels and packed INT4/INT8 chosen by actual SIMD/NUMA, memory bandwidth and matrix shapes. Quantization error may be more acute in expert down projections; calibrate per layer and per expert. A smaller checkpoint with higher-quality precision could outperform an ultra-low-bit full model **if** actual kernels and quality support it. A smaller bit count does not guarantee faster execution.

### Heterogeneous scheduling and movement

Track true per-expert placement, resident bytes across the **shared** Spark pool, quant format availability, working-set locality, CPU+GPU bandwidth contention, network RTT/jitter and per-layer barriers. Options: static hot/warm/cold, epoch-level dynamic expert relocation, prefetch, CPU execution vs weight transfer, per-phase layouts, replicated hot experts, timeout/fallback. Migration or re-quantization must earn back its cost. Remote CPU could hurt: include negative controls and publish break-even regimes.

### Speculative decoding and prefix cache

Optimize committed tokens per complete target-verification cycle, not proposed draft tokens. Any substantial target-model change likely requires drafter revalidation/retraining. Cache the exact sparse-attention and recurrent states for 45K prefixes; maintain session affinity and verify state against full recomputation. Cold TTFT, cached TTFT, and sustained decode must be met **for the same checkpoint/configuration**, not a cherry-picked union of different runs.

## Experiment design and success/failure language

- Frozen general-purpose calibration/development/holdout splits, independent prompts, domain-stratified metrics and preregistered quality margins.
- CPU/network and CUDA/Triton kernel microbenchmarks on **actual hardware**, then end-to-end execution traces. Store all run metadata, seeds, hashes, clock/power state and cache hit counts.
- Variant ladder: T teacher; S0 Spark original; S1 Spark compressed GPU-only; S2 Spark+Arm; S3 Spark+remote CPU; S4 all-three; S5 fixed vs dynamic placement; S6 per-tier mixed precision; S7 final speculation/cache optimizations; S8 optional smaller-depth student.
- Separate component ablations from end-to-end gains; quantify p50/p95/p99 and confidence intervals. Student quality tested against teacher (possibly hierarchical multiple testing); speed tested S1/S2... against S0 on same Spark.
- Hard thresholds should not be silently relaxed. If they cannot be met, report bounds, full Pareto frontier and negative results faithfully. **No fake measurements or guaranteed outcomes.**

## Work completed already (as of 2026-11-06)

From `CURRENT_STATUS.md`:

- Research roadmap, architecture/system-design document, Hecate-transfer analysis and formal-ish experiment protocol drafted.
- `src/placement_sim.py`: deterministic **synthetic** 42-layer expert placement/bit-allocation model with capacity checks. The example placement and ~38.845ms illustrative cost proxy use fabricated popularity and timing constants; **never** treat them as observed throughput.
- `src/router_surgery.py`: tested pure Python GLM-like sigmoid/correction-bias top-k and reference-normalization algebra. **Not** integrated into the real model and no weights modified.
- `src/paired_analysis.py`: basic paired bootstrap; needs richer metadata and tests for real data.
- `src/hardware_probe.sh`: safe local inventory script to run on both machines.
- Ten unit tests reportedly passed in the initial scaffold. **Rerun tests to confirm in the new environment.**
- `paper/main.tex`, `paper/main.pdf`: three-page methodological research draft. **No actual performance, model quality, or training results.**
- No access to user's actual Spark or workstation in this conversation; no real GLM run, actual router trace, KLD measurement, trained compressed checkpoint, NIC calibration or three-SLA result.

## Prioritized next actions for Codex

1. Inspect project and rerun unit tests; report environment and any missing runtime dependencies without fabricating GPU availability.
2. Build real model router tracing and checkpoint-version pinning. Add a loader that reads actual traces, replacing synthetic popularity in the placement simulator.
3. Make a hardware calibration schema and microbenchmark driver for Spark GPU+Arm and workstation CPU/NIC; calculate per-expert per-tier q-format cost tables from measurements.
4. Implement an opt-in GLM top-k/reference-normalization patch with original-top-8 parity checks, followed by paired KL collection and a held-out benchmark harness.
5. Research/replicate Hecate source mechanisms on a pinned commit **without assuming its large-host system design is directly portable**. Prototype concurrent activation-only remote CPU expert execution and exact partial-sum correctness, after confirming CPU/NIC feasibility.
6. Expand optimization solver to joint expert retention, quant-type assignment and physical placement/migration with measured costs and memory constraints.
7. Maintain `CURRENT_STATUS.md` and the paper's distinction between hypotheses, methods and measured results after each experiment.

## Artifact map

- `AGENTS.md` — Codex's high-priority repository constraints and coding guidance.
- `docs/CODEX_KICKOFF.md` — copy/paste kickoff prompt for Codex.
- `docs/CONVERSATION_CONTEXT.md` — this conversation and decision handoff.
- `README.md` — project overview.
- `ROADMAP.md` — stage-gated M0–M9 execution program.
- `CURRENT_STATUS.md` — honest work log.
- `docs/HECATE_TRANSFER.md` — upstream transfer/limitations.
- `docs/SYSTEM_DESIGN.md` — cross-processor MoE scheduler and quantization tiers.
- `docs/EXPERIMENT_PROTOCOL.md` — experimental validity, quality and speed controls.
- `src/` — prototype, benchmark/analysis helper, hardware probe.
- `tests/` — scaffold tests.
- `paper/` — draft manuscript source and references.

## Useful starting sources (verify versions and licenses)

- Hecate: https://github.com/third-party/hecate
- Official model: https://huggingface.co/zai-org/GLM-5.3-Flash
- HF reference model source: https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm5_next/modeling_glm5_next.py
- NVIDIA model architecture: https://docs.nvidia.com/nemo/automodel/latest/model-coverage/vision-language-models/thudm/GLM-5.3-Flash
- NVIDIA mixed-precision/QAD research: https://nvidia.github.io/Model-Optimizer/announcements/20261028-glm53-flash-iq2.html
- Cerebras REAP: https://github.com/CerebrasResearch/reap
- Spark hardware: https://docs.nvidia.com/dgx/dgx-spark/hardware.html

## Research ethos

The user wants a *real attempt* to close a difficult hardware-constrained model-serving gap, with systems creativity and statistical rigor. They do **not** want another generic explanation that today's single-Spark baseline is slow, and they do **not** want the two Max-Q GPUs consumed by deployed inference. Preserve rigor: an ambitious research question is not an achieved result, and a negative measured finding is still useful science.
