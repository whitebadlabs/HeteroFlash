# Hecate transfer study — checked 2026-11-06

Repository: https://github.com/third-party/hecate

**Do not conflate Hecate with GLM-5.3-Flash.** Hecate is a GLM-5.3 753B system for DGX Station GB300, an RTX PRO 6000 Blackwell and a 72-core Grace CPU. The target here is GLM-5.3-Flash 320B/18B-active on one GB10 Spark with remote CPU-only workstation assistance. Claimed Hecate results are repository authors' own, not measured on our machine.

## Evidence that actually informs this project
- Places distinct expert bands on different processors **within every MoE layer**, not merely swapping an expert bank at request boundaries.
- Uses 144 popular experts/layer as GPU int4, 85 medium-frequency experts/layer as RTX trellis ~3-bit, and 27 cold experts/layer on Grace Arm int8 (these are Hecate's GLM-5.3 placement counts, not GLM-Flash design choices).
- Composes partial expert outputs and overlaps the bands, using device-visible mailboxes, device-side signaling and one CUDA graph for the secondary GPU; removes host synchronization where possible.
- Runs cold experts *on the CPU* rather than sending their entire weights to the primary GPU, and tunes CPU kernel representation to NEON/int8 dot products.
- Separate prefill and decode algorithms; prefill grouped MoE GEMMs avoid full dequantize-once-per-chunk overhead. Prefill can have different preferred tiering than decode.
- Calibrates quantization **causally**: use activations from preceding already-quantized layers, calibrate expert down projection on quantized gate/up output, not pristine BF16 activations.
- Reports KL on a specified number of token positions and explicitly warns that KL from different evaluation panels is not comparable.

## Hardware topology difference: critical
Hecate's Grace CPU accesses GB300 memory via coherent NVLink-C2C, with high bandwidth and low overhead; the GB300 has 252 GB HBM3e and there is an additional 96 GB RTX GPU. A **single DGX Spark GB10** has just 128 GB of shared CPU/GPU LPDDR5X and no second GPU allowed for inference. Our remote CPU over Ethernet/Infiniband is neither an extra coherent memory tier nor a remotely CUDA-graphable GPU. Its actual bandwidth/RTT/CPU SIMD capability is unknown until probed. It cannot simply reproduce Hecate's +22% GPU tier, or any numerical benchmark.

## Adaptation plan
**Tier A** — Spark GPU: fast routed experts in a native or tested compact GPU format (quantized int4/NVFP4/W4A16; dequant fused with GEMV or grouped GEMM). Dense/shared experts need independent calibration.

**Tier B** — Spark local Arm CPU: small group of sporadic experts on 20 Arm cores, conditional only when parallel CPU compute beats shared-memory contention. CPU and GPU share **one** LPDDR5X bandwidth budget, so do NOT double-count 273 GB/s.

**Tier C** — Workstation CPU: only selected experts; keep their weights in remote RAM and transmit small activations + routing info and return partial sums. Use AVX2/AVX512/VNNI/AMX where available **after detecting actual CPU**. Remote NIC & CPU memory bandwidth need measured calibration. Batch multiple rows in prefill, or skip remote CPU entirely for large prefill if synchronization dominates.

**Tier D** — Remote RAM **as storage/prefetch**, as a separate experimental variant. Do not transfer 10+ MiB experts for each router miss on the critical path if remote compute can be cheaper. Compare cost against exact execution on Spark.

## Placement / migration details
1. Profile per-expert `Pr(select | layer, workload, prefill/decode)` and quality sensitivity, not just total selection count. Hotness can drift by request and language.
2. Offline prepack candidate quantized representations per processor and per expert. Real-time re-quantizing every move is likely counterproductive. At run time select already-packed replicas; background recalibration and repacking only when workload distribution changes.
3. Choose a **frozen placement epoch** to avoid churn while verifying each step, unless measured benefit exceeds migration cost. Admission rule includes expected saved milliseconds over a horizon minus migration/repacking cost, not just LRU.
4. Route then issue independent processor tasks; compute available bands in parallel; combine exactly with original gating weights. Only proceed to next layer when all required partials have arrived. Measure exposed stall = max(band finish times) - available local work, per layer.
5. Permit per-phase placements: decode top-k and prefill grouped GEMM may have different best processor formats and schedules. Count model replicas and activation buffers against total RAM.
6. First preserve exact original top-8 outputs for a correctness baseline; then test router/top-k modifications with separately reported quality loss.

## Suggested experiments adapted from Hecate
H0: GPU-only (same quant model). H1: Spark GPU+Arm split. H2: GPU+remote CPU split. H3: all three concurrent. H4: dynamic epoch-wise placement. H5: per-tier mixed quant (fixed placement). H6: H4+H5. H7: prefill/decode distinct placements. H8: routing/pruning/KD student + H6/H7.

## Beware conflating speedups
Hecate reports ~184 tok/s greedy on a significantly larger, *different* machine, with separate large memory/compute devices. Its measured placement gains are not transferable by multiplying percentages. Its long-context cached-turn TTFT is not the same metric as the requested 45K+suffix <=200ms.

## Concrete observations from the upstream repository
- Readme describes tier placement 144/85/27 and fused router, mailbox signaling, prefill `moe3p/moe4p`, GPU-only long-prefill, and causally calibrated quantization.
- Upstream config defaults include 8192-token prefill chunks and 3 MTP draft tokens. These are starting hypotheses only.
- Code clone could not be performed in the current execution environment because GitHub DNS is unavailable. Analysis is based on publicly accessible repository pages, not a local build or source audit. Exact commit must be pinned when downloaded on research hardware.
