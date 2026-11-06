# Research execution log: 2026-11-06

## What was actually completed in this session
1. Researched Hecate's public README and configuration documentation, original GLM-5.3-Flash NVIDIA/HF documentation, NVIDIA's October 28 quantization findings, REAP/HOPE and recent heterogeneous-expert routing literature. Key non-transferability: Hecate runs GLM-5.3 753B on GB300+RTX+Grace; our target is GLM-5.3-Flash 320B on one GB10 with remote workstation CPU ONLY.
2. Drafted architecture and stage-gated roadmap with compression, recovery training, mixed precision, CPU/GPU heterogeneous execution, cache optimization and publication strategy.
3. Wrote and ran a **synthetic** expert placement prototype for 42x288 experts with shared Spark memory accounting and remote DDR capacity checks. Its output has no external validity until routing traces and measured cost tables are supplied.
4. Wrote and tested a pure-Python prototype of GLM's sigmoid-router selection semantics, including correction biases and independently configured activated/reference top-k. Compared reference top-8 vs experimental top-4/ref-8 on synthetic logits. No neural network weights were changed.
5. Implemented a paired bootstrap analysis harness and unit tests for quality/latency comparison of *future real measurements*. No real benchmark CSV is included.
6. Produced a three-page two-column LaTeX **research-design manuscript**, compiled to PDF and visually inspected. This is an initial methods-oriented draft, not a claim of empirical results or a publishable completed arXiv paper.
7. Attempted to clone `third-party/hecate`; the execution environment could not resolve `github.com`, so no local Hecate code was built or executed. Public repository pages were reviewed instead.

## Results from local synthetic/prototype run (not hardware experiments)
- 10 unit tests passed (router semantics, placement capacity/invariants, paired analysis).
- Example synthetic 42-layer placement: Spark GPU 71.999 GB expert weights, Spark Arm 7.991 GB expert weights, remote CPU 144.476 GB expert weights. These numbers result from fabricated Zipf expert popularities and assumed formats/budgets, **not** a real GLM routing trace, actual production memory plan, or measurement.
- Synthetic tier-overlap cost proxy: 38.845 ms per notional target pass with arbitrary assumed effective bandwidths and RTT. This number has **no performance predictive validity**, excludes attention, draft and real CPU/NIC measurements, and must not be compared with 125 tok/s.
- Synthetic example of router reference-normalization changes total weight assigned to 4 chosen experts from 2.5 (ref-4) to roughly 1.25 (ref-8). This only tests normalization algebra; it does not imply preserved GLM quality.

## Missing to claim scientific progress on hardware
- No direct access to user's Spark or workstation, so no real CPU/NIC bandwidth, exact GPU power, memory topology, or GLM model execution.
- No dataset collection, no trained/pruned/requantized real GLM weights, no teacher logits, no calibrated KL and no domain quality evaluation.
- No Hecate source-level code audit; its transference hypotheses are based on documented architecture.
- No actual remote CPU worker or GPU kernel; no measured improvement on the three SLAs.

## Immediate hardware execution order
1. Collect inventory via `sh src/hardware_probe.sh` on Spark and workstation; record outputs in private run logs.
2. Capture exact model revision IDs and checkpoint hashes; run single-Spark S0 and temporary two-Max-Q teacher T.
3. Measure workstation NIC actual link/RTT, CPU SIMD/NUMA and sustained effective CPU GEMV/GEMM; measure concurrent Spark GPU+Arm memory traffic. Build a cost calibration CSV.
4. Hook `Glm5NextTextTopkRouter` at layer 3..44, record exact original top-8 routing and per-layer expert quality saliency on general-purpose calibration samples.
5. Patch the real model's router with tested active/reference top-k math and execute first paired KL/accuracy sweep on the Max-Q research rig.

## Integrity statement
This project has started at the reproducible analysis/software-scaffolding phase. The hard SLAs remain experimental targets, not achieved performance claims.
