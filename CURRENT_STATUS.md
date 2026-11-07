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

# Hardware session: 2026-11-06 (M0, partial)

- Ran `src/hardware_probe.sh` on both machines and microbenchmarks in `bench/`. Results are in `results/2026-11-06-m0-calibration/MEASUREMENTS.md`.
- Measured: workstation DRAM read 246-251 GB/s; int8 VNNI expert kernel 206-217 GB/s (~0.12 ms per expert-token); Spark Arm read ~100 GB/s; Spark<->workstation TCP RTT 0.54 ms at 8 KB over 1 GbE.
- Found: both machines link at 1 Gb/s; the Spark's CX7 QSFP interfaces are not visible; no RDMA tooling is installed.
- Reran the placement simulator with the measured bandwidths: 38.8 -> 29.1 ms/pass (measured TCP RTT) and 21.1 ms/pass (assumed RDMA RTT). Expert traffic only; routing still synthetic.
- Unit tests: 10/10 pass.
- Tuned int4 CPU kernel v2: 52-61 us per expert-token (237-241 GB/s, DRAM-bound), verified exact vs scalar reference; 1.87x over v1. Activation quantization, scales and SiLU are not yet included.
- Added `src/router_trace.py`: trace schema v1 (gzip JSONL; meta + per-layer route records with ids, gate weights and optional top-N candidates), validated writer and reader, statistics (selection counts, coverage vs resident fraction, all-hot probability, speculative-window expert union), and an optional HF `*TopkRouter` forward-hook adapter (upstream transformers fd3cf01).
- `src/placement_sim.py` now accepts `--trace` (real routing replaces synthetic Zipf; sensitivity is flagged as unmeasured) and `--calibration` (`configs/calibration/2026-11-06.json`, every value labelled measured/published/assumed). The output status reads MEASURED only when every input is measured.
- Tests: 17/17 pass (7 new).
- Added `AGENTS.md`, `docs/CONVERSATION_CONTEXT.md` and `docs/CODEX_KICKOFF.md` (handoff from the planning conversation).
- Still missing for M0: Spark GPU bandwidth/GEMM, CPU kernel with activation quant/scales/SiLU, tuned/RDMA link latency, Spark Arm kernel under contention, model baselines S0/T.

## M0/M1 continued: 2026-11-06 (later)
- Spark GPU measured (`bench/spark_gpu_bench.py`): read 241 GB/s; expert-weight streaming ~222 GB/s at decode shapes; dense GEMM at [T x 4096 x 2048]: BF16 ~100, FP8 ~195, NVFP4 ~300 TFLOPS (T >= 4096). Calibration updated: Spark GPU bandwidth 222 GB/s (measured).
- M1 capture pipeline built (owner approved use of the Max-Q GPUs for teacher/trace work):
  - `src/build_calibration_prompts.py`: 364 general-purpose public prompts across 7 domains (writing/chat, knowledge QA, math, code, multilingual, instruction following, 4 x ~40K-token long-context), with a deterministic 25% holdout. No private data.
  - `src/trace_hook/sitecustomize.py`: global torch hooks that save full 288-way router logits (fp16) per MoE layer per forward. Validated on a fake 42-layer MoE (exact layer order, token counts and row totals).
  - `ops/maxq-trace/compose.yaml`: the serving recipe in eager mode, without MTP or prefix caching, max-num-seqs 1, loopback port 8001.
  - `ops/maxq-heteroflash.sh` / `ops/spark-heteroflash.sh`: one sudo script per machine (temporary CPU idle-state disable for link-latency tests, RDMA tools, ConnectX diagnosis on the Spark; trace container start + hook self-check on MaxQ), each with an `-off.sh` undo script.
  - `src/run_trace_client.py`: sequential, resumable request driver.
- Not yet run: the trace capture itself (waiting for the owner to run the MaxQ sudo script), and the RDMA/idle-state latency test.
- End-to-end TTFT harness added (`src/ttft_bench.py`, `docs/TTFT_PROTOCOL.md`): exact 45,000-token cold prompts with per-trial nonce, cached 45K + 16/32/64/128/256 suffixes, server-side cache-state verification per trial, interleaved trials, p50/p95/p99 + bootstrap CIs, SLA judged on the p95 upper CI. Tests 20/20. Not yet run on a valid target (needs the Spark S0 deployment or the original serving config; the trace server is eager/no-prefix-cache by design).

## Re-prioritised plan after external review (2026-11-07)
The verify step must be measured together with committed tokens per cycle; tok/s = committed/step (e.g. 49 ms needs ~6.1 committed tokens for 125 tok/s). Priority order:
1. **Baseline TTFT + TPOT + ITL + MTP acceptance** on the original serving config (`ops/maxq-baseline-window.sh`, one sudo run): cold 45K, cached 45K+{16,32,64,128,256}, prose and mixed-domain decode with spec-decode counters from /metrics. `src/ttft_bench.py decode` adds TPOT, per-chunk ITL p50/p95/p99, committed tokens per verify, implied verify-step time and committed tokens needed for 125 tok/s (21 tests).
2. **Top-k quality sweep** (M2, hook v2/v3 + `src/m2_router_surgery.py`).
3. **Dense/fixed-weight precision sensitivity:** hook v3 applies simulated FP8 (128x128 blocks) or NVFP4 (16-blocks, e4m3 scales) to attention, shared-expert, dense-MLP and lm_head weights in place, live from the control file; originals kept on CPU, restore verified bit-exact; module inventory dumped. Included in the M2 grid (fixed_fp8, fixed_nvfp4, k4_fixed_nvfp4).
4. **Heterogeneous placement** on full multi-domain traces (`src/decode_sim.py`, now reporting tau needed for 125 tok/s per configuration).

## 2026-11-07: framing correction
- The owner restated the design (`docs/RESEARCH_CHARTER.md`): Control = original GLM-5.3-Flash on the dual-Max-Q rig; treatments = modified versions of the original model; deployment = one Spark + CPU/RAM. Published Spark checkpoints are background only.
- Aborted an attempt to stand up a community single-Spark checkpoint/recipe as "S0". Partial downloads and the recipe clone were deleted; nothing from it was used or committed.
- All other work so far fits the framing: Control routing traces (M1) and the baseline window measure the original model on the Max-Q rig; the surgery/precision hooks act on that model; the hardware calibration and simulator target the deployment.

- M1 complete: 364 prompts, 0 errors, 6.4 GB raw logits. Curated results in `results/2026-11-07-m1-routing/` (coverage, all-8-resident probability, window unions, domain overlap, gate mass, holdout transfer, decode simulation). Raw traces stay on the workstation (not committed).

- 2026-11-07: owner approved surgery/requantization from the official BF16 weights (zai-org/GLM-5.3-Flash-BF16 rev f9fed3bbc822, downloading to the workstation). Checkpoint strategy written (`docs/CHECKPOINT_STRATEGY.md`). Control baseline window started by the owner.

- Control baseline measured (`results/2026-11-07-control-baseline/`): cold 45K TTFT p50 11.10 s / p95 11.30 s; cached 45K+32 p95 161 ms; prose TPOT p50 6.70 ms (~151 tok/s), 2.43 committed tokens/verify (47.6% draft acceptance), implied verify step ~16.3 ms; mixed 167 tok/s, 2.73 committed/verify.
- M2 launched 03:15Z. A first launch attempt never ran: the duplicate-run check matched its own command line. The workstation's 1 GbE link is saturated by the BF16 download, which causes intermittent SSH timeouts.
- G3 first result: remote expert server (activation-only TCP, multi-token batched int4 kernel, verified) measured from the Spark over 1 GbE. Network dominates verify windows (1.0-1.5 ms/layer at 4-8 tokens), so a 10G or ConnectX link is required. Remote union must be <= ~3-4 experts/layer at 6 tokens. BF16 official weights downloaded (599 GB, rev f9fed3b).
- Expert quantization error on the official BF16 weights: int4 RTN 12.1% weight / 21.6% FFN-output error; int4 clip-search 10.1% / 17.0%; int8 0.68% / 1.2%. `src/quantize_experts_int4.py` also exports kernel-layout layer files (next: real-weight loader in the expert server, GPTQ with captured expert inputs).
- Expert server now loads real exported weights (`serve-weights`); verified end to end on official BF16 experts (server vs its own int4 weights 2.5-3.1%, dominated by per-token activation quantization).
