# HeteroFlash research plan, 2026-11-16: closing the gap to the targets

**Labels.** [M] measured on our hardware. [D] derived arithmetically from measurements. [V] vendor-, paper- or third-party-claimed, with a URL. [A] assumption. [H] hypothesis. Every throughput or latency that is not [M] is **modeled**, not achieved.

**Targets (unchanged).** Decode 80 tok/s (stretch 120-130); cold 45K TTFT ≤ 10.8 s; cached 45K+32 ≤ 0.2 s; quality no worse than the dual-Max-Q control (balanced eval: dNLL +0.0148, KLD 0.042).

**Model and scripts for this plan.** `src/cycle_bytes.py` (bytes per decode cycle, from checkpoint headers), `src/gap_model_2026_11_16.py` (every rung below; inputs labelled), `ops/r16/cycle_anatomy.sh` (today's measurement). Outputs: `results/2026-11-16-cycle/`.

## A. Where the time goes

### A1. The measured path so far

The README headline (6.46 tok/s, 838 s cold, 3.10 s cached) is the first llama.cpp RPC deployment over 1 GbE (E2E-v0). The project has moved well past it. The gaps should be read from the newer measurements.

| Step | What changed | Decode | Cold 45K | Cached 45K+32 | dNLL vs Q8 proxy |
|---|---|---|---|---|---|
| E2E-v0 | llama.cpp RPC, 1 GbE, Q4_K experts on layers 24-44 remote | 6.46 [M] | 838 s [M] | 3.10 s [M] | not measured |
| E2E-v1 | direct 10 GbE link, GPU placement fixed | 10.5; 17.0 with MTP [M] | 223 s [M] | 0.68 s [M] | not measured |
| Config A | own runtime (Spark-native vLLM), GPU-only, FP8 fixed weights, 2-bit experts, MTP-3 | 25.0-25.7 [M] | 37.9-39.0 s [M] | 0.35-0.42 s warm [M] | +0.182 (fails) [M] |
| Config A, model runner V2 | same, V2 runner (delegated run, 2026-11-10, single run) | 34.79 [M] | not run | not run | +0.182 (fails) |
| Config B | all routed experts on the workstation CPU in Q4_K + imatrix, mailbox handoff, MTP-3 | 15.3-19.5 [M] | 994 s [M] | 1.17 s [M] | **+0.0069** (passes) [M] |
| Config C | prompt read with GPU 2-bit experts, answer with CPU Q4_K | 15.3 [M] | 42.2 s [M] | 0.54 s [M] | +0.0209 (fails) [M] |

So the best configuration at control quality is config B: decode 4-5x short of 80, cold 92x short, cached 6x short.

### A2. Why E2E-v0 was 6.46 tok/s and 838 s (from existing profiles)

- **Decode.** About 13.9 GB was read per token, at Q8_0 attention and Q4_K experts [M, GGUF tensor table]. Half the expert layers were remote over 1 GbE with ~0.5 ms RTT per layer [M]. The CPU work and the network round trips ran strictly serially with the GPU. The v0 → v1 step (10 GbE, fixed placement) alone gave 1.6x.
- **Cold prefill.** The cold prefill is linear in tokens at ~200 tok/s [M]. A 4K nsys profile (v1) shows the GPU idle 55% of the time waiting on remote experts. The CPU ran experts at ~1.5 TMAC/s, about 10x below VNNI peak. On the GPU, the Q4_K MoE kernel reached ~7 TFLOPS [M]. Attention, KDA and the DSA indexer were under 1.4 s of 17.9 s GPU time [M]. Nothing in v0/v1 was long-context-bound.
- **Network share now.** In config B the mailbox exchange is 0.06-0.11 ms per layer [M], so about 2.6-4.6 ms of a 118-150 ms cycle (**about 2-4% [D]**).
  - This answers the open follow-up "share of decode time waiting on the network". It is small at config B's structure.
  - The 10 GbE link matters again only under the dual-memory split (A3, Bet 1), which needs about twice as many exchanges per cycle.
  - It also matters for cold prefill, where it carries weights (Bet 3).

### A3. Bytes per decode cycle of the actual FP8 + MTP-3 build [M, headers]

`results/2026-11-16-cycle/bytes_full_iq2xxs_fp8_k3.json`; read model stated in the script.

| Component | GB read per MTP-3 cycle |
|---|---|
| KDA attention (34 layers, FP8 projections) | 4.95 |
| MLA attention incl. BF16 kv_b (11 layers) | 1.48 |
| Shared experts | 1.06 |
| lm_head (FP8), verify | 0.65 |
| Dense MLP (layers 0-2) | 0.47 |
| DSA indexer + router + mHC + norms | 0.33 |
| **Verify fixed path** | **8.94** |
| Routed experts, U = 22.7 per layer, IQ2_XXS 6.49 MB | 6.19 (Q4_K 14.16 MB: 13.50) |
| 3 MTP draft passes (0.244 MTP non-expert + 0.052 experts + **0.654 lm_head** each) | 2.85 |
| **Cycle total (GPU-only, 2-bit experts)** | **17.97** |

**Reconciliation.**
- V1 runner: 25.67 tok/s at the measured prose acceptance of 2.217 committed tokens per cycle is 86.4 ms per cycle, so the GPU streams 17.97 GB at **~208 GB/s effective [D]**. That is the measured GB10 streaming rate (210-243 GB/s [M]), so the V1 cycle is bandwidth-bound, as the 11-09 report said.
- V2 runner: 34.79 tok/s cannot be bandwidth-bound at the same acceptance; it would need 282 GB/s, above the 273 GB/s spec. Either acceptance or union differ under V2, or the bytes model overcounts.
- Today's measurement (section E) resolves which, because the ceiling every bet is sized against depends on it.

**Notable.** The three draft passes re-read the full lm_head: 1.96 GB per cycle, **11% of all bytes**. A draft-only head is lossless to output quality.

### A4. The structural bound, stated once

Single-stream decode at batch 1 is bound by bytes per committed token over the memory bandwidth that reads them in parallel.

- In config B the fixed path (GPU) and the experts (CPU) are serial within each layer. So the workstation's 250 GB/s helps only the expert phase, and the Spark's 208 GB/s alone carries 11.8 GB of fixed and draft bytes per cycle.
- With both memories streaming every phase, aggregate bandwidth is ~378 GB/s [D].
- **80 tok/s then needs ≤ 4.7 GB per committed token; today's quality config reads ~11.4 [D].**
- The expert term barely amortizes with speculation. U(4)/A = 22.7/2.2 and U(8)/A = 38/3.5 both give ~10.5 experts per layer per committed token [D]. So above roughly 40 tok/s the gap is mainly an expert-bytes problem, not a fixed-path problem.

## B. Literature and technique scan (delta to docs/research, 2026-11-16)

Prior notes (11-09) cover Fiddler, KTransformers, HybriMoE, DALI, DAOP, Pre-gated, HOBBIT, ProMoE, Fate, MoE-Infinity, the OSDI'26 stream-loading prefill, Hecate, MC-MoE/MxMoE, QAD and DFlash2/DSpine. New or newly quantified:

| Technique | Claim [label, source] | Effect on our targets (modeled with our numbers) | Cost / risk |
|---|---|---|---|
| **Heterogeneous tensor parallelism** (split non-expert weights across unequal devices): HeteGen, HexGen | HeteGen picks the CPU:GPU split so T_cpu = T_gpu + T_comm, at batch 1 over PCIe [V, paper summary; speedup not retrieved]. HexGen: asymmetric TP/PP across heterogeneous GPUs, 2.3x lower latency on Llama-2-70B [V, arXiv 2311.11514]. No batch-1 GPU+CPU split across a network link was found. | Decode: the core of Bet 1. Takes config-B structure from ~16 to ~27-30 tok/s at unchanged quality [modeled, R2-R3]. | Runtime work (head-split attention + KDA state on the CPU); exact math, no quality risk. |
| **FR-Spec** frequency-ranked draft vocabulary | Draft LM-head compute -75%; verification keeps the full vocabulary (lossless); 1.12x over EAGLE-2 [V, arXiv 2502.14856] | Decode: cuts the 1.96 GB/cycle of draft lm_head reads to ~0.5 GB; +8% on the R3 rung [modeled R4] | Small patch to the MTP drafter; acceptance loss to measure |
| **KTransformers expert deferral** | ≤ 0.5% average accuracy drop, up to 1.45x throughput, on DS-V3, DS-V2.5 and Qwen2-57B [V, SOSP'25 paper; LMSYS blog] | Decode: an alternative to Bet 1 that hides CPU expert time behind the next layer's attention. Bet 1 achieves the overlap exactly, so deferral is the fallback. | Changes the model; never measured on GLM-style hybrids |
| **NVFP4 grouped MoE GEMM on GB10** | Dense NVFP4 356 TFLOPS; MoE grouped GEMM 120-154 TFLOPS [V, NVIDIA developer forum, community CUTLASS] | Cold prefill: the kernel class that brings the compute floor from 18-20 s to 11-13 s [D from V, prefill note §6] | Whether the sm_121 FP4 MoE path is native is disputed in the forum; a driver regression was reported |
| **Spark ConnectX-7 to a workstation** | 2 ports, each 2x100G MACs on separate PCIe Gen5 x4 links. Single rail ~109 Gbit/s ib_write_bw [V, forum]. Both rails 185-190 Gbit/s RoCE perftest [V, ServeTheHome 2025-12-01]. QSFP112 cages accept QSFP56/QSFP28 [V, NVIDIA docs]. | Cold prefill: streams the 52 GB cold-expert set per 45K prefill in ~4.8 s (1x100G) or ~2.4 s (2x100G), against 45 s on 10 GbE [modeled]. Decode: sync cost for Bet 1. | Used dual-port CX-6 Dx 100GbE ~$430-900, new ~$1,390 [V, listings, undated]. DAC price not found. **No report of a Spark linked to a non-Spark host.** Forum reports of 13 Gbit/s caps from slot-power throttling and a kernel regression. |
| **Threadripper PRO 9000WX bandwidth** | 8x DDR5-6400, 410 GB/s peak [V, AMD]. No measured STREAM found for the 9975WX/9985WX/9995WX. | Our 9975WX measures ~250 GB/s [M], which suggests a CCD/fabric limit (4 CCDs). An 8-CCD part could add CPU-side bandwidth. +7.7 tok/s at the R6 rung if it reached 250 GB/s on verify windows [modeled R7, A]. | Expensive and unmeasured. Measure first on a borrowed or cloud 64-core part before any purchase. |
| **MegaFlux** (arXiv 2610.00671) | Hot-expert replication inside MoE megakernels, 8xB200: 1.45x forward geomean; DeepSeek-V4-Pro prefill 1.13-1.26x end to end [V] | Low relevance: multi-GPU, large batch, skew-driven | none |
| **HiSparse** (arXiv 2608.07009) | Exact hierarchical KV for top-k sparse attention; host-memory KV with a GPU cache inside the CUDA graph; up to 4.7x peak throughput, per-token latency "comparable" [V] | None at batch 1 on unified memory, unless the KV exceeds the Spark's memory | none |
| **DFlash / DFlash2 for MLA targets** | DFlash repo lists GLM-5.1 / DeepSeek-V4 as "coming soon" [V, third-party summary]. No MLA support found. | Decode: A 2.2 → ~3.5 is worth little at control quality because the verify union grows (R5 → R6: +2%) [modeled] | Days of porting (MLA page layout); deprioritized |
| **MoE-aware drafting that limits verify-window expert union** | Not found in 2025-2026 literature | Decode: the largest remaining lever (A4). This is the gap where a new result would sit. | New research |

Not found (asked for, absent): measured Spark ib_write_lat, FP8/NVFP4-activation quality on long-context prefill, EAGLE-3 acceptance on a GLM/DeepSeek-class MoE.

## C. Hardware levers

- **QSFP link, Spark CX-7 to the workstation.**
  - **Needs:** a ConnectX-6 Dx (100GbE, QSFP56) or ConnectX-7 card in a free PCIe x16 slot of the workstation, plus one QSFP56/QSFP28 passive DAC (≤ 1-2 m). Two cables and a dual-port card would use both Spark rails.
  - **Cost [V]:** used dual-port CX-6 Dx ~$430-900; new ~$1,390; DAC price not found.
  - **Bandwidth:** ~11 GB/s single rail, ~22 GB/s both rails [V]; 10 GbE today is 1.15 GB/s [M].
  - **RDMA latency:** not measured on a Spark [A: ~2-15 us].
  - **What it buys:**
    - **Cold prefill:** makes layer-wise streaming of control-precision cold experts fit under the GPU compute time (Bet 3). Without it the stream alone is 45 s [modeled].
    - **Decode:** under Bet 1 the 90 exchanges per cycle drop from ~9 ms (10 GbE busy-poll, [M] per exchange) to ~1.4 ms [modeled]. Worth +3 tok/s at R2 → R3.
  - **Risks:**
    - A Spark-to-non-Spark RoCE link has no public precedent.
    - The Spark boot log already shows "insufficient power on the PCIe slot (27W)" with no cable. That message is linked in forum reports to 13 Gbit/s caps.
    - Fallback: the card is useful for any later second Spark.
- **RDMA on the existing 10 GbE.** The Realtek 10GBASE-T NIC has no RDMA, so busy-poll TCP (50 us RTT [M]) is the floor on this link.
- **Can prefill run entirely on the Spark GPU with experts streamed?**
  - **Yes in principle.** 200 hot experts per layer stay resident in control format (~95 GB [D, memory budget not validated by a boot]). The other 88 per layer (52 GB as Q4_K per 45K prefill) are streamed per layer into a double buffer and dequantized to the GPU kernel's format.
  - **Precision:** prefill reads at decode precision, which removes the measured +0.014 nat penalty of 2-bit prefill [M].
  - **Time:** cold TTFT is then max(compute, stream) ≈ 11-13 s with best-published kernels and a 100G link, or 18-20 s with today's kernels [modeled].
  - **Still missing:** the EXL3 trellis prefill path ran at only 28 TFLOPS [M]. Prefill therefore needs a dequantize-to-FP8 (or NVFP4) staging pass per layer feeding a grouped GEMM. The staging throughput is unmeasured.
- **Workstation CPU memory bandwidth.**
  - The 9975WX has 4 CCDs and measures ~250 GB/s [M] against a 410 GB/s spec.
  - An 8-CCD part is the only hardware lever that raises the decode ceiling inside the charter (R7, +18% [modeled]).
  - **Do not buy on spec:** first measure STREAM on a 64-core Zen 5 Threadripper PRO (cloud or borrowed).

## D. Ranked bets

Ranking = expected movement of a target × probability, per unit of cost. All throughput numbers are modeled from the rungs in `results/2026-11-16-cycle/gap_model.json` unless marked [M].

### Bet 1: Dual-memory decode, an exact split of every layer across Spark GPU and workstation CPU

- **Hypothesis.** Splitting each layer exactly lets both memories stream in parallel during every phase, not only the expert phase:
  - attention/KDA by heads;
  - shared and dense MLP by intermediate columns;
  - routed experts by expert;
  - lm_head by vocabulary shard.
  - Two partial-sum exchanges per layer (~32 KB at 4 tokens).
- **Expected:**
  - config B 15.3-19.5 [M] → ~27 tok/s on 10 GbE, ~30 on RDMA (MTP-3, prose acceptance) [modeled R2/R3];
  - cached turn: the 32-token step splits the same way (supports the 0.2 s goal).
  - Quality unchanged (exact arithmetic up to summation order).
  - Changes the **fixed-path term** (from /208 to /378 GB/s).
- **Experiments:**
  1. **Today, Spark-only:** cycle anatomy of the fastest GPU-only runtime (section E). It sizes the GPU half and settles whether the V2 runner moved the ceiling.
  2. **Workstation CPU, once free:** FP8/int8 GEMV at attention, KDA, shared and lm_head shapes, T = 1-4. Target ≥ 170 GB/s.
  3. **Workstation CPU, once free:** a two-party exchange loop of 90 x 32 KB per cycle over 10 GbE busy-poll. Expected ≤ 10 ms per cycle.
  4. Head-split KDA/MLA on one layer: compare against the GPU layer output (rel. error ≤ 1e-3).
  5. Then the full-model runtime.
- **Success:** a measured ≥ 25 tok/s on prose at config-B quality (+0.0069 or better).
- **Budget:** about 2 days of runtime work; 0 Max-Q GPU-h; workstation CPU only.
- **Approvals:** none beyond existing ones (workstation CPU/RAM use, sudo-less).

### Bet 2: Lossless bytes diet on the draft and fixed paths (training-free)

- **Hypothesis.** About 3 GB per cycle are avoidable without training:
  - a draft-only frequency-ranked head (FR-Spec), cutting 1.96 → ~0.5 GB per cycle;
  - the measured training-free FP4 mix (KDA 0-21 + lm_head: +0.0045 nats [M]), saving ~1.4 GB per cycle;
  - control-format (EXL3, 11.3 MB) experts for the GPU's share instead of Q4_K (14.16 MB).
- **Expected:** stacked on Bet 1, ~36 tok/s prose and ~41 on Hermes-like workloads [modeled R5/R5h]. Quality +0.0114 if additive [D] (bar +0.0148).
- **Changes:** fixed-path and draft terms.
- **Experiments:**
  1. Spark-only: FR-Spec head on the MTP drafter. Measure acceptance per position and cycle time against MTP-3.
  2. Spark-only quality: B + FP4 mix on the balanced eval with the clamp-fixed server.
- **Success:**
  - acceptance loss ≤ 3% with a draft-byte cut ≥ 60%;
  - combined dNLL ≤ +0.0125 (leaves margin to the bar).
- **Budget:** ~1 day Spark; 0 Max-Q GPU-h.
- **Approvals:** none.

### Bet 3: Cold prefill on the Spark GPU at decode precision, with cold experts streamed over a QSFP link

- **Hypothesis.** Cold 45K is compute-bound once experts are on the GPU. Precision must match decode. All 288 experts don't fit, so the 88 coldest per layer stream from workstation RAM, layer-ahead, double-buffered, into an FP8 staging buffer feeding a grouped GEMM.
- **Expected:** 994 s [M, config B] → 11-13 s with best-published kernels and a 100G link, 18-20 s with today's kernels [modeled]. Over 10 GbE the stream alone is ~45 s [modeled]. 10.8 s needs NVFP4 activations on the experts, or kernel efficiency beyond published results [D].
- **Changes:** cold TTFT (and removes the measured +0.014 prefill-precision penalty).
- **Experiments:**
  1. Spark-only: the per-layer staging pass (dequantize 288 experts to FP8). Measure GB/s and whether staging + FP8 grouped GEMM at 8-16K-token chunks beats 28 TFLOPS.
  2. Spark-only: an NVFP4-activation quality probe on long-context prefill (balanced eval, long-context domain).
  3. After purchase: the streamed-prefill prototype.
- **Success:** a one-layer staged prefill ≥ 120 TFLOPS effective; a modeled full TTFT ≤ 13 s from measured per-layer parts.
- **Budget:** ~2 days Spark; the purchase below.
- **Approvals:** **yes, a hardware purchase**: ConnectX-6 Dx 100GbE (or CX-7) + QSFP DAC for the workstation, ~$450-1,400.

### Bet 4: Union-aware routing, so the model cooperates with the verify window

- **Hypothesis.** Above ~40 tok/s the binding term is expert bytes per committed token (A4).
  - Decode routes already reuse experts: 49% of a token's experts were used in the previous 4 tokens [M].
  - A router nudged to reuse experts already selected in the same verify window lowers U per window at small quality cost.
  - Two forms:
    - **training-free:** at verify time, swap a selected expert for an in-window expert whose score is within δ;
    - **trained:** router fine-tuning with a λ·E[|U_window|] penalty, already in the 11-07 gate plan.
  - This is the project's scientific question in its sharpest form, and **no prior work was found**.
- **Expected:**
  - U4 22.7 → 16 lifts the R5h rung from ~41 to ~49 tok/s [modeled];
  - with the full stack (R8) and U scaled by 0.53, ~78 tok/s [modeled].
  - **This is the only modeled path that reaches the 80 goal.**
- **Changes:** the expert term.
- **Experiments:**
  1. Spark-only: δ sweep of window-sticky routing on the GPU-only build. Measure U per window (the union hook exists) and dNLL vs its own baseline.
  2. Then the same on config B for absolute quality.
  3. Only if the δ curve shows ≥ 25% union cut at ≤ +0.004 nats: a bounded router fine-tune.
- **Success:** a Pareto point with U cut ≥ 25% at ≤ +0.004 nats.
- **Budget:** step 1 is ~1 day Spark. Step 3 needs Max-Q GPU time beyond the approved QAT pilot, which is 5.31 of 8 GPU-h used and closed no-go.
- **Approvals:** step 3 only, needs Carmen's explicit go.

### Bet 5: Cached turn ≤ 0.2 s (software)

- **Hypothesis.** Fixing two measured causes brings the GPU-only cached turn from 0.35-0.56 s to ≤ 0.25 s [M causes, modeled effect]:
  - recompute since the last KDA state checkpoint (block-aligned, ~4.6-4.9K tokens);
  - ~250 ms host idle per request.
- Bet 1's split of the 32-token expert reads then gives ≤ 0.2 s at quality: 0.14-0.17 s [modeled].
- **Experiments:** Spark-only: checkpoint the KDA state at request end (align-mode patch) and remove the host overhead. Today's run measures the V2-runner cached turn, which may already cut host time.
- **Success:** p95 ≤ 0.25 s GPU-only over 20 trials.
- **Budget:** ~1 day.
- **Approvals:** none.

### What the stack gives, and what remains (modeled, honest)

- **Without training or purchases (Bets 1, 2, 5):** decode ~36 prose / ~41 Hermes; cached ~0.2 s; cold ~18-20 s (needs Bet 3's kernels, but not the link, for the resident part).
- **With the link (Bet 3):** cold 11-13 s.
- **80 tok/s needs Bet 4 to work**, and likely also:
  - fixed-path NVFP4 via QAD (Max-Q training);
  - a block drafter;
  - possibly the CPU bandwidth upgrade.
- 125 tok/s is not reached by any modeled combination (needs ≤ 3.0-3.7 GB per committed token).

## E. Today's measurement (Bet 1, step 1): cycle anatomy of the fastest GPU-only runtime

RESULTS_PLACEHOLDER
