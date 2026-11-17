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
| Config A, model runner V2 | same, V2 runner. 2026-11-10 delegated single run: 34.79. **Not reproduced on 2026-11-16** (section E). | 27.0-28.1 [M, 3 reps] | 41.6 s [M] | 0.37-0.40 s warm [M] | +0.182 (fails) |
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

### C2. Cheapest link that still meets cold 45K ≤ 10.8 s (added after Carmen's 14:05 answer: no purchase for now unless streaming absolutely needs it)

**Workload.**
- The Spark keeps ~200 hot experts per layer resident in control format. The memory budget is ~89 GiB for experts [D, not boot-validated].
- The other 88 per layer stream in per prefill as Q4_K: 52.3 GB per cold 45K prompt [D].
- The stream must finish inside the GPU compute time it overlaps with. That compute is 11-13 s with best-published kernels, and ≤ 10.8 s only with NVFP4-activation expert kernels [D from V]. So the link must carry 52 GB in ≲ 10 s, about 5+ GB/s effective, and preferably with slack.

**Workstation side.**
- Board: ASUS Pro WS WRX90E-SAGE SE.
- Onboard: Intel X710 2x10GBASE-T, one port on the direct 10G link, one on the LAN [M, lspci].
- At least two PCIe slots are empty [M, sysfs]. Physical clearance beside the two dual-slot GPUs needs checking by hand.
- Every candidate card below is PCIe Gen3 and fits a Gen5 x16/x8 slot.

**Spark side.**
- ConnectX-7 firmware lists 25/40/50/100GbE, with 50/100GbE in both NRZ and PAM4 modes [V, NVIDIA CX-7 firmware release notes].
- Forum reports say a 100G QSFP28 DAC and a 40G link both came up on a Spark [V, forum anecdote].
- No report pairs a Spark with a ConnectX-4 specifically.

| Option | Card price [V, listings, Nov 2026, undated] | Effective GB/s [A] | Cold-set stream time [modeled] | Meets ≤ 10.8 s? |
|---|---|---|---|---|
| Existing 10 GbE | $0 | 1.15 [M] | 45.5 s | **No** |
| ConnectX-4 Lx 25GbE | ~$30-60 (Carmen's figure; not re-checked) | ~2.9 | ~18 s | **No** |
| ConnectX-3 Pro 40GbE (QSFP+) | price not found | ~4.4 | ~11.9 s | **No**: the stream alone exceeds 10.8 s |
| ConnectX-4 Lx 50GbE (MCX4131A-GCAT) | ~$94-129 used | ~5.6 | ~9.3 s | Only on paper. No slack against 11-13 s compute, and 50G-CR2 interop with CX-7 is unverified. |
| **ConnectX-4 100GbE (MCX416A-CCAT dual / MCX415A single)** | **~$150-270** (new-old-stock ~$150, used ~$185-270) | ~11 (PCIe Gen3 x16 ≈ 12.5 max; Spark rail ~109 Gbit/s [V]) | **~4.8 s** | **Yes**: off the critical path, with ~2x slack |
| ConnectX-6 Dx 100GbE | ~$430-900 used | ~11 | ~4.8 s | Yes, at 2-5x the price |

Cable: 1 m passive QSFP28 100G DAC (MCP1600-C001-compatible), ~$23-35 [V, Optcore / Naddod listings]. Passive copper is safer than optics on the Spark [V, forum].

**Can more Spark-resident experts at lower precision avoid streaming?**
- To fit all 12,096 experts in ~96 GB, prefill copies must average ≤ ~2.5-2.6 bpw [D].
- Reading the prompt with 2-bit experts already costs +0.014 nats on the answers [M]. A ~2.6 bpw prefill copy is unmeasured; a linear interpolation [A] puts it near +0.008. Added to config B's +0.0069, that reaches the +0.0148 bar and leaves nothing for the decode-side FP4 mix.
- It also costs decode about 25% [modeled]: the GPU's share of decode experts would have to come from those low-bit copies or move to the CPU.
- **Verdict: not a reliable route to 10.8 s at control quality.** It can be measured cheaply (an IQ2_M/IQ2_S ~2.6 bpw GPU prefill copy with Q4_K CPU decode, as in config C). That test needs the workstation CPU to requantize from the BF16 GGUF, and is queued after the sweep.

**Recommendation.**
- **Buy:** one ConnectX-4 100GbE card (MCX416A-CCAT or MCX415A-CCAT) plus a 1 m QSFP28 passive DAC, about **$175-300 in total**.
- **Modeled cold 45K TTFT:** 11-13 s with best-published kernels. ~10.8 s needs NVFP4-activation expert kernels as well. The link is then not on the critical path.
- **Without a card,** the best modeled cold TTFT at control quality is ~45 s (10 GbE stream-bound).
- **Not yet:** buy only after the Spark-only staging experiment (Bet 3, step 1) shows a one-layer staged prefill ≥ 120 TFLOPS. Until then the link is not the bottleneck.

**Workstation CPU bandwidth (Carmen: Q3 yes, ≤ ~$20 rental).**
- No hourly rental of a 64-core Threadripper PRO 9000WX was found.
- The one bare-metal offer found (ip-projects.de, 9985WX) is €658/month with a setup fee for 1-month terms. Its example 64 GB RAM would not populate 8 channels anyway.
- **Proposed instead, free:** once the workstation is free, measure read bandwidth pinned to 1, 2, 3 and 4 CCDs of the existing 9975WX.
  - If bandwidth scales ~linearly per CCD (~62 GB/s each [D, 250/4]) and is not DRAM-saturated at 4 CCDs, an 8-CCD part should reach its DRAM limit. That is roughly 330-350 GB/s (85% of 410 GB/s spec) [A].
  - If 4 CCDs already saturate, the upgrade buys nothing.
- No rental is needed unless that result is ambiguous.

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
- **Approvals:** a hardware purchase, **declined for now (2026-11-16)**. The cheapest adequate link is a ConnectX-4 100GbE card + QSFP28 DAC, ~$175-300 (section C2). It is requested only after step 1 shows the kernels make the link the bottleneck.

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
  - Status 2026-11-16: the training-free step **failed its gate** (E2).
- **Changes:** the expert term.
- **Experiments:**
  1. Spark-only: δ sweep of window-sticky routing on the GPU-only build. Measure U per window (the union hook exists) and dNLL vs its own baseline.
  2. Then the same on config B for absolute quality.
  3. Only if the δ curve shows ≥ 25% union cut at ≤ +0.004 nats: a bounded router fine-tune.
- **Success:** a Pareto point with U cut ≥ 25% at ≤ +0.004 nats.
- **Budget:** step 1 is ~1 day Spark. Step 3 needs Max-Q GPU time beyond the approved QAT pilot, which is 5.31 of 8 GPU-h used and closed no-go.
- **Approvals:** step 3 is **approved conditionally (Carmen, 2026-11-16)**.
  - Condition: step 1 must show ≥ 25% fewer distinct experts per window at ≤ +0.004 nats.
  - A protocol with a GPU-hour budget and a stop rule is pre-registered in docs/ before any training.
  - It starts only after the workstation sweep ends and the workstation is declared free.

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
- **80 tok/s:** the training-free form of Bet 4 failed its gate (E2), so no measured path reaches 80 yet. It would need a trained union-aware router (not triggered by the pre-registered gate) or a larger expert-byte cut, together with:
  - fixed-path NVFP4 via QAD (Max-Q training);
  - a block drafter;
  - possibly the CPU bandwidth upgrade.
- 125 tok/s is not reached by any modeled combination (needs ≤ 3.0-3.7 GB per committed token).

## F. Evaluation design for the paper

This is a research project aimed at a paper, not only an engineering target. The plan above is organised by targets. This section organises the same work by claims, benchmarks, baselines and ablations, so every result lands on a frontier and can be compared.

### F1. Scientific question and candidate claims

**Question (charter).** On the quality-latency frontier, does hardware-aware *joint* adaptation of MoE expert weights, precision and placement beat *independently* optimised compression plus scheduling?

**Candidate primary claims.** Each is falsifiable, and each is tied to the bet that tests it.

- **C1 (system, Bet 1).** On a unified-memory GPU plus a remote CPU over Ethernet, an exact per-layer split of both fixed and expert weights moves the decode frontier beyond any expert-only placement at equal quality. The two memory systems stream concurrently instead of in series.
  - Test: measured tok/s at equal dNLL against the best expert-only split.
- **C2 (model-system co-design, Bet 4).** Making routing aware of the speculative verify window (causal stickiness, then a trained router) cuts expert bytes per committed token at smaller quality cost than lowering expert precision for the same byte saving.
  - Test: frontier of dNLL against bytes per committed token. Two curves share the axes: union-aware routing vs uniform or mixed lower-bit experts.
  - This is the claim that answers the charter's question directly.
- **C3 (precision as a phase constraint, measured).** The prompt must be read at near-decode expert precision.
  - Measured: 2-bit prefill with 4-bit answers costs +0.014 nats (config C).
  - Consequence: placement can't simply use low-bit prefill copies to save memory. This is the reason cold TTFT needs streamed weights (Bet 3, C2 of the link analysis).
- **C4 (bound).** At batch 1, expert bytes per committed token barely amortise with speculation (measured unions U(4) = 22.7, U(8) = 38).
  - Consequence: above ~40 tok/s on this hardware class, drafter quality matters less than routing locality.

### F2. Benchmarks (fixed before final runs)

**Quality**
- **Balanced 7-domain eval:** dNLL, KLD and same-top-1 against the BF16 proxy (Q8_0), reported per domain.
  - **Upgrade before the paper:** a BF16-logit reference on a subset, to bound the proxy's own error.
  - Fix the eval-size resolution: SE ~0.004 nats at 40K tokens, so the confirmatory set is ~400K tokens.
- **Task accuracy on a fixed suite:**
  - the Hermes regression suite (280 items, already scored for control and B);
  - plus public tasks with fixed prompts and seeds: GSM8K-style math, HumanEval/MBPP-style code, knowledge QA (MMLU-Pro subset), IFEval-style instruction following.
- **Long-context quality at 45K:** a needle/retrieval set plus long-document QA at 16K/32K/45K. This is where prefill precision (C3) should show.

**Latency (batch 1, single stream, fixed decoding settings)**
- Decode tok/s, with p50/p95 ITL, committed tokens per cycle, and per-position acceptance, on prose, chat, code, reasoning and tool workloads (`ops/r16/cycle_client.py`).
- Cold TTFT at 8K/16K/32K/45K.
- Cached TTFT at 45K + {16, 32, 64, 128, 256}, with first-hit and warm cases reported separately (`src/ttft_bench.py`, `docs/TTFT_PROTOCOL.md`).
- A sweep over context length for decode (KV/state growth).

**Cost axes**
- Bytes read per committed token, split by fixed, draft and expert terms, from headers plus measured unions (`src/cycle_bytes.py`).
- Link bytes per token and per prefill, and link bandwidth used.
- Energy per token (J): the Spark from its power telemetry, the workstation from the existing 5-minute GPU/CPU power log. Report only if the sampling resolution allows a per-run integral; otherwise mark it not measured.
- Memory footprint per device.

### F3. Baselines

1. **Control:** the original GLM-5.3-Flash, EXL3 3.5 bpw, on 2x Max-Q with vLLM. This is the quality and latency reference (measured: 151 tok/s, 11.1 s cold, 0.156 s cached, dNLL +0.0148).
2. **Initial deployment:** llama.cpp RPC (E2E-v0 over 1 GbE, E2E-v1 over 10 GbE), measured.
3. **Independent compression plus default scheduling:** the best compression alone, deployed with the stock engine's own offload policy and no co-design. That means config A/B-style checkpoints under stock vLLM CPU offload, or llama.cpp `-ot` expert offload, on the same hardware.
4. **Best published CPU-offload systems that can run the model:**
   - KTransformers / SGLang-kt (supports CPU experts; check GLM-5.3-Flash support);
   - llama.cpp/ik_llama.cpp with expert offload.
   - All run local-CPU style: same box over PCIe is impossible here, so each runs with its CPU tier on the workstation, or is reported as not runnable with the reason.
   - Remote-CPU support over Ethernet is itself part of what we compare.

Community single-Spark checkpoints remain background only (charter); they are not baselines.

### F4. Ablations (each contribution isolated, same checkpoint and settings)

| Ablation | Off | On | Metric |
|---|---|---|---|
| Dual-memory split (C1) | expert-only split (fixed path on GPU) | exact fixed + expert split | tok/s and cycle anatomy at equal quality |
| Link for the split | 10 GbE busy-poll (measured) | RDMA (if a link is ever bought), otherwise modeled only, labelled | cycle sync time |
| Byte diet: draft head | full lm_head in drafts | FR-Spec draft head | acceptance, cycle ms |
| Byte diet: FP4 mix | FP8 fixed | KDA 0-21 + lm_head NVFP4 | dNLL, cycle ms |
| Byte diet: expert format | Q4_K everywhere | control-format GPU share + Q4_K CPU | dNLL, bytes per token |
| Union-aware routing (C2) | stock | sticky β sweep; trained router | U4, dNLL frontier, acceptance |
| Expert precision (C2 comparator) | 4-bit | 3-bit-class at the same byte saving | dNLL at equal bytes per token |
| Prefill precision (C3) | prompt read at decode precision | prompt read with 2-bit / ~2.6-bit copies | per-domain dNLL, long-context accuracy |
| Link speed for cold prefill | 10G measured | 25/50/100G modeled from measured per-layer parts; measured if hardware appears | cold TTFT, stream share |

### F5. Reporting rules

- **Frontiers, not single points.** Quality (dNLL / task score) against each latency axis, and against bytes per committed token. Every treatment is a curve where a knob exists: β, λ, precision mix, split fraction.
- **Uncertainty.** Paired bootstrap 95% CIs over prompts (`src/paired_analysis.py`). Run-to-run spread for speed, with ≥ 3 boots per configuration (MTP decode has already shown 15.3-19.5 tok/s spread).
- **Fixed seeds, fixed prompts, fixed decoding settings,** recorded with every result. Model revisions and checkpoint hashes are logged.
- **Pre-registration.** Each confirmatory comparison gets a protocol in `docs/` before its runs, as the QAT pilot and the union-router pilot did. Exploratory runs are labelled exploratory.
- **Labels.** Measured [M], derived [D], modeled, vendor-claimed [V]. Modeled numbers never appear in the results tables, only in a separate projections table.

### F6. Results that matter even if negative

- **The network is not the bottleneck at the expert-only structure:** 2-4% of the cycle (A2). This contradicts the common assumption that Ethernet-attached CPU tiers are latency-bound.
- **125 tok/s is unreachable on this hardware class at control quality** (A4 bound), with the bound decomposed into measured terms.
- **Speculation barely amortises expert bytes at batch 1** (C4). This explains why drafter gains reported on dense models don't transfer.
- **Prefill precision constrains placement** (C3).
- **Layer-local QAT recovers only 15-58% at 3 bits** (QAT pilot, measured), while end-to-end methods remain open.
- **If union-aware routing fails** (no ≥ 25% union cut at ≤ +0.004 nats), that is a clean negative answer to the co-design question at this scale: compression and scheduling then can't be beaten by routing adaptation, and the paper reports where the frontier sits.

## E. Today's measurement (Bet 1, step 1): cycle anatomy of the fastest GPU-only runtime

**Run.** 2026-11-16, Spark only, one boot.
- Build: `full_iq2xxs_fp8` (FP8 fixed weights, IQ2_XXS experts, MTP layer kept). Settings: MTP-3, `VLLM_USE_V2_MODEL_RUNNER=1`, CUDA graphs, BLHNC KV layout.
- Script: `ops/r16/cycle_anatomy.sh`. Raw data: `results/2026-11-16-cycle/v2_fp8_mtp3/client.jsonl`.
- GPU-only diagnostic of the runtime; its quality fails the bar by construction.

**Decode, with spec-decode counters per request [M]:**

| Workload | tok/s | Committed per cycle | ms per cycle | Accept rate, positions 1/2/3 |
|---|---|---|---|---|
| Essay (serve_eval benchmark), rep 0/1/2 | 27.37 / 28.13 / 27.03 | 2.31 / 2.42 / 2.33 | 84.3 / 85.9 / 86.1 | ~0.71-0.77 / 0.40-0.45 / 0.16-0.20 |
| Prose, 400 tokens | 25.59 | 2.25 | 88.3 | 0.68 / 0.37 / 0.20 |
| Chat | 25.30 | 2.19 | 86.9 | 0.65 / 0.39 / 0.15 |
| Code | 24.62 | 2.20 | 89.8 | 0.61 / 0.40 / 0.20 |
| Reasoning | 28.06 | 2.49 | 89.1 | 0.66 / 0.49 / 0.34 |

**Profiled 64-token decode: per-cycle GPU anatomy [M]** (`v2_fp8_mtp3_decode64_cycle.json`; 28 cycles, 2.25 committed per cycle).
- **88.6 ms GPU-busy per cycle, 98% busy.** Runtime/host overhead is about 2%.
- Dense GEMMs (verify fixed path + 3 draft passes): **58.6 ms**. Against the 11.8 GB header byte count that is **~201 GB/s** [D].
- Routed-expert kernels: **32.0 ms**, against ~6.24 GB, i.e. **~195 GB/s** [D].
- KDA, MLA/DSA, norms and sampling: about 5 ms in total.

**Cold and cached [M]:**
- Cold 45K: 41.6 s.
- Cached 45K+32: 1.30 s (profiled), then 6.79, 0.40 and 0.37 s.
  - The first unprofiled follow-up again recomputes from the last KDA block checkpoint, as on the V1 runner (5.2 s on 11-09).
  - The profiled cached turn is GPU-busy only 36% of a 746 ms span; 179 ms of the 267 ms GPU time is the W4A16 MoE prefill kernel on the recomputed tail.

**What this settles:**
1. **The 58.5 ms fixed path is now measured, not inferred.** It was earlier inferred from a 34% MoE-share assumption; the measurement is 58.6 ms. Every term in the cycle runs at the GB10's streaming bandwidth.
   - So no runtime trick on the GPU alone moves decode much: overhead is about 2%.
   - Decode speed comes only from fewer bytes per committed token (Bets 2 and 4) or a second memory system reading in parallel (Bet 1).
   - The gap model's GPU bandwidth of 0.208 TB/s matches the measured ~0.20.
2. **The V2 runner does not change the ceiling.** Same essay benchmark: 27.0-28.1 tok/s today against 25.67 on V1 (11-08), a +5-10% runner effect.
   - The 34.79 tok/s single run of 11-10 is not reproduced. It is treated as unreplicated and is not used for planning.
3. **The cached-turn problem is unchanged.** Bet 5 remains a software fix: checkpoint the KDA state at request end, and remove host idle. Its first measurement on this build is next.
4. **Implication for the bets.** Bet 1's case strengthens: the GPU half is purely bandwidth-bound, so every byte moved to the workstation's own memory comes off the critical path one for one. The draft path's 3 × 0.95 GB is measured inside the 58.6 ms, so Bet 2's draft diet is sized correctly.

### E2. Bet 4 step 1: training-free causal window-sticky routing [M]

**Setup.**
- Build and runner: the same GPU-only FP8 build, run eagerly through the quality pipeline.
- Hook: `ops/r16/sticky_hook`, chained before the logits capture.
- Eval: balanced eval, 41,943 tokens, scored against the Q8_0 BF16 proxy.
- Routing change: token i gets a bonus β for experts the previous W = 3 tokens picked under stock routing. It is causal and non-recursive.
- U4 = distinct experts per 4-token window, measured over the eval prefill.
- Summary file: `results/2026-11-16-cycle/sticky/sticky_sweep_summary.json`.

| β | U4 stock → sticky | Union cut | Picks changed | dNLL (SE) | Δ vs β = 0 | KLD | Same top-1 |
|---|---|---|---|---|---|---|---|
| 0 (control) | 21.85 → 21.85 | 0% | 0% | +0.1832 (0.0042) | 0 | 0.239 | 84.9% |
| 0.02 | 21.85 → 20.00 | 8.5% | 10.1% | +0.1859 (0.0042) | +0.0027 | 0.242 | 84.7% |
| 0.05 | 21.87 → 18.06 | 17.4% | 22.2% | +0.1972 (0.0044) | +0.0140 | 0.254 | 84.3% |
| 0.10 | 21.89 → 16.53 | 24.5% | 33.8% | +0.2226 (0.0046) | +0.0394 | 0.281 | 83.4% |

- **Hook validity.** β = 0 reproduces the stock build's +0.182 [M, 11-08] with 0 changed picks.
- **Gate (pre-registered): ≥ 25% union cut at ≤ +0.004 nats. NOT MET.**
  - At +0.0027 the cut is only 8.5%.
  - A ~25% cut costs about +0.039, roughly 10x the allowance.
  - So the bounded router pilot is **not triggered**, per `docs/UNION_ROUTER_PILOT_PROTOCOL.md`.
- **Caveats.**
  - The costs are relative to a 2-bit-expert build; the cost on config B is unmeasured.
  - SEs are per configuration (unpaired), so the small-β difference is near resolution.
  - A constant bonus is the crudest form of union-awareness. A trained router could sit on a better curve, but this result gives no evidence that it would.
- **Comparator for claim C2 (from existing measurements).**
  - IQ3_XXS PTQ cuts expert bytes ~35% (4.5 → 3.06 bpw) at about +0.05 nats on config B (≈ +0.00118 per layer × 42 [M, QAT addendum]).
  - Training-free stickiness cuts ~25% for +0.039 here, a similar cost per byte saved.
  - **The training-free form is not better than lowering precision.** That is an informative negative for the co-design claim at the training-free level.
- **Modeled decode effect at the allowed quality cost** (8.5% union cut): about +2 tok/s on the R5h rung (~41 → ~43). Bet 4 drops from "the only path to 80" to unproven. 80 tok/s at control quality now has no modeled training-free path.

**Next:**
- **Bet 5 (Spark):** KDA checkpoint at request end, plus host-idle removal on the cached turn.
- **Bet 2 (Spark):** FR-Spec draft head.
- **Bet 1 workstation microbenchmarks:** when the workstation is free.
- **Per-CCD bandwidth on the workstation's own CPU:** replaces the cloud rental.

### E3. Bet 1 steps 2-3 + prototype: both memories at once [M] (2026-11-16 evening)

Full write-up: `results/2026-11-16-dualmem/README.md`.
- **Simultaneity: yes.** Spark GPU 239.7 GB/s alone and while the workstation streams; workstation CPU 251.2 GB/s alone and while the Spark GPU + Arm stream.
  - Correction to M0: the Spark Arm CPU takes ~33 GB/s *from* the GPU (GPU 239.7 → 202). It adds no bandwidth during decode.
- **Step 2:** workstation FP8 GEMV in the checkpoint's exact E4M3 + E8M0 format, T = 1-4, reads at 229-246 GB/s at every split shape (attention, MLP, shared expert, lm_head). The modeled value was 170.
- **Per-CCD (Q2):** 63 / 125 / 187 / 251 GB/s on 1-4 CCDs. That is linear, ~63 GB/s per CCD link, and not DRAM-bound (peak 358 at DDR5-5600).
  - An 8-CCD part would be DRAM-limited, not CCD-limited: ~305-350 GB/s [A: efficiency unmeasured].
- **Step 3:** the 90-swap loop at T = 4 costs 4.5-5.6 ms per cycle with Energy-Efficient Ethernet off, and 15-18 ms with NIC defaults.
  - The "~300 us idle-gap penalty" of 2026-11-08 was EEE low-power idle (Spark LPI timer 115 us). No heartbeat is needed once EEE is off.
- **Prototype:** a full 90-phase cycle of split layers, real FP8 kernels on both sides, swaps over the real link.
  - 56.7 ms against 100.1 ms GPU-only on the same 20.8 GB: **1.77x, 366 GB/s effective** (EEE off; 1.45x with NIC defaults).
- **Modeled from these measured parts:** ~26-28 tok/s at config-B quality with drafts on the GPU, and ~28-31 with the draft passes split too. Config B measures 15.3-19.5. This agrees with R2/R3.
- **Next:** head-split exactness on real weights (step 4), then the runtime.
