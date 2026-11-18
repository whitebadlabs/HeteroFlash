# HeteroFlash

**Running a 320B-class MoE model interactively on one DGX Spark plus a CPU box: measured findings**

*Independent research, November 2026. Every number below is measured unless it is marked* modeled *or* derived,
*and each one points to the committed result file it comes from (paths are relative to this repository).*

---

## 1. The problem

GLM-5.3-Flash is a Mixture-of-Experts model: in each of its 42 MoE layers a router picks 8 of 288 routed experts per
token. Only a small part of the model is active per token, but every expert must be readable. At 4 bits the routed
experts alone take about 171-177 GB, more than one DGX Spark (128 GB of unified memory) can hold next to the
attention weights, the KV cache and a draft head.

The question: **can GLM-5.3-Flash run interactively (batch 1, one user) on one DGX Spark, helped only by the CPU and
RAM of an ordinary workstation, at quality no worse than a two-GPU server deployment of the same model?**

Hardware, by role:

| role | hardware | what it does |
|---|---|---|
| GPU node | one DGX Spark (GB10, 128 GB unified LPDDR5X, 273 GB/s rated) | attention, KV cache, a hot subset of experts, the MTP draft head |
| CPU node | a 32-core, 4-chiplet workstation CPU, 8-channel DDR5-5600, ~230 GB free RAM; its GPUs are **not** used at inference | every routed expert in RAM, part of the vocabulary projection, optionally some attention projections |
| link | one direct 10GBASE-T cable, MTU 9000, TCP (no usable RDMA device) | per-layer exchanges |
| control (quality and speed reference) | the same model on a two-GPU workstation server: vLLM, EXL3 3.5 bpw weights, tensor parallel 2, MTP with 3 draft tokens | 151 tok/s prose, and the quality bar in section 3 |

The project's targets are 80 tok/s (stretch 120-130) at control quality, a cold 45K-token TTFT of at most 10.8 s and a
cached 45K + 32-token TTFT of at most 0.2 s. **None of these is met today** (section 6).

## 2. Architecture

### 2.1 Placement

Weights come from the official BF16 release of GLM-5.3-Flash (revision `f9fed3bbc822`). The non-expert ("fixed path")
weights run in FP8 on the Spark. Every routed expert exists as Q4_K + importance-matrix GGUF bytes in workstation RAM,
and the Spark keeps a resident hot set of the *same* Q4_K bytes.

A decode cycle is one verify pass over 4 positions plus 3 MTP draft passes. Its work is split three ways
(`results/2026-11-17-dualrt/README.md`, section 0):

- **Routed experts (layers 3-44): split by expert, re-balanced every step.** The Spark holds 112-120 hot experts per
  layer, chosen from decode-route counts on a calibration slice disjoint from the measured prompts
  (`results/2026-11-17-dualrt/hotset_calib.json`). Per layer and step, an in-graph assign kernel takes the verify
  window's expert union and runs the resident experts on the GPU with its own Q4_K kernel; the rest go to the
  workstation. Measured per layer and step: a union of 23.3 experts, 12.1 computed on the GPU, 11 on the workstation.
- **Vocabulary projection (lm_head): split by row.** Rows 0-84,991 on the GPU, the rest on the workstation; exact bf16
  logits come back (no top-k approximation).
- **Linear-attention (KDA) q/k/v projections: split by head.** Optional; it does not pay (finding F7) and is off in the
  recommended configuration.

### 2.2 The mailbox exchange

The GPU stays the master, and the whole step stays inside one CUDA graph
(`results/2026-11-08-route3/README.md`, "Runtime v2 mailbox"; `docs/RESEARCH_PLAN_2026-11-16.md`, section E4):

1. A pack kernel writes the layer's routes, gate weights and hidden rows into a pinned, host-mapped mailbox and sets a
   release flag with a device-side sequence counter.
2. A relay thread on the Spark CPU forwards the request over the direct link to the workstation's expert server
   (`runtime/expert_server_gguf.c`, 32 pinned threads, busy-poll sockets).
3. A spin-wait kernel queued later in the same graph waits for the reply and adds the workstation's partial sums
   (bf16 by default, fp32 optional).

Exchange points are therefore not graph breaks: about 91 in-graph exchange points per verify pass and 3 per draft pass.
The mailbox is exact against the eager path and under CUDA-graph replay (`results/2026-11-08-route3/README.md`).
The design follows published hybrid CPU/GPU MoE systems (device-visible mailboxes, one captured graph); it is not
claimed as new here.

Code: `bench/dualrt/` (hook `hook/sitecustomize.py`, CUDA extension `hook/hf_drt.cu`) and `runtime/expert_server_gguf.c`.

## 3. Methodology

- **Speed.** Batch 1, greedy, MTP with 3 draft tokens, CUDA graphs. Each frontier "point" is a run-time switch on one
  server boot, so points run interleaved, round-robin, on the same server. A round sends 15 requests: one 256-token
  essay, 4 chat prompts (prose, code, chat, reasoning; 400 tokens) and 10 agent-style items (2 each of tool calling,
  multi-turn, coding, prose and reasoning, taken from public benchmarks; 300 tokens; the `hermes` group in the result
  files). The reported tok/s pools all 15 requests over 3 rounds.
- **Quality.** A balanced evaluation set: 41 chunks, 41,943 scored tokens, 7 domains (writing/chat, knowledge QA, math,
  code, multilingual, instruction following, long context), with prompt *and* answer text
  (`results/2026-11-08-balanced-eval/q_eval_balanced_v1.manifest.json`). Every position runs through the split path in decode-sized batches
  (16 tokens per step, eager mode). Logits are compared with a Q8_0 quantization of the BF16 weights (the "proxy"):
  dNLL (mean extra negative log-likelihood in nats per token, with its standard error), mean KL divergence, and the
  same-top-1 rate. Scorer: `bench/dualrt/compare_q8.py`.
- **Control bar.** The two-GPU control, scored against the same proxy on the same tokens:
  **dNLL +0.0148 (SE 0.0017), KLD 0.042, top-1 94.0%**
  (`results/2026-11-08-balanced-eval/control_exl3_3.5bpw.json`, key `_overall`). A point is *inside* the bar only if
  all three metrics are at least as good as the control's.
- **Anchors.** Every speed boot re-runs reference points from the first boot. If an anchor misses by more than
  0.3 tok/s, the boot's other points are not trusted. That rule rejected one boot (finding F9).

## 4. Measured results

### 4.1 The two memories add up

Coordinated, time-stamped streaming on both machines at once (`results/2026-11-16-dualmem/README.md`, section 1;
raw output `overlap_legA.txt`, `overlap_legC.txt`):

| stream | alone | while the other machine streams |
|---|---|---|
| Spark GPU, bf16 read-reduce | 239.7 GB/s | 239.8 GB/s |
| workstation CPU, 32 cores | 251.2 GB/s | 251.2 GB/s (Spark GPU *and* Spark Arm cores streaming) |

The two memories do not interfere, so together they offer about 240 + 251 GB/s of raw read bandwidth. Two measured
caveats:

- **The Spark's own Arm cores add nothing.** They share the GPU's memory: a 20-thread Arm stream cut the GPU from
  239.7 to 202 GB/s (`overlap_legB.txt`).
- **The workstation CPU is limited by its chiplet links** (about 63 GB/s per chiplet, linear up to 4), not by DRAM
  (`results/2026-11-16-dualmem/README.md`, section 2).

A real decode cycle gets less than the sum. In a 90-phase prototype of one verify pass, splitting each layer between
the two machines read **366 GB/s effective, against 207 GB/s for the GPU alone on the same bytes: 56.7 vs 100.1 ms per
cycle, 1.77x**. That figure is with Energy-Efficient Ethernet and receive coalescing off; at NIC defaults it was 1.45x
(`results/2026-11-16-dualmem/README.md`, section 4; `proto_full_t32.txt`, `proto_gpu_only_full_bytes.txt`).

### 4.2 The split is exact

- **In math.** Splitting real FP8 attention layers by head across the two machines matches the unsplit computation to
  1.1e-6 (linear-attention layer) and 7.8e-7 (MLA layer) relative L2 in fp32
  (`results/2026-11-16-dualmem/headsplit_compare.json`).
- **Kernels.** The Spark's Q4_K expert kernel matches an fp32 dequantized reference to 3-6e-7 relative L2
  (`results/2026-11-17-dualrt/unit_q4k_L3.json`).
- **End to end, paired on the same 40,920 tokens.** Unsplit vs split with bf16 partial sums: **dNLL +0.0007
  (SE 0.0008)**, direct KL 0.010, same top-1 96.9% (`results/2026-11-17-dualrt/paired_q_unsplit_vs_q_split_bf16.json`).
  For scale: 96.8-97.0% top-1 agreement is what summation order alone produces between any two of these runs, and
  fp32 vs bf16 partial sums differ by -0.0007 (SE 0.0008) (`paired_q_split_fp32_vs_q_split_bf16.json`).
  The split against the proxy: +0.0044 (SE 0.0014), KLD 0.027, top-1 95.1% (`q_split_bf16_vs_q8_bal.json`), well
  inside the control bar.

### 4.3 Speed progression

The rows use different workloads, so compare within a group of rows, not across groups.

| stage | workload | tok/s | source |
|---|---|---|---|
| control, two-GPU server | prose | 150.8 | `results/2026-11-07-control-baseline/README.md` |
| first end-to-end deployment (llama.cpp RPC over 1 GbE, no drafting) | prose | 6.46 | `results/2026-11-07-e2e-v0/README.md` |
| same, direct 10G link + MTP drafting | prose | 17.0 | `results/2026-11-07-e2e-v1/README.md` |
| "config B": vLLM, FP8 fixed path, **all** routed experts on the workstation CPU, mailbox, MTP-3 (3 reps) | essay / chat / agent | 16.58 / 15.43 / 18.00 | `results/2026-11-17-dualrt/cfgB_base_r1_client.jsonl` |
| dual-memory split, same session (3 reps) | essay / chat / agent | 22.82 / 22.05 / 23.03 | `results/2026-11-17-dualrt/m4_final_g11a7_client.jsonl` |
| frontier boot f1: split, KDA split on | 15-request pool | 20.75 | `results/2026-11-17-frontier/f1_frontier_speed.json` (`base`) |
| f1: KDA split off (lossless) | 15-request pool | 23.21 | same file (`kdaoff`) |
| f1: KDA split off + workstation NIC interrupt coalescing off (lossless) | 15-request pool | 24.68 | same file (`coal_kdaoff_c0`, 6 rounds, ABBA) |
| **f3: top-7 experts + KDA split off (recommended)** | 15-request pool | **25.62** (25.7 on warm rounds 2-3, derived) | `results/2026-11-17-frontier/f3_frontier_speed.json` (`k7_kdaoff`) |

The split against config B in the same session: cycle time 148.5 -> 120.6 ms (essay) and 150.9 -> 119.9 ms (chat),
1.21-1.26x (`results/2026-11-17-dualrt/README.md`, section 3).

For context, the Spark alone, with every expert in 2 bits on the GPU, reaches about 27-28 tok/s but scores dNLL +0.18
against the proxy, twelve times the control's +0.0148 (`results/2026-11-16-dualmem/README.md`, section 5;
`results/2026-11-17-frontier/gpuonly/r17_drop_base_vs_q8_bal.json`). The workstation is what buys the quality.

### 4.4 The speed-quality frontier

The lever: drop routed experts at run time, either with a fixed top-k (keep the k highest-scoring of the 8) or with a
weight threshold tau (drop experts whose gate weight is below tau). The dropped weight is **not** redistributed.
All points have the KDA split off. Speed is pooled tok/s over 3 interleaved rounds; quality as in section 3
(`results/2026-11-17-frontier/README.md`).

| point | tok/s (boot) | dNLL (SE) | KLD | top-1 | against the control bar |
|---|---|---|---|---|---|
| control bar | | +0.0148 (0.0017) | 0.042 | 94.0% | |
| all 8 experts (lossless reference) | 23.2 (f1); 23.9 (f3 rounds 2-3, derived) | +0.0044 (0.0014) | 0.027 | 95.0% | inside |
| tau 0.04 | 24.3 (f3) | +0.0055 (0.0014) | 0.030 | 94.9% | inside |
| **top-7** | **25.6 (f3)** | **+0.0099 (0.0016)** | **0.037** | **93.9%** | **dNLL and KLD inside; top-1 0.1 pt under** |
| top-7 + tau 0.04 | 25.8 (f3) | +0.0112 (0.0017) | 0.040 | 93.8% | top-1 0.2 pt under |
| top-6 | 25.8 (f1) | +0.0181 (0.0020) | 0.059 | 92.4% | outside |
| tau 0.08 | 26.4 (f1); 26.3 (f3) | +0.0313 (0.0022) | 0.077 | 91.4% | outside |
| top-4 | 27.6 (f1) | +0.0886 (0.0031) | 0.156 | 86.9% | far outside |
| top-4, weights renormalised | not timed | +0.1234 (0.0034) | 0.174 | 86.6% | far outside |

Files, all under `results/2026-11-17-frontier/`: speed in `f1_frontier_speed.json` and `f3_frontier_speed.json`;
quality in `q_fq1_{base,k6,k4,k4rn}_vs_q8_bal.json` and `q_fq2_{k7,tau0.04,k7tau0.04,tau0.08}_vs_q8_bal.json`
(key `_overall`; per-token arrays in `*_pertoken.npz`).

Timed but not quality-scored (past the bar by construction, since each drops more than a scored point that is already
outside): top-5 27.1, top-6 + tau 0.08 27.6, top-4 with 3-bit experts on 15 CPU layers 28.0, top-4 + coalescing off
30.4 tok/s (all f1).

**Recommended operating point: top-7 with the KDA split off, 25.7 tok/s on warm rounds** (25.62 pooled, including the
cold first round).

- That is about +7.5% over lossless on the same boot (23.9, derived from rounds 2-3) and +10% over f1's lossless 23.2.
- dNLL and KLD are inside the control bar.
- Same-top-1 is 93.88% against the control's 93.97%, so it fails the strict all-three rule by 0.1 point. With a top-1
  standard error of about 0.12 point we read it as *at* the bar, not inside it.
- Top-7 combined with coalescing off is **modeled** at about 27 tok/s. It has not been measured.

## 5. Findings

Each finding is labelled. *Likely new* means we did not find it stated in the prior work we checked; it is not a
claim that nobody has seen it. *Known* means established practice or published; those are listed because the measured
size matters here.

**F1. Renormalising after dropping experts hurts.** *Likely new as a measured ablation.*
The usual implementation divides the kept experts' gate weights by their sum, so they add up to 1 again. Leaving the
dropped mass out instead was better at every level we measured:
- Model level, frontier eval: top-4 renormalised +0.1234 vs plain top-4 +0.0886 dNLL
  (`results/2026-11-17-frontier/q_fq1_k4rn_vs_q8_bal.json`, `q_fq1_k4_vs_q8_bal.json`).
- Layer level, real inputs: renormalised top-4 output error 0.23-0.69 per layer, against 0.05-0.47 with the original
  top-8 normalisation. The latter is within 0.011 of the least-squares-optimal weights for the same 4 experts
  (`results/2026-11-07-g2-topk-decomposition/README.md`).
- An earlier model-level run on a different eval set agrees: 0.257 vs 0.205 nats (`CURRENT_STATUS.md`, M2 round 3).

Related work on dynamic expert skipping exists; we found no direct renormalised-vs-not ablation in it.

**F2. Fixed top-k beat a weight threshold at matched speed.** *Setting-specific; likely new for this model.*
- On the same boot, top-6 gave 25.8 tok/s at +0.0181 dNLL, while tau 0.08 gave 26.4 tok/s at +0.0313: about 1.7x the
  quality cost for +0.6 tok/s.
- The gentle threshold tau 0.04 was nearly free (+0.0055 vs +0.0044) but bought only +0.4 tok/s (+1.6%), inside the
  ±0.5 tok/s measurement band.
- tau 0.04 saves little because it drops little: 0.33 of 8 experts per token (1% of the gate weight), and it shrank
  the 4-position expert union by 4.3% (GPU-only screen, `results/2026-11-17-frontier/gpuonly/r17_drop_tau0.04_drop_stats.json`).
  Why tau 0.08 costs more quality than top-6 at similar speed is not established.
- Caveat: only two thresholds were tested. Published threshold policies on other models report good frontiers.

**F3. A ~100-115 ms cycle floor caps what dropping experts can buy; overlap is the next lever.** *The floor is
measured; the conclusion is our inference.*
- Within the control bar, trading accuracy buys +7-10% (23.2-23.9 -> 25.7 tok/s).
- Even top-4, far past the bar, only reaches a 100 ms cycle (`f1_frontier_speed.json`, `k4_kdaoff`: 100.2 ms at 2.77
  committed tokens per cycle).
- The floor is made of sequential exchanges. Each expert exchange costs 0.91 ms of server compute but 1.23 ms round
  trip, so about 0.32-0.36 ms per exchange is outside compute. Times ~80 exchanges per cycle that is ~26 ms per cycle
  (derived; `results/2026-11-17-dualrt/README.md`, section 3), on top of the CPU expert time itself.
- Dropping experts shortens neither the per-exchange latency nor the number of exchanges. The next speedup has to come
  from overlapping the exchange with GPU work, or from fewer, fused exchanges. That work is in progress, with no
  results yet.

**F4. The two memories really are additive, and the Spark's Arm cores are not a third tier.** *The concept is known
(hybrid CPU/GPU MoE inference); the measurements are new for this pair.* Section 4.1: 239.8 + 251.2 GB/s
concurrently, 1.77x effective in a full-cycle prototype; a 20-thread Arm stream takes 16% of the Spark GPU's
bandwidth.

**F5. Splitting by head, expert or vocabulary row is exact, and bf16 partial sums cost nothing measurable.**
*Known technique (tensor-parallel-style decomposition); validated here.* See section 4.2. A pre-registered per-layer
criterion (split vs GPU-only error at most 1e-3) was **not** met (2.3-3.0e-3). A control that splits the same heads
across two GPU runs, with no CPU involved, gives the same 2.3-2.5e-3, so the gap is bf16 output rounding: the
criterion was set below bf16 resolution (`results/2026-11-16-dualmem/README.md`, section 6).

**F6. Turning NIC interrupt coalescing off is a lossless lever, and so is turning Energy-Efficient Ethernet off.**
*Known low-latency networking practice; the size of the effect is the result.*
- Workstation receive interrupt coalescing off (ABBA order, same boot): cycle 122.8 -> 115.5 ms, 23.3 -> 24.7 tok/s at
  full quality; 99.9 -> 91.3 ms, 27.8 -> 30.4 tok/s at top-4 (`results/2026-11-17-frontier/f1_frontier_speed.json`,
  `coal_*`; `coal_ab_run2.log`). It changes no arithmetic, so quality is unchanged.
- An early ~300 µs "idle-gap penalty" per exchange turned out to be Energy-Efficient Ethernet. Any quiet gap longer
  than the NIC's low-power-idle timer (115 µs) puts the link to sleep, and waking it costs ~150-250 µs. With EEE off,
  90 swaps per cycle took 5.6 ms instead of 15-18 ms (`results/2026-11-16-dualmem/README.md`, section 3;
  `results/2026-11-08-runtime-v1/README.md`). Where EEE must stay on, a software keep-alive byte during idle gaps
  recovers part of it.

**F7. Splitting the linear-attention projections does not pay.** *Measured; specific to this runtime.* On the same
boot, turning the KDA head split off raised throughput from 20.75 to 23.21 tok/s (cycle 138.9 -> 122.8 ms) with no
quality change (`f1_frontier_speed.json`, `base` vs `kdaoff`). One extra exchange per linear-attention layer (34 per
pass) costs more latency than the bandwidth it saves.

**F8. The vocabulary-projection split does pay.** *Measured, 1 rep.* Same boot: 20.07 -> 20.85 tok/s essay
(119.2 -> 114.7 ms per cycle) and 19.74 -> 21.89 tok/s chat (`results/2026-11-17-dualrt/README.md`, section 3,
`m4_g00` vs `m4_g10`). One large exchange that halves a large read is worth it; many small exchanges are not
(compare F7).

**F9. Two benchmark pitfalls: KV-cache size variance and cold first rounds.** *Known classes of problem, stated
because they nearly produced wrong conclusions here.*
- **KV-cache size.** One boot came up with a 230,589-token KV cache instead of 302,193 at the same memory-utilisation
  setting (non-torch memory +1.8 GiB; the cause is unexplained). Long agent prompts were then evicted from the prefix
  cache (hit rate 33% instead of 93%) and re-prefilled, and the lossless anchor fell from 23.21 to 21.33 tok/s. The
  anchor rule rejected the boot; the rerun pinned the cache size with `--kv-cache-memory-bytes`
  (`results/2026-11-17-frontier/README.md`, "Boots and anchor checks"; `f2_frontier_speed.json`). Pin the KV size
  explicitly when comparing boots.
- **Cold first round.** The first point after a boot runs with a cold prefix cache: 20.88 tok/s, then 23.83 and 24.01
  in the next rounds (`f3_frontier_speed.json`, `kdaoff`; f1 showed the same, 18.93 vs ~21.8). Run a throwaway warm-up
  point, or interleave and report rounds separately.

## 6. Limitations and what is not measured

- **Far from the targets.** 25.7 tok/s against a goal of 80 and a control at 151. With 4-bit experts and FP8 fixed
  weights, 80 tok/s exceeds the combined read bandwidth of both machines by about 1.6-1.7x *regardless of placement*
  (derived, `results/2026-11-09-audit/README.md`). Getting there needs fewer bytes per committed token, not only
  better scheduling.
- **TTFT is not measured on the split runtime.** Prefill still sends every expert to the workstation, as in config B.
  Config B measured 994 s for a cold 45K prompt and 1.17 s for cached 45K + 32 (`docs/RESEARCH_PLAN_2026-11-16.md`,
  first table), against targets of 10.8 s and 0.2 s.
- **The recommended point is not confirmed on a clean boot.** On boot f3 the lossless anchor missed f1 by 0.40 tok/s
  pooled (all of it in the cold first round); the other anchor matched within 0.1. A confirmation boot with a warm-up
  point has not been run. Treat absolute f3 numbers as ±0.5 tok/s and its comparisons as same-boot only.
- **Top-7 + coalescing off** (~27 tok/s) is modeled, not measured.
- **Quality is teacher-forced agreement** with a Q8_0 proxy of the BF16 weights, on 41,943 tokens. There are no
  downstream task scores for top-7 yet; the agent-suite scores in `results/2026-11-09-audit/` are for config B.
- **Speed is batch 1, greedy,** on one hardware pair and one link type (TCP over 10GBASE-T).
- **Unexplained:** the KV-size difference between boots (F9), and a race between the hook's 30-second stats-dump
  thread and CUDA-graph capture that failed one boot (`results/2026-11-17-frontier/f2_bootfail_2217/`).
- **The hot set** is calibrated on agent-suite prompts only; the top 120 experts per layer cover 66% of routes
  in-sample (`results/2026-11-17-dualrt/hotset_calib.json`).
- **Not built:** an MLA projection split, a shared-expert split, a split draft layer.
- Two power outages interrupted quality runs. The interrupted partial passes are kept aside and unused
  (`results/2026-11-17-frontier/partial_fq2_pre1653/`, `partial_fq2_pre1919/`).

## 7. Reproduction

The scripts were written for two specific machines and still contain host-specific paths and addresses (some come from
`ops/local.env`, which is not in this repository). Expect to edit paths. The order of operations:

1. **Weights.** From the official BF16 release, build the FP8 fixed-path checkpoint for the Spark-native vLLM build
   (`docs/ROUTE3_FORMATS.md`) and a GGUF with Q4_K + importance-matrix routed experts (llama.cpp tools; importance
   matrix from a calibration split disjoint from the eval set).
2. **Workstation expert server.** Build `runtime/expert_server_gguf.c` (C with OpenMP). Run it bound to the direct-link
   address with 32 pinned threads, active OpenMP wait, and the dense-row pack for the lm_head rows
   (`bench/dualrt/prep_dense.py`).
3. **Link.** Direct 10GBASE-T cable, MTU 9000, static addresses. Record the EEE and interrupt-coalescing state: it moves
   results by 5-10%.
4. **Bandwidth and prototype (optional):** `bench/dualmem/run_overlap.sh`, `cpu_suite.sh`, `link_suite.sh`,
   `run_proto.sh`, `headsplit.py`.
5. **Kernel unit tests:** `bench/dualrt/run_unit.sh`, `run_unit_dense.sh`, `test_q4k.py`, `test_dense.py`,
   `test_drop.py`.
6. **Hot set:** `bench/dualrt/run_calib.sh`, then `hotset.py`.
7. **Speed frontier:** boot once with `bench/dualrt/boot_frontier.sh <tag>`, pinning the KV cache with
   `EXTRA_ARGS="--kv-cache-memory-bytes <bytes>"`. Then run `POINTS_ONLY="..." ROUNDS=3 bash bench/dualrt/run_f1_points.sh`
   and aggregate with `frontier_agg.py`.
8. **Quality:** `bench/dualrt/run_frontier_quality.sh "<name>|<k> <tau> 0|none|1 0 0" ...` (eager, 16-token batches),
   scored with `compare_q8.py` against the Q8 proxy logits; paired comparisons with `paired_quality.py`.

Unit tests for the analysis code: `python3 -m unittest discover -s tests -v`.

## 8. Repository map

- `results/<date>-<topic>/README.md`: one write-up per experiment, raw files beside it. Start with
  `2026-11-17-frontier`, `2026-11-17-dualrt` and `2026-11-16-dualmem`.
- `bench/dualmem/`: bandwidth, link and prototype benchmarks. `bench/dualrt/`: the split-runtime hook, clients and
  scorers. `runtime/`: the workstation expert server.
- `docs/`: protocols and plans (`EXPERIMENT_PROTOCOL.md`, `RESEARCH_PLAN_2026-11-16.md`), literature notes in
  `docs/research/`.

## License and data

Code license: to be chosen before release. No model weights are redistributed here; the official GLM-5.3-Flash model
card lists the weights as MIT. Evaluation and calibration text comes from public datasets and public encyclopedia text
under their own licenses (`results/2026-11-08-balanced-eval/q_eval_balanced_v1.manifest.json`).
