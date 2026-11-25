# Bet 1, dual-memory decode: can both memories be read at once, and what does it buy? (2026-11-16)

Question (the project lead, 2026-11-16 21:09): "Can we read both spark and maxq ram at the same time? Can we test it?"

**Answer: yes, measured.** The Spark GPU and the workstation CPU stream their own memories at the same time with no
measurable interference. Inside a real 90-exchange decode-cycle prototype the two together read **~366 GB/s
effective** (vs 207 GB/s for the GPU alone on the same bytes): **56.7 ms vs 100.1 ms per cycle, 1.77x** [M].

Labels: [M] measured here, [D] derived, [A] assumption. Anything in tok/s is **modeled** from measured parts.
Hardware: Spark GB10 (whisper idle, nothing else on the GPU), workstation Threadripper PRO 9975WX (4 CCDs, 8 x DDR5
configured 5600 MT/s, 1 NUMA node, GPUs idle and unused), direct 10GBASE-T link SPARK_LINK_IP <-> .1 (MTU 9000).

## 1. Simultaneity [M] (`bench/dualmem/stream_*.{c,py}`, `run_overlap*.sh`, `overlap_leg{A,B,C}.txt`)
Coordinated start by epoch time; 1 s windows; per-window epoch stamps prove the overlap.

- Spark GPU (bf16 read-reduce, 2 GiB): **239.7 GB/s alone, 239.8 GB/s while the workstation CPU streams** (leg A).
- Workstation CPU (32 cores pinned, 8 GB buffer): **251.2 GB/s alone, 251.2 GB/s while the Spark GPU and Spark Arm both stream** (leg C).
- Spark Arm CPU (20 threads) + Spark GPU, same LPDDR5X (legs B, C): Arm 101.5 GB/s alone -> **32.8** together;
  GPU 239.7 -> **202.0-202.4** together (-16%). Sum ~235 GB/s, i.e. one shared pool.
  - Controls (legs D, E): 20 busy Arm threads on a 512 KB cache-resident buffer (no DRAM traffic) leave the GPU at
    **238.5 GB/s**, so the drop is memory arbitration, not the GPU launcher losing its core. 16 Arm threads (4 cores
    free) streaming DRAM: Arm 83.9 -> 26.3, GPU 239.7 -> **210.3** (-12%).
  - **This contradicts the M0 note** ("GPU loses nothing", Arm 17-19 GB/s). That used a different Arm kernel (NEON
    expert GEMV) and a 1 GiB GPU buffer, so it is not like for like. With a plain read stream the Arm takes ~26-33 GB/s
    *from* the GPU. So the Spark's Arm cores add no bandwidth during decode; the split must not use them for weight
    streaming.

## 2. Workstation CPU GEMV at split shapes [M] (`bench/dualmem/gemv_cpu.c`, `cpu_suite.sh`, `cpu_suite_*.txt`)
Formats: **fp8** = the checkpoint's own E4M3 weights + E8M0 32-block scales, decoded on the fly to bf16 with the scale
folded into the exponent, `vdpbf16ps` (exact for normal E4M3; `check` rel. L2 7e-8 vs scalar); **bf16** native;
**int8** VNNI as a speed reference only (not the checkpoint format). 32 cores pinned, OMP active wait, a barrier after every
matrix, weights cycled over >= 2 GB of distinct copies (pools under ~0.6 GB were re-run; L3 is 128 MB and inflated them).

| shape (CPU share f = 0.5 unless noted) | T=1 GB/s | T=4 GB/s | T=4 us |
|---|---|---|---|
| KDA attention half: q,k,v [4096x4096] + o [4096x4096], fp8 | 241.5 | 236.4 | 293 |
| KDA f = 0.25, fp8 | 238.9 | 234.9 | 147 |
| KDA small bf16 projections (b, f_a, g_a, f_b, g_b slices) | 177.5 | 160.4 | 29 |
| MLA half: q_a, kv_a, q_b slice, o slice, fp8 | 237.5 | 229.1 | 245 |
| MLA half bf16 parts: kv_b slice, indexer wq_b/wk/weights | 237.0 | 224.1 | 137 |
| dense MLP half (layers 0-2), fp8 | 243.0 | 236.4 | 329 |
| shared expert half / full, fp8 | 255.5 / 249.4 | 246.0 / 243.1 | 53 / 107 |
| lm_head vocab half [77440x4096], fp8 | 250.3 | 245.1 | 1334 |
| bf16 / int8, KDA half shapes | 245.8 / 246.8 | 242.6 / 240.2 | |
| KDA half, 64 threads (SMT) | | 231.4 | |

- **FP8 at the checkpoint's exact format reads at 229-246 GB/s, 91-98% of STREAM.** The plan targeted >= 170 GB/s and
  modeled 170 for everything; that was the Q4_K expert rate (165-175), not a fixed-path rate.
- Inside the prototype (with swaps between phases) the same kernels ran at 212-218 GB/s.

### Per-chiplet bandwidth (decision Q2) [M]
Read stream pinned by L3 group (CCD k = cores 8k..8k+7):
- 1 / 2 / 3 / 4 CCDs, 8 cores each: **63.4 / 124.7 / 187.3 / 251.2 GB/s** (SMT 16/CCD: 61.8 / 122.5 / 182.1 / 239.3).
- FP8 GEMV, same pinning: 61.0 / 119.2 / 172.5 / 235.9 GB/s.
- 4 cores on one CCD: 62.4. 8 cores spread 2 per CCD: 244.9. 16 cores spread 4 per CCD: 248.6 (read stream).
  FP8 GEMV with the same spread: 8 cores 154.7, 16 cores 205.7, 32 cores 235.9 (the E4M3 decode needs cores; the read alone does not).
  M0's "thread scaling roughly linear in cores" was close-pinning onto fewer CCDs.
- So the limit is **~63 GB/s per CCD (the CCD's fabric link), perfectly linear to 4 CCDs**, not DRAM.
  4 x 63 = 251 against a DRAM peak of 8 x 5600 MT/s x 8 B = 358 GB/s [D] (70%).
- Implication, **conditional on [A]: an 8-CCD part keeps the same ~63 GB/s per-CCD link** (link width per CCD can differ by
  SKU; not measured here). Then it would not be CCD-limited (8 x 63 = 504 > DRAM) and would hit the DRAM ceiling: at ~85-90%
  of 358 that is ~305-320 GB/s with these DIMMs, ~350 with DDR5-6400 [A: DRAM efficiency not measured]. That would be +20-40%
  CPU-side bandwidth. Measured: the 9975WX is fabric-limited, not DRAM-limited. Before any purchase, the per-CCD link rate of
  the candidate SKU is the fact to confirm.

## 3. Link exchange loop [M] (`bench/dualmem/phase_swap.c`, `link_suite.sh`, `link_eee_*.txt`)
Symmetric swap per phase (both sides compute, send their partial, receive the other's), 90 phases per cycle, 150 cycles,
busy-poll spin recv (mode 1) unless noted. Overhead = per-phase time minus compute.

| NIC state | 8 KB (T=1 bf16) | 32 KB (T=4 bf16) | 64 KB (T=4 fp32) |
|---|---|---|---|
| defaults (EEE on), compute 150-300 us | 158-193 us | 166-199 us | 260-270 us |
| defaults + 20 us heartbeat, compute 50-300 us | 62-110 us | 98-122 us | 119-177 us |
| **EEE off** (both NICs), compute 150-300 us | 34-38 us | 62-63 us | |
| **EEE off + workstation RX coalescing off** | 30 us | 50 us | 79 us |
| blocking recv (no busy-poll), NIC defaults | 200-304 us | 246-303 us | 284-363 us |
| blocking recv, EEE off (with or without coalescing) | 144-211 us | 87-209 us | 107-226 us |

- **Surprise: the ~300 us "idle-gap penalty" from 2026-11-08 is Energy-Efficient Ethernet.** Both 10GBASE-T NICs had EEE
  active (Spark r8127 Tx LPI timer 115 us). Any quiet gap longer than that puts the link into low-power idle, and waking it
  costs ~150-250 us. With EEE off the overhead is flat in compute time and no heartbeat is needed.
- 90 swaps per cycle at T=4: **4.5 ms (EEE + coalescing off), 5.6 ms (EEE off), 15-18 ms (defaults)**. The plan expected <= 10 ms.
- RDMA: `ibv_devinfo` finds no device on either side (the Spark's CX-7 is still off the bus), so TCP busy-poll is the floor.
- NIC settings were changed only for these runs and restored afterwards (EEE on, LPI 115/0 us, adaptive RX on, rx-usecs 50).
  Making EEE off permanent is a one-line `ethtool --set-eee <if> eee off` per host (not done).

## 4. One-layer-split prototype, run as a full 90-phase cycle [M] (`bench/dualmem/dual_srv.c`, `dual_cli.py`, `run_proto.sh`, `proto_*.txt`)
Each side runs its half of every layer on its own memory, then the two swap a 4 x 4096 bf16 partial over the link.
- Spark: torch `_scaled_mm` FP8 (W8A8, T padded to 16) in eager mode, `cuda.synchronize` + pinned D2H per phase, Python busy-poll socket.
- Workstation: the exact-format FP8 GEMV above, 32 threads, C busy-poll socket.
- Phase A = KDA attention half (4 x [4096x4096], 67 MB per side). Phase B = shared-expert half + 6 FP8 expert matrices
  (164 MB per side). The 6 experts stand in, byte for byte, for ~11.5 Q4_K experts (the U4 = 22.7 union split by expert).
  10 distinct layer copies per side; 45 layers = 90 phases = 20.8 GB per cycle in total. Random weights: this is a timing
  prototype; the head-split exactness test (plan step 4) is not done.

| run | cycle ms (p50) | effective GB/s | swap+D2H per phase |
|---|---|---|---|
| GPU only, both halves' bytes (no network) | 100.1 | 207 | (sync+D2H 9 us) |
| GPU only, one half (no network) | 51.0 | 204 | |
| workstation CPU only, one half (server compute) | 50.3 [D: 45 x (323 + 795 us)] | 213 | |
| **split, EEE off + RX coalescing off** | **56.7** | **366** | 50 + 10 us |
| split, EEE off | 58.4 | 356 | 62 + 10 us |
| split, NIC defaults (EEE on) | 69.0 | 301 | 195 + 10 us |

- **Measured speedup from reading both memories: 1.77x (EEE + coalescing off), 1.71x (EEE off), 1.45x (defaults)** on the same bytes.
- `proto_full_t32_transitional_nic.txt` (67.6 ms) was taken while the NICs were half-restored (coalescing off; EEE reported
  "enabled - inactive" just before and "active" again minutes later, so its EEE state during the run is unverified); kept, not used.
- **Scope of the prototype:** it emulates one verify pass (45 layers, T = 4) with FP8 stand-in experts and no draft passes.
  The 1.77x is a memory-read speedup on equal bytes, not a tok/s gain; tok/s is modeled in section 5.
- The composition rule used below (per-phase max of the two sides + per-phase swap) predicts 55.8 ms for this run against
  56.7 ms measured (within 2%) [D].
- GPU-side cost per exchange: sync + 8-32 KB D2H ~10 us; the eager launch of 4-21 large kernels per phase still streamed at
  204 GB/s. A real layer has many small kernels and currently runs as one CUDA graph; 90 graph breaks per cycle are not
  measured with the real model (sensitivity +10 us per phase below).

## 5. Measured-input decode estimate (MODELED composition) (`src/dualmem_compose_2026_11_16.py`, `compose.json`)
Config-B quality (FP8 fixed path, Q4_K experts, MTP-3, prose acceptance 2.217 [M]). Inputs: GPU 201 GB/s fixed / 195 experts
[M], CPU 215 GB/s FP8 fixed [M, in-prototype] / 170 Q4_K experts [M, 11-09], per-phase swap 60 / 72 / 205 us [M], bytes from
headers [M]. Every phase balanced; 91 phases (45 x 2 + lm_head).

- **EEE off, drafts on the GPU only: ~26-28 tok/s** (cycle 78.5-85.3 ms; 0-10% imbalance; +0/10 us per graph break).
- **EEE off, draft passes split too: ~28-31 tok/s** (71.8-78.6 ms).
- NIC defaults (EEE on): 22.5-25.7 tok/s.
- Same inputs, config B structure: 15.5 tok/s, against **15.3-19.5 measured** (sanity check on the inputs).
- Baselines: config B 15.3-19.5 tok/s [M] at quality (+0.0069). GPU-only 27-28 tok/s [M] fails quality (+0.18).
- **Against the plan's model (R2 27, R3 30 tok/s):** the result agrees. The parts moved in opposite directions: the CPU reads
  fixed-path shapes faster than modeled (215-246 vs 170 GB/s), the swap costs less than modeled with EEE off (60-72 vs
  110 us), but the draft passes (14 ms of ~80) stay GPU-bound unless they are split too.
  RDMA (R3) would save at most ~4-5 ms per cycle on top of EEE off; the swap is no longer the problem.

## Not measured / open
- Head-split exactness on real weights (plan step 4); the KDA recurrence and MLA KV reads on the CPU side; expert-ownership
  imbalance per step (the GPU can hold only a hot subset of the 171 GB Q4_K expert set, so per-step balance needs dynamic
  assignment over replicated hot experts) [A]; CUDA-graph segmentation cost in the real runtime.
- The DRAM efficiency an 8-CCD part would reach (only the per-CCD linearity is measured).

## 6. Step 4: head-split exactness on real weights [M] (`bench/dualmem/headsplit.py`, `hf_gemv_lib.c`, `headsplit_compare.json`, `headsplit_controls.txt`)
Real FP8 weights of KDA layer 4 and MLA layer 3 (full_iq2xxs_fp8: MXFP8 E8M0-32 for KDA, FP32 128x128 block scales for MLA,
BF16 kv_b and small projections). Inputs: 36 tokens, RMS-normalised embedding rows of fixed pseudo-random token ids
(not real layer residuals) [A]. KDA runs token by token from a zero state (conv, bounded gate lb = -5, L2-normed q/k, delta
rule, sigmoid-gated o_norm), following the vLLM glm5next code. MLA runs causal attention over the 36 tokens; DSA top-k
2048 selects all of them, so the indexer is not exercised.
- Spark GPU: heads 0-31 (torch). Workstation CPU: heads 32-63 through the C kernels on the checkpoint's own bytes
  (MXFP8 + a new FP32-block-scale FP8 kernel + BF16). Partials are summed; replicated ops (f_a/g_a; q_a, kv_a + norms) run on both sides.
- "rt" = runtime-like numerics: bf16 activations at op boundaries, fp32 accumulation and recurrence. "ref" = fp32 throughout.

| rel. L2 over [36, 4096] | KDA L4 | MLA L3 |
|---|---|---|
| split vs unsplit, both fp32 (is the decomposition exact?) | 1.1e-6 | 7.8e-7 |
| GPU-only rt vs fp32 ref (the runtime's own error) | 5.17e-3 | 4.08e-3 |
| split rt vs fp32 ref, fp32 partial exchange | 5.17e-3 | 4.14e-3 |
| split rt vs fp32 ref, bf16 partial exchange | 5.48e-3 | 4.51e-3 |
| split rt vs GPU-only rt, fp32 / bf16 exchange | 2.34e-3 / 2.86e-3 | 2.48e-3 / 2.97e-3 |
| control: GPU + GPU split (no CPU) vs GPU-only rt | 2.34e-3 | 2.45e-3 |
| CPU C half vs GPU half, same heads | 6.8e-4 | 1.16e-3 |

- **The split is exact in math (1e-6) and as accurate as the GPU-only path** against the fp32 reference: 1.00x (KDA) and 1.01x
  (MLA) of the GPU-only error with fp32 partials; 1.06x / 1.11x with bf16 partials.
- **The pre-stated criterion "split vs GPU-only <= 1e-3" is NOT met (2.3-3.0e-3)**, but the control shows why: splitting the same
  heads across two GPU runs, with no CPU involved, gives the same 2.3-2.5e-3. It is the bf16 rounding of the outputs
  (partials rounded separately vs one rounded sum), not the CPU kernels. The criterion was below bf16 output resolution.
  Proposed replacement: split error vs the fp32 reference <= 1.1x the GPU-only error. Met with fp32 exchange; bf16 exchange is
  at the edge for MLA (1.11x).
- Recommendation: exchange fp32 partials (64 KB at T = 4; measured swap 79-82 us vs 50-62 us at 32 KB with EEE off, about +2.5 ms
  per cycle), or verify bf16 exchange end to end on dNLL. Per-layer error is not the quality metric; the balanced-eval dNLL is.

## 7. Split draft passes, prototype [M] (`bench/dualmem/run_draft.sh`, `proto_draft3_default_nic.txt`)
Three MTP draft passes at T = 1, NIC at defaults (EEE on; the project lead has not decided on EEE). Per pass and per side:
- phase A = MTP attention half plus eh_proj half, with q_a/kv_a replicated (105 MB);
- phase B = lm_head vocab half plus shared-expert half plus one FP8 expert as a byte stand-in for the MTP experts (355 MB).

Real FP8 kernels on both sides; FP8 byte stand-ins for the BF16 kv_b / eh_proj halves. Same bytes as the 0.95 GB draft pass.

| run | 3 draft passes, ms (p50) |
|---|---|
| GPU only, same bytes | 12.7 (215 GB/s) |
| GPU half only | 6.6 |
| workstation half only (server compute) | 5.9 [D: 3 x (476 + 1481 us)] |
| **split, NIC defaults** | **7.5** (1.69x; swap 122 us per 8 KB phase) |

- One swap per pass is not modeled: the lm_head phase was merged into phase B, so 6 phases ran instead of 9. That adds about 3 x 0.13 ms → ~7.9 ms [D].
- The section-5 model assumed 8.7 ms at defaults and 7.4 ms with EEE off, so the measurement is slightly better than modeled.
- **Updated measured-input estimate with split drafts:**
  - NIC defaults: 77.6 + 7.9 = 85.5 ms per cycle → **~25.9 tok/s** (24.2 with drafts on the GPU only).
  - EEE off: ~31 tok/s (modeled from the measured EEE-off swap).
