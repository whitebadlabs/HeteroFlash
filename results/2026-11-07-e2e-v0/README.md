# E2E-v0: GLM-5.3-Flash running end to end on Spark GPU + workstation RAM (2026-11-07)

**First working deployment of the target topology.** Built from the owner's official BF16 weights (zai-org/GLM-5.3-Flash-BF16 rev f9fed3b): BF16 -> GGUF (llama.cpp converter, 642 GB) -> Q8_0 for attention, shared experts, dense MLPs, embeddings and output; Q4_K for routed experts (177 GB, 4.63 bpw). Runtime: mainline llama.cpp (31424b7) with CUDA on the Spark (CUDA 13.0, sm_121) and `ggml-rpc-server` on the workstation CPU (32 threads, AVX-512), 1 GbE link. Placement: everything on the Spark GPU except the routed-expert tensors of MoE layers 24-44, which live in workstation RAM and are computed by its CPU. No speculative decoding. Measured with the same harness as the Control (`src/ttft_bench.py --engine llamacpp`); every TTFT trial passed its cache-state check.

| Hard target | E2E-v0 measured | Control (dual Max-Q) | Gap |
|---|---|---|---|
| Prose decode 120-130 tok/s | **6.46 tok/s** (TPOT p50 156.436 ms, p95 165.729 ms; chunk ITL p95 172.739 ms) | 151 tok/s | ~19x |
| Cold 45K TTFT <= 10.8 s | **838 s** p50 (n=3, ~54 input tok/s) | 11.10 s | ~78x |
| Cached 45K + 32 TTFT <= 0.2 s | **3.10 s** p50 (n=3) | 0.156 s | ~15x |
| Cached 45K + 128 (reported) | 5.59 s p50 | 0.201 s | |

Load: 13.6 min on first start with a warm workstation tensor cache (the first attempt was OOM-killed at 122 GB because an experiment vLLM server still held ~130 GB of workstation RAM; fixed by stopping it). The rpc-server held ~175 GB RSS.

## Where the time goes (estimates from the design measurements; not yet profiled per kernel)
- **Whole-layer placement over 1 GbE.** Each of the 21 remote MoE layers is a separate RPC round trip carrying the hidden state both ways, plus llama.cpp's RPC protocol overhead (several commands per graph split). At ~0.5 ms+ per round trip that is tens of ms per token before any compute. In prefill, activations for 45K tokens cross the 1 GbE link at every remote layer boundary (~15 GB at bf16, ~2 min at 1 GbE alone).
- **CPU-side prefill:** the routed experts of 21 layers for 45K tokens on the CPU (~0.38 PFLOP) take minutes on 32 Zen5 cores without AMX.
- **No speculative decoding** in this run (the Control uses MTP x3: 2.43 committed tokens/verify).
- **Generic kernels:** llama.cpp's glm5next path on GB10 is not tuned for the KDA/DSA/MLA mix (cf. community single-Spark GGUF runs ~15 tok/s).

## What closes which part (HeteroFlash components, measured or built)
| Gap | HeteroFlash piece | Status |
|---|---|---|
| per-layer RPC overhead | activation-only expert server, one message per layer, multi-token batched (`runtime/expert_server.c`) | built + verified; 0.5 ms/layer RTT on 1 GbE measured |
| 1 GbE transfers | direct 10G Cat6 (Spark 10GbE RJ45 <-> workstation spare 10G port); ConnectX needs a QSFP cable + NIC | free option identified |
| fixed-weight bytes | FP8 all fixed weights (+0.006 nats) | quality measured |
| expert placement | hot experts on the Spark by measured usage (M1), cold in RAM at int8/int4 | data + kernels ready |
| cold prefill | GPU-centric prefill with expert staging (G4); needs the faster link | not started |
| decode cycle | top-6 + original-top-8 normalisation (+0.060 nats); drafter saturates at ~2.7 committed tokens (G5) | measured |

The hard targets are **not** met. E2E-v0 is the working baseline of the deployment topology that every later HeteroFlash variant is measured against.
