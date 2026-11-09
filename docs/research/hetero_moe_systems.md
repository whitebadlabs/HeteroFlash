# Heterogeneous CPU-GPU MoE inference: literature + code review (for GLM-5.3-Flash on GB10 + remote Zen5 over 10 GbE)

Compiled 2026-11-08. "M" = the paper measured it. "C" = the authors claim it, or it is a headline number whose baseline is weak or unclear. "E" = my own back-of-envelope estimate for our setup.

## 0. Our numbers recomputed (E), so the transfer judgments have a scale

- Routed expert = 3 x 4096 x 2048 = 25.17M params. 42 layers x 288 = 12,096 experts = **304B params**. That leaves only about 6B non-expert params out of ~310B.
- Q4_K (4.5 bpw) = 14.2 MB per expert, **~171 GB** for all of them.
- Decode, one layer: 8 experts x 14.2 MB = 113 MB in 0.47 ms = **241 GB/s, about 96% of the 250 GB/s measured on the workstation**. The CPU decode expert kernel is already at the memory-bandwidth roofline. Better CPU kernels cannot speed it up; only reading fewer bytes (precision, placement) or overlap can.
- The GB10 reading the same 113 MB at 273 GB/s takes at least 0.41 ms. **GPU and CPU have about the same bandwidth (273 vs 250 GB/s).** This is the key difference from every PCIe paper below, where GPU HBM/GDDR is 3-10x faster than host DRAM.
- Token budget at 13.4 tok/s: 74.6 ms. CPU experts take 42 x 0.47 = 19.7 ms (26%). The 10 GbE round trips are at least 42 x ~50 us = 2.1 ms. That leaves ~50 ms for the GPU path plus synchronization. Reading the ~6B non-expert params once in FP8 takes about 22 ms at 273 GB/s, plus the shared experts (~1B params). So **roughly 20-25 ms of the token time is probably mailbox/sync/launch overhead, not bandwidth**. This needs a timeline to confirm. KTransformers measured the same failure mode: Fiddler spent 73% of GPU time on kernel launches, and CUDA Graph capture brought overhead from >20% to ~0 ([KT paper](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf), §2.3/§6.2).
- Prefill, all on the CPU: 45K tokens / 994 s = 45 tok/s. Routed compute is 42 x 8 x 2 x 25.17M = 16.9 GFLOP/token, so the CPU runs at about **0.77 TOPS effective**. Zen5 AVX-512 VNNI peak is about 32 cores x 256 int8 ops/cycle x ~4 GHz ≈ 33 TOPS (E, from public microarchitecture data; not measured). Prefill is far from the compute roofline, unlike decode.

## 1. Systems

**Fiddler** (Kamahori et al., ICLR'25; [arXiv 2402.07033](https://arxiv.org/abs/2402.07033), [code](https://github.com/efeslab/fiddler)).
- Algorithm: runs a CPU-resident expert on the CPU and sends only activations, instead of copying weights over PCIe. A per-expert latency model decides at runtime: GPU latency is a constant, CPU latency is linear in the number of tokens. The expert's weights are copied to the GPU only if `cpu_lat(s) > gpu_lat + transfer_lat`, so long prefills move weights to the GPU ([v3 HTML](https://arxiv.org/html/2402.07033v3)).
- Placement: the most popular experts are put on the GPU from an offline ShareGPT profile. This only gives 25.2% vs 21.9% (random) hit rate with 56/256 experts on the GPU, and 53.0% vs 48.8% with 125 (M).
- **Overlap: none.** The CPU-GPU-CPU path is sequential.
- Hardware: Quadro RTX 6000 on PCIe3, and RTX 6000 Ada on PCIe4 with a Xeon 8480+. Model: Mixtral-8x7B fp16.
- Results: 1.26x average tokens/s over llama.cpp b2956, 1.07x TTFT over DeepSpeed-MII, 11.57x on beam search (M, against old baselines).
- Quality: not evaluated. The paper says it is exact.

**KTransformers** (Chen et al., SOSP'25; [paper](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf), [code](https://github.com/kvcache-ai/ktransformers), [SGLang integration](https://lmsys.org/blog/2025-10-22-KTransformers)).
- Hardware (M): 2x Xeon 8452Y (36 cores each, 220 GB/s per socket, 125 GB/s cross-socket), A100-40G or RTX 4080, PCIe4, batch 1. Models: DS-V3 (17B on GPU, 654B on CPU), DS-V2.5, Qwen2-57B. Attention and shared experts run on the GPU, routed experts on the CPU.
- Kernels: weights are repacked into an AMX-tile layout (Int8/Int4 with separate scales, 64-byte aligned). An AVX-512 kernel takes over when arithmetic intensity is ≤4, i.e. in decode.
- Kernel throughput (M): 21.3 TFLOPS per socket on DS-3 MoE, versus 5.4 TFLOPS for PyTorch/oneDNN AMX and 1.8 for AVX-512. Theoretical AMX peak is 73.7.
- Other optimizations: NUMA-aware tensor parallelism, and the whole decode step captured as one CUDA Graph, with the CPU calls embedded via host functions.
- **Expert Deferral** (decode only): the top experts of layer k are "immediate" and feed layer k+1's attention input. The remaining "deferred" experts run on the CPU while the GPU does layer k+1 attention, and their outputs are added to the residual one layer later. This changes the model.
  - Timeline (M, DS-3 BF16): without deferral, CPU utilization 74%, GPU 28%, overlap only 5%, sync/transfer 3% idle. With 3 deferred experts, layer time −26% and end-to-end decode +33%.
  - Configurations used: 5+3 for DS-3 BF16, 2+6 for Int4.
  - Up to +45% on top of the other optimizations. Overall decode is 1.66-2.56x llama.cpp; llama.cpp was patched by the authors to support expert-level offload.
  - Quality (M): DS-3 8+0 → 2+6 gives HumanEval 83.0→83.0, MBPP 71.2→70.2, GSM8K 94.8→95.2, StrategyQA 83.0→82.9, plus LiveBench. The claim is "within 0.5%" on average.
- Prefill: AMX GEMM on the CPU. 4.62-19.74x over the baselines for 32-8192 tokens (C). Absolute values are only in Fig. 11.
- **Newer kt-kernel**: `--kt-gpu-prefill-token-threshold` switches long prompts to **layerwise GPU prefill**, which streams expert weights to the GPU per layer and needs 9 GB+ extra VRAM ([Kimi-K2 doc](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/kt-kernel/Kimi-K2-Thinking-Native.md), [issue 1608](https://github.com/kvcache-ai/ktransformers/issues/1608)).
  - Other kt-kernel flags: `--kt-num-gpu-experts N` places hot experts on the GPU, and `--kt-max-deferred-experts-per-token` exposes deferral in the SGLang path.
  - A third-party GLM-5.3-Flash tutorial uses `--kt-method FP8`, a 2048-token GPU-prefill threshold and 14 GPU experts on 2x RTX PRO 6000. It reports ~11 tok/s (C, secondary; [DataCamp](https://www.datacamp.com/zh/tutorial/ktransformers-tutorial)).

**OSDI'26 "Cloud-grade SLOs for local MoE"** (Wang et al.; [arXiv 2606.10493](https://arxiv.org/abs/2606.10493), [PDF](https://www.usenix.org/system/files/osdi26-wang-wenxin.pdf)). This is the closest prior art on prefill.
- Hardware (M): 2x EPYC 9355 (1.23 TB/s theoretical DDR5) + RTX 5090 on PCIe5.
- **Stream-loading prefill**: loader, compute and unloader threads/streams, a ring buffer of expert weights, and sub-layer events. Reaches 1,200 tok/s, and 32K tokens in under 30 s. With two 5090s (DSLP) it reaches 1,800 tok/s.
- The CPU FP8/AVX path is used up to 4K tokens and streaming beyond that. They note CPU MoE saturates above ~512 tokens ([HTML](https://arxiv.org/html/2606.10493)).
- Decode: an AVX-512 FP8 GEMV runs at 947 GB/s effective. Results are 21.5 tok/s (FP8 R1), 22.4 (Kimi-K2), 28 tok/s (INT4), versus their reruns of KTransformers ~22 and ik_llama ~14 tok/s.
- Its attention/MoE overlap is **dual-batch**: about 350 us of attention overlaps about 450 us of MoE per layer. It **needs ≥2 concurrent requests** and does nothing at batch 1.
- Quality (M): MMLU-Redux/Pro within 1-1.2 points of the official scores.
- Code: not stated.

**Hecate** ([github.com/third-party/hecate](https://github.com/third-party/hecate)). Self-reported, 16 commits. This is GLM-5.3, the 753B/256-expert model, not Flash.
- Each MoE layer's experts are split by popularity across three devices: 144 hot experts in int4 GPTQ on the GB300, 85 mid experts in 3-bit EXL3 on an RTX PRO 6000 over PCIe (no P2P, pinned-memory mailboxes), and 27 cold experts in int8 NEON on 72-core Grace, coherent over C2C.
- **The tiers run concurrently within each layer.** The RTX band finishes after the hot band in 39% of layers.
- Ablation gains: RTX coprocessor +22%, Grace cold tier +11%, one CUDA graph for the whole decode-verify forward +4.1%, pipelined decode +5.4%. ~184 tok/s at concurrency 1 (C; the repo text elsewhere says 145-170).
- Prefill: 8K chunks. The RTX band's weights are read from a pinned Grace copy over C2C (C2C bandwidth not stated in the README). ~4,800 tok/s.
- Quality (M): KLD 0.0735 vs 0.1013 for the stock EXL3 3.0 bpw quant on the same 18 windows (36,848 positions). The author calls this preliminary.
- **The design (joint precision + placement per popularity tier, profiled per layer) is very close to what we would build.** The interconnects differ: C2C and PCIe, against our Ethernet link.

**HybriMoE** (DAC'25; [arXiv 2504.05897](https://arxiv.org/html/2504.05897), [code](https://github.com/PKU-SEC-Lab/HybriMoE), built on KTransformers).
- Hardware (M): A6000 + Xeon 5220R limited to 10 cores.
- **Intra-layer split**: GPU queue = cached experts by load descending. CPU queue = uncached experts by load ascending, and the CPU steals cached experts when idle. High-load uncached experts are transferred over PCIe. A greedy simulation minimizes max(CPU, GPU) time.
- Prefetch uses the gates of the next 3 layers, ranked by simulated benefit. Cache policy is an EMA of routing scores.
- Results: 1.33x prefill and 1.70x decode over KTransformers (M). Qwen2 ablation: decode 0.21→0.11 s/token.
- Quality: none reported. That is fine, because the method is exact.

**DALI** ([arXiv 2602.03495](https://arxiv.org/html/2602.03495v1)).
- Same idea as HybriMoE: a 0-1 assignment with a greedy heuristic (92% of optimal), plus "residual-corrected" next-layer gate prefetch.
- Measured at batch 8-128 on a 3090 + EPYC 16 cores: decode 1.32x HybriMoE and 2.16x KTransformers.
- Prefetch accuracy is notable: **naive next-gate top-1 accuracy for high-load experts on DeepSeek-V2-Lite is only 36.7-40.6% (HybriMoE method)**.
- Not batch 1, no quality numbers.

**DAOP** (DATE'25; [arXiv 2501.10375](https://arxiv.org/html/2501.10375)).
- Hardware (M): i9-10980XE + A6000 on PCIe4. Models: Mixtral and Phi-3.5-MoE.
- **Predictive pre-calculation**: applies layer i+1's gate to layer i's post-attention hidden state (84.1% one-ahead accuracy, from layer 4 on). The predicted CPU experts are computed early on that approximate input, so their output is an approximation.
- Graceful degradation replaces the second CPU expert with a GPU-resident one.
- Results (M): +35-40% decode over Fiddler (4.52 tok/s Mixtral).
- **Quality hit**: GSM8K 58.9→33.5 on Mixtral at 25% cache, 86.9→74.1 on Phi.

**Pre-gated MoE** (ISCA'24; [arXiv 2308.12066](https://arxiv.org/html/2308.12066)).
- Block k's gate is fine-tuned to select block k+1's experts, which requires fine-tuning (2,048 steps).
- Models: Switch-Base/Large on an A100 + PCIe4. Results (M): 1.7x lower latency than on-demand fetch, 19% overhead vs GPU-only.
- One-block lookahead gave the best accuracy; looking further ahead hurt.

**HOBBIT** ([arXiv 2411.01433](https://arxiv.org/html/2411.01433)).
- Mechanism: on a cache miss, a less important expert is loaded in low precision. The cumulative gate-weight threshold T1=0.6 selects low precision and T2=0.9 skips the expert. On Mixtral this gives 67% high / 30% low / 3% skipped.
- Pairings: fp16+int4 on a 4090, int8+int2 on a Jetson Orin. Next-layer prefetch top-1 accuracy is ~96% on Mixtral (top-2 of 8).
- Results (M): 1.31-1.42x llama.cpp and 0.99-1.46x Fiddler on the 4090+CPU. Headline 9.93x vs MoE-Infinity on Orin.
- Quality: GSM8K/TruthfulQA drop ≤1 point.

**ProMoE** ([arXiv 2410.22134](https://arxiv.org/html/2410.22134), [code](https://github.com/promoe-opensource/promoe)).
- A learned per-layer MLP predictor (~2M params, 200 us on the CPU) gives 84.7% accuracy versus 58.3% for token-based prediction.
- Hardware (M): 4090 + PCIe4. Results: 2.20x prefill and 2.07x decode against LRU/static caches in transformers and llama.cpp. llama.cpp layer offload beat it on Mixtral decode.

**Fate** ([arXiv 2502.12224](https://arxiv.org/html/2502.12224v2)): cross-layer gate prefetch with 97.15% accuracy, but on coarse-expert models.

**MoE-Infinity** ([arXiv 2401.14361](https://arxiv.org/abs/2401.14361)): request-level activation traces drive caching and prefetch at batch 1. 3.1-16.7x vs vLLM/Ollama/DeepSpeed offload (C, weak baselines).

**AdapMoE** ([arXiv 2408.10284](https://arxiv.org/abs/2408.10284)): sensitivity-based adaptive gating cuts activated experts by 25% for 1.35x, "without accuracy degradation" (C).

**EdgeMoE** ([arXiv 2308.14352](https://arxiv.org/abs/2308.14352), code in [mllm](https://github.com/UbiquitousLearning/mllm)): per-expert bitwidth chosen offline, plus preloading from flash.

**SwapMoE** ([arXiv 2308.15030](https://arxiv.org/abs/2308.15030)): a dynamic "virtual expert" subset. Switch Transformer 14.2→4.7 GiB at −0.041 Rouge-2. This is an approximate method.

**Throughput-only systems, not batch-1-relevant**:
- MoE-Lightning ([arXiv 2411.11217](https://arxiv.org/abs/2411.11217)): CGOPipe + hierarchical roofline, T4, 10.3x.
- MoE-Gen ([arXiv 2503.09716](https://arxiv.org/abs/2503.09716)): module-based batching, 8-31x.
- Klotski ([arXiv 2502.06888](https://arxiv.org/abs/2502.06888)): multi-batch pipeline, 85x.
- Their gains come from amortizing weight transfers over large batches.

**HetRoute** ([arXiv 2608.00577](https://arxiv.org/abs/2608.00577)): geo-distributed edge servers.
- Offline: placement + GPU/CPU residency + **per-replica precision** under a quality-penalty budget. Online: routes the whole top-k set to minimize the bottleneck cost.
- Evaluation is trace-driven on Switch/Qwen-A2.7B/Mixtral, up to 59% lower latency (C).
- This is the conceptual neighbor for "joint precision + placement", but it is simulated, uses small models, and does not handle decode critical-path overlap.

**llama.cpp / ik_llama.cpp practice**:
- `-ot exps=CPU` / `--cpu-moe` keeps the routed experts on the CPU.
- **Op offload**: when a batch has ≥32 tokens (`GGML_OP_OFFLOAD_MIN_BATCH`), the CPU-resident expert weights are *copied to the GPU per op* for prefill ([HF guide](https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide); `--no-op-offload` in the [server README](https://raw.githubusercontent.com/ggml-org/llama.cpp/HEAD/tools/server/README.md)).
- ik_llama.cpp uses a threshold of 32 x total/active experts. For us that would be 32 x 288/8 = 1,152 tokens (E).
- This is weight streaming in all but name, and it is designed for PCIe.
- Community DS-R1 numbers on ik_llama: ~40 tok/s prompt processing, ~14 tok/s decode ([discussion 477](https://github.com/ikawrakow/ik_llama.cpp/discussions/477)). This is anecdotal.

## 2. Answers

### Q1. Hiding a CPU expert path that sits on the decode critical path

Four mechanisms exist:

- **(a) Cross-layer deferral** (KTransformers).
- **(b) Splitting each layer's experts across devices concurrently** (HybriMoE, DALI, Hecate, kt `--kt-num-gpu-experts`).
- **(c) Early or approximate pre-calculation of the next layer's experts on the CPU** (DAOP), or fine-tuned pre-gating (Pre-gated MoE).
- **(d) Dual-batch overlap** (OSDI'26). This needs ≥2 requests.

Prefetch prediction (HOBBIT, ProMoE, Fate, MoE-Infinity) hides *weight transfer*, not CPU compute.

Best reported numbers at batch 1:
- **KTransformers: GPU+CPU overlap went from 5% to the CPU being fully saturated. Layer time −26%, end-to-end +33% (DS-3 BF16), up to +45%.** In their regime the CPU is the bottleneck (74% CPU vs 28% GPU), so what deferral hides is the GPU and sync time behind the CPU work. The 26% reduction is about 93% of the 28% that could be hidden (E).
- Hecate: RTX experts fully hidden behind GB300 experts in 61% of layers. The Grace tier adds +11%.
- DAOP: +35-40% over Fiddler, but approximate (GSM8K loss).
- HybriMoE: 1.86x decode in one ablation, versus a weaker KTransformers configuration.

**Direct answer: none of the surveyed systems reports hiding CPU expert time at batch 1 in a GPU-dominant regime like ours.** KTransformers hid ~93% of the *GPU and sync* time behind a CPU that was the bottleneck. Hecate hides its mid-tier experts in 61% of layers.

**For us**, the CPU expert time (~19.7 ms) is smaller than the GPU path (~50 ms), so deferral would hide the *CPU and network* behind GPU work. Deferral also hides the 50 us RTT, which no PCIe paper had to deal with. Estimates (E):
- Bound if everything were hidden: 74.6 → ~52 ms (1.43x). That would mean zero immediate experts, which KTransformers says breaks stability.
- At their most aggressive validated configuration (2 immediate + 6 deferred), only 6/8 of the CPU time can be deferred: ~60 ms, **~1.25x**.

KTransformers validated deferral only on DeepSeek and Qwen, always with at least 2 immediate experts. The quality cost has to be re-measured on GLM-5.3-Flash.

### Q2. Prefill with CPU-resident experts

The strategies seen:
- GPU computes after moving weights per expert or per op: Fiddler's latency model, llama.cpp op offload at ≥32 tokens.
- CPU AMX GEMM: KTransformers paper, 21.3 TFLOPS/socket.
- Layerwise or ring-buffer weight streaming: kt-kernel layerwise prefill; OSDI'26 SLP at 1,200 tok/s on PCIe5 + 5090.
- Hecate: GPU reads cold weights over coherent C2C, about 4,800 tok/s.

**Every streaming design was built for ≥32-64 GB/s links. On our 10 GbE link it still works for long prompts.** The right comparison is the quality-passing all-CPU Q4 prefill (994 s), not the 2-bit GPU path (39 s, which fails quality). Estimates (E):
- **Layer-major streaming** (kt layerwise / OSDI'26 SLP pattern): send each layer's Q4_K blob (288 x 14.2 MB = 4.1 GB, which fits easily in a GPU ring buffer) once, run the whole prompt through that layer on the GPU, discard it.
  - Cost is about fixed per prefill: 171 GB / 1.1 GB/s ≈ **155 s, quality identical to the decode weights**. That is ~6x faster than today's 994 s.
  - It only works if the prompt is processed layer-major, all tokens per layer. Chunked prefill would re-stream the weights once per chunk.
  - It needs Q4_K GEMM kernels on sm_121.
  - Hot experts kept resident on the GPU shrink the streamed bytes proportionally.
- **Crossover with today's CPU path.** CPU cost scales with prompt length (994 s / 45K ≈ 22 ms per token); streaming is fixed. The crossover is 155 / 994 x 45K ≈ **7K tokens**. This is our analogue of kt-kernel's `--kt-gpu-prefill-token-threshold`. ik_llama's 32 x 288/8 = 1,152 heuristic would be wrong for us.
- **Per-expert view, two regimes.** Transfer costs 12.9 ms per expert.
  - At the measured 0.77 TOPS, one token-expert takes ~65 us, so break-even is ~200 tokens. A 45K prompt routes ~1,250 tokens to each expert, so streaming wins ~6x.
  - At a hypothetical well-tuned ~8 TOPS Zen5 VNNI GEMM (E, unmeasured; KTransformers got 1.8 TFLOPS with PyTorch AVX-512 on Xeon and 21.3 with AMX, which Zen5 lacks), a token-expert takes 6.3 us. Break-even rises to ~2,000 tokens, CPU-only 45K drops to ~95 s, and the crossover moves to ~45K.
  - On PCIe5 (~50 GB/s) break-even would be ≈45 tokens.
- **Combined option.** Run streaming and CPU concurrently: stream some experts per layer while the CPU computes the rest. With a better CPU kernel this could land at ~60 s for 45K at full Q4 quality, versus 39 s for the 2-bit path (E).
- **CPU-path activation floor.** Each layer sends 45K x 4096 x 2 B = 369 MB each way: 14 s per prefill with full duplex overlapped, 28 s without, 7-14 s with FP8 activations (E). Streaming avoids this cost.
- Third option: GPU low-bit prefill (39 s) followed by high-bit decode. See Q5.

### Q3. Precision per expert or per device

- **HOBBIT**: dynamic, per cache-miss expert, chosen by gate-weight rank. ≤1-point loss on GSM8K and TruthfulQA.
- **EdgeMoE**: offline per-expert bitwidth.
- **HetRoute**: per-replica precision under a quality-penalty budget, simulated.
- **Hecate**: per tier/device (int4 / 3-bit trellis / int8), with layer-causal GPTQ calibration and KLD vs BF16 as the metric.
- **KTransformers**: per device in practice (CPU Int4/Int8 kernels, GPU-side quants), evaluated on HumanEval/MBPP/GSM8K/StrategyQA/LiveBench.
- **OSDI'26**: native FP8 on the CPU, evaluated on MMLU-Redux/Pro.

Evaluation quality is weak across the set: mostly 2-4 short benchmarks with small n. None uses per-expert sensitivity measured on long-horizon agentic tasks. **Hecate's same-window KLD with a CI is the most rigorous.**

### Q4. What transfers to our topology

Transfers:
- **Per-layer concurrent split.** Hot experts at Q4 stay resident in the Spark's 128 GB, cold experts go to the CPU. The split should be proportional to *bandwidth* (273:250), not the GPU-dominant split PCIe papers use.
  - Ideal concurrent per-layer expert time is 113 MB / 523 GB/s ≈ 0.22 ms vs 0.47 ms (E). The real gain is smaller because GPU-side expert reads compete with the non-expert reads in the same LPDDR.
  - The Spark's own Arm CPU cores share the same 273 GB/s, so using them adds no bandwidth.
- **Expert Deferral**, to hide CPU time and RTT behind GPU attention.
- **Whole-step CUDA-graph capture with mailbox flags** (KTransformers, Hecate). Our ~20-25 ms of unexplained time suggests this is the first thing to fix (E).
- **Popularity profiling.** Fiddler showed that offline popularity gives weak hit gains for coarse experts. Fine-grained 288-expert routing is likely more skewed, but that needs measuring.

- **Layer-major weight streaming for long-prompt prefill**, with a ~7K-token threshold at today's CPU speed (Q2). Per-op streaming in llama.cpp's op-offload style does not transfer: it re-streams per ubatch at a 32-token threshold.

Does not transfer:
- Decode-time weight movement of any kind.
- Cache-miss expert fetch and prefetch: HOBBIT, ProMoE, MoE-Infinity, Pre-gated, AdapMoE. Over 10 GbE, a 14.2 MB fetch costs 12.9 ms, about 27x one layer's entire CPU expert time.
- Dual-batch overlap.
- Throughput-batching systems.
- HybriMoE/DALI's "transfer high-load uncached expert" queue.
- Prediction-based schemes are also risky for us on accuracy: high-load expert prediction on DeepSeek-V2-Lite is only 37-41% (DALI Table 2).

### Q5. What would be novel for us

Hecate (popularity tiers x per-tier precision x concurrent per-layer split) and HetRoute (placement x replica precision under a quality budget) already cover the *concept* of joint precision + placement. A genuine contribution would need to rest on:

1. **Bandwidth parity across a high-latency link.** Placement is set by bytes-per-device, which is ~1:1. Execution has to schedule around a 50 us RTT per layer (deferral depth as a tunable). No surveyed system targets a remote CPU over Ethernet.
2. **Precision-asymmetric prefill and decode.** Prefill all experts on the GPU in low bit (or hot experts Q4 + cold 2-bit) to build the KV cache, then decode with Q4 split across GPU and CPU. **None of the surveyed systems does this.** All of them prefill and decode with the same weights. It must be tested for KV-mismatch quality loss.
   - The 128 GB capacity forces a joint choice between "full low-bit copy for prefill" and "more hot Q4 experts on the GPU for decode". That joint choice is exactly the optimization problem.
3. **Profiling-guided optimization for one model** (KDA/MLA-DSA hybrid, MTP-3 verify, which raises the number of unique experts per step). The objective would be an end-to-end measured per-token timeline plus a KLD/long-task quality gate, rather than generic cache heuristics.
4. **Quality-aware deferral selection.** Choose which layers and how many experts to defer from measured per-layer sensitivity. KTransformers uses one fixed count for all layers.

Caveats on the evidence:
- Most speedups are reported against weak or self-modified baselines: llama.cpp b2956, llama.cpp patched by the KTransformers authors, HybriMoE capped at 10 CPU cores, DALI at batch ≥8.
- Several headline figures are internally inconsistent: Hecate 184 vs 145-170 tok/s; KTransformers at 16 vs 22 tok/s in the OSDI'26 paper.
