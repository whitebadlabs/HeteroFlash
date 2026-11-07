# M0 partial calibration: 2026-11-06

First measurements on the real machines. They replace guessed inputs in `configs/synthetic_demo.json`.
Routing is still synthetic, and nothing here is an SLA prediction.

## Inventory (from `src/hardware_probe.sh`, addresses omitted)
| | Spark | Workstation |
|---|---|---|
| CPU | 20 Arm cores (Cortex-X925 + Cortex-A725) | AMD Threadripper PRO 9975WX, 32 cores / 64 threads, AVX-512 (F/BW/VNNI/BF16/VBMI2), no AMX, 1 NUMA node |
| Memory | 121 GiB unified LPDDR5X visible, ~107 GiB in use by another model during the test | 246 GiB DDR5 RDIMM, 8 channels populated (16 x 16 GB ranks), ~240 GiB available |
| GPU | GB10, driver 580.178.04 | 2 x RTX PRO 6000 Blackwell Max-Q (97,887 MiB each), driver 615.71.09. **Not used for inference.** |
| NICs | One RJ45 port linked at **1000 Mb/s**; Wi-Fi down. **The ConnectX-7 QSFP interfaces do not appear in /sys/class/net.** | 10GBASE-T port 0 linked at **1000 Mb/s**; 10GBASE-T port 1 unused (no link) |
| RDMA | `ibv_devinfo` not installed | `ibv_devinfo` not installed |

Action items: find out why the Spark's CX7 interfaces are missing (driver, firmware, or naming); install `rdma-core` / `perftest` on both machines.

## Memory bandwidth (`bench/mem_bw.c`, read only)
| Machine | Threads | GB/s |
|---|---|---|
| Workstation | 8 / 16 / 32 / 64 | 246 / 250 / 251 / 240 |
| Spark (Arm CPU side, 2 GB buffer, shared with a running GPU model) | 4 / 10 / 20 | 84 / 93 / 100 |

The Spark GPU side was not measured here because it would disturb the live model. The published GB10 STREAM-triad figure is ~200-213 GB/s.
The Arm and GPU share one memory interface, so their bandwidth figures must not be added together.

## Workstation CPU expert kernel (`bench/moe_cpu_vnni.c`, 32 threads pinned to cores)
| Format | Experts x tokens (distinct) | ms per layer | Weight GB/s |
|---|---|---|---|
| int8 | 1 x 1 | 0.122 | 206 |
| int8 | 8 x 1 | 0.926 | 217 |
| int8 | 3 x 4 (12) | 1.420 | 213 |
| int4 (naive unpack) | 1 x 1 | 0.106 | 118 |
| int4 (naive unpack) | 8 x 1 | 0.784 | 128 |
| int4 (naive unpack) | 3 x 4 (12) | 1.185 | 127 |

- int8 runs at about 85% of measured DRAM bandwidth: roughly 0.12 ms per distinct expert per token.
- The naive int4 kernel is compute/unpack-bound and only ~13% faster than int8. A tuned int4 kernel (for example kt-kernel RAWINT4) is the next thing to measure.
- With 16 threads every number is ~1.7x slower. Reserving cores for research jobs directly costs CPU-tier speed.

## Spark GPU (`bench/spark_gpu_bench.py`, vllm/vllm-openai:v0.32.0, torch 2.13 cu130, GB10 sm_121; idle co-resident model, 2.2 GB allocated)
| Test | Result |
|---|---|
| Read-reduce, 1 GiB bf16 | **241 GB/s** |
| Copy (read + write) | 233 GB/s |
| Expert gate+up GEMV [T x 4096] @ [4096 x 4096], weights cycled over 24 experts | BF16 T1 162 GB/s; BF16 T4/T8 222 GB/s; FP8 T16/T32 222-224 GB/s |
| Prefill GEMM [T x 4096] @ [4096 x 2048] TFLOPS | BF16 22 / 74 / 88 / 96 / 102 at T 64 / 256 / 1024 / 4096 / 8192 |
| | FP8 82 / 127 / 193 / 192 / 195 |
| | NVFP4 (vLLM cutlass_scaled_fp4_mm) 227 / 252 / 297 / 300 at T 256 / 1024 / 4096 / 8192 |

- Weight streaming of ~222 GB/s beats the published 200-213 GB/s STREAM figures. The calibration value moves from 170 (assumed) to 222 (measured).
- Single dense NVFP4 GEMMs reach ~300 TFLOPS at prefill sizes. Grouped MoE GEMM with per-expert token counts has not been measured yet.

## Tuned CPU expert kernel v2 (`bench/moe_cpu_vnni_v2.c`; verified exact vs scalar reference by `bench/check_v2.c`)
4-row register blocking; int4 as unsigned nibbles with the zero point folded into one per-row correction.

| Format | Experts x tokens (distinct) | Threads | ms per layer | us per expert | Weight GB/s |
|---|---|---|---|---|---|
| int8 | 1 x 1 | 32 | 0.110 | 110.4 | 228 |
| int8 | 8 x 1 | 32 | 0.842 | 105.2 | 239 |
| int8 | 3 x 4 (12) | 32 | 1.221 | 101.8 | 247 |
| int8 | 24 x 1 | 32 | 2.428 | 101.2 | 249 |
| int4 | 1 x 1 | 32 | 0.061 | 60.6 | 208 |
| int4 | 8 x 1 | 32 | 0.425 | 53.1 | 237 |
| int4 | 3 x 4 (12) | 32 | 0.631 | 52.6 | 239 |
| int4 | 24 x 1 | 32 | 1.254 | 52.3 | 241 |
| int4 | 8 x 1 | 16 | 0.822 | 102.8 | 122 |
| int8 | 8 x 1 | 16 | 1.605 | 200.7 | 125 |

- int4 v2 is 1.87x faster than v1 (0.425 vs 0.795 ms for 8 experts) and is now DRAM-bound: about 52-61 us per distinct expert, ~96% of the measured 251 GB/s read bandwidth.
- Not yet included: dynamic activation quantization (bf16 -> u8 + scale), per-group weight scales, SiLU(gate)*up and the down-projection input requantization. These need measuring before the per-expert cost goes into the planner.

## Spark <-> workstation link (`bench/tcp_pingpong.py`, current 1 GbE switched path)
| Payload each way | Median RTT | p90 |
|---|---|---|
| 64 B | 0.477 ms | 0.813 ms |
| 8 KB (one 4096 x bf16 hidden state) | 0.538 ms | 0.640 ms |
| 16 KB | 0.607 ms | 0.698 ms |
| 64 KB | 1.441 ms | 1.564 ms |

ICMP ping average was 1.195 ms (min 0.59). Published Spark data suggests most small-message latency comes from CPU idle states; this needs to be tested on these machines.

## Placement simulator rerun (`src/placement_sim.py`)
| Config | Spark GPU BW | Arm BW | Remote CPU BW | RTT | Proxy ms / target pass |
|---|---|---|---|---|---|
| `synthetic_demo.json` (original guesses) | 160 | 30 | 45 | 0.25 ms | 38.845 |
| `measured_2026-11-06_tcp_lan.json` | 170 | 100 | 200 | 0.54 ms (measured) | 29.058 |
| `measured_2026-11-06_rdma_assumed.json` | 170 | 100 | 200 | 0.01 ms (assumed) | 21.122 |

The proxy covers routed-expert traffic only. It leaves out attention, KDA/MLA projections, shared experts and lm_head (~9B parameters read every step), the drafter, and kernel overheads.
**Do not compare it with the 8 ms/token decode target.**

## Model facts read from the served checkpoint's config.json
- `glm5_next` with 45 layers: first 3 dense, then 42 MoE layers; 288 routed experts, top-8, 1 shared expert; hidden 4096; moe_intermediate 2048 (25.2M params per expert).
- MLA: q_lora 1536, kv_lora 512, qk/v head dim 256, rope dim 0. DSA indexer: 32 heads x 128, topk 2048. 1 MTP layer. Upstream config also has 34 KDA linear-attention layers + 11 DSA layers and mHC hc_mult 4.
- Served checkpoint: EXL3 mixed 3/4-bit routed experts (146 GB on disk). KV ~17 KB/token as reported by the serving engine.

## Reference point (old serving setup on the two Max-Q GPUs, before research mode)
122-132 tok/s decode, 45K cold prompt 10.8 s, cached second turn 0.2 s (vLLM + EXL3 3.5 bpw, TP2). These are the SLA targets.

## Next M0 steps
1. Spark GPU bandwidth and grouped-GEMM microbenchmarks at the real expert shapes. Needs a window with the Spark GPU free.
2. Tuned int4 CPU kernel (kt-kernel AVX-512 RAWINT4) on the workstation.
3. RTT with CPU idle states disabled; then 10G and CX7/RoCE links once cabled (`ib_write_lat`).
4. Spark Arm expert kernel (NEON int8) while measuring contention against GPU traffic.

## Link latency with deep CPU idle states disabled on both machines (`ops/*-heteroflash.sh`, temporary)
| | Default idle states | Deep idle states off |
|---|---|---|
| ICMP ping avg | 1.195 ms | **0.347 ms** (min 0.339, mdev 0.009) |
| TCP 64 B median | 0.477 ms | **0.211 ms** (p90 0.308) |
| TCP 8 KB median | 0.538 ms | 0.522 ms (p90 0.538) |
| TCP 16 KB median | 0.607 ms | 0.607 ms |
| TCP 64 KB median | 1.441 ms | 1.372 ms |

- Idle-state exit latency was most of the small-message RTT. Activation-sized messages (8-16 KB) are now dominated by 1 GbE serialization through the switch (~0.13 ms per hop each way), so the next step is a 10G/ConnectX link.
- RDMA tools installed on both machines; `ibv_devinfo` finds no RDMA devices on either.
- Spark ConnectX-7: the mlx5 drivers are loaded, but no ConnectX PCI function is present on the bus. The kernel log shows all four functions (0000:01:00.0/.1, 0002:01:00.0/.1) unloading on 2026-10-18 21:46 and never coming back. The QSFP ports are unusable until this is fixed (PCI rescan / reboot / firmware check).

## Complete CPU expert FFN v3 (`bench/moe_cpu_expert_v3.c`; `check` mode: relative L2 error 2.0e-5 vs scalar reference)
Dynamic per-token u8 activation quantization, int4 weights with an fp32 scale per 128-weight group (+ folded zero-point constant; 4.5 bpw incl. metadata), gate/up -> SiLU(gate)*up -> requantize -> down, gate-weighted sum over experts. 32 threads pinned to cores unless noted. The M1 trace capture was running on the same machine (GPU-bound, light CPU load).

| Variant | Experts | ms per layer | us per expert | GB/s (weights + scales) |
|---|---|---|---|---|
| per-expert barriers | 8 | 0.640 | 80.0 | 177 |
| per-expert barriers, 24 threads | 8 | 0.843 | 105.4 | 134 |
| per-expert barriers, 16 threads | 8 | 1.100 | 137.5 | 103 |
| layer-batched | 2 | 0.171 | 85.3 | 166 |
| layer-batched | 4 | 0.324 | 81.0 | 175 |
| layer-batched | 8 | 0.588 | 73.5 | 193 |
| layer-batched | 12 | 0.878 | 73.1 | 194 |
| layer-batched | 24 | 1.765 | 73.5 | 192 |
| layer-batched, 16 threads | 8 | 1.100 | 137.4 | 103 |

- Planner input: **~73.5 us per distinct cold expert per layer** (complete FFN, 32 cores); about 1.4x the bare int4 dot product (52 us). The remaining overhead is per-group scale/correction work (next optimization: apply scales per 2 groups, vectorize corrections across rows).
- Thread scaling is roughly linear in cores, so CPU-tier capacity is set by how many workstation cores research jobs leave free.

## Spark Arm expert kernel and GPU contention (`bench/moe_arm_neon.c`, `bench/gpu_stream_load.py`)
NEON SDOT kernel (int8, and int4 as unsigned nibbles with zero-point fold), 4-row blocking; dot kernels verified exact vs scalar. Built with `-march=armv8.4-a+dotprod` (GCC 13 `-mcpu=native` does not enable dotprod here). Pool of 48 experts, 8 experts per layer, threads pinned to cores.

| Condition | int8 us/expert (GB/s) | int4 us/expert (GB/s) |
|---|---|---|
| Arm alone, 20 threads | 162.3 (155) | 86.7 (145) |
| Arm alone, 10 threads | 297.5 (85) | 156.2 (81) |
| Arm, 20 threads, **while the GPU streams memory** | 1498.6 (**17**) | 663.1 (**19**) |
| GPU read bandwidth during the same window | 243 GB/s, unchanged from GPU alone (242-243) | |

- GB10 memory arbitration strongly favours the GPU. Under full GPU streaming, Arm cores get ~8x less bandwidth while the GPU loses nothing.
- **Implication:** a concurrent Spark-Arm expert tier is ineffective during bandwidth-bound decode (the GPU is streaming nearly the whole step). It might help only in phases where the GPU is compute-bound (prefill GEMMs) or idle (network waits). Treat it as a hypothesis to test, not a planned tier.
- The earlier Arm STREAM figure (~100 GB/s) was measured with a co-resident model loaded. Arm-alone weight streaming reaches 145-155 GB/s.

## Served checkpoint byte breakdown (`src/checkpoint_bytes.py`, safetensors headers; EXL3 TR3 3.5bpw mixed)
| Category | GB | Read every decode step? |
|---|---|---|
| routed experts (EXL3 K3/K4) | 133.62 | only the selected ones |
| attention projections (KDA + MLA, native BF16) | 11.95 | yes |
| MTP layer | 3.55 | per draft step, if MTP is used |
| shared experts | 2.11 | yes |
| lm_head | 1.27 | yes |
| embeddings | 1.27 | no (row gather) |
| vision tower | 1.13 | no (text-only) |
| dense MLP (layers 0-2) | 0.91 | yes |
| DSA indexer | 0.16 | yes |
| router | 0.10 | yes |
| norms / hyper-connections | 0.07 | yes |
| **total** | **156.15** | |

- **Fixed bytes per decode step at served precision: ~16.6 GB** (attention + shared + lm_head + dense + indexer + router + norms). At the measured 222 GB/s Spark expert-streaming rate that is **~75 ms per step before any routed expert**. This replaces the earlier 9-10 GB estimate.
- The same fixed weights at FP8 would be ~8.3 GB (~37 ms) and at NVFP4 (~4.5 bpw) ~4.7 GB (~21 ms). Fixed-weight precision is therefore the largest single decode lever on the Spark, ahead of expert placement. Its quality cost must be measured (M5).

## Spark ConnectX-7 after reboot (2026-11-07)
- At boot the kernel enumerates all four functions (15b3:1021 at 0000:01:00.0/.1 and 0002:01:00.0/.1). mlx5_core loads them (firmware 28.45.4028, PCIe Gen5 x4 = 126 Gb/s available), then reports **"Port module event: Cable unplugged"** and "insufficient power on the PCIe slot (27W)". After boot the functions are gone from the PCI bus. The ports power down because **no QSFP cable is connected**; the hardware is not faulty.
- An RDMA path to the workstation needs a QSFP cable **and** a QSFP NIC in the workstation (its ports are 10GBASE-T RJ45 only), i.e. a purchase (not approved).
- Free 10 Gb/s option: the Spark's RJ45 is a 10GbE Realtek 8127 (enP7s7, linked at 1G only because of the 1 GbE switch), and the workstation has a spare 10GBASE-T port. A direct Cat6 cable gives 10 Gb/s point-to-point. The Spark's Wi-Fi (wlP9s9) is up and could carry its LAN traffic, or a 10G switch keeps both on the LAN.
