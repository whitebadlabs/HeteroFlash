# G3: remote CPU expert calls with real network synchronization (2026-11-07)

Setup: `runtime/expert_server.c` on the workstation (32 cores pinned, int4 g128 experts, synthetic weights, 48 resident experts x 42 layers = 28.5 GB; selftest rel L2 2.0e-5 vs scalar reference). `runtime/expert_client_bench.c` on the Spark: 42 sequential per-layer requests per emulated verify step, 20 steps per cell. Link: current 1 GbE switched path, deep CPU idle states disabled on both ends. The M2 sweep was running on the workstation's GPUs at the same time (light CPU load).

| tokens/verify | remote experts/layer | layer RTT p50 | p95 | server compute p50 | network p50 | step (42 layers) p50 |
|---|---|---|---|---|---|---|
| 1 | 1 | 464 us | 548 | 149 | 315 | 20.1 ms |
| 1 | 2 | 488 | 763 | 218 | 272 | 21.3 |
| 1 | 4 | 857 | 920 | 343 | 515 | 36.3 |
| 1 | 8 | 1077 | 1142 | 582 | 494 | 44.7 |
| 1 | 16 | 1497 | 1630 | 989 | 522 | 62.6 |
| 4 | 1 | 1154 | 1190 | 158 | 993 | 48.1 |
| 4 | 2 | 1204 | 1251 | 236 | 967 | 50.6 |
| 4 | 4 | 1384 | 1468 | 411 | 969 | 58.4 |
| 4 | 8 | 1706 | 1852 | 737 | 969 | 71.7 |
| 4 | 16 | 2229 | 2492 | 1270 | 966 | 94.0 |
| 6 | 1 | 1253 | 1300 | 168 | 1080 | 52.6 |
| 6 | 2 | 1286 | 1595 | 271 | 1019 | 55.3 |
| 6 | 4 | 1705 | 1787 | 462 | 1248 | 71.6 |
| 6 | 8 | 2080 | 2229 | 840 | 1240 | 87.3 |
| 6 | 16 | 2737 | 2992 | 1497 | 1246 | 115.0 |
| 8 | 1 | 1716 | 1797 | 185 | 1530 | 69.0 |
| 8 | 2 | 1830 | 1901 | 295 | 1537 | 76.5 |
| 8 | 4 | 2042 | 2128 | 513 | 1538 | 85.6 |
| 8 | 8 | 2437 | 2580 | 912 | 1532 | 102.1 |
| 8 | 16 | 3165 | 3444 | 1637 | 1531 | 133.2 |

Findings
- **Multi-token batching works:** 16 experts cost 0.99 ms for 1 token and 1.50 ms for 6 tokens (weights streamed once per expert per request).
- **The 1 GbE link is the dominant cost for verify windows:** network time is ~1.0 / 1.25 / 1.53 ms per layer at 4 / 6 / 8 tokens (bf16 activations out and partial sums back = 16 KB per token per layer). Over 42 layers that alone is 40-65 ms per step, more than the 40 ms cycle budget. A 10G link (~10x less serialization) is the minimum; the Spark's ConnectX-7 (RoCE) is the target.
- **The CPU branch must be kept small:** at 6 tokens, 8 remote experts per layer cost 0.84 ms compute, ~35 ms over 42 layers if every layer is remote. To hide under the GPU branch the window's remote union must be ~3-4 experts per layer, or overlapped (deferral). This is the measured basis for expert-union regularisation in router training.
- **Fixed per-call overhead ~150 us** (barriers, activation quantization, bf16 conversion) at 1 expert: next optimization target (persistent thread team, fused quant, fewer barriers).
