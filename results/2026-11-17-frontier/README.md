# Frontier: trading accuracy for tok/s (task 35, 2026-11-17)

Carmen, 07:06: "Try to decrease some accuracy to increase tok/s as much as possible."

Setup: GLM-5.3-Flash dual-memory split runtime (Spark GPU + MaxQ CPU experts over the direct 10G link), spec mode MTP-3, NIC defaults, batch 1.
Every lever below is a run-time switch on one boot, so speed points are interleaved on the same server.

- **Speed**: all 15 requests (essay + 4 chat + 10 Hermes) pooled, 3 interleaved rounds. Measured tok/s.
- **Quality**: balanced eval (41 chunks, 41,943 tokens) in agree mode, scored vs the Q8_0 BF16 proxy. Measured.
- **Control bar** (dual Max-Q EXL3 served control vs the same proxy): **dNLL +0.0148, KLD 0.042, top-1 94.0%**.
  A point is inside only if all three pass; "at the bar" = within noise (dNLL SE ~0.002, top-1 SE ~0.12 pt).

## Frontier (all measured)

| point (KDA split off unless noted) | tok/s | dNLL (SE) | KLD | top-1 | vs bar |
|---|---|---|---|---|---|
| base, KDA split on | 20.8 (f1) | +0.0044 (0.0014) | 0.027 | 95.0% | inside (lossless) |
| kdaoff | 23.2 (f1); 22.8 (f3) | lossless (same math) | | | inside |
| kdaoff + MaxQ i40e coalescing off | 24.7 (f1, ABBA) | lossless | | | inside |
| **tau0.04** (drop routed experts with weight < 0.04) | **24.3 (f3)** | +0.0055 (0.0014) | 0.030 | 94.9% | **inside** |
| **k7** (top-7 of 8 experts) | **25.6 (f3)** | +0.0099 (0.0016) | 0.037 | 93.9% | **at the bar** (top-1 0.1 pt under) |
| k7 + tau0.04 | 25.8 (f3) | +0.0112 (0.0017) | 0.040 | 93.8% | at the bar (top-1 0.2 pt under) |
| top-6 | 25.8 (f1) | +0.0181 (0.0020) | 0.059 | 92.4% | just outside |
| tau0.08 | 26.4 (f1); 26.3 (f3) | +0.0313 (0.0022) | 0.077 | 91.4% | outside |
| top-5 | 27.1 (f1) | not measured | | | (worse than top-6) |
| top-4 | 27.6 (f1) | +0.0886 (0.0031) | 0.156 | 86.9% | far outside |
| top-4 renormalised | | +0.1234 | 0.174 | 86.6% | far outside |
| top-6 + tau0.08 | 27.6 (f1) | not measured | | | outside (both parts are) |
| top-4 + Q3_K on 15 CPU layers | 28.0 (f1) | not measured | | | outside |
| top-4 + coalescing off | 30.4 (f1, ABBA) | = top-4 | | | far outside |

Speed files: `f1_frontier_speed.json`, `f3_frontier_speed.json`. Quality files: `q_fq1_*_vs_q8_bal.json`, `q_fq2_*_vs_q8_bal.json` (per-token arrays in `*_pertoken.npz`).

## Boots and anchor checks

- **f1** (11-17 08:36-13:37 PT): the reference speed boot.
- **f2** (11-17 22:36 PT): **failed the anchor check**. kdaoff came in at 21.33 vs 23.21. Its KV cache came up 230,589 tokens vs f1's 302,193: non-torch memory was +1.8 GiB at the same 0.84 utilisation. Hermes prompts were evicted from the prefix cache and re-prefilled (hit rate 33%). Points kept, not used.
- **f3** (11-17 23:42 PT): same config with `--kv-cache-memory-bytes 8181829386`, which pins f1's 302,193 tokens. Prefix hit rate 93%.
  - Anchors, pooled over 3 rounds: tau0.08_kdaoff 26.30 vs f1 26.39 (ok). kdaoff 22.81 vs 23.21, a 0.40 miss.
  - The kdaoff miss is all round 1. kdaoff ran first on a fresh boot with a cold prefix cache: 20.88, then 23.83 and 24.01. f1's first point (base) showed the same cold-round dip, 18.93 vs ~21.8.
  - Rounds 2-3 only: kdaoff 23.9, tau0.08 26.4, k7 25.7, tau0.04 24.3, k7tau0.04 25.6.
  - Read the f3 numbers as same-boot comparisons; absolute values are good to about ±0.5 tok/s.
- The first f2 boot attempt failed at CUDA-graph capture. The hook's 30 s stats-dump thread raced the capture. That race is in existing code (f1 got lucky); logs are in `f2_bootfail_2217/`.

## Recommendation

- **Best point at the control bar: k7 + KDA split off, 25.6 tok/s measured.** That is +10% over lossless kdaoff on the same boot (22.8, or 23.2 on f1).
  - dNLL +0.0099 and KLD 0.037 are inside the bar.
  - top-1 93.9% vs 94.0% is within one SE.
- **Strictly inside the bar on every metric: tau0.04, 24.3 tok/s** (+6.5% same boot). Its quality is almost the base's.
- k7 + tau0.04 adds nothing over k7: +0.1 tok/s, within noise, for slightly worse quality.
- Add the lossless lever, MaxQ i40e interrupt coalescing off.
  - Measured: +1.4 tok/s on kdaoff, +2.6 on top-4.
  - Not yet measured with k7. **Modeled** k7 + coalescing off: ~27 tok/s.
  - Making it permanent is a NIC setting on the MaxQ, so it needs Carmen's OK.
- Everything faster than ~26 tok/s costs quality past the bar: top-6, tau0.08, top-5, top-4. This frontier is flat. Within the bar, accuracy-for-speed buys about +10%, nowhere near the 80 tok/s goal. The remaining gap is the per-cycle floor (~100-115 ms with ~2.9 tokens committed per cycle): exchange latency plus CPU expert time.

## Outages

- Power cuts at 16:53 and 19:19 PT.
- Partials from cut passes are kept aside and unused: `partial_fq2_pre1653/` (tau0.08, 16/41 chunks), `partial_fq2_pre1919/` (k7, 16/41).
- The resumed passes are in `fq2b.log`.
