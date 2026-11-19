## gpt-oss-120b (128 experts, top-4, 36 MoE layers, task types: chat, code, math)

Tokens used (calib fit / holdout test): chat 20000/20000 (102 holdout requests); code 20000/20000 (89 holdout requests); math 20000/20000 (104 holdout requests)

- Load skew (mixed tasks, mean over layers): busiest expert 5.18x the mean; top 10% of experts get 28.0% of picks; CV 0.86
- Within-layer co-selection, top-7 partners per expert, measured on holdout: P(partner | expert) 0.212 vs P(partner) 0.089; lift same task 2.97, partners from another task 3.14, mixed->mixed 3.26
  - lift per layer (same task, mean over tasks): min 2.00 (layer 29), max 4.10 (layer 18), first 3.70, last 3.41
- Layer N -> N+1 recall@4: same task 0.474, other task 0.257, mixed 0.457; popularity baseline 0.182; random 0.031
  - mixed fit by phase: prefill 0.450 (20000 tok), decode 0.437 (20000 tok)
  - per layer (same task): min 0.391, max 0.562
- Per-task recall@4 (fit row -> test column):

| fit \ test | chat | code | math |
|---|---|---|---|
| chat | 0.466 | 0.240 | 0.400 |
| code | 0.219 | 0.423 | 0.200 |
| math | 0.368 | 0.117 | 0.534 |

Simulation, holdout replay, mean over task types. sends = cross-GPU sends per token per layer; imb = max/mean GPU load over the whole holdout set; imb_b = same per 2048-token batch.

| GPUs | placement | sends | vs Default | distinct GPUs | imb | imb_b |
|---|---|---|---|---|---|---|
| 8 | random equal-size | 2.930 | -0.6% | 3.349 | 1.544 | 1.563 |
| 8 | DefaultEplbPolicy, mixed load | 2.949 | +0.0% | 3.370 | 1.362 | 1.387 |
| 8 | DefaultEplbPolicy, per-task load | 2.992 | +1.4% | 3.419 | 1.057 | 1.144 |
| 8 | correlated, mixed fit, cap 1.0x | 2.470 | -16.2% | 2.823 | 1.708 | 1.736 |
| 8 | correlated, mixed fit, cap 1.1x | 2.160 | -26.8% | 2.468 | 2.005 | 2.042 |
| 8 | correlated, per-task fit, cap 1.0x | 2.713 | -8.0% | 3.100 | 1.072 | 1.173 |
| 8 | oracle (fit on the test tokens, mixed set only): Default -> correlated | 2.950 -> 2.475 | -16.1% | | 1.003 -> 1.002 | |
| 16 | random equal-size | 3.457 | -0.9% | 3.688 | 1.997 | 2.043 |
| 16 | DefaultEplbPolicy, mixed load | 3.488 | +0.0% | 3.721 | 1.673 | 1.726 |
| 16 | DefaultEplbPolicy, per-task load | 3.523 | +1.0% | 3.759 | 1.189 | 1.318 |
| 16 | correlated, mixed fit, cap 1.0x | 2.948 | -15.5% | 3.144 | 2.072 | 2.153 |
| 16 | correlated, mixed fit, cap 1.1x | 2.741 | -21.4% | 2.923 | 2.302 | 2.389 |
| 16 | correlated, per-task fit, cap 1.0x | 3.069 | -12.0% | 3.273 | 1.238 | 1.436 |
| 16 | oracle (fit on the test tokens, mixed set only): Default -> correlated | 3.483 -> 2.948 | -15.3% | | 1.016 -> 1.015 | |
| 32 | random equal-size | 3.740 | -1.0% | 3.860 | 2.884 | 2.966 |
| 32 | DefaultEplbPolicy, mixed load | 3.777 | +0.0% | 3.898 | 2.240 | 2.353 |
| 32 | DefaultEplbPolicy, per-task load | 3.803 | +0.7% | 3.926 | 2.004 | 2.098 |
| 32 | correlated, mixed fit, cap 1.0x | 3.233 | -14.4% | 3.338 | 2.933 | 3.087 |
| 32 | correlated, mixed fit, cap 1.1x | 3.167 | -16.2% | 3.269 | 3.149 | 3.312 |
| 32 | correlated, per-task fit, cap 1.0x | 3.211 | -15.0% | 3.315 | 2.135 | 2.454 |
| 32 | oracle (fit on the test tokens, mixed set only): Default -> correlated | 3.777 -> 3.225 | -14.6% | | 1.350 -> 1.349 | |

Correlated search: swaps per layer max 110, mean 73 (cap 5000; 0 layers hit the cap)

Analytic distinct GPUs for a random placement: 8: 3.342, 16: 3.680, 32: 3.860

## Qwen3-30B-A3B (128 experts, top-8, 48 MoE layers, task types: chat, code, math)

Tokens used (calib fit / holdout test): chat 20000/20000 (102 holdout requests); code 20000/20000 (89 holdout requests); math 20000/20000 (104 holdout requests)

- Load skew (mixed tasks, mean over layers): busiest expert 3.18x the mean; top 10% of experts get 19.3% of picks; CV 0.55
- Within-layer co-selection, top-7 partners per expert, measured on holdout: P(partner | expert) 0.360 vs P(partner) 0.159; lift same task 3.26, partners from another task 4.31, mixed->mixed 4.22
  - lift per layer (same task, mean over tasks): min 2.41 (layer 30), max 10.95 (layer 47), first 4.94, last 10.95
- Layer N -> N+1 recall@8: same task 0.504, other task 0.238, mixed 0.516; popularity baseline 0.254; random 0.062
  - mixed fit by phase: prefill 0.515 (20000 tok), decode 0.508 (20000 tok)
  - per layer (same task): min 0.446, max 0.605
- Per-task recall@8 (fit row -> test column):

| fit \ test | chat | code | math |
|---|---|---|---|
| chat | 0.493 | 0.390 | 0.326 |
| code | 0.233 | 0.480 | 0.134 |
| math | 0.238 | 0.103 | 0.538 |

Simulation, holdout replay, mean over task types. sends = cross-GPU sends per token per layer; imb = max/mean GPU load over the whole holdout set; imb_b = same per 2048-token batch.

| GPUs | placement | sends | vs Default | distinct GPUs | imb | imb_b |
|---|---|---|---|---|---|---|
| 8 | random equal-size | 4.653 | -1.4% | 5.318 | 1.463 | 1.485 |
| 8 | DefaultEplbPolicy, mixed load | 4.717 | +0.0% | 5.391 | 1.346 | 1.372 |
| 8 | DefaultEplbPolicy, per-task load | 4.858 | +3.0% | 5.552 | 1.058 | 1.139 |
| 8 | correlated, mixed fit, cap 1.0x | 3.513 | -25.5% | 4.014 | 1.931 | 1.952 |
| 8 | correlated, mixed fit, cap 1.1x | 3.090 | -34.5% | 3.531 | 2.263 | 2.273 |
| 8 | correlated, per-task fit, cap 1.0x | 4.346 | -7.9% | 4.967 | 1.075 | 1.181 |
| 8 | oracle (fit on the test tokens, mixed set only): Default -> correlated | 4.721 -> 3.594 | -23.9% | | 1.003 -> 1.003 | |
| 16 | random equal-size | 6.169 | -1.6% | 6.581 | 1.841 | 1.890 |
| 16 | DefaultEplbPolicy, mixed load | 6.270 | +0.0% | 6.689 | 1.638 | 1.687 |
| 16 | DefaultEplbPolicy, per-task load | 6.460 | +3.0% | 6.891 | 1.149 | 1.284 |
| 16 | correlated, mixed fit, cap 1.0x | 4.754 | -24.2% | 5.071 | 2.228 | 2.306 |
| 16 | correlated, mixed fit, cap 1.1x | 4.354 | -30.6% | 4.644 | 2.502 | 2.581 |
| 16 | correlated, per-task fit, cap 1.0x | 5.427 | -13.5% | 5.789 | 1.180 | 1.383 |
| 16 | oracle (fit on the test tokens, mixed set only): Default -> correlated | 6.263 -> 4.791 | -23.5% | | 1.014 -> 1.013 | |
| 32 | random equal-size | 7.129 | -1.4% | 7.359 | 2.570 | 2.652 |
| 32 | DefaultEplbPolicy, mixed load | 7.228 | +0.0% | 7.461 | 2.123 | 2.241 |
| 32 | DefaultEplbPolicy, per-task load | 7.374 | +2.0% | 7.613 | 1.636 | 1.834 |
| 32 | correlated, mixed fit, cap 1.0x | 5.917 | -18.1% | 6.108 | 2.609 | 2.776 |
| 32 | correlated, mixed fit, cap 1.1x | 5.636 | -22.0% | 5.818 | 2.825 | 3.000 |
| 32 | correlated, per-task fit, cap 1.0x | 6.053 | -16.3% | 6.249 | 1.749 | 2.088 |
| 32 | oracle (fit on the test tokens, mixed set only): Default -> correlated | 7.219 -> 5.862 | -18.8% | | 1.050 -> 1.049 | |

Correlated search: swaps per layer max 115, mean 75 (cap 5000; 0 layers hit the cap)

Analytic distinct GPUs for a random placement: 8: 5.339, 16: 6.597, 32: 7.359

## DeepSeek-V2-Lite (64 experts, top-6, 26 MoE layers, task types: chat, code, math)

Tokens used (calib fit / holdout test): chat 20000/20000 (102 holdout requests); code 20000/20000 (89 holdout requests); math 20000/20000 (104 holdout requests)

- Load skew (mixed tasks, mean over layers): busiest expert 1.68x the mean; top 10% of experts get 13.0% of picks; CV 0.18
- Within-layer co-selection, top-7 partners per expert, measured on holdout: P(partner | expert) 0.273 vs P(partner) 0.119; lift same task 2.38, partners from another task 2.24, mixed->mixed 2.48
  - lift per layer (same task, mean over tasks): min 1.98 (layer 13), max 2.90 (layer 4), first 2.40, last 2.76
- Layer N -> N+1 recall@6: same task 0.485, other task 0.346, mixed 0.477; popularity baseline 0.169; random 0.094
  - mixed fit by phase: prefill 0.449 (20000 tok), decode 0.479 (20000 tok)
  - per layer (same task): min 0.411, max 0.561
- Per-task recall@6 (fit row -> test column):

| fit \ test | chat | code | math |
|---|---|---|---|
| chat | 0.486 | 0.396 | 0.436 |
| code | 0.357 | 0.446 | 0.327 |
| math | 0.321 | 0.239 | 0.523 |

Simulation, holdout replay, mean over task types. sends = cross-GPU sends per token per layer; imb = max/mean GPU load over the whole holdout set; imb_b = same per 2048-token batch.

| GPUs | placement | sends | vs Default | distinct GPUs | imb | imb_b |
|---|---|---|---|---|---|---|
| 8 | random equal-size | 3.948 | -0.9% | 4.511 | 1.210 | 1.231 |
| 8 | DefaultEplbPolicy, mixed load | 3.984 | +0.0% | 4.553 | 1.175 | 1.196 |
| 8 | DefaultEplbPolicy, per-task load | 4.004 | +0.5% | 4.576 | 1.046 | 1.090 |
| 8 | correlated, mixed fit, cap 1.0x | 3.189 | -20.0% | 3.644 | 1.233 | 1.260 |
| 8 | correlated, mixed fit, cap 1.1x | 2.970 | -25.5% | 3.394 | 1.345 | 1.373 |
| 8 | correlated, per-task fit, cap 1.0x | 3.147 | -21.0% | 3.597 | 1.062 | 1.121 |
| 8 | oracle (fit on the test tokens, mixed set only): Default -> correlated | 4.006 -> 3.150 | -21.4% | | 1.017 -> 1.015 | |
| 16 | random equal-size | 4.972 | -0.6% | 5.303 | 1.410 | 1.446 |
| 16 | DefaultEplbPolicy, mixed load | 5.002 | +0.0% | 5.335 | 1.315 | 1.357 |
| 16 | DefaultEplbPolicy, per-task load | 5.025 | +0.5% | 5.361 | 1.141 | 1.207 |
| 16 | correlated, mixed fit, cap 1.0x | 4.126 | -17.5% | 4.401 | 1.406 | 1.461 |
| 16 | correlated, mixed fit, cap 1.1x | 3.977 | -20.5% | 4.242 | 1.533 | 1.590 |
| 16 | correlated, per-task fit, cap 1.0x | 4.150 | -17.0% | 4.427 | 1.163 | 1.257 |
| 16 | oracle (fit on the test tokens, mixed set only): Default -> correlated | 5.014 -> 4.148 | -17.3% | | 1.052 -> 1.049 | |
| 32 | random equal-size | 5.561 | -0.8% | 5.741 | 1.796 | 1.865 |
| 32 | DefaultEplbPolicy, mixed load | 5.608 | +0.0% | 5.789 | 1.635 | 1.710 |
| 32 | DefaultEplbPolicy, per-task load | 5.622 | +0.3% | 5.804 | 1.496 | 1.591 |
| 32 | correlated, mixed fit, cap 1.0x | 5.009 | -10.7% | 5.171 | 1.730 | 1.817 |
| 32 | correlated, mixed fit, cap 1.1x | 4.914 | -12.4% | 5.073 | 1.866 | 1.951 |
| 32 | correlated, per-task fit, cap 1.0x | 5.036 | -10.2% | 5.199 | 1.520 | 1.650 |
| 32 | oracle (fit on the test tokens, mixed set only): Default -> correlated | 5.615 -> 5.020 | -10.6% | | 1.173 -> 1.173 | |

Correlated search: swaps per layer max 49, mean 32 (cap 5000; 0 layers hit the cap)

Analytic distinct GPUs for a random placement: 8: 4.536, 16: 5.316, 32: 5.762

