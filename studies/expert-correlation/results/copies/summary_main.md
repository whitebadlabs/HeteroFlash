
## copies_glm.json.ckpt  (E=288, layers=42)

Simulated replay on holdout. 'tasks' = mean over task types for sends/imb; imb_b p50/p95 over full 2048-token batches pooled across task types (full batches per set: {'code': 1, 'instruction_following': 1, 'knowledge_qa': 1, 'long_context': 14, 'math': 1, 'multilingual': 1, 'writing_chat': 2, 'mix': 26}). 'mix' = mix holdout (task-ordered concatenation, so its batches are mostly single-task). vs (a) = vs DefaultEplbPolicy at the same R.

| G | R | policy | sends tasks | vs (a) | imb tasks | imb_b p50/p95 tasks | sends mix | vs (a) | imb mix | imb_b p50/p95 mix | kappa p50/p95 mix | kappa_slot p95 mix | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 0 | default | 4.656 | +0.0% | 1.254 | 1.333/1.467 | 4.641 | +0.0% | 1.163 | 1.325/1.493 | 3.035/5.086 | 5.086 |  |
| 8 | 0 | corr_static | 3.857 | -17.2% | 1.765 | 2.438/2.979 | 3.707 | -20.1% | 1.799 | 2.287/2.958 | 3.035/5.086 | 5.086 |  |
| 8 | 0 | anti | 4.992 | +7.2% | 1.141 | 1.180/1.217 | 4.967 | +7.0% | 1.062 | 1.172/1.253 | 3.035/5.086 | 5.086 | intra-cosel x0.79, swap cap hit 0 layers |
| 8 | 1 | default | 4.473 | +0.0% | 1.239 | 1.343/1.466 | 4.508 | +0.0% | 1.161 | 1.307/1.481 | 3.035/5.086 | 4.869 | dup copies 12 |
| 8 | 1 | corr_static | 3.710 | -17.0% | 1.743 | 2.421/2.960 | 3.598 | -20.2% | 1.793 | 2.273/2.945 | 3.035/5.086 | 4.896 |  |
| 8 | 1 | corr_dynamic | 3.706 | -17.1% | 1.510 | 2.166/2.670 | 3.569 | -20.8% | 1.672 | 2.031/2.656 | 3.035/5.086 | 3.934 | sees the batch |
| 8 | 1 | anti | 4.750 | +6.2% | 1.131 | 1.187/1.242 | 4.775 | +5.9% | 1.072 | 1.186/1.264 | 3.035/5.086 | 4.842 | intra-cosel x0.80, swap cap hit 0 layers |
| 8 | 2 | default | 4.367 | +0.0% | 1.229 | 1.308/1.459 | 4.404 | +0.0% | 1.165 | 1.307/1.484 | 3.035/5.086 | 4.691 | dup copies 27 |
| 8 | 2 | corr_static | 3.636 | -16.7% | 1.716 | 2.401/2.965 | 3.535 | -19.8% | 1.787 | 2.249/2.950 | 3.035/5.086 | 4.824 |  |
| 8 | 2 | corr_dynamic | 3.605 | -17.4% | 1.389 | 1.951/2.387 | 3.487 | -20.8% | 1.548 | 1.806/2.388 | 3.035/5.086 | 3.275 | sees the batch |
| 8 | 2 | anti | 4.618 | +5.8% | 1.123 | 1.175/1.254 | 4.647 | +5.5% | 1.077 | 1.177/1.286 | 3.035/5.086 | 4.674 | intra-cosel x0.81, swap cap hit 0 layers |
| 8 | 9 | default | 3.923 | +0.0% | 1.166 | 1.245/1.367 | 3.966 | +0.0% | 1.130 | 1.240/1.392 | 3.035/5.086 | 4.501 | dup copies 176 |
| 8 | 9 | corr_static | 3.308 | -15.7% | 1.604 | 2.306/2.844 | 3.241 | -18.3% | 1.746 | 2.151/2.807 | 3.035/5.086 | 4.722 |  |
| 8 | 9 | corr_dynamic | 3.137 | -20.0% | 1.102 | 1.148/1.283 | 3.106 | -21.7% | 1.097 | 1.125/1.301 | 3.035/5.086 | 2.092 | sees the batch |
| 8 | 9 | anti | 4.096 | +4.4% | 1.079 | 1.128/1.191 | 4.128 | +4.1% | 1.056 | 1.119/1.224 | 3.035/5.086 | 4.476 | intra-cosel x0.89, swap cap hit 0 layers |
| 16 | 0 | default | 6.136 | +0.0% | 1.500 | 1.711/1.822 | 6.127 | +0.0% | 1.295 | 1.647/1.867 | 3.035/5.086 | 5.086 |  |
| 16 | 0 | corr_static | 5.045 | -17.8% | 2.280 | 3.162/4.083 | 4.909 | -19.9% | 2.213 | 2.950/4.032 | 3.035/5.086 | 5.086 |  |
| 16 | 0 | anti | 6.540 | +6.6% | 1.329 | 1.416/1.472 | 6.526 | +6.5% | 1.143 | 1.379/1.512 | 3.035/5.086 | 5.086 | intra-cosel x0.64, swap cap hit 0 layers |
| 16 | 1 | default | 5.855 | +0.0% | 1.393 | 1.596/1.757 | 5.900 | +0.0% | 1.274 | 1.565/1.807 | 3.035/5.086 | 4.695 | dup copies 18 |
| 16 | 1 | corr_static | 4.835 | -17.4% | 2.196 | 3.117/4.072 | 4.741 | -19.6% | 2.218 | 2.882/4.017 | 3.035/5.086 | 4.800 |  |
| 16 | 1 | corr_dynamic | 4.860 | -17.0% | 1.475 | 1.952/2.411 | 4.815 | -18.4% | 1.507 | 1.846/2.387 | 3.035/5.086 | 3.403 | sees the batch |
| 16 | 1 | anti | 6.239 | +6.6% | 1.230 | 1.337/1.450 | 6.271 | +6.3% | 1.127 | 1.334/1.476 | 3.035/5.086 | 4.673 | intra-cosel x0.66, swap cap hit 0 layers |
| 16 | 2 | default | 5.702 | +0.0% | 1.340 | 1.545/1.706 | 5.745 | +0.0% | 1.248 | 1.516/1.742 | 3.035/5.086 | 4.550 | dup copies 29 |
| 16 | 2 | corr_static | 4.734 | -17.0% | 2.132 | 3.067/3.984 | 4.651 | -19.1% | 2.188 | 2.841/3.904 | 3.035/5.086 | 4.704 |  |
| 16 | 2 | corr_dynamic | 4.715 | -17.3% | 1.268 | 1.480/1.805 | 4.706 | -18.1% | 1.281 | 1.415/1.781 | 3.035/5.086 | 2.827 | sees the batch |
| 16 | 2 | anti | 6.067 | +6.4% | 1.198 | 1.326/1.429 | 6.096 | +6.1% | 1.124 | 1.331/1.479 | 3.035/5.086 | 4.543 | intra-cosel x0.67, swap cap hit 0 layers |
| 16 | 5 | default | 5.423 | +0.0% | 1.228 | 1.404/1.519 | 5.466 | +0.0% | 1.191 | 1.367/1.548 | 3.035/5.086 | 4.591 | dup copies 64 |
| 16 | 5 | corr_static | 4.582 | -15.5% | 1.933 | 2.858/3.671 | 4.518 | -17.4% | 2.077 | 2.642/3.650 | 3.035/5.086 | 4.772 |  |
| 16 | 5 | corr_dynamic | 4.407 | -18.7% | 1.091 | 1.135/1.276 | 4.404 | -19.4% | 1.092 | 1.127/1.282 | 3.035/5.086 | 2.137 | sees the batch |
| 16 | 5 | anti | 5.740 | +5.8% | 1.107 | 1.216/1.329 | 5.765 | +5.5% | 1.084 | 1.191/1.366 | 3.035/5.086 | 4.555 | intra-cosel x0.71, swap cap hit 0 layers |
| 32 | 0 | default | 7.071 | +0.0% | 1.946 | 2.370/2.641 | 7.064 | +0.0% | 1.610 | 2.290/2.720 | 3.035/5.086 | 5.086 |  |
| 32 | 0 | corr_static | 5.984 | -15.4% | 2.877 | 3.993/5.024 | 5.890 | -16.6% | 2.552 | 3.807/5.133 | 3.035/5.086 | 5.086 |  |
| 32 | 0 | anti | 7.371 | +4.2% | 1.723 | 1.902/2.107 | 7.373 | +4.4% | 1.359 | 1.865/2.143 | 3.035/5.086 | 5.086 | intra-cosel x0.50, swap cap hit 0 layers |
| 32 | 1 | default | 6.764 | +0.0% | 1.610 | 1.935/2.260 | 6.797 | +0.0% | 1.410 | 1.937/2.335 | 3.035/5.086 | 4.539 | dup copies 15 |
| 32 | 1 | corr_static | 5.751 | -15.0% | 2.568 | 3.773/4.637 | 5.685 | -16.4% | 2.409 | 3.592/4.793 | 3.035/5.086 | 4.613 |  |
| 32 | 1 | corr_dynamic | 5.811 | -14.1% | 1.340 | 1.586/1.910 | 5.833 | -14.2% | 1.295 | 1.532/1.879 | 3.035/5.086 | 2.888 | sees the batch |
| 32 | 1 | anti | 7.115 | +5.2% | 1.401 | 1.610/1.837 | 7.140 | +5.0% | 1.241 | 1.630/1.927 | 3.035/5.086 | 4.512 | intra-cosel x0.51, swap cap hit 0 layers |
| 32 | 2 | default | 6.618 | +0.0% | 1.504 | 1.755/2.038 | 6.653 | +0.0% | 1.343 | 1.736/2.191 | 3.035/5.086 | 4.597 | dup copies 25 |
| 32 | 2 | corr_static | 5.684 | -14.1% | 2.363 | 3.601/4.380 | 5.627 | -15.4% | 2.336 | 3.381/4.449 | 3.035/5.086 | 4.628 |  |
| 32 | 2 | corr_dynamic | 5.638 | -14.8% | 1.182 | 1.284/1.463 | 5.669 | -14.8% | 1.149 | 1.252/1.474 | 3.035/5.086 | 2.354 | sees the batch |
| 32 | 2 | anti | 6.970 | +5.3% | 1.307 | 1.512/1.719 | 6.996 | +5.2% | 1.197 | 1.513/1.827 | 3.035/5.086 | 4.493 | intra-cosel x0.53, swap cap hit 0 layers |

Go/no-go (fixed before results: >= 15% fewer sends than (a) at the same R AND p95 batch imbalance <= 1.15 or <= (a)+0.05; judged on task-pooled batches and on mix):
- G8 R1 corr_static [tasks]: send cut vs (a) 17.0%, p95 imb_b 2.960 (a: 1.466) -> no-go
- G8 R1 corr_static [mix]: send cut vs (a) 20.2%, p95 imb_b 2.945 (a: 1.481) -> no-go
- G8 R1 corr_dynamic [tasks]: send cut vs (a) 17.1%, p95 imb_b 2.670 (a: 1.466) -> no-go
- G8 R1 corr_dynamic [mix]: send cut vs (a) 20.8%, p95 imb_b 2.656 (a: 1.481) -> no-go
- G8 R2 corr_static [tasks]: send cut vs (a) 16.7%, p95 imb_b 2.965 (a: 1.459) -> no-go
- G8 R2 corr_static [mix]: send cut vs (a) 19.8%, p95 imb_b 2.950 (a: 1.484) -> no-go
- G8 R2 corr_dynamic [tasks]: send cut vs (a) 17.4%, p95 imb_b 2.387 (a: 1.459) -> no-go
- G8 R2 corr_dynamic [mix]: send cut vs (a) 20.8%, p95 imb_b 2.388 (a: 1.484) -> no-go
- G8 R9 corr_static [tasks]: send cut vs (a) 15.7%, p95 imb_b 2.844 (a: 1.367) -> no-go
- G8 R9 corr_static [mix]: send cut vs (a) 18.3%, p95 imb_b 2.807 (a: 1.392) -> no-go
- G8 R9 corr_dynamic [tasks]: send cut vs (a) 20.0%, p95 imb_b 1.283 (a: 1.367) -> GO
- G8 R9 corr_dynamic [mix]: send cut vs (a) 21.7%, p95 imb_b 1.301 (a: 1.392) -> GO
- G16 R1 corr_static [tasks]: send cut vs (a) 17.4%, p95 imb_b 4.072 (a: 1.757) -> no-go
- G16 R1 corr_static [mix]: send cut vs (a) 19.6%, p95 imb_b 4.017 (a: 1.807) -> no-go
- G16 R1 corr_dynamic [tasks]: send cut vs (a) 17.0%, p95 imb_b 2.411 (a: 1.757) -> no-go
- G16 R1 corr_dynamic [mix]: send cut vs (a) 18.4%, p95 imb_b 2.387 (a: 1.807) -> no-go
- G16 R2 corr_static [tasks]: send cut vs (a) 17.0%, p95 imb_b 3.984 (a: 1.706) -> no-go
- G16 R2 corr_static [mix]: send cut vs (a) 19.1%, p95 imb_b 3.904 (a: 1.742) -> no-go
- G16 R2 corr_dynamic [tasks]: send cut vs (a) 17.3%, p95 imb_b 1.805 (a: 1.706) -> no-go
- G16 R2 corr_dynamic [mix]: send cut vs (a) 18.1%, p95 imb_b 1.781 (a: 1.742) -> GO
- G16 R5 corr_static [tasks]: send cut vs (a) 15.5%, p95 imb_b 3.671 (a: 1.519) -> no-go
- G16 R5 corr_static [mix]: send cut vs (a) 17.4%, p95 imb_b 3.650 (a: 1.548) -> no-go
- G16 R5 corr_dynamic [tasks]: send cut vs (a) 18.7%, p95 imb_b 1.276 (a: 1.519) -> GO
- G16 R5 corr_dynamic [mix]: send cut vs (a) 19.4%, p95 imb_b 1.282 (a: 1.548) -> GO
- G32 R1 corr_static [tasks]: send cut vs (a) 15.0%, p95 imb_b 4.637 (a: 2.260) -> no-go
- G32 R1 corr_static [mix]: send cut vs (a) 16.4%, p95 imb_b 4.793 (a: 2.335) -> no-go
- G32 R1 corr_dynamic [tasks]: send cut vs (a) 14.1%, p95 imb_b 1.910 (a: 2.260) -> no-go
- G32 R1 corr_dynamic [mix]: send cut vs (a) 14.2%, p95 imb_b 1.879 (a: 2.335) -> no-go
- G32 R2 corr_static [tasks]: send cut vs (a) 14.1%, p95 imb_b 4.380 (a: 2.038) -> no-go
- G32 R2 corr_static [mix]: send cut vs (a) 15.4%, p95 imb_b 4.449 (a: 2.191) -> no-go
- G32 R2 corr_dynamic [tasks]: send cut vs (a) 14.8%, p95 imb_b 1.463 (a: 2.038) -> no-go
- G32 R2 corr_dynamic [mix]: send cut vs (a) 14.8%, p95 imb_b 1.474 (a: 2.191) -> no-go

## copies_qwen.json.ckpt  (E=256, layers=40)

Simulated replay on holdout. 'tasks' = mean over task types for sends/imb; imb_b p50/p95 over full 2048-token batches pooled across task types (full batches per set: {'chat': 9, 'code': 9, 'math': 9, 'mix': 29}). 'mix' = mix holdout (task-ordered concatenation, so its batches are mostly single-task). vs (a) = vs DefaultEplbPolicy at the same R.

| G | R | policy | sends tasks | vs (a) | imb tasks | imb_b p50/p95 tasks | sends mix | vs (a) | imb mix | imb_b p50/p95 mix | kappa p50/p95 mix | kappa_slot p95 mix | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 0 | default | 4.662 | +0.0% | 1.231 | 1.278/1.326 | 4.662 | +0.0% | 1.034 | 1.268/1.321 | 2.611/3.607 | 3.607 |  |
| 8 | 0 | corr_static | 3.433 | -26.4% | 1.701 | 1.669/2.076 | 3.432 | -26.4% | 1.063 | 1.666/2.024 | 2.611/3.607 | 3.607 |  |
| 8 | 0 | anti | 5.140 | +10.3% | 1.087 | 1.133/1.154 | 5.140 | +10.3% | 1.025 | 1.133/1.147 | 2.611/3.607 | 3.607 | intra-cosel x0.71, swap cap hit 0 layers |
| 8 | 1 | default | 4.458 | +0.0% | 1.208 | 1.239/1.308 | 4.457 | +0.0% | 1.049 | 1.234/1.314 | 2.611/3.607 | 2.775 | dup copies 14 |
| 8 | 1 | corr_static | 3.275 | -26.5% | 1.676 | 1.637/2.061 | 3.274 | -26.5% | 1.079 | 1.657/2.010 | 2.611/3.607 | 2.800 |  |
| 8 | 1 | corr_dynamic | 3.288 | -26.2% | 1.466 | 1.463/1.662 | 3.289 | -26.2% | 1.077 | 1.448/1.631 | 2.611/3.607 | 2.598 | sees the batch |
| 8 | 1 | anti | 4.829 | +8.3% | 1.104 | 1.145/1.159 | 4.828 | +8.3% | 1.032 | 1.141/1.161 | 2.611/3.607 | 2.761 | intra-cosel x0.76, swap cap hit 0 layers |
| 8 | 2 | default | 4.332 | +0.0% | 1.214 | 1.234/1.308 | 4.332 | +0.0% | 1.049 | 1.233/1.306 | 2.611/3.607 | 2.534 | dup copies 33 |
| 8 | 2 | corr_static | 3.208 | -26.0% | 1.654 | 1.636/1.994 | 3.207 | -26.0% | 1.082 | 1.644/1.948 | 2.611/3.607 | 2.563 |  |
| 8 | 2 | corr_dynamic | 3.197 | -26.2% | 1.355 | 1.363/1.493 | 3.201 | -26.1% | 1.086 | 1.358/1.474 | 2.611/3.607 | 2.278 | sees the batch |
| 8 | 2 | anti | 4.673 | +7.9% | 1.091 | 1.130/1.149 | 4.673 | +7.9% | 1.035 | 1.125/1.150 | 2.611/3.607 | 2.522 | intra-cosel x0.78, swap cap hit 0 layers |
| 8 | 8 | default | 3.899 | +0.0% | 1.137 | 1.159/1.203 | 3.897 | +0.0% | 1.045 | 1.153/1.201 | 2.611/3.607 | 2.296 | dup copies 139 |
| 8 | 8 | corr_static | 2.998 | -23.1% | 1.531 | 1.501/1.764 | 2.996 | -23.1% | 1.087 | 1.501/1.728 | 2.611/3.607 | 2.333 |  |
| 8 | 8 | corr_dynamic | 2.872 | -26.3% | 1.119 | 1.146/1.174 | 2.876 | -26.2% | 1.064 | 1.147/1.173 | 2.611/3.607 | 1.679 | sees the batch |
| 8 | 8 | anti | 4.147 | +6.4% | 1.040 | 1.050/1.095 | 4.147 | +6.4% | 1.021 | 1.049/1.094 | 2.611/3.607 | 2.285 | intra-cosel x0.86, swap cap hit 0 layers |
| 16 | 0 | default | 6.178 | +0.0% | 1.446 | 1.541/1.604 | 6.178 | +0.0% | 1.058 | 1.522/1.619 | 2.611/3.607 | 3.607 |  |
| 16 | 0 | corr_static | 4.588 | -25.7% | 1.967 | 1.967/2.552 | 4.588 | -25.7% | 1.114 | 1.936/2.484 | 2.611/3.607 | 3.607 |  |
| 16 | 0 | anti | 6.733 | +9.0% | 1.237 | 1.300/1.377 | 6.733 | +9.0% | 1.054 | 1.284/1.384 | 2.611/3.607 | 3.607 | intra-cosel x0.53, swap cap hit 0 layers |
| 16 | 1 | default | 5.822 | +0.0% | 1.343 | 1.381/1.546 | 5.822 | +0.0% | 1.068 | 1.372/1.561 | 2.611/3.607 | 2.559 | dup copies 12 |
| 16 | 1 | corr_static | 4.354 | -25.2% | 1.868 | 1.844/2.483 | 4.354 | -25.2% | 1.139 | 1.855/2.420 | 2.611/3.607 | 2.569 |  |
| 16 | 1 | corr_dynamic | 4.411 | -24.2% | 1.400 | 1.456/1.609 | 4.412 | -24.2% | 1.118 | 1.448/1.614 | 2.611/3.607 | 2.326 | sees the batch |
| 16 | 1 | anti | 6.337 | +8.8% | 1.157 | 1.231/1.280 | 6.337 | +8.8% | 1.048 | 1.218/1.278 | 2.611/3.607 | 2.564 | intra-cosel x0.57, swap cap hit 0 layers |
| 16 | 2 | default | 5.664 | +0.0% | 1.260 | 1.328/1.419 | 5.664 | +0.0% | 1.064 | 1.325/1.418 | 2.611/3.607 | 2.441 | dup copies 32 |
| 16 | 2 | corr_static | 4.289 | -24.3% | 1.786 | 1.769/2.313 | 4.288 | -24.3% | 1.140 | 1.769/2.283 | 2.611/3.607 | 2.440 |  |
| 16 | 2 | corr_dynamic | 4.293 | -24.2% | 1.255 | 1.327/1.372 | 4.295 | -24.2% | 1.110 | 1.318/1.371 | 2.611/3.607 | 1.981 | sees the batch |
| 16 | 2 | anti | 6.134 | +8.3% | 1.129 | 1.182/1.242 | 6.134 | +8.3% | 1.038 | 1.169/1.242 | 2.611/3.607 | 2.433 | intra-cosel x0.60, swap cap hit 0 layers |
| 16 | 4 | default | 5.456 | +0.0% | 1.195 | 1.252/1.353 | 5.455 | +0.0% | 1.057 | 1.234/1.351 | 2.611/3.607 | 2.395 | dup copies 65 |
| 16 | 4 | corr_static | 4.239 | -22.3% | 1.662 | 1.696/2.045 | 4.237 | -22.3% | 1.138 | 1.702/2.021 | 2.611/3.607 | 2.398 |  |
| 16 | 4 | corr_dynamic | 4.144 | -24.0% | 1.123 | 1.160/1.197 | 4.146 | -24.0% | 1.068 | 1.153/1.200 | 2.611/3.607 | 1.726 | sees the batch |
| 16 | 4 | anti | 5.888 | +7.9% | 1.075 | 1.102/1.168 | 5.889 | +8.0% | 1.031 | 1.103/1.168 | 2.611/3.607 | 2.386 | intra-cosel x0.62, swap cap hit 0 layers |
| 32 | 0 | default | 7.114 | +0.0% | 1.830 | 2.001/2.224 | 7.115 | +0.0% | 1.125 | 1.940/2.208 | 2.611/3.607 | 3.607 |  |
| 32 | 0 | corr_static | 5.556 | -21.9% | 2.383 | 2.463/3.122 | 5.556 | -21.9% | 1.213 | 2.437/3.111 | 2.611/3.607 | 3.607 |  |
| 32 | 0 | anti | 7.498 | +5.4% | 1.642 | 1.726/2.113 | 7.498 | +5.4% | 1.130 | 1.687/2.096 | 2.611/3.607 | 3.607 | intra-cosel x0.35, swap cap hit 0 layers |
| 32 | 1 | default | 6.754 | +0.0% | 1.464 | 1.613/1.705 | 6.754 | +0.0% | 1.108 | 1.548/1.727 | 2.611/3.607 | 2.455 | dup copies 14 |
| 32 | 1 | corr_static | 5.341 | -20.9% | 2.082 | 2.239/2.777 | 5.340 | -20.9% | 1.248 | 2.230/2.697 | 2.611/3.607 | 2.458 |  |
| 32 | 1 | corr_dynamic | 5.439 | -19.5% | 1.320 | 1.444/1.492 | 5.435 | -19.5% | 1.159 | 1.448/1.495 | 2.611/3.607 | 2.103 | sees the batch |
| 32 | 1 | anti | 7.227 | +7.0% | 1.238 | 1.379/1.459 | 7.228 | +7.0% | 1.091 | 1.369/1.471 | 2.611/3.607 | 2.432 | intra-cosel x0.38, swap cap hit 0 layers |
| 32 | 2 | default | 6.615 | +0.0% | 1.348 | 1.459/1.603 | 6.615 | +0.0% | 1.107 | 1.432/1.602 | 2.611/3.607 | 2.437 | dup copies 28 |
| 32 | 2 | corr_static | 5.338 | -19.3% | 1.921 | 2.064/2.463 | 5.336 | -19.3% | 1.239 | 2.088/2.447 | 2.611/3.607 | 2.449 |  |
| 32 | 2 | corr_dynamic | 5.341 | -19.3% | 1.174 | 1.227/1.309 | 5.342 | -19.3% | 1.101 | 1.220/1.319 | 2.611/3.607 | 1.833 | sees the batch |
| 32 | 2 | anti | 7.074 | +6.9% | 1.172 | 1.274/1.355 | 7.074 | +6.9% | 1.071 | 1.261/1.353 | 2.611/3.607 | 2.404 | intra-cosel x0.39, swap cap hit 0 layers |

Go/no-go (fixed before results: >= 15% fewer sends than (a) at the same R AND p95 batch imbalance <= 1.15 or <= (a)+0.05; judged on task-pooled batches and on mix):
- G8 R1 corr_static [tasks]: send cut vs (a) 26.5%, p95 imb_b 2.061 (a: 1.308) -> no-go
- G8 R1 corr_static [mix]: send cut vs (a) 26.5%, p95 imb_b 2.010 (a: 1.314) -> no-go
- G8 R1 corr_dynamic [tasks]: send cut vs (a) 26.2%, p95 imb_b 1.662 (a: 1.308) -> no-go
- G8 R1 corr_dynamic [mix]: send cut vs (a) 26.2%, p95 imb_b 1.631 (a: 1.314) -> no-go
- G8 R2 corr_static [tasks]: send cut vs (a) 26.0%, p95 imb_b 1.994 (a: 1.308) -> no-go
- G8 R2 corr_static [mix]: send cut vs (a) 26.0%, p95 imb_b 1.948 (a: 1.306) -> no-go
- G8 R2 corr_dynamic [tasks]: send cut vs (a) 26.2%, p95 imb_b 1.493 (a: 1.308) -> no-go
- G8 R2 corr_dynamic [mix]: send cut vs (a) 26.1%, p95 imb_b 1.474 (a: 1.306) -> no-go
- G8 R8 corr_static [tasks]: send cut vs (a) 23.1%, p95 imb_b 1.764 (a: 1.203) -> no-go
- G8 R8 corr_static [mix]: send cut vs (a) 23.1%, p95 imb_b 1.728 (a: 1.201) -> no-go
- G8 R8 corr_dynamic [tasks]: send cut vs (a) 26.3%, p95 imb_b 1.174 (a: 1.203) -> GO
- G8 R8 corr_dynamic [mix]: send cut vs (a) 26.2%, p95 imb_b 1.173 (a: 1.201) -> GO
- G16 R1 corr_static [tasks]: send cut vs (a) 25.2%, p95 imb_b 2.483 (a: 1.546) -> no-go
- G16 R1 corr_static [mix]: send cut vs (a) 25.2%, p95 imb_b 2.420 (a: 1.561) -> no-go
- G16 R1 corr_dynamic [tasks]: send cut vs (a) 24.2%, p95 imb_b 1.609 (a: 1.546) -> no-go
- G16 R1 corr_dynamic [mix]: send cut vs (a) 24.2%, p95 imb_b 1.614 (a: 1.561) -> no-go
- G16 R2 corr_static [tasks]: send cut vs (a) 24.3%, p95 imb_b 2.313 (a: 1.419) -> no-go
- G16 R2 corr_static [mix]: send cut vs (a) 24.3%, p95 imb_b 2.283 (a: 1.418) -> no-go
- G16 R2 corr_dynamic [tasks]: send cut vs (a) 24.2%, p95 imb_b 1.372 (a: 1.419) -> GO
- G16 R2 corr_dynamic [mix]: send cut vs (a) 24.2%, p95 imb_b 1.371 (a: 1.418) -> GO
- G16 R4 corr_static [tasks]: send cut vs (a) 22.3%, p95 imb_b 2.045 (a: 1.353) -> no-go
- G16 R4 corr_static [mix]: send cut vs (a) 22.3%, p95 imb_b 2.021 (a: 1.351) -> no-go
- G16 R4 corr_dynamic [tasks]: send cut vs (a) 24.0%, p95 imb_b 1.197 (a: 1.353) -> GO
- G16 R4 corr_dynamic [mix]: send cut vs (a) 24.0%, p95 imb_b 1.200 (a: 1.351) -> GO
- G32 R1 corr_static [tasks]: send cut vs (a) 20.9%, p95 imb_b 2.777 (a: 1.705) -> no-go
- G32 R1 corr_static [mix]: send cut vs (a) 20.9%, p95 imb_b 2.697 (a: 1.727) -> no-go
- G32 R1 corr_dynamic [tasks]: send cut vs (a) 19.5%, p95 imb_b 1.492 (a: 1.705) -> GO
- G32 R1 corr_dynamic [mix]: send cut vs (a) 19.5%, p95 imb_b 1.495 (a: 1.727) -> GO
- G32 R2 corr_static [tasks]: send cut vs (a) 19.3%, p95 imb_b 2.463 (a: 1.603) -> no-go
- G32 R2 corr_static [mix]: send cut vs (a) 19.3%, p95 imb_b 2.447 (a: 1.602) -> no-go
- G32 R2 corr_dynamic [tasks]: send cut vs (a) 19.3%, p95 imb_b 1.309 (a: 1.603) -> GO
- G32 R2 corr_dynamic [mix]: send cut vs (a) 19.3%, p95 imb_b 1.319 (a: 1.602) -> GO
