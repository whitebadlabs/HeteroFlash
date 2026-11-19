
## balance_glm.json.ckpt  (E=288, layers=42)

Simulated replay on holdout. 'tasks' = mean over task types for sends/imb; imb_b p50/p95 over full 2048-token batches pooled across task types (full batches per set: {'code': 1, 'instruction_following': 1, 'knowledge_qa': 1, 'long_context': 14, 'math': 1, 'multilingual': 1, 'writing_chat': 2, 'mix': 26}). 'mix' = mix holdout (task-ordered concatenation, so its batches are mostly single-task). vs (a) = vs DefaultEplbPolicy at the same R.

| G | R | policy | sends tasks | vs (a) | imb tasks | imb_b p50/p95 tasks | sends mix | vs (a) | imb mix | imb_b p50/p95 mix | kappa p50/p95 mix | kappa_slot p95 mix | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 0 | default | 4.656 | +0.0% | 1.254 | 1.333/1.467 | 4.641 | +0.0% | 1.163 | 1.325/1.493 | 3.035/5.086 | 5.086 |  |
| 8 | 0 | corr_static | 3.857 | -17.2% | 1.765 | 2.438/2.979 | 3.707 | -20.1% | 1.799 | 2.287/2.958 | 3.035/5.086 | 5.086 |  |
| 8 | 1 | default | 4.655 | +0.0% | 1.181 | 1.293/1.407 | 4.653 | +0.0% | 1.145 | 1.271/1.428 | 3.035/5.086 | 4.950 | dup copies 12 |
| 8 | 1 | corr_static | 3.941 | -15.3% | 1.643 | 2.334/2.835 | 3.791 | -18.5% | 1.752 | 2.204/2.795 | 3.035/5.086 | 5.180 |  |
| 8 | 1 | corr_dynamic | 4.143 | -11.0% | 1.234 | 1.527/1.750 | 4.155 | -10.7% | 1.282 | 1.412/1.733 | 3.035/5.086 | 4.935 | sees the batch |
| 8 | 2 | default | 4.659 | +0.0% | 1.132 | 1.218/1.336 | 4.661 | +0.0% | 1.121 | 1.211/1.356 | 3.035/5.086 | 4.980 | dup copies 27 |
| 8 | 2 | corr_static | 4.005 | -14.0% | 1.537 | 2.218/2.669 | 3.870 | -17.0% | 1.689 | 2.100/2.661 | 3.035/5.086 | 5.247 |  |
| 8 | 2 | corr_dynamic | 4.191 | -10.0% | 1.130 | 1.249/1.387 | 4.222 | -9.4% | 1.139 | 1.181/1.374 | 3.035/5.086 | 4.450 | sees the batch |
| 8 | 9 | default | 4.648 | +0.0% | 1.022 | 1.057/1.106 | 4.659 | +0.0% | 1.037 | 1.051/1.125 | 3.035/5.086 | 5.149 | dup copies 176 |
| 8 | 9 | corr_static | 4.220 | -9.2% | 1.177 | 1.637/1.855 | 4.173 | -10.4% | 1.363 | 1.562/1.855 | 3.035/5.086 | 5.815 |  |
| 8 | 9 | corr_dynamic | 4.197 | -9.7% | 1.073 | 1.075/1.086 | 4.195 | -10.0% | 1.032 | 1.072/1.085 | 3.035/5.086 | 2.710 | sees the batch |
| 16 | 0 | default | 6.136 | +0.0% | 1.500 | 1.711/1.822 | 6.127 | +0.0% | 1.295 | 1.647/1.867 | 3.035/5.086 | 5.086 |  |
| 16 | 0 | corr_static | 5.045 | -17.8% | 2.280 | 3.162/4.083 | 4.909 | -19.9% | 2.213 | 2.950/4.032 | 3.035/5.086 | 5.086 |  |
| 16 | 1 | default | 6.150 | +0.0% | 1.316 | 1.525/1.687 | 6.147 | +0.0% | 1.242 | 1.496/1.743 | 3.035/5.086 | 4.810 | dup copies 18 |
| 16 | 1 | corr_static | 5.232 | -14.9% | 1.970 | 2.866/3.608 | 5.110 | -16.9% | 2.046 | 2.683/3.599 | 3.035/5.086 | 5.201 |  |
| 16 | 1 | corr_dynamic | 5.497 | -10.6% | 1.259 | 1.468/1.667 | 5.558 | -9.6% | 1.246 | 1.406/1.644 | 3.035/5.086 | 4.368 | sees the batch |
| 16 | 2 | default | 6.159 | +0.0% | 1.238 | 1.405/1.528 | 6.169 | +0.0% | 1.183 | 1.376/1.570 | 3.035/5.086 | 4.856 | dup copies 29 |
| 16 | 2 | corr_static | 5.347 | -13.2% | 1.776 | 2.581/3.281 | 5.258 | -14.8% | 1.889 | 2.442/3.105 | 3.035/5.086 | 5.314 |  |
| 16 | 2 | corr_dynamic | 5.541 | -10.0% | 1.163 | 1.256/1.352 | 5.593 | -9.3% | 1.129 | 1.230/1.348 | 3.035/5.086 | 3.627 | sees the batch |
| 16 | 5 | default | 6.155 | +0.0% | 1.101 | 1.209/1.297 | 6.172 | +0.0% | 1.109 | 1.185/1.329 | 3.035/5.086 | 5.069 | dup copies 64 |
| 16 | 5 | corr_static | 5.581 | -9.3% | 1.393 | 1.997/2.435 | 5.542 | -10.2% | 1.557 | 1.873/2.359 | 3.035/5.086 | 5.697 |  |
| 16 | 5 | corr_dynamic | 5.575 | -9.4% | 1.118 | 1.143/1.173 | 5.591 | -9.4% | 1.066 | 1.139/1.160 | 3.035/5.086 | 2.629 | sees the batch |

Go/no-go (fixed before results: >= 15% fewer sends than (a) at the same R AND p95 batch imbalance <= 1.15 or <= (a)+0.05; judged on task-pooled batches and on mix):
- G8 R1 corr_static [tasks]: send cut vs (a) 15.3%, p95 imb_b 2.835 (a: 1.407) -> no-go
- G8 R1 corr_static [mix]: send cut vs (a) 18.5%, p95 imb_b 2.795 (a: 1.428) -> no-go
- G8 R1 corr_dynamic [tasks]: send cut vs (a) 11.0%, p95 imb_b 1.750 (a: 1.407) -> no-go
- G8 R1 corr_dynamic [mix]: send cut vs (a) 10.7%, p95 imb_b 1.733 (a: 1.428) -> no-go
- G8 R2 corr_static [tasks]: send cut vs (a) 14.0%, p95 imb_b 2.669 (a: 1.336) -> no-go
- G8 R2 corr_static [mix]: send cut vs (a) 17.0%, p95 imb_b 2.661 (a: 1.356) -> no-go
- G8 R2 corr_dynamic [tasks]: send cut vs (a) 10.0%, p95 imb_b 1.387 (a: 1.336) -> no-go
- G8 R2 corr_dynamic [mix]: send cut vs (a) 9.4%, p95 imb_b 1.374 (a: 1.356) -> no-go
- G8 R9 corr_static [tasks]: send cut vs (a) 9.2%, p95 imb_b 1.855 (a: 1.106) -> no-go
- G8 R9 corr_static [mix]: send cut vs (a) 10.4%, p95 imb_b 1.855 (a: 1.125) -> no-go
- G8 R9 corr_dynamic [tasks]: send cut vs (a) 9.7%, p95 imb_b 1.086 (a: 1.106) -> no-go
- G8 R9 corr_dynamic [mix]: send cut vs (a) 10.0%, p95 imb_b 1.085 (a: 1.125) -> no-go
- G16 R1 corr_static [tasks]: send cut vs (a) 14.9%, p95 imb_b 3.608 (a: 1.687) -> no-go
- G16 R1 corr_static [mix]: send cut vs (a) 16.9%, p95 imb_b 3.599 (a: 1.743) -> no-go
- G16 R1 corr_dynamic [tasks]: send cut vs (a) 10.6%, p95 imb_b 1.667 (a: 1.687) -> no-go
- G16 R1 corr_dynamic [mix]: send cut vs (a) 9.6%, p95 imb_b 1.644 (a: 1.743) -> no-go
- G16 R2 corr_static [tasks]: send cut vs (a) 13.2%, p95 imb_b 3.281 (a: 1.528) -> no-go
- G16 R2 corr_static [mix]: send cut vs (a) 14.8%, p95 imb_b 3.105 (a: 1.570) -> no-go
- G16 R2 corr_dynamic [tasks]: send cut vs (a) 10.0%, p95 imb_b 1.352 (a: 1.528) -> no-go
- G16 R2 corr_dynamic [mix]: send cut vs (a) 9.3%, p95 imb_b 1.348 (a: 1.570) -> no-go
- G16 R5 corr_static [tasks]: send cut vs (a) 9.3%, p95 imb_b 2.435 (a: 1.297) -> no-go
- G16 R5 corr_static [mix]: send cut vs (a) 10.2%, p95 imb_b 2.359 (a: 1.329) -> no-go
- G16 R5 corr_dynamic [tasks]: send cut vs (a) 9.4%, p95 imb_b 1.173 (a: 1.297) -> no-go
- G16 R5 corr_dynamic [mix]: send cut vs (a) 9.4%, p95 imb_b 1.160 (a: 1.329) -> no-go

## balance_qwen.json.ckpt  (E=256, layers=40)

Simulated replay on holdout. 'tasks' = mean over task types for sends/imb; imb_b p50/p95 over full 2048-token batches pooled across task types (full batches per set: {'chat': 9, 'code': 9, 'math': 9, 'mix': 29}). 'mix' = mix holdout (task-ordered concatenation, so its batches are mostly single-task). vs (a) = vs DefaultEplbPolicy at the same R.

| G | R | policy | sends tasks | vs (a) | imb tasks | imb_b p50/p95 tasks | sends mix | vs (a) | imb mix | imb_b p50/p95 mix | kappa p50/p95 mix | kappa_slot p95 mix | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 0 | default | 4.662 | +0.0% | 1.231 | 1.278/1.326 | 4.662 | +0.0% | 1.034 | 1.268/1.321 | 2.611/3.607 | 3.607 |  |
| 8 | 0 | corr_static | 3.433 | -26.4% | 1.701 | 1.669/2.076 | 3.432 | -26.4% | 1.063 | 1.666/2.024 | 2.611/3.607 | 3.607 |  |
| 8 | 1 | default | 4.661 | +0.0% | 1.153 | 1.178/1.246 | 4.660 | +0.0% | 1.049 | 1.167/1.251 | 2.611/3.607 | 2.885 | dup copies 14 |
| 8 | 1 | corr_static | 3.542 | -24.0% | 1.559 | 1.526/1.915 | 3.537 | -24.1% | 1.104 | 1.566/1.879 | 2.611/3.607 | 3.245 |  |
| 8 | 1 | corr_dynamic | 3.782 | -18.9% | 1.241 | 1.234/1.348 | 3.770 | -19.1% | 1.078 | 1.226/1.338 | 2.611/3.607 | 3.234 | sees the batch |
| 8 | 2 | default | 4.665 | +0.0% | 1.122 | 1.142/1.197 | 4.664 | +0.0% | 1.046 | 1.139/1.195 | 2.611/3.607 | 2.811 | dup copies 33 |
| 8 | 2 | corr_static | 3.643 | -21.9% | 1.442 | 1.460/1.684 | 3.638 | -22.0% | 1.101 | 1.460/1.668 | 2.611/3.607 | 3.200 |  |
| 8 | 2 | corr_dynamic | 3.854 | -17.4% | 1.145 | 1.143/1.209 | 3.842 | -17.6% | 1.056 | 1.137/1.205 | 2.611/3.607 | 3.057 | sees the batch |
| 8 | 8 | default | 4.642 | +0.0% | 1.020 | 1.020/1.031 | 4.642 | +0.0% | 1.013 | 1.020/1.033 | 2.611/3.607 | 2.791 | dup copies 139 |
| 8 | 8 | corr_static | 3.972 | -14.4% | 1.134 | 1.112/1.252 | 3.964 | -14.6% | 1.052 | 1.142/1.251 | 2.611/3.607 | 3.228 |  |
| 8 | 8 | corr_dynamic | 3.950 | -14.9% | 1.090 | 1.092/1.112 | 3.939 | -15.1% | 1.038 | 1.093/1.115 | 2.611/3.607 | 2.169 | sees the batch |
| 16 | 0 | default | 6.178 | +0.0% | 1.446 | 1.541/1.604 | 6.178 | +0.0% | 1.058 | 1.522/1.619 | 2.611/3.607 | 3.607 |  |
| 16 | 0 | corr_static | 4.588 | -25.7% | 1.967 | 1.967/2.552 | 4.588 | -25.7% | 1.114 | 1.936/2.484 | 2.611/3.607 | 3.607 |  |
| 16 | 1 | default | 6.158 | +0.0% | 1.279 | 1.330/1.415 | 6.156 | +0.0% | 1.079 | 1.327/1.414 | 2.611/3.607 | 2.743 | dup copies 12 |
| 16 | 1 | corr_static | 4.796 | -22.1% | 1.687 | 1.724/2.113 | 4.792 | -22.2% | 1.152 | 1.724/2.063 | 2.611/3.607 | 2.953 |  |
| 16 | 1 | corr_dynamic | 5.109 | -17.0% | 1.240 | 1.251/1.384 | 5.093 | -17.3% | 1.089 | 1.254/1.374 | 2.611/3.607 | 2.857 | sees the batch |
| 16 | 2 | default | 6.166 | +0.0% | 1.186 | 1.232/1.315 | 6.166 | +0.0% | 1.070 | 1.226/1.313 | 2.611/3.607 | 2.674 | dup copies 32 |
| 16 | 2 | corr_static | 4.993 | -19.0% | 1.507 | 1.541/1.830 | 4.988 | -19.1% | 1.144 | 1.557/1.829 | 2.611/3.607 | 3.031 |  |
| 16 | 2 | corr_dynamic | 5.198 | -15.7% | 1.152 | 1.171/1.226 | 5.183 | -15.9% | 1.065 | 1.178/1.220 | 2.611/3.607 | 2.527 | sees the batch |
| 16 | 4 | default | 6.156 | +0.0% | 1.113 | 1.126/1.180 | 6.152 | +0.0% | 1.053 | 1.126/1.178 | 2.611/3.607 | 2.693 | dup copies 65 |
| 16 | 4 | corr_static | 5.247 | -14.8% | 1.307 | 1.343/1.626 | 5.240 | -14.8% | 1.112 | 1.349/1.621 | 2.611/3.607 | 3.076 |  |
| 16 | 4 | corr_dynamic | 5.273 | -14.3% | 1.131 | 1.145/1.186 | 5.262 | -14.5% | 1.056 | 1.146/1.186 | 2.611/3.607 | 2.140 | sees the batch |

Go/no-go (fixed before results: >= 15% fewer sends than (a) at the same R AND p95 batch imbalance <= 1.15 or <= (a)+0.05; judged on task-pooled batches and on mix):
- G8 R1 corr_static [tasks]: send cut vs (a) 24.0%, p95 imb_b 1.915 (a: 1.246) -> no-go
- G8 R1 corr_static [mix]: send cut vs (a) 24.1%, p95 imb_b 1.879 (a: 1.251) -> no-go
- G8 R1 corr_dynamic [tasks]: send cut vs (a) 18.9%, p95 imb_b 1.348 (a: 1.246) -> no-go
- G8 R1 corr_dynamic [mix]: send cut vs (a) 19.1%, p95 imb_b 1.338 (a: 1.251) -> no-go
- G8 R2 corr_static [tasks]: send cut vs (a) 21.9%, p95 imb_b 1.684 (a: 1.197) -> no-go
- G8 R2 corr_static [mix]: send cut vs (a) 22.0%, p95 imb_b 1.668 (a: 1.195) -> no-go
- G8 R2 corr_dynamic [tasks]: send cut vs (a) 17.4%, p95 imb_b 1.209 (a: 1.197) -> GO
- G8 R2 corr_dynamic [mix]: send cut vs (a) 17.6%, p95 imb_b 1.205 (a: 1.195) -> GO
- G8 R8 corr_static [tasks]: send cut vs (a) 14.4%, p95 imb_b 1.252 (a: 1.031) -> no-go
- G8 R8 corr_static [mix]: send cut vs (a) 14.6%, p95 imb_b 1.251 (a: 1.033) -> no-go
- G8 R8 corr_dynamic [tasks]: send cut vs (a) 14.9%, p95 imb_b 1.112 (a: 1.031) -> no-go
- G8 R8 corr_dynamic [mix]: send cut vs (a) 15.1%, p95 imb_b 1.115 (a: 1.033) -> GO
- G16 R1 corr_static [tasks]: send cut vs (a) 22.1%, p95 imb_b 2.113 (a: 1.415) -> no-go
- G16 R1 corr_static [mix]: send cut vs (a) 22.2%, p95 imb_b 2.063 (a: 1.414) -> no-go
- G16 R1 corr_dynamic [tasks]: send cut vs (a) 17.0%, p95 imb_b 1.384 (a: 1.415) -> GO
- G16 R1 corr_dynamic [mix]: send cut vs (a) 17.3%, p95 imb_b 1.374 (a: 1.414) -> GO
- G16 R2 corr_static [tasks]: send cut vs (a) 19.0%, p95 imb_b 1.830 (a: 1.315) -> no-go
- G16 R2 corr_static [mix]: send cut vs (a) 19.1%, p95 imb_b 1.829 (a: 1.313) -> no-go
- G16 R2 corr_dynamic [tasks]: send cut vs (a) 15.7%, p95 imb_b 1.226 (a: 1.315) -> GO
- G16 R2 corr_dynamic [mix]: send cut vs (a) 15.9%, p95 imb_b 1.220 (a: 1.313) -> GO
- G16 R4 corr_static [tasks]: send cut vs (a) 14.8%, p95 imb_b 1.626 (a: 1.180) -> no-go
- G16 R4 corr_static [mix]: send cut vs (a) 14.8%, p95 imb_b 1.621 (a: 1.178) -> no-go
- G16 R4 corr_dynamic [tasks]: send cut vs (a) 14.3%, p95 imb_b 1.186 (a: 1.180) -> no-go
- G16 R4 corr_dynamic [mix]: send cut vs (a) 14.5%, p95 imb_b 1.186 (a: 1.178) -> no-go
