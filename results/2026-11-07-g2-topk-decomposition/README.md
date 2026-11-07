# G2: where the top-k loss comes from (layer level, real inputs, official BF16 experts), 2026-11-07

Method: `src/layer_topk_analysis.py`. Expert inputs (hidden states entering each MoE layer) and router logits were captured from the Control model by hook v4 on 36 general-purpose calibration prompts (6 per domain). For each MoE layer, 192 tokens are compared against the original routed output (top-8, sigmoid+bias selection, normalised, x2.5) using the official BF16 expert weights. Metric: mean relative L2 error of the routed-expert output per token. The shared expert is identical in all variants and omitted.

Variants: `kN_active` = top-N renormalised over the kept experts (vLLM/router default; what M2 round 1 measured); `kN_ref8` = top-N with weights normalised over the original top-8 (dropped mass not redistributed); `k4_oracle_ls` = per-token least-squares optimal weights for the top-4 (upper bound for any re-weighting); `k4_contrib_ref8` = the 4 of the original 8 with the largest contribution ||w_i E_i(x)||, ref8 weights.

| layer | k4_active | k4_ref8 | k4_oracle_ls | k4_contrib_ref8 | k5_ref8 | k6_active | k6_ref8 | k3_ref8 | k2_ref8 |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 0.230 | 0.054 | 0.053 | 0.046 | 0.039 | 0.097 | 0.029 | 0.071 | 0.112 |
| 4 | 0.383 | 0.216 | 0.215 | 0.198 | 0.170 | 0.181 | 0.130 | 0.276 | 0.359 |
| 5 | 0.428 | 0.261 | 0.260 | 0.247 | 0.207 | 0.204 | 0.157 | 0.321 | 0.412 |
| 6 | 0.358 | 0.214 | 0.213 | 0.203 | 0.170 | 0.170 | 0.128 | 0.277 | 0.380 |
| 7 | 0.451 | 0.273 | 0.272 | 0.248 | 0.221 | 0.224 | 0.171 | 0.339 | 0.424 |
| 8 | 0.459 | 0.269 | 0.268 | 0.242 | 0.212 | 0.216 | 0.160 | 0.341 | 0.443 |
| 9 | 0.416 | 0.249 | 0.249 | 0.235 | 0.195 | 0.195 | 0.147 | 0.326 | 0.431 |
| 10 | 0.365 | 0.215 | 0.214 | 0.205 | 0.161 | 0.163 | 0.120 | 0.290 | 0.421 |
| 11 | 0.452 | 0.270 | 0.269 | 0.248 | 0.213 | 0.208 | 0.156 | 0.343 | 0.449 |
| 12 | 0.554 | 0.354 | 0.351 | 0.320 | 0.282 | 0.261 | 0.207 | 0.440 | 0.556 |
| 13 | 0.566 | 0.356 | 0.352 | 0.334 | 0.279 | 0.268 | 0.210 | 0.445 | 0.565 |
| 14 | 0.542 | 0.342 | 0.340 | 0.318 | 0.271 | 0.258 | 0.201 | 0.429 | 0.542 |
| 15 | 0.490 | 0.300 | 0.298 | 0.284 | 0.231 | 0.226 | 0.173 | 0.386 | 0.505 |
| 16 | 0.547 | 0.351 | 0.349 | 0.332 | 0.276 | 0.258 | 0.206 | 0.438 | 0.560 |
| 17 | 0.606 | 0.397 | 0.392 | 0.375 | 0.317 | 0.296 | 0.240 | 0.486 | 0.604 |
| 18 | 0.597 | 0.392 | 0.388 | 0.371 | 0.312 | 0.294 | 0.239 | 0.482 | 0.594 |
| 19 | 0.620 | 0.414 | 0.411 | 0.391 | 0.328 | 0.298 | 0.244 | 0.513 | 0.621 |
| 20 | 0.669 | 0.446 | 0.442 | 0.423 | 0.362 | 0.335 | 0.277 | 0.538 | 0.645 |
| 21 | 0.687 | 0.467 | 0.464 | 0.430 | 0.383 | 0.353 | 0.296 | 0.561 | 0.668 |
| 22 | 0.623 | 0.406 | 0.404 | 0.387 | 0.326 | 0.306 | 0.249 | 0.505 | 0.617 |
| 23 | 0.630 | 0.407 | 0.405 | 0.384 | 0.324 | 0.304 | 0.245 | 0.510 | 0.618 |
| 24 | 0.623 | 0.389 | 0.387 | 0.359 | 0.312 | 0.305 | 0.240 | 0.478 | 0.591 |
| 25 | 0.548 | 0.343 | 0.341 | 0.325 | 0.272 | 0.269 | 0.212 | 0.436 | 0.548 |
| 26 | 0.558 | 0.357 | 0.356 | 0.335 | 0.279 | 0.260 | 0.207 | 0.446 | 0.564 |
| 27 | 0.546 | 0.345 | 0.345 | 0.317 | 0.277 | 0.260 | 0.206 | 0.431 | 0.548 |
| 28 | 0.631 | 0.405 | 0.404 | 0.377 | 0.327 | 0.318 | 0.256 | 0.489 | 0.602 |
| 29 | 0.571 | 0.331 | 0.330 | 0.295 | 0.267 | 0.270 | 0.204 | 0.408 | 0.513 |
| 30 | 0.571 | 0.332 | 0.331 | 0.299 | 0.268 | 0.272 | 0.205 | 0.410 | 0.516 |
| 31 | 0.607 | 0.373 | 0.373 | 0.339 | 0.301 | 0.297 | 0.233 | 0.445 | 0.539 |
| 32 | 0.608 | 0.359 | 0.359 | 0.334 | 0.296 | 0.300 | 0.233 | 0.435 | 0.532 |
| 33 | 0.582 | 0.353 | 0.352 | 0.322 | 0.285 | 0.284 | 0.220 | 0.429 | 0.540 |
| 34 | 0.532 | 0.314 | 0.314 | 0.293 | 0.254 | 0.255 | 0.195 | 0.391 | 0.497 |
| 35 | 0.560 | 0.336 | 0.335 | 0.313 | 0.271 | 0.269 | 0.208 | 0.414 | 0.516 |
| 36 | 0.544 | 0.325 | 0.325 | 0.313 | 0.257 | 0.257 | 0.198 | 0.412 | 0.528 |
| 37 | 0.582 | 0.346 | 0.346 | 0.324 | 0.277 | 0.269 | 0.207 | 0.432 | 0.546 |
| 38 | 0.589 | 0.363 | 0.363 | 0.339 | 0.294 | 0.289 | 0.228 | 0.443 | 0.545 |
| 39 | 0.576 | 0.350 | 0.349 | 0.333 | 0.287 | 0.283 | 0.221 | 0.429 | 0.538 |
| 40 | 0.584 | 0.341 | 0.341 | 0.315 | 0.276 | 0.277 | 0.211 | 0.429 | 0.538 |
| 41 | 0.642 | 0.372 | 0.371 | 0.331 | 0.302 | 0.303 | 0.231 | 0.460 | 0.565 |
| 42 | 0.673 | 0.385 | 0.383 | 0.349 | 0.304 | 0.316 | 0.235 | 0.479 | 0.587 |
| 43 | 0.671 | 0.372 | 0.368 | 0.348 | 0.298 | 0.307 | 0.226 | 0.455 | 0.569 |
| 44 | 0.615 | 0.350 | 0.339 | 0.300 | 0.279 | 0.282 | 0.205 | 0.437 | 0.554 |
| **mean** | **0.550** | **0.336** | **0.334** | **0.311** | **0.268** | **0.263** | **0.204** | **0.417** | **0.524** |

Findings
- **Renormalising the kept experts is the largest avoidable error.** Top-4 with ref8 weights cuts the mean layer error from 0.550 to 0.336 (-39%) with no training. It is within 0.002 of the least-squares oracle, so no better re-weighting exists for a fixed top-4.
- Choosing the 4 experts by actual contribution helps only modestly (0.336 -> 0.311). The remaining error is the dropped experts' contribution, which needs training (router KD, dropped-expert compensation).
- **k5_ref8 (0.268) ~ k6_active (0.263)**: one fewer expert at the same layer error.
- Strong layer dependence: layer 3 tolerates even k2_ref8 (0.112); layers 17-24 are hardest (k4_ref8 0.39-0.47). Per-layer k is a natural lever.
- Model-level effect of ref8 is measured in M2 round 3 (hook v5 implements ref8 exactly as y' = r*y + (1-r)*shared with TP-aware reduction of the shared-expert output).

## Linear dropped-expert compensation: does not generalize across domains (negative result)
Capture v2: 120 calibration prompts (20 per domain, 64 generated tokens), ~12K tokens per layer. `src/compensation_probe.py` fits y_k + X W P (ridge lambda in {1, 1e2, 1e3, 1e4, 1e5}, reduced rank in {8, 16, 32, 64, 128}) to the routed output of the original top-8, for k = 4 and 5 with original-top-8 normalisation. Two held-out splits: **random** tokens (same prompts/domains as training) and **tail** (the last 25% of prompts, i.e. unseen prompts and mostly unseen domains). Values are the best configuration per split, as the fraction of the k_ref8 routed-output error recovered.

| layer | k4 random | k4 tail | k5 random | k5 tail |
|---|---|---|---|---|
| 4 | 34% | 5% | 34% | 5% |
| 12 | 23% | 3% | 22% | 1% |
| 20 | 21% | 4% | 21% | 3% |
| 32 | 21% | 2% | 20% | 2% |
| 44 | 27% | 11% | 27% | 9% |

- In-distribution, a rank-64/128 linear term recovers 20-34% of the dropped-expert error. On unseen prompts/domains it recovers only 1-11%, so the linear term mostly memorises prompt- and domain-specific structure. For a general-purpose model, linear compensation is **not** a robust lever.
- The earlier first-probe numbers (20-35% on 1.8K tokens) were noise/overfit and are superseded.
- Next: learned non-linear capacity evaluated on held-out domains (a distilled per-layer "dropped-expert" MLP), and router distillation.

## Non-linear distilled compensation also fails to generalize (negative result)
`src/nonlinear_compensation.py`: per layer, a 4096 -> h -> 4096 SiLU MLP trained (AdamW, early stopping on a random 10% of the training prompts) to predict the dropped-expert residual for k = 4 (original-top-8 normalisation), on the first 75% of capture-v2 prompts, evaluated on the last 25% (unseen prompts/domains).

| layer | k4_ref8 error | MLP h=256 (2.1M params) recovered | MLP h=1024 (8.4M params) recovered |
|---|---|---|---|
| 12 | 0.311 | 4.1% | 3.5% |
| 20 | 0.369 | 0.8% | 0.2% |
| 32 | 0.356 | -1.0% | -1.7% |

- At most 4% recovered, and negative in layer 32. With ~8K training tokens per layer, neither linear nor small non-linear compensation can predict what the dropped experts contribute on unseen domains: the information is token-specific.
- Caveat: distillation at much larger data scale (millions of tokens, full end-to-end KD) is untested and could behave differently.
