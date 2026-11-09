# FP4 (NVFP4) fixed-weight PTQ sensitivity on our stack, balanced eval vs Q8_0 BF16 proxy [M, 2026-11-09]

**Method.** The trace hook (`HF_FQ_NO_RESTORE=1`) fake-quantizes BF16 fixed weights in place with NVFP4: 16-element blocks, E4M3 block scales, a global scale. One spec per boot.

| run | modules (GB BF16) | dNLL | KLD | same-top | increment vs base |
|---|---|---|---|---|---|
| GPU-only IQ2_XXS experts, BF16 fixed (base) | 0 | +0.1839 +/- 0.0042 | 0.237 | 85.07% | - |
| + NVFP4 lm_head (spec meant shared/dense/lm_head; the regex matched only lm_head) | 1 (1.27) | +0.1876 | 0.242 | 84.61% | +0.0037 |
| + NVFP4 attention (KDA + MLA projections) | 214 (11.94) | +0.1960 | 0.251 | 84.36% | +0.0121 |
| + NVFP4 attention + lm_head + dense MLP (shared experts **not** included: the hook's exclude pattern `experts\.` drops `shared_experts.*`) | 221 (14.12) | +0.1987 | 0.256 | 84.13% | +0.0148 |
| **config B (CPU Q4_K experts) + BF16 fixed + NVFP4 attention** | 214 | **+0.0173 +/- 0.0018** | **0.046** | **93.67%** | about +0.010 over B (FP8 fixed, +0.0069) |
| control (EXL3 3.5 bpw, 2x Max-Q) | | +0.0148 +/- 0.0017 | 0.042 | 94.0% | |

**Reading.**

- **The increment is smaller here than in M2.** NVFP4 attention costs about +0.010-0.012 nats here, against +0.030 in M2. M2 used an older prose-heavy eval and measured against the control model.
- **B with NVFP4 attention misses the bar by about 0.0025 nats** (+0.0173 vs +0.0148), and its KLD is 0.046 vs 0.042.
- **Recovery needed to meet the bar:**
  - about 25% of the attention increment at point estimates;
  - about 50% to keep a margin (target +0.012).
- **lm_head NVFP4 costs +0.0037.**
- **Shared experts at NVFP4 are untested**; M2 measured +0.008.
- **Untested caveat:** MLA `kv_b_proj` is absorbed at load, so its precision is unaffected by the hook.
