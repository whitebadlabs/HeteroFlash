# NVFP4 sensitivity by attention group (cumulative, one boot) [M, 2026-11-09]

Base: GPU-only IQ2_XXS experts, BF16 fixed weights. Weights are fake-quantized in place (`HF_FQ_NO_RESTORE`). Balanced eval against the Q8_0 BF16 proxy.

| pass | dNLL | KLD | same-top | increment |
|---|---|---|---|---|
| c0 none | +0.1809 | 0.2362 | 84.83% | |
| c1 + MLA attention (11 layers, 44 modules, 2.58 GB BF16) | +0.1872 | 0.2424 | 84.69% | **+0.0063** |
| c2 + KDA attention, layers 0-21 (17 layers) | +0.1892 | 0.2430 | 84.63% | **+0.0020** |
| c3 + KDA attention, layers 22-44 (17 layers) | +0.1973 | 0.2512 | 84.37% | **+0.0081** |
| c4 + lm_head | +0.1998 | 0.2554 | 84.09% | **+0.0025** |

**Noise.** The no-FP4 baseline was +0.1839 in an earlier boot, so increments carry roughly +/-0.003.

**Reading.**

- MLA attention and the later KDA layers carry almost all of the FP4 sensitivity. The early KDA layers (0-21) are nearly free.
- **Training-free mix:** NVFP4 on KDA layers 0-21 and lm_head, FP8 elsewhere.
  - Estimated +0.0045 on top of B (+0.0069), so about +0.011 against the +0.0148 bar.
  - It cuts about 1.35 GB of about 11.8 GB of fixed-path reads per MTP-3 cycle.
- **All-attention NVFP4** (about 2.7 GB saved) costs about +0.010-0.016. That needs recovery (QAD) to fit the bar.
