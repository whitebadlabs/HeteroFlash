# CPU-tier expert quantization error on the official BF16 weights (2026-11-07)

`src/quantize_experts_int4.py measure` on zai-org/GLM-5.3-Flash-BF16 (rev f9fed3b), 12 experts (layers 3, 20, 44 x experts 0, 57, 144, 287). Group size 128 along the input dimension. FFN output error uses 16 random N(0,1) inputs. That overstates error relative to real activations and serves only as a format comparison; model-level KL comes in G2.

| Format | gate rel err | up rel err | down rel err | expert FFN output rel err | Bytes vs int4 | CPU us/expert (layer-batched, measured) |
|---|---|---|---|---|---|---|
| int4 symmetric RTN | 12.08% | 12.09% | 12.19% | 21.56% | 1x | 73.5 |
| int4 + per-group clip search (MSE) | 10.07% | 10.06% | 10.15% | 16.96% | 1x | 73.5 |
| int8 symmetric | 0.68% | 0.68% | 0.68% | 1.17% | ~1.9x | ~101 (v2 dot: 101-110) |

- Error is uniform across layers and experts (std of per-expert values < 0.5 pp), so no layer stands out at the weight level. Activation-aware sensitivity (GPTQ/AWQ with captured expert inputs) is the next measurement.
- Design implication: workstation RAM is plentiful, so cold experts can be stored in int8 (near-lossless) at ~1.38x CPU time per expert. Placement then trades GPU-resident experts against CPU precision: int8 for all 50%-cold experts needs ~162 GB, which fits in ~230 GB.
