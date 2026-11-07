# CPU-tier expert quantization error on the official BF16 weights (2026-11-07)

`src/quantize_experts_int4.py measure` on zai-org/GLM-5.3-Flash-BF16 (rev f9fed3b), 12 experts (layers 3, 20, 44 x experts 0, 57, 144, 287). Group size 128 along the input dimension. FFN output error uses 16 random N(0,1) inputs. That overstates error relative to real activations and serves only as a format comparison; model-level KL comes in G2.

| Format | gate rel err | up rel err | down rel err | expert FFN output rel err | Bytes vs int4 | CPU us/expert (layer-batched, measured) |
|---|---|---|---|---|---|---|
| int4 symmetric RTN | 12.08% | 12.09% | 12.19% | 21.56% | 1x | 73.5 |
| int4 + per-group clip search (MSE) | 10.07% | 10.06% | 10.15% | 16.96% | 1x | 73.5 |
| int8 symmetric | 0.68% | 0.68% | 0.68% | 1.17% | ~1.9x | ~101 (v2 dot: 101-110) |

- Error is uniform across layers and experts (std of per-expert values < 0.5 pp), so no layer stands out at the weight level. Activation-aware sensitivity (GPTQ/AWQ with captured expert inputs) is the next measurement.
- Design implication: workstation RAM is plentiful, so cold experts can be stored in int8 (near-lossless) at ~1.38x CPU time per expert. Placement then trades GPU-resident experts against CPU precision: int8 for all 50%-cold experts needs ~162 GB, which fits in ~230 GB.

## End-to-end through the expert server on real weights
`runtime/expert_server.c serve-weights` loaded layers 20 and 44 (all 288 experts each, int4 RTN g128, 4.08 GB per layer, exported in ~35 s per layer). `runtime/check_real_weights.py` sent 3-token requests over the socket and compared:

| Request | server vs same int4 weights (numpy) | server vs original BF16 | int4 weights alone vs BF16 |
|---|---|---|---|
| layer 20, experts 0/57/144/287 | 2.48% | 21.2% | 21.1% |
| layer 44, experts 3/99 | 3.09% | 22.1% | 21.8% |

- The serving path is correct. Its own extra error (2.5-3.1%) comes from per-token u8 activation quantization (absmax over 4096 inputs and over the post-SiLU hidden vector) plus bf16 output rounding. Next: per-128-group activation scales.
- Total error is dominated by int4 RTN weights. That motivates clip-search/GPTQ int4 or int8 for the CPU tier (see table above), decided by model-level KL in G2.
- First request 5.6 ms (cold pages), then 0.27 ms for 2 experts x 3 tokens.

## int8 CPU tier (expert server `HFQ8` files)
The int8 kernel path (signed int8 weights, per-128 scales, the +128 activation offset folded into one constant per group) is selftest-verified: the same integer weights stored as int8 reproduce the int4 path to 1.1e-6. Real layers 20 and 44, exported in int8 (7.70 GB per layer, ~28 s each) and served over the socket:

| Request | server vs original BF16 (int8) | (int4 for comparison) |
|---|---|---|
| layer 20, experts 0/57/144/287, 3 tokens | **1.95%** | 21.2% |
| layer 44, experts 3/99, 3 tokens | **2.15%** | 22.1% |

- int8 cold experts are ~10x more accurate end to end; the remaining ~2% is per-group activation quantization plus bf16 I/O.
- All 42 layers in int8 would need 323 GB, more than the ~230 GB of workstation RAM, so the CPU tier is mixed int8/int4 by measured sensitivity and usage. The placement planner should allocate it.
- The int4 export of all 42 layers (160 GB, int4 RTN g128) is complete.
