"""GB10 attention-projection GEMM: BF16 vs FP8 (torch._scaled_mm) vs NVFP4 (vLLM cutlass_scaled_fp4_mm).

Shapes are representative of GLM-5.3-Flash projections (hidden 4096); M covers decode (1), verify windows (4, 8) and
prefill chunks (512, 4096). Reports time, effective weight-read bandwidth (GB/s) and TFLOPS.
Run inside a vLLM container on the Spark: python3 attn_precision_bench.py --out results.json
"""
import argparse
import json
import time

import torch

SHAPES = [(4096, 4096), (4096, 12288), (12288, 4096), (4096, 2048)]  # (K in, N out)
MS = [1, 4, 8, 512, 4096]


def timeit(fn, iters=50, warm=10):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda"
    res = {"gpu": torch.cuda.get_device_name(), "rows": []}
    try:
        from vllm import _custom_ops as ops
        fp4_ok = hasattr(ops, "cutlass_scaled_fp4_mm") and hasattr(ops, "scaled_fp4_quant")
    except Exception as e:  # noqa: BLE001
        ops, fp4_ok = None, False
        res["fp4_import_error"] = repr(e)[:200]
    for K, N in SHAPES:
        wb = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02
        w8 = wb.to(torch.float8_e4m3fn)
        one = torch.ones((), device=dev, dtype=torch.float32)
        if fp4_ok:
            gs_w = (448 * 6 / wb.abs().max().float()).reshape(())
            w4, w4s = ops.scaled_fp4_quant(wb, gs_w)
        for M in MS:
            x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
            row = {"K": K, "N": N, "M": M}
            t = timeit(lambda: x @ wb.t())
            row["bf16_us"] = round(t * 1e6, 1); row["bf16_GBs"] = round(N * K * 2 / t / 1e9, 1)
            x8 = x.to(torch.float8_e4m3fn)
            mp = max(M, 16)  # scaled_mm wants M multiple of 16 on some backends
            x8p = torch.zeros(mp, K, device=dev, dtype=torch.float8_e4m3fn); x8p[:M] = x8
            try:
                t = timeit(lambda: torch._scaled_mm(x8p, w8.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16))
                row["fp8_us"] = round(t * 1e6, 1); row["fp8_GBs"] = round(N * K / t / 1e9, 1)
            except Exception as e:  # noqa: BLE001
                row["fp8_error"] = repr(e)[:160]
            if fp4_ok:
                try:
                    gs_x = (448 * 6 / x.abs().max().float()).reshape(())
                    alpha = (1 / (gs_x * gs_w)).reshape(())

                    def fp4():
                        xq, xs = ops.scaled_fp4_quant(x, gs_x)
                        return ops.cutlass_scaled_fp4_mm(xq, w4, xs, w4s, alpha, torch.bfloat16)
                    t = timeit(fp4)
                    row["nvfp4_us"] = round(t * 1e6, 1); row["nvfp4_GBs"] = round(N * K * 0.5625 / t / 1e9, 1)
                except Exception as e:  # noqa: BLE001
                    row["nvfp4_error"] = repr(e)[:160]
            for k in ("bf16", "fp8", "nvfp4"):
                if f"{k}_us" in row:
                    row[f"{k}_tflops"] = round(2 * M * N * K / (row[f"{k}_us"] * 1e-6) / 1e12, 1)
            res["rows"].append(row)
            print(json.dumps(row), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
