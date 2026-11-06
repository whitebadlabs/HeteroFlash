#!/usr/bin/env python3
"""Spark GB10 GPU microbenchmarks at GLM-5.3-Flash expert shapes.

1. Device memory bandwidth (read-reduce and copy).
2. Decode-shape GEMV/skinny GEMM: [T x 4096] @ [4096 x 4096] (gate+up of 1 expert = 2x2048 rows) in BF16, FP8.
3. Prefill-shape GEMM: [T x 4096] @ [4096 x 2048] for T in 64..8192, BF16 / FP8 (torch._scaled_mm) /
   NVFP4 (vLLM cutlass_scaled_fp4_mm, if the build exposes it on this GPU).
Keeps total allocation small (< ~8 GB) so it can run beside a resident model.
Prints one JSON object.
"""
import json
import time

import torch

dev = torch.device("cuda")
out = {"device": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0)),
       "torch": torch.__version__}


def timeit(fn, iters=50, warm=10):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters


# 1. bandwidth
n = 512 * 1024 * 1024  # 2 GiB of fp32... use 1 GiB bf16 buffers
a = torch.empty(n, dtype=torch.bfloat16, device=dev).fill_(1)
b = torch.empty_like(a)
t = timeit(lambda: a.sum(dtype=torch.float32), 30)
out["read_reduce_GBps"] = round(a.numel() * 2 / t / 1e9, 1)
t = timeit(lambda: b.copy_(a), 30)
out["copy_GBps_read_plus_write"] = round(2 * a.numel() * 2 / t / 1e9, 1)
del a, b
torch.cuda.empty_cache()

# 2. decode-shape: weights cycled over a pool larger than L2 so reads hit DRAM
H, I = 4096, 2048
pool = 24
Wb = [torch.randn(2 * I, H, dtype=torch.bfloat16, device=dev) for _ in range(pool)]
res = {}
for T in (1, 4, 8):
    x = torch.randn(T, H, dtype=torch.bfloat16, device=dev)
    k = [0]

    def f():
        k[0] = (k[0] + 1) % pool
        return x @ Wb[k[0]].t()
    t = timeit(f, 200, 20)
    res[f"bf16_T{T}"] = {"us": round(t * 1e6, 1), "weight_GBps": round(2 * I * H * 2 / t / 1e9, 1)}
fp8 = getattr(torch, "float8_e4m3fn", None)
if fp8 is not None:
    W8 = [w.to(fp8) for w in Wb]
    one = torch.ones((), device=dev, dtype=torch.float32)
    for T in (16, 32):  # _scaled_mm needs M multiple of 16
        x8 = torch.randn(T, H, device=dev).to(fp8)
        k = [0]

        def f8():
            k[0] = (k[0] + 1) % pool
            return torch._scaled_mm(x8, W8[k[0]].t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16)
        try:
            t = timeit(f8, 200, 20)
            res[f"fp8_T{T}"] = {"us": round(t * 1e6, 1), "weight_GBps": round(2 * I * H / t / 1e9, 1)}
        except Exception as e:  # noqa: BLE001
            res[f"fp8_T{T}"] = {"error": repr(e)[:200]}
    del W8
out["decode_gemv"] = res
del Wb
torch.cuda.empty_cache()

# 3. prefill-shape GEMM
W = torch.randn(I, H, dtype=torch.bfloat16, device=dev)
pre = {}
for T in (64, 256, 1024, 4096, 8192):
    x = torch.randn(T, H, dtype=torch.bfloat16, device=dev)
    t = timeit(lambda: x @ W.t(), 30, 5)
    pre[f"bf16_T{T}"] = round(2 * T * H * I / t / 1e12, 1)
    if fp8 is not None:
        x8, w8 = x.to(fp8), W.to(fp8)
        one = torch.ones((), device=dev, dtype=torch.float32)
        try:
            t = timeit(lambda: torch._scaled_mm(x8, w8.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16), 30, 5)
            pre[f"fp8_T{T}"] = round(2 * T * H * I / t / 1e12, 1)
        except Exception as e:  # noqa: BLE001
            pre[f"fp8_T{T}"] = repr(e)[:160]
# NVFP4 via vLLM custom ops if present
try:
    from vllm import _custom_ops as ops
    gs = torch.tensor([448.0 * 6.0 / W.abs().max().float()], device=dev)
    wq, wsc = ops.scaled_fp4_quant(W, gs)
    for T in (256, 1024, 4096, 8192):
        x = torch.randn(T, H, dtype=torch.bfloat16, device=dev)
        xs = torch.tensor([448.0 * 6.0 / x.abs().max().float()], device=dev)
        xq, xsc = ops.scaled_fp4_quant(x, xs)
        alpha = 1.0 / (xs * gs)
        t = timeit(lambda: ops.cutlass_scaled_fp4_mm(xq, wq, xsc, wsc, alpha, torch.bfloat16), 30, 5)
        pre[f"nvfp4_T{T}"] = round(2 * T * H * I / t / 1e12, 1)
except Exception as e:  # noqa: BLE001
    pre["nvfp4"] = "unavailable: " + repr(e)[:200]
out["prefill_gemm_TFLOPS_4096x2048"] = pre
out["max_mem_allocated_GB"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
print(json.dumps(out, indent=1))
