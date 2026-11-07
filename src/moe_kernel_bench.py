"""GB10 MoE prefill kernel bench: vLLM fused_experts throughput for one GLM-5.3-Flash MoE layer.

288 experts, hidden 4096, expert intermediate 2048, top-8. Reports effective TFLOPS (2*M*topk*3*H*I / t) so it can be
compared with llama.cpp's Q4_K MMQ path measured in the E2E-v1 nsys trace (~7 TFLOPS).
Run inside a vLLM container on the Spark: python3 moe_kernel_bench.py --out results.json
"""
import argparse
import json
import time

import torch

E, H, I, K = 288, 4096, 2048, 8


def timeit(fn, iters=10, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters


def routing(m, dev):
    logits = torch.randn(m, E, device=dev, dtype=torch.float32)
    w, ids = torch.topk(torch.softmax(logits, -1), K, dim=-1)
    return (w / w.sum(-1, keepdim=True)).to(torch.float32), ids.to(torch.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda"
    from vllm.model_executor.layers.fused_moe import fused_experts
    res = {"gpu": torch.cuda.get_device_name(), "E": E, "H": H, "I": I, "topk": K, "rows": []}
    w1 = (torch.randn(E, 2 * I, H, device=dev, dtype=torch.bfloat16) * 0.02)
    w2 = (torch.randn(E, H, I, device=dev, dtype=torch.bfloat16) * 0.02)
    # dense FP8 GEMM roofline for reference
    try:
        A = torch.randn(4096, H, device=dev).to(torch.float8_e4m3fn)
        B = torch.randn(2 * I, H, device=dev).to(torch.float8_e4m3fn).t()
        one = torch.ones((), device=dev)
        t = timeit(lambda: torch._scaled_mm(A, B, scale_a=one, scale_b=one, out_dtype=torch.bfloat16))
        res["dense_fp8_gemm_tflops"] = round(2 * 4096 * H * 2 * I / t / 1e12, 1)
    except Exception as e:  # noqa: BLE001
        res["dense_fp8_gemm_error"] = repr(e)[:200]
    tb = torch.randn(4096, H, device=dev, dtype=torch.bfloat16)
    t = timeit(lambda: tb @ w1[0].t())
    res["dense_bf16_gemm_tflops"] = round(2 * 4096 * H * 2 * I / t / 1e12, 1)
    # FP8 per-tensor weights for the fused path
    w1_8, w2_8 = w1.to(torch.float8_e4m3fn), w2.to(torch.float8_e4m3fn)
    s1 = torch.ones(E, device=dev, dtype=torch.float32)
    s2 = torch.ones(E, device=dev, dtype=torch.float32)
    fp8_cfg = None
    try:
        from vllm.model_executor.layers.fused_moe.config import fp8_w8a8_moe_quant_config
        fp8_cfg = fp8_w8a8_moe_quant_config(w1_scale=s1, w2_scale=s2, a1_scale=None, a2_scale=None)
    except Exception as e:  # noqa: BLE001
        res["fp8_cfg_error"] = repr(e)[:200]
    for m in a.tokens:
        x = torch.randn(m, H, device=dev, dtype=torch.bfloat16)
        tw, ti = routing(m, dev)
        flop = 2 * m * K * 3 * H * I
        row = {"tokens": m}
        try:
            t = timeit(lambda: fused_experts(x, w1, w2, tw, ti))
            row["bf16_ms"], row["bf16_tflops"] = round(t * 1e3, 2), round(flop / t / 1e12, 1)
        except Exception as e:  # noqa: BLE001
            row["bf16_error"] = repr(e)[:200]
        if fp8_cfg is not None:
            try:
                t = timeit(lambda: fused_experts(x, w1_8, w2_8, tw, ti, quant_config=fp8_cfg))
                row["fp8_ms"], row["fp8_tflops"] = round(t * 1e3, 2), round(flop / t / 1e12, 1)
            except Exception as e:  # noqa: BLE001
                row["fp8_error"] = repr(e)[:200]
        res["rows"].append(row)
        print(json.dumps(row), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}))


if __name__ == "__main__":
    main()
