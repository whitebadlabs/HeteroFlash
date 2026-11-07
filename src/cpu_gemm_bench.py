"""Workstation CPU GEMM roofline for remote-expert prefill: one expert FFN (4096 -> 2x2048 -> 4096) at M tokens/expert."""
import json, sys, time
import torch
torch.set_num_threads(int(sys.argv[1]) if len(sys.argv) > 1 else 32)
H, I = 4096, 2048
def tm(f, it=5):
    f(); t = time.perf_counter()
    for _ in range(it): f()
    return (time.perf_counter() - t) / it
out = {"threads": torch.get_num_threads(), "rows": []}
w1 = torch.randn(2 * I, H, dtype=torch.bfloat16); w2 = torch.randn(H, I, dtype=torch.bfloat16)
w1i = torch.randint(-127, 127, (H, 2 * I), dtype=torch.int8); w2i = torch.randint(-127, 127, (I, H), dtype=torch.int8)
for m in (16, 64, 128, 512, 2048):
    x = torch.randn(m, H, dtype=torch.bfloat16); flop = 2 * m * 3 * H * I
    t = tm(lambda: (x @ w1.t())[:, :I] @ w2.t())
    r = {"m": m, "bf16_tflops": round(flop / t / 1e12, 2)}
    try:
        xi = torch.randint(-127, 127, (max(m, 17), H), dtype=torch.int8)
        hi = torch.randint(-127, 127, (max(m, 17), I), dtype=torch.int8)
        t = tm(lambda: (torch._int_mm(xi, w1i), torch._int_mm(hi, w2i)))
        r["int8_tops"] = round(2 * max(m, 17) * 3 * H * I / t / 1e12, 2)
    except Exception as e:
        r["int8_error"] = repr(e)[:120]
    out["rows"].append(r); print(json.dumps(r), flush=True)
