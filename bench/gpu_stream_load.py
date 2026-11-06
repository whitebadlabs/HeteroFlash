#!/usr/bin/env python3
"""Keep the GPU streaming memory (bf16 read-reduce over 1 GiB) for --seconds; print achieved GB/s per 2 s window."""
import sys
import time
import torch
secs = float(sys.argv[1]) if len(sys.argv) > 1 else 30
a = torch.ones(512 * 1024 * 1024, dtype=torch.bfloat16, device="cuda")
end = time.time() + secs
while time.time() < end:
    torch.cuda.synchronize(); t = time.perf_counter(); n = 0
    while time.perf_counter() - t < 2:
        a.sum(dtype=torch.float32); n += 1
    torch.cuda.synchronize()
    print(f"gpu_read_GBps {n * a.numel() * 2 / (time.perf_counter() - t) / 1e9:.1f}", flush=True)
