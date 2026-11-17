#!/usr/bin/env python3
"""Timestamped GPU read-bandwidth stream (bf16 read-reduce over a 2 GiB buffer).
   stream_gpu.py <start_epoch> <duration_s> <window_s> <tag>   ->  "<tag> <epoch_mid> <GB/s>" per window."""
import sys, time, torch
st, dur, win, tag = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
a = torch.ones(1024 * 1024 * 1024, dtype=torch.bfloat16, device="cuda")
for _ in range(5): a.sum(dtype=torch.float32)
torch.cuda.synchronize()
print(f"# {tag} ready, start in {st - time.time():.1f}s", file=sys.stderr, flush=True)
while time.time() < st: time.sleep(0.001)
end = st + dur
while time.time() < end:
    t0 = time.time(); n = 0
    while time.time() - t0 < win:
        a.sum(dtype=torch.float32); n += 1
        if n % 4 == 0: torch.cuda.synchronize()
    torch.cuda.synchronize(); t1 = time.time()
    print(f"{tag} {(t0 + t1) / 2:.3f} {n * a.numel() * 2 / (t1 - t0) / 1e9:.1f}", flush=True)
