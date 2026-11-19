#!/usr/bin/env python3
"""Unit test (2026-11-18) of mean_sub (hook_mean/hf_drt.cu) vs torch, eager and in a CUDA graph with the flag switched."""
import json, os, sys, torch
from torch.utils.cpp_extension import load
hd = sys.argv[1]
ext = load(name="hf_drt_mean", sources=[os.path.join(hd, "hf_drt.cu")], extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=False)
torch.manual_seed(0); dev = "cuda"; res = []
mean = (torch.randn(288, 4096, device=dev) * 0.05).to(torch.bfloat16).contiguous()
for T in (1, 4, 16):
    for dt in (torch.int32, torch.int64):
        w = torch.rand(T, 8, device=dev); ids_o = torch.stack([torch.randperm(288, device=dev)[:8] for _ in range(T)]).to(dt).contiguous()
        kmax = torch.full((42,), 5, dtype=torch.int32, device=dev); tr = torch.tensor([0.0, 0.0, 0.0], device=dev)
        ids = torch.empty_like(ids_o); w2 = torch.empty_like(w)
        ext.drop(ids_o, w.contiguous(), kmax, 3, tr, ids, w2)
        y0 = torch.randn(T, 4096, device=dev); y = y0.clone()
        flag = torch.tensor([1.0], device=dev)
        ext.mean_sub(ids_o, ids, w.contiguous(), mean, flag, y)
        dropped = (ids < 0) & (ids_o >= 0)
        ref = y0 + (torch.where(dropped, w, 0 * w)[:, :, None] * mean.float()[ids_o.long()]).sum(1)
        err = float((y - ref).abs().max())
        res.append({"T": T, "dtype": str(dt), "dropped_per_tok": float(dropped.sum()) / T, "max_abs_err": err, "ok": err < 1e-4})
        y = y0.clone(); flag.zero_(); ext.mean_sub(ids_o, ids, w.contiguous(), mean, flag, y)
        res.append({"T": T, "flag0_unchanged": bool(torch.equal(y, y0)), "ok": bool(torch.equal(y, y0))})
# graph: switch flag at run time
T = 4; w = torch.rand(T, 8, device=dev); ids_o = torch.stack([torch.randperm(288, device=dev)[:8] for _ in range(T)]).contiguous()
ids = ids_o.clone(); ids[:, 5:] = -1; y = torch.zeros(T, 4096, device=dev); flag = torch.tensor([0.0], device=dev)
s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s): ext.mean_sub(ids_o, ids, w, mean, flag, y)
torch.cuda.current_stream().wait_stream(s)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g): ext.mean_sub(ids_o, ids, w, mean, flag, y)
y.zero_(); g.replay(); a0 = float(y.abs().sum()); flag.fill_(1.0); y.zero_(); g.replay(); a1 = float(y.abs().sum())
res.append({"graph_switch": [a0, a1], "ok": a0 == 0 and a1 > 0})
print(json.dumps({"all_ok": all(r["ok"] for r in res), "cases": len(res)}))
for r in res: print(r)
