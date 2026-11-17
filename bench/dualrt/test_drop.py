#!/usr/bin/env python3
"""Unit test of the expert-drop kernel (hf_drt.cu drop) against a torch reference, int32 and int64 ids."""
import json, os, torch
from torch.utils.cpp_extension import load
ext = load(name="hf_drt", sources=[os.path.join(os.path.dirname(os.path.abspath(__file__)), "hook", "hf_drt.cu")],
           extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=False)
torch.manual_seed(0); dev = "cuda"; res = []
for T in (1, 4, 16, 300):
    w = torch.rand(T, 8, device=dev) ** 3; w = w / w.sum(1, keepdim=True) * 2.5
    ids = torch.stack([torch.randperm(288, device=dev)[:8] for _ in range(T)])
    for dt in (torch.int32, torch.int64):
        for K, tau, rn in ((8, 0.0, 0), (6, 0.0, 0), (8, 0.05, 0), (5, 0.08, 1), (2, 0.0, 1)):
            kmax = torch.full((42,), K, dtype=torch.int32, device=dev); kmax[7] = 3
            tr = torch.tensor([tau, float(rn)], device=dev)
            i_in = ids.to(dt).contiguous(); oi = torch.empty_like(i_in); ow = torch.empty_like(w)
            ext.drop(i_in, w.contiguous(), kmax, 4, tr, oi, ow)
            rank = torch.argsort(torch.argsort(w, dim=1, descending=True, stable=True), dim=1)
            keep = (rank < K) & (w >= tau * w.sum(1, keepdim=True))
            rw = torch.where(keep, w, 0 * w)
            if rn: rw = rw * w.sum(1, keepdim=True) / rw.sum(1, keepdim=True)
            ri = torch.where(keep, i_in, torch.full_like(i_in, -1))
            ok = bool(torch.equal(oi, ri)) and float((ow - rw).abs().max()) < 1e-5
            res.append({"T": T, "dtype": str(dt), "K": K, "tau": tau, "renorm": rn, "kept_per_tok": round(float(keep.sum()) / T, 2), "ok": ok})
            ext.drop(i_in, w.contiguous(), kmax, 7, tr, oi, ow)   # per-layer kmax entry
            assert int((oi >= 0).sum(1).max()) <= 3
print(json.dumps({"all_ok": all(r["ok"] for r in res), "cases": len(res)}))
for r in res:
    if not r["ok"]: print(r)
