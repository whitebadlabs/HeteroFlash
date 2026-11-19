#!/usr/bin/env python3
"""Unit test (2026-11-18) of the union-aware drop (hook_union/hf_drt.cu drop with tau_rn[3]) vs a torch reference."""
import json, os, sys, torch
from torch.utils.cpp_extension import load
hd = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "hook_union")
ext = load(name="hf_drt_union", sources=[os.path.join(hd, "hf_drt.cu")], extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=False)
torch.manual_seed(0); dev = "cuda"; res = []
for T in (1, 4, 16, 300):
    w = torch.rand(T, 8, device=dev) ** 3; w = w / w.sum(1, keepdim=True) * 2.5
    ids = torch.stack([torch.randperm(40, device=dev)[:8] for _ in range(T)])   # 40 experts: unions overlap a lot
    for dt in (torch.int32, torch.int64):
        for K, tau, rn, uni in ((8, 0.0, 0, 0), (6, 0.0, 0, 0), (5, 0.08, 1, 0), (4, 0.0, 0, 1), (5, 0.0, 1, 1), (3, 0.05, 0, 1), (2, 0.0, 1, 1)):
            kmax = torch.full((42,), K, dtype=torch.int32, device=dev)
            tr = torch.tensor([tau, float(rn), float(uni)], device=dev)
            i_in = ids.to(dt).contiguous(); oi = torch.empty_like(i_in); ow = torch.empty_like(w)
            ext.drop(i_in, w.contiguous(), kmax, 4, tr, oi, ow)
            rank = torch.argsort(torch.argsort(w, dim=1, descending=True, stable=True), dim=1)
            core = (rank < K) & (w >= tau * w.sum(1, keepdim=True))
            keep = core.clone()
            if uni and T <= 16:
                present = torch.zeros(288, dtype=torch.bool, device=dev); present[i_in[core].long()] = True
                keep = core | present[i_in.long()]
            rw = torch.where(keep, w, 0 * w)
            if rn: rw = rw * w.sum(1, keepdim=True) / rw.sum(1, keepdim=True)
            ri = torch.where(keep, i_in, torch.full_like(i_in, -1))
            ok = bool(torch.equal(oi, ri)) and float((ow - rw).abs().max()) < 1e-5
            # union invariant: experts read == core union
            if uni and T <= 16:
                ok = ok and set(oi[oi >= 0].tolist()) == set(i_in[core].tolist())
            res.append({"T": T, "dtype": str(dt), "K": K, "tau": tau, "rn": rn, "uni": uni,
                        "core_per_tok": round(float(core.sum()) / T, 2), "kept_per_tok": round(float(keep.sum()) / T, 2), "ok": ok})
    # graph capture: union flag switched at run time without recapture
    kmax = torch.full((42,), 4, dtype=torch.int32, device=dev); tr = torch.tensor([0.0, 0.0, 0.0], device=dev)
    if T <= 16:
        i_in = ids.to(torch.int64).contiguous(); oi = torch.empty_like(i_in); ow = torch.empty_like(w); wc = w.contiguous()
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s): ext.drop(i_in, wc, kmax, 4, tr, oi, ow)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g): ext.drop(i_in, wc, kmax, 4, tr, oi, ow)
        g.replay(); n0 = int((oi >= 0).sum()); tr[2] = 1.0; g.replay(); n1 = int((oi >= 0).sum())
        res.append({"T": T, "graph_switch": [n0, n1], "ok": n1 >= n0 and n0 == 4 * T})
print(json.dumps({"all_ok": all(r["ok"] for r in res), "cases": len(res)}))
for r in res:
    if not r["ok"] or "graph_switch" in r or r.get("uni"): print(r)
