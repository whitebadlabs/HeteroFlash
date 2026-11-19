#!/usr/bin/env python3
"""Expert mean-output table (2026-11-18): mean over calibration picks of f_e(x) = down(silu(min(gate x,10)) * clip(up x,+-10))
per MoE layer (3..44) and routed expert, computed with the runtime's own GPU Q4_K kernel (hook_mean/hf_drt.cu q4k_moe) on the
same GGUF bytes. Input: HF_XCAP_DIR npz chunks (x bf16 bits, ids, w per layer). Output: fp16 npy [42, 288, 4096] + counts.
  python3 expert_means.py XCAP_DIR OUT.npy"""
import glob, json, os, struct, sys
import numpy as np, torch
from torch.utils.cpp_extension import load
HERE = os.path.dirname(os.path.abspath(__file__))
ext = load(name="hf_drt_mean", sources=[os.path.join(HERE, "hook_mean", "hf_drt.cu")], extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=False)
GGUF = os.environ.get("GGUF", "/gguf/GLM-5.3-Flash-Q8attn-G0_q4kimexp.gguf"); IDX = os.path.join(HERE, "q4k_experts.idx")
H, I, NE, NL = 4096, 2048, 288, 42
recs = {}
raw = open(IDX, "rb").read()
for i in range(len(raw) // 48):
    l, p, ty, rows, cols, ne, rb, off, nb = struct.unpack("<6i3Q", raw[i * 48:(i + 1) * 48]); recs[(l, p)] = (ty, rows, cols, ne, rb, off)
files = sorted(glob.glob(os.path.join(sys.argv[1], "xcap_*.npz")))
assert files, "no capture files"
table = np.zeros((NL, NE, H), np.float32); cnt = np.zeros((NL, NE), np.int64)
dev = "cuda"
glist = torch.zeros(64, dtype=torch.int32, device=dev); gcount = torch.zeros(1, dtype=torch.int32, device=dev)
gw = torch.zeros(64 * 16, device=dev); hbuf = torch.zeros(64 * 4 * I, device=dev); stats = torch.zeros(4, dtype=torch.int64, device=dev)
remote = torch.ones(NE, dtype=torch.bool, device=dev); AB = torch.tensor([1.0, 0.0], device=dev)
for li in range(NL):
    L = li + 3
    xs, ids = [], []
    for f in files:
        z = np.load(f)
        if f"x_{li}" in z.files:
            xs.append(z[f"x_{li}"]); ids.append(z[f"ids_{li}"].astype(np.int64))
    if not xs:
        continue
    X = torch.from_numpy(np.concatenate(xs).view(np.int16)).view(torch.bfloat16)   # [N, 4096]
    ID = np.concatenate(ids)                                                        # [N, 8]
    def blob(p):
        ty, rows, cols, ne, rb, off = recs[(L, p)]
        with open(GGUF, "rb") as fh:
            fh.seek(off); a = np.frombuffer(fh.read(ne * rows * rb), dtype=np.uint8)
        return torch.from_numpy(a.copy()).to(dev).view(ne, rows * rb)
    G, U, D = blob(0), blob(1), blob(2)
    PTRS = torch.tensor([G.data_ptr(), U.data_ptr(), D.data_ptr()], dtype=torch.int64, device=dev)
    slot_of = torch.arange(NE, dtype=torch.int32, device=dev)
    for e in range(NE):
        rows_e = np.nonzero((ID == e).any(1))[0]
        if len(rows_e) == 0:
            continue
        acc = torch.zeros(H, dtype=torch.float64, device=dev)
        for s0 in range(0, len(rows_e), 16):
            r = rows_e[s0:s0 + 16]; T = len(r)
            xb = X[torch.from_numpy(r)].to(dev).contiguous()
            ii = torch.full((T, 8), -1, dtype=torch.int32, device=dev); ii[:, 0] = e
            ww = torch.zeros(T, 8, device=dev); ww[:, 0] = 1.0
            y = torch.zeros(T, H, device=dev)
            ext.assign(ii, ww, slot_of, AB, 2, remote, glist, gcount, gw, stats)
            ext.q4k_moe(xb, glist, gcount, gw, PTRS, hbuf, y)
            acc += y.double().sum(0)
        table[li, e] = (acc / len(rows_e)).float().cpu().numpy(); cnt[li, e] = len(rows_e)
    del G, U, D; torch.cuda.empty_cache()
    print(json.dumps({"layer": L, "tokens": int(X.shape[0]), "experts_seen": int((cnt[li] > 0).sum()), "min_n_seen": int(cnt[li][cnt[li] > 0].min())}), flush=True)
np.save(sys.argv[2], table.astype(np.float16)); np.save(sys.argv[2].replace(".npy", "_counts.npy"), cnt)
print("saved", sys.argv[2])
