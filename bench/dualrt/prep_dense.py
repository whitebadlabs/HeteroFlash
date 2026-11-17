#!/usr/bin/env python3
"""Write the workstation's rows of row-split linears (dense pack for expert_server_gguf HF_DENSE_PACK), 2026-11-17.
  prep_dense.py CKPT_DIR SPEC.json OUT.pack
SPEC = [{"lid": 0, "name": "lm_head", "fmt": 0, "rows": [[start, end], ...]}, ...]  (rows taken in the listed order;
        end -1 = to the last row), or {"lid":..., "fmt": 0, "parts": [[tensor_prefix, start, end], ...]} (start < 0: from the end)
fmt 0 = MXFP8 (weight e4m3 [N, K] + weight_scale E8M0 [N, K/32])."""
import json, struct, sys
import torch
from safetensors import safe_open
ck, spec, outp = sys.argv[1], json.load(open(sys.argv[2])), sys.argv[3]
wm = json.load(open(f"{ck}/model.safetensors.index.json"))["weight_map"]
def get(name):
    with safe_open(f"{ck}/{wm[name]}", framework="pt") as f:
        return f.get_tensor(name)
blobs, recs = [], []
off = 8 + 32 * len(spec)
off = (off + 63) // 64 * 64
pad0 = off - (8 + 32 * len(spec))
for s in spec:
    Ws, Ss = [], []
    for nm, a, b in (s["parts"] if "parts" in s else [(s["name"], a, b) for a, b in s["rows"]]):
        w = get(nm + ".weight").view(torch.uint8); sc = get(nm + ".weight_scale").view(torch.uint8)
        a = w.shape[0] + a if a < 0 else a; b = w.shape[0] if b is None or b == 0 or b < 0 and a > 0 and b == -1 else b
        Ws.append(w[a:b]); Ss.append(sc[a:b])
    W = torch.cat(Ws).contiguous(); S = torch.cat(Ss).contiguous()
    N, K = W.shape; assert S.shape == (N, K // 32), (S.shape, N, K)
    wb, sb = W.numpy().tobytes(), S.numpy().tobytes()
    wo = off; off += len(wb); off = (off + 63) // 64 * 64; so = off; off += len(sb); off = (off + 63) // 64 * 64
    recs.append(struct.pack("<4I2Q", s["lid"], s["fmt"], N, K, wo, so)); blobs.append((wo, wb)); blobs.append((so, sb))
    print(s.get("name", s.get("parts", [["?"]])[0][0]), N, K, flush=True)
with open(outp, "wb") as f:
    f.write(b"HFDP" + struct.pack("<I", len(spec)) + b"".join(recs) + b"\0" * pad0)
    for o, b in blobs:
        f.seek(o); f.write(b)
    f.truncate(off)
print("wrote", outp, off / 1e9, "GB")
