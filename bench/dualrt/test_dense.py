"""Unit test + bandwidth of the MXFP8 row GEMV (GPU, hook/hf_drt.cu) and of the workstation HFD1 dense rows (lm_head
lid 0: rows 84992-154879; KDA q/k/v lids 1-34), against an fp32 dequantized reference.   python3 test_dense.py [ip:port]"""
import json, os, socket, struct, sys, time
import numpy as np, torch
from safetensors import safe_open
from torch.utils.cpp_extension import load
HERE = os.path.dirname(os.path.abspath(__file__))
ext = load(name="hf_drt", sources=[os.path.join(HERE, "hook", "hf_drt.cu")], extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=False)
CK = "/m"; wm = json.load(open(f"{CK}/model.safetensors.index.json"))["weight_map"]
def get(n):
    with safe_open(f"{CK}/{wm[n]}", framework="pt") as f: return f.get_tensor(n)
W8 = get("lm_head.weight"); S8 = get("lm_head.weight_scale").view(torch.uint8)
def deq(w, s):  # fp32 [N, K]
    return w.to(torch.float32) * torch.pow(2.0, s.to(torch.float32) - 127).repeat_interleave(32, dim=1)
res = {}
torch.manual_seed(0)
x = (torch.randn(4, 4096) * 0.5).to(torch.bfloat16)
Vg = 84992
ref = deq(W8[:Vg].cuda(), S8[:Vg].cuda()) @ x.float().cuda().T          # [Vg, 4]
W = W8.view(torch.uint8)[:Vg].contiguous().cuda(); S = S8[:Vg].contiguous().cuda()
out = torch.zeros(4, Vg, dtype=torch.bfloat16, device="cuda")
ext.mxfp8_gemv(W, S, x.cuda(), out); torch.cuda.synchronize()
res["gpu_rel_l2"] = float((out.float().T - ref).norm() / ref.norm()); print(res, flush=True)
for T in (1, 4):
    xx = x[:T].cuda().contiguous(); o = torch.zeros(T, Vg, dtype=torch.bfloat16, device="cuda")
    for _ in range(3): ext.mxfp8_gemv(W, S, xx, o)
    torch.cuda.synchronize(); n = 50; t0 = time.perf_counter()
    for _ in range(n): ext.mxfp8_gemv(W, S, xx, o)
    torch.cuda.synchronize(); dt = (time.perf_counter() - t0) / n
    res[f"gpu_T{T}_us"] = round(dt * 1e6, 1); res[f"gpu_T{T}_GBs"] = round(Vg * 4096 * (1 + 1 / 32) / dt / 1e9, 1); print(res, flush=True)
if len(sys.argv) > 1:
    host, port = sys.argv[1].rsplit(":", 1); s = socket.create_connection((host, int(port))); s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    def rn(n):
        b = bytearray()
        while len(b) < n: b += s.recv(n - len(b))
        return bytes(b)
    refc = deq(W8[Vg:].cuda(), S8[Vg:].cuda()) @ x.float().cuda().T
    for rep in range(6):
        t0 = time.perf_counter()
        s.sendall(struct.pack("<IHHHH", 0x31444648, 0, 4, 0, 0) + x.view(torch.int16).numpy().tobytes())
        h = rn(12); assert h[:4] == b"HFE1", h; T, st, us = struct.unpack("<HHI", h[4:])
        y = np.frombuffer(rn(4 * (154880 - Vg) * 2), dtype=np.uint16).astype(np.uint32) << 16
        dt = time.perf_counter() - t0
    yc = torch.from_numpy(y.view(np.float32).reshape(4, -1)).cuda()
    res["server_rel_l2"] = float((yc.T - refc).norm() / refc.norm()); res["server_us"] = us; res["roundtrip_us"] = round(dt * 1e6)
    print(res, flush=True)
json.dump(res, open(os.environ.get("OUT", "/tmp/test_dense.json"), "w"), indent=1)
