"""Unit test + bandwidth of the GPU Q4_K routed-expert kernels (bench/dualrt/hook/hf_drt.cu), Spark GPU only.
Checks, on real Q4_K + imatrix experts of one layer (Spark copy of the GGUF):
  (1) GPU kernel vs an fp32 numpy dequantize-and-matmul reference (kernel exactness);
  (2) optional: GPU kernel vs the workstation expert_server_gguf on the same routes (q8_K activations there);
  (3) GB/s of the kernels at the decode shape (T = 1..4, 1..24 distinct GPU experts).
  python3 test_q4k.py LAYER [server ip:port]"""
import json, os, socket, struct, sys, time
import numpy as np
import torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
ext = load(name="hf_drt", sources=[os.path.join(HERE, "hook", "hf_drt.cu")], extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=False)
GGUF = os.environ.get("GGUF", "/gguf/GLM-5.3-Flash-Q8attn-G0_q4kimexp.gguf"); IDX = os.environ.get("IDX", os.path.join(HERE, "q4k_experts.idx"))
L = int(sys.argv[1]); H, I, NE = 4096, 2048, 288
recs = {}
raw = open(IDX, "rb").read()
for i in range(len(raw) // 48):
    l, p, ty, rows, cols, ne, rb, off, nb = struct.unpack("<6i3Q", raw[i * 48:(i + 1) * 48]); recs[(l, p)] = (ty, rows, cols, ne, rb, off)

def read_expert(p, e):
    ty, rows, cols, ne, rb, off = recs[(L, p)]
    with open(GGUF, "rb") as f:
        f.seek(off + e * rows * rb); return np.frombuffer(f.read(rows * rb), dtype=np.uint8).reshape(rows, rb)

def deq(q):  # q: [rows, nblk*144] -> fp32 [rows, nblk*256]
    rows = q.shape[0]; b = q.reshape(rows, -1, 144); nb = b.shape[1]
    d = b[:, :, 0:2].copy().view(np.float16)[..., 0].astype(np.float32); dmin = b[:, :, 2:4].copy().view(np.float16)[..., 0].astype(np.float32)
    s = b[:, :, 4:16].astype(np.int32); qs = b[:, :, 16:144]
    sc = np.zeros((rows, nb, 8), np.float32); mn = np.zeros((rows, nb, 8), np.float32)
    for j in range(8):
        if j < 4: sc[..., j] = s[..., j] & 63; mn[..., j] = s[..., j + 4] & 63
        else: sc[..., j] = (s[..., j + 4] & 0xF) | ((s[..., j - 4] >> 6) << 4); mn[..., j] = (s[..., j + 4] >> 4) | ((s[..., j] >> 6) << 4)
    out = np.zeros((rows, nb, 256), np.float32)
    for c in range(4):
        lo = (qs[:, :, c * 32:(c + 1) * 32] & 0xF).astype(np.float32); hi = (qs[:, :, c * 32:(c + 1) * 32] >> 4).astype(np.float32)
        out[:, :, c * 64:c * 64 + 32] = d[..., None] * sc[..., 2 * c, None] * lo - dmin[..., None] * mn[..., 2 * c, None]
        out[:, :, c * 64 + 32:c * 64 + 64] = d[..., None] * sc[..., 2 * c + 1, None] * hi - dmin[..., None] * mn[..., 2 * c + 1, None]
    return out.reshape(rows, nb * 256)

NS = 24
experts = list(range(0, 288, 12))[:NS]
G = torch.empty((NS, I * 2304), dtype=torch.uint8, device="cuda"); U = torch.empty_like(G); D = torch.empty((NS, H * 1152), dtype=torch.uint8, device="cuda")
for j, e in enumerate(experts):
    G[j].copy_(torch.from_numpy(read_expert(0, e).reshape(-1).copy())); U[j].copy_(torch.from_numpy(read_expert(1, e).reshape(-1).copy()))
    D[j].copy_(torch.from_numpy(read_expert(2, e).reshape(-1).copy()))
PTRS = torch.tensor([G.data_ptr(), U.data_ptr(), D.data_ptr()], dtype=torch.int64, device="cuda")
slot_of = torch.full((NE,), -1, dtype=torch.int32); slot_of[torch.tensor(experts)] = torch.arange(NS, dtype=torch.int32); slot_of = slot_of.cuda()
remote = torch.ones(NE, dtype=torch.bool, device="cuda"); glist = torch.zeros(64, dtype=torch.int32, device="cuda")
gcount = torch.zeros(1, dtype=torch.int32, device="cuda"); gw = torch.zeros(64 * 16, device="cuda"); hbuf = torch.zeros(64 * 4 * I, device="cuda")
stats = torch.zeros(4, dtype=torch.int64, device="cuda"); AB = torch.tensor([1.0, 0.0], device="cuda")
torch.manual_seed(0); rng = np.random.default_rng(0)
deqc = {}
def ref_moe(x, ids, w):  # fp32 reference over the same routes
    y = np.zeros((x.shape[0], H), np.float64)
    for t in range(x.shape[0]):
        for k in range(8):
            e = int(ids[t, k])
            if e not in experts: continue
            if e not in deqc: deqc[e] = (deq(read_expert(0, e)), deq(read_expert(1, e)), deq(read_expert(2, e)))
            g, u, dn = deqc[e]
            gv = np.minimum(g @ x[t], 10.0); uv = np.clip(u @ x[t], -10.0, 10.0)
            hh = gv / (1 + np.exp(-gv)) * uv
            y[t] += w[t, k] * (dn @ hh)
    return y
res = {"layer": L, "checks": []}
for T in (1, 2, 4, 7):
    x = (torch.randn(T, H) * 0.6).to(torch.bfloat16)
    ids = torch.tensor(np.stack([rng.choice(experts, 8, replace=False) for _ in range(T)]), dtype=torch.int32)
    w = torch.rand(T, 8) * 0.3
    y = torch.zeros(T, H, device="cuda")
    ext.assign(ids.cuda().contiguous(), w.cuda().contiguous(), slot_of, AB, 2, remote, glist, gcount, gw, stats)
    ext.q4k_moe(x.cuda().contiguous(), glist, gcount, gw, PTRS, hbuf, y)
    torch.cuda.synchronize()
    r = ref_moe(x.float().numpy().astype(np.float64), ids.numpy(), w.numpy().astype(np.float64))
    err = float(np.linalg.norm(y.cpu().numpy() - r) / np.linalg.norm(r))
    res["checks"].append({"T": T, "gcount": int(gcount.item()), "rel_l2_vs_fp32_ref": err, "remote_any": bool(remote[torch.tensor(experts)].any().item())})
    print(json.dumps(res["checks"][-1]), flush=True)
    if len(sys.argv) > 2 and T <= 4:   # same routes through the workstation server (fp32 reply)
        host, port = sys.argv[2].rsplit(":", 1); s = socket.create_connection((host, int(port))); s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        wire = np.zeros(T * 8, dtype=[("tok", "<u2"), ("e", "<u2"), ("w", "<f4")])
        wire["tok"] = np.repeat(np.arange(T), 8); wire["e"] = ids.numpy().reshape(-1); wire["w"] = w.numpy().reshape(-1)
        s.sendall(struct.pack("<IHHHH", 0x31584648, L - 3, T, T * 8, 4) + wire.tobytes() + x.view(torch.int16).numpy().tobytes())
        def rn(n):
            b = bytearray()
            while len(b) < n: b += s.recv(n - len(b))
            return bytes(b)
        hdr = rn(12); assert hdr[:4] == b"HFY1", hdr; ys = np.frombuffer(rn(T * H * 4), dtype=np.float32).reshape(T, H); s.close()
        e2 = float(np.linalg.norm(ys - r) / np.linalg.norm(r)); e3 = float(np.linalg.norm(ys - y.cpu().numpy()) / np.linalg.norm(r))
        res["checks"][-1].update({"server_rel_l2_vs_fp32_ref": e2, "server_vs_gpu_rel_l2": e3}); print(json.dumps(res["checks"][-1]), flush=True)
# bandwidth: T = 1..4, n_g GPU experts per call, 200 calls (weights for 24 experts = 340 MB > L2)
bw = []
for T in (1, 4):
    for ng in (4, 8, 12, 16, 24):
        x = (torch.randn(T, H) * 0.6).to(torch.bfloat16).cuda()
        ids = torch.tensor([[experts[(t * 8 + k) % ng] for k in range(8)] for t in range(T)], dtype=torch.int32).cuda()
        if ng < 8:
            ids = torch.tensor([[experts[k % ng] if k < ng else experts[ng + k] for k in range(8)] for t in range(T)], dtype=torch.int32).cuda()
        w = torch.rand(T, 8, device="cuda") * 0.3; y = torch.zeros(T, H, device="cuda")
        sl = torch.full((NE,), -1, dtype=torch.int32); sl[torch.tensor(experts[:ng])] = torch.arange(ng, dtype=torch.int32); sl = sl.cuda()
        for _ in range(5):
            ext.assign(ids, w, sl, AB, 2, remote, glist, gcount, gw, stats); ext.q4k_moe(x, glist, gcount, gw, PTRS, hbuf, y)
        torch.cuda.synchronize(); n = 200; t0 = time.perf_counter()
        for _ in range(n):
            ext.assign(ids, w, sl, AB, 2, remote, glist, gcount, gw, stats); ext.q4k_moe(x, glist, gcount, gw, PTRS, hbuf, y)
        torch.cuda.synchronize(); dt = (time.perf_counter() - t0) / n
        gc = int(gcount.item()); byts = gc * (2 * I * 2304 + H * 1152)
        bw.append({"T": T, "n_gpu_experts": gc, "us": round(dt * 1e6, 1), "GB_s": round(byts / dt / 1e9, 1)}); print(json.dumps(bw[-1]), flush=True)
res["bandwidth"] = bw
json.dump(res, open(os.environ.get("OUT", "/tmp/test_q4k.json"), "w"), indent=1)
