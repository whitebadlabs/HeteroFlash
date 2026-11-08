"""Mailbox unit test on the Spark GPU against a live expert_server (one layer loaded).
  ref  : python socket client (same wire format as the eager hook)  -> saves reference partial sums
  mb   : mailbox ext, eager pack/wait_add, then the same inside a CUDA graph (capture once, replay twice)."""
import json, os, socket, struct, sys, time
import numpy as np, torch
HOST, PORT, LAYER = os.environ["HF_SERVER_IP"], int(os.environ.get("HF_SERVER_PORT", "47210")), int(os.environ.get("HF_LAYER", "20"))
torch.manual_seed(0)
T = 3
x = (torch.randn(T, 4096) * 0.5).to(torch.bfloat16)
ids = torch.stack([torch.randperm(288)[:8] for _ in range(T)]).to(torch.int64)
w = torch.rand(T, 8)
remote = torch.zeros(288, dtype=torch.bool); remote[:144] = True
mode = sys.argv[1]
if mode == "ref":
    pairs = [(t, int(ids[t, k]), float(w[t, k])) for t in range(T) for k in range(8) if remote[ids[t, k]]]
    toks = sorted({p[0] for p in pairs}); tm = {t: i for i, t in enumerate(toks)}
    msg = struct.pack("<IHHHH", 0x31584648, LAYER - 3, len(toks), len(pairs), 0) + b"".join(struct.pack("<HHf", tm[t], e, wv) for t, e, wv in pairs) + x[toks].contiguous().view(torch.int16).numpy().tobytes()
    s = socket.create_connection((HOST, PORT)); s.sendall(msg)
    def rd(n):
        b = b""
        while len(b) < n: b += s.recv(n - len(b))
        return b
    _, nt, st, us = struct.unpack("<IHHI", rd(12))
    yr = torch.from_numpy(np.frombuffer(rd(nt * 8192), np.int16).copy()).view(torch.bfloat16).view(nt, 4096)
    y = torch.zeros(T, 4096, dtype=torch.bfloat16); y[toks] = yr
    torch.save({"y": y, "pairs": len(pairs)}, "/tmp/hf_mb_ref.pt"); print("ref saved", len(pairs), "pairs, server_us", us)
else:
    from torch.utils.cpp_extension import load
    t0 = time.time()
    ext = load(name="hf_mailbox", sources=[os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_mailbox.cu")], extra_cuda_cflags=["-O3"])
    print("compiled in", round(time.time() - t0, 1), "s")
    ext.init(64, HOST, PORT, 3)
    ref = torch.load("/tmp/hf_mb_ref.pt")["y"].float()
    xd, idd, wd, rd_ = x.cuda(), ids.cuda(), w.cuda(), remote.cuda()
    def run():
        out = torch.zeros(T, 4096, dtype=torch.bfloat16, device="cuda")
        wg = ext.pack(5, LAYER - 3, xd, idd, wd, rd_)
        ext.wait_add(5, out)
        return out, wg
    out, wg = run(); torch.cuda.synchronize()
    err = ((out.float().cpu() - ref).norm() / ref.norm()).item()
    wexp = torch.where(remote[ids], torch.zeros_like(w), w)
    print("eager: rel err vs python ref", round(err, 6), "| gpu weights masked ok:", bool(torch.allclose(wg.cpu(), wexp)))
    t1 = time.perf_counter()
    for _ in range(20): out, _ = run()
    torch.cuda.synchronize(); print("eager round trip mean us", round((time.perf_counter() - t1) / 20 * 1e6))
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        run(); torch.cuda.synchronize()
        with torch.cuda.graph(g):
            outg, _ = run()
    for i in range(3):
        g.replay(); torch.cuda.synchronize()
        print("graph replay", i, "rel err", round(((outg.float().cpu() - ref).norm() / ref.norm()).item(), 6))
    t1 = time.perf_counter()
    for _ in range(20): g.replay()
    torch.cuda.synchronize(); print("graph round trip mean us", round((time.perf_counter() - t1) / 20 * 1e6), "stats", ext.stats())
    sys.stdout.flush(); os._exit(0)
