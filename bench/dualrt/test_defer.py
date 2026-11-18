#!/usr/bin/env python3
"""Unit test (2026-11-18) of the deferral pieces of hf_drt.cu against a fake workstation server on 127.0.0.1:
gated pack (late slot), gated wait (layer L vs L+1), the GPU wait timers, eager and inside a CUDA graph."""
import json, os, socket, struct, threading, time
import torch
from torch.utils.cpp_extension import load

H = 4096
SLEEP_S = 0.001


def readn(c, n):
    b = b""
    while len(b) < n:
        k = c.recv(n - len(b))
        if not k:
            raise EOFError
        b += k
    return b


def serve(srv):
    c, _ = srv.accept()
    c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    while True:
        try:
            m = readn(c, 1)
            while m == b"\x00":
                m = readn(c, 1)
            m += readn(c, 3)
            hdr = readn(c, 8)
        except EOFError:
            return
        lay, nt, npairs, flags = struct.unpack("<4H", hdr)
        pairs = [struct.unpack("<HHf", readn(c, 8)) for _ in range(npairs)]
        readn(c, nt * H * 2)
        y = torch.zeros(nt, H)
        for t, e, w in pairs:               # fake partial: weight * (expert + 1) in every column
            y[t] += w * (e + 1)
        time.sleep(SLEEP_S)
        body = y.to(torch.float32).numpy().tobytes() if flags & 4 else y.to(torch.bfloat16).view(torch.int16).numpy().tobytes()
        c.sendall(b"HFY1" + struct.pack("<HHI", nt, 0, 900) + body)


def expected(ids, w, remote):
    y = torch.zeros(ids.shape[0], H)
    for t in range(ids.shape[0]):
        for k in range(ids.shape[1]):
            e = int(ids[t, k])
            if 0 <= e < 288 and bool(remote[e]):
                y[t] += float(w[t, k]) * (e + 1)
    return y


def main():
    srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); srv.bind(("127.0.0.1", 0)); srv.listen(1)
    port = srv.getsockname()[1]
    threading.Thread(target=serve, args=(srv,), daemon=True).start()
    ext = load(name="hf_drt", sources=[os.path.join(os.path.dirname(os.path.abspath(__file__)), "hook", "hf_drt.cu")],
               extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=False)
    nsl = 42 * 8
    ext.mb_init(nsl, "127.0.0.1", port, 3 | 4)      # fp32 reply so the check is exact up to the bf16 output rounding
    dev = "cuda"; torch.manual_seed(0); res = {}
    none = torch.zeros(0, dtype=torch.int32, device=dev)
    T = 4
    x = torch.randn(T, H, device=dev).to(torch.bfloat16)
    ids = torch.stack([torch.randperm(288, device=dev)[:8] for _ in range(T)]).to(torch.int64)
    w = (torch.rand(T, 8, device=dev) / 4).contiguous()
    remote = torch.rand(288, device=dev) < 0.5
    ref = expected(ids.cpu(), w.cpu(), remote.cpu())

    def close(o, r):
        return float((o.float().cpu() - r).abs().max()) <= 0.02 * float(r.abs().max() + 1)

    # 1. ungated (existing behaviour)
    out = torch.zeros(T, H, dtype=torch.bfloat16, device=dev)
    ext.mb_pack(0, 0, x, ids, w, remote, none, 0); ext.mb_wait_add(0, out, none, 0); torch.cuda.synchronize()
    res["ungated"] = close(out, ref)
    # 2. deferral gate: defer[1] = 1 -> the "own" wait (want 0) is a no-op, the next layer's wait (want 1) adds
    defer = torch.tensor([0, 1], dtype=torch.int32, device=dev)
    o1 = torch.zeros_like(out); o2 = torch.zeros_like(out)
    ext.mb_pack(8, 1, x, ids, w, remote, none, 0)
    ext.mb_wait_add(8, o1, defer[1:2], 0); ext.mb_wait_add(8, o2, defer[1:2], 1); torch.cuda.synchronize()
    res["defer_own_noop"] = bool((o1.float() == 0).all()); res["defer_late_adds"] = close(o2, ref)
    # 3. gated pack off: no request reaches the server
    late = torch.zeros(1, dtype=torch.int32, device=dev)
    r0 = ext.mb_stats()[0]
    ext.mb_pack(9, 1, x, ids, w, remote, late, 1); o3 = torch.zeros_like(out); ext.mb_wait_add(9, o3, late, 1); torch.cuda.synchronize()
    time.sleep(0.05); res["late_off_no_request"] = ext.mb_stats()[0] == r0 and bool((o3.float() == 0).all())
    # 4. CUDA graph: pack + both gated waits captured once, gate flipped between replays
    s = torch.cuda.Stream(); og_now = torch.zeros_like(out); og_late = torch.zeros_like(out)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s):
        torch.cuda.synchronize()
        with torch.cuda.graph(g, stream=s):
            og_now.zero_(); og_late.zero_()
            ext.mb_pack(16, 2, x, ids, w, remote, none, 0)
            ext.mb_wait_add(16, og_now, defer[0:1], 0)    # defer[0] is 0 at first
            ext.mb_wait_add(16, og_late, defer[0:1], 1)
    for d in (0, 1, 0):
        defer[0] = d; g.replay(); torch.cuda.synchronize()
        res[f"graph_defer{d}_{'now' if d == 0 else 'late'}"] = close(og_now if d == 0 else og_late, ref) and bool(((og_late if d == 0 else og_now).float() == 0).all())
    # 5. timers: slot 16 decode bucket, n = 3 (replays), blocked ~ server sleep
    v = ext.mb_wait_stats(); i = (0 * nsl + 16) * 4
    n, blk, ovl, zero = v[i:i + 4]
    res["timer_slot16"] = {"n": n, "blocked_ms": round(blk / max(n, 1) / 1e6, 3), "overlap_us": round(ovl / max(n, 1) / 1e3, 1), "zero": zero}
    res["timer_ok"] = n == 3 and 0.8 < blk / n / 1e6 < 20
    res["all_ok"] = all(v for k, v in res.items() if isinstance(v, bool))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
