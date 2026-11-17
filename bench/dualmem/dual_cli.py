#!/usr/bin/env python3
"""Bet 1 one-layer-split prototype, Spark side. Per phase: run this side's FP8 slice on the GB10 (torch._scaled_mm,
W8A8 per-tensor scales, activations padded to 16 rows), synchronize, copy the T x 4096 partial to pinned host memory,
swap it with the workstation over TCP (busy-poll spin recv), copy the peer partial back and add it.
  dual_cli.py --peer IP --port P --T 4 --copies 10 --A 4096x4096,... --B ... --layers 45 --cycles 20 --mode full|gpu_nosync|gpu_sync
A cycle = `layers` x (phase A + phase B) = 90 phases for 45 layers."""
import argparse, json, socket, statistics, time, torch
ap = argparse.ArgumentParser()
ap.add_argument("--peer"); ap.add_argument("--port", type=int); ap.add_argument("--T", type=int, default=4)
ap.add_argument("--copies", type=int, default=10); ap.add_argument("--A"); ap.add_argument("--B")
ap.add_argument("--layers", type=int, default=45); ap.add_argument("--cycles", type=int, default=20)
ap.add_argument("--mode", default="full"); ap.add_argument("--tag", default="")
a = ap.parse_args()
dev = "cuda"; f8 = torch.float8_e4m3fn
def shapes(s): return [tuple(map(int, x.split("x"))) for x in s.split(",")]
SA, SB = shapes(a.A), shapes(a.B)
g = torch.Generator(device=dev).manual_seed(0)
def mk(N, K): return (torch.randn(N, K, device=dev, generator=g) * 0.05).to(f8)
W = [[[mk(N, K) for N, K in SA], [mk(N, K) for N, K in SB]] for _ in range(a.copies)]
bytesA = sum(N * K for N, K in SA); bytesB = sum(N * K for N, K in SB)
xs = {K: (torch.randn(16, K, device=dev, generator=g) * 0.1).to(f8) for K in {k for _, k in SA + SB}}
one = torch.tensor(1.0, device=dev)
T = a.T; m = T * 4096 * 2
host_out = torch.empty(T * 4096, dtype=torch.bfloat16, pin_memory=True)
host_in = torch.empty(T * 4096, dtype=torch.bfloat16, pin_memory=True)
acc = torch.zeros(T, 4096, dtype=torch.bfloat16, device=dev)
sbuf = bytearray(m + 1); sbuf[0] = 0xAA; rview = memoryview(host_in.view(torch.int16).numpy().view("uint8"))
def run_phase(l, p):
    y = None
    for w in W[l][p]:
        y = torch._scaled_mm(xs[w.shape[1]], w.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16)
    return y
sock = None
if a.mode == "full":
    sock = socket.create_connection((a.peer, a.port)); sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.setsockopt(socket.SOL_SOCKET, 46, 50)  # SO_BUSY_POLL 50 us
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 << 20); sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 << 20)
    sock.sendall(bytes(sbuf)); sock.setblocking(False)
def recv_frame():
    h = bytearray(1)
    while True:
        try:
            if sock.recv_into(h, 1) == 1 and h[0] == 0xAA: break
        except BlockingIOError: pass
    got = 0
    while got < m:
        try: got += sock.recv_into(rview[got:], m - got)
        except BlockingIOError: pass
def send_all(buf):
    mv = memoryview(buf); s = 0
    while s < len(buf):
        try: s += sock.send(mv[s:])
        except BlockingIOError: pass
for _ in range(3): run_phase(0, 0); run_phase(0, 1)
torch.cuda.synchronize()
cyc, t_gpu, t_sync, t_net = [], [], [], []
for c in range(a.cycles + 2):
    t0 = time.perf_counter()
    for L in range(a.layers):
        l = L % a.copies
        for p in (0, 1):
            g0 = time.perf_counter()
            y = run_phase(l, p)
            if a.mode == "gpu_nosync": continue
            torch.cuda.synchronize(); g1 = time.perf_counter()
            host_out.copy_(y[:T, :4096].reshape(-1)); g2 = time.perf_counter()
            if a.mode == "full":
                sbuf[1:] = host_out.view(torch.int16).numpy().view("uint8").tobytes(); send_all(sbuf); recv_frame()
                acc.copy_(host_in.view(T, 4096), non_blocking=True)
            g3 = time.perf_counter()
            if c >= 2: t_gpu.append(g1 - g0); t_sync.append(g2 - g1); t_net.append(g3 - g2)
    torch.cuda.synchronize()
    if c >= 2: cyc.append(time.perf_counter() - t0)
med = lambda x: statistics.median(x) * 1e6 if x else 0
res = {"tag": a.tag, "mode": a.mode, "T": T, "phases_per_cycle": 2 * a.layers, "phaseA_MB": bytesA / 1e6, "phaseB_MB": bytesB / 1e6,
       "cycle_ms_p50": statistics.median(cyc) * 1e3, "cycle_ms_min": min(cyc) * 1e3,
       "gpu_GBps_cycle": a.layers * (bytesA + bytesB) / statistics.median(cyc) / 1e9,
       "per_phase_us_p50": {"gpu_launch_to_sync": med(t_gpu), "d2h_copy": med(t_sync), "swap_send_recv_h2d": med(t_net)}}
print(json.dumps(res), flush=True)
