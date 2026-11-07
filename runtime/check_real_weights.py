#!/usr/bin/env python3
"""End-to-end check of the expert server on real exported weights.
Sends one request (layer, tokens, experts, gate weights, bf16 x) and compares the server's
partial sum with numpy: (a) same int4 weights dequantized (serving correctness) and
(b) original BF16 weights (total quantization error)."""
import socket, struct, sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import quantize_experts_int4 as qe

host, port, model, layer = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
experts = [int(e) for e in sys.argv[5].split(",")]
ck = qe.Checkpoint(model)
rng = np.random.default_rng(1)
ntok = 3
x = rng.standard_normal((ntok, 4096)).astype(np.float32)
xb = (x.view(np.uint32) + 0x7FFF + ((x.view(np.uint32) >> 16) & 1)) >> 16
x_bf = (xb.astype(np.uint32) << 16).view(np.float32)
pairs = [(t, e, float(w)) for t in range(ntok) for e, w in zip(experts, rng.dirichlet(np.ones(len(experts))) * 2.5)]
msg = struct.pack("<IHHHH", 0x31584648, layer - 3, ntok, len(pairs), 0) + b"".join(struct.pack("<HHf", t, e, w) for t, e, w in pairs) + xb.astype(np.uint16).tobytes()
s = socket.create_connection((host, port)); s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1); s.sendall(msg)
def rd(n):
    b = b""
    while len(b) < n: b += s.recv(n - len(b))
    return b
magic, nt, st, us = struct.unpack("<IHHI", rd(12))
y = (np.frombuffer(rd(ntok * 4096 * 2), np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(ntok, 4096)
ref_q = np.zeros_like(y); ref_bf = np.zeros_like(y)
for t, e, w in pairs:
    g, u, d = ck.expert(layer, e)
    gu = np.concatenate([g, u], 0)
    gq = qe.dequant(*qe.quant_int4_g128(gu)); dq = qe.dequant(*qe.quant_int4_g128(d))
    ref_q[t] += w * qe.ffn(x_bf[t:t + 1], gq[:2048], gq[2048:], dq)[0]
    ref_bf[t] += w * qe.ffn(x_bf[t:t + 1], g, u, d)[0]
rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
print({"status": st, "server_us": us, "vs_same_int4_dequant": rel(y, ref_q), "vs_bf16_original": rel(y, ref_bf),
       "int4_dequant_vs_bf16": rel(ref_q, ref_bf)})
