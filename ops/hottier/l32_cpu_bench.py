"""Hot-expert tier experiment, step 3 (Spark client -> workstation Q4_K expert server). KERNEL+LINK MICROBENCHMARK.

For each hot-set size N, the cold routes of the real test tokens (layer 32) are sent to expert_server_gguf (clamped
SiLU build) over the existing HFX1 socket protocol. Measures, per request size (T = 1 decode, 4 = MTP verify window,
16 = server max): round trip, server compute time, so link+sync = RTT - server; and Q4_K correctness of the cold
partial sum vs BF16 and vs the control's EXL3 experts.
Usage: python3 l32_cpu_bench.py DATA_DIR OUT_JSON   (env HF_SERVER_IP, HF_SERVER_PORT)
"""
import json, os, socket, struct, sys, time
import numpy as np, torch
from safetensors.torch import load_file

D, OUTJ = sys.argv[1], sys.argv[2]
HOST, PORT, MOE = os.environ["HF_SERVER_IP"], int(os.environ.get("HF_SERVER_PORT", "47201")), 29
E, H, TOPK = 288, 4096, 8
NS = [int(v) for v in os.environ.get("HOT_NS", "288,256,216,160,128,96,64").split(",")]
tk = load_file(f"{D}/l32_tokens.safetensors")
rc = json.load(open(f"{D}/l32_route_counts.json"))
x, ids, w = tk["x"].bfloat16(), tk["ids"].long(), tk["w"].float()
route_b, route_x = tk["route_bf16"].float(), tk["route_exl3"].float()
T0 = x.shape[0]
hot_order = torch.argsort(sum(torch.tensor(v) for v in rc["counts"].values()), descending=True)
sock = socket.create_connection((HOST, PORT)); sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)


def rd(n):
    b = bytearray()
    while len(b) < n:
        c = sock.recv(n - len(b))
        if not c: raise ConnectionError("server closed")
        b += c
    return bytes(b)


def call(tok_ids, cold_mask):
    """Send the cold routes of tokens tok_ids; returns (partial [len(tok_ids),H] or None, rtt_us, server_us)."""
    pairs = [(i, int(ids[t, k]), float(w[t, k])) for i, t in enumerate(tok_ids) for k in range(TOPK) if cold_mask[t, k]]
    if not pairs:
        return None, 0.0, 0
    toks = sorted({p[0] for p in pairs}); tm = {t: j for j, t in enumerate(toks)}
    msg = struct.pack("<IHHHH", 0x31584648, MOE, len(toks), len(pairs), 0) + \
        b"".join(struct.pack("<HHf", tm[i], e, wv) for i, e, wv in pairs) + \
        x[[tok_ids[i] for i in toks]].contiguous().view(torch.int16).numpy().tobytes()
    t0 = time.perf_counter_ns(); sock.sendall(msg)
    _, nt, st, us = struct.unpack("<IHHI", rd(12))
    yr = torch.from_numpy(np.frombuffer(rd(nt * H * 2), np.int16).copy()).view(torch.bfloat16).view(nt, H).float()
    rtt = (time.perf_counter_ns() - t0) / 1e3
    y = torch.zeros(len(tok_ids), H); y[toks] = yr
    return y, rtt, us


def stats(v):
    v = np.asarray(v, float)
    return {"n": int(v.size), "mean": round(float(v.mean()), 1), "p50": round(float(np.percentile(v, 50)), 1),
            "p90": round(float(np.percentile(v, 90)), 1), "p99": round(float(np.percentile(v, 99)), 1)} if v.size else None


for _ in range(20): call([0], torch.ones_like(ids, dtype=torch.bool))     # warm
res = {"label": "kernel+link microbenchmark (Q4_K experts on workstation CPU, Spark client, one MoE layer)", "N": {}}
for N in NS:
    hot = torch.zeros(E, dtype=torch.bool); hot[hot_order[:N]] = True
    cold = ~hot[ids]
    r = {"cold_routes_per_token": round(float(cold.sum(1).float().mean()), 3),
         "tokens_blocking_T1": round(float(cold.any(1).float().mean()), 4)}
    for T in (1, 4, 16):
        rtts, svs, link, y_all, blocked = [], [], [], torch.zeros(T0, H), 0
        n_req = 0
        for s in range(0, T0, T):
            tok = list(range(s, min(s + T, T0)))
            y, rtt, us = call(tok, cold)
            n_req += 1
            if y is None: continue
            blocked += 1; rtts.append(rtt); svs.append(us); link.append(rtt - us); y_all[tok] = y
            if T == 16 and len(rtts) >= 160: break
        r[f"T{T}"] = {"requests": n_req, "blocking_fraction": round(blocked / n_req, 4), "rtt_us": stats(rtts),
                      "server_us": stats(svs), "link_sync_us": stats(link)}
        if T == 4:                                                  # correctness on all tokens
            m = cold.unsqueeze(-1).float()
            cb, cx = (route_b * m).sum(1), (route_x * m).sum(1)
            nz = cold.any(1)
            if nz.any():
                r["q4k_cold_partial_vs_bf16_rel"] = round(float((y_all[nz] - cb[nz]).norm() / cb[nz].norm()), 5)
                r["exl3_cold_partial_vs_bf16_rel"] = round(float((cx[nz] - cb[nz]).norm() / cb[nz].norm()), 5)
    res["N"][N] = r
    print(json.dumps({N: r}), flush=True)
json.dump(res, open(OUTJ, "w"), indent=1)
print("CPU_BENCH_DONE", flush=True)
