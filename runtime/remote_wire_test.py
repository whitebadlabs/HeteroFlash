import json, os, socket, struct, sys
import numpy as np, torch
from safetensors import safe_open
sys.argv += []
B = "/m/GLM-5.3-Flash-BF16"; idx = json.load(open(B + "/model.safetensors.index.json"))["weight_map"]
L = 20; H = 4096
def W(n):
    with safe_open(B + "/" + idx[n], "pt") as f: return f.get_tensor(n).float()
torch.manual_seed(0)
ntok = 5; x = (torch.randn(ntok, H) * 0.5).to(torch.bfloat16)
pairs = {"sparse": [(0, 3, 0.7), (0, 100, 0.4), (2, 3, 1.1), (4, 287, 0.25), (4, 50, 0.9), (1, 7, 0.6)],
         "dense": [(t, e, 0.5) for t in range(5) for e in (3, 100)],
         "sparse_sorted_tok": [(0, 3, 0.7), (0, 100, 0.4), (1, 7, 0.6), (2, 3, 1.1), (4, 287, 0.25), (4, 50, 0.9)],
         "one": [(0, 3, 1.0)]}[os.environ.get("PS", "sparse")]
ref = torch.zeros(ntok, H)
for t, e, w in pairs:
    p = f"model.language_model.layers.{L}.mlp.experts.{e}."
    g, u, d = W(p + "gate_proj.weight"), W(p + "up_proj.weight"), W(p + "down_proj.weight")
    xf = x[t].float()
    ref[t] += w * (d @ (torch.nn.functional.silu(g @ xf) * (u @ xf)))
wire = np.zeros(len(pairs), dtype=[("tok", "<u2"), ("e", "<u2"), ("w", "<f4")])
wire["tok"], wire["e"], wire["w"] = [p[0] for p in pairs], [p[1] for p in pairs], [p[2] for p in pairs]
s = socket.create_connection((os.environ["HF_SERVER_IP"], int(os.environ.get("HF_SERVER_PORT", "47210"))))
s.sendall(struct.pack("<IHHHH", 0x31584648, L - 3, ntok, len(pairs), int(os.environ.get("FL","0"))) + wire.tobytes() + x.contiguous().view(torch.int16).numpy().tobytes())
def rn(n):
    b = b""
    while len(b) < n: b += s.recv(n - len(b))
    return b
while True:
    c = rn(1)
    if c != b"\x00": break
assert c + rn(3) == b"HFY1"
nt, st, us = struct.unpack("<HHI", rn(8))
y = torch.from_numpy(np.frombuffer(rn(nt * H * 2), dtype=np.int16).copy()).view(torch.bfloat16).float().view(nt, H)
for t in range(ntok):
    r = ref[t].norm().item()
    print("tok", t, "ref_norm", round(r, 3), "rel_err", round(((y[t] - ref[t]).norm() / max(r, 1e-9)).item(), 4) if r > 0 else ("zero ok" if y[t].abs().max() == 0 else "NONZERO"))
print("status", st, "server_us", us, "ntok", nt, "y_abs_max_tok3", float(y[3].abs().max()))
