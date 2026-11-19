# Diagnostic for mean stand-in on Qwen3: (a) eager expert formula == experts() impl;
# (b) for tail picks (rank>=K), does w*mean reduce the error vs dropping (zero)? out-of-sample (calib means) and in-sample (eval means).
import json, torch, torch.nn.functional as F, sys
sys.argv = ["x"]
import os
os.environ["KS"] = "5"
src = open("/srv/cache/heteroflash/union-qwen/union_qwen.py").read()
src = src[:src.index("# ---- calibration means")]
exec(src)
MEAN = torch.load(f"{W}/qwen_mean_outputs.pt")["mean"].float().to(dev)
agg = {}
def diag_forward(self, hidden_states):
    B, T, Hd = hidden_states.shape
    x = hidden_states.view(-1, Hd)
    _, w, ids = self.gate(x)
    L = self._lid; ex = self.experts
    y = self.experts(x, ids, w)
    # eager full output
    fo = torch.zeros(x.shape[0], TOPK, Hd, device=x.device, dtype=torch.float32)
    for e in ids.unique().tolist():
        tk, kp = torch.where(ids == e)
        g, u = F.linear(x[tk], ex.gate_up_proj[e]).chunk(2, dim=-1)
        fo[tk, kp] = F.linear(ex.act_fn(g) * u, ex.down_proj[e]).float()
    yy = (fo * w.float().unsqueeze(-1)).sum(1)
    a = agg.setdefault(L, torch.zeros(8, dtype=torch.float64, device=x.device))
    a[0] += ((yy - y.float()).norm() / y.float().norm())
    a[7] += 1
    for K in (5,):
        tail = fo[:, K:]; wt = w[:, K:].float().unsqueeze(-1); mu = MEAN[L][ids[:, K:]]
        err0 = (wt * tail).pow(2).sum()                 # drop -> error = w f
        err1 = (wt * (tail - mu)).pow(2).sum()          # stand-in -> error = w (f - mu)
        # sum-over-tail vector error (what the residual stream actually sees)
        e0v = (wt * tail).sum(1).pow(2).sum(); e1v = (wt * (tail - mu)).sum(1).pow(2).sum()
        a[1] += err0; a[2] += err1; a[3] += e0v; a[4] += e1v
        a[5] += (wt * tail).sum(1).norm(dim=-1).sum(); a[6] += y.float().norm(dim=-1).sum()
    return y
M.Qwen3MoeSparseMoeBlock.forward = diag_forward
with torch.no_grad():
    for ci in (0, 4, 6, 8):
        model(input_ids=torch.tensor([chunks[ci][1]], device=dev), use_cache=False)
print("layer relerr(eager vs impl) | per-pick err ratio stand/drop | tail-sum err ratio | |tail sum|/|y|")
for L in sorted(agg):
    a = agg[L].tolist()
    print(L, f"{a[0]/a[7]:.2e}", f"{a[2]/a[1]:.3f}", f"{a[4]/a[3]:.3f}", f"{a[5]/a[6]:.3f}")
tot = torch.stack([agg[L] for L in agg]).sum(0).tolist()
print("ALL", f"{tot[2]/tot[1]:.3f}", f"{tot[4]/tot[3]:.3f}", f"{tot[5]/tot[6]:.3f}")
