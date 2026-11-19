# Follow-up: mean stand-in variants on Qwen3 for K=5,4.
#  s_oracle: means computed in-sample from the eval chunks themselves (upper bound, not deployable)
#  s_sel: calib means, applied only on layers where calib-held-out check says they help (chosen on calib, not eval)
import os, json, torch
os.environ["KS"] = "5"
W = "/srv/cache/heteroflash/union-qwen"
src = open(f"{W}/union_qwen.py").read()
pre = src[:src.index("# ---- calibration means")]
exec(pre)
# calib means from tokens 0..8191, held-out check on calib tokens 8192..10239 to select layers
def run_calib(token_lists):
    mean_sum.zero_(); mean_cnt.zero_(); S["mode"] = "calib"
    for t in token_lists: logits_of(t)
    return (mean_sum / mean_cnt.clamp(min=1).unsqueeze(-1)).to(torch.bfloat16).float()
MEAN_CAL = torch.load(f"{W}/qwen_mean_outputs.pt")["mean"].float().to(dev)
MEAN_ORA = run_calib([c[1] for c in chunks])
# layer selection: err ratio on held-out calib chunk using means from first 4 calib chunks
MEAN_A = run_calib([calib_ids[i:i+CTX] for i in range(0, 4*CTX, CTX)])
ratio = torch.zeros(NL, 2, dtype=torch.float64, device=dev)
def sel_forward(self, hs):
    B, T, Hd = hs.shape; x = hs.view(-1, Hd); _, w, ids = self.gate(x); L = self._lid; ex = self.experts
    for e in ids[:, 5:].unique().tolist():
        tk, kp = torch.where(ids[:, 5:] == e)
        g, u = F.linear(x[tk], ex.gate_up_proj[e]).chunk(2, dim=-1)
        f = F.linear(ex.act_fn(g) * u, ex.down_proj[e]).float()
        wt = w[tk, 5 + kp].float().unsqueeze(-1)
        ratio[L, 0] += (wt * f).pow(2).sum(); ratio[L, 1] += (wt * (f - MEAN_A[L, e])).pow(2).sum()
    return orig_forward(self, hs)
M.Qwen3MoeSparseMoeBlock.forward = sel_forward
with torch.no_grad(): model(input_ids=torch.tensor([calib_ids[4*CTX:5*CTX]], device=dev), use_cache=False)
M.Qwen3MoeSparseMoeBlock.forward = hooked_forward
r = (ratio[:, 1] / ratio[:, 0]).tolist()
SEL = [L for L in range(NL) if r[L] < 0.97]
print("held-out calib stand/drop err ratio per layer:", [round(v, 3) for v in r], flush=True)
print("selected layers (ratio<0.97):", SEL, flush=True)
MEAN_SEL = MEAN_CAL.clone(); mask = torch.ones(NL, dtype=torch.bool); mask[SEL] = False; MEAN_SEL[mask.to(dev)] = 0
res = {}
for K in (5, 4):
    for mname, MM in (("s_oracle", MEAN_ORA), ("s_sel", MEAN_SEL)):
        for un in (False, True):
            name = f"k{K}{'u' if un else ''}{mname}"
            MEAN = MM; tot = 0.0; n = 0; t1 = 0
            for ci, (dom, ids) in enumerate(chunks):
                S["mode"] = "orig"; lref = F.log_softmax(logits_of(ids), -1)
                S.update(mode="var", K=K, union=un, stand=True)
                lv = F.log_softmax(logits_of(ids), -1)
                tot += (lref.exp() * (lref - lv)).sum().item(); n += lv.shape[0]; t1 += (lv.argmax(-1) == lref.argmax(-1)).sum().item()
                del lref, lv
            res[name] = {"kl": tot / n, "top1": t1 / n}
            print(name, res[name], flush=True)
json.dump({"selected_layers": SEL, "ratio": r, "res": res}, open(f"{W}/followup_mean.json", "w"), indent=1)
