#!/usr/bin/env python3
"""Paired comparison of two captured balanced-eval runs (same tokens, same positions): per-token NLL difference on the
true next token (B - A, mean and SE over tokens) and the direct KL(A || B) between the two runs' distributions.
  paired_quality.py TOKENS.json CAPDIR_A CAPDIR_B OUT.json      (capture files rows [1024, 2047) of each 2048 chunk)"""
import glob, json, sys
import numpy as np
toks = json.load(open(sys.argv[1]))["chunks"]
A = sorted(glob.glob(sys.argv[2] + "/logits_*.npy")); B = sorted(glob.glob(sys.argv[3] + "/logits_*.npy"))
assert len(A) == len(B) == len(toks), (len(A), len(B), len(toks))
d_nll, kl, same = [], [], []
for i, (fa, fb) in enumerate(zip(A, B)):
    la = np.load(fa).astype(np.float32); lb = np.load(fb).astype(np.float32)
    tgt = np.array(toks[i][1025:2048])                     # row r predicts token at position 1024 + r + 1
    ia = la[np.arange(len(tgt)), tgt]; ib = lb[np.arange(len(tgt)), tgt]
    d_nll.append(ia - ib)                                  # NLL_B - NLL_A = logp_A - logp_B
    pa = np.exp(la); kl.append((pa * (la - lb)).sum(-1)); same.append(la.argmax(-1) == lb.argmax(-1))
d = np.concatenate(d_nll); k = np.concatenate(kl); s = np.concatenate(same)
out = {"A": sys.argv[2], "B": sys.argv[3], "tokens": int(d.size), "paired_dnll_B_minus_A": float(d.mean()),
       "paired_dnll_se": float(d.std(ddof=1) / np.sqrt(d.size)), "kld_A_B_mean": float(k.mean()), "kld_A_B_p99": float(np.quantile(k, 0.99)),
       "same_top1": float(s.mean())}
print(json.dumps(out)); json.dump(out, open(sys.argv[4], "w"), indent=1)
