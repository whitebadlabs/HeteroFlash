#!/usr/bin/env python3
"""Same-boot sensitivity (2026-11-18 deferral map): KL(ref || run), same top-1 and mean |dlogp of the ref argmax| between
two HF_SPLIT_CAPTURE dirs of the same chunks (files in the same order, log-softmax fp16 rows). Much less noisy than
two separate scores against the Q8 proxy.   kld_pair.py REFDIR RUNDIR OUT.json"""
import glob, json, os, sys
import numpy as np
ra = sorted(glob.glob(os.path.join(sys.argv[1], "logits_*.npy"))); rb = sorted(glob.glob(os.path.join(sys.argv[2], "logits_*.npy")))
ks, ss = [], []
for fa, fb in zip(ra, rb):
    a = np.load(fa).astype(np.float32); b = np.load(fb).astype(np.float32)
    a -= np.log(np.exp(a).sum(1, keepdims=True)); b -= np.log(np.exp(b).sum(1, keepdims=True))
    pa = np.exp(a)
    ks.append((pa * (a - b)).sum(1)); ss.append((a.argmax(1) == b.argmax(1)).astype(np.float32))
k, s = np.concatenate(ks), np.concatenate(ss)
res = {"files": [len(ra), len(rb)], "tokens": int(k.size), "kld_mean": float(k.mean()), "kld_se": float(k.std() / np.sqrt(k.size)),
       "kld_p99": float(np.percentile(k, 99)), "same_top": float(s.mean())}
json.dump(res, open(sys.argv[3], "w")); print(json.dumps(res))
