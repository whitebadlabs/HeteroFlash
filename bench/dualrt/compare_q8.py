#!/usr/bin/env python3
"""Score split-hook captures against the Q8_0 BF16-proxy reference: the same arithmetic as src/control_kld.py compare
(KL(ref || run), dNLL, same top-1, per domain), for captures that hold one file per chunk with rows 1024..2046
(HF_SPLIT_CAPTURE layout). Chunks without a capture file are skipped and reported.
  compare_q8.py REF.kld CAPDIR MANIFEST OUT.json"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from control_kld import read_header
ref, capdir, manifest, out = sys.argv[1:5]
f, n_ctx, n_vocab, n_chunk, toks = read_header(ref)
nv = 2 * ((n_vocab + 1) // 2) + 4; first = n_ctx // 2; n_rec = n_ctx - 1 - first
files = sorted(glob.glob(os.path.join(capdir, "logits_*.npy")))
man = json.load(open(manifest)); dom_of = {}
for d in man["domains"]:
    for c in range(d["chunks"][0], d["chunks"][1] + 1): dom_of[c] = d["domain"]
acc = {}
for c in range(n_chunk):
    raw = np.frombuffer(f.read(n_rec * nv * 2), dtype=np.uint16).reshape(n_rec, nv)
    if c >= len(files): continue
    hdr = raw[:, :4].copy().view(np.float32); q = raw[:, 4:4 + n_vocab].astype(np.float32)
    lp_ref = hdr[:, 1:2] + hdr[:, 0:1] * q
    lp = np.load(files[c])[:n_rec, :n_vocab].astype(np.float32); lp -= np.log(np.exp(lp).sum(1, keepdims=True))
    tgt = np.array(toks[c * n_ctx + first + 1: c * n_ctx + first + 1 + n_rec])
    p_ref = np.where(lp_ref > -16.0, np.exp(lp_ref), 0.0)
    kld = (p_ref * (lp_ref - lp)).sum(1); dnll = -lp[np.arange(n_rec), tgt] + lp_ref[np.arange(n_rec), tgt]
    same = (lp_ref.argmax(1) == lp.argmax(1)).astype(np.float32)
    a = acc.setdefault(dom_of.get(c, "?"), {"kld": [], "dnll": [], "same": []}); a["kld"].append(kld); a["dnll"].append(dnll); a["same"].append(same)
res = {}; allk, alld, alls = [], [], []
for dname, a in acc.items():
    k, dn, sm = np.concatenate(a["kld"]), np.concatenate(a["dnll"]), np.concatenate(a["same"]); allk.append(k); alld.append(dn); alls.append(sm)
    res[dname] = {"tokens": int(k.size), "kld_mean": float(k.mean()), "kld_p99": float(np.percentile(k, 99)), "dnll_mean": float(dn.mean()), "same_top": float(sm.mean())}
k, dn, sm = np.concatenate(allk), np.concatenate(alld), np.concatenate(alls)
res["_overall"] = {"tokens": int(k.size), "chunks_scored": min(len(files), n_chunk), "chunks_total": n_chunk, "kld_mean": float(k.mean()),
                   "kld_p99": float(np.percentile(k, 99)), "dnll_mean": float(dn.mean()), "dnll_se": float(dn.std() / np.sqrt(dn.size)), "same_top": float(sm.mean())}
json.dump(res, open(out, "w"), indent=1); print(json.dumps(res["_overall"]))
