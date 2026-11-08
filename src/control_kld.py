#!/usr/bin/env python3
"""Score the served EXL3 control (vLLM on the Max-Q rig) against llama.cpp reference logits, token for token.

  tokens  REF.kld OUT.json                 extract the evaluation token ids (n_ctx x n_chunk) from a llama.cpp
                                           --kl-divergence-base file (stdlib only)
  client  TOKENS.json URL                  send each chunk's ids to /v1/completions (prompt_logprobs=1, max_tokens=1),
                                           sequentially; the trace hook's capture_logits saves rows [n_ctx/2, n_ctx-1)
  compare REF.kld LOGITS_DIR MANIFEST OUT  exact per-token KL(ref || control), dNLL and same-top, per domain (numpy)

llama.cpp file layout: "_logits_", n_ctx u32, n_vocab i32, n_chunk i32, tokens i32[n_ctx*n_chunk], then per chunk
(n_ctx-1-n_ctx/2) records of nv = 2*((n_vocab+1)/2)+4 uint16: float scale, float min_log_prob, uint16[n_vocab]
with log_prob_i = min_log_prob + scale * q_i (q_i = 0 means <= min_log_prob).
"""
import json
import struct
import sys
import urllib.request


def read_header(path):
    f = open(path, "rb")
    assert f.read(8) == b"_logits_"
    n_ctx = struct.unpack("<I", f.read(4))[0]
    n_vocab, n_chunk = struct.unpack("<ii", f.read(8))
    toks = struct.unpack(f"<{n_ctx * n_chunk}i", f.read(4 * n_ctx * n_chunk))
    return f, n_ctx, n_vocab, n_chunk, toks


def cmd_tokens(ref, out):
    f, n_ctx, n_vocab, n_chunk, toks = read_header(ref)
    json.dump({"n_ctx": n_ctx, "n_vocab": n_vocab, "n_chunk": n_chunk,
               "chunks": [list(toks[i * n_ctx:(i + 1) * n_ctx]) for i in range(n_chunk)]}, open(out, "w"))
    print(json.dumps({"n_ctx": n_ctx, "n_vocab": n_vocab, "n_chunk": n_chunk}))


def cmd_client(tokens, url, model="GLM-5.3-Flash"):
    t = json.load(open(tokens))
    for i, ids in enumerate(t["chunks"]):
        body = {"model": model, "prompt": ids, "max_tokens": 1, "temperature": 0.0, "prompt_logprobs": 1}
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=900) as r:
            o = json.load(r)
        print(json.dumps({"chunk": i, "prompt_tokens": o.get("usage", {}).get("prompt_tokens")}), flush=True)


def cmd_compare(ref, logits_dir, manifest, out):
    import glob
    import os

    import numpy as np
    f, n_ctx, n_vocab, n_chunk, toks = read_header(ref)
    nv = 2 * ((n_vocab + 1) // 2) + 4
    first = n_ctx // 2
    n_rec = n_ctx - 1 - first
    # this vLLM build computes prompt logits in 1024-row slices: two captured calls per 2048-token chunk
    files = sorted(glob.glob(os.path.join(logits_dir, "logits_*.npy")), key=os.path.getmtime)
    assert len(files) >= 2 * n_chunk, (len(files), 2 * n_chunk)
    files = files[-2 * n_chunk:]                               # the last 2*n_chunk captures belong to this run
    man = json.load(open(manifest))
    dom_of = {}
    for d in man["domains"]:
        for c in range(d["chunks"][0], d["chunks"][1] + 1):
            dom_of[c] = d["domain"]
    acc = {}
    for c in range(n_chunk):
        raw = np.frombuffer(f.read(n_rec * nv * 2), dtype=np.uint16).reshape(n_rec, nv)
        hdr = raw[:, :4].copy().view(np.float32)              # scale, min_log_prob
        q = raw[:, 4:4 + n_vocab].astype(np.float32)
        lp_ref = hdr[:, 1:2] + hdr[:, 0:1] * q                # exactly as llama.cpp: no renormalisation
        full = np.concatenate([np.load(files[2 * c]), np.load(files[2 * c + 1])], 0)   # positions 0..2047
        lp_ctl = full[first:first + n_rec, :n_vocab].astype(np.float32)          # rows predicting tokens first+1..
        lp_ctl -= np.log(np.exp(lp_ctl).sum(1, keepdims=True))
        tgt = np.array(toks[c * n_ctx + first + 1: c * n_ctx + first + 1 + n_rec])
        p_ref = np.where(lp_ref > -16.0, np.exp(lp_ref), 0.0)   # llama.cpp counts only log p_base > -16
        kld = (p_ref * (lp_ref - lp_ctl)).sum(1)
        dnll = -lp_ctl[np.arange(n_rec), tgt] + lp_ref[np.arange(n_rec), tgt]
        same = (lp_ref.argmax(1) == lp_ctl.argmax(1)).astype(np.float32)
        a = acc.setdefault(dom_of.get(c, "?"), {"kld": [], "dnll": [], "same": []})
        a["kld"].append(kld); a["dnll"].append(dnll); a["same"].append(same)
        print(json.dumps({"chunk": c, "domain": dom_of.get(c), "kld": float(kld.mean()), "dnll": float(dnll.mean())}), flush=True)
    res = {}
    allk, alld, alls = [], [], []
    for dname, a in acc.items():
        k, dn, sm = np.concatenate(a["kld"]), np.concatenate(a["dnll"]), np.concatenate(a["same"])
        allk.append(k); alld.append(dn); alls.append(sm)
        res[dname] = {"tokens": int(k.size), "kld_mean": float(k.mean()), "kld_p99": float(np.percentile(k, 99)),
                      "dnll_mean": float(dn.mean()), "same_top": float(sm.mean())}
    k, dn, sm = np.concatenate(allk), np.concatenate(alld), np.concatenate(alls)
    res["_overall"] = {"tokens": int(k.size), "kld_mean": float(k.mean()), "kld_p99": float(np.percentile(k, 99)),
                       "dnll_mean": float(dn.mean()), "dnll_se": float(dn.std() / np.sqrt(dn.size)), "same_top": float(sm.mean())}
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res["_overall"]))


if __name__ == "__main__":
    cmd = sys.argv[1]
    {"tokens": cmd_tokens, "client": cmd_client, "compare": cmd_compare}[cmd](*sys.argv[2:])
