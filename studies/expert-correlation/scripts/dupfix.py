#!/usr/bin/env python3
"""Does vLLM DefaultEplbPolicy (main == 6092bb9 vendored) put 2+ replicas of one expert on the same GPU, and does
a packing step that avoids it improve balance? CPU replay on real calib/holdout traces (reuses copies.py)."""
import os, sys, json
import numpy as np
sys.path.insert(0, "/tmp/ecorr-copies-20261118")
os.environ.setdefault("ECORR_SRC", "NAS_PATH")
import copies  # noqa
D = copies.ecorr.DefaultEplbPolicy


def pack_nodup(w, logid, num_packs):
    """balanced_packing, but never put two replicas of one logical expert in the same pack if avoidable."""
    L, n = w.shape
    per = n // num_packs
    pack_index = np.full((L, n), -1, np.int64); rank = np.full((L, n), -1, np.int64)
    for l in range(L):
        pw = np.zeros(num_packs); pi = np.zeros(num_packs, np.int64); held = [set() for _ in range(num_packs)]
        for g in np.argsort(-w[l], kind="stable"):
            free = pi < per
            ok = free & np.array([logid[l, g] not in h for h in held])
            cand = np.nonzero(ok if ok.any() else free)[0]
            p = int(cand[np.argmin(pw[cand])])
            pack_index[l, g] = p; rank[l, g] = pi[p]; pw[p] += w[l, g]; pi[p] += 1; held[p].add(int(logid[l, g]))
    return pack_index, rank


def default_fixed(weight, num_phy, G):
    # num_groups = num_nodes = 1 path of rebalance_experts_hierarchical, step 3 replaced by pack_nodup
    L, E = weight.shape
    per_gpu = num_phy // G
    phy2log, logcnt = D.replicate_experts(weight, num_phy)
    tokens_per_phy = np.take_along_axis(weight / logcnt, phy2log, axis=1)
    pack_index, rank = pack_nodup(tokens_per_phy, phy2log, G)
    phy2pphy = pack_index * per_gpu + rank
    out = np.empty_like(phy2log)
    np.put_along_axis(out, phy2pphy, phy2log, axis=1)
    return out


def dups(phy, G):
    S = phy.shape[1] // G
    return int(sum(len(r) - len(set(r)) for l in range(phy.shape[0]) for r in phy[l].reshape(G, S).tolist()))


def main(prefix, tc, ec, out, gpus):
    E, K, Lr, domains, train, test = copies.load_split(prefix, tc, ec)
    load = np.stack([np.bincount(train["mix"][:, l].ravel().astype(np.int64), minlength=E) for l in range(Lr)]).astype(float)
    res = {}
    for G in gpus:
        for R in copies.r_list(E, G):
            N = E + R * G
            pd = D.rebalance_experts_hierarchical(load, N, 1, 1, G)
            pf = default_fixed(load, N, G)
            for name, phy in (("default", pd), ("default_nodup", pf)):
                ev = copies.evaluate(test["mix"], G, 2048, lambda l, b, phy=phy: phy[l], E)
                r = {k: ev[k] for k in ("sends", "imbalance", "imb_b_p50", "imb_b_p95")}
                r["dup_copies"] = dups(phy, G); r["slots"] = int(R * G * Lr)
                res[f"G{G}|R{R}|{name}"] = r
                print(G, R, name, json.dumps(r), flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], [int(x) for x in sys.argv[5:]])
