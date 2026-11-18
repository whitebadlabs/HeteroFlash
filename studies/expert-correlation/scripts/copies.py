#!/usr/bin/env python3
"""Copy-budget experiment: do a few redundant expert copies restore load balance for the
correlation-aware placement while keeping most of its cross-GPU send cut? (CPU replay only.)

Usage:
  copies.py --reduced PREFIX --out OUT.json [--gpus 8 16 32] [--train-cap 12000 --eval-cap 30000]
  copies.py --summarize OUT.json [OUT2.json]      # tables + go/no-go from checkpoints, any time
  ecorr is imported from $ECORR_SRC if set, else ../src relative to this file (repo/scripts -> repo/src).

Data split = analyze.py's (copied verbatim): calib/holdout by request, train seed 1 / test seed 2, same caps,
'mix' = concatenation of the per-task sets in sorted task order. Everything is fitted on calib 'mix';
everything reported is measured on holdout (each task type and mix).

Placements (each GPU holds E/G primaries + R copy slots; R=0 rows are the no-copy references):
  a  default   vLLM DefaultEplbPolicy (vendored, unchanged) called with num_physical = E + R*G
               (rebalance_experts_hierarchical(load, E+R*G, 1, 1, G), as vLLM calls it for num_groups = 1).
               It may put two copies of one expert on the same GPU; counted as 'dup_copies'.
  b  corr_static  correlated greedy (ecorr.correlated_placement_layer, cap 1.0x of the no-copy Default max)
               for the E primaries, then R*G copies, one at a time: expert = largest calib load per copy
               (as vLLM replicate_experts); GPU = least estimated calib load among GPUs with a free slot that do
               not hold it; near-ties (within TIE_TOL of mean GPU load from the minimum) broken by the largest
               co-selection with the experts already on the GPU.
  c  corr_dynamic same primaries as b; per 2048-token batch and layer, a greedy planner that sees that batch's
               own expert counts places the R*G copies: repeatedly take the busiest GPU (estimated load =
               sum count/copies), copy its expert with the largest count per copy that can still be copied to the
               least-loaded GPU with a free slot (same tie rule). Sees the batch: an oracle-ish optimistic case,
               not a guaranteed upper bound (the planner's load proxy differs from the routing rule).
  d  anti      start from a's physical slots; greedy slot swaps that LOWER within-GPU co-selection (the same swap
               search as ecorr.correlated_placement_layer, run on the negated slot co-selection matrix; two copies
               of one expert weigh C[e,e], so they are pushed apart), estimated calib GPU load <= a's max.

Token routing with copies (this study's rule, applied to every policy; NOT vLLM's runtime replica choice):
  per batch and layer: experts with one copy-GPU go there. For an expert with copies on several GPUs:
  1. a copy on the token's home GPU, else
  2. a copy on a GPU the token already sends to (from its other picks; iterated until no change), else
  3. the least-loaded copy: remaining picks are water-filled per expert (largest first) onto its copy GPUs,
     using the batch's running GPU load.
Home GPU = default_rng(0).integers(0, G, T) per test set, as ecorr.replay.

Metrics (per G, R, policy, test set): sends per token per layer; whole-holdout imbalance (max/mean GPU load,
mean over layers, = analyze.py 'imb'); per full 2048-token batch imbalance (mean over layers per batch,
= analyze.py 'imb_b') with p50/p95 over batches, and p50/p95 over (batch, layer) cells; MegaFlux
kappa = E * sum(share^2) of expert load per batch-layer (a property of the batch, the same for every placement)
and kappa_slot = N_slots * sum(share^2) of per-copy load (placement-dependent).
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.environ.get("ECORR_SRC") or
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import ecorr  # noqa: E402

TIE_TOL = 0.01          # near-tie window for copy GPU choice, as a fraction of the mean GPU load
ANTI_MAX_SWAPS = 1500   # swap cap per layer for the anti-paired search (reported if hit)
GATE_SENDS = 0.15       # go/no-go: >= 15% fewer sends than (a) at the same R ...
GATE_P95 = 1.15         # ... with p95 batch imbalance <= 1.15 or <= (a) + 0.05
GATE_P95_SLACK = 0.05
POLS = ["default", "corr_static", "corr_dynamic", "anti"]
# routing rule: "send" = home > already-sent GPU > least-loaded (main); "balance" = least-loaded copy only
# (sensitivity run; set COPIES_ROUTE=balance and use a separate --out)
ROUTE = os.environ.get("COPIES_ROUTE", "send")


def log(*x):
    print(time.strftime("%H:%M:%S"), *x, flush=True)


# ----------------------------------------------------------------------------------------------
# data (verbatim from analyze.main)
# ----------------------------------------------------------------------------------------------
def load_split(prefix, train_cap, eval_cap):
    reqs, meta = ecorr.load_reduced(prefix)
    E = int(meta.get("num_experts") or 1 + max(int(r["ids"].max()) for r in reqs))
    K = reqs[0]["ids"].shape[2]
    Lr = reqs[0]["ids"].shape[1]
    domains = sorted({r["domain"] for r in reqs})
    by = {(d, s): [r for r in reqs if r["domain"] == d and r["split"] == s] for d in domains for s in ("calib", "holdout")}
    train = {d: ecorr.stack(by[(d, "calib")], cap=train_cap, seed=1) for d in domains}
    test = {d: ecorr.stack(by[(d, "holdout")], cap=eval_cap, seed=2) for d in domains}
    train["mix"] = np.concatenate([train[d] for d in domains])
    test["mix"] = np.concatenate([test[d] for d in domains])
    return E, K, Lr, domains, train, test


# ----------------------------------------------------------------------------------------------
# placements: phy2log [L, N] with slot p on GPU p // S (S = N / G)
# ----------------------------------------------------------------------------------------------
def gpu_of_to_phy(gpu_of_l):
    return np.argsort(gpu_of_l, kind="stable")  # E/G experts per GPU, GPU-major


def holds_matrix(phy2log_l, E, G):
    S = len(phy2log_l) // G
    H = np.zeros((E, G), np.int64)
    np.add.at(H, (phy2log_l, np.arange(len(phy2log_l)) // S), 1)
    return H


def add_copies(prim_gpu, w, W, G, R, mode):
    """Place R*G copies on top of primaries prim_gpu [E] -> phy2log [G*(E/G+R)].

    w: expert load used for the plan (calib load for 'static', the batch's counts for 'dynamic').
    W: calib co-selection (diag 0) for the tie-break.
    """
    E = len(prim_gpu)
    H = np.zeros((E, G), bool); H[np.arange(E), prim_gpu] = True
    cnt = np.ones(E)
    free = np.full(G, R)
    extra = [[] for _ in range(G)]
    tol = TIE_TOL * w.sum() / G
    for _ in range(R * G):
        per = w / cnt
        gl = H.T.astype(np.float64) @ per
        if mode == "static":
            order = np.argsort(-per, kind="stable")
        else:  # dynamic: lower the busiest GPU
            b = int(np.argmax(gl))
            on_b = np.nonzero(H[:, b])[0]
            order = np.concatenate([on_b[np.argsort(-per[on_b], kind="stable")],
                                    np.argsort(-per, kind="stable")])
        e = -1
        for x in order:
            if ((~H[x]) & (free > 0)).any():
                e = int(x); break
        if e < 0:
            break
        feas = (~H[e]) & (free > 0)
        glm = np.where(feas, gl, np.inf)
        cand = np.nonzero(feas & (glm <= glm.min() + tol))[0]
        if len(cand) > 1:
            g = int(cand[np.argmax(W[e] @ H[:, cand])])
        else:
            g = int(cand[0])
        H[e, g] = True; cnt[e] += 1; free[g] -= 1; extra[g].append(e)
    phy = []
    for g in range(G):
        phy.extend(np.nonzero(prim_gpu == g)[0].tolist())
        phy.extend(extra[g])
    phy = np.array(phy, np.int64)
    assert len(phy) == G * (E // G + R), (len(phy), G, R)
    return phy


def anti_layer(phy_l, C, load, G, cap):
    """Slot swaps that lower within-GPU co-selection. Returns (new phy2log GPU-major, swaps)."""
    N = len(phy_l)
    S = N // G
    E = C.shape[0]
    cnt = np.bincount(phy_l, minlength=E).astype(np.float64)
    Wp = C[np.ix_(phy_l, phy_l)].astype(np.float64)
    slot_load = load[phy_l] / cnt[phy_l]
    g0 = np.arange(N) // S
    ecorr.SWAPS.clear()
    g = ecorr.correlated_placement_layer(-Wp, slot_load, g0, G, cap=cap, max_swaps=ANTI_MAX_SWAPS)
    return phy_l[np.argsort(g, kind="stable")], int(ecorr.SWAPS[-1])


def est_gpu_load(phy_l, load, G):
    E = len(load)
    cnt = np.bincount(phy_l, minlength=E).astype(np.float64)
    S = len(phy_l) // G
    return np.bincount(np.arange(len(phy_l)) // S, weights=load[phy_l] / cnt[phy_l], minlength=G)


def intra_cosel(phy_l, C, G):
    S = len(phy_l) // G
    tot = 0.0
    for g in range(G):
        u = np.unique(phy_l[g * S:(g + 1) * S])
        sub = C[np.ix_(u, u)]
        tot += sub.sum() - np.trace(sub)
    return tot


# ----------------------------------------------------------------------------------------------
# routing
# ----------------------------------------------------------------------------------------------
def waterfill(l, n):
    """Integer counts c >= 0, sum n, minimizing max(l + c)."""
    m_all = len(l)
    order = np.argsort(l, kind="stable")
    ls = l[order].astype(np.float64)
    cs = np.cumsum(ls)
    m = m_all
    while m > 1 and (n + cs[m - 1]) / m < ls[m - 1]:
        m -= 1
    level = (n + cs[m - 1]) / m
    c = np.zeros(m_all, np.int64)
    c[:m] = np.maximum(np.floor(level - ls[:m]), 0).astype(np.int64)
    rem = n - c.sum()
    if rem > 0:
        idx = np.argsort(ls[:m] + c[:m], kind="stable")[:rem]
        c[idx] += 1
    out = np.zeros(m_all, np.int64); out[order] = c
    return out


def route(ids_b, home_b, Hb, G):
    """ids_b [B, K] int64, home_b [B], Hb [E, G] bool (expert has a copy on GPU). -> dst [B, K]."""
    B, K = ids_b.shape
    nc = Hb.sum(1)
    gpu1 = np.argmax(Hb, 1)
    multi = nc[ids_b] > 1
    dst = np.where(multi, -1, gpu1[ids_b])
    if not multi.any():
        return dst
    t, k = np.nonzero(multi)
    e = ids_b[t, k]
    if ROUTE == "balance":  # sensitivity variant: skip rules 1-2, every multi-copy pick is water-filled
        gl = np.bincount(dst[dst >= 0], minlength=G).astype(np.float64)
        ue, inv, cnts = np.unique(e, return_inverse=True, return_counts=True)
        for j in np.argsort(-cnts, kind="stable"):
            sel = np.nonzero(inv == j)[0]
            gs = np.nonzero(Hb[ue[j]])[0]
            c = waterfill(gl[gs], len(sel))
            dst[t[sel], k[sel]] = np.repeat(gs, c)
            gl[gs] += c
        return dst
    # rule 1: home copy
    r1 = Hb[e, home_b[t]]
    dst[t[r1], k[r1]] = home_b[t[r1]]
    t, k, e = t[~r1], k[~r1], e[~r1]
    # rule 2: a GPU the token already sends to (iterate: resolved picks add GPUs)
    have = np.zeros((B, G), bool)
    fixed = dst >= 0
    have[np.nonzero(fixed)[0], dst[fixed]] = True
    while len(t):
        cand = Hb[e] & have[t]
        hit = cand.any(1)
        if not hit.any():
            break
        gsel = np.argmax(cand[hit], 1)
        dst[t[hit], k[hit]] = gsel
        have[t[hit], gsel] = True
        t, k, e = t[~hit], k[~hit], e[~hit]
    # rule 3: least-loaded copy, water-filled per expert (largest group first) on the running batch load
    if len(t):
        gl = np.bincount(dst[dst >= 0], minlength=G).astype(np.float64)
        ue, inv, cnts = np.unique(e, return_inverse=True, return_counts=True)
        for j in np.argsort(-cnts, kind="stable"):
            sel = np.nonzero(inv == j)[0]
            gs = np.nonzero(Hb[ue[j]])[0]
            c = waterfill(gl[gs], len(sel))
            dst[t[sel], k[sel]] = np.repeat(gs, c)
            gl[gs] += c
    return dst


def evaluate(test_ids, G, batch, plan_fn, E):
    """plan_fn(l, blk_ids_l) -> phy2log for layer l (static placements ignore the block)."""
    T, Lr, K = test_ids.shape
    home = np.random.default_rng(0).integers(0, G, T)
    sends = np.zeros(Lr); distinct = np.zeros(Lr)
    gl_all = np.zeros((Lr, G))
    nfull = T // batch
    cell_imb = np.zeros((nfull, Lr)); cell_kap = np.zeros((nfull, Lr)); cell_kslot = np.zeros((nfull, Lr))
    for bi, s0 in enumerate(range(0, T, batch)):
        blk = test_ids[s0:s0 + batch].astype(np.int64)
        hb = home[s0:s0 + batch]
        for l in range(Lr):
            ids_l = blk[:, l]
            phy = plan_fn(l, ids_l)
            Hc = holds_matrix(phy, E, G)
            dst = route(ids_l, hb, Hc > 0, G)
            assert (Hc[ids_l, dst] > 0).all()
            s = np.sort(dst, 1)
            nv = np.ones_like(s, bool); nv[:, 1:] = s[:, 1:] != s[:, :-1]
            dist = nv.sum(1)
            hh = (dst == hb[:, None]).any(1)
            distinct[l] += dist.sum(); sends[l] += (dist - hh).sum()
            g = np.bincount(dst.ravel(), minlength=G)
            gl_all[l] += g
            if bi < nfull:
                cell_imb[bi, l] = g.max() / g.mean()
                ce = np.bincount(ids_l.ravel(), minlength=E).astype(np.float64)
                cell_kap[bi, l] = E * ((ce / ce.sum()) ** 2).sum()
                # per-copy load: (expert, GPU) load split evenly over duplicate copies on that GPU
                eg = np.bincount(ids_l.ravel() * G + dst.ravel(), minlength=E * G).reshape(E, G)
                gslot = np.arange(len(phy)) // (len(phy) // G)
                sl = eg[phy, gslot] / Hc[phy, gslot]
                cell_kslot[bi, l] = len(phy) * ((sl / sl.sum()) ** 2).sum()
    assert np.allclose(gl_all.sum(1), T * K)
    imb_l = gl_all.max(1) / gl_all.mean(1)
    bimb = cell_imb.mean(1)
    q = lambda a, p: float(np.percentile(a, p)) if a.size else None  # noqa: E731
    return {"sends": float((sends / T).mean()), "distinct": float((distinct / T).mean()),
            "sends_per_layer": [round(x, 4) for x in sends / T],
            "imbalance": float(imb_l.mean()),
            "imbalance_batch": float(bimb.mean()) if nfull else None,
            "imb_b_p50": q(bimb, 50), "imb_b_p95": q(bimb, 95),
            "imb_cell_p50": q(cell_imb, 50), "imb_cell_p95": q(cell_imb, 95),
            "kappa_p50": q(cell_kap.mean(1), 50), "kappa_p95": q(cell_kap.mean(1), 95),
            "kappa_slot_p50": q(cell_kslot.mean(1), 50), "kappa_slot_p95": q(cell_kslot.mean(1), 95),
            "batch_imb": [round(x, 5) for x in bimb], "batch_kappa": [round(x, 4) for x in cell_kap.mean(1)],
            "batch_kappa_slot": [round(x, 4) for x in cell_kslot.mean(1)],
            "tokens": int(T), "full_batches": int(nfull)}


# ----------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------
def r_list(E, G):
    return sorted({1, 2, int(np.floor(0.25 * E / G + 0.5))})  # 25% of E/G, half up (GLM G=16: 4.5 -> 5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reduced")
    ap.add_argument("--out")
    ap.add_argument("--gpus", type=int, nargs="*", default=[8, 16, 32])
    ap.add_argument("--train-cap", type=int, default=12000)
    ap.add_argument("--eval-cap", type=int, default=30000)
    ap.add_argument("--batch-tokens", type=int, default=2048)
    ap.add_argument("--layers", type=int, default=0, help="smoke test: first N layers only")
    ap.add_argument("--sets", nargs="*", default=None, help="smoke test: subset of test sets")
    ap.add_argument("--policies", nargs="*", default=POLS)
    ap.add_argument("--R", type=int, nargs="*", default=None, help="override R list (bare --R = R=0 only)")
    ap.add_argument("--summarize", nargs="*", default=None)
    a = ap.parse_args()
    if a.summarize:
        print(summarize(a.summarize))
        return

    E, K, Lr, domains, train, test = load_split(a.reduced, a.train_cap, a.eval_cap)
    if a.layers:
        Lr = a.layers
        train = {d: v[:, :Lr] for d, v in train.items()}
        test = {d: v[:, :Lr] for d, v in test.items()}
    sets = a.sets or (domains + ["mix"])
    log("E", E, "K", K, "layers", Lr, "sets", sets, {d: len(test[d]) for d in sets})
    ck = a.out + ".ckpt"
    os.makedirs(ck, exist_ok=True)
    meta = {"E": E, "K": K, "layers": Lr, "domains": domains, "sets": sets, "reduced": os.path.abspath(a.reduced),
            "train_cap": a.train_cap, "eval_cap": a.eval_cap, "batch_tokens": a.batch_tokens,
            "tie_tol": TIE_TOL, "anti_max_swaps": ANTI_MAX_SWAPS}
    if ROUTE != "send":  # keeps meta of the main run unchanged
        meta["route"] = ROUTE
    mf = os.path.join(ck, "meta.json")
    if os.path.exists(mf):
        old = json.load(open(mf))
        if old != meta:
            sys.exit(f"checkpoint {ck} was made with different settings:\n{old}\nvs\n{meta}")
    else:
        json.dump(meta, open(mf, "w"))

    load = np.stack([np.bincount(train["mix"][:, l].ravel().astype(np.int64), minlength=E)
                     for l in range(Lr)]).astype(np.float64)
    C = [ecorr.coselect_matrix(train["mix"][:, l], E).astype(np.float64) for l in range(Lr)]
    Wd = []
    for c in C:
        w = c.copy(); np.fill_diagonal(w, 0.0); Wd.append(w)
    log("calib matrices done")

    def npy(name, fn):
        f = os.path.join(ck, name + ".npy")
        if os.path.exists(f):
            return np.load(f)
        x = fn(); np.save(f + ".tmp.npy", x); os.replace(f + ".tmp.npy", f)
        return x

    def done(name):
        return os.path.exists(os.path.join(ck, name + ".json"))

    def save(name, obj):
        f = os.path.join(ck, name + ".json")
        json.dump(obj, open(f + ".tmp", "w")); os.replace(f + ".tmp", f)

    for G in a.gpus:
        assert E % G == 0
        Rs = a.R if a.R is not None else r_list(E, G)
        Rs = [0] + [r for r in Rs if r > 0]
        t0 = time.time()
        gof_def = ecorr.default_placement(load, G)
        phy_def0 = np.stack([gpu_of_to_phy(gof_def[l]) for l in range(Lr)])

        def fit_corr():
            out = np.empty_like(gof_def)
            for l in range(Lr):
                gl = np.bincount(gof_def[l], weights=load[l], minlength=G)
                out[l] = ecorr.correlated_placement_layer(C[l], load[l], gof_def[l], G, cap=gl.max())
            return out
        gof_corr = npy(f"G{G}_corr_primaries", fit_corr)
        log(f"G{G} primaries ready ({time.time() - t0:.0f}s)")

        def default_phy(R):
            if R == 0:
                return phy_def0
            return npy(f"G{G}_R{R}_default", lambda: ecorr.DefaultEplbPolicy.rebalance_experts_hierarchical(
                load, E + R * G, 1, 1, G).astype(np.int64))

        for R, pol in [(R, p) for p in POLS for R in Rs]:
            if pol not in a.policies or (pol == "corr_dynamic" and R == 0):
                continue
            name = f"G{G}_R{R}_{pol}"
            if done(name):
                log(name, "(checkpoint)")
                continue
            t1 = time.time()
            info = {"G": G, "R": R, "policy": pol}
            if pol == "default":
                phy = default_phy(R)
                info["dup_copies"] = int(sum((holds_matrix(phy[l], E, G) > 1).sum() for l in range(Lr)))
            elif pol == "corr_static":
                phy = npy(name, lambda: np.stack(
                    [add_copies(gof_corr[l], load[l], Wd[l], G, R, "static") if R else gpu_of_to_phy(gof_corr[l])
                     for l in range(Lr)]))
            elif pol == "anti":
                base = default_phy(R)
                f_sw = os.path.join(ck, name + "_swaps.json")
                f_np = os.path.join(ck, name + ".npy")
                if os.path.exists(f_np) and os.path.exists(f_sw):
                    phy = np.load(f_np); sw = json.load(open(f_sw))
                else:
                    res = [anti_layer(base[l], C[l], load[l], G, cap=est_gpu_load(base[l], load[l], G).max())
                           for l in range(Lr)]
                    phy = np.stack([x[0] for x in res]); sw = [x[1] for x in res]
                    json.dump(sw, open(f_sw, "w"))
                    np.save(f_np + ".tmp.npy", phy); os.replace(f_np + ".tmp.npy", f_np)
                info["swaps_per_layer"] = sw
                info["swap_cap_hit_layers"] = int(sum(x >= ANTI_MAX_SWAPS for x in sw))
                info["fit_seconds"] = round(time.time() - t1, 1)
                info["intra_cosel_ratio_vs_default"] = float(np.mean(
                    [intra_cosel(phy[l], C[l], G) / max(intra_cosel(base[l], C[l], G), 1e-9) for l in range(Lr)]))
            if pol != "corr_dynamic":
                assert phy.shape[1] == E + R * G
                for l in range(Lr):
                    assert (np.bincount(phy[l], minlength=E) >= 1).all()
                info["calib_est_imb"] = float(np.mean(
                    [(lambda x: x.max() / x.mean())(est_gpu_load(phy[l], load[l], G)) for l in range(Lr)]))
                fn = (lambda P: (lambda l, ids_l: P[l]))(phy)
            else:  # corr_dynamic: plan per batch from the batch's own counts
                fn = (lambda R_: (lambda l, ids_l: add_copies(
                    gof_corr[l], np.bincount(ids_l.ravel(), minlength=E).astype(np.float64), Wd[l], G, R_,
                    "dynamic")))(R)
            info["sets"] = {B: evaluate(test[B], G, a.batch_tokens, fn, E) for B in sets}
            info["seconds"] = round(time.time() - t1, 1)
            save(name, info)
            m = info["sets"].get("mix") or next(iter(info["sets"].values()))
            log(f"{name} {time.time() - t1:.0f}s  mix: sends {m['sends']:.3f} imb {m['imbalance']:.3f} "
                f"imb_b p50/p95 {m['imb_b_p50']}/{m['imb_b_p95']}")
    allr = {}
    for f in sorted(glob.glob(os.path.join(ck, "G*_R*_*.json"))):
        if not f.endswith("_swaps.json"):
            allr[os.path.basename(f)[:-5]] = json.load(open(f))
    json.dump({"meta": meta, "runs": allr}, open(a.out, "w"))
    log("wrote", a.out)
    print(summarize([a.out]), flush=True)


# ----------------------------------------------------------------------------------------------
# summary + go/no-go
# ----------------------------------------------------------------------------------------------
def summarize(paths):
    lines = []
    f3 = lambda x: "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"  # noqa: E731
    for p in paths:
        ck = p if p.endswith(".ckpt") else p + ".ckpt"
        meta = json.load(open(os.path.join(ck, "meta.json")))
        runs = {}
        for f in sorted(glob.glob(os.path.join(ck, "G*_R*_*.json"))):
            if not f.endswith("_swaps.json"):
                r = json.load(open(f)); runs[(r["G"], r["R"], r["policy"])] = r
        tasks = [s for s in meta["sets"] if s != "mix"]

        def agg(rr):
            st = rr["sets"]
            tk = [t for t in tasks if t in st]
            if not tk:
                return (np.nan,) * 4
            pool = np.concatenate([st[t]["batch_imb"] for t in tk])
            return (np.mean([st[t]["sends"] for t in tk]), np.mean([st[t]["imbalance"] for t in tk]),
                    np.percentile(pool, 50) if pool.size else np.nan, np.percentile(pool, 95) if pool.size else np.nan)

        nb = {t: runs[next(iter(runs))]["sets"][t]["full_batches"] for t in meta["sets"]} if runs else {}
        lines.append(f"\n## {os.path.basename(ck)}  (E={meta['E']}, layers={meta['layers']})\n")
        lines.append(f"Simulated replay on holdout. 'tasks' = mean over task types for sends/imb; imb_b p50/p95 over "
                     f"full 2048-token batches pooled across task types (full batches per set: {nb}). 'mix' = mix "
                     f"holdout (task-ordered concatenation, so its batches are mostly single-task). vs (a) = vs "
                     f"DefaultEplbPolicy at the same R.\n")
        lines.append("| G | R | policy | sends tasks | vs (a) | imb tasks | imb_b p50/p95 tasks | sends mix | vs (a) | "
                     "imb mix | imb_b p50/p95 mix | kappa p50/p95 mix | kappa_slot p95 mix | note |")
        lines.append("|" + "---|" * 14)
        gate = []
        for key in sorted(runs, key=lambda x: (x[0], x[1], POLS.index(x[2]))):
            G, R, pol = key
            r = runs[key]; m = r["sets"].get("mix")
            ref = runs.get((G, R, "default"))
            s_t, i_t, p50_t, p95_t = agg(r)
            note = {"default": f"dup copies {r.get('dup_copies', 0)}" if r.get("dup_copies") else "",
                    "corr_dynamic": "sees the batch",
                    "anti": f"intra-cosel x{r.get('intra_cosel_ratio_vs_default', np.nan):.2f}, swap cap hit "
                            f"{r.get('swap_cap_hit_layers')} layers" if pol == "anti" else ""}.get(pol, "")
            vs_t = vs_m = ""
            if ref:
                rs_t, _, _, rp95_t = agg(ref)
                vs_t = f"{100 * (s_t / rs_t - 1):+.1f}%"
                if m and "mix" in ref["sets"]:
                    vs_m = f"{100 * (m['sends'] / ref['sets']['mix']['sends'] - 1):+.1f}%"
            mixcols = (f"{f3(m['sends'])} | {vs_m} | {f3(m['imbalance'])} | {f3(m['imb_b_p50'])}/{f3(m['imb_b_p95'])} | "
                       f"{f3(m['kappa_p50'])}/{f3(m['kappa_p95'])} | {f3(m['kappa_slot_p95'])}") if m else " | | | | | "
            lines.append(f"| {G} | {R} | {pol} | {f3(s_t)} | {vs_t} | {f3(i_t)} | {f3(p50_t)}/{f3(p95_t)} | {mixcols} | {note} |")
            if pol in ("corr_static", "corr_dynamic") and R > 0 and ref:
                cases = [("tasks", s_t, p95_t, rs_t, rp95_t)]
                if m and "mix" in ref["sets"]:
                    cases.append(("mix", m["sends"], m["imb_b_p95"], ref["sets"]["mix"]["sends"],
                                  ref["sets"]["mix"]["imb_b_p95"]))
                for lab, sa, pa, sr, pr in cases:
                    if pa is None or pr is None or np.isnan(sa) or np.isnan(pa):
                        continue
                    cut = 1 - sa / sr
                    ok = cut >= GATE_SENDS and (pa <= GATE_P95 or pa <= pr + GATE_P95_SLACK)
                    gate.append(f"- G{G} R{R} {pol} [{lab}]: send cut vs (a) {100 * cut:.1f}%, p95 imb_b {pa:.3f} "
                                f"(a: {pr:.3f}) -> {'GO' if ok else 'no-go'}")
        lines.append("\nGo/no-go (fixed before results: >= 15% fewer sends than (a) at the same R AND p95 batch "
                     "imbalance <= 1.15 or <= (a)+0.05; judged on task-pooled batches and on mix):")
        lines.extend(gate)
    return "\n".join(lines)


if __name__ == "__main__":
    main()
