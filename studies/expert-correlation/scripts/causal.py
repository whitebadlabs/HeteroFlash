#!/usr/bin/env python3
"""Causal copy planner: does the per-batch redundant-copy planner still work when it can only see the past?
(CPU replay only; reuses copies.py (copied from the sibling run) for data split, placements, copy planner, routing.)

Usage:
  causal.py --reduced PREFIX --out OUT.json [--gpus 8 16 32] [--train-cap 12000 --eval-cap 30000] [--layers N]
  causal.py --summarize OUT.json [...]

Policies at each (G, R); every one uses the same routing rule as copies.py ('send': home > already-sent > least loaded):
  def_static  (a)  vLLM DefaultEplbPolicy with E+R*G physical slots, fitted on calib mix (= copies.py 'default').
  def_lag1    (a1) Default primaries (R=0 Default placement) + the copies.py per-batch greedy copy planner fed with
                   the PREVIOUS batch's per-expert counts of the same stream (batch 0: calib-mix load).
  def_ema     (a2) same, fed with an EMA of past batches' count shares (alpha=EMA_ALPHA; seeded with calib share).
  def_oracle  (a0) same, fed with the CURRENT batch's counts (bound).
  corr_lag1   (c1) correlated primaries (copies.py 'corr' primaries) + lag-1 planner.
  corr_ema    (c2) correlated primaries + EMA planner.
  corr_oracle (c0) correlated primaries + current-batch planner (= copies.py 'corr_dynamic').
  corr_static (b)  R=0 only (parity reference).
Planner state is per stream and per layer and is reset for every test set.

Test sets ("streams", all built from the same holdout tokens as copies.py / analyze.py):
  per task type d:   copies.py test[d] (ecorr.stack, cap eval_cap, seed 2; request order as in the trace).
  mix:               copies.py test['mix'] (task-ordered concatenation; parity only).
  mixstream_s{k}:    the tokens of all test[d], split back into their requests (token order inside a request kept),
                     request order shuffled with default_rng(k) across task types, concatenated. k = 0, 1, 2.
Batches = consecutive 2048-token blocks of a stream.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import copies as cp  # noqa: E402  (copies.py imports ecorr from $ECORR_SRC)
from copies import ecorr, load_split, add_copies, route, holds_matrix, gpu_of_to_phy, log  # noqa: E402

assert cp.ROUTE == "send", "run with COPIES_ROUTE unset"
EMA_ALPHA = 0.5
MIX_SEEDS = (0, 1, 2)
GATE_SENDS, GATE_P95, GATE_SLACK = 0.15, 1.15, 0.05
DYN = {"def_lag1": ("def", "lag1"), "def_ema": ("def", "ema"), "def_oracle": ("def", "oracle"),
       "corr_lag1": ("corr", "lag1"), "corr_ema": ("corr", "ema"), "corr_oracle": ("corr", "oracle")}
POLS = ["def_static", "def_lag1", "def_ema", "def_oracle", "corr_static", "corr_lag1", "corr_ema", "corr_oracle"]
LABEL = {"def_static": "a", "def_lag1": "a1", "def_ema": "a2", "def_oracle": "a0", "corr_static": "b",
         "corr_lag1": "c1", "corr_ema": "c2", "corr_oracle": "c0"}


# ----------------------------------------------------------------------------------------------
# streams
# ----------------------------------------------------------------------------------------------
def request_segments(prefix, domains, eval_cap, test):
    """Reproduce ecorr.stack's token choice for each holdout set and split it back into requests.
    -> {d: list of arrays [t_i, L, K]} whose concatenation == test[d]."""
    reqs, _ = ecorr.load_reduced(prefix)
    out = {}
    for d in domains:
        rr = [r["ids"] for r in reqs if r["domain"] == d and r["split"] == "holdout" and len(r["ids"])]
        lens = np.array([len(a) for a in rr])
        n = lens.sum()
        idx = np.arange(n)
        if eval_cap is not None and n > eval_cap:
            idx = np.sort(np.random.default_rng(2).choice(n, eval_cap, replace=False))
        rid = np.searchsorted(np.cumsum(lens), idx, side="right")
        offs = np.concatenate([[0], np.cumsum(lens)[:-1]])
        segs = []
        for i in np.unique(rid):
            sel = idx[rid == i] - offs[i]
            segs.append(rr[i][sel])
        cat = np.concatenate(segs)
        assert cat.shape[0] == test[d].shape[0] and np.array_equal(cat[:, :test[d].shape[1]], test[d]), d
        out[d] = [s[:, :test[d].shape[1]] for s in segs]
    return out


def mixed_stream(segs, domains, seed):
    items = [(d, s) for d in domains for s in segs[d]]
    perm = np.random.default_rng(seed).permutation(len(items))
    ids = np.concatenate([items[i][1] for i in perm])
    dom = np.concatenate([np.full(len(items[i][1]), domains.index(items[i][0]), np.int16) for i in perm])
    return ids, dom, len(items)


# ----------------------------------------------------------------------------------------------
# planner + replay
# ----------------------------------------------------------------------------------------------
class Planner:
    """Per-layer copy plan for one stream. mode: static | lag1 | ema | oracle."""

    def __init__(self, mode, prim_gpu, calib_load, Wd, G, R, E, static_phy=None):
        self.mode, self.prim, self.Wd, self.G, self.R, self.E = mode, prim_gpu, Wd, G, R, E
        self.static_phy = static_phy
        Lr = len(calib_load)
        self.state = [calib_load[l] / calib_load[l].sum() for l in range(Lr)]  # lag1: counts; ema: shares
        self.prev_copies = [None] * Lr
        self.churn = []  # copy slots (expert, GPU) changed vs the previous batch, per batch-layer (from batch 1 on)

    def plan(self, l, ids_l):
        if self.mode == "static":
            return self.static_phy[l]
        cnt = np.bincount(ids_l.ravel(), minlength=self.E).astype(np.float64)
        w = cnt if self.mode == "oracle" else self.state[l]
        phy = add_copies(self.prim[l], w, self.Wd[l], self.G, self.R, "dynamic") if self.R else gpu_of_to_phy(self.prim[l])
        # update state AFTER planning (the plan never sees the current batch unless oracle)
        if self.mode == "lag1":
            self.state[l] = cnt
        elif self.mode == "ema":
            self.state[l] = EMA_ALPHA * cnt / max(cnt.sum(), 1.0) + (1 - EMA_ALPHA) * self.state[l]
        S = len(phy) // self.G
        cop = holds_matrix(phy, self.E, self.G) - holds_matrix(gpu_of_to_phy(self.prim[l]), self.E, self.G)
        if self.prev_copies[l] is not None:
            self.churn.append(int(np.maximum(cop - self.prev_copies[l], 0).sum()))
        self.prev_copies[l] = cop
        assert S == self.E // self.G + self.R
        return phy


def evaluate(test_ids, G, batch, planner, E, dom=None):
    T, Lr, K = test_ids.shape
    home = np.random.default_rng(0).integers(0, G, T)
    sends = np.zeros(Lr)
    gl_all = np.zeros((Lr, G))
    nfull = T // batch
    cell_imb = np.zeros((nfull, Lr))
    for bi, s0 in enumerate(range(0, T, batch)):
        blk = test_ids[s0:s0 + batch].astype(np.int64)
        hb = home[s0:s0 + batch]
        for l in range(Lr):
            ids_l = blk[:, l]
            phy = planner.plan(l, ids_l)
            Hc = holds_matrix(phy, E, G)
            dst = route(ids_l, hb, Hc > 0, G)
            s = np.sort(dst, 1)
            nv = np.ones_like(s, bool); nv[:, 1:] = s[:, 1:] != s[:, :-1]
            dist = nv.sum(1)
            hh = (dst == hb[:, None]).any(1)
            sends[l] += (dist - hh).sum()
            g = np.bincount(dst.ravel(), minlength=G)
            gl_all[l] += g
            if bi < nfull:
                cell_imb[bi, l] = g.max() / g.mean()
    assert np.allclose(gl_all.sum(1), T * K)
    imb_l = gl_all.max(1) / gl_all.mean(1)
    bimb = cell_imb.mean(1)
    q = lambda a, p: float(np.percentile(a, p)) if a.size else None  # noqa: E731
    r = {"sends": float((sends / T).mean()), "imbalance": float(imb_l.mean()),
         "imb_b_p50": q(bimb, 50), "imb_b_p95": q(bimb, 95), "imb_b_max": q(bimb, 100),
         "imb_b_p95_excl0": q(bimb[1:], 95),
         "batch_imb": [round(x, 5) for x in bimb], "tokens": int(T), "full_batches": int(nfull)}
    if planner.churn:
        r["churn_mean"] = float(np.mean(planner.churn))
        r["churn_frac"] = float(np.mean(planner.churn)) / max(planner.R * G, 1)
    if dom is not None and nfull:
        r["tasks_per_batch"] = [int(len(np.unique(dom[i * batch:(i + 1) * batch]))) for i in range(nfull)]
    return r


# ----------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reduced")
    ap.add_argument("--out")
    ap.add_argument("--gpus", type=int, nargs="*", default=[8, 16, 32])
    ap.add_argument("--train-cap", type=int, default=12000)
    ap.add_argument("--eval-cap", type=int, default=30000)
    ap.add_argument("--batch-tokens", type=int, default=2048)
    ap.add_argument("--layers", type=int, default=0)
    ap.add_argument("--policies", nargs="*", default=POLS)
    ap.add_argument("--R", type=int, nargs="*", default=None)
    ap.add_argument("--max-R-G32", type=int, default=2, help="G=32 only runs R <= this (cost)")
    ap.add_argument("--summarize", nargs="*", default=None)
    a = ap.parse_args()
    if a.summarize:
        print(summarize(a.summarize)); return

    E, K, Lr, domains, train, test = load_split(a.reduced, a.train_cap, a.eval_cap)
    segs = request_segments(a.reduced, domains, a.eval_cap, test)
    if a.layers:
        Lr = a.layers
        train = {d: v[:, :Lr] for d, v in train.items()}
        test = {d: v[:, :Lr] for d, v in test.items()}
        segs = {d: [s[:, :Lr] for s in v] for d, v in segs.items()}
    streams = {d: (test[d], None) for d in domains}
    streams["mix"] = (test["mix"], None)
    nreq = 0
    for k in MIX_SEEDS:
        ids, dom, nreq = mixed_stream(segs, domains, k)
        assert ids.shape == test["mix"].shape
        streams[f"mixstream_s{k}"] = (ids, dom)
    log("E", E, "K", K, "layers", Lr, "requests in mixed stream", nreq, {s: len(v[0]) for s, v in streams.items()})
    ck = a.out + ".ckpt"
    os.makedirs(ck, exist_ok=True)
    meta = {"E": E, "K": K, "layers": Lr, "domains": domains, "sets": list(streams), "reduced": os.path.abspath(a.reduced),
            "train_cap": a.train_cap, "eval_cap": a.eval_cap, "batch_tokens": a.batch_tokens, "tie_tol": cp.TIE_TOL,
            "ema_alpha": EMA_ALPHA, "mix_seeds": list(MIX_SEEDS), "mixstream_requests": int(nreq)}
    mf = os.path.join(ck, "meta.json")
    if os.path.exists(mf):
        old = json.load(open(mf))
        if old != meta:
            sys.exit(f"checkpoint {ck} made with different settings:\n{old}\nvs\n{meta}")
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

    for G in a.gpus:
        assert E % G == 0
        Rs = a.R if a.R is not None else [r for r in cp.r_list(E, G) if r >= 2]
        if G >= 32:
            Rs = [r for r in Rs if r <= a.max_R_G32]
        Rs = [0] + [r for r in Rs if r > 0]
        t0 = time.time()
        gof_def = ecorr.default_placement(load, G)

        def fit_corr():
            out = np.empty_like(gof_def)
            for l in range(Lr):
                gl = np.bincount(gof_def[l], weights=load[l], minlength=G)
                out[l] = ecorr.correlated_placement_layer(C[l], load[l], gof_def[l], G, cap=gl.max())
            return out
        gof_corr = npy(f"G{G}_corr_primaries", fit_corr)
        prim = {"def": gof_def, "corr": gof_corr}
        log(f"G{G} primaries ready ({time.time() - t0:.0f}s)")
        for R in Rs:
            for pol in a.policies:
                if R == 0 and pol not in ("def_static", "corr_static"):
                    continue
                if R > 0 and pol == "corr_static":
                    continue
                name = f"G{G}_R{R}_{pol}"
                f = os.path.join(ck, name + ".json")
                if os.path.exists(f):
                    log(name, "(checkpoint)"); continue
                t1 = time.time()
                if pol == "def_static":
                    phy = (np.stack([gpu_of_to_phy(gof_def[l]) for l in range(Lr)]) if R == 0 else
                           npy(f"G{G}_R{R}_default", lambda: ecorr.DefaultEplbPolicy.rebalance_experts_hierarchical(
                               load, E + R * G, 1, 1, G).astype(np.int64)))
                    mk = lambda: Planner("static", None, load, Wd, G, R, E, static_phy=phy)  # noqa: E731
                elif pol == "corr_static":
                    phy = np.stack([gpu_of_to_phy(gof_corr[l]) for l in range(Lr)])
                    mk = lambda: Planner("static", None, load, Wd, G, R, E, static_phy=phy)  # noqa: E731
                else:
                    pk, mode = DYN[pol]
                    mk = (lambda pk_, mode_: (lambda: Planner(mode_, prim[pk_], load, Wd, G, R, E)))(pk, mode)
                sets = list(streams) if (R == 0 or pol == "corr_oracle") else [s for s in streams if s != "mix"]
                info = {"G": G, "R": R, "policy": pol,
                        "sets": {s: evaluate(streams[s][0], G, a.batch_tokens, mk(), E, streams[s][1]) for s in sets}}
                info["seconds"] = round(time.time() - t1, 1)
                json.dump(info, open(f + ".tmp", "w")); os.replace(f + ".tmp", f)
                m = info["sets"]["mixstream_s0"]
                log(f"{name} {info['seconds']:.0f}s  mixstream_s0: sends {m['sends']:.3f} imb {m['imbalance']:.3f} "
                    f"p50/p95 {m['imb_b_p50']:.3f}/{m['imb_b_p95']:.3f}" +
                    (f"  | mix: sends {info['sets']['mix']['sends']:.4f} imb {info['sets']['mix']['imbalance']:.4f} "
                     f"p95 {info['sets']['mix']['imb_b_p95']:.3f}" if "mix" in info["sets"] else ""))
    runs = {os.path.basename(f)[:-5]: json.load(open(f)) for f in sorted(glob.glob(os.path.join(ck, "G*_R*_*.json")))}
    json.dump({"meta": meta, "runs": runs}, open(a.out, "w"))
    log("wrote", a.out)
    print(summarize([a.out]), flush=True)


# ----------------------------------------------------------------------------------------------
# summary
# ----------------------------------------------------------------------------------------------
def pooled(r, sets):
    st = [r["sets"][s] for s in sets if s in r["sets"]]
    if not st:
        return None
    pool = np.concatenate([x["batch_imb"] for x in st])
    pool1 = np.concatenate([x["batch_imb"][1:] for x in st])
    ch = [x["churn_frac"] for x in st if "churn_frac" in x]
    return {"sends": float(np.mean([x["sends"] for x in st])), "imb": float(np.mean([x["imbalance"] for x in st])),
            "p50": float(np.percentile(pool, 50)), "p95": float(np.percentile(pool, 95)), "max": float(pool.max()),
            "p95x0": float(np.percentile(pool1, 95)) if pool1.size else np.nan, "n": int(pool.size),
            "churn": float(np.mean(ch)) if ch else np.nan}


def gate(x, ref):
    cut = 1 - x["sends"] / ref["sends"]
    ok = cut >= GATE_SENDS and (x["p95"] <= GATE_P95 or x["p95"] <= ref["p95"] + GATE_SLACK)
    return cut, ok


def summarize(paths):
    L = []
    f3 = lambda v: "" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"  # noqa: E731
    for p in paths:
        ck = p if p.endswith(".ckpt") else p + ".ckpt"
        meta = json.load(open(os.path.join(ck, "meta.json")))
        runs = {}
        for f in sorted(glob.glob(os.path.join(ck, "G*_R*_*.json"))):
            r = json.load(open(f)); runs[(r["G"], r["R"], r["policy"])] = r
        doms = meta["domains"]
        ms = [s for s in meta["sets"] if s.startswith("mixstream")]
        groups = [("mixstream (3 shuffles pooled)", ms), ("task streams pooled", doms)]
        L.append(f"\n## {os.path.basename(ck)}  (E={meta['E']}, layers={meta['layers']}, EMA alpha={meta['ema_alpha']}, "
                 f"mixstream requests={meta['mixstream_requests']})\n")
        any_r = next(iter(runs.values()), None)
        if any_r:
            nb = {s: any_r["sets"][s]["full_batches"] for s in meta["sets"] if s in any_r["sets"]}
            tpb = np.concatenate([any_r["sets"][s]["tasks_per_batch"] for s in ms if s in any_r["sets"]])
            L.append(f"Full 2048-token batches per stream: {nb}. Mixed-stream batches: task types per batch "
                     f"mean {tpb.mean():.2f}, share with >=3 types {np.mean(tpb >= 3):.2f}, single-type "
                     f"{np.mean(tpb == 1):.2f}.\n")
        for gname, sets in groups:
            L.append(f"\n### {gname}\n")
            L.append("| G | R | policy | sends/tok/layer | vs a | vs same-planner Default | imb (whole) | "
                     "batch imb p50/p95 (max) | p95 excl. batch 0 | copy churn/batch | n batches |")
            L.append("|" + "---|" * 11)
            for key in sorted(runs, key=lambda k: (k[0], k[1], POLS.index(k[2]))):
                G, R, pol = key
                x = pooled(runs[key], sets)
                if x is None:
                    continue
                ra = pooled(runs.get((G, R, "def_static")), sets) if (G, R, "def_static") in runs else None
                same = {"corr_lag1": "def_lag1", "corr_ema": "def_ema", "corr_oracle": "def_oracle"}.get(pol)
                rs = pooled(runs[(G, R, same)], sets) if same and (G, R, same) in runs else None
                va = f"{100 * (x['sends'] / ra['sends'] - 1):+.1f}%" if ra else ""
                vs = f"{100 * (x['sends'] / rs['sends'] - 1):+.1f}%" if rs else ""
                L.append(f"| {G} | {R} | {LABEL[pol]} {pol} | {f3(x['sends'])} | {va} | {vs} | {f3(x['imb'])} | "
                         f"{f3(x['p50'])}/{f3(x['p95'])} ({f3(x['max'])}) | {f3(x['p95x0'])} | {f3(x['churn'])} | {x['n']} |")
            L.append(f"\nGate ({gname}): >= 15% fewer sends AND p95 <= 1.15 or <= ref p95 + 0.05; "
                     "verdict GO only if it passes vs a AND vs the same-planner Default.\n")
            for key in sorted(runs, key=lambda k: (k[0], k[1], POLS.index(k[2]))):
                G, R, pol = key
                if R == 0 or not pol.startswith("corr_"):
                    continue
                same = {"corr_lag1": "def_lag1", "corr_ema": "def_ema", "corr_oracle": "def_oracle"}[pol]
                if (G, R, "def_static") not in runs or (G, R, same) not in runs:
                    continue
                x = pooled(runs[key], sets)
                ra, rs = pooled(runs[(G, R, "def_static")], sets), pooled(runs[(G, R, same)], sets)
                ca, oka = gate(x, ra)
                cs, oks = gate(x, rs)
                L.append(f"- G{G} R{R} {LABEL[pol]} {pol}: vs a: cut {100 * ca:.1f}%, p95 {x['p95']:.3f} vs "
                         f"{ra['p95']:.3f} -> {'pass' if oka else 'fail'}; vs {LABEL[same]}: cut {100 * cs:.1f}%, "
                         f"p95 vs {rs['p95']:.3f} -> {'pass' if oks else 'fail'}  => {'GO' if oka and oks else 'no-go'}")
    return "\n".join(L)


if __name__ == "__main__":
    main()
