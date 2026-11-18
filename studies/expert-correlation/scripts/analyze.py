#!/usr/bin/env python3
"""Steps 4 and 5 on a reduced routing trace (CPU only).

Usage: python3 analyze.py --reduced PREFIX --out results.json [--gpus 8 16 32] [--train-cap 12000]

Train / test: every request is in the 'calib' or 'holdout' split (by prompt). Everything that is
fitted (partners, transition model, placements, the load fed to DefaultEplbPolicy) uses calib
tokens; every number reported as a result is measured on holdout tokens.
'mix' = all task types, calib tokens capped at --train-cap per task type so that the 4 long-context
prompts (~160K prefill tokens) do not dominate.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import ecorr  # noqa: E402


def log(*x):
    print(time.strftime("%H:%M:%S"), *x, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reduced", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpus", type=int, nargs="*", default=[8, 16, 32])
    ap.add_argument("--train-cap", type=int, default=12000)
    ap.add_argument("--eval-cap", type=int, default=30000)
    ap.add_argument("--slack", type=float, nargs="*", default=[1.0, 1.1])
    ap.add_argument("--random-seeds", type=int, default=3)
    ap.add_argument("--batch-tokens", type=int, default=2048)
    a = ap.parse_args()

    reqs, meta = ecorr.load_reduced(a.reduced)
    E = int(meta.get("num_experts") or 1 + max(int(r["ids"].max()) for r in reqs))
    K = reqs[0]["ids"].shape[2]
    Lr = reqs[0]["ids"].shape[1]
    domains = sorted({r["domain"] for r in reqs})
    by = {(d, s): [r for r in reqs if r["domain"] == d and r["split"] == s] for d in domains for s in ("calib", "holdout")}
    train = {d: ecorr.stack(by[(d, "calib")], cap=a.train_cap, seed=1) for d in domains}
    test = {d: ecorr.stack(by[(d, "holdout")], cap=a.eval_cap, seed=2) for d in domains}
    train["mix"] = np.concatenate([train[d] for d in domains])
    test["mix"] = np.concatenate([test[d] for d in domains])
    sets = domains + ["mix"]
    R = {"E": E, "K": K, "layers": Lr, "domains": domains,
         "tokens": {d: {"calib_all": int(sum(len(r["ids"]) for r in by[(d, "calib")])) if d != "mix" else None,
                        "holdout_all": int(sum(len(r["ids"]) for r in by[(d, "holdout")])) if d != "mix" else None,
                        "train_used": int(len(train[d])), "test_used": int(len(test[d])),
                        "requests_calib": len(by[(d, "calib")]) if d != "mix" else None,
                        "requests_holdout": len(by[(d, "holdout")]) if d != "mix" else None} for d in sets}}
    R["moe_layers"] = meta.get("moe_layers")
    log("tokens", json.dumps(R["tokens"]))

    # checkpoints: <out>.ckpt/stage1.json (step 4) and sim_G<G>_<fit>.json (step 5), so a restart resumes
    ck = a.out + ".ckpt"
    os.makedirs(ck, exist_ok=True)

    def ck_load(name):
        f = os.path.join(ck, name + ".json")
        return json.load(open(f)) if os.path.exists(f) else None

    def ck_save(name, obj):
        f = os.path.join(ck, name + ".json")
        json.dump(obj, open(f + ".tmp", "w")); os.replace(f + ".tmp", f)

    stage1 = ck_load("stage1")

    # ---------------- per-set matrices (calib) ----------------
    loads = {d: np.stack([np.bincount(train[d][:, l].ravel().astype(np.int64), minlength=E) for l in range(Lr)])
             .astype(np.float64) for d in sets}
    C_tr = {d: [ecorr.coselect_matrix(train[d][:, l], E) for l in range(Lr)] for d in sets}
    C_te = {d: [ecorr.coselect_matrix(test[d][:, l], E) for l in range(Lr)] for d in (sets if stage1 is None else ["mix"])}
    log("matrices done")
    if stage1 is not None:
        R.update(stage1)
        log("stage1 loaded from checkpoint")
    else:
        stage4(R, a, sets, domains, by, train, test, loads, C_tr, C_te, E, K, Lr)
        ck_save("stage1", {k: R[k] for k in ("load_skew", "coselect_partner", "next_layer_recall",
                                              "next_layer_recall_by_phase")})
    stage5(R, a, sets, train, test, loads, C_tr, C_te, E, K, Lr, ck_load, ck_save)
    json.dump(R, open(a.out, "w"), indent=1)
    log("wrote", a.out)


def stage4(R, a, sets, domains, by, train, test, loads, C_tr, C_te, E, K, Lr):
    M_tr = {d: [ecorr.transition_matrix(train[d][:, l], train[d][:, l + 1], E) for l in range(Lr - 1)] for d in sets}

    # ---------------- step 4a: load skew ----------------
    R["load_skew"] = {}
    for d in sets:
        both = np.concatenate([train[d], test[d]])
        sk = [ecorr.load_skew(np.bincount(both[:, l].ravel().astype(np.int64), minlength=E)) for l in range(Lr)]
        R["load_skew"][d] = {k: float(np.mean([s[k] for s in sk])) for k in sk[0]} | \
            {"max_over_mean_per_layer": [round(s["max_over_mean"], 3) for s in sk]}

    # ---------------- step 4b: within-layer co-selection (partners fit on A calib, measured on B holdout) ----
    R["coselect_partner"] = {}
    for A in sets:
        for B in sets:
            vals = [ecorr.partner_lift(C_tr[A][l], C_te[B][l], len(test[B])) for l in range(Lr)]
            R["coselect_partner"][f"{A}->{B}"] = {
                "p_partner_given_i": float(np.mean([v[0] for v in vals])),
                "p_partner_marginal": float(np.mean([v[1] for v in vals])),
                "lift": float(np.mean([v[2] for v in vals])),
                "lift_per_layer": [round(v[2], 3) for v in vals]}
    log("partners done")

    # ---------------- step 4c: layer N -> N+1 predictability ----------------
    R["next_layer_recall"] = {}
    for A in sets:
        for B in sets:
            rec, pop = [], []
            for l in range(Lr - 1):
                load_next = loads[A][l + 1]
                pred = ecorr.predict_next(M_tr[A][l], load_next, test[B][:, l], K)
                rec.append(ecorr.recall_at_k(pred, test[B][:, l + 1].astype(np.int64)))
                pop.append(ecorr.recall_at_k(ecorr.predict_popular(load_next, len(test[B]), K),
                                             test[B][:, l + 1].astype(np.int64)))
            R["next_layer_recall"][f"{A}->{B}"] = {"transition": float(np.mean(rec)), "popularity": float(np.mean(pop)),
                                                   "transition_per_layer": [round(x, 4) for x in rec],
                                                   "popularity_per_layer": [round(x, 4) for x in pop]}
    log("transitions done")
    # phase split for mix (fit on mix calib, measured on holdout prefill vs decode tokens)
    R["next_layer_recall_by_phase"] = {}
    for ph, code in (("prefill", 0), ("decode", 1)):
        te = ecorr.stack([r for d in domains for r in by[(d, "holdout")]], phase=code, cap=a.eval_cap, seed=3)
        rec = [ecorr.recall_at_k(ecorr.predict_next(M_tr["mix"][l], loads["mix"][l + 1], te[:, l], K),
                                 te[:, l + 1].astype(np.int64)) for l in range(Lr - 1)]
        R["next_layer_recall_by_phase"][ph] = {"transition": float(np.mean(rec)), "tokens": int(len(te))}
    log("stage1 done")


def stage5(R, a, sets, train, test, loads, C_tr, C_te, E, K, Lr, ck_load, ck_save):
    # ---------------- step 5: placement simulation ----------------
    R["sim"] = {"random_analytic_distinct": {G: ecorr.expected_distinct_random(E, G, K) for G in a.gpus}}
    sims = {}
    for G in a.gpus:
        assert E % G == 0
        for A in sets:
            got = ck_load(f"sim_G{G}_{A}")
            if got is not None:
                sims.update(got)
                log("sim", G, A, "(checkpoint)")
                continue
            done_before = set(sims)
            plc = {"default": ecorr.default_placement(loads[A], G)}
            swaps = {}
            for s in a.slack:
                ecorr.SWAPS.clear()
                init = plc["default"]
                out = np.empty_like(init)
                for l in range(Lr):
                    gl = np.bincount(init[l], weights=loads[A][l], minlength=G)
                    out[l] = ecorr.correlated_placement_layer(C_tr[A][l], loads[A][l], init[l], G, cap=s * gl.max())
                plc[f"correlated_s{s}"] = out
                swaps[f"correlated_s{s}"] = [int(x) for x in ecorr.SWAPS]
            if A == "mix":
                for sd in range(a.random_seeds):
                    plc[f"random_{sd}"] = ecorr.random_placement(Lr, E, G, sd)
            for name, gp in plc.items():
                for B in sets:
                    r = ecorr.replay(test[B], gp, G)
                    # batch-level imbalance: consecutive holdout token blocks (order within requests kept)
                    bi = []
                    for s0 in range(0, len(test[B]) - a.batch_tokens + 1, a.batch_tokens):
                        blk = test[B][s0:s0 + a.batch_tokens]
                        rr = [np.bincount(gp[l][blk[:, l].astype(np.int64)].ravel(), minlength=G) for l in range(Lr)]
                        bi.append(np.mean([x.max() / x.mean() for x in rr]))
                    sims[f"G{G}|{A}|{name}|{B}"] = {
                        "sends": float(r["sends"].mean()), "distinct": float(r["distinct"].mean()),
                        "sends_nodedup": float(r["sends_nodedup"].mean()), "imbalance": float(r["imbalance"].mean()),
                        "imbalance_batch": float(np.mean(bi)) if bi else None,
                        "sends_per_layer": [round(x, 4) for x in r["sends"]]}
                    if name in swaps:
                        sims[f"G{G}|{A}|{name}|{B}"]["swaps_per_layer"] = swaps[name]
            ck_save(f"sim_G{G}_{A}", {k: v for k, v in sims.items() if k not in done_before})
            log("sim", G, A)
        # in-sample oracle bracket for mix: placement fitted on the holdout tokens themselves
        Lte = np.stack([np.bincount(test["mix"][:, l].ravel().astype(np.int64), minlength=E) for l in range(Lr)]).astype(float)
        init = ecorr.default_placement(Lte, G)
        out = np.empty_like(init)
        for l in range(Lr):
            gl = np.bincount(init[l], weights=Lte[l], minlength=G)
            out[l] = ecorr.correlated_placement_layer(C_te["mix"][l], Lte[l], init[l], G, cap=gl.max())
        for name, gp in (("oracle_default", init), ("oracle_correlated_s1.0", out)):
            r = ecorr.replay(test["mix"], gp, G)
            sims[f"G{G}|holdout-in-sample|{name}|mix"] = {"sends": float(r["sends"].mean()),
                                                          "distinct": float(r["distinct"].mean()),
                                                          "imbalance": float(r["imbalance"].mean())}
    R["sim"]["runs"] = sims


if __name__ == "__main__":
    main()
