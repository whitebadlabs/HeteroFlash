#!/usr/bin/env python3
"""Decode-step continuous-batching replay: do causal copy planners keep balance when consecutive steps share requests?
(CPU replay only. Reuses copies.py / causal.py from /tmp/ecorr-causal-20261118 (copied verbatim, sha-checked):
same data split + calib fit (calib mix, all phases, same caps), same primaries, same add_copies planner, same route.)

Stream model: B request slots; every step each slot emits its request's next DECODE token (phase==1 rows only).
When a request ends, the next request of a shuffled holdout queue (all task types interleaved, default_rng(seed))
takes the slot. The queue is recycled with a fresh shuffle when it runs out ('pass' > 0 = repeated request).
Initial slots start at a random offset into their first request (stagger), so slots do not turn over in lockstep.
Every step has exactly B tokens. Home GPU per request instance (u ~ U[0,1) per instance, home = floor(u*G)).

Planners (per step, per layer; state per layer; R*G copies on top of the primaries):
  oracle (a0/c0): counts of THIS step.     lag1 (a1/c1): counts of the previous step (step 0: calib share).
  ema{alpha} (a2/c2): EMA of past step shares, alpha weights the newest step, seeded with calib share.
  lag1_every16 (a3/c3): replanned only at steps 16k from the counts of step 16k-1 (step 0: calib share).
  static (a): vLLM DefaultEplbPolicy with E+R*G slots fitted on calib.  R=0: a (Default) and b (corr) static.
Plan sharing: lag1 plan(t) == oracle plan(t-1) (add_copies is deterministic), so it is reused, not recomputed.

Usage:
  decode.py --reduced PREFIX --train-cap N --eval-cap N --out OUT.json --B 64 --steps 800 [--gpus 8 16] [--layer-stride 4]
  decode.py --parity --reduced ... --sibling-ckpt DIR     (step evaluator on the sibling's mixstream_s0, 2048 blocks)
  decode.py --stability ...                               (consecutive-step load cosine, decode vs sibling batches)
  decode.py --summarize OUT.json ...
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
import copies as cp  # noqa: E402
import causal as cz  # noqa: E402
from copies import ecorr, load_split, add_copies, route, holds_matrix, gpu_of_to_phy, log  # noqa: E402

assert cp.ROUTE == "send"
ALPHAS = (0.5, 0.9)
REPLAN_EVERY = 16
GATE_SENDS, GATE_P95, GATE_SLACK = 0.15, 1.15, 0.05
MODES = ["static", "oracle", "lag1"] + [f"ema{a}" for a in ALPHAS] + ["lag1_every16"]
LAB = {"oracle": "0", "lag1": "1", "ema0.5": "2(.5)", "ema0.9": "2(.9)", "lag1_every16": "3"}


def label(prim, mode, R):
    if mode == "static":
        return ("a" if prim == "def" else "b")
    return ("a" if prim == "def" else "c") + LAB[mode]


# ----------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------
def decode_requests(prefix):
    reqs, _ = ecorr.load_reduced(prefix)
    out = []
    for r in reqs:
        if r["split"] != "holdout":
            continue
        a = r["ids"][r["phase"] == 1]
        if len(a):
            out.append((a, r["domain"], r["req"]))
    return out


def build_stream(dreqs, B, steps, seed):
    """-> ids [S, B, L, K] uint16, inst [S, B] (request-instance id), info dict."""
    rng = np.random.default_rng(seed)
    n = len(dreqs)
    queue, inst_req, inst_pass = [], [], []
    state = {"pass": -1}

    def take():
        if not queue:
            state["pass"] += 1
            queue.extend(rng.permutation(n).tolist())
        r = queue.pop(0)
        inst_req.append(r); inst_pass.append(state["pass"])
        return len(inst_req) - 1
    slot_inst = np.zeros(B, np.int64); slot_pos = np.zeros(B, np.int64)
    for j in range(B):
        slot_inst[j] = take()
        slot_pos[j] = rng.integers(0, len(dreqs[inst_req[slot_inst[j]]][0]))
    L, K = dreqs[0][0].shape[1:]
    ids = np.zeros((steps, B, L, K), np.uint16)
    inst = np.zeros((steps, B), np.int64)
    for t in range(steps):
        for j in range(B):
            a = dreqs[inst_req[slot_inst[j]]][0]
            ids[t, j] = a[slot_pos[j]]
            inst[t, j] = slot_inst[j]
            slot_pos[j] += 1
            if slot_pos[j] == len(a):
                slot_inst[j] = take(); slot_pos[j] = 0
    inst_req = np.array(inst_req); inst_pass = np.array(inst_pass)
    rep = inst_pass[inst] > 0
    dup = np.mean([len(np.unique(inst_req[inst[t]])) < B for t in range(steps)])
    doms = sorted({d for _, d, _ in dreqs})
    dom_of = np.array([doms.index(dreqs[r][1]) for r in inst_req])
    tps = np.array([len(np.unique(dom_of[inst[t]])) for t in range(steps)])
    shares = {d: float(np.mean(dom_of[inst] == i)) for i, d in enumerate(doms)}
    # step-to-step request overlap
    ov = np.mean([np.isin(inst[t], inst[t - 1]).mean() for t in range(1, steps)])
    info = {"B": B, "steps": steps, "seed": seed, "decode_requests": n,
            "decode_tokens_available": int(sum(len(a) for a, _, _ in dreqs)),
            "tokens_used": int(steps * B), "instances": int(len(inst_req)),
            "repeat_token_share": float(rep.mean()), "steps_with_concurrent_duplicate_request": float(dup),
            "passes_started": int(inst_pass.max() + 1), "task_share": shares,
            "task_types_per_step_mean": float(tps.mean()), "step_request_overlap": float(ov)}
    return ids, inst, info


# ----------------------------------------------------------------------------------------------
# step evaluator (shared plans)
# ----------------------------------------------------------------------------------------------
def churn_of(phy, prim_l, prev, E, G):
    cop = holds_matrix(phy, E, G) - holds_matrix(gpu_of_to_phy(prim_l), E, G)
    c = None if prev is None else int(np.maximum(cop - prev, 0).sum())
    return cop, c


def run_cell(batches, homes, G, R, prims, static_phy, calib, E, Wd, modes=None):
    """batches: list of ids [b, L, K]; homes: list of [b] home GPUs.
    prims: {'def': gof [L,E], 'corr': gof [L,E]}; static_phy: {'def': [L,N] or None, 'corr': [L,N] or None}.
    Returns {policy_key: {'sends': [S], 'imb': [S], 'churn': [...]}} with per-step means over layers."""
    S = len(batches)
    Lr = batches[0].shape[1]
    pols = []
    for pk in ("def", "corr"):
        if static_phy.get(pk) is not None:
            pols.append((pk, "static"))
    if R > 0:
        for pk in ("def", "corr"):
            for m in (modes or MODES[1:]):
                pols.append((pk, m))
    res = {f"{pk}|{m}": {"sends": np.zeros((S, Lr)), "imb": np.zeros((S, Lr)), "churn": [], "tok": np.zeros(S)}
           for pk, m in pols}
    calib_share = [calib[l] / calib[l].sum() for l in range(Lr)]
    st = {}
    for pk in ("def", "corr"):
        for l in range(Lr):
            calib_plan = add_copies(prims[pk][l], calib_share[l], Wd[l], G, R, "dynamic") if R else None
            st[(pk, l)] = {"prev_oracle": calib_plan, "held": calib_plan,
                           **{f"ema{a}": calib_share[l].copy() for a in ALPHAS},
                           "prevcop": {}}
    for t in range(S):
        blk = batches[t].astype(np.int64)
        hb = homes[t]
        for l in range(Lr):
            ids_l = blk[:, l]
            cnt = np.bincount(ids_l.ravel(), minlength=E).astype(np.float64)
            for pk in ("def", "corr"):
                s = st[(pk, l)]
                plans = {}
                if R > 0:
                    if t % REPLAN_EVERY == 0 and t > 0:
                        s["held"] = s["prev_oracle"]
                    want = modes or MODES[1:]
                    orc = add_copies(prims[pk][l], cnt, Wd[l], G, R, "dynamic")
                    plans["oracle"] = orc
                    plans["lag1"] = s["prev_oracle"]
                    plans["lag1_every16"] = s["held"]
                    for a in ALPHAS:
                        k = f"ema{a}"
                        if k in want:
                            plans[k] = add_copies(prims[pk][l], s[k], Wd[l], G, R, "dynamic")
                            s[k] = a * cnt / max(cnt.sum(), 1.0) + (1 - a) * s[k]
                    s["prev_oracle"] = orc
                    plans = {m: p for m, p in plans.items() if m in want}
                if static_phy.get(pk) is not None:
                    plans["static"] = static_phy[pk][l]
                for m, phy in plans.items():
                    key = f"{pk}|{m}"
                    if key not in res:
                        continue
                    r = res[key]
                    Hc = holds_matrix(phy, E, G)
                    dst = route(ids_l, hb, Hc > 0, G)
                    srt = np.sort(dst, 1)
                    nv = np.ones_like(srt, bool); nv[:, 1:] = srt[:, 1:] != srt[:, :-1]
                    hh = (dst == hb[:, None]).any(1)
                    r["sends"][t, l] = (nv.sum(1) - hh).sum()
                    g = np.bincount(dst.ravel(), minlength=G)
                    r["imb"][t, l] = g.max() / g.mean()
                    r["tok"][t] = len(ids_l)
                    if m != "static" and (m != "lag1_every16" or t % REPLAN_EVERY == 0):
                        cop, c = churn_of(phy, prims[pk][l], s["prevcop"].get(m), E, G)
                        s["prevcop"][m] = cop
                        if c is not None:
                            r["churn"].append(c)
    return res


def finish(r, R, G, nfull=None):
    """per-step series -> summary; sends per token per layer = sum sends / tokens, mean over layers (sibling def)."""
    T = r["tok"].sum()
    out = {"sends": float((r["sends"].sum(0) / T).mean()),
           "step_imb": [round(float(x), 5) for x in r["imb"].mean(1)],
           "step_sends": [round(float(x), 4) for x in (r["sends"].sum(1) / r["tok"] / r["sends"].shape[1])]}
    si = r["imb"].mean(1) if nfull is None else r["imb"][:nfull].mean(1)
    out["p50"] = float(np.percentile(si, 50)); out["p95"] = float(np.percentile(si, 95)); out["max"] = float(si.max())
    if r["churn"]:
        out["churn_mean"] = float(np.mean(r["churn"])); out["churn_frac"] = float(np.mean(r["churn"])) / max(R * G, 1)
    return out


# ----------------------------------------------------------------------------------------------
# shared setup
# ----------------------------------------------------------------------------------------------
def setup(a, ck):
    E, K, Lr, domains, train, test = load_split(a.reduced, a.train_cap, a.eval_cap)
    load = np.stack([np.bincount(train["mix"][:, l].ravel().astype(np.int64), minlength=E)
                     for l in range(Lr)]).astype(np.float64)
    C = [ecorr.coselect_matrix(train["mix"][:, l], E).astype(np.float64) for l in range(Lr)]
    Wd = []
    for c in C:
        w = c.copy(); np.fill_diagonal(w, 0.0); Wd.append(w)
    return E, K, Lr, domains, train, test, load, C, Wd


def npy(ck, name, fn):
    f = os.path.join(ck, name + ".npy")
    if os.path.exists(f):
        return np.load(f)
    x = fn(); np.save(f + ".tmp.npy", x); os.replace(f + ".tmp.npy", f)
    return x


def placements(ck, G, R, E, Lr, load, C, sib_ck=None):
    gof_def = ecorr.default_placement(load, G)

    def fit_corr():
        out = np.empty_like(gof_def)
        for l in range(Lr):
            gl = np.bincount(gof_def[l], weights=load[l], minlength=G)
            out[l] = ecorr.correlated_placement_layer(C[l], load[l], gof_def[l], G, cap=gl.max())
        return out
    gof_corr = npy(ck, f"G{G}_corr_primaries", fit_corr)
    if sib_ck and os.path.exists(os.path.join(sib_ck, f"G{G}_corr_primaries.npy")):
        assert np.array_equal(np.load(os.path.join(sib_ck, f"G{G}_corr_primaries.npy")), gof_corr), "corr primaries differ"
    if R == 0:
        sp = {"def": np.stack([gpu_of_to_phy(gof_def[l]) for l in range(Lr)]),
              "corr": np.stack([gpu_of_to_phy(gof_corr[l]) for l in range(Lr)])}
    else:
        dphy = npy(ck, f"G{G}_R{R}_default", lambda: ecorr.DefaultEplbPolicy.rebalance_experts_hierarchical(
            load, E + R * G, 1, 1, G).astype(np.int64))
        if sib_ck and os.path.exists(os.path.join(sib_ck, f"G{G}_R{R}_default.npy")):
            assert np.array_equal(np.load(os.path.join(sib_ck, f"G{G}_R{R}_default.npy")), dphy), "default differs"
        sp = {"def": dphy, "corr": None}
    return {"def": gof_def, "corr": gof_corr}, sp


# ----------------------------------------------------------------------------------------------
# parity: step evaluator on the sibling's mixstream_s0 (2048-token blocks, per-token home)
# ----------------------------------------------------------------------------------------------
def parity(a):
    ck = a.out + ".ckpt"; os.makedirs(ck, exist_ok=True)
    E, K, Lr, domains, train, test, load, C, Wd = setup(a, ck)
    segs = cz.request_segments(a.reduced, domains, a.eval_cap, test)
    ids, dom, _ = cz.mixed_stream(segs, domains, 0)
    T = len(ids); G = 8
    home = np.random.default_rng(0).integers(0, G, T)
    batches = [ids[s:s + 2048] for s in range(0, T, 2048)]
    homes = [home[s:s + 2048] for s in range(0, T, 2048)]
    nfull = T // 2048
    out = {}
    for R in (0, 2):
        prims, sp = placements(ck, G, R, E, Lr, load, C, a.sibling_ckpt)
        t0 = time.time()
        res = run_cell(batches, homes, G, R, prims, sp, load, E, Wd, modes=["oracle", "lag1", "ema0.5"])
        log(f"parity G{G} R{R} {time.time() - t0:.0f}s")
        sib = {("def", "static"): "def_static", ("corr", "static"): "corr_static", ("def", "oracle"): "def_oracle",
               ("def", "lag1"): "def_lag1", ("def", "ema0.5"): "def_ema", ("corr", "oracle"): "corr_oracle",
               ("corr", "lag1"): "corr_lag1", ("corr", "ema0.5"): "corr_ema"}
        for key, r in res.items():
            pk, m = key.split("|")
            mine = finish(r, R, G, nfull)
            f = os.path.join(a.sibling_ckpt, f"G{G}_R{R}_{sib[(pk, m)]}.json")
            if not os.path.exists(f):
                continue
            ref = json.load(open(f))["sets"]["mixstream_s0"]
            d_s = abs(mine["sends"] - ref["sends"])
            d_b = float(np.max(np.abs(np.array(mine["step_imb"][:nfull]) - np.array(ref["batch_imb"]))))
            d_c = abs(mine.get("churn_mean", 0) - ref.get("churn_mean", 0))
            ok = d_s < 1e-9 and d_b < 1e-4 and d_c < 1e-9
            out[f"G{G}_R{R}_{sib[(pk, m)]}"] = {"mine_sends": mine["sends"], "sib_sends": ref["sends"], "d_sends": d_s,
                                                "max_d_batch_imb": d_b, "d_churn": d_c, "ok": bool(ok)}
            log(f"  {sib[(pk, m)]:12s} sends {mine['sends']:.6f} vs {ref['sends']:.6f}  max|d batch imb| {d_b:.2e} "
                f"d churn {d_c:.2e} -> {'OK' if ok else 'MISMATCH'}")
    json.dump(out, open(a.out, "w"), indent=1)


# ----------------------------------------------------------------------------------------------
# stability of per-expert load vectors
# ----------------------------------------------------------------------------------------------
def cos_rows(X, Y):
    """X, Y [N, L, E] -> mean over N, L of cosine."""
    num = (X * Y).sum(-1)
    den = np.linalg.norm(X, axis=-1) * np.linalg.norm(Y, axis=-1)
    return float(np.mean(num / np.maximum(den, 1e-12)))


def counts_of(blocks, E):
    """blocks: list of ids [b, L, K] -> [N, L, E] counts."""
    out = np.zeros((len(blocks), blocks[0].shape[1], E))
    for i, b in enumerate(blocks):
        for l in range(b.shape[1]):
            out[i, l] = np.bincount(b[:, l].ravel().astype(np.int64), minlength=E)
    return out


def stab(Cn, calib_share, far=100):
    N = len(Cn)
    r = {"n": int(N), "cos_lag1": cos_rows(Cn[1:], Cn[:-1])}
    f = min(far, N // 3)
    r["far"] = int(f)
    r["cos_far"] = cos_rows(Cn[f:], Cn[:-f]) if f >= 1 else None
    r["cos_calib"] = cos_rows(Cn, np.broadcast_to(calib_share, Cn.shape))
    # EMA predictor (alpha 0.5) vs next, causal
    for a in (0.5, 0.1):
        s = calib_share.copy(); pred = np.zeros_like(Cn)
        for t in range(N):
            pred[t] = s
            s = a * Cn[t] / np.maximum(Cn[t].sum(-1, keepdims=True), 1) + (1 - a) * s
        r[f"cos_ema{a}"] = cos_rows(Cn[1:], pred[1:])
    # i.i.d. sampling floor: two independent multinomial draws of the same size from the block's own distribution
    rng = np.random.default_rng(0)
    idx = rng.choice(N, min(N, 40), replace=False)
    vals = []
    for i in idx:
        for l in range(Cn.shape[1]):
            n = int(Cn[i, l].sum()); p = Cn[i, l] / n
            vals.append(_cos1(rng.multinomial(n, p), rng.multinomial(n, p)))
    r["cos_iid_same_dist"] = float(np.mean(vals))
    return r


def _cos1(x, y):
    return float(x @ y / max(np.linalg.norm(x) * np.linalg.norm(y), 1e-12))


def stability(a):
    ck = a.out + ".ckpt"; os.makedirs(ck, exist_ok=True)
    E, K, Lr, domains, train, test, load, C, Wd = setup(a, ck)
    calib_share = load / load.sum(1, keepdims=True)
    out = {}
    # sibling batches: mixstream_s0 (2048 blocks), and also cut into B-token blocks
    segs = cz.request_segments(a.reduced, domains, a.eval_cap, test)
    # phase share of the sibling's holdout token choice (replicates ecorr.stack index choice)
    reqs, _ = ecorr.load_reduced(a.reduced)
    pre = []
    for d in domains:
        ph = np.concatenate([r["phase"] for r in reqs if r["domain"] == d and r["split"] == "holdout" and len(r["ids"])])
        n = len(ph)
        idx = np.arange(n) if n <= a.eval_cap else np.sort(np.random.default_rng(2).choice(n, a.eval_cap, replace=False))
        pre.append(ph[idx] == 0)
    out["sibling_prefill_token_share"] = float(np.concatenate(pre).mean())
    for s in (0, 1, 2):
        ids, dom, _ = cz.mixed_stream(segs, domains, s)
        T = len(ids)
        for bs in (2048,) + tuple(a.B_list):
            blocks = [ids[i:i + bs] for i in range(0, T - bs + 1, bs)]
            out.setdefault(f"sibling_mixstream_{bs}", []).append(stab(counts_of(blocks, E), calib_share))
        log("stability sibling seed", s)
    dreqs = decode_requests(a.reduced)
    for B in a.B_list:
        steps = a.steps_for.get(B, a.steps)
        for seed in a.seeds:
            ids, inst, info = build_stream(dreqs, B, steps, seed)
            Cn = counts_of(list(ids), E)
            out.setdefault(f"decode_B{B}", []).append({**stab(Cn, calib_share), "stream": info})
            # pooled to 2048 tokens: consecutive windows of 2048/B steps
            w = 2048 // B
            P = np.stack([Cn[i:i + w].sum(0) for i in range(0, len(Cn) - w + 1, w)])
            out.setdefault(f"decode_B{B}_pooled2048", []).append(stab(P, calib_share, far=10))
            log("stability decode B", B, "seed", seed)
    # average over seeds
    summ = {}
    for k, v in out.items():
        if isinstance(v, list):
            summ[k] = {m: float(np.mean([x[m] for x in v])) for m in v[0] if isinstance(v[0][m], (int, float))}
    out["mean_over_seeds"] = summ
    json.dump(out, open(a.out, "w"), indent=1)
    for k, v in summ.items():
        log(k, {m: round(x, 4) for m, x in v.items()})


# ----------------------------------------------------------------------------------------------
# main decode run
# ----------------------------------------------------------------------------------------------
def main_run(a):
    ck = a.out + ".ckpt"; os.makedirs(ck, exist_ok=True)
    E, K, Lr_all, domains, train, test, load, C, Wd = setup(a, ck)
    layers = list(range(0, Lr_all, a.layer_stride))[:a.max_layers or None]
    dreqs = decode_requests(a.reduced)
    meta = {"reduced": os.path.abspath(a.reduced), "train_cap": a.train_cap, "eval_cap": a.eval_cap, "E": E,
            "layers_used": layers, "B": a.B, "steps": a.steps, "seed": a.seed, "gpus": a.gpus,
            "alphas": list(ALPHAS), "replan_every": REPLAN_EVERY}
    mf = os.path.join(ck, "meta.json")
    if os.path.exists(mf):
        old = json.load(open(mf))
        if old != meta:
            sys.exit(f"checkpoint {ck} made with different settings:\n{old}\nvs\n{meta}")
    else:
        json.dump(meta, open(mf, "w"))
    ids, inst, info = build_stream(dreqs, a.B, a.steps, a.seed)
    json.dump(info, open(os.path.join(ck, "stream.json"), "w"), indent=1)
    log("stream", info)
    ids = ids[:, :, layers]
    u = np.random.default_rng(1000 + a.seed).random(int(inst.max()) + 1)
    load_l = load[layers]; Wd_l = [Wd[l] for l in layers]; C_l = [C[l] for l in layers]
    for G in a.gpus:
        Rs = a.R if a.R else [0, 2, int(np.floor(0.25 * E / G + 0.5))]
        homes = list(np.floor(u[inst] * G).astype(np.int64))
        for R in Rs:
            f = os.path.join(ck, f"G{G}_R{R}.json")
            if os.path.exists(f):
                log(f"G{G} R{R} (checkpoint)"); continue
            t0 = time.time()
            prims_all, sp_all = placements(ck, G, R, E, Lr_all, load, C)
            prims = {k: v[layers] for k, v in prims_all.items()}
            sp = {k: (None if v is None else v[layers]) for k, v in sp_all.items()}
            res = run_cell(list(ids), homes, G, R, prims, sp, load_l, E, Wd_l)
            info_c = {"G": G, "R": R, "seconds": round(time.time() - t0, 1),
                      "policies": {k: finish(v, R, G) for k, v in res.items()}}
            json.dump(info_c, open(f + ".tmp", "w")); os.replace(f + ".tmp", f)
            log(f"G{G} R{R} {info_c['seconds']:.0f}s " + "  ".join(
                f"{k} {v['sends']:.3f}/{v['p95']:.3f}" for k, v in info_c["policies"].items()))
    log("done")


# ----------------------------------------------------------------------------------------------
# summary
# ----------------------------------------------------------------------------------------------
def summarize(paths):
    Lns = []
    f3 = lambda v: "" if v is None else f"{v:.3f}"  # noqa: E731
    for p in paths:
        ck = p + ".ckpt"
        meta = json.load(open(os.path.join(ck, "meta.json")))
        sinfo = json.load(open(os.path.join(ck, "stream.json")))
        Lns.append(f"\n## {os.path.basename(p)}  (B={meta['B']}, steps={meta['steps']}, layers used "
                   f"{len(meta['layers_used'])} (stride {meta['layers_used'][1] - meta['layers_used'][0] if len(meta['layers_used']) > 1 else 1}))\n")
        Lns.append(f"Stream: {sinfo['decode_requests']} holdout requests with decode tokens, "
                   f"{sinfo['decode_tokens_available']} decode tokens available; used {sinfo['tokens_used']}; "
                   f"repeat-token share {sinfo['repeat_token_share']:.2f}; steps with a request in 2 slots "
                   f"{sinfo['steps_with_concurrent_duplicate_request']:.2f}; task types/step "
                   f"{sinfo['task_types_per_step_mean']:.2f}; request overlap step-to-step {sinfo['step_request_overlap']:.3f}.\n")
        Lns.append("| G | R | policy | sends/tok/layer | vs a | vs same-planner Default | step imb p50/p95 (max) | "
                   "copy churn/replan (frac of R*G) | gate |")
        Lns.append("|" + "---|" * 9)
        for f in sorted(glob.glob(os.path.join(ck, "G*_R*.json")), key=lambda x: tuple(
                int(s) for s in os.path.basename(x)[1:-5].split("_R"))):
            c = json.load(open(f)); G, R = c["G"], c["R"]; P = c["policies"]
            order = ["def|static", "corr|static"] + [f"{pk}|{m}" for pk in ("def", "corr") for m in MODES[1:]]
            for k in order:
                if k not in P:
                    continue
                x = P[k]; pk, m = k.split("|")
                ra = P.get("def|static")
                same = P.get(f"def|{m}") if pk == "corr" and m != "static" else None
                va = f"{100 * (x['sends'] / ra['sends'] - 1):+.1f}%" if ra else ""
                vs = f"{100 * (x['sends'] / same['sends'] - 1):+.1f}%" if same else ""
                g = ""
                if same:
                    cut = 1 - x["sends"] / same["sends"]
                    ok = cut >= GATE_SENDS and (x["p95"] <= GATE_P95 or x["p95"] <= same["p95"] + GATE_SLACK)
                    g = "GO" if ok else "no-go"
                ch = f"{x['churn_mean']:.1f} ({x['churn_frac']:.2f})" if "churn_mean" in x else ""
                Lns.append(f"| {G} | {R} | {label(pk, m, R)} {m} | {x['sends']:.3f} | {va} | {vs} | "
                           f"{f3(x['p50'])}/{f3(x['p95'])} ({f3(x['max'])}) | {ch} | {g} |")
        Lns.append("\nGate: corr variant >= 15% fewer sends than the SAME-planner Default AND step p95 <= 1.15 or "
                   "<= that Default's p95 + 0.05.\n")
    return "\n".join(Lns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reduced"); ap.add_argument("--out")
    ap.add_argument("--train-cap", type=int, default=12000); ap.add_argument("--eval-cap", type=int, default=30000)
    ap.add_argument("--B", type=int, default=64); ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpus", type=int, nargs="*", default=[8, 16])
    ap.add_argument("--R", type=int, nargs="*", default=None)
    ap.add_argument("--layer-stride", type=int, default=1); ap.add_argument("--max-layers", type=int, default=0)
    ap.add_argument("--parity", action="store_true"); ap.add_argument("--sibling-ckpt")
    ap.add_argument("--stability", action="store_true")
    ap.add_argument("--B-list", type=int, nargs="*", default=[64, 256])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--steps-for", default="{}", help='JSON {"B": steps} for --stability')
    ap.add_argument("--summarize", nargs="*")
    a = ap.parse_args()
    a.steps_for = {int(k): v for k, v in json.loads(a.steps_for).items()}
    if a.summarize:
        print(summarize(a.summarize)); return
    if a.parity:
        parity(a); return
    if a.stability:
        stability(a); return
    main_run(a)


if __name__ == "__main__":
    main()
