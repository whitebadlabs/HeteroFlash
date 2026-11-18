"""Expert co-selection statistics and EP placement simulation (CPU only, numpy).

Terms (as in the study plan):
  load            number of tokens an expert received in a layer
  co-selection    how often two experts are picked by the same token in the same layer
  cross-layer     how often a token that picked expert A in layer N picks expert B in layer N+1
  cross-GPU send  one transfer of a token's data to a GPU that holds one of its picked experts
                  (one send per (token, destination GPU); the token's own GPU is not a send)
"""
import hashlib
import json
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VENDOR = os.path.join(HERE, "..", "vendor", "vllm_6092bb9")
DEFAULT_PY_SHA256 = "9b8643b3632ce73b7d0492d5906d944293d933312c8a3ce9077bfe1dbd788d49"


# ----------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------
def load_reduced(prefix):
    """-> list of dicts {req, domain, split, ids [T, L, K] uint16, phase [T] (0 prefill, 1 decode)}."""
    meta = json.load(open(prefix + ".json"))
    z = np.load(prefix + ".npz")
    out = []
    for r in meta["requests"]:
        parts, ph = [], []
        for p, code in (("prefill", 0), ("decode", 1)):
            k = f"ids_{r['req']}_{p}"
            if k in z.files:
                a = z[k]
                parts.append(a)
                ph.append(np.full(a.shape[0], code, np.int8))
        if not parts:
            continue
        out.append(dict(r, ids=np.concatenate(parts), phase=np.concatenate(ph)))
    return out, meta


def stack(reqs, phase=None, cap=None, seed=0):
    """Concatenate token rows of requests -> [T, L, K]; optional phase filter and random token cap."""
    rows = []
    for r in reqs:
        a = r["ids"] if phase is None else r["ids"][r["phase"] == phase]
        if len(a):
            rows.append(a)
    if not rows:
        return np.zeros((0, 0, 0), np.uint16)
    a = np.concatenate(rows)
    if cap is not None and len(a) > cap:
        idx = np.random.default_rng(seed).choice(len(a), cap, replace=False)
        a = a[np.sort(idx)]
    return a


# ----------------------------------------------------------------------------------------------
# step 4 statistics
# ----------------------------------------------------------------------------------------------
def onehot_counts(ids_l, E):
    """ids_l [T, K] -> X [T, E] uint8 membership."""
    T = ids_l.shape[0]
    X = np.zeros((T, E), np.float32)
    X[np.arange(T)[:, None], ids_l.astype(np.int64)] = 1.0
    return X


def coselect_matrix(ids_l, E):
    """Within-layer co-selection counts C[i, j] (diagonal = load)."""
    X = onehot_counts(ids_l, E)
    return X.T @ X


def transition_matrix(ids_a, ids_b, E):
    """Cross-layer counts M[i, j] = # tokens with i in layer N and j in layer N+1."""
    return onehot_counts(ids_a, E).T @ onehot_counts(ids_b, E)


def recall_at_k(pred, true):
    """pred, true [T, K] -> mean |pred ∩ true| / K."""
    K = true.shape[1]
    hits = (pred[:, :, None] == true[:, None, :]).any(2).sum(1)
    return float(hits.mean() / K)


def predict_next(M_train, load_next_train, ids_a, K, alpha=1.0):
    """Predict layer N+1 top-K from layer N ids with a naive-Bayes-like sum of log P(j | i).

    P(j | i) = (M[i, j] + alpha * p_j) / (load_i + alpha), p_j = popularity prior of j at N+1.
    Score_j = sum_{i in S_N} log P(j | i)  (tokens pick K distinct experts; we take the top K scores).
    """
    E = M_train.shape[0]
    p = (load_next_train + 1.0) / (load_next_train.sum() + E)
    row = M_train.sum(1) / K
    P = (M_train + alpha * p[None, :]) / (row[:, None] + alpha)
    logP = np.log(np.maximum(P, 1e-12))
    S = logP[ids_a.astype(np.int64)].sum(1)            # [T, E]
    return np.argpartition(-S, K, axis=1)[:, :K]


def predict_popular(load_next_train, T, K):
    top = np.argsort(-load_next_train)[:K]
    return np.broadcast_to(top, (T, K))


def partner_lift(C_train, C_test, n_test_tokens, top=7, min_load=20):
    """Pick each expert's `top` partners on train; measure on test P(j | i) and lift = P(j|i) / P(j).

    Returns mean P(partner | i), mean P(partner), mean lift over experts with test load >= min_load.
    """
    E = C_train.shape[0]
    Ct = C_train.copy(); np.fill_diagonal(Ct, -1)
    load_te = np.diag(C_test)
    pj = load_te / max(n_test_tokens, 1)
    pc, pm = [], []
    for i in range(E):
        if load_te[i] < min_load:
            continue
        part = np.argsort(-Ct[i])[:top]
        pc.append(C_test[i, part].mean() / load_te[i])
        pm.append(pj[part].mean())
    pc, pm = np.array(pc), np.array(pm)
    return float(pc.mean()), float(pm.mean()), float((pc / np.maximum(pm, 1e-12)).mean()), int(len(pc))


def load_skew(load):
    """Load skew of one layer: max/mean expert load, coefficient of variation, share of top-10% experts."""
    load = np.asarray(load, np.float64)
    E = len(load)
    s = np.sort(load)[::-1]
    return {"max_over_mean": float(s[0] / load.mean()), "cv": float(load.std() / load.mean()),
            "top10pct_share": float(s[: max(1, E // 10)].sum() / load.sum())}


# ----------------------------------------------------------------------------------------------
# step 5: placements
# ----------------------------------------------------------------------------------------------
def _load_default_policy():
    """Import the vendored, unmodified vLLM DefaultEplbPolicy without torch.

    The algorithm (balanced_packing / replicate_experts / rebalance_experts_hierarchical) is
    pure numpy; torch is only used for the tensor-typed entry point, which we do not call.
    """
    src = open(os.path.join(VENDOR, "default.py"), "rb").read()
    got = hashlib.sha256(src).hexdigest()
    assert got == DEFAULT_PY_SHA256, f"vendored default.py changed: {got}"
    fake_torch = types.ModuleType("torch"); fake_torch.Tensor = object
    pkg = types.ModuleType("vllm_eplb_policy"); pkg.__path__ = []
    abstract = types.ModuleType("vllm_eplb_policy.abstract")

    class AbstractEplbPolicy:  # stand-in for the ABC; only the numpy classmethods are used
        pass

    abstract.AbstractEplbPolicy = AbstractEplbPolicy
    abstract.EplbStats = object
    saved = sys.modules.get("torch")
    sys.modules["torch"] = fake_torch
    sys.modules["vllm_eplb_policy"] = pkg
    sys.modules["vllm_eplb_policy.abstract"] = abstract
    try:
        mod = types.ModuleType("vllm_eplb_policy.default")
        mod.__package__ = "vllm_eplb_policy"
        exec(compile(src, "vendor/vllm_6092bb9/default.py", "exec"), mod.__dict__)
    finally:
        if saved is None:
            del sys.modules["torch"]
        else:
            sys.modules["torch"] = saved
    return mod.DefaultEplbPolicy


DefaultEplbPolicy = _load_default_policy()
SWAPS = []


def default_placement(load, G):
    """DefaultEplbPolicy as vLLM calls it for GLM-5.3-Flash (n_group = 1 -> num_groups = 1).

    With num_groups = 1, vLLM uses the hierarchical function with (groups, nodes) = (1, 1) for any
    node count (eplb_state.py: num_groups % num_nodes must be 0, else global policy = (1, 1)).
    num_replicas = num_logical_experts (no redundant experts). Returns gpu_of[layer, expert].
    """
    load = np.asarray(load, np.float64)
    Lr, E = load.shape
    phy2log = DefaultEplbPolicy.rebalance_experts_hierarchical(load, E, 1, 1, G)
    per = E // G
    gpu_of = np.empty((Lr, E), np.int64)
    for l in range(Lr):
        gpu_of[l, phy2log[l]] = np.arange(E) // per
    return gpu_of


def random_placement(Lr, E, G, seed):
    rng = np.random.default_rng(seed)
    per = E // G
    return np.stack([rng.permutation(E) // per for _ in range(Lr)])


def correlated_placement_layer(C, load, gpu_init, G, cap, max_swaps=5000):
    """Equal-size, load-capped placement that maximizes co-selection inside each GPU.

    Kernighan-Lin style pairwise swaps (as in arXiv 2610.09372): start from `gpu_init`
    (the DefaultEplbPolicy placement), repeatedly apply the expert swap with the largest gain in
    intra-GPU co-selection weight, subject to every GPU load <= cap. Equal experts per GPU always.
    Gain of swapping a (on GPU p) with b (on GPU q): (W[a,q] - W[a,p]) + (W[b,p] - W[b,q]) - 2 W[a,b],
    where W[x, g] = sum of co-selection of x with experts on g (excluding x).
    """
    E = C.shape[0]
    W = C.astype(np.float64).copy(); np.fill_diagonal(W, 0.0)
    g = gpu_init.copy()
    load = np.asarray(load, np.float64)
    gl = np.bincount(g, weights=load, minlength=G)
    n = 0
    for n in range(max_swaps + 1):
        if n == max_swaps:
            break
        H = np.zeros((E, G)); H[np.arange(E), g] = 1.0
        WG = W @ H                                   # [E, G]
        own = WG[np.arange(E), g]                    # W[x, gpu(x)]
        to = WG[:, g]                                # to[a, b] = W[a, gpu(b)]
        gain = (to - own[:, None]) + (to.T - own[None, :]) - 2 * W
        same = g[:, None] == g[None, :]
        # load feasibility: new load of gpu(a) = gl[p] - load[a] + load[b] etc.
        dl = load[None, :] - load[:, None]           # moving b in, a out of p
        newp = gl[g][:, None] + dl
        newq = gl[g][None, :] - dl
        ok = (~same) & (newp <= cap + 1e-9) & (newq <= cap + 1e-9)
        gain = np.where(ok, gain, -np.inf)
        a, b = np.unravel_index(np.argmax(gain), gain.shape)
        if not np.isfinite(gain[a, b]) or gain[a, b] <= 1e-9:
            break
        p, q = g[a], g[b]
        gl[p] += load[b] - load[a]; gl[q] += load[a] - load[b]
        g[a], g[b] = q, p
    SWAPS.append(n)  # swaps applied in this layer (== max_swaps means the search was cut off)
    return g


def correlated_placement(ids, load, G, slack=1.0):
    """Per-layer correlated placement. cap = slack * (max GPU load of the default placement)."""
    Lr = ids.shape[1]
    E = load.shape[1]
    init = default_placement(load, G)
    out = np.empty_like(init)
    for l in range(Lr):
        C = coselect_matrix(ids[:, l], E)
        gl = np.bincount(init[l], weights=load[l], minlength=G)
        out[l] = correlated_placement_layer(C, load[l], init[l], G, cap=slack * gl.max())
    return out


# ----------------------------------------------------------------------------------------------
# step 5: replay
# ----------------------------------------------------------------------------------------------
def replay(ids, gpu_of, G, home_seed=0):
    """Replay routes on a placement.

    ids [T, L, K]; gpu_of [L, E]. Each token gets a home GPU (uniform random, fixed seed; in vLLM
    the home is the token's data-parallel attention rank, and results return home after each MoE layer).
    Returns per-layer arrays:
      distinct   mean distinct destination GPUs per token (= dispatch rows with per-rank dedup)
      sends      mean cross-GPU sends per token (distinct destination GPUs other than the home GPU)
      sends_nodedup mean (token, expert) transfers to non-home GPUs (no per-rank dedup)
      imbalance  max GPU load / mean GPU load (load = token-expert pairs received)
    """
    T, Lr, K = ids.shape
    home = np.random.default_rng(home_seed).integers(0, G, T)
    res = {"distinct": [], "sends": [], "sends_nodedup": [], "imbalance": []}
    for l in range(Lr):
        dst = gpu_of[l][ids[:, l].astype(np.int64)]           # [T, K]
        s = np.sort(dst, axis=1)
        newv = np.ones_like(s, bool); newv[:, 1:] = s[:, 1:] != s[:, :-1]
        distinct = newv.sum(1)
        home_hit = (dst == home[:, None]).any(1)
        res["distinct"].append(distinct.mean())
        res["sends"].append((distinct - home_hit).mean())
        res["sends_nodedup"].append((dst != home[:, None]).sum(1).mean())
        gl = np.bincount(dst.ravel(), minlength=G)
        res["imbalance"].append(gl.max() / gl.mean())
    return {k: np.array(v) for k, v in res.items()}


def expected_distinct_random(E, G, K):
    """Analytic E[distinct GPUs] for a uniformly random equal-size placement (K distinct experts)."""
    from math import comb
    per = E // G
    return G * (1 - comb(E - per, K) / comb(E, K))
