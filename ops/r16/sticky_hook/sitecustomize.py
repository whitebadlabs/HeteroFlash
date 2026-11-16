"""Training-free causal window-sticky routing (Bet 4, step 1). Eager mode only.
For every MoE router call with T > 1 tokens (one sequence per batch), token i's selection score becomes
  biased_sigmoid_score[i, e] + BETA * #{j in 1..W : e in ids0[i-j]}
where ids0 are the unmodified top-k picks of the previous W tokens in the same batch (non-recursive, so the model is
causal and deterministic). Gate weights use the original unbiased sigmoid scores of the new picks, renormalized and
scaled exactly as the stock router does. BETA = 0 reproduces the stock routing.
Env: HF_STICKY_BETA (float), HF_STICKY_W (int, default 3), HF_STICKY_STATS (json path).
Stats: mean distinct experts per 4-token window (stride 4) before/after, and the fraction of changed picks.
HF_CHAIN_HOOK (another sitecustomize.py, e.g. the logits-capture hook) is executed first."""
import os

if os.environ.get("HF_CHAIN_HOOK") and os.path.exists(os.environ["HF_CHAIN_HOOK"]):
    _src = open(os.environ["HF_CHAIN_HOOK"]).read()
    exec(compile(_src, os.environ["HF_CHAIN_HOOK"], "exec"), {"__name__": "hf_chained_hook", "__file__": os.environ["HF_CHAIN_HOOK"]})

if os.environ.get("HF_STICKY_BETA") is not None:
    import atexit, builtins, json
    BETA = float(os.environ["HF_STICKY_BETA"]); W = int(os.environ.get("HF_STICKY_W", "3"))
    _p = os.environ.get("HF_STICKY_STATS"); _st = {"done": False, "n": 0, "win": 0, "u0": 0.0, "u1": 0.0, "picks": 0, "changed": 0}

    def _flush():
        if not _p: return
        try:
            w = max(_st["win"], 1)
            json.dump({"beta": BETA, "W": W, "calls": _st["n"], "windows4": _st["win"], "U4_stock": round(_st["u0"] / w, 3),
                       "U4_sticky": round(_st["u1"] / w, 3), "U4_ratio": round(_st["u1"] / max(_st["u0"], 1), 4),
                       "changed_pick_frac": round(_st["changed"] / max(_st["picks"], 1), 4)}, open(_p, "w"))
        except Exception:  # noqa: BLE001
            pass

    def _u4(ids, torch):
        T, k = ids.shape; n = (T // 4) * 4
        if n == 0: return 0, 0.0
        g = ids[:n].reshape(-1, 4 * k).to(torch.int64)
        s, _ = torch.sort(g, dim=1)
        distinct = 1 + (s[:, 1:] != s[:, :-1]).sum(dim=1)
        return g.shape[0], float(distinct.sum())

    def _install():
        import sys
        fr = sys.modules.get("vllm.model_executor.layers.fused_moe.router.fused_moe_router")
        if fr is None or not hasattr(fr, "FusedMoERouter"):
            return
        import torch
        R = fr.FusedMoERouter; orig = R.select_experts

        def select_experts(self, hidden_states, router_logits, *a, **k):
            w0, ids0 = orig(self, hidden_states, router_logits, *a, **k)
            T = int(ids0.shape[0])
            if T <= 1 or (torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()):
                return w0, ids0
            if getattr(self, "scoring_func", "sigmoid") != "sigmoid":
                return w0, ids0
            topk = ids0.shape[1]
            scores = router_logits.float().sigmoid()
            bias = getattr(self, "e_score_correction_bias", None)
            sel = scores + (bias.data.float().unsqueeze(0) if bias is not None else 0.0)
            if BETA != 0.0:
                M = torch.zeros_like(scores); M.scatter_(1, ids0.long(), 1.0)
                others = torch.zeros_like(scores)
                for j in range(1, W + 1):
                    if j < T: others[j:] += M[:-j]
                sel = sel + BETA * others
            ids = torch.topk(sel, k=topk, dim=-1)[1]
            w = scores.gather(1, ids)
            if getattr(self, "renormalize", True):
                w = w / w.sum(dim=-1, keepdim=True)
            rsf = getattr(self, "routed_scaling_factor", 1.0) or 1.0
            if rsf != 1.0: w = w * rsf
            ids = ids.to(ids0.dtype); w = w.to(w0.dtype)
            nw, d0 = _u4(ids0, torch); _, d1 = _u4(ids, torch)
            _st["win"] += nw; _st["u0"] += d0; _st["u1"] += d1; _st["n"] += 1
            s0, _ = torch.sort(ids0.long(), 1); s1, _ = torch.sort(ids.long(), 1)
            _st["picks"] += T * topk
            _st["changed"] += int((~(s1.unsqueeze(2) == s0.unsqueeze(1)).any(2)).sum())
            if _st["n"] % 200 == 0: _flush()
            return w, ids
        R.select_experts = select_experts
        _st["done"] = True; atexit.register(_flush)

    _imp = builtins.__import__

    def _hook(name, *a, **k):
        mod = _imp(name, *a, **k)
        if not _st["done"] and name.startswith("vllm"):
            try: _install()
            except Exception:  # noqa: BLE001
                pass
        return mod
    builtins.__import__ = _hook
