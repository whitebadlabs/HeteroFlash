"""Training-free routed-expert dropping (2026-11-17, speed-quality frontier). Eager mode only (GPU-only diagnostic build).
After the stock router picks its top-8 (sigmoid + correction bias, renormalised, scaled), an expert is dropped when
  rank >= K   or   w_i / sum_j w_j < TAU
Dropped experts get weight 0 (identical to not computing them). RENORM=0 keeps the original top-8 normalisation
("ref8": the dropped mass is not redistributed, the better choice per G2), RENORM=1 rescales the kept weights to the
original sum ("active"). Env: HF_DROP_K (8), HF_DROP_TAU (0), HF_DROP_RENORM (0), HF_DROP_STATS (json path).
Stats: mean kept experts per token, distinct experts per 4-token window (stride 4) before/after (bytes proxy).
HF_CHAIN_HOOK (another sitecustomize.py, e.g. the logits-capture hook) is executed first."""
import os

if os.environ.get("HF_CHAIN_HOOK") and os.path.exists(os.environ["HF_CHAIN_HOOK"]):
    _src = open(os.environ["HF_CHAIN_HOOK"]).read()
    exec(compile(_src, os.environ["HF_CHAIN_HOOK"], "exec"), {"__name__": "hf_chained_hook", "__file__": os.environ["HF_CHAIN_HOOK"]})

if os.environ.get("HF_DROP_K") is not None or os.environ.get("HF_DROP_TAU") is not None:
    import atexit, builtins, json
    K = int(os.environ.get("HF_DROP_K", "8")); TAU = float(os.environ.get("HF_DROP_TAU", "0")); RENORM = os.environ.get("HF_DROP_RENORM", "0") == "1"
    _p = os.environ.get("HF_DROP_STATS"); _st = {"done": False, "n": 0, "win": 0, "u0": 0.0, "u1": 0.0, "picks": 0, "kept": 0, "wdrop": 0.0, "tok": 0}

    def _flush():
        if not _p: return
        try:
            w = max(_st["win"], 1)
            json.dump({"K": K, "TAU": TAU, "renorm": RENORM, "calls": _st["n"], "windows4": _st["win"],
                       "U4_stock": round(_st["u0"] / w, 3), "U4_drop": round(_st["u1"] / w, 3), "U4_ratio": round(_st["u1"] / max(_st["u0"], 1), 4),
                       "kept_per_token": round(_st["kept"] / max(_st["tok"], 1), 3),
                       "dropped_weight_frac": round(_st["wdrop"] / max(_st["tok"], 1), 4)}, open(_p, "w"))
        except Exception:  # noqa: BLE001
            pass

    def _u4(ids, keep, torch):
        T, k = ids.shape; n = (T // 4) * 4
        if n == 0: return 0, 0.0
        g = torch.where(keep[:n], ids[:n].long(), torch.full_like(ids[:n].long(), -1)).reshape(-1, 4 * k)
        s, _ = torch.sort(g, dim=1)
        distinct = (s[:, :1] >= 0).long().squeeze(1) + ((s[:, 1:] != s[:, :-1]) & (s[:, 1:] >= 0)).sum(dim=1)
        return g.shape[0], float(distinct.sum())

    def drop(w, ids, torch):
        """keep mask + new weights; graph-safe (no host sync)."""
        wf = w.float()
        tot = wf.sum(dim=-1, keepdim=True)
        rank = wf.argsort(dim=-1, descending=True).argsort(dim=-1)
        keep = (rank < K) & (wf >= TAU * tot)
        nw = torch.where(keep, wf, torch.zeros_like(wf))
        if RENORM:
            nw = nw * (tot / nw.sum(dim=-1, keepdim=True).clamp_min(1e-20))
        return keep, nw

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
            if T == 0 or (torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()):
                return w0, ids0
            keep, nw = drop(w0, ids0, torch)
            ones = torch.ones_like(keep)
            nwin, d0 = _u4(ids0, ones, torch); _, d1 = _u4(ids0, keep, torch)
            _st["win"] += nwin; _st["u0"] += d0; _st["u1"] += d1; _st["n"] += 1; _st["tok"] += T
            _st["kept"] += int(keep.sum())
            wf = w0.float(); _st["wdrop"] += float(((wf - torch.where(keep, wf, 0 * wf)).sum(-1) / wf.sum(-1)).sum())
            if _st["n"] % 200 == 0: _flush()
            return nw.to(w0.dtype), ids0
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
