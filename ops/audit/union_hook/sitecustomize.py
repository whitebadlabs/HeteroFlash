"""Measure distinct routed experts per verify batch (eager mode only).
HF_UNION_STATS=<path>: for every MoE router call with 2..16 tokens, record (tokens, distinct experts in that layer).
Written as JSON {T: [calls, mean_distinct, max_distinct]} every 500 calls and at exit."""
import os
if os.environ.get("HF_UNION_STATS"):
    import atexit, builtins, json
    _p = os.environ["HF_UNION_STATS"]; _st = {"done": False, "n": 0, "agg": {}}

    def _flush():
        try:
            out = {str(t): [c, round(s / c, 2), m] for t, (c, s, m) in sorted(_st["agg"].items())}
            json.dump(out, open(_p, "w"))
        except Exception:  # noqa: BLE001
            pass

    def _install():
        import sys
        fr = sys.modules.get("vllm.model_executor.layers.fused_moe.router.fused_moe_router")
        if fr is None or not hasattr(fr, "FusedMoERouter"):
            return
        R = fr.FusedMoERouter; orig = R.select_experts
        import torch

        def select_experts(self, *a, **k):
            w, ids = orig(self, *a, **k)
            T = int(ids.shape[0])
            if 1 < T <= 16 and not (torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()):
                u = int(torch.unique(ids).numel())
                c, s, m = _st["agg"].get(T, (0, 0, 0)); _st["agg"][T] = (c + 1, s + u, max(m, u))
                _st["n"] += 1
                if _st["n"] % 500 == 0:
                    _flush()
            return w, ids
        R.select_experts = select_experts
        _st["done"] = True
        atexit.register(_flush)

    _imp = builtins.__import__

    def _hook(name, *a, **k):
        mod = _imp(name, *a, **k)
        if not _st["done"] and name.startswith("vllm"):
            try:
                _install()
            except Exception:  # noqa: BLE001
                pass
        return mod
    builtins.__import__ = _hook
