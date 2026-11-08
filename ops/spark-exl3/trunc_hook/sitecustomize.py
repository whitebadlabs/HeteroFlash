"""Test-only: when HF_TRUNCATE_LAYERS is set, drop checkpoint weights for decoder layers >= num_hidden_layers so a
truncated GLM-5.3-Flash (fewer layers) loads from the full checkpoint. Never used for quality measurements."""
import os
import re

if os.environ.get("HF_TRUNCATE_LAYERS"):
    _pat = re.compile(r"(?:^|\.)layers\.(\d+)\.")

    _state = {"done": False}

    def _install():
        import sys as _s
        m = _s.modules.get("vllm.models.glm5next.nvidia.model")     # never import here: only patch once loaded
        if m is None:
            return False
        cls = getattr(m, "Glm5NextModel", None)
        if cls is None:                      # module still initialising (circular import); try again later
            return False
        if getattr(cls, "_hf_trunc", False):
            return True
        orig = m.Glm5NextModel.load_weights

        def load_weights(self, weights):
            n = int(self.config.num_hidden_layers)

            def keep(it):
                for name, w in it:
                    mm = _pat.search(name)
                    if mm and int(mm.group(1)) >= n:
                        continue
                    yield name, w
            return orig(self, keep(weights))
        m.Glm5NextModel.load_weights = load_weights
        m.Glm5NextModel._hf_trunc = True
        _state["done"] = True
        return True

    import builtins
    _orig_import = builtins.__import__

    def _hook(name, *a, **k):
        mod = _orig_import(name, *a, **k)
        if not _state["done"] and name.startswith("vllm"):
            try:
                _install()
            except Exception:  # noqa: BLE001
                pass
        return mod
    builtins.__import__ = _hook

# Test-only b12x compatibility shim (HF_B12X_COMPAT=1): the Spark vLLM passes recover_speculative_state to b12x KDA
# Caps; the control image's b12x 1.3.0 Caps lacks it. Drop it when False (no speculative decoding in these tests),
# fail loudly if it is ever True.
if os.environ.get("HF_B12X_COMPAT"):
    import builtins as _b
    import inspect as _inspect
    _imp = _b.__import__
    _done = set()

    def _shim_caps(mod):
        c = getattr(mod, "Caps", None)
        if c is None or id(c) in _done:
            return
        _done.add(id(c))
        try:
            params = _inspect.signature(c.__init__).parameters
        except (TypeError, ValueError):
            return
        if "recover_speculative_state" in params:
            return
        orig_init = c.__init__

        def __init__(self, *a, **k):
            if k.pop("recover_speculative_state", False):
                raise RuntimeError("b12x 1.3.0 Caps cannot recover speculative state")
            orig_init(self, *a, **k)
        c.__init__ = __init__

    def _hook2(name, *a, **k):
        m = _imp(name, *a, **k)
        if name.startswith("b12x"):                 # only b12x imports; never scan sys.modules (O(n^2) startup)
            try:
                import sys as _sys
                mm = _sys.modules.get(name)
                if mm is not None and "Caps" in getattr(mm, "__dict__", {}):
                    _shim_caps(mm)
                for sub in (k.get("fromlist") or (a[2] if len(a) > 2 else None) or ()):
                    sm = _sys.modules.get(f"{name}.{sub}")
                    if sm is not None and "Caps" in getattr(sm, "__dict__", {}):
                        _shim_caps(sm)
            except Exception:  # noqa: BLE001
                pass
        return m
    _b.__import__ = _hook2

# Test-only remote-expert mask (HF_REMOTE_MASK=neg|zero, HF_REMOTE_MOD=k): experts with id % k == 0 are "remote".
#  neg:  their topk_ids slots become -1 (b12x should skip them: no compute, weight unused)
#  zero: their topk_weights become 0 (computed, contributes nothing) -- the reference for neg.
# Identical outputs for neg and zero (and different from no mask) show the -1 skip path is exact.
if os.environ.get("HF_REMOTE_MASK"):
    import builtins as _b3
    _imp3 = _b3.__import__
    _m3 = {"done": False}
    _mode = os.environ["HF_REMOTE_MASK"]
    _k = int(os.environ.get("HF_REMOTE_MOD", "4"))

    def _install3():
        import sys as _s
        m = _s.modules.get("vllm.model_executor.layers.fused_moe.router.fused_moe_router")
        c = getattr(m, "FusedMoERouter", None) if m else None
        if c is None:
            return
        orig = c.select_experts

        def select_experts(self, *a, **k):
            w, ids = orig(self, *a, **k)
            remote = (ids % _k) == 0
            if _mode == "neg":
                ids = ids.masked_fill(remote, -1)
            elif _mode == "zero":
                w = w.masked_fill(remote, 0.0)
            return w, ids
        c.select_experts = select_experts
        _m3["done"] = True

    def _hook3(name, *a, **k):
        mod = _imp3(name, *a, **k)
        if not _m3["done"] and name.startswith("vllm"):
            try:
                _install3()
            except Exception:  # noqa: BLE001
                pass
        return mod
    _b3.__import__ = _hook3

# Test-only expert-map path (HF_ROUTE_MAP=k): plan b12x MoE with route_num_experts=288 and bind every MoE call with
# route_expert_map[e] = e, or -1 for e % k == 0 (all weights stay resident). Exact if it matches HF_REMOTE_MASK=zero.
if os.environ.get("HF_ROUTE_MAP"):
    import builtins as _b4
    _imp4 = _b4.__import__
    _m4 = {"done": False, "maps": {}}
    _k4 = int(os.environ["HF_ROUTE_MAP"])

    def _install4():
        import sys as _s
        fm = _s.modules.get("b12x.moe.fused_moe")
        if fm is None or not hasattr(fm, "bind") or not hasattr(fm, "ExecutionCapacity"):
            return
        import torch
        cap_cls, orig_bind = fm.ExecutionCapacity, fm.bind

        def ExecutionCapacity(*a, **k):
            if not k.get("route_num_experts"):
                k["route_num_experts"] = 288
            return cap_cls(*a, **k)

        def bind(plan, **k):
            dev = k["topk_ids"].device
            m = _m4["maps"].get(dev)
            if m is None:
                m = torch.arange(288, dtype=torch.int32, device=dev)
                m = torch.where(m % _k4 == 0, torch.full_like(m, -1), m)
                _m4["maps"][dev] = m
            k["route_expert_map"] = m
            return orig_bind(plan, **k)
        fm.ExecutionCapacity, fm.bind = ExecutionCapacity, bind
        _m4["done"] = True

    def _hook4(name, *a, **k):
        mod = _imp4(name, *a, **k)
        if not _m4["done"] and (name.startswith("b12x") or name.startswith("vllm")):
            try:
                _install4()
            except Exception:  # noqa: BLE001
                pass
        return mod
    _b4.__import__ = _hook4
