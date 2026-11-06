"""HeteroFlash router hook v2, loaded via PYTHONPATH into every vLLM process.

Active only when HF_TRACE_DIR is set. Behaviour is controlled LIVE by a JSON
control file ($HF_TRACE_CONTROL, default $HF_TRACE_DIR/../control.json), re-read
at most once per second, so experiments switch without restarting the server:

  {"capture": true,            # save full router logits (fp16) per MoE layer per forward
   "k_active": 8,              # routing surgery: keep only the top-k_active experts (by s + bias)
   "k_ref": "active",          # "active": renormalise over the kept experts (router default)
                               # int (e.g. 8): weights normalised over the original top-k_ref
   "routed_class": null,       # module class whose OUTPUT is the routed-expert sum (needed for int k_ref)
   "diag": false}              # dump the module call sequence of the next forward to diag_calls_<pid>.json

Surgery mechanics (exact, no kernel changes): the gate's logits for experts
outside the kept set are set to -1e4, so sigmoid(score) ~ 0. The router still
selects 8 ids, but the extra ones get weight 0, and the kept weights are
renormalised over the kept set (k_ref = "active"). For an int k_ref, the
routed-expert output is multiplied per token by S_kept / S_ref, where S is the
sum of sigmoid scores over the respective sets. That reproduces
w_i = scaling * s_i / S_ref. Compute is not reduced (masked experts still run
with weight 0), so this mode measures quality, not speed.

Requires eager execution (no CUDA graphs, no torch.compile). Only tensor-parallel
rank 0 writes capture files; all ranks apply surgery identically.
"""
import os

_DIR = os.environ.get("HF_TRACE_DIR")
if _DIR:
    import json
    import threading
    import time

    import numpy as np
    import torch

    E = int(os.environ.get("HF_TRACE_EXPERTS", "288"))
    TOPK = int(os.environ.get("HF_TRACE_TOPK", "8"))
    FLUSH = int(os.environ.get("HF_TRACE_FLUSH_ROWS", "200000"))
    CTRL = os.environ.get("HF_TRACE_CONTROL", os.path.join(os.path.dirname(_DIR.rstrip("/")), "control.json"))
    os.makedirs(_DIR, exist_ok=True)
    _st = {"fwd": -1, "layer": 0, "cls": None, "rows": 0, "chunk": 0, "buf": [], "diag": None,
           "first_fwd_counts": None, "ratio": {}, "calls": None}
    _ctl = {"capture": True, "k_active": TOPK, "k_ref": "active", "routed_class": None, "diag": False,
            "_t": 0.0, "_mtime": None}
    _lock = threading.Lock()
    _TOP = ("ForCausalLM", "ForConditionalGeneration")

    def _control():
        now = time.time()
        if now - _ctl["_t"] < 1.0:
            return _ctl
        _ctl["_t"] = now
        try:
            mt = os.path.getmtime(CTRL)
            if mt != _ctl["_mtime"]:
                with open(CTRL) as f:
                    new = json.load(f)
                for k in ("capture", "k_active", "k_ref", "routed_class", "diag"):
                    if k in new:
                        _ctl[k] = new[k]
                _ctl["_mtime"] = mt
        except FileNotFoundError:
            pass
        except Exception:  # noqa: BLE001  (half-written file: keep previous settings)
            pass
        return _ctl

    def _rank0():
        try:
            import torch.distributed as dist
            return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
        except Exception:  # noqa: BLE001
            return True

    def _flush():
        if not _st["buf"]:
            return
        b = _st["buf"]
        path = os.path.join(_DIR, f"r0_{os.getpid()}_{_st['chunk']:05d}.npz")
        np.savez(path + ".tmp.npz", fwd=np.array([x[0] for x in b], np.int64), layer=np.array([x[1] for x in b], np.int16),
                 ntok=np.array([x[2] for x in b], np.int32), t=np.array([x[3] for x in b], np.float64),
                 kact=np.array([x[5] for x in b], np.int16), logits=np.concatenate([x[4] for x in b], axis=0))
        os.replace(path + ".tmp.npz", path)
        _st["chunk"] += 1
        _st["buf"], _st["rows"] = [], 0

    def _find(tensors, last):
        for i, x in enumerate(tensors):
            if torch.is_tensor(x) and x.dim() == 2 and x.shape[-1] == last and x.is_floating_point():
                return i, x
        return None, None

    def _flat(o):
        return list(o) if isinstance(o, (tuple, list)) else [o]

    def _pre(mod, args):
        name = type(mod).__name__
        if name.endswith(_TOP):
            c = _control()
            with _lock:
                if _st["fwd"] >= 0 and _st["first_fwd_counts"] is None and _st["cls"]:
                    _st["first_fwd_counts"] = _st["layer"]
                if _st["calls"] is not None and _st["calls"]["done"] is False and _st["calls"]["started"]:
                    _st["calls"]["done"] = True
                    with open(os.path.join(_DIR, f"diag_calls_{os.getpid()}.json"), "w") as f:
                        json.dump(_st["calls"]["seq"][:4000], f)
                if c["diag"] and _st["calls"] is None:
                    _st["calls"] = {"seq": [], "started": True, "done": False}
                _st["fwd"] += 1
                _st["layer"] = 0
                _st["ratio"] = {}

    def _record_call(mod, out):
        cs = _st["calls"]
        if cs is not None and cs["started"] and not cs["done"] and len(cs["seq"]) < 4000:
            shapes = [list(x.shape) for x in _flat(out) if torch.is_tensor(x)]
            cs["seq"].append([type(mod).__name__, shapes])

    def _post(mod, args, out):
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return None
        _record_call(mod, out)
        cls = type(mod).__name__
        c = _ctl
        # routed-output rescale for integer k_ref
        if c["routed_class"] and cls == c["routed_class"] and isinstance(c["k_ref"], int):
            li = _st["layer"] - 1
            r = _st["ratio"].get(li)
            if r is not None:
                outs = _flat(out)
                idx, y = _find(outs, outs[0].shape[-1] if torch.is_tensor(outs[0]) else -1)
                if y is not None and y.shape[0] == r.shape[0]:
                    y2 = y * r.to(y.dtype).unsqueeze(1)
                    if isinstance(out, (tuple, list)):
                        outs[idx] = y2
                        return type(out)(outs)
                    return y2
            return None
        if _st["cls"] is not None and cls != _st["cls"]:
            return None
        outs = _flat(out)
        idx, x = _find(outs, E)
        where = "output"
        if x is None:
            _, x = _find(_flat(args), E)
            where = "input"
        if x is None:
            return None
        ka = int(c["k_active"])
        new_out = None
        with _lock:
            if _st["cls"] is None:
                _st["cls"] = cls
                _st["diag"] = {"matched_class": cls, "where": where, "pid": os.getpid(),
                               "has_bias_attr": hasattr(mod, "e_score_correction_bias"), "time": time.time()}
            li = _st["layer"]
            _st["layer"] += 1
            if c["capture"] and _rank0():
                _st["buf"].append((_st["fwd"], li, int(x.shape[0]), time.time(),
                                   x.detach().to(torch.float16).cpu().numpy(), ka))
                _st["rows"] += int(x.shape[0])
            if _st["diag"] is not None and _st["fwd"] >= 3 and "written" not in _st["diag"]:
                _st["diag"]["layers_in_first_forward"] = _st["first_fwd_counts"]
                _st["diag"]["written"] = True
                with open(os.path.join(_DIR, f"diag_{os.getpid()}.json"), "w") as f:
                    json.dump(_st["diag"], f)
            if _st["rows"] >= FLUSH:
                _flush()
        if ka < TOPK and where == "output":
            bias = getattr(mod, "e_score_correction_bias", None)
            with torch.no_grad():
                lf = x.float()
                s = lf.sigmoid()
                choice = s + (bias.float() if bias is not None else 0.0)
                keep = choice.topk(ka, dim=-1).indices
                mask = torch.ones_like(lf, dtype=torch.bool)
                mask.scatter_(1, keep, False)
                xm = x.masked_fill(mask, -1e4)
                if isinstance(c["k_ref"], int):
                    ref = choice.topk(int(c["k_ref"]), dim=-1).indices
                    s_keep = s.gather(1, keep).sum(1)
                    s_ref = s.gather(1, ref).sum(1)
                    _st["ratio"][li] = (s_keep / s_ref.clamp_min(1e-20))
            outs[idx] = xm
            new_out = type(out)(outs) if isinstance(out, (tuple, list)) else xm
        return new_out

    torch.nn.modules.module.register_module_forward_pre_hook(_pre)
    torch.nn.modules.module.register_module_forward_hook(_post)

    def _periodic():
        while True:
            time.sleep(30)
            with _lock:
                _flush()
    threading.Thread(target=_periodic, daemon=True).start()

    import atexit
    atexit.register(_flush)
