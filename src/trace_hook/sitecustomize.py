"""HeteroFlash router-logit capture, loaded via PYTHONPATH into every vLLM process.

Active only when HF_TRACE_DIR is set. Registers GLOBAL torch module hooks:
- a forward pre-hook on the top-level model (*ForCausalLM / *ForConditionalGeneration)
  that starts a new "forward" record and resets the per-forward MoE-layer counter;
- a forward hook that finds the router logits: the first 2-D float tensor with
  last dim == HF_TRACE_EXPERTS (288) in the module's OUTPUT (preferred) or INPUT.
  The first module class that matches is locked in, so the same logits are not
  recorded twice (e.g. by both the gate and the fused-MoE layer).
Full logits are saved as fp16 so selection (with the checkpoint's correction
bias), top-k and normalization variants can be recomputed exactly offline.

Requires eager execution (no CUDA graphs, no torch.compile), otherwise hooks do
not fire. Only tensor-parallel rank 0 writes. Writes:
  $HF_TRACE_DIR/r0_<pid>_<chunk>.npz   arrays: fwd, layer, ntok, t, logits(N_rows x E fp16)
  $HF_TRACE_DIR/diag_<pid>.json         first-forward diagnostics (matched class, counts)
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
    FLUSH = int(os.environ.get("HF_TRACE_FLUSH_ROWS", "200000"))
    os.makedirs(_DIR, exist_ok=True)
    _st = {"fwd": -1, "layer": 0, "cls": None, "rows": 0, "chunk": 0, "buf": [], "diag": None,
           "seen_cls": {}, "first_fwd_counts": None}
    _lock = threading.Lock()
    _TOP = ("ForCausalLM", "ForConditionalGeneration")

    def _rank0():
        try:
            import torch.distributed as dist
            return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
        except Exception:  # noqa: BLE001
            return True

    def _flush(final=False):
        if not _st["buf"]:
            return
        fwd = np.array([b[0] for b in _st["buf"]], dtype=np.int64)
        layer = np.array([b[1] for b in _st["buf"]], dtype=np.int16)
        ntok = np.array([b[2] for b in _st["buf"]], dtype=np.int32)
        t = np.array([b[3] for b in _st["buf"]], dtype=np.float64)
        logits = np.concatenate([b[4] for b in _st["buf"]], axis=0)
        path = os.path.join(_DIR, f"r0_{os.getpid()}_{_st['chunk']:05d}.npz")
        np.savez(path + ".tmp.npz", fwd=fwd, layer=layer, ntok=ntok, t=t, logits=logits)
        os.replace(path + ".tmp.npz", path)
        _st["chunk"] += 1
        _st["buf"], _st["rows"] = [], 0

    def _find(tensors):
        for x in tensors:
            if torch.is_tensor(x) and x.dim() == 2 and x.shape[-1] == E and x.is_floating_point():
                return x
        return None

    def _flat(o):
        if isinstance(o, (tuple, list)):
            return list(o)
        return [o]

    def _pre(mod, args):
        name = type(mod).__name__
        if name.endswith(_TOP):
            with _lock:
                if _st["fwd"] >= 0 and _st["first_fwd_counts"] is None and _st["cls"]:
                    _st["first_fwd_counts"] = _st["layer"]
                _st["fwd"] += 1
                _st["layer"] = 0

    def _post(mod, args, out):
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return
        cls = type(mod).__name__
        if _st["cls"] is not None and cls != _st["cls"]:
            return
        x = _find(_flat(out))
        where = "output"
        if x is None:
            x = _find(_flat(args))
            where = "input"
        if x is None:
            return
        with _lock:
            if _st["cls"] is None:
                _st["cls"] = cls
                _st["diag"] = {"matched_class": cls, "where": where, "pid": os.getpid(),
                               "has_bias_attr": hasattr(mod, "e_score_correction_bias"), "time": time.time()}
            if not _rank0():
                return
            li = _st["layer"]
            _st["layer"] += 1
            _st["buf"].append((_st["fwd"], li, int(x.shape[0]), time.time(), x.detach().to(torch.float16).cpu().numpy()))
            _st["rows"] += int(x.shape[0])
            if _st["diag"] is not None and _st["fwd"] >= 3 and "written" not in _st["diag"]:
                _st["diag"]["layers_in_first_forward"] = _st["first_fwd_counts"]
                _st["diag"]["written"] = True
                with open(os.path.join(_DIR, f"diag_{os.getpid()}.json"), "w") as f:
                    json.dump(_st["diag"], f)
            if _st["rows"] >= FLUSH:
                _flush()

    torch.nn.modules.module.register_module_forward_pre_hook(_pre)
    torch.nn.modules.module.register_module_forward_hook(_post)

    def _periodic():
        while True:
            time.sleep(30)
            with _lock:
                _flush()
    threading.Thread(target=_periodic, daemon=True).start()

    import atexit
    atexit.register(lambda: _flush(True))
