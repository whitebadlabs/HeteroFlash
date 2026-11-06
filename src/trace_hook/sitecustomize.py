"""HeteroFlash router hook v2, loaded via PYTHONPATH into every vLLM process.

Active only when HF_TRACE_DIR is set. Behaviour is controlled LIVE by a JSON
control file ($HF_TRACE_CONTROL, default $HF_TRACE_DIR/../control.json), re-read
at most once per second, so experiments switch without restarting the server:

  {"capture": true,            # save full router logits (fp16) per MoE layer per forward
   "k_active": 8,              # routing surgery: keep only the top-k_active experts (by s + bias)
   "k_ref": "active",          # "active": renormalise over the kept experts (router default)
                               # int (e.g. 8): weights normalised over the original top-k_ref
   "routed_class": null,       # module class whose OUTPUT is the routed-expert sum (needed for int k_ref)
   "diag": false,              # dump the module call sequence of the next forward to diag_calls_<pid>.json
   "fixed_quant": null}        # simulated quantization of fixed (non-routed) weights, applied in place:
                               # {"fmt": "fp8" | "nvfp4", "include": regex, "exclude": regex}; null restores BF16.
                               # fp8: e4m3 with one scale per 128x128 block; nvfp4: E2M1 values, e4m3 scale per
                               # 16 inputs under a per-tensor fp32 scale. Only 2-D bf16/fp16 .weight tensors are
                               # touched (EXL3-packed routed experts are skipped by dtype). Originals are kept on CPU.
A module inventory (name, class, weight dtype/shape) is written once to modules_<pid>.json.

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
            "fixed_quant": None, "_t": 0.0, "_mtime": None}
    _fq = {"applied": None, "orig": {}, "inventory_done": False, "report": None}
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
                for k in ("capture", "k_active", "k_ref", "routed_class", "diag", "fixed_quant"):
                    if k in new:
                        _ctl[k] = new[k]
                _ctl["_mtime"] = mt
        except FileNotFoundError:
            pass
        except Exception:  # noqa: BLE001  (half-written file: keep previous settings)
            pass
        return _ctl

    _E2M1 = None

    def _fake_fp8_block(w, blk=128):
        o, i = w.shape
        po, pi = (-o) % blk, (-i) % blk
        x = torch.nn.functional.pad(w.float(), (0, pi, 0, po)).view((o + po) // blk, blk, (i + pi) // blk, blk)
        sc = x.abs().amax(dim=(1, 3), keepdim=True).clamp_min(1e-12) / 448.0
        q = (x / sc).to(torch.float8_e4m3fn).float() * sc
        return q.view(o + po, i + pi)[:o, :i].to(w.dtype)

    def _fake_nvfp4(w, blk=16):
        global _E2M1
        if _E2M1 is None or _E2M1.device != w.device:
            _E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device=w.device)
        o, i = w.shape
        pi = (-i) % blk
        x = torch.nn.functional.pad(w.float(), (0, pi)).view(o, (i + pi) // blk, blk)
        g = x.abs().amax().clamp_min(1e-12) / (448.0 * 6.0)
        bs = (x.abs().amax(-1, keepdim=True) / (6.0 * g)).to(torch.float8_e4m3fn).float().clamp_min(1e-12) * g
        y = (x / bs).clamp(-6, 6)
        idx = (y.abs().unsqueeze(-1) - _E2M1).abs().argmin(-1)
        q = _E2M1[idx] * y.sign() * bs
        return q.view(o, i + pi)[:, :i].to(w.dtype)

    def _apply_fixed_quant(model, spec):
        import re
        # restore originals first
        for name, (param, cpu) in _fq["orig"].items():
            for r in range(0, cpu.shape[0], 1024):
                param.data[r:r + 1024].copy_(cpu[r:r + 1024].to(param.device))
        if not spec:
            _fq["applied"] = None
            return
        inc = re.compile(spec.get("include", r"(attn|attention|shared_expert|lm_head|mlp\.(gate_up_proj|down_proj))"))
        exc = re.compile(spec.get("exclude", r"(experts\.|\.gate$|indexer|norm|embed|visual|vision|mtp)"))
        fn = {"fp8": _fake_fp8_block, "nvfp4": _fake_nvfp4}[spec["fmt"]]
        n, nbytes, err = 0, 0, 0.0
        with torch.no_grad():
            for mname, mod in model.named_modules():
                w = getattr(mod, "weight", None)
                if (w is None or not torch.is_tensor(w) or w.dim() != 2 or w.dtype not in (torch.bfloat16, torch.float16)
                        or not inc.search(mname) or exc.search(mname)):
                    continue
                if mname not in _fq["orig"]:
                    _fq["orig"][mname] = (w, w.detach().to("cpu", copy=True))
                cpu = _fq["orig"][mname][1]
                rows = 128 if spec["fmt"] == "fp8" else 256       # small GPU temporaries (server holds ~98% of VRAM)
                num = den = 0.0
                for r in range(0, cpu.shape[0], rows):
                    o = cpu[r:r + rows].to(w.device)
                    q = fn(o)
                    num += float(((q.float() - o.float()) ** 2).sum())
                    den += float((o.float() ** 2).sum())
                    w.data[r:r + rows].copy_(q)
                    del o, q
                err += num / max(den, 1e-30)
                n += 1
                nbytes += w.numel() * w.element_size()
        _fq["applied"] = json.dumps(spec, sort_keys=True)
        _fq["report"] = {"spec": spec, "modules": n, "GB_bf16": round(nbytes / 1e9, 3),
                         "mean_rel_sq_err": round(err / max(n, 1), 6), "pid": os.getpid(), "time": time.time()}
        with open(os.path.join(_DIR, f"fixed_quant_{os.getpid()}.json"), "w") as f:
            json.dump(_fq["report"], f)

    def _inventory(model):
        inv = []
        for mname, mod in model.named_modules():
            w = getattr(mod, "weight", None)
            inv.append([mname, type(mod).__name__, str(w.dtype) if torch.is_tensor(w) else None,
                        list(w.shape) if torch.is_tensor(w) else None])
        with open(os.path.join(_DIR, f"modules_{os.getpid()}.json"), "w") as f:
            json.dump(inv, f)

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
            if not _fq["inventory_done"]:
                _fq["inventory_done"] = True
                try:
                    _inventory(mod)
                except Exception:  # noqa: BLE001
                    pass
            want = json.dumps(c["fixed_quant"], sort_keys=True) if c["fixed_quant"] else None
            if want != _fq["applied"]:
                _apply_fixed_quant(mod, c["fixed_quant"])
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
