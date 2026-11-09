"""HeteroFlash runtime-v2 prototype: routed experts computed on the workstation CPU (runtime-v1 expert_server).

Active when HF_REMOTE_EXPERTS=<plan.json> and HF_REMOTE_ADDR=<ip:port> are set; eager mode only (--enforce-eager).
plan.json maps absolute decoder layer -> list of expert ids whose routes go to the CPU server.

Per MoE layer:
  1. FusedMoERouter.select_experts (wrapped): the routes (token, expert, weight) whose expert is remote are queued to a
     sender thread as expert_server requests (<= 16 tokens, <= 128 pairs each); their GPU weight is set to 0, which is
     an exact mask on b12x (docs/RUNTIME_V2_VLLM_HOOKS.md).
  2. The GPU runs the resident routed experts and the shared expert meanwhile (kernels are launched asynchronously).
  3. MoERunner.forward (wrapped): after the GPU launch, receive the CPU partial sums (gate-weighted, routed scale
     already folded into the weights) and add them to the layer output per token.
Weights on the GPU therefore stay resident (IQ2 copy); the CPU holds a higher-precision (int4) copy of the planned
experts: "remote experts as precision upgrade".

If HF_CHAIN_HOOK points to another sitecustomize.py (e.g. the logits-capture trace hook), it is executed first.
Stats (requests, pairs, wait time) are appended to $HF_REMOTE_STATS every 200 MoE calls.
"""
import os

if os.environ.get("HF_CHAIN_HOOK") and os.path.exists(os.environ["HF_CHAIN_HOOK"]):
    _src = open(os.environ["HF_CHAIN_HOOK"]).read()
    exec(compile(_src, os.environ["HF_CHAIN_HOOK"], "exec"), {"__name__": "hf_chained_hook", "__file__": os.environ["HF_CHAIN_HOOK"]})

if os.environ.get("HF_REMOTE_EXPERTS"):
    import builtins
    import json
    import queue
    import re
    import socket
    import struct
    import threading
    import time

    _plan_raw = {int(k): v for k, v in json.load(open(os.environ["HF_REMOTE_EXPERTS"])).items()}
    _host, _port = os.environ["HF_REMOTE_ADDR"].rsplit(":", 1)
    _FIRST_MOE = 3
    _H = 4096
    _MAXTOK, _MAXPAIR = 16, 128
    _FLAGS = int(os.environ.get("HF_REMOTE_FLAGS", "3"))      # bit0 busy-poll, bit1 heartbeat
    _st = {"done": False, "sock": None, "q": None, "masks": {}, "calls": 0, "req": 0, "pairs": 0, "wait": 0.0,
           "routes": 0, "server_us": 0}
    _tl = threading.local()
    _layer_re = re.compile(r"layers\.(\d+)\.")

    def _sock():
        if _st["sock"] is None:
            s = socket.create_connection((_host, int(_port)))
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            _st["sock"] = s
            _st["q"] = queue.Queue()

            def sender():
                while True:
                    b = _st["q"].get()
                    s.sendall(b)
            threading.Thread(target=sender, daemon=True).start()
        return _st["sock"]

    def _recvn(s, n):
        buf = bytearray(n); mv = memoryview(buf); o = 0
        while o < n:
            k = s.recv_into(mv[o:], n - o)
            if k == 0:
                raise ConnectionError("expert_server closed")
            o += k
        return buf

    def _recv_reply(s):
        # heartbeat mode: skip leading 0x00 bytes before the 'HFY1' magic
        while True:
            b = _recvn(s, 1)
            if b[0] != 0:
                break
        rest = _recvn(s, 3)
        magic = bytes(b) + bytes(rest)
        if magic != b"HFY1":
            raise ConnectionError(f"bad reply magic {magic!r}")
        ntok, status, us = struct.unpack("<HHI", _recvn(s, 8))
        if status:
            raise RuntimeError(f"expert_server status {status} (missing expert)")
        return ntok, us, _recvn(s, ntok * _H * 2)

    def _mask_for(layer, device):
        key = (layer, device)
        m = _st["masks"].get(key)
        if m is None:
            import torch
            m = torch.zeros(288, dtype=torch.bool)
            m[torch.tensor(_plan_raw[layer], dtype=torch.long)] = True
            m = m.to(device)
            _st["masks"][key] = m
        return m

    _MAILBOX = os.environ.get("HF_REMOTE_MODE", "") == "mailbox"
    _NSUB, _MAXT = 8, 16
    _mb = {"ext": None}

    def _ext():
        if _mb["ext"] is None:
            from torch.utils.cpp_extension import load
            src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_mailbox.cu")
            ext = load(name="hf_mailbox", sources=[src], extra_cuda_cflags=["-O3"], verbose=False)
            ext.init(64 * _NSUB, _host, int(_port), _FLAGS)
            _mb["ext"] = ext
        return _mb["ext"]

    def _mb_select(L, x, w, ids):
        """Pack the first _NSUB 16-token chunks into this layer's mailbox sub-slots; the GPU keeps the rest."""
        ext = _ext()
        rem = _mask_for(L, ids.device)
        T = int(x.shape[0])
        chunks = [(t0, min(T, t0 + _MAXT)) for t0 in range(0, T, _MAXT)]
        xb = x if x.dtype == __import__("torch").bfloat16 else x.to(__import__("torch").bfloat16)
        wg = []
        for c, (t0, t1) in enumerate(chunks):
            if c < _NSUB:
                wg.append(ext.pack((L - _FIRST_MOE) * _NSUB + c, L - _FIRST_MOE, xb[t0:t1].contiguous(), ids[t0:t1].contiguous(), w[t0:t1].contiguous(), rem))
            else:
                import torch
                wg.append(w[t0:t1].masked_fill(rem[ids[t0:t1].long()], 0.0))
        _tl.mb = (L, xb, ids, w, rem, chunks)
        import torch
        return (wg[0] if len(wg) == 1 else torch.cat(wg, 0)), ids

    def _mb_finish(mb, out):
        """Wait for each chunk's CPU partial sum (pipelined over _NSUB sub-slots) and add it to the layer output."""
        ext = _ext()
        L, xb, ids, w, rem, chunks = mb
        base = (L - _FIRST_MOE) * _NSUB
        for c, (t0, t1) in enumerate(chunks):
            ext.wait_add(base + c % _NSUB, out[t0:t1])
            n = c + _NSUB
            if n < len(chunks):
                a0, a1 = chunks[n]
                ext.pack(base + n % _NSUB, L - _FIRST_MOE, xb[a0:a1].contiguous(), ids[a0:a1].contiguous(), w[a0:a1].contiguous(), rem)
        return out

    def _install():
        import sys
        fr = sys.modules.get("vllm.model_executor.layers.fused_moe.router.fused_moe_router")
        mr = sys.modules.get("vllm.model_executor.layers.fused_moe.runner.moe_runner")
        if fr is None or mr is None or not hasattr(fr, "FusedMoERouter") or not hasattr(mr, "MoERunner"):
            return
        import numpy as np
        import torch
        R, M = fr.FusedMoERouter, mr.MoERunner
        orig_sel, orig_fwd = R.select_experts, M.forward

        def select_experts(self, hidden_states, router_logits, *a, **k):
            w, ids = orig_sel(self, hidden_states, router_logits, *a, **k)
            L = getattr(_tl, "layer", None)
            if L is None or L not in _plan_raw or hidden_states.shape[0] == 0:
                return w, ids
            if _MAILBOX:
                return _mb_select(L, hidden_states, w, ids)
            remote = _mask_for(L, ids.device)[ids.long()]
            nz = remote.nonzero()
            if nz.numel() == 0:
                _tl.pending = None
                return w, ids
            toks = nz[:, 0]
            pe = ids[nz[:, 0], nz[:, 1]].to(torch.int64)
            pw = w[nz[:, 0], nz[:, 1]].float()
            utok, inv = torch.unique(toks, return_inverse=True)
            x = hidden_states.index_select(0, utok).to(torch.bfloat16).contiguous().view(torch.int16).cpu().numpy()
            inv_c, pe_c, pw_c, utok_c = inv.cpu().numpy(), pe.cpu().numpy(), pw.cpu().numpy(), utok
            s = _sock()
            groups = []
            nt = int(utok.shape[0])
            for t0 in range(0, nt, _MAXTOK):
                t1 = min(nt, t0 + _MAXTOK)
                sel = np.nonzero((inv_c >= t0) & (inv_c < t1))[0]
                wire = np.zeros(len(sel), dtype=[("tok", "<u2"), ("e", "<u2"), ("w", "<f4")])
                wire["tok"], wire["e"], wire["w"] = inv_c[sel] - t0, pe_c[sel], pw_c[sel]
                hdr = struct.pack("<IHHHH", 0x31584648, L - _FIRST_MOE, t1 - t0, len(sel), _FLAGS)
                _st["q"].put(hdr + wire.tobytes() + x[t0:t1].tobytes())
                groups.append(t1 - t0)
                _st["req"] += 1
            _st["pairs"] += int(nz.shape[0]); _st["routes"] += int(ids.numel())
            _tl.pending = (s, groups, utok_c)
            return w.masked_fill(remote, 0.0), ids

        def forward(self, hidden_states, router_logits, *a, **k):
            m = _layer_re.search(str(getattr(self, "layer_name", "")))
            _tl.layer = int(m.group(1)) if m else None
            _tl.pending = None
            _tl.mb = None
            out = orig_fwd(self, hidden_states, router_logits, *a, **k)
            if _MAILBOX:
                mb = getattr(_tl, "mb", None)
                _tl.layer, _tl.mb = None, None
                if mb is not None:
                    out = _mb_finish(mb, out)
                _st["calls"] += 1
                return out
            p = getattr(_tl, "pending", None)
            _tl.layer, _tl.pending = None, None
            if p is not None:
                s, groups, utok = p
                t0 = time.perf_counter()
                parts = []
                for g in groups:
                    ntok, us, yb = _recv_reply(s)
                    assert ntok == g
                    parts.append(np.frombuffer(yb, dtype=np.int16).reshape(ntok, _H))
                    _st["server_us"] += us
                y = torch.from_numpy(np.concatenate(parts)).view(torch.bfloat16).to(out.device, non_blocking=False)
                out = out.index_add(0, utok, y.to(out.dtype))
                _st["wait"] += time.perf_counter() - t0
            _st["calls"] += 1
            if _st["calls"] % 200 == 0 and os.environ.get("HF_REMOTE_STATS"):
                with open(os.environ["HF_REMOTE_STATS"], "a") as f:
                    f.write(json.dumps({k: _st[k] for k in ("calls", "req", "pairs", "routes", "wait", "server_us")}) + "\n")
            return out

        R.select_experts, M.forward = select_experts, forward
        _st["done"] = True

    # HF_REMOTE_EXPERT_MAP=1: skip the CPU's experts on the GPU entirely (b12x route_expert_map with -1 for remote ids,
    # verified exact within accumulation noise, docs/RUNTIME_V2_VLLM_HOOKS.md) instead of computing them with weight 0.
    _EMAP = os.environ.get("HF_REMOTE_EXPERT_MAP") == "1"
    _em = {"done": not _EMAP, "maps": {}}

    def _install_emap():
        import sys
        fm = sys.modules.get("b12x.moe.fused_moe")
        if fm is None or not hasattr(fm, "bind") or not hasattr(fm, "ExecutionCapacity"):
            return
        import torch
        cap_cls, orig_bind = fm.ExecutionCapacity, fm.bind

        def ExecutionCapacity(*a, **k):
            if not k.get("route_num_experts"):
                k["route_num_experts"] = 288
            return cap_cls(*a, **k)

        def bind(plan, **k):
            L = getattr(_tl, "layer", None)
            dev = k["topk_ids"].device
            key = (L if L in _plan_raw else None, dev)
            m = _em["maps"].get(key)
            if m is None:
                g = torch.arange(288, dtype=torch.int32)
                if key[0] is not None:
                    rem = torch.zeros(288, dtype=torch.bool); rem[torch.tensor(_plan_raw[key[0]], dtype=torch.long)] = True
                    g = torch.where(rem, torch.full_like(g, -1), g)
                m = g.to(dev)
                _em["maps"][key] = m
            k["route_expert_map"] = m
            return orig_bind(plan, **k)
        fm.ExecutionCapacity, fm.bind = ExecutionCapacity, bind
        _em["done"] = True

    _orig_import = builtins.__import__

    def _hook(name, *a, **k):
        mod = _orig_import(name, *a, **k)
        if not _st["done"] and name.startswith("vllm"):
            try:
                _install()
            except Exception:  # noqa: BLE001
                pass
        if not _em["done"] and (name.startswith("b12x") or name.startswith("vllm")):
            try:
                _install_emap()
            except Exception:  # noqa: BLE001
                pass
        return mod
    builtins.__import__ = _hook
