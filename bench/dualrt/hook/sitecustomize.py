"""HeteroFlash Bet 1 dual-memory runtime hook (2026-11-17), stage M1: routed experts split by expert, both memories at once.

Replaces ops/route3/remote_hook (config B) for the MoE layers 3-44:
  * The workstation CPU still holds every routed expert as Q4_K + imatrix (expert_server_gguf, same GGUF rows).
  * The Spark GPU frees its unused IQ2_XXS copies of layers 3-44 after loading and instead keeps a resident hot set of
    the SAME Q4_K bytes (read from the Spark's copy of the GGUF), so a route computed on either side uses identical
    weights (config-B quality by construction; only activation rounding differs: fp32 on the GPU, q8_K on the CPU).
  * Per layer and step, an in-graph assign kernel takes the window's expert union U, sends every non-resident expert
    and a balancing remainder to the CPU, and keeps cap = round(A*U + B) resident experts on the GPU (per-step
    rebalancing). The GPU computes its experts while the CPU computes the rest; the CPU partial (fp32 by default) is
    added after the shared expert, as in config B. Everything stays inside the CUDA graph (mailbox, no Python breaks).
  * Windows larger than HF_SPLIT_MAXT tokens (prefill chunks) send every route to the CPU (config B behaviour).
  * The MTP layer (45) is unchanged (GPU IQ2 experts, as in config B).

Env:
  HF_REMOTE_ADDR ip:port   workstation expert_server_gguf          HF_REMOTE_FLAGS  wire flags (3 = busy-poll + heartbeat)
  HF_SPLIT_MODE 0|1|2      0 every route to the CPU (config B on this mailbox), 1 balanced, 2 every resident expert on GPU
  HF_SPLIT_A / HF_SPLIT_B  balance rule (cap = round(A*U + B))     HF_SPLIT_MAXT    largest window that is split (16)
  HF_SPLIT_F32 0|1         bf16 (default, Carmen 2026-11-17: optimise tok/s) or fp32 partial-sum reply
  HF_SPLIT_HOT json        {layer: [experts, hottest first]}       HF_SPLIT_NRES    resident experts per layer
  HF_SPLIT_GGUF / HF_SPLIT_IDX  Q4_K GGUF + expert index (runtime/gguf_expert_index.py format)
  HF_SPLIT_COUNT path      count decode-window routes per layer (hot-set calibration), dumped every 2000 MoE calls
  HF_SPLIT_STATS path      assign + exchange counters, dumped every 2000 MoE calls
If HF_CHAIN_HOOK points to another sitecustomize.py (e.g. the logits-capture trace hook), it is executed first.
"""
import os

if os.environ.get("HF_CHAIN_HOOK") and os.path.exists(os.environ["HF_CHAIN_HOOK"]):
    _src = open(os.environ["HF_CHAIN_HOOK"]).read()
    exec(compile(_src, os.environ["HF_CHAIN_HOOK"], "exec"), {"__name__": "hf_chained_hook", "__file__": os.environ["HF_CHAIN_HOOK"]})

import builtins as _bi
if os.environ.get("HF_SPLIT", "0") == "1" and not getattr(_bi, "_hf_drt_loaded", False):
    _bi._hf_drt_loaded = True      # guard: some launch paths execute sitecustomize twice in one process
    import builtins
    import json
    import re
    import struct
    import threading

    _host, _port = os.environ["HF_REMOTE_ADDR"].rsplit(":", 1)
    _FIRST, _LAST = 3, 44
    _H, _I, _NE = 4096, 2048, 288
    _NSUB, _MAXT = 8, 16
    _MODE = int(os.environ.get("HF_SPLIT_MODE", "1"))
    _A = float(os.environ.get("HF_SPLIT_A", "0.5"))
    _B = float(os.environ.get("HF_SPLIT_B", "-1.0"))
    _SPLIT_MAXT = int(os.environ.get("HF_SPLIT_MAXT", "16"))
    _F32 = os.environ.get("HF_SPLIT_F32", "0") == "1"   # Carmen 11-17: bf16 partial exchange is the default (faster); fp32 is the option
    _FLAGS = int(os.environ.get("HF_REMOTE_FLAGS", "3")) | (4 if _F32 else 0)
    _NRES = int(os.environ.get("HF_SPLIT_NRES", "0"))
    _HOT = os.environ.get("HF_SPLIT_HOT", "")
    _GGUF = os.environ.get("HF_SPLIT_GGUF", "/gguf/GLM-5.3-Flash-Q8attn-G0_q4kimexp.gguf")
    _IDX = os.environ.get("HF_SPLIT_IDX", "/drt/q4k_experts.idx")
    _COUNT = os.environ.get("HF_SPLIT_COUNT", "")
    _STATS = os.environ.get("HF_SPLIT_STATS", "")
    _LMH_VG = int(os.environ.get("HF_SPLIT_LMHEAD_VG", "0"))   # >0: lm_head rows [0, VG) on the GPU, the rest on the CPU (lid 0)
    _CKPT = os.environ.get("HF_SPLIT_CKPT", "/m")
    _tl = threading.local()
    _layer_re = re.compile(r"layers\.(\d+)\.")
    _S = {"ext": None, "buf": None, "pool": None, "calls": 0, "done": False, "pw_done": False, "fm_done": False}

    def _log(msg):
        print(f"[hf-drt] {msg}", flush=True)

    def _ext():
        if _S["ext"] is None:
            from torch.utils.cpp_extension import load
            src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_drt.cu")
            ext = load(name="hf_drt", sources=[src], extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=False)
            ext.mb_init((_LAST - _FIRST + 1) * _NSUB, _host, int(_port), _FLAGS)
            _S["ext"] = ext
            _log(f"mailbox up: mode={_MODE} A={_A} B={_B} maxT={_SPLIT_MAXT} f32={_F32} flags={_FLAGS}")
        return _S["ext"]

    def _bufs(device):
        if _S["buf"] is None:
            import torch
            nl = _LAST - _FIRST + 1
            b = {
                "remote": torch.ones(_NE, dtype=torch.bool, device=device),
                "remote_all": torch.ones(_NE, dtype=torch.bool, device=device),
                "glist": torch.zeros(64, dtype=torch.int32, device=device),
                "gcount": torch.zeros(1, dtype=torch.int32, device=device),
                "gw": torch.zeros(64 * _MAXT, dtype=torch.float32, device=device),
                "hbuf": torch.zeros(64 * 4 * _I, dtype=torch.float32, device=device),
                "y": torch.zeros(_MAXT * _H, dtype=torch.float32, device=device),
                "stats": torch.zeros(nl, 4, dtype=torch.int64, device=device),
                "counts": torch.zeros(nl, _NE, dtype=torch.int64, device=device),
            }
            if _S["pool"] is None:   # resident pool not loaded yet: dummy bases, every slot -1 (all routes to the CPU)
                dummy = torch.zeros(64, dtype=torch.uint8, device=device)
                _S["pool"] = {"dummy": dummy, "ptrs": torch.tensor([dummy.data_ptr()] * 3, dtype=torch.int64, device=device),
                              "slot_of": torch.full((nl, _NE), -1, dtype=torch.int32, device=device)}
            _S["buf"] = b
            if _COUNT or _STATS:
                threading.Thread(target=_dump_loop, daemon=True).start()
        return _S["buf"]

    # ------------------------------------------------------------------ resident Q4_K hot set
    def _read_idx():
        recs = {}
        raw = open(_IDX, "rb").read()
        for i in range(len(raw) // 48):
            L, p, ty, rows, cols, ne, rb, off, nb = struct.unpack("<6i3Q", raw[i * 48:(i + 1) * 48])
            recs[(L, p)] = (ty, rows, cols, ne, rb, off)
        return recs

    def _load_pool(device):
        import torch
        nl = _LAST - _FIRST + 1
        slot_of = torch.full((nl, _NE), -1, dtype=torch.int32)
        if _MODE == 0 or _NRES <= 0:
            _log("no resident experts (mode 0 or NRES 0)")
            return
        hot = {int(k): v for k, v in json.load(open(_HOT)).items()} if _HOT else {}
        sel = {}
        for L in range(_FIRST, _LAST + 1):
            order = list(hot.get(L, [])) + [e for e in range(_NE) if e not in set(hot.get(L, []))]
            sel[L] = order[:_NRES]
        ns = nl * _NRES
        recs = _read_idx()
        gate_rb, down_rb = recs[(_FIRST, 0)][4], recs[(_FIRST, 2)][4]
        G = torch.empty((ns, _I * gate_rb), dtype=torch.uint8, device=device)
        U = torch.empty((ns, _I * gate_rb), dtype=torch.uint8, device=device)
        D = torch.empty((ns, _H * down_rb), dtype=torch.uint8, device=device)
        import time
        t0 = time.time()
        fd = os.open(_GGUF, os.O_RDONLY)
        pin = torch.empty(_I * gate_rb, dtype=torch.uint8).pin_memory()
        s = 0
        for L in range(_FIRST, _LAST + 1):
            for p, dst in ((0, G), (1, U), (2, D)):
                ty, rows, cols, ne, rb, off = recs[(L, p)]
                assert ty == 12, f"layer {L} proj {p} is ggml type {ty}, not Q4_K"
                per = rows * rb
                for j, e in enumerate(sel[L]):
                    o = off + e * per
                    n = os.preadv(fd, [memoryview(pin.numpy())[:per]], o)
                    assert n == per
                    dst[s + j].copy_(pin[:per], non_blocking=False)
                    try:
                        os.posix_fadvise(fd, o, per, os.POSIX_FADV_DONTNEED)
                    except Exception:  # noqa: BLE001
                        pass
            for j, e in enumerate(sel[L]):
                slot_of[L - _FIRST, e] = s + j
            s += _NRES
        os.close(fd)
        torch.cuda.synchronize()
        P = _S["pool"]          # in place: captured CUDA graphs read these two tensors at run time
        P.update({"G": G, "U": U, "D": D, "sel": sel})
        P["ptrs"].copy_(torch.tensor([G.data_ptr(), U.data_ptr(), D.data_ptr()], dtype=torch.int64))
        P["slot_of"].copy_(slot_of)
        torch.cuda.synchronize()
        _log(f"resident Q4_K experts: {ns} ({_NRES}/layer), {(G.numel() + U.numel() + D.numel()) / 1e9:.1f} GB in {time.time() - t0:.0f} s")

    def _free_iq2(model):
        """Drop the unused IQ2 routed-expert tensors of layers 3-44 wherever the b12x method keeps them (module params,
        buffers, plain attributes, and attributes of objects one or two levels down, e.g. the quant method)."""
        import gc
        import torch
        freed, seen, where, keep = 0, set(), {}, []

        def big(v):
            return isinstance(v, torch.Tensor) and v.is_cuda and v.numel() * v.element_size() > (1 << 20)

        def strip(obj, path, depth):
            nonlocal freed
            if id(obj) in seen or depth > 2:
                return
            seen.add(id(obj))
            if isinstance(obj, torch.nn.Module):
                for pn, p in list(obj.named_parameters(recurse=False)):
                    if big(p):
                        freed += p.numel() * p.element_size(); where[path + "." + pn] = tuple(p.shape); keep.append(p)
                for bn, bb in list(obj.named_buffers(recurse=False)):
                    if big(bb):
                        freed += bb.numel() * bb.element_size(); where[path + "." + bn] = tuple(bb.shape); keep.append(bb)
            d = getattr(obj, "__dict__", None)
            if not isinstance(d, dict):
                return
            for k, v in list(d.items()):
                if k in ("_parameters", "_buffers", "_modules"):
                    continue
                if big(v) and not isinstance(v, torch.nn.Parameter):
                    freed += v.numel() * v.element_size(); where[path + "." + k] = tuple(v.shape); keep.append(v)
                elif isinstance(v, (list, tuple)) and v and all(big(t) for t in v):
                    for t in v:
                        freed += t.numel() * t.element_size(); keep.append(t)
                    where[path + "." + k] = f"{len(v)} tensors"
                elif hasattr(v, "__dict__") and not isinstance(v, (torch.Tensor, type)) and not isinstance(v, torch.nn.Module):
                    strip(v, path + "." + k, depth + 1)
        n = 0
        for name, mod in model.named_modules():
            if type(mod).__name__ != "RoutedExperts":
                continue
            m = _layer_re.search(name)
            if not m or not (_FIRST <= int(m.group(1)) <= _LAST):
                continue
            strip(mod, name, 0)
            n += 1
        _S["iq2_keep"] = keep
        ex = list(where.items())[:4]
        _log(f"found {freed / 1e9:.1f} GB of IQ2 routed-expert tensors in {n} modules (layers {_FIRST}-{_LAST}); released after startup (lazy resident load); e.g. {ex}")

    def _release_iq2():
        """Free the IQ2 storages in place (resize to 0): every alias (b12x plans etc.) loses them, which is safe because
        the routed path of layers 3-44 never runs b12x again (forward_modular is replaced)."""
        import gc
        import torch
        torch.cuda.synchronize()
        n = 0
        for t in _S.pop("iq2_keep", []):
            st = t.untyped_storage()
            if st.size() > 0:
                n += st.size(); st.resize_(0)
        gc.collect(); torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        _log(f"released {n / 1e9:.1f} GB of IQ2 storage; cuda free {free / 1e9:.1f} / {total / 1e9:.1f} GB")

    # ------------------------------------------------------------------ lm_head vocabulary split (stage M3a)
    def _load_lmhead(device):
        if _LMH_VG <= 0:
            return
        import torch
        from safetensors import safe_open
        wm = json.load(open(f"{_CKPT}/model.safetensors.index.json"))["weight_map"]
        with safe_open(f"{_CKPT}/{wm['lm_head.weight']}", framework="pt") as f:
            w = f.get_slice("lm_head.weight"); V = w.get_shape()[0]
            W = f.get_tensor("lm_head.weight").view(torch.uint8)[:_LMH_VG].contiguous().to(device)
            S = f.get_tensor("lm_head.weight_scale").view(torch.uint8)[:_LMH_VG].contiguous().to(device)
        ext = _ext()
        si = ext.dense_register(0, 4096, V - _LMH_VG)
        _S["lmh"] = {"W": W, "S": S, "V": V, "si": si}
        _log(f"lm_head split: GPU rows 0-{_LMH_VG - 1}, workstation rows {_LMH_VG}-{V - 1} (dense slot {si})")

    _LOAD_REQ = os.environ.get("HF_SPLIT_LOAD_REQ", "/out/drt_load.req")

    def _maybe_lazy_load():
        _S["ah_calls"] = _S.get("ah_calls", 0) + 1
        if _S.get("pool_loaded") or _S["ah_calls"] % 16 or not os.path.exists(_LOAD_REQ):
            return
        import torch
        _S["pool_loaded"] = True
        try:
            _release_iq2()
            _load_pool(torch.device("cuda", torch.cuda.current_device()))
            free, total = torch.cuda.mem_get_info()
            msg = f"resident pool live; cuda free {free / 1e9:.1f} / {total / 1e9:.1f} GB"
        except Exception as ex:  # noqa: BLE001
            msg = f"resident load FAILED: {ex!r}"
        _log(msg)
        with open(_LOAD_REQ.replace(".req", ".done"), "w") as f:
            f.write(msg + "\n")
        os.remove(_LOAD_REQ)

    def _apply_head_split(orig, self, lm_head, hidden_states, embedding_bias):
        import torch
        _maybe_lazy_load()
        L = _S.get("lmh")
        T = int(hidden_states.shape[0])
        if L is None or embedding_bias is not None or T == 0 or T > 16:
            return orig(self, lm_head, hidden_states, embedding_bias)
        if "proto" not in L:      # learn the output layout once (first call is an uncaptured warmup/profile run)
            o = orig(self, lm_head, hidden_states, embedding_bias)
            L["proto"] = (o.dtype, int(o.shape[-1]))
            _log(f"lm_head output dtype {o.dtype}, width {o.shape[-1]} (checkpoint vocab {L['V']})")
            return o
        dt, Vp = L["proto"]
        ext = _ext()
        x = hidden_states.to(torch.bfloat16).contiguous()
        ext.dense_pack(L["si"], x)
        out = torch.zeros(T, Vp, dtype=torch.bfloat16, device=x.device) if Vp > L["V"] else torch.empty(T, Vp, dtype=torch.bfloat16, device=x.device)
        ext.mxfp8_gemv(L["W"], L["S"], x, out)
        ext.dense_wait_copy(L["si"], T, out, _LMH_VG)
        return out if dt == torch.bfloat16 else out.to(dt)

    # ------------------------------------------------------------------ dumps (a thread: CUDA graph replays run no Python)
    import contextlib
    _nullctx = contextlib.nullcontext

    def _dump_loop():
        import time
        import torch
        st = torch.cuda.Stream()
        n = 0
        while True:
            time.sleep(1.0)
            n += 1
            req = (_STATS or _COUNT) + ".req"
            if os.path.exists(req) or n % 30 == 0:
                try:
                    _dump(st)
                    if os.path.exists(req):
                        os.remove(req)
                except Exception as ex:  # noqa: BLE001
                    _log(f"dump failed: {ex!r}")
    def _dump(stream=None):
        import torch
        b = _S["buf"]
        if b is None:
            return
        with torch.cuda.stream(stream) if stream is not None else _nullctx():
            counts = b["counts"].to("cpu") if _COUNT else None
            st = b["stats"].to("cpu").tolist()
        if _COUNT:
            json.dump({str(L): counts[L - _FIRST].tolist() for L in range(_FIRST, _LAST + 1)}, open(_COUNT, "w"))
        if _STATS:
            ms = _ext().mb_stats()
            json.dump({"mode": _MODE, "A": _A, "B": _B, "f32": _F32, "nres": _NRES, "moe_calls": _S["calls"],
                       "per_layer_calls_U_cap_R": {str(L): st[L - _FIRST] for L in range(_FIRST, _LAST + 1)},
                       "mailbox_req_pairs_serverus_relayus": ms}, open(_STATS, "w"))

    # ------------------------------------------------------------------ the MoE path
    def _moe_split(L, x, topk_weights, topk_ids):
        import torch
        ext = _ext()
        b = _bufs(x.device)
        P = _S["pool"]
        T = int(x.shape[0])
        xb = (x if x.dtype == torch.bfloat16 else x.to(torch.bfloat16)).contiguous()
        w = topk_weights.float().contiguous()
        ids = topk_ids.contiguous()
        li = L - _FIRST
        if T <= _SPLIT_MAXT:
            if _COUNT:
                b["counts"][li].index_add_(0, ids.reshape(-1).long(), torch.ones_like(ids.reshape(-1), dtype=torch.int64))
            ext.assign(ids, w, P["slot_of"][li], _A, _B, _MODE, b["remote"], b["glist"], b["gcount"], b["gw"], b["stats"][li])
            ext.mb_pack(li * _NSUB, li, xb, ids, w, b["remote"])
            y = b["y"][: T * _H].view(T, _H)
            ext.q4k_moe(xb, b["glist"], b["gcount"], b["gw"], P["ptrs"], b["hbuf"], y)
            _tl.mb = (li, [(0, T)], None)
            return y.to(x.dtype)
        chunks = [(t0, min(T, t0 + _MAXT)) for t0 in range(0, T, _MAXT)]
        for c, (t0, t1) in enumerate(chunks[:_NSUB]):
            ext.mb_pack(li * _NSUB + c, li, xb[t0:t1].contiguous(), ids[t0:t1].contiguous(), w[t0:t1].contiguous(), b["remote_all"])
        _tl.mb = (li, chunks, (xb, ids, w))
        return torch.zeros_like(x)

    def _mb_finish(mb, out):
        ext = _ext()
        li, chunks, rest = mb
        b = _S["buf"]
        for c, (t0, t1) in enumerate(chunks):
            ext.mb_wait_add(li * _NSUB + c % _NSUB, out[t0:t1])
            n = c + _NSUB
            if n < len(chunks):
                xb, ids, w = rest
                a0, a1 = chunks[n]
                ext.mb_pack(li * _NSUB + n % _NSUB, li, xb[a0:a1].contiguous(), ids[a0:a1].contiguous(), w[a0:a1].contiguous(), b["remote_all"])
        return out

    def _install():
        import sys
        mr = sys.modules.get("vllm.model_executor.layers.fused_moe.runner.moe_runner")
        re_ = sys.modules.get("vllm.model_executor.layers.fused_moe.routed_experts")
        if not _S["fm_done"] and mr is not None and re_ is not None and hasattr(mr, "MoERunner") and hasattr(re_, "RoutedExperts"):
            M, R = mr.MoERunner, re_.RoutedExperts
            orig_fwd, orig_fm = M.forward, R.forward_modular

            def forward_modular(self, x, topk_weights, topk_ids, shared_experts=None, shared_experts_input=None, workspace=None):
                L = getattr(_tl, "layer", None)
                if L is None or not (_FIRST <= L <= _LAST) or x.shape[0] == 0:
                    if workspace is None:
                        return orig_fm(self, x, topk_weights, topk_ids, shared_experts=shared_experts, shared_experts_input=shared_experts_input)
                    return orig_fm(self, x, topk_weights, topk_ids, shared_experts=shared_experts, shared_experts_input=shared_experts_input, workspace=workspace)
                return _moe_split(L, x, topk_weights, topk_ids)

            def forward(self, hidden_states, router_logits, *a, **k):
                m = _layer_re.search(str(getattr(self, "layer_name", "")))
                _tl.layer = int(m.group(1)) if m else None
                _tl.mb = None
                out = orig_fwd(self, hidden_states, router_logits, *a, **k)
                mb = getattr(_tl, "mb", None)
                _tl.layer, _tl.mb = None, None
                if mb is not None:
                    out = _mb_finish(mb, out)
                _S["calls"] += 1
                return out

            R.forward_modular, M.forward = forward_modular, forward
            _S["fm_done"] = True
            _log("MoE hooks installed")
        lp = sys.modules.get("vllm.model_executor.layers.logits_processor")
        if not _S.get("lp_done") and lp is not None and hasattr(lp, "LogitsProcessor") and hasattr(lp.LogitsProcessor, "_apply_head"):
            orig_ah = lp.LogitsProcessor._apply_head

            def _apply_head(self, lm_head, hidden_states, embedding_bias=None, _orig=orig_ah):
                return _apply_head_split(_orig, self, lm_head, hidden_states, embedding_bias)
            lp.LogitsProcessor._apply_head = _apply_head
            _S["lp_done"] = True
            _log("lm_head hook installed")
        bl = sys.modules.get("vllm.model_executor.model_loader.base_loader")
        if not _S["pw_done"] and bl is not None and hasattr(bl, "process_weights_after_loading"):
            orig_pw = bl.process_weights_after_loading

            def process_weights_after_loading(model, model_config, target_device, _orig=orig_pw):
                r = _orig(model, model_config, target_device)
                if _S.get("loaded"):
                    return r
                _S["loaded"] = True
                import torch
                dev = torch.device("cuda", torch.cuda.current_device())
                _free_iq2(model)
                _bufs(dev)
                _load_lmhead(dev)
                free, total = torch.cuda.mem_get_info()
                _log(f"after resident load: cuda free {free / 1e9:.1f} / {total / 1e9:.1f} GB")
                return r
            bl.process_weights_after_loading = process_weights_after_loading
            _S["pw_done"] = True
            _log("load hook installed (base_loader.process_weights_after_loading)")
        _S["done"] = _S["fm_done"] and _S["pw_done"] and _S.get("lp_done", False)

    _orig_import = builtins.__import__

    def _hook(name, *a, **k):
        mod = _orig_import(name, *a, **k)
        if not _S["done"] and name.startswith("vllm"):
            try:
                _install()
            except Exception as ex:  # noqa: BLE001
                if repr(ex) != _S.get("last_ex"):
                    _S["last_ex"] = repr(ex)
                    _log(f"install deferred: {ex!r}")
        return mod
    builtins.__import__ = _hook
