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
  HF_SPLIT_F32 0|1         bf16 (default, the project lead 2026-11-17: optimise tok/s) or fp32 partial-sum reply
  HF_SPLIT_HOT json        {layer: [experts, hottest first]}       HF_SPLIT_NRES    resident experts per layer
  HF_SPLIT_GGUF / HF_SPLIT_IDX  Q4_K GGUF + expert index (runtime/gguf_expert_index.py format)
  HF_SPLIT_COUNT path      count decode-window routes per layer (hot-set calibration), dumped every 2000 MoE calls
  HF_SPLIT_STATS path      assign + exchange counters, dumped every 2000 MoE calls
2026-11-17 speed-quality frontier (all switchable at run time on one boot, CUDA graphs stay valid):
  HF_SPLIT_DROP 0|1        compile the expert-drop kernel into layers 3-44 (no-op until set): per-layer kmax, tau, renorm
                           (req file drt_drop.req: "K TAU RENORM" or "json {layer: K, ...} TAU RENORM")
  HF_SPLIT_FRSPEC ids.json FR-Spec draft head: the drafter's argmax over these lm_head rows only (GPU, no exchange);
                           gates[2] switches it (req drt_gates.req "LMH KDA FRS")
  HF_SPLIT_ALT_LAYERS l,.. decode windows of these layers use the workstation's alt (lower-bit) expert table
                           (server env HF_ALT_*); req file drt_alt.req: "l1,l2,..." or "none"
2026-11-18 union-aware drop: drt_drop.req "K TAU RENORM [UNION]"; UNION 1 keeps tail picks whose expert is in the
  window union of core picks (decode windows), so the experts read stay those of plain top-K.
2026-11-18 expert deferral study (decode-size windows T <= HF_SPLIT_MAXT only; prefill chunks unchanged):
  HF_SPLIT_DEFER 0|1       compile a second, gated wait per layer: layer L's workstation partial is added after layer
                           L+1's MoE instead of L's (defer depth 1, the GPU no longer blocks at L). Per-layer mask at run time,
                           req file drt_defer.req: "none" | "all" | "l1,l2,..." [LATE 0|1]; layer 44 is never deferred.
  HF_SPLIT_LATE 0|1        with HF_SPLIT_DROP: experts removed by the drop kernel (e.g. expert 8 under k7) are sent to the
                           workstation in a second slot and added one layer late ("add expert 8 late" vs "drop expert 8");
                           switched by the LATE field of drt_defer.req (layers 3-43; layer 44 drops them).
  GPU wait timers (always on): per layer, blocked ns in the wait kernel and GPU work between pack and wait, in the stats dump.
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
    _F32 = os.environ.get("HF_SPLIT_F32", "0") == "1"   # The project lead 11-17: bf16 partial exchange is the default (faster); fp32 is the option
    _FLAGS = int(os.environ.get("HF_REMOTE_FLAGS", "3")) | (4 if _F32 else 0)
    _NRES = int(os.environ.get("HF_SPLIT_NRES", "0"))
    _HOT = os.environ.get("HF_SPLIT_HOT", "")
    _GGUF = os.environ.get("HF_SPLIT_GGUF", "/gguf/GLM-5.3-Flash-Q8attn-G0_q4kimexp.gguf")
    _IDX = os.environ.get("HF_SPLIT_IDX", "/drt/q4k_experts.idx")
    _COUNT = os.environ.get("HF_SPLIT_COUNT", "")
    _STATS = os.environ.get("HF_SPLIT_STATS", "")
    _LMH_VG = int(os.environ.get("HF_SPLIT_LMHEAD_VG", "0"))   # >0: lm_head rows [0, VG) on the GPU, the rest on the CPU (lid 0)
    _CKPT = os.environ.get("HF_SPLIT_CKPT", "/m")
    _KDA_HC = int(os.environ.get("HF_SPLIT_KDA_HC", "0"))   # >0: q/k/v rows of the last HC heads of every KDA layer on the CPU
    _DROP = os.environ.get("HF_SPLIT_DROP", "0") == "1"
    _FRS = os.environ.get("HF_SPLIT_FRSPEC", "")
    _DEFER = os.environ.get("HF_SPLIT_DEFER", "0") == "1"
    _LATE = os.environ.get("HF_SPLIT_LATE", "0") == "1" and _DROP and _DEFER   # needs both
    _tl = threading.local()
    _layer_re = re.compile(r"layers\.(\d+)\.")
    _S = {"ext": None, "buf": None, "pool": None, "calls": 0, "done": False, "pw_done": False, "fm_done": False}

    def _log(msg):
        print(f"[hf-drt] {msg}", flush=True)

    _ext_lock = threading.Lock()

    def _ext():
        if _S["ext"] is not None:
            return _S["ext"]
        with _ext_lock:
            if _S["ext"] is None:
                from torch.utils.cpp_extension import load
                src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_drt.cu")
                ext = load(name="hf_drt_union", sources=[src], extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=False)
                ext.mb_init((_LAST - _FIRST + 1) * _NSUB, _host, int(_port), _FLAGS)
                if os.environ.get("HF_SPLIT_ALT_LAYERS"):
                    m, Ls = _alt_mask(os.environ["HF_SPLIT_ALT_LAYERS"]); ext.mb_set_alt(m); _log(f"alt expert table on layers {Ls} at start")
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
                "ab": torch.tensor([_A, _B], dtype=torch.float32, device=device),
                # run-time switches read by the kernels (CUDA graphs stay valid): [lm_head split on, KDA split on]
                "gates": torch.tensor([int(os.environ.get("HF_SPLIT_LMH_ON", "1")), int(os.environ.get("HF_SPLIT_KDA_ON", "1")),
                                       int(os.environ.get("HF_SPLIT_FRS_ON", "1" if _FRS else "0"))], dtype=torch.int32, device=device),
                "drop_k": torch.full((nl,), int(os.environ.get("HF_SPLIT_DROP_K", "8")), dtype=torch.int32, device=device),
                "drop_tr": torch.tensor([float(os.environ.get("HF_SPLIT_DROP_TAU", "0")), float(os.environ.get("HF_SPLIT_DROP_RENORM", "0")),
                                        float(os.environ.get("HF_SPLIT_DROP_UNION", "0"))],
                                        dtype=torch.float32, device=device),
                "counts": torch.zeros(nl, _NE, dtype=torch.int64, device=device),
                "defer": torch.zeros(nl, dtype=torch.int32, device=device),     # 1: layer's partial added at layer + 1
                "late": torch.zeros(1, dtype=torch.int32, device=device),       # 1: dropped experts sent + added late
            }
            if _S["pool"] is None:   # resident pool not loaded yet: dummy bases, every slot -1 (all routes to the CPU)
                dummy = torch.zeros(64, dtype=torch.uint8, device=device)
                _S["pool"] = {"dummy": dummy, "ptrs": torch.tensor([dummy.data_ptr()] * 3, dtype=torch.int64, device=device),
                              "slot_of": torch.full((nl, _NE), -1, dtype=torch.int32, device=device)}
            _S["buf"] = b
            _S["none"] = torch.zeros(0, dtype=torch.int32, device=device)
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
    def _ckpt_get(name):
        from safetensors import safe_open
        wm = _S.get("wm")
        if wm is None:
            wm = _S["wm"] = json.load(open(f"{_CKPT}/model.safetensors.index.json"))["weight_map"]
        with safe_open(f"{_CKPT}/{wm[name]}", framework="pt") as f:
            return f.get_tensor(name)

    def _load_lmhead(device):
        if _LMH_VG <= 0:
            return
        import torch
        from safetensors import safe_open
        wm = json.load(open(f"{_CKPT}/model.safetensors.index.json"))["weight_map"]
        w8 = _ckpt_get("lm_head.weight").view(torch.uint8); V = w8.shape[0]
        W = w8[:_LMH_VG].contiguous().to(device); Wc = w8[_LMH_VG:].contiguous().to(device); del w8
        s8 = _ckpt_get("lm_head.weight_scale").view(torch.uint8)
        S = s8[:_LMH_VG].contiguous().to(device); Sc = s8[_LMH_VG:].contiguous().to(device)
        ext = _ext()
        si = ext.dense_register(0, 4096, V - _LMH_VG)
        _S["lmh"] = {"W": W, "S": S, "Wc": Wc, "Sc": Sc, "V": V, "si": si}
        if _FRS:   # FR-Spec draft-only head: rows of the frequent ids (exact FP8 bytes + E8M0 scales of the checkpoint)
            fids = torch.tensor(json.load(open(_FRS))["ids"], dtype=torch.long)
            w8f = _ckpt_get("lm_head.weight").view(torch.uint8)
            _S["frs"] = {"W": w8f[fids].contiguous().to(device), "S": s8[fids].contiguous().to(device), "ids": fids.to(device)}
            del w8f
            _log(f"FR-Spec draft head: {fids.numel()} rows ({_S['frs']['W'].numel() / 1e6:.0f} MB FP8), switch gates[2]")
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

    _AB_REQ = os.environ.get("HF_SPLIT_AB_REQ", "/out/drt_ab.req")

    _G_REQ = os.environ.get("HF_SPLIT_GATES_REQ", "/out/drt_gates.req")

    def _maybe_set_gates():
        if _S.get("ah_calls", 0) % 16 or not os.path.exists(_G_REQ) or _S["buf"] is None:
            return
        import torch
        try:
            v = [int(t) for t in open(_G_REQ).read().split()[:3]]
            torch.cuda.synchronize()
            _S["buf"]["gates"][: len(v)].copy_(torch.tensor(v, dtype=torch.int32))
            torch.cuda.synchronize()
            _log(f"split switches set: {_S['buf']['gates'].tolist()} (lm_head, kda, frspec)")
        except Exception as ex:  # noqa: BLE001
            _log(f"bad gates request: {ex!r}")
        os.remove(_G_REQ)

    def _maybe_set_ab():
        if _S.get("ah_calls", 0) % 16 or not os.path.exists(_AB_REQ) or _S["buf"] is None:
            return
        try:
            a, b = (float(v) for v in open(_AB_REQ).read().split()[:2])
            _S["buf"]["ab"].copy_(torch_tensor([a, b]))
            _log(f"balance rule set: cap = round({a} * U + {b})")
        except Exception as ex:  # noqa: BLE001
            _log(f"bad ab request: {ex!r}")
        os.remove(_AB_REQ)

    _D_REQ = os.environ.get("HF_SPLIT_DROP_REQ", "/out/drt_drop.req")
    _ALT_REQ = os.environ.get("HF_SPLIT_ALT_REQ", "/out/drt_alt.req")
    _DEF_REQ = os.environ.get("HF_SPLIT_DEFER_REQ", "/out/drt_defer.req")

    def _alt_mask(spec):
        spec = spec.strip()
        if spec in ("", "none"):
            return 0, []
        Ls = [int(v) for v in spec.split(",") if v.strip()]
        return sum(1 << (L - _FIRST) for L in Ls if _FIRST <= L <= _LAST), Ls

    def _maybe_set_drop_alt():
        if _S.get("ah_calls", 0) % 16 or _S["buf"] is None:
            return
        import torch
        if os.path.exists(_D_REQ):
            try:
                parts = open(_D_REQ).read().split()
                if parts[0] == "json":
                    per = {int(k): int(v) for k, v in json.loads(parts[1]).items()}; kk = [per.get(L, 8) for L in range(_FIRST, _LAST + 1)]
                    tau, rn = float(parts[2]), float(parts[3]); uni = float(parts[4]) if len(parts) > 4 else 0.0
                else:
                    kk = [int(parts[0])] * (_LAST - _FIRST + 1); tau, rn = float(parts[1]), float(parts[2]); uni = float(parts[3]) if len(parts) > 3 else 0.0
                torch.cuda.synchronize()
                _S["buf"]["drop_k"].copy_(torch.tensor(kk, dtype=torch.int32)); _S["buf"]["drop_tr"].copy_(torch.tensor([tau, rn, uni], dtype=torch.float32))
                torch.cuda.synchronize()
                _log(f"expert drop set: kmax {kk if len(set(kk)) > 1 else kk[0]} tau {tau} renorm {rn} union {uni} (compiled in: {_DROP})")
            except Exception as ex:  # noqa: BLE001
                _log(f"bad drop request: {ex!r}")
            os.remove(_D_REQ)
        if os.path.exists(_DEF_REQ):
            try:
                parts = open(_DEF_REQ).read().split()
                spec = parts[0] if parts else "none"
                Ls = list(range(_FIRST, _LAST)) if spec == "all" else ([] if spec == "none" else [int(v) for v in spec.split(",") if v])
                Ls = [L for L in Ls if _FIRST <= L < _LAST]          # layer 44 has no next MoE layer
                late = int(parts[1]) if len(parts) > 1 else 0
                m = [1 if L in Ls else 0 for L in range(_FIRST, _LAST + 1)]
                torch.cuda.synchronize()
                _S["buf"]["defer"].copy_(torch.tensor(m, dtype=torch.int32)); _S["buf"]["late"].fill_(late)
                torch.cuda.synchronize()
                _log(f"deferral set: layers {Ls} ({len(Ls)}) late {late} (compiled in: defer {_DEFER} late {_LATE})")
            except Exception as ex:  # noqa: BLE001
                _log(f"bad defer request: {ex!r}")
            os.remove(_DEF_REQ)
        if os.path.exists(_ALT_REQ):
            try:
                m, Ls = _alt_mask(open(_ALT_REQ).read())
                _ext().mb_set_alt(m)
                _log(f"alt expert table on layers {Ls} (mask {m:#x})")
            except Exception as ex:  # noqa: BLE001
                _log(f"bad alt request: {ex!r}")
            os.remove(_ALT_REQ)

    def torch_tensor(v):
        import torch
        return torch.tensor(v, dtype=torch.float32)

    # Quality capture for small-batch evals (HF_SPLIT_CAPTURE=dir): the eval client writes the request index to
    # dir/capture.cur before each 2048-token request; every logits row of that request is appended in order and rows
    # [1024, 2047) are saved as logits_<pid>_<n>.npy (log-softmax, fp16), the layout src/control_kld.py compare reads.
    _CAP = os.environ.get("HF_SPLIT_CAPTURE", "")
    _cap = {"id": None, "rows": [], "n": 0, "log": []}

    def _cap_flush():
        import numpy as np
        import torch
        if _cap["id"] is None or not _cap["rows"]:
            return
        # chunked prefill emits, per 16-token step, the prompt-logprob rows (T > 1) and one sampling row (T = 1, a
        # throw-away for partial prefills): keep the multi-row calls only, in order = prompt positions 0..2047.
        rows = [r for r in _cap["rows"] if r.shape[0] > 1]
        allr = torch.cat(rows, 0) if rows else torch.zeros(0)
        _cap["log"].append((_cap["id"], int(allr.shape[0]), len(_cap["rows"]), len(rows)))
        if allr.shape[0] >= 2047:
            _cap["n"] += 1
            np.save(os.path.join(_CAP, f"logits_{os.getpid()}_{_cap['n']:05d}.npy"), allr[1024:2047].numpy())
        with open(os.path.join(_CAP, "capture_rows.json"), "w") as f:
            json.dump(_cap["log"], f)
        _cap["rows"] = []

    def _cap_rows(out):
        import torch
        try:
            cur = open(os.path.join(_CAP, "capture.cur")).read().strip()
        except OSError:
            return
        if cur != _cap["id"]:
            _cap_flush(); _cap["id"] = cur
        if cur == "stop":
            return
        _cap["rows"].append(torch.log_softmax(out.float(), dim=-1).to(torch.float16).cpu())

    def _apply_head_split(orig, self, lm_head, hidden_states, embedding_bias):
        out = _apply_head_split2(orig, self, lm_head, hidden_states, embedding_bias)
        if _CAP:
            _cap_rows(out[..., : getattr(self, "org_vocab_size", out.shape[-1])])
        return out

    def _apply_head_split2(orig, self, lm_head, hidden_states, embedding_bias):
        import torch
        _maybe_lazy_load()
        _maybe_set_ab()
        _maybe_set_gates()
        _maybe_set_drop_alt()
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
        g = _S["buf"]["gates"][0:1]
        ext.dense_pack(L["si"], x, g)
        out = torch.zeros(T, Vp, dtype=torch.bfloat16, device=x.device) if Vp > L["V"] else torch.empty(T, Vp, dtype=torch.bfloat16, device=x.device)
        ext.mxfp8_gemv(L["W"], L["S"], x, out, _S["none"], 0)
        ext.mxfp8_gemv(L["Wc"], L["Sc"], x, out.narrow(1, _LMH_VG, L["V"] - _LMH_VG), g, 0)   # split off: GPU does these rows
        ext.dense_wait_copy(L["si"], T, out, _LMH_VG, 0, -1, g)
        return out if dt == torch.bfloat16 else out.to(dt)

    def _top_tokens(orig, self, lm_head, hidden_states, embedding_bias=None):
        """Draft argmax (LogitsProcessor.get_top_tokens, used by the MTP drafter only). gates[2] = 1: FR-Spec sub-head
        on the GPU, no exchange; 0: the full split head (as _apply_head). Both branches are captured; kernels no-op by gate."""
        import torch
        F, L = _S.get("frs"), _S.get("lmh")
        T = int(hidden_states.shape[0])
        if F is None or L is None or "proto" not in L or embedding_bias is not None or T == 0 or T > 16 \
                or getattr(self, "soft_cap", None) is not None or getattr(self, "scale", 1.0) != 1.0:
            return orig(self, lm_head, hidden_states, embedding_bias)
        ext = _ext(); g = _S["buf"]["gates"]
        fr = g[2:3]
        gl = g[0:1] * (1 - fr)            # exchange on: lm_head split on and FR-Spec off
        gwc = g[0:1] + 2 * fr             # GPU computes the workstation's rows: split off and FR-Spec off (== 0)
        x = hidden_states.to(torch.bfloat16).contiguous()
        ext.dense_pack(L["si"], x, gl)
        out = torch.empty(T, L["V"], dtype=torch.bfloat16, device=x.device)
        ext.mxfp8_gemv(L["W"], L["S"], x, out, fr, 0)
        ext.mxfp8_gemv(L["Wc"], L["Sc"], x, out.narrow(1, _LMH_VG, L["V"] - _LMH_VG), gwc, 0)
        ext.dense_wait_copy(L["si"], T, out, _LMH_VG, 0, -1, gl)
        sub = torch.empty(T, F["W"].shape[0], dtype=torch.bfloat16, device=x.device)
        ext.mxfp8_gemv(F["W"], F["S"], x, sub, fr, 1)
        tok = torch.where(fr.bool(), F["ids"][sub.argmax(dim=-1)], out.argmax(dim=-1))
        _S["frs_calls"] = _S.get("frs_calls", 0) + 1
        if _S["frs_calls"] == 1:
            _log("FR-Spec get_top_tokens path active")
        return tok

    # ------------------------------------------------------------------ KDA q/k/v projection rows by head (stage M3b)
    def _load_kda(model, device):
        if _KDA_HC <= 0:
            return
        import torch
        from safetensors import safe_open
        wm = json.load(open(f"{_CKPT}/model.safetensors.index.json"))["weight_map"]
        ext = _ext()
        n = 0
        for name, mod in model.named_modules():
            if not hasattr(mod, "in_proj_qkv") or not hasattr(mod, "project_split_input"):
                continue
            m = _layer_re.search(name); L = int(m.group(1))
            pre = f"model.language_model.layers.{L}.self_attn."
            parts = []
            for pr in ("q_proj", "k_proj", "v_proj"):
                w = _ckpt_get(pre + pr + ".weight").view(torch.uint8); sc = _ckpt_get(pre + pr + ".weight_scale").view(torch.uint8)
                P = w.shape[0]; g = P - _KDA_HC * 128
                parts.append((w[:g].contiguous().to(device), sc[:g].contiguous().to(device), P, g,
                              w[g:].contiguous().to(device), sc[g:].contiguous().to(device)))
            si = ext.dense_register(1 + n, 4096, 3 * _KDA_HC * 128)
            mod.in_proj_qkv._hf_kda = {"parts": parts, "si": si, "L": L}
            orig_fwd = mod.in_proj_qkv.forward

            def fwd(x, *a, _orig=orig_fwd, _mod=mod.in_proj_qkv, **k):
                T = int(x.shape[0])
                if T == 0 or T > 16 or a or k:
                    return _orig(x, *a, **k)
                d = _mod._hf_kda
                gt = _S["buf"]["gates"][1:2]
                xb = x.to(torch.bfloat16).contiguous()
                ext.dense_pack(d["si"], xb, gt)
                Ptot = sum(p[2] for p in d["parts"])
                out = torch.empty(T, Ptot, dtype=torch.bfloat16, device=x.device)
                off = 0
                for (W, S, P, g, Wc, Sc) in d["parts"]:
                    ext.mxfp8_gemv(W, S, xb, out.narrow(1, off, g), _S["none"], 0)
                    ext.mxfp8_gemv(Wc, Sc, xb, out.narrow(1, off + g, P - g), gt, 0)    # split off: GPU does these rows
                    off += P
                off, yoff, hc = 0, 0, _KDA_HC * 128
                for (W, S, P, g, Wc, Sc) in d["parts"]:
                    ext.dense_wait_copy(d["si"], T, out, off + g, yoff, hc, gt)
                    off += P; yoff += hc
                return out, None
            mod.in_proj_qkv.forward = fwd
            n += 1
        _log(f"KDA q/k/v split: {n} layers, last {_KDA_HC} heads of q, k and v on the workstation (dense slots 1-{n})")

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
            if os.path.exists(req):   # 2026-11-18: request-only (the 30 s dump raced CUDA-graph capture on f2)
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
                       "mailbox_req_pairs_serverus_relayus": ms[:4], "mailbox_decode_T<=4_req_pairs_serverus_relayus": ms[4:8], "alt_requests": ms[8] if len(ms) > 8 else 0,
                       "dense_req_serverus_relayus": _ext().dense_stats(), "wait": _wait_stats()}, open(_STATS, "w"))

    def _wait_stats():
        """GPU wait timers (host memory, no CUDA op): per MoE layer, decode (T <= 4) and other windows, main slot and late slot:
        [n, blocked_ns, overlap_ns (pack -> wait start), zero-wait n]; dense slots [n, blocked_ns]."""
        ext = _ext()
        if not hasattr(ext, "mb_wait_stats"):
            return None
        v = ext.mb_wait_stats(); ns = (_LAST - _FIRST + 1) * _NSUB; out = {}
        for bk, bn in ((0, "decode"), (1, "other")):
            for sub, sn in ((0, "main"), (1, "late")):
                out[f"{bn}_{sn}"] = {str(L): v[((bk * ns) + (L - _FIRST) * _NSUB + sub) * 4:((bk * ns) + (L - _FIRST) * _NSUB + sub) * 4 + 4]
                                     for L in range(_FIRST, _LAST + 1)}
        out["dense"] = ext.dense_wait_stats()
        return out

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
        ids_o, w_o = ids, w
        if _DROP:
            ids2, w2 = torch.empty_like(ids), torch.empty_like(w)
            ext.drop(ids, w, b["drop_k"], li, b["drop_tr"], ids2, w2)
            ids, w = ids2, w2
        if T <= _SPLIT_MAXT and _LATE and L < _LAST:   # dropped experts -> workstation, second slot, added at L + 1 (gate "late")
            lost = (ids < 0) & (ids_o >= 0)
            ext.mb_pack(li * _NSUB + 1, li, xb, torch.where(lost, ids_o, torch.full_like(ids_o, -1)).contiguous(),
                        torch.where(lost, w_o, torch.zeros_like(w_o)).contiguous(), b["remote_all"], b["late"], 1)
        if T <= _SPLIT_MAXT:
            if _COUNT:
                b["counts"][li].index_add_(0, ids.reshape(-1).long(), torch.ones_like(ids.reshape(-1), dtype=torch.int64))
            ext.assign(ids, w, P["slot_of"][li], b["ab"], _MODE, b["remote"], b["glist"], b["gcount"], b["gw"], b["stats"][li])
            ext.mb_pack(li * _NSUB, li, xb, ids, w, b["remote"], _S["none"], 0)
            y = b["y"][: T * _H].view(T, _H)
            ext.q4k_moe(xb, b["glist"], b["gcount"], b["gw"], P["ptrs"], b["hbuf"], y)
            _tl.mb = (li, [(0, T)], None)
            return y.to(x.dtype)
        chunks = [(t0, min(T, t0 + _MAXT)) for t0 in range(0, T, _MAXT)]
        for c, (t0, t1) in enumerate(chunks[:_NSUB]):
            ext.mb_pack(li * _NSUB + c, li | 0x100, xb[t0:t1].contiguous(), ids[t0:t1].contiguous(), w[t0:t1].contiguous(), b["remote_all"], _S["none"], 0)
        _tl.mb = (li, chunks, (xb, ids, w))
        return torch.zeros_like(x)

    def _mb_finish(mb, out):
        ext = _ext()
        li, chunks, rest = mb
        b = _S["buf"]
        for c, (t0, t1) in enumerate(chunks):
            ext.mb_wait_add(li * _NSUB + c % _NSUB, out[t0:t1], _S["none"], 0)
            n = c + _NSUB
            if n < len(chunks):
                xb, ids, w = rest
                a0, a1 = chunks[n]
                ext.mb_pack(li * _NSUB + n % _NSUB, li | 0x100, xb[a0:a1].contiguous(), ids[a0:a1].contiguous(), w[a0:a1].contiguous(), b["remote_all"], _S["none"], 0)
        return out

    def _mb_finish_decode(mb, out):
        """Decode-size window (one slot per layer). Gated waits: own partial now (defer[li] == 0), and the previous layer's
        partial (defer[li-1] == 1) and late experts (late == 1) here, one layer late. Both waits are captured either way."""
        ext = _ext(); b = _S["buf"]
        li = mb[0]; T = int(out.shape[0])
        if not _DEFER:
            ext.mb_wait_add(li * _NSUB, out, _S["none"], 0)
            return out
        ext.mb_wait_add(li * _NSUB, out, b["defer"][li:li + 1], 0)
        p = getattr(_tl, "pend", None)
        if p is not None and p[0] == li - 1 and p[1] == T:
            ext.mb_wait_add((li - 1) * _NSUB, out, b["defer"][li - 1:li], 1)
            if _LATE:
                ext.mb_wait_add((li - 1) * _NSUB + 1, out, b["late"], 1)
        elif li > 0:
            _S["pend_miss"] = _S.get("pend_miss", 0) + 1
            if _S["pend_miss"] <= 5:
                _log(f"deferral: no pending layer {li - 1 + _FIRST} before layer {li + _FIRST} (T {T}, pend {p})")
        _tl.pend = (li, T)
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
                if shared_experts is not None and shared_experts_input is not None:   # runs only if the kernel owned it
                    from vllm.model_executor.layers.fused_moe.runner.shared_experts import SharedExpertsOrder
                    shared_experts(shared_experts_input, SharedExpertsOrder.MK_INTERNAL_OVERLAPPED)
                return _moe_split(L, x, topk_weights, topk_ids)

            def forward(self, hidden_states, router_logits, *a, **k):
                m = _layer_re.search(str(getattr(self, "layer_name", "")))
                _tl.layer = int(m.group(1)) if m else None
                _tl.mb = None
                out = orig_fwd(self, hidden_states, router_logits, *a, **k)
                mb = getattr(_tl, "mb", None)
                _tl.layer, _tl.mb = None, None
                if mb is not None and mb[2] is None:
                    out = _mb_finish_decode(mb, out)
                elif mb is not None:
                    _tl.pend = None
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
            if _FRS and hasattr(lp.LogitsProcessor, "get_top_tokens"):
                orig_tt = lp.LogitsProcessor.get_top_tokens

                def get_top_tokens(self, lm_head, hidden_states, embedding_bias=None, _orig=orig_tt):
                    return _top_tokens(_orig, self, lm_head, hidden_states, embedding_bias)
                lp.LogitsProcessor.get_top_tokens = get_top_tokens
                _log("FR-Spec get_top_tokens hook installed")
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
                _load_kda(model, dev)
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
