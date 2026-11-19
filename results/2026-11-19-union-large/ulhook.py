# Worker-side hooks for union-aware routing eval in vLLM (imported in every worker via collective_rpc).
import torch
import torch.nn.functional as F

WIN = 4
ST = {"route": False, "K": None, "union": False, "T": 0, "capture": None,
      "lref": None, "pref": None, "aref": None, "out": None, "stats": None, "calls": 0, "slice": 256}


def apply_mask(router, w, ids):
    T = ST["T"]; K = ST["K"]
    wT, iT = w[:T], ids[:T].long()
    N, k = wT.shape
    E = router.global_num_experts
    # rank by weight desc, ties -> lower slot index first (matches literal kernel port)
    order = torch.argsort(-wT.float(), dim=1, stable=True)
    rank = torch.empty_like(order); rank.scatter_(1, order, torch.arange(k, device=w.device).expand(N, k).contiguous())
    core = rank < K
    nw = N // WIN
    idw = iT[: nw * WIN].reshape(nw, WIN * k)
    corew = core[: nw * WIN].reshape(nw, WIN * k)
    pres = torch.zeros(nw, E + 1, dtype=torch.bool, device=w.device)
    pres.scatter_(1, torch.where(corew, idw, torch.full_like(idw, E)), True)
    inunion = torch.zeros_like(core)
    inunion[: nw * WIN] = pres[:, :E].gather(1, idw).view(nw * WIN, k)
    keep = (core | inunion) if ST["union"] else core
    st = ST["stats"]
    tail = ~core
    kw = keep[: nw * WIN].reshape(nw, WIN * k)
    pk = torch.zeros(nw, E + 1, dtype=torch.bool, device=w.device)
    pk.scatter_(1, torch.where(kw, idw, torch.full_like(idw, E)), True)
    pa = torch.zeros(nw, E + 1, dtype=torch.bool, device=w.device); pa.scatter_(1, idw, True)
    st += torch.stack([tail.sum().double(), (tail & inunion).sum().double(), pk[:, :E].sum().double(),
                       torch.tensor(float(nw), device=w.device, dtype=torch.float64), pa[:, :E].sum().double(),
                       (wT.double() * keep).sum(), wT.double().sum(), keep.sum().double()])
    w2 = w.clone()
    w2[:T] = torch.where(keep, wT, torch.zeros_like(wT))
    return w2


def install(worker):
    from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
    if not getattr(BaseRouter, "_ul_patched", False):
        orig = BaseRouter._select_experts

        def patched(self, hidden_states, router_logits, topk_indices_dtype=None, *, input_ids=None):
            w, ids = orig(self, hidden_states, router_logits, topk_indices_dtype, input_ids=input_ids)
            if ST["route"] and w.shape[0] >= ST["T"] > 0:
                ST["calls"] += 1
                w = apply_mask(self, w, ids)
            return w, ids
        BaseRouter._select_experts = patched
        BaseRouter._ul_patched = True
    try:
        import vllm.model_executor.layers.fused_moe.experts.gpt_oss_triton_kernels_moe as G
        if not getattr(G, "_ul_patched", False):
            orig_tk = G.triton_kernel_moe_forward

            class _R:  # minimal router stand-in for apply_mask
                pass

            def tk_patched(hidden_states, w1, w2, gating_output, topk, renormalize, activation=None,
                           quant_config=None, apply_router_weight_on_input=False, global_num_experts=-1,
                           expert_map=None, **kw):
                if not (ST["route"] and gating_output.shape[0] >= ST["T"] > 0 and expert_map is None):
                    kw2 = dict(kw)
                    if activation is not None: kw2["activation"] = activation
                    return orig_tk(hidden_states, w1, w2, gating_output, topk, renormalize, quant_config=quant_config,
                                   apply_router_weight_on_input=apply_router_weight_on_input,
                                   global_num_experts=global_num_experts, expert_map=expert_map, **kw2)
                ST["calls"] += 1
                E = gating_output.shape[-1]
                if activation is None:
                    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
                    activation = MoEActivation.SWIGLUOAI
                from triton_kernels.topk import topk as topk_fn
                lg = gating_output if renormalize else torch.softmax(gating_output, -1)
                tr = topk_fn(lg, topk, apply_softmax=renormalize)
                if isinstance(tr, tuple):
                    w, ids = tr[0], tr[1]
                else:
                    w, ids = tr.vals, tr.indx
                ST["wdtype"] = str(w.dtype)
                r = _R(); r.global_num_experts = E
                w = apply_mask(r, w, ids)
                routing_data, gather_idx, scatter_idx = G.make_routing_data(ids.long(), w, E)
                out = torch.empty_like(hidden_states)
                from vllm.model_executor.layers.fused_moe.config import FUSED_MOE_UNQUANTIZED_CONFIG
                return G.triton_kernel_fused_experts(out, hidden_states, w1, w2, routing_data, gather_idx, scatter_idx,
                    topk=topk, activation=activation, quant_config=quant_config if quant_config is not None else FUSED_MOE_UNQUANTIZED_CONFIG,
                    apply_router_weight_on_input=apply_router_weight_on_input, global_num_experts=E, expert_map=None)
            G.triton_kernel_moe_forward = tk_patched
            G._ul_patched = True
    except Exception as e:
        print("triton patch failed", repr(e), flush=True)
    model = worker.get_model()
    nrouters = 0; info = []
    for n, m in model.named_modules():
        r = getattr(m, "router", None)
        if isinstance(r, BaseRouter):
            nrouters += 1
            if len(info) < 1:
                qm = getattr(getattr(m, "quant_method", None), "__class__", type(None)).__name__
                info.append(dict(name=n, router=type(r).__name__, top_k=r.top_k, E=r.global_num_experts, qm=qm,
                                 mono=bool(getattr(getattr(m, "quant_method", None), "is_monolithic", None))))
    model.register_forward_hook(fwd_hook)
    ST["model"] = model
    import torch.distributed as dist
    return dict(nrouters=nrouters, info=info, rank=(dist.get_rank() if dist.is_initialized() else 0))


@torch.no_grad()
def fwd_hook(mod, args, out):
    cap = ST["capture"]
    if cap is None:
        return
    h = out[0] if isinstance(out, (tuple, list)) else out
    T = ST["T"]
    if not torch.is_tensor(h) or h.shape[0] < T:
        return
    ST["capture"] = None
    model = ST["model"]
    S = ST["slice"]
    kls, agree, maxd = [], 0, 0.0
    if cap == "ref":
        ST["lref"] = []; ST["aref"] = []
    for s in range(0, T, S):
        lg = model.compute_logits(h[s:min(T, s + S)])
        if lg is None:
            continue
        lp = F.log_softmax(lg.float(), -1)
        if cap == "ref":
            ST["lref"].append(lp); ST["aref"].append(lp.argmax(-1))
            if ST.get("ids") is not None:
                e = min(T, s + S); tg = torch.tensor(ST["ids"][s + 1:e + 1], device=lp.device)
                m = tg.shape[0]
                ST.setdefault("nll", []).append(float(-lp[:m].gather(1, tg[:, None]).sum()))
                ST.setdefault("nlln", []).append(m)
                ST.setdefault("ent", []).append(float(-(lp.exp() * lp).sum(-1).sum())); ST.setdefault("entn", []).append(lp.shape[0])
        else:
            lr = ST["lref"][s // S]
            kls.append((lr.exp() * (lr - lp)).sum(-1).cpu())
            agree += int((lp.argmax(-1) == ST["aref"][s // S]).sum())
            maxd = max(maxd, float((lp - lr).abs().max()))
    if cap == "ref":
        ST["out"] = dict(kind="ref", n=sum(x.shape[0] for x in ST["lref"]))
        if ST.get("nll"):
            ST["out"].update(nll=sum(ST["nll"]) / sum(ST["nlln"]), ent=sum(ST["ent"]) / sum(ST["entn"]))
            ST["nll"] = []; ST["nlln"] = []; ST["ent"] = []; ST["entn"] = []
    else:
        ST["out"] = dict(kind="var", kl=torch.cat(kls).tolist() if kls else None, agree=agree, maxd=maxd)


def set_state(worker, **kw):
    ST.update(kw)
    if kw.get("reset_stats"):
        ST["stats"] = torch.zeros(8, dtype=torch.float64, device="cuda" if torch.cuda.is_available() else "cpu"); ST["calls"] = 0
    return True


def get_out(worker):
    o = ST["out"]; ST["out"] = None
    if o is None:
        return None
    o = dict(o); o["calls"] = ST["calls"]
    o["stats"] = ST["stats"].tolist() if ST["stats"] is not None else None
    return o
