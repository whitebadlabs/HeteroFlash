"""FR-Spec draft-only head for the GLM-5.3 MTP drafter (Bet 2). Lossless: the verifier keeps the full head.
HF_FRSPEC_IDS=<json from build_ids.py>, HF_FRSPEC_CKPT=<model dir> (lm_head.weight F8_E4M3 + weight_scale U8 E8M0
per 1x32 group, read straight from the safetensors). The draft argmax runs over a BF16 sub-head of those rows."""
import os
if os.environ.get("HF_FRSPEC_IDS"):
    import builtins, json
    _st = {"done": False}

    def _load_subhead(dev):
        import struct, torch
        d = os.environ["HF_FRSPEC_CKPT"]; idx = json.load(open(f"{d}/model.safetensors.index.json"))["weight_map"]
        def tensor(name):
            fn = f"{d}/{idx[name]}"
            with open(fn, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(n)); m = h[name]
                f.seek(8 + n + m["data_offsets"][0]); buf = f.read(m["data_offsets"][1] - m["data_offsets"][0])
            dt = {"F8_E4M3": torch.float8_e4m3fn, "U8": torch.uint8, "BF16": torch.bfloat16}[m["dtype"]]
            return torch.frombuffer(bytearray(buf), dtype=dt).reshape(m["shape"])
        ids = torch.tensor(json.load(open(os.environ["HF_FRSPEC_IDS"]))["ids"], dtype=torch.long)
        w = tensor("lm_head.weight")[ids].float()
        s = tensor("lm_head.weight_scale")[ids].float()
        g = w.shape[1] // s.shape[1]
        w = (w.view(w.shape[0], s.shape[1], g) * torch.exp2(s - 127.0).unsqueeze(-1)).view(w.shape[0], -1)
        return w.to(torch.bfloat16).to(dev).contiguous(), ids.to(dev)

    def _install():
        import sys
        m = sys.modules.get("vllm.model_executor.models.glm5next.nvidia.mtp")
        if m is None or not hasattr(m, "Glm5NextMultiTokenPredictor"):
            return
        import torch
        P = m.Glm5NextMultiTokenPredictor

        def get_top_tokens(self, hidden_states, spec_step_idx=0):
            sub = getattr(self, "_hf_frspec", None)
            if sub is None:
                sub = _load_subhead(hidden_states.device); self._hf_frspec = sub
                print(f"[frspec] draft sub-head rows={sub[0].shape[0]} bytes={sub[0].numel()*2/1e6:.0f}MB", flush=True)
            W, ids = sub
            logits = torch.matmul(hidden_states.to(torch.bfloat16), W.t())
            return ids[logits.argmax(dim=-1)]
        P.get_top_tokens = get_top_tokens
        _st["done"] = True

    _imp = builtins.__import__

    def _hook(name, *a, **k):
        mod = _imp(name, *a, **k)
        if not _st["done"] and name.startswith("vllm"):
            try: _install()
            except Exception:  # noqa: BLE001
                pass
        return mod
    builtins.__import__ = _hook
