#!/usr/bin/env python3
"""Step 1: log the top-k routed expert ids per token per MoE layer with vLLM. Needs one GPU.

Hook point (no vLLM patch needed). vLLM has a routed-experts capture path:
  --enable-return-routed-experts  ->  RoutedExpertsCapturer (fused_moe/routed_experts_capturer.py)
  binds BaseRouter.set_capture_fn(...) on every MoE layer; BaseRouter._select_experts calls
  capture_fn(topk_ids) right after _compute_routing and BEFORE the EPLB logical->physical mapping,
  so the ids are logical expert ids. CompletionOutput.routed_experts is [seq_len, num_layers, top_k]
  (uint8 if <= 256 experts else uint16), indexed over all hidden layers.
vLLM requires prefix caching for this path ("AuxOutput Connector requires prefix caching"); routing for
cached prompt blocks is returned from the connector's per-block store, so every prompt token still has a row.
The runner checks that rows == prompt + completion - 1 for every request.

Output: OUT_PREFIX.npz  ids_<r>_prefill / ids_<r>_decode  uint16 [T, num_moe_layers, top_k]
        OUT_PREFIX.json request metadata + run metadata (same layout as scripts/reduce_m1.py output)
Usage:
  python3 trace_routes.py --model /path/Qwen3.6-35B-A3B-FP8 --prompts prompts.jsonl --out routes/qwen \
      --max-tokens 256 [--limit N] [--gpu-mem 0.85] [--max-model-len 8192]
"""
import argparse
import json
import os
import time

import numpy as np


def moe_layers_from_config(cfg, num_layers):
    tc = cfg.get("text_config", cfg)
    mlp_only = set(tc.get("mlp_only_layers") or [])
    step = tc.get("decoder_sparse_step") or 1
    first_dense = tc.get("first_k_dense_replace") or 0
    return [l for l in range(num_layers) if l not in mlp_only and l >= first_dense and (l + 1) % step == 0]


def num_experts(cfg):
    tc = cfg.get("text_config", cfg)
    return int(tc.get("num_experts") or tc.get("n_routed_experts") or tc.get("num_local_experts"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-prompt-tokens", type=int, default=4096)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--batch", type=int, default=64, help="requests per generate() call")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--tp", type=int, default=1)
    a = ap.parse_args()

    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    rows = [json.loads(x) for x in open(a.prompts)]
    if a.limit:
        rows = rows[: a.limit]
    tok = AutoTokenizer.from_pretrained(a.model)
    cfg = json.load(open(os.path.join(a.model, "config.json")))
    prompts = []
    for r in rows:
        if r["kind"] == "chat":
            text = tok.apply_chat_template([{"role": "user", "content": r["text"]}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
            ids = tok(text, add_special_tokens=False)["input_ids"]
        else:
            ids = tok(r["text"])["input_ids"]
        prompts.append([int(x) for x in ids[: a.max_prompt_tokens]])

    extra = {"limit_mm_per_prompt": {"image": 0, "video": 0}} if "vision_config" in cfg else {}
    llm = LLM(model=a.model, enable_return_routed_experts=True, enable_prefix_caching=True,
              gpu_memory_utilization=a.gpu_mem, max_model_len=a.max_model_len, enforce_eager=a.enforce_eager,
              tensor_parallel_size=a.tp, **extra)
    sp = SamplingParams(max_tokens=a.max_tokens, temperature=a.temperature, seed=0)

    arrays, meta = {}, []
    num_layers = None
    moe = None
    t0 = time.time()
    for s in range(0, len(rows), a.batch):
        outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts[s:s + a.batch]], sp,
                            use_tqdm=False)
        for j, o in enumerate(outs):
            i = s + j
            c = o.outputs[0]
            re = c.routed_experts
            assert re is not None, "routed_experts missing: was capture enabled?"
            if num_layers is None:
                num_layers = re.shape[1]
                moe = moe_layers_from_config(cfg, num_layers)
            P = len(o.prompt_token_ids)
            n_out = len(c.token_ids)
            re = np.asarray(re)[:, moe].astype(np.uint16)
            # rows = tokens that went through a forward pass: prompt + all but the last sampled token
            arrays[f"ids_{i}_prefill"] = re[:P]
            if re.shape[0] > P:
                arrays[f"ids_{i}_decode"] = re[P:]
            meta.append(dict(req=i, prompt_id=rows[i]["prompt_id"], domain=rows[i]["task"],
                             split="holdout" if (int(rows[i]["prompt_id"].split(":")[1], 16) % 4 == 0) else "calib",
                             source=rows[i]["source"], prompt_tokens=P, completion_tokens=n_out,
                             routed_rows=int(re.shape[0]), finish_reason=c.finish_reason))
        print(f"{min(s + a.batch, len(rows))}/{len(rows)} requests, {time.time() - t0:.0f} s", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out + ".npz", **arrays)
    json.dump({"requests": meta, "model": os.path.basename(a.model.rstrip("/")), "vllm": vllm.__version__,
               "num_hidden_layers_in_capture": num_layers, "moe_layers": moe,
               "num_experts": num_experts(cfg),
               "top_k": int(next(iter(arrays.values())).shape[2]),
               "sampling": {"temperature": a.temperature, "max_tokens": a.max_tokens, "seed": 0},
               "prefix_caching": True,
               "rows_check_failures": sum(1 for m in meta if m["routed_rows"] != m["prompt_tokens"] + m["completion_tokens"] - 1), "wall_s": time.time() - t0}, open(a.out + ".json", "w"), indent=1)


if __name__ == "__main__":
    main()
