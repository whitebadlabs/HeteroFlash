"""Hot-expert tier experiment, item 5: capture layer-32 routing on Hermes-style and long-context workloads.

Runs against the control (EXL3, trace variant: eager, trace hook, no prefix cache, max-num-seqs 1) on the workstation.
For each selected Hermes-suite item, sets the hook's capture_inputs tag (MoE layer 29 = model layer 32 only) and sends
the item's first turn. The hook writes inputs_<tag>_*.npz with per-forward router logits, so routing for prefill
(ntok > 1) and decode (ntok == 1) tokens is recoverable per item.
Usage: python3 route_capture_client.py SUITE_ITEMS_DIR CONTROL_JSON OUT_LOG
"""
import json, os, sys, time, urllib.request

ITEMS, CTRL, LOG = sys.argv[1:4]
URL = "http://127.0.0.1:8001/v1/chat/completions"
PLAN = {"long_context": 6, "tool_calling": 10, "multi_turn": 10, "coding": 10, "instruction_following": 10,
        "reasoning": 10, "prose": 10}


def set_tag(tag):
    base = json.load(open(CTRL))
    base["capture_inputs"] = None if tag is None else {"max_rows": 5_000_000, "layers": [29], "tag": tag}
    tmp = CTRL + ".tmp"
    json.dump(base, open(tmp, "w")); os.replace(tmp, CTRL)
    time.sleep(2.5)                                   # hook re-reads the control file at most once per second


def post(body):
    r = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=3600))


log = open(LOG, "a")
for cat, n in PLAN.items():
    rows = [json.loads(l) for l in open(f"{ITEMS}/{cat}.jsonl")][:n]
    for it in rows:
        tag = f"hx_{cat}_{it['id']}".replace("/", "_")[:80]
        body = {"model": "GLM-5.3-Flash", "messages": it["messages"], "temperature": 0,
                "max_tokens": 64 if cat == "long_context" else 256,
                "chat_template_kwargs": {"enable_thinking": False}}
        if it.get("tools"): body["tools"] = it["tools"]
        set_tag(tag)
        t0 = time.time()
        try:
            o = post(body); u = o.get("usage", {}); err = None
        except Exception as exc:                          # noqa: BLE001
            u, err = {}, repr(exc)[:200]
        rec = {"tag": tag, "category": cat, "prompt_tokens": u.get("prompt_tokens"),
               "completion_tokens": u.get("completion_tokens"), "s": round(time.time() - t0, 1), "error": err}
        log.write(json.dumps(rec) + "\n"); log.flush(); print(json.dumps(rec), flush=True)
set_tag("hx_end")                                     # flush the last item's buffer
set_tag(None)
print("CAPTURE_DONE", flush=True)
