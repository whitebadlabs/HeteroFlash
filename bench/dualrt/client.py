#!/usr/bin/env python3
"""Bet 1 runtime client (2026-11-17): batch-1 decode speed + MTP acceptance per workload, greedy (temperature 0).
Workloads = the 11-16 cycle_anatomy set (essay completions x REPS, prose/code/chat/reasoning chat 400 tokens) plus a
fixed Hermes-like set (first 2 items of 5 Hermes-suite categories, their own tools/messages, 300 tokens).
Prints one JSON line per request; output token text is saved for free-running agreement checks."""
import json, os, re, sys, time, urllib.request
U = os.environ.get("URL", "http://127.0.0.1:8095"); N = int(os.environ.get("MTP_N", "3")); REPS = int(os.environ.get("REPS", "3"))
HS = os.environ.get("HERMES", "/srv/vllm-research/work/cache/hermes_suite/items")
def post(path, body, timeout=3600):
    r = urllib.request.Request(U + path, data=json.dumps(dict(body, model="GLM-5.3-Flash")).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); o = json.load(urllib.request.urlopen(r, timeout=timeout)); return o, time.perf_counter() - t0
def metrics():
    txt = urllib.request.urlopen(U + "/metrics").read().decode(); acc = {}; tot = {}
    for line in txt.splitlines():
        m = re.match(r'vllm:spec_decode_num_accepted_tokens_per_pos_total\{.*position="(\d+)".*\} ([0-9.e+]+)', line)
        if m: acc[int(m.group(1))] = float(m.group(2))
        m = re.match(r'vllm:spec_decode_num_(drafts|draft_tokens|accepted_tokens)_total\{.*\} ([0-9.e+]+)', line)
        if m: tot[m.group(1)] = float(m.group(2))
    return acc, tot
def measured(name, fn, rep):
    a0, t0 = metrics(); ntok, dt, text = fn(); a1, t1 = metrics()
    d = t1.get("drafts", 0) - t0.get("drafts", 0); acc = t1.get("accepted_tokens", 0) - t0.get("accepted_tokens", 0)
    per = [a1.get(i, 0) - a0.get(i, 0) for i in range(N)]
    out = {"workload": name, "rep": rep, "tokens": ntok, "s": round(dt, 3), "tok_s": round(ntok / dt, 2), "drafts": d,
           "committed_per_step": round(1 + acc / max(d, 1), 3), "per_pos_accept": [round(p / max(d, 1), 3) for p in per],
           "ms_per_cycle": round(1e3 * dt / max(d, 1), 2) if d else None, "text": text}
    print(json.dumps(out), flush=True); return out
chats = {
  "prose": "Write a detailed essay about the history of river navigation and its effect on European trade.",
  "code": "Write a Python module implementing an LRU cache with TTL expiry, thread safety, and unit tests using pytest.",
  "chat": "I'm planning a 3-day trip to Kyoto in November. Suggest an itinerary with food recommendations.",
  "reasoning": "A train leaves at 9:40 and travels 230 km at 82 km/h, then 140 km at 61 km/h with a 12 minute stop between. When does it arrive? Explain step by step.",
}
hermes = []
_a, _b = (int(v) for v in os.environ.get("HERMES_SLICE", "0:2").split(":"))   # calibration runs use a disjoint slice
for cat in ["tool_calling", "multi_turn", "coding", "prose", "reasoning"]:
    for line in open(os.path.join(HS, cat + ".jsonl")).readlines()[_a:_b]:
        it = json.loads(line); hermes.append((it["id"], it["messages"], it.get("tools")))
post("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8})
sel = os.environ.get("ONLY", "essay,chat,hermes").split(",")
for rep in range(REPS):
    if "essay" in sel:
        def essay():
            _, d1 = post("/v1/completions", {"prompt": "Write a long essay about rivers.", "max_tokens": 1, "ignore_eos": True, "temperature": 0})
            o, d2 = post("/v1/completions", {"prompt": "Write a long essay about rivers.", "max_tokens": 257, "ignore_eos": True, "temperature": 0})
            return 256, d2 - d1, o["choices"][0]["text"]
        measured("essay_completions", essay, rep)
    if "chat" in sel:
        for name, p in chats.items():
            def run(p=p):
                o, dt = post("/v1/chat/completions", {"messages": [{"role": "user", "content": p}], "max_tokens": 400, "temperature": 0})
                m = o["choices"][0]["message"]; return o["usage"]["completion_tokens"], dt, (m.get("reasoning_content") or m.get("reasoning") or "") + "|" + (m.get("content") or "")
            measured(name, run, rep)
    if "hermes" in sel:
        for hid, msgs, tools in hermes:
            def run(msgs=msgs, tools=tools):
                b = {"messages": msgs, "max_tokens": 300, "temperature": 0}
                if tools: b["tools"] = tools
                o, dt = post("/v1/chat/completions", b)
                m = o["choices"][0]["message"]
                return o["usage"]["completion_tokens"], dt, (m.get("reasoning_content") or m.get("reasoning") or "") + "|" + (m.get("content") or "") + "|" + json.dumps(m.get("tool_calls") or [])
            measured("hermes:" + hid, run, rep)
