#!/usr/bin/env python3
"""Client for cycle_anatomy.sh: acceptance + speed per workload (spec-decode counters before/after each request),
a profiled 64-token decode, cold 45K and cached 45K+32 turns. Prints one JSON object per line."""
import json, os, random, re, time, urllib.request
U = os.environ.get("URL", "http://127.0.0.1:8095"); N = int(os.environ["MTP_N"]); PROF = os.environ.get("PROFILE", "1") == "1"
def post(path, body, timeout=1800):
    r = urllib.request.Request(U + path, data=json.dumps(dict(body, model="GLM-5.3-Flash")).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); o = json.load(urllib.request.urlopen(r, timeout=timeout)); return o, time.perf_counter() - t0
def ctl(p): urllib.request.urlopen(urllib.request.Request(U + p, data=b"", method="POST"))
def metrics():
    txt = urllib.request.urlopen(U + "/metrics").read().decode(); acc = {}; tot = {}
    for line in txt.splitlines():
        m = re.match(r'vllm:spec_decode_num_accepted_tokens_per_pos_total\{.*position="(\d+)".*\} ([0-9.e+]+)', line)
        if m: acc[int(m.group(1))] = float(m.group(2))
        m = re.match(r'vllm:spec_decode_num_(drafts|draft_tokens|accepted_tokens)_total\{.*\} ([0-9.e+]+)', line)
        if m: tot[m.group(1)] = float(m.group(2))
    return acc, tot
def measured(name, fn):
    a0, t0 = metrics(); ntok, dt = fn(); a1, t1 = metrics()
    d = t1.get("drafts", 0) - t0.get("drafts", 0); acc = t1.get("accepted_tokens", 0) - t0.get("accepted_tokens", 0)
    per = [a1.get(i, 0) - a0.get(i, 0) for i in range(N)]
    out = {"workload": name, "tokens": ntok, "s": round(dt, 3), "tok_s": round(ntok / dt, 2), "drafts": d,
           "committed_per_step": round(1 + acc / max(d, 1), 3), "per_pos_accept": [round(p / max(d, 1), 3) for p in per],
           "ms_per_cycle": round(1e3 * dt / max(d, 1), 2) if d else None}
    print(json.dumps(out), flush=True); return out
chats = {
  "prose": "Write a detailed essay about the history of river navigation and its effect on European trade.",
  "code": "Write a Python module implementing an LRU cache with TTL expiry, thread safety, and unit tests using pytest.",
  "chat": "I'm planning a 3-day trip to Kyoto in November. Suggest an itinerary with food recommendations.",
  "reasoning": "A train leaves at 9:40 and travels 230 km at 82 km/h, then 140 km at 61 km/h with a 12 minute stop between. When does it arrive? Explain step by step.",
}
post("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8})
CACHED_ONLY = os.environ.get("CACHED_ONLY") == "1"
if CACHED_ONLY: PROF = False
# (1) the serve_eval essay benchmark (completions API, ignore_eos), 3 repeats, decode time = 257-token minus 1-token request
for rep in range(0 if CACHED_ONLY else 3):
    def essay():
        _, d1 = post("/v1/completions", {"prompt": "Write a long essay about rivers.", "max_tokens": 1, "ignore_eos": True, "temperature": 0})
        _, d2 = post("/v1/completions", {"prompt": "Write a long essay about rivers.", "max_tokens": 257, "ignore_eos": True, "temperature": 0})
        return 256, d2 - d1
    measured(f"essay_completions_rep{rep}", essay)
# (2) chat workloads, 400 tokens, temperature 0
for name, p in ([] if CACHED_ONLY else chats.items()):
    def run(p=p):
        o, dt = post("/v1/chat/completions", {"messages": [{"role": "user", "content": p}], "max_tokens": 400, "temperature": 0})
        return o["usage"]["completion_tokens"], dt
    measured(name, run)
# (3) profiled decode: 64 tokens of the essay prompt
if PROF:
    ctl("/start_profile")
    def prof():
        o, dt = post("/v1/completions", {"prompt": "Write a long essay about rivers.", "max_tokens": 64, "ignore_eos": True, "temperature": 0})
        return 64, dt
    measured("profiled_decode64", prof)
    ctl("/stop_profile")
# (4) cold 45K, then cached 45K+32 (first one profiled), 4 repeats
random.seed(7); ids = [random.randrange(1000, 60000) for _ in range(45000)]
_, cold = post("/v1/completions", {"prompt": ids, "max_tokens": 1, "temperature": 0})
print(json.dumps({"cold_45k_s": round(cold, 2)}), flush=True); time.sleep(2)
for rep in range(int(os.environ.get("CACHED_REPS", "4"))):
    if PROF and rep == 0: ctl("/start_profile")
    o, c = post("/v1/completions", {"prompt": ids + [random.randrange(1000, 60000) for _ in range(32)], "max_tokens": 1, "temperature": 0})
    if PROF and rep == 0: ctl("/stop_profile")
    print(json.dumps({"cached_45k_plus32_s": round(c, 3), "rep": rep, "profiled": PROF and rep == 0,
                      "cached_tokens": ((o.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens")}), flush=True)
