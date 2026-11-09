#!/bin/bash
# Audit run (one boot): MTP acceptance per draft position, decode speed, and torch-profiler traces of
# (c) an 8K-token prefill + 32 decode steps and (d) a cached 45K + 32 turn. Runs ON the Spark; stops qfn and restarts it.
#   prof_run.sh MODEL_DIR TAG MTP_N
set -u
M=$1; TAG=$2; N=${3:-5}
H=/srv/vllm-research/work/HeteroFlash; IMG=local/hf-spark-native:1
OUT=$H/results/2026-11-09-audit/$TAG; mkdir -p $OUT
PROF=/srv/vllm-research/work/cache/prof/$TAG; rm -rf $PROF; mkdir -p $PROF
t() { date -u +%H:%M:%SZ; }
SPEC_DEF="{\"method\": \"mtp\", \"num_speculative_tokens\": $N, \"attention_backend\": \"B12X\"}"
docker stop qfn >/dev/null 2>&1
for i in $(seq 40); do [ "$(free -g | awk '/^Mem:/{print $7}')" -ge 100 ] && break; sleep 3; done
docker rm -f hf-r3 >/dev/null 2>&1
docker run -d --name hf-r3 --gpus all --ipc=host -p 127.0.0.1:8095:8095 -v $M:/m:ro -v $PROF:/prof -v /srv/vllm-research/work/cache/control_chat_template.jinja:/tmpl.jinja:ro \
  -v /srv/vllm-research/work/cache/b12x:/cache/b12x -e B12X_COMPILE_CACHE_DIR=/cache/b12x \
  -e VLLM_KV_CACHE_LAYOUT=BLHNC -e B12X_LUT_E4M3_SMEM=0 -e HF_HUB_OFFLINE=1 -e VLLM_ENGINE_READY_TIMEOUT_S=3600 ${EXTRA_ENV:-} \
  --entrypoint vllm $IMG serve /m --port 8095 --host 0.0.0.0 --served-model-name GLM-5.3-Flash \
  --quantization modelopt_mixed --dtype bfloat16 --max-num-seqs 1 --gpu-memory-utilization 0.90 --trust-remote-code \
  --block-size 256 --kv-cache-dtype fp8 --mamba-ssm-cache-dtype bfloat16 --attention-backend B12X --linear-backend b12x --moe-backend b12x \
  --max-model-len 65536 --max-num-batched-tokens 8192 \
  --speculative-config "${SPEC_JSON:-$SPEC_DEF}" ${EXTRA_ARGS:-} \
  --profiler-config '{"profiler": "torch", "torch_profiler_dir": "/prof"}' \
  --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice --chat-template /tmpl.jinja \
  --default-chat-template-kwargs.reasoning_effort=high >/dev/null
for i in $(seq 240); do sleep 15; docker ps -q --filter name=hf-r3 | grep -q . || break; curl -fs http://127.0.0.1:8095/health >/dev/null && break; done
docker logs hf-r3 > $OUT/serve.log 2>&1
if ! curl -fs http://127.0.0.1:8095/health >/dev/null; then echo "$(t) NOT HEALTHY"; grep -iE "Error" $OUT/serve.log | tail -5; docker rm -f hf-r3 >/dev/null; docker start qfn >/dev/null; exit 1; fi
echo "$(t) healthy"
MTP_N=$N SKIP8K=${SKIP8K:-} python3 -I - <<'PY' | tee $OUT/acceptance_speed.txt
import json, os, re, time, urllib.request, random
U = "http://127.0.0.1:8095"
N = int(os.environ["MTP_N"])
def req(body):
    r = urllib.request.Request(U + "/v1/completions", data=json.dumps(dict(body, model="GLM-5.3-Flash")).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); o = json.load(urllib.request.urlopen(r, timeout=1800)); return o, time.perf_counter() - t0
def chat(msgs, n, temp=0.0):
    r = urllib.request.Request(U + "/v1/chat/completions", data=json.dumps({"model": "GLM-5.3-Flash", "messages": msgs, "max_tokens": n, "temperature": temp}).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); o = json.load(urllib.request.urlopen(r, timeout=1800)); return o, time.perf_counter() - t0
def metrics():
    txt = urllib.request.urlopen(U + "/metrics").read().decode()
    acc = {}; tot = {}
    for line in txt.splitlines():
        m = re.match(r'vllm:spec_decode_num_accepted_tokens_per_pos_total\{.*position="(\d+)".*\} ([0-9.e+]+)', line)
        if m: acc[int(m.group(1))] = float(m.group(2))
        m = re.match(r'vllm:spec_decode_num_(drafts|draft_tokens|accepted_tokens)_total\{.*\} ([0-9.e+]+)', line)
        if m: tot[m.group(1)] = float(m.group(2))
    return acc, tot
prompts = {
  "prose": [{"role": "user", "content": "Write a detailed essay about the history of river navigation and its effect on European trade."}],
  "code": [{"role": "user", "content": "Write a Python module implementing an LRU cache with TTL expiry, thread safety, and unit tests using pytest."}],
  "chat": [{"role": "user", "content": "I'm planning a 3-day trip to Kyoto in November. Suggest an itinerary with food recommendations."}],
  "reasoning": [{"role": "user", "content": "A train leaves at 9:40 and travels 230 km at 82 km/h, then 140 km at 61 km/h with a 12 minute stop between. When does it arrive? Explain step by step."}],
  "tool": [{"role": "system", "content": "You are an agent. When you need information, reply ONLY with a JSON object {\"tool\": name, \"args\": {...}}. Tools: search(query), read_file(path), run(cmd)."},
           {"role": "user", "content": "Find out which Python files in the repo import numpy and summarise what each does."}],
}
chat([{"role": "user", "content": "hi"}], 8)
res = {}
for name, msgs in prompts.items():
    a0, t0 = metrics()
    o, dt = chat(msgs, 400)
    a1, t1 = metrics()
    ntok = o["usage"]["completion_tokens"]
    drafts = t1.get("drafts", 0) - t0.get("drafts", 0); acc = t1.get("accepted_tokens", 0) - t0.get("accepted_tokens", 0)
    per = [a1.get(i, 0) - a0.get(i, 0) for i in range(N)]
    res[name] = {"tokens": ntok, "s": round(dt, 2), "tok_s": round(ntok / dt, 2), "drafts": drafts,
                 "mean_accepted_per_draft": round(acc / max(drafts, 1), 3), "committed_per_step": round(1 + acc / max(drafts, 1), 3),
                 "per_pos_accept_rate": [round(p / max(drafts, 1), 3) for p in per]}
    print(json.dumps({name: res[name]}), flush=True)
json.dump(res, open("/dev/stdout", "w")) if False else None
# (b2) decode-only profile: short prompt, 64 tokens
urllib.request.urlopen(urllib.request.Request(U + "/start_profile", data=b"", method="POST"))
o, dt = chat(prompts["prose"], 64)
urllib.request.urlopen(urllib.request.Request(U + "/stop_profile", data=b"", method="POST"))
print(json.dumps({"profiled_decode64_s": round(dt, 3), "tokens": o["usage"]["completion_tokens"]}), flush=True)
if os.environ.get("SKIP8K"):
    pass
else:
  # (c) profiled 8K prefill + 32 decode steps
  pass
# (c) profiled 8K prefill + 32 decode steps
random.seed(11); ids8 = [random.randrange(1000, 60000) for _ in range(8192)]
urllib.request.urlopen(urllib.request.Request(U + "/start_profile", data=b"", method="POST"))
o, dt = req({"prompt": ids8, "max_tokens": 32, "temperature": 0, "ignore_eos": True})
urllib.request.urlopen(urllib.request.Request(U + "/stop_profile", data=b"", method="POST"))
print(json.dumps({"profiled_8k_plus32_s": round(dt, 3)}), flush=True)
# (d) cold 45K then profiled cached 45K+32
random.seed(7); ids = [random.randrange(1000, 60000) for _ in range(45000)]
o, cold = req({"prompt": ids, "max_tokens": 1, "temperature": 0})
print(json.dumps({"cold_45k_s": round(cold, 2)}), flush=True)
time.sleep(3)
urllib.request.urlopen(urllib.request.Request(U + "/start_profile", data=b"", method="POST"))
o, cached = req({"prompt": ids + [random.randrange(1000, 60000) for _ in range(32)], "max_tokens": 1, "temperature": 0})
urllib.request.urlopen(urllib.request.Request(U + "/stop_profile", data=b"", method="POST"))
print(json.dumps({"cached_45k_plus32_s_profiled": round(cached, 3)}), flush=True)
for rep in range(3):
    o, cached2 = req({"prompt": ids + [random.randrange(1000, 60000) for _ in range(32)], "max_tokens": 1, "temperature": 0})
    print(json.dumps({"cached_45k_plus32_s_unprofiled": round(cached2, 3), "rep": rep, "cached_tokens": (o.get("usage") or {}).get("prompt_tokens_details")}), flush=True)
PY
if [ -n "${HERMES:-}" ]; then   # Hermes-style workload: suite slice, spec-decode counters before/after
  m() { curl -s http://127.0.0.1:8095/metrics | awk '/^vllm:spec_decode_num_drafts_total/{d=$2} /^vllm:spec_decode_num_accepted_tokens_total/{a=$2} END{print d+0, a+0}'; }
  read d0 a0 < <(m); S=/srv/vllm-research/work/cache/hermes_suite
  docker run --rm --network host -e PYTHONPATH=/s/pylib -e NLTK_DATA=/s/pylib/nltk_data -v $S:/s -w /s --entrypoint python3 $IMG \
    run_suite.py --endpoint http://127.0.0.1:8095 --model GLM-5.3-Flash --out answers_prof_$TAG.jsonl --thinking off --limit 8 \
    --categories reasoning coding tool_calling instruction_following multi_turn 2>&1 | tail -1
  read d1 a1 < <(m)
  docker run --rm --network none -e PYTHONPATH=/s/pylib -e NLTK_DATA=/s/pylib/nltk_data -v $S:/s -w /s --entrypoint python3 $IMG \
    score.py answers_prof_$TAG.jsonl 2>&1 | sed -n '/-- ops/,$p' | tee $OUT/hermes_slice_ops.txt
  python3 -c "d=$d1-$d0; a=$a1-$a0; print({'hermes_slice_drafts': d, 'accepted': a, 'committed_per_step': round(1+a/max(d,1),3)})" | tee -a $OUT/hermes_slice_ops.txt
fi
sleep 10; docker logs hf-r3 > $OUT/serve.log 2>&1
docker rm -f hf-r3 >/dev/null 2>&1; docker start qfn >/dev/null 2>&1
echo "$(t) traces: $(find $PROF -type f | wc -l) files, $(du -sh $PROF | cut -f1)"; echo "$(t) PROF_DONE; qfn $(docker inspect -f '{{.State.Status}}' qfn)"
