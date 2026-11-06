#!/bin/bash
# One-shot baseline window on MaxQ (run after the M1 trace run has finished):
#   ssh -t maxq sudo bash /srv/tools/heteroflash/ops/maxq-baseline-window.sh
# 1. stop glm53-trace   2. start the ORIGINAL serving config (graphs, MTP, prefix cache) on :8000
# 3. as lok: cold 45K TTFT, cached 45K+{16..256} TTFT, prose decode TPOT/ITL/MTP acceptance
# 4. stop serving and restart glm53-trace with the current hook (v3) for M2.  Noo Noo stays on Qwen (LiteLLM unchanged).
set -u
OPS=/srv/tools/heteroflash/ops
OUT=/srv/cache/heteroflash/baseline/$(date -u +%Y%m%dT%H%M%SZ)
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
mkdir -p "$OUT"; chown -R lok:lok /srv/cache/heteroflash/baseline
exec > >(tee -a "$OUT/window.log") 2>&1
t() { date -u +%H:%M:%SZ; }
DD="sudo -u dockerd env XDG_RUNTIME_DIR=/run/user/1002 DOCKER_HOST=unix:///run/user/1002/docker.sock"
if pgrep -f run_trace_client.py >/dev/null; then echo "M1 trace client still running - nothing changed"; exit 1; fi
echo "$(t) 1. stop glm53-trace"
(cd "$OPS/maxq-trace" && $DD docker compose -f compose.run.yaml down) >/dev/null 2>&1
echo "$(t) 2. start serving glm53 (boot ~13 min)"
$DD sh -c 'cd /srv/dockerd/services/glm53 && docker compose up -d' || { echo "serving start failed"; exit 1; }
for i in $(seq 150); do sleep 20; curl -fs -o /dev/null http://127.0.0.1:8000/health && break; done
curl -fs -o /dev/null http://127.0.0.1:8000/health || { echo "serving not healthy"; $DD docker logs --tail 40 glm53; exit 1; }
$DD docker inspect -f '{{.Image}}' glm53 > "$OUT/serving_image_id.txt"
echo "$(t)    healthy; warm-up"
curl -fs http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' -d '{"model":"GLM-5.3-Flash","prompt":"Hello","max_tokens":16}' >/dev/null
echo "$(t) 3a. TTFT cold/cached (45K)"
sudo -u lok python3 -I /srv/tools/heteroflash/ttft_bench.py --base http://127.0.0.1:8000 --corpus /srv/cache/heteroflash/prompts/longctx_corpus.txt \
  --n 45000 --suffixes 32 16 64 128 256 --repeats 10 --out "$OUT/ttft.jsonl" | tail -40
echo "$(t) 3b. prose decode TPOT/ITL/MTP acceptance (holdout writing_chat, 512 tokens, temperature 1)"
sudo -u lok python3 -I /srv/tools/heteroflash/ttft_bench.py decode --base http://127.0.0.1:8000 \
  --prompts /srv/cache/heteroflash/prompts/m1_calib_v1.jsonl --domains writing_chat --split holdout --limit 20 \
  --max-tokens 512 --temperature 1.0 --out "$OUT/decode_prose.jsonl" | tail -25
sudo -u lok python3 -I /srv/tools/heteroflash/ttft_bench.py decode --base http://127.0.0.1:8000 \
  --prompts /srv/cache/heteroflash/prompts/m1_calib_v1.jsonl --domains writing_chat knowledge_qa math code multilingual instruction_following \
  --split holdout --limit 60 --max-tokens 256 --temperature 1.0 --out "$OUT/decode_mixed.jsonl" | tail -25
echo "$(t) 4. stop serving, restart trace server (hook v3)"
$DD sh -c 'cd /srv/dockerd/services/glm53 && docker compose stop' >/dev/null 2>&1
bash "$OPS/maxq-trace-restart.sh"
chown -R lok:lok "$OUT"
echo "$(t) done: results in $OUT - tell Claude"
