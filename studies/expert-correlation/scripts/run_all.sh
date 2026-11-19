#!/bin/bash
# Step 4 + 5 for both models on CPU (NAS: 4 cores, ~1.5 h). Rerunning resumes from out/<model>.json.ckpt/.
# Runs as transient systemd user units (ecorr-glm, ecorr-qwen): setsid/nohup children were killed by a Claude
# session restart (cgroup kill) on 2026-11-18. DATA = the working dir holding data/ (traces are not in git).
set -e
DATA=${DATA:-NAS_PATH}
HERE=$(cd "$(dirname "$0")/.." && pwd)
run() {  # unit-name log args...
  local u=$1 log=$2; shift 2
  systemd-run --user --unit="$u" --collect -p WorkingDirectory="$DATA" -E PYTHONPATH="$DATA/pylib" \
    -E OPENBLAS_NUM_THREADS=2 -E OMP_NUM_THREADS=2 \
    /bin/bash -c "exec /usr/bin/python3 -u $HERE/scripts/analyze.py $* >> $log 2>&1"
}
run ecorr-glm out/glm_run3.log --reduced data/reduced/m1_top8_v2 --out out/glm.json
run ecorr-qwen out/qwen_run3.log --reduced data/qwen36/routes/qwen36_v1 --out out/qwen.json --train-cap 20000 --eval-cap 20000
