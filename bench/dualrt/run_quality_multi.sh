#!/bin/bash
# 2026-11-17 frontier: several balanced-eval quality points on ONE split boot (agree mode: eager, 16-token batches, every
# position through the split path), switching the run-time knobs between passes. Each pass's capture is scored against
# the Q8_0 BF16 proxy (compare_q8.py, per-token arrays kept for paired comparisons), then its 13 GB capture is deleted.
#   run_quality_multi.sh TAG "name|DROP|ALT|GATES[|DEFER]" ...   (2026-11-18: DEFER = "none|all|l1,l2.. LATE"; env FO,
#   TOKENS = subset tokens file, HF_CHUNKS = its reference chunk ids; KEEPCAP=name keeps that pass's capture as the same-boot reference)      DROP = "K TAU RENORM" / "json {..} TAU RENORM"; ALT = layers|none
# env: HF_SPLIT_* as run_quality.sh (HF_SPLIT_DROP=1 needed for drop points), GPU_UTIL
set -u
TAG=$1; shift
H=/srv/vllm-research/work/HeteroFlash; OUT=$H/results/2026-11-17-dualrt; FO=${FO:-$H/results/2026-11-17-frontier}; IMG=local/hf-spark-native:1
CAP=/srv/vllm-research/work/cache/dualrt/cap_$TAG; REF=/srv/vllm-research/models/gguf/kld/base_q8_0_bal.kld
mkdir -p $FO; rm -rf $CAP; mkdir -p $CAP; chmod 777 $CAP
export HF_SPLIT=1 HF_SPLIT_CAPTURE=/cap HF_SPLIT_STATS=/out/${TAG}_stats.json MNBT=16
echo stop > $CAP/capture.cur
EXTRA_DOCKER="-v $CAP:/cap ${EXTRA_DOCKER:-}" bash $H/bench/dualrt/serve.sh $TAG agree || { docker rm -f hf-drt >/dev/null 2>&1; exit 1; }
hi() { curl -s -m 300 http://127.0.0.1:8095/v1/completions -H 'Content-Type: application/json' -d '{"model":"GLM-5.3-Flash","prompt":"Hello","max_tokens":4}' >/dev/null; }
rm -f $OUT/drt_load.done
if [ "${HF_SPLIT_NRES:-0}" -gt 0 ]; then
  touch $OUT/drt_load.req
  for i in $(seq 120); do hi; [ -s $OUT/drt_load.done ] && break; sleep 2; done
  echo "$(date -u +%H:%M:%SZ) $(cat $OUT/drt_load.done 2>/dev/null || echo 'pool load did not report')"
fi
for P in "$@"; do
  IFS='|' read NAME DROP ALT GATES DEFER <<< "$P"
  echo "$DROP" > $OUT/drt_drop.req; echo "$ALT" > $OUT/drt_alt.req; echo "${GATES:-1 1 0}" > $OUT/drt_gates.req; echo "${DEFER:-none 0}" > $OUT/drt_defer.req
  for i in $(seq 60); do hi; [ -e $OUT/drt_drop.req ] || [ -e $OUT/drt_alt.req ] || [ -e $OUT/drt_gates.req ] || [ -e $OUT/drt_defer.req ] || break; done
  touch $OUT/${TAG}_stats.json.req; sleep 3; cp $OUT/${TAG}_stats.json $FO/q_${TAG}_${NAME}_stats0.json
  echo "$(date -u +%H:%M:%SZ) pass $NAME: drop [$DROP] alt [$ALT] gates [${GATES:-1 1 0}] defer [${DEFER:-none 0}]"
  python3 -I $H/bench/dualrt/quality_client.py ${TOKENS:-$H/data/q_eval_balanced_v1.tokens.json} http://127.0.0.1:8095/v1/completions $CAP > $FO/q_${TAG}_${NAME}_client.log 2>&1
  echo stop > $CAP/capture.cur; hi; sleep 3
  touch $OUT/${TAG}_stats.json.req; sleep 3; cp $OUT/${TAG}_stats.json $FO/q_${TAG}_${NAME}_stats1.json
  mkdir -p $CAP/$NAME; mv $CAP/logits_*.npy $CAP/$NAME/ 2>/dev/null; cp $CAP/capture_rows.json $FO/q_${TAG}_${NAME}_capture_rows.json 2>/dev/null
  echo "$(date -u +%H:%M:%SZ) $NAME logits files: $(ls $CAP/$NAME/logits_*.npy 2>/dev/null | wc -l)"
  docker run --rm -e HF_CHUNKS=${HF_CHUNKS:-} -e HF_PERTOKEN=/h/${FO#$H/}/q_${TAG}_${NAME}_pertoken.npz -v $H:/h -v $CAP:/cap -v $(dirname $REF):/ref:ro \
    --entrypoint python3 $IMG /h/bench/dualrt/compare_q8.py /ref/$(basename $REF) /cap/$NAME /h/data/q_eval_balanced_v1.manifest.json \
    /h/${FO#$H/}/q_${TAG}_${NAME}_vs_q8_bal.json 2>&1 | tail -2
  RK=$CAP/../keep_${TAG}_${KEEPCAP:-none}
  if [ -d $RK ] && [ "${KEEPCAP:-}" != "$NAME" ]; then
    docker run --rm -v $H:/h -v $CAP:/cap -v $RK:/refcap:ro --entrypoint python3 $IMG /h/bench/dualrt/kld_pair.py /refcap /cap/$NAME \
      /h/${FO#$H/}/q_${TAG}_${NAME}_vs_${KEEPCAP}.json 2>&1 | tail -1
  fi
  [ "${KEEPCAP:-}" = "$NAME" ] && mv $CAP/$NAME $CAP/../keep_${TAG}_$NAME || rm -rf $CAP/$NAME
done
docker logs hf-drt > $OUT/${TAG}_serve.log 2>&1; cp $OUT/${TAG}_serve.log $FO/; docker rm -f hf-drt >/dev/null 2>&1
rm -rf $CAP
echo "$(date -u +%H:%M:%SZ) DONE $TAG"
