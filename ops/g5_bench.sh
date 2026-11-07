#!/bin/bash
# G5 step 1: existing MTP drafter vs draft length, on the Control (serveN: serving config, CUDA graphs) and on the
# target treatment (mtpN: eager + hook, top-6 original-top-8 normalisation + FP8 fixed weights).
# Runs as lok; uses ops/lok/hf-run (sudo-less, lok rootless Docker). Results: /srv/cache/heteroflash/g5/<variant>/
set -u
T=/srv/tools/heteroflash
OUT=/srv/cache/heteroflash/g5
CTL=/srv/cache/heteroflash/traces/control.json
P=/srv/cache/heteroflash/prompts/m1_calib_v1.jsonl
mkdir -p "$OUT"
set_ctl() {  # $1 = json
  printf '%s' "$1" > "$CTL.tmp" && mv -f "$CTL.tmp" "$CTL"
}
PLAIN='{"capture": false, "k_active": 8, "k_ref": "active", "routed_class": null, "diag": false, "fixed_quant": null, "capture_inputs": null, "dump_source": null, "shared_class": "Glm5NextMLP", "ratio_override": null}'
TARGET='{"capture": false, "k_active": 6, "k_ref": 8, "routed_class": "MoERunner", "diag": false, "fixed_quant": {"fmt": "fp8", "include": "(self_attn|shared_experts|lm_head|mlp\\.(gate_up_proj|down_proj))", "exclude": "(mlp\\.experts\\.|\\.gate$|indexer|norm|embed|visual|vision|mtp)"}, "capture_inputs": null, "dump_source": null, "shared_class": "Glm5NextMLP", "ratio_override": null}'
for V in ${G5_VARIANTS:-serve1 serve3 serve5 serve7 mtp1 mtp3 mtp5 mtp7}; do
  D=$OUT/$V; mkdir -p "$D"
  [ -s "$D/decode_mixed.jsonl" ] && grep -q '"type": "summary"' "$D/decode_mixed.jsonl" && { echo "$V already done"; continue; }
  case $V in mtp*) set_ctl "$TARGET" ;; *) set_ctl "$PLAIN" ;; esac
  echo "$(date -u +%H:%M:%SZ) restart $V"
  /srv/tools/heteroflash/ops/lok/hf-run "$V" > "$D/restart.log" 2>&1 || { echo "restart $V failed"; tail -20 "$D/restart.log"; continue; }
  case $V in mtp*) sleep 5; curl -fs http://127.0.0.1:8001/v1/completions -H 'Content-Type: application/json' -d '{"model":"GLM-5.3-Flash","prompt":"warm","max_tokens":4}' >/dev/null; sleep 3 ;; esac
  python3 -I $T/ttft_bench.py decode --base http://127.0.0.1:8001 --prompts $P --domains writing_chat --split holdout --limit 14 \
    --max-tokens 256 --temperature 1.0 --out "$D/decode_prose.jsonl" > "$D/prose.log" 2>&1
  python3 -I $T/ttft_bench.py decode --base http://127.0.0.1:8001 --prompts $P --domains writing_chat knowledge_qa math code multilingual instruction_following \
    --split holdout --limit 36 --max-tokens 256 --temperature 1.0 --out "$D/decode_mixed.jsonl" > "$D/mixed.log" 2>&1
  ls /srv/cache/heteroflash/traces/raw/fixed_quant_*.json >/dev/null 2>&1 && cp -f $(ls -t /srv/cache/heteroflash/traces/raw/fixed_quant_*.json | head -1) "$D/fixed_quant_applied.json" 2>/dev/null
  echo "$(date -u +%H:%M:%SZ) $V done: $(tail -1 "$D/decode_prose.jsonl" | head -c 300)"
done
set_ctl "$PLAIN"
/srv/tools/heteroflash/ops/lok/hf-run trace > "$OUT/final_restart.log" 2>&1
echo G5_BENCH_DONE
