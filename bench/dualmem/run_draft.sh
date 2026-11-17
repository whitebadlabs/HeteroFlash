#!/bin/bash
# Split MTP draft passes (T = 1): 3 draft passes x 2 phases per "cycle". Phase A = MTP attention + eh_proj half (q_a/kv_a
# replicated), phase B = lm_head vocab half + shared-expert half + one FP8 expert (byte stand-in for 4 IQ2 experts).
# FP8 byte stand-ins for the BF16 kv_b / eh_proj halves. usage: run_draft.sh <port> <tag>
port=$1; tag=$2
A=1536x4096,512x4096,8192x1536,16384x1024,4096x8192,4096x8192
B=77440x4096,1024x4096,1024x4096,2048x4096,2048x4096,4096x2048,4096x1024
A2=1536x4096,512x4096,16384x1536,32768x1024,4096x16384,4096x16384
B2=154880x4096,2048x4096,2048x4096,2048x4096,2048x4096,4096x2048,2048x4096,2048x4096,4096x2048,4096x2048
IMG="docker run --rm --name hfdual-cli --network host --user \$(id -u):\$(id -g) -e HOME=/tmp --device nvidia.com/gpu=all -v /srv/vllm-research/work/HeteroFlash/bench/dualmem:/b:ro --entrypoint python3 vllm/vllm-openai:v0.32.0 /b/dual_cli.py"
ssh spark "$IMG --T 1 --copies 3 --A $A2 --B $B2 --layers 3 --cycles 50 --mode gpu_sync --tag ${tag}_gpu_only_same_bytes 2>&1 | grep '^{'"
ssh spark "$IMG --T 1 --copies 3 --A $A --B $B --layers 3 --cycles 50 --mode gpu_sync --tag ${tag}_gpu_half 2>&1 | grep '^{'"
ssh maxq "cd /srv/cache/heteroflash/dualmem && rm -f srv_$tag.log; tmux -L hfdual new -d -s srv_$port 'OMP_NUM_THREADS=32 OMP_PLACES=cores OMP_PROC_BIND=close OMP_WAIT_POLICY=active timeout 600 ./dual_srv WS_LINK_IP $port 1 3 $A $B 2> srv_$tag.log'"
until ssh maxq "grep -q ready /srv/cache/heteroflash/dualmem/srv_$tag.log 2>/dev/null"; do sleep 2; done
ssh spark "$IMG --peer WS_LINK_IP --port $port --T 1 --copies 3 --A $A --B $B --layers 3 --cycles 50 --mode full --tag ${tag}_split 2>&1 | grep '^{'"
sleep 2; ssh maxq "cat /srv/cache/heteroflash/dualmem/srv_$tag.log"
