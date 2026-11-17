#!/bin/bash
# usage: run_proto.sh <port> <threads> <tag> [cycles]  -- MaxQ dual_srv + Spark dual_cli (full mode), f = 0.5 shapes
port=$1; nt=$2; tag=$3; cyc=${4:-20}
A=4096x4096,4096x4096,4096x4096,4096x4096; E=2048x4096,2048x4096,4096x2048; B=$E,$E,$E,$E,$E,$E,1024x4096,1024x4096,4096x1024
ssh maxq "cd /srv/cache/heteroflash/dualmem && rm -f srv_$tag.log; tmux -L hfdual new -d -s srv_$port 'OMP_NUM_THREADS=$nt OMP_PLACES=cores OMP_PROC_BIND=close OMP_WAIT_POLICY=active timeout 600 ./dual_srv WS_LINK_IP $port 4 10 $A $B 2> srv_$tag.log'"
until ssh maxq "grep -q ready /srv/cache/heteroflash/dualmem/srv_$tag.log 2>/dev/null"; do sleep 2; done
ssh spark "cd /srv/vllm-research/work/HeteroFlash && timeout 400 docker run --rm --name hfdual-cli --network host --user \$(id -u):\$(id -g) -e HOME=/tmp --device nvidia.com/gpu=all -v \$PWD/bench/dualmem:/b:ro --entrypoint python3 vllm/vllm-openai:v0.32.0 /b/dual_cli.py --peer WS_LINK_IP --port $port --T 4 --copies 10 --A $A --B $B --cycles $cyc --mode full --tag $tag 2>&1 | grep '^{'"
sleep 2; ssh maxq "cat /srv/cache/heteroflash/dualmem/srv_$tag.log"
