#!/bin/bash
# Push HeteroFlash branch expert-correlation from the Spark to Forgejo via agentbox (adapted from heteroflash/r17/push_forgejo.sh). Forgejo only.
set -e
B=expert-correlation
ssh spark "cd /srv/vllm-research/work/HeteroFlash && git bundle create /tmp/hf-ecorr.bundle refs/heads/$B 2>/dev/null"
ssh spark 'cat /tmp/hf-ecorr.bundle' | sudo -n docker exec -i -u agents agentbox bash -c 'cat > /tmp/hf-ecorr.bundle'
sudo -n docker exec -u agents agentbox bash -lc "cd /home/agent/work/research/HeteroFlash && git fetch -q /tmp/hf-ecorr.bundle refs/heads/$B:refs/remotes/spark/$B && git push -q origin refs/remotes/spark/$B:refs/heads/$B && git ls-remote origin refs/heads/$B; rm -f /tmp/hf-ecorr.bundle"
ssh spark 'rm -f /tmp/hf-ecorr.bundle'
