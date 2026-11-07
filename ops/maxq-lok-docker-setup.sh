#!/bin/bash
# ONE-TIME: give lok its own rootless Docker engine on MaxQ (GPU via CDI), so HeteroFlash experiments run
# with no sudo at all afterwards. Containers then run as lok (no root). The owner's GLM serving under the
# 'dockerd' account is not changed.
#   ssh -t maxq sudo bash /srv/tools/heteroflash/ops/maxq-lok-docker-setup.sh
set -eu
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
U=lok; UID_L=$(id -u $U); RUN=/run/user/$UID_L
IMG=localhost/vllm-glm53f-exl3-sm120-turbo:r6
DATA=/srv/cache/heteroflash/docker-lok
CACHE=/srv/cache/heteroflash/vllm-cache
DD="sudo -u dockerd env XDG_RUNTIME_DIR=/run/user/1002 DOCKER_HOST=unix:///run/user/1002/docker.sock"
AS_LOK="sudo -u $U env XDG_RUNTIME_DIR=$RUN DBUS_SESSION_BUS_ADDRESS=unix:path=$RUN/bus PATH=/usr/share/docker.io/contrib:/usr/bin:/bin:/usr/sbin:/sbin HOME=/home/$U"
t() { date -u +%H:%M:%SZ; }

echo "$(t) 1. linger + memlock for $U's user services"
loginctl enable-linger $U
install -d /etc/systemd/system/user@$UID_L.service.d
printf '[Service]\nLimitMEMLOCK=infinity\n' > /etc/systemd/system/user@$UID_L.service.d/memlock.conf
systemctl daemon-reload
systemctl restart user@$UID_L.service
for i in $(seq 20); do [ -S $RUN/bus ] && break; sleep 1; done

echo "$(t) 2. rootless Docker for $U (data in $DATA, not in the home directory)"
install -d -o $U -g $U "$DATA" "$CACHE" /home/$U/.config/docker
printf '{\n  "data-root": "%s"\n}\n' "$DATA" > /home/$U/.config/docker/daemon.json
chown $U:$U /home/$U/.config/docker/daemon.json
$AS_LOK dockerd-rootless-setuptool.sh install 2>&1 | tail -5
$AS_LOK systemctl --user enable --now docker.service
for i in $(seq 30); do [ -S $RUN/docker.sock ] && break; sleep 1; done
LOKDOCKER="sudo -u $U env XDG_RUNTIME_DIR=$RUN DOCKER_HOST=unix://$RUN/docker.sock docker"
$LOKDOCKER info --format '{{.ServerVersion}} rootless={{.SecurityOptions}} root={{.DockerRootDir}}'

echo "$(t) 3. stop the dockerd-owned trace container (frees GPUs and port 8001)"
$DD docker rm -f glm53-trace >/dev/null 2>&1 || true

echo "$(t) 4. copy the vLLM image to $U's engine (~40 GB, a few minutes)"
$DD docker save $IMG | $LOKDOCKER load
echo "$(t) 5. copy the compiled-kernel cache"
cp -a /srv/cache/vllm-glm53f-exl3/. "$CACHE"/ 2>/dev/null || true
chown -R $U:$U "$CACHE"

echo "$(t) 6. GPU check as $U"
$LOKDOCKER run --rm --device nvidia.com/gpu=all --entrypoint nvidia-smi $IMG -L

echo "$(t) 7. remove the 24 h scoped sudo grant (no longer needed)"
rm -f /etc/sudoers.d/91-heteroflash-24h
systemctl stop heteroflash-revoke.timer 2>/dev/null || true
echo "$(t) done: lok's rootless Docker is ready (DOCKER_HOST=unix://$RUN/docker.sock) - tell Claude"
