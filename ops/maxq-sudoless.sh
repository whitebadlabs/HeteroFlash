#!/bin/bash
# ONE-TIME on the workstation: give lok passwordless sudo (owner's request: no more sudo prompts), then apply the
# direct-link network setup.  Undo the sudo grant: sudo rm /etc/sudoers.d/90-lok-nopasswd
set -eu
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
printf '%s\n' '# lok: passwordless sudo (HeteroFlash research ops, granted by the owner)' 'lok ALL=(ALL:ALL) NOPASSWD: ALL' > /root/hf-sudoers-check
visudo -cf /root/hf-sudoers-check >/dev/null || { echo "sudoers validation failed; nothing changed"; exit 1; }
install -o root -g root -m 440 /root/hf-sudoers-check /etc/sudoers.d/90-lok-nopasswd
echo "lok now has passwordless sudo on $(hostname)"
exec bash "$(dirname "$0")/maxq-direct-link.sh"
