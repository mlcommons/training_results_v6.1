#!/usr/bin/env bash
# One-time administrator setup for the exact 16 nodes used by this launcher.
set -euo pipefail
[[ $(id -u) != 0 ]] || { echo 'Run as your normal account; the script invokes sudo for the admin job.' >&2; exit 1; }
RL_ROOT="/lustre/$(id -un)/mlperf-rl"
source "$RL_ROOT/site.sh"
nodes='xtyslurma4-a4xmaxnodeset1-[0,2-16]'
submit_user=$(id -un)
[[ "$submit_user" =~ ^[a-z_][a-z0-9_-]*$ ]] || { echo 'Unsupported account name' >&2; exit 1; }
sudo "$(command -v srun)" -p a4xmaxpartition -N16 -n16 --segment=16 \
    --gpus-per-node=4 --ntasks-per-node=1 --exclusive --chdir=/tmp \
    --time=00:05:00 --input=none --nodelist="$nodes" \
    bash -euc '
        test "$(id -u)" = 0
        user=$1
        shared=$2
        test "$(findmnt -n -o FSTYPE -T "$shared")" = lustre
        test -c /dev/fuse
        visudo -cq
        tmp=$(mktemp /etc/sudoers.d/.mlperf-cache.XXXXXX)
        printf "%s ALL=(root) NOPASSWD: /sbin/sysctl vm.drop_caches\\=3\n" "$user" > "$tmp"
        chmod 0440 "$tmp"
        visudo -cf "$tmp"
        mv "$tmp" "/etc/sudoers.d/99-mlperf-drop-caches-$user"
        visudo -cq
        sync
        runuser -u "$user" -- sudo -n -k /sbin/sysctl vm.drop_caches=3
        runuser -u "$user" -- env ENROOT_CACHE_PATH="$shared/enroot-cache" enroot list >/dev/null
        echo "GB300_NODE_READY $(hostname)"
    ' -- "$submit_user" "$RL_ROOT"
printf '%s\n' "$nodes" > "$RL_ROOT/gb300_1013_nodes_ready"
echo 'ALL_16_NODES_READY'
