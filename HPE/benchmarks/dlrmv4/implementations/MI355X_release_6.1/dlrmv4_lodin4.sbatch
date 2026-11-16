#!/bin/bash
#SBATCH --partition=MDF-High
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --job-name=monnetb-dlrmv4
#SBATCH --output=%x-%j.out
#
# HPE, lodin4 (lodi cluster, user monnetb): DLRMv4 convergence study.
#  1. stage data + time-window index from /nas to /lvol0 (skips files already
#     there with the right size), md5 against the MLCommons list
#  2. load the image (lodin4 deletes Docker images at job end)
#  3. run each arm of ARMS="<lr>:<runs>;<lr>:<runs>" (dense = sparse LR)
# Logs in $HOME/mlperf61/results/dlrmv4/<TAG>-lr<lr>. Host VAR=VAL arguments
# are exported and forwarded into the container.
#
# Usage (from lodi-login, in ~/slurm-logs):
#   sbatch -w lodin4 --export=ALL,TAG=l4,ARMS="1e-6:4;1.25e-6:4" dlrmv4_lodin4.sbatch [VAR=VAL ...]
set -uo pipefail
TAG=${TAG:?set TAG}
ARMS=${ARMS:-1e-6:1}
NAS=/nas/data/SHARED/MLPERF
SRC=${NAS}/dlrmv4/data
DST=/lvol0/mlperf/dlrmv4/data
IMPL=${HOME}/mlperf61/dlrmv4
export CONT=rocm/amd-mlperf:dlrmv4_training_6.1
log() { echo "=== $(date -u +%H:%M:%S) $*"; }

log "$(hostname) TAG=${TAG} ARMS=${ARMS} overrides: $*"
mkdir -p "${DST}"
( docker image inspect "${CONT}" >/dev/null 2>&1 || docker load -q -i "${NAS}/containers/amd-mlperf_dlrmv4_training_6.1.tar" ) &
imgpid=$!
(cd "${SRC}" && find . -type f ! -path './.fetch/*' -printf '%P %s\n') | while read -r f s; do
    [[ -f "${DST}/${f}" && $(stat -c %s "${DST}/${f}") -eq ${s} ]] || echo "${f}"
done | xargs -r -P 4 -I{} sh -c 'mkdir -p "$(dirname "$2/{}")" && cp "$1/{}" "$2/{}.part" && mv "$2/{}.part" "$2/{}"' _ "${SRC}" "${DST}" || { echo "FATAL: copy"; exit 1; }
log "data copied: $(du -sh "${DST}" | cut -f1)"
if [[ ! -f "${DST}/.md5_ok" ]]; then
    (cd "${DST}" && awk '{print $2}' dlrmv4_dataset.md5 | xargs -P 32 -n 1 md5sum) | sort -k2 > /tmp/md5.local.$$
    diff <(sort -k2 "${DST}/dlrmv4_dataset.md5") /tmp/md5.local.$$ || { echo "FATAL: md5 mismatch"; exit 1; }
    date -Is > "${DST}/.md5_ok"
fi
log "md5 OK"
wait ${imgpid} || { echo "FATAL: image load"; exit 1; }
docker image inspect "${CONT}" >/dev/null 2>&1 || { echo "FATAL: no image"; exit 1; }
log "image ready"

export DGXSYSTEM=MI355X_1x8x1 DATADIR=${DST}
export RUN_RUNTIME_TUNABLES=0
export MLLOG_SUBMISSION_ORG=HPE
for kv in "$@"; do export "${kv?}"; done
export HPE_FORWARD="DENSE_LR SPARSE_LR $(for kv in "$@"; do printf '%s ' "${kv%%=*}"; done)"
IFS=';' read -r -a arms <<< "${ARMS}"
for arm in "${arms[@]}"; do
    lr=${arm%%:*}; n=${arm##*:}
    export LOGDIR=${HOME}/mlperf61/results/dlrmv4/${TAG}-lr${lr}
    export NEXP=${n} CONT_NAME=mlperf_dlrmv4_${SLURM_JOB_ID}
    mkdir -p "${LOGDIR}"
    CODE=${LOGDIR}/code-${SLURM_JOB_ID}-$(date +%y%m%d%H%M%S)
    cp -a "${IMPL}" "${CODE}" && cd "${CODE}" || exit 1
    source "config_${DGXSYSTEM}.sh"
    export MLPERF_SUBMISSION_PLATFORM=XD685_MI355X DENSE_LR=${lr} SPARSE_LR=${lr}
    for kv in "$@"; do export "${kv?}"; done
    log "arm lr=${lr} runs=${n} -> ${LOGDIR}"
    ( while sleep 10; do echo "$(date -u +%T) $(awk '/^Committed_AS/{c=$2} /^CommitLimit/{l=$2} END{printf "committed %d GB of %d GB", c/1048576, l/1048576}' /proc/meminfo)"; done ) > "${LOGDIR}/commit-${SLURM_JOB_ID}.log" 2>&1 &
    monpid=$!
    bash run_with_docker.sh
    kill ${monpid} 2>/dev/null
    log "arm lr=${lr} rc=$?"
done
log "done"
