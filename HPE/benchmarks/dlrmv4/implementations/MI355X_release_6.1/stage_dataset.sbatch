#!/bin/bash
#SBATCH --partition=def
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --comment=docker
#SBATCH --time=6:00:00
#SBATCH --job-name=bruno-dlrmv4-stage
#SBATCH --output=/home/hpebench/slurm-logs/%x-%j.out
#
# HPE: put the DLRMv4 dataset on node-local disk and check it.
#  1. wait for fetch_dataset.sh to finish the durable /nas copy
#  2. copy /nas -> /lvol0 with AMD's stage_dataset_local.sh
#  3. md5 against the MLCommons list, then AMD's verify_dataset.sh (sha256)
#  4. prebuild the time-window index (otherwise built inside the first timed
#     run) and save it to /nas next to the dataset
#
# Usage: sbatch -w <node> stage_dataset.sbatch
set -uo pipefail
if [[ -d /nas/data/SHARED/MLPERF/containers ]]; then NAS=/nas/data/SHARED/MLPERF; else NAS=/nas/SHARED/MLPERF; fi
SRC=${SRC:-${NAS}/dlrmv4/data}
DST=${DST:-/lvol0/mlperf/dlrmv4/data}
CONT=${CONT:-rocm/amd-mlperf:dlrmv4_training_6.1}
IMPL=$(dirname "$(readlink -f "$0")")
[[ -f "${IMPL}/stage_dataset_local.sh" ]] || IMPL=${REPO:-/home/hpebench/mlcommons_mlperf_training_61a}/HPE/benchmarks/dlrmv4/implementations/MI355X_release_6.1
CACHE=processed_5b/hstu_cache_L4086
IDX=(ts_indices_L4086_W86400.npy ts_offsets_L4086_W86400.npy)
log() { echo "=== $(date -u +%H:%M:%S) $*"; }

log "$(hostname): ${SRC} -> ${DST}"
for _ in $(seq 1 720); do [[ -f "${SRC}/.fetch/COMPLETE" ]] && break; sleep 30; done
[[ -f "${SRC}/.fetch/COMPLETE" ]] || { echo "FATAL: download not complete"; exit 1; }

sudo mkdir -p "$(dirname "${DST}")" && sudo chown "$(id -u):$(id -g)" "$(dirname "${DST}")"
DLRM_STAGE_SRC=${SRC} DLRM_LOCAL_DATADIR=${DST} DLRM_STAGE_STREAMS=8 bash "${IMPL}/stage_dataset_local.sh" || exit 1
cp "${SRC}/dlrmv4_dataset.md5" "${DST}/"

log "md5 (MLCommons list)"
(cd "${DST}" && awk '{print $2}' dlrmv4_dataset.md5 | xargs -P 20 -n 1 md5sum) | sort -k2 > "${DST}/.md5.local"
if ! diff <(sort -k2 "${DST}/dlrmv4_dataset.md5") "${DST}/.md5.local"; then
    echo "FATAL: md5 mismatch (local copy)"; exit 1
fi
log "md5 OK"
DLRM_DATA_PATH=${DST} bash "${IMPL}/scripts/verify_dataset.sh" || exit 1

for _ in $(seq 1 60); do [[ $(squeue -h -w "$(hostname -s)" -t CG | wc -l) -eq 0 ]] && break; sleep 10; done
if [[ ! -f "${DST}/${CACHE}/${IDX[0]}" || ! -f "${DST}/${CACHE}/${IDX[1]}" ]]; then
    if [[ -f "${SRC}/${CACHE}/${IDX[0]}" && -f "${SRC}/${CACHE}/${IDX[1]}" ]]; then
        log "index: copy from /nas"; cp "${SRC}/${CACHE}/${IDX[0]}" "${SRC}/${CACHE}/${IDX[1]}" "${DST}/${CACHE}/"
    else
        docker image inspect "${CONT}" >/dev/null 2>&1 || docker load -q -i "${NAS}/containers/amd-mlperf_${CONT##*:}.tar" || exit 1
        log "index: build"
        docker run --rm -v "${DST}:/data" "${CONT}" python3 -m generative_recommenders.dlrm_v4.datasets.offline_ts_indices \
            --cache-dir /data/${CACHE} --history-length 4086 --min-history 4086 --window-seconds 86400 || exit 1
        sudo chown "$(id -u):$(id -g)" "${DST}/${CACHE}"/ts_*.npy
        log "index: save to /nas"; cp "${DST}/${CACHE}/${IDX[0]}" "${DST}/${CACHE}/${IDX[1]}" "${SRC}/${CACHE}/"
    fi
fi
ls -la "${DST}/${CACHE}"
du -sh "${DST}"
log "done"
