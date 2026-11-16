#!/usr/bin/env bash
# HPE: download the MLCommons DLRMv4 dataset (~227 GB, 20 files) with many
# parallel HTTP range requests. One stream from lodin gets ~4 MB/s; striker
# gets ~70 MB/s, so run this on striker into /nas (the durable copy), then
# stage to node-local disk with stage_dataset_local.sh and check it there with
# scripts/verify_dataset.sh (local reads are much faster than NFS).
#
# Each file is split into CHUNK-byte ranges written in place; a chunk is
# marked done only after its bytes are on disk, so a rerun resumes.
#
# Usage: fetch_dataset.sh <dest dir> [streams]
set -euo pipefail
DST=${1:?destination directory}
STREAMS=${2:-16}
CHUNK=${CHUNK:-1073741824}
BASE=https://training.mlcommons-storage.org
LIST=${BASE}/metadata/dlrmv4_dataset.md5
DATA=${BASE}/dlrmv4_dataset

mkdir -p "${DST}/.fetch"
curl -fsS "${LIST}" -o "${DST}/dlrmv4_dataset.md5"

jobs=${DST}/.fetch/jobs
: > "${jobs}"
while read -r _ f; do
    size=$(curl -fsSI "${DATA}/${f}" | awk 'tolower($1)=="content-length:" {print $2+0}')
    mkdir -p "$(dirname "${DST}/${f}")"
    [[ -f "${DST}/${f}" ]] || touch "${DST}/${f}"
    if (( size == 0 )); then : > "${DST}/${f}"; continue; fi
    truncate -s "${size}" "${DST}/${f}"
    for ((off = 0, i = 0; off < size; off += CHUNK, i++)); do
        end=$(( off + CHUNK - 1 < size - 1 ? off + CHUNK - 1 : size - 1 ))
        [[ -f "${DST}/.fetch/${f//\//_}.${i}" ]] || echo "${f} ${off} ${end} ${i}" >> "${jobs}"
    done
done < "${DST}/dlrmv4_dataset.md5"
echo "[fetch] $(date -u +%H:%M:%S) $(wc -l < "${jobs}") chunks to fetch, ${STREAMS} streams -> ${DST}"

fetch_chunk() {
    local f=$1 off=$2 end=$3 i=$4 n
    set -o pipefail
    for n in 1 2 3 4 5; do
        if curl -fsS --retry 3 --http1.1 --speed-limit 100000 --speed-time 60 -r "${off}-${end}" "${DATA}/${f}" |
           dd of="${DST}/${f}" bs=4M iflag=fullblock seek="${off}" oflag=seek_bytes conv=notrunc status=none; then
            touch "${DST}/.fetch/${f//\//_}.${i}"
            return 0
        fi
        sleep $(( n * 10 ))
    done
    echo "[fetch] FAILED ${f} chunk ${i}" >&2
    return 1
}
export -f fetch_chunk
export DST DATA

start=$(date +%s)
rc=0
xargs -P "${STREAMS}" -L 1 bash -c 'fetch_chunk "$@"' _ < "${jobs}" || rc=$?
echo "[fetch] $(date -u +%H:%M:%S) done in $(( ($(date +%s) - start) / 60 )) min, rc=${rc}, $(du -sh --apparent-size "${DST}" | cut -f1)"
[[ ${rc} -eq 0 ]] && date -Is > "${DST}/.fetch/COMPLETE"
exit ${rc}
