#!/bin/bash
#SBATCH --partition=MDF-High
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --job-name=monnetb-flux-stage
#SBATCH --output=%x-%j.out
#
# HPE, lodin4 (lodi cluster, user monnetb): copy the raw MLCommons FLUX.1 data
# (2.3 TB) from /nas to node-local /lvol0 and check every file against the
# MLCommons md5 lists. /nas is reached over 1 GbE (~115 MB/s): about 6 h.
# Resumable: files already copied with the right size are skipped.
#
# Usage (from lodi-login): sbatch -w lodin4 stage_flux_lodin4.sbatch
set -uo pipefail
SRC=${SRC:-/nas/data/SHARED/MLPERF/sd-flux}
DST=${DST:-/lvol0/mlperf/flux1_mlc}
STREAMS=${STREAMS:-4}
META=https://training.mlcommons-storage.org/metadata
declare -A LIST=([cc12m_preprocessed]=flux-1-cc12m-preprocessed [coco_preprocessed]=flux-1-coco-preprocessed [empty_encodings]=flux-1-empty-encodings)
log() { echo "=== $(date -u +%H:%M:%S) $*"; }

log "$(hostname): ${SRC} -> ${DST}, ${STREAMS} streams"
start=$(date +%s)
for d in "${!LIST[@]}"; do
    mkdir -p "${DST}/${d}"
    curl -fsS "${META}/${LIST[$d]}.md5" -o "${DST}/${d}.md5" || { echo "FATAL: md5 list ${d}"; exit 1; }
    (cd "${SRC}/${d}" && find . -maxdepth 1 -type f ! -name '._*' -printf '%f %s\n') |
    while read -r f s; do
        [[ -f "${DST}/${d}/${f}" && $(stat -c %s "${DST}/${d}/${f}") -eq ${s} ]] || echo "${d}/${f}"
    done
done | xargs -P "${STREAMS}" -I{} sh -c 'cp "$1/{}" "$2/{}.part" && mv "$2/{}.part" "$2/{}"' _ "${SRC}" "${DST}" || { echo "FATAL: copy"; exit 1; }
log "copy done in $(( ($(date +%s) - start) / 60 )) min, $(du -sh "${DST}" | cut -f1)"

bad=0
for d in "${!LIST[@]}"; do
    n_list=$(wc -l < "${DST}/${d}.md5"); n_have=$(find "${DST}/${d}" -maxdepth 1 -type f | wc -l)
    (cd "${DST}/${d}" && awk '{print $2}' "../${d}.md5" | xargs -P 64 -n 8 md5sum) | sort -k2 > "${DST}/${d}.md5.local"
    if diff -q <(sort -k2 "${DST}/${d}.md5") "${DST}/${d}.md5.local" >/dev/null; then
        log "md5 OK ${d}: ${n_list} files (${n_have} on disk)"
    else
        log "md5 MISMATCH ${d}"; diff <(sort -k2 "${DST}/${d}.md5") "${DST}/${d}.md5.local" | head; bad=1
    fi
done
[[ ${bad} -eq 0 ]] && date -Is > "${DST}/.stage_complete"
log "done rc=${bad}"
exit ${bad}
