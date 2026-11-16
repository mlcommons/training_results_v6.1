#!/bin/bash
#SBATCH --partition=def
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --comment=docker
#SBATCH --time=12:00:00
#SBATCH --job-name=bruno-flux1-stage-mlc
#SBATCH --output=/home/hpebench/slurm-logs/%x-%j.out
#
# Prepare a node for AMD's Primus v26.7 Flux.1 recipe (examples/mlperf/flux1):
#  1. copy the raw MLCommons preprocessed data (2.28 TB of .arrow + 2 .npy)
#     from durable /nas to this node's /lvol0 (the recipe reads this format
#     directly, not the energon tree used by the v6.0 image);
#  2. get the recipe's image onto the node and keep a tar on /nas.
# Both run in parallel. Restartable: rsync skips finished files.
#
# Superseded by rebuild_flux_mlc.sbatch: on 2026-10-06 /nas delivered only
# ~22 MB/s (2.28 TB = ~28 h); rebuilding from the local energon copy is hours.
#
# Usage: sbatch -w lodin1 --dependency=afterany:<last job on node> stage_flux_mlc_from_nas.sbatch
set -uo pipefail
if [[ -d /nas/data/SHARED/MLPERF/sd-flux ]]; then NAS=/nas/data/SHARED/MLPERF; else NAS=/nas/SHARED/MLPERF; fi
SRC=${NAS}/sd-flux
DST=${DST:-/lvol0/mlperf/flux1_mlc}
CONT=${CONT:-zirui3/primus-v26.3-flux:v0.1}
tar="${NAS}/containers/$(echo "${CONT#rocm/}" | tr '/:' '__').tar"

echo "=== stage flux1 (MLCommons format) $(date -u) on $(hostname): ${SRC} -> ${DST}"
for _ in $(seq 1 60); do [[ $(squeue -h -w "$(hostname -s)" -t CG | wc -l) -eq 0 ]] && break; sleep 10; done
for d in cc12m_preprocessed coco_preprocessed empty_encodings; do
    [[ -n $(ls -A "${SRC}/${d}" 2>/dev/null) ]] || { echo "FATAL: ${SRC}/${d} missing or empty"; exit 1; }
done
need=$(du -s --block-size=1G "${SRC}" | cut -f1)
have=$(du -s --block-size=1G "${DST}" 2>/dev/null | cut -f1); have=${have:-0}
avail=$(df -BG --output=avail /lvol0 | tail -1 | tr -dc 0-9)
[[ $(( avail + have - need )) -gt 200 ]] || { echo "FATAL: need ${need} GB, ${avail} GB free on /lvol0"; exit 1; }
sudo mkdir -p "${DST}" && sudo chown "$(id -u):$(id -g)" "${DST}"

# Image in the background (Docker Hub ~5 MB/s: ~75 min for 22.6 GB).
(
    docker info >/dev/null 2>&1 || { echo "IMAGE FATAL: docker daemon not available"; exit 1; }
    if ! docker image inspect "${CONT}" >/dev/null 2>&1; then
        if [[ -f "${tar}" ]]; then docker load -q -i "${tar}"; else docker pull -q "${CONT}" || { echo "IMAGE FATAL: pull failed"; exit 1; }; fi
    fi
    [[ -f "${tar}" ]] || { docker save -o "${tar}.part" "${CONT}" && mv "${tar}.part" "${tar}" && chmod 644 "${tar}"; }
    echo "IMAGE OK $(date -u +%H:%M) ${CONT} (tar $(du -h "${tar}" | cut -f1))"
    docker run --rm "${CONT}" bash -c 'python -c "import torch; print(\"torch\", torch.__version__)"; pip list 2>/dev/null | grep -iE "^(primus|torchtitan|aiter|transformer.engine|mlperf.logging|triton) "'
) &
img=$!

s=$(date +%s)
rsync -a "${SRC}/empty_encodings" "${SRC}/coco_preprocessed" "${DST}/"
mkdir -p "${DST}/cc12m_preprocessed"
ls "${SRC}/cc12m_preprocessed" | xargs -P 8 -I{} rsync -a "${SRC}/cc12m_preprocessed/{}" "${DST}/cc12m_preprocessed/{}"
echo "copied in $(( ($(date +%s)-s)/60 )) min"

( cd "${SRC}" && find . -type f -printf '%P %s\n' | grep -v '/\._' | sort ) > /tmp/src.$$
( cd "${DST}" && find . -type f -printf '%P %s\n' | sort ) > /tmp/dst.$$
if diff -q /tmp/src.$$ /tmp/dst.$$ >/dev/null; then echo "VERIFY OK: $(wc -l < /tmp/dst.$$) files, $(du -sh "${DST}" | cut -f1)"
else echo "FATAL: mismatch"; diff /tmp/src.$$ /tmp/dst.$$ | head; rc=1; fi
rm -f /tmp/src.$$ /tmp/dst.$$

wait ${img} || rc=1
echo "=== stage done $(date -u)"
exit ${rc:-0}
