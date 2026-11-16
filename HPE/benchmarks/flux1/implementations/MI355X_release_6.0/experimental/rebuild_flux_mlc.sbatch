#!/bin/bash
#SBATCH --partition=def
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --comment=docker
#SBATCH --time=14:00:00
#SBATCH --job-name=bruno-flux1-rebuild
#SBATCH --output=/home/hpebench/slurm-logs/%x-%j.out
#
# Prepare a node for AMD's Primus v26.7 Flux.1 recipe, which reads the raw
# MLCommons HF datasets (.arrow). Instead of copying 2.28 TB from /nas
# (measured 2026-10-06: ~22 MB/s total, ~28 h), rebuild the .arrow files from
# the energon copy already on this node's /lvol0 (energon_to_hf_arrow.py, the
# exact inverse of scripts/to_webdataset.py). The rebuilt files are
# byte-identical to the MLCommons release: every file is checked against
# MLCommons' official md5 lists (flux_md5/), and any mismatch fails the job.
# Only the tiny metadata files and empty encodings come from /nas.
# In parallel, gets the recipe's image onto the node (tar on /nas, else pull).
#
# Usage: sbatch -w <node> [--dependency=afterany:<last job>] rebuild_flux_mlc.sbatch
set -uo pipefail
if [[ -d /nas/data/SHARED/MLPERF/sd-flux ]]; then NAS=/nas/data/SHARED/MLPERF; else NAS=/nas/SHARED/MLPERF; fi
HERE=/home/hpebench/mlcommons_mlperf_training/HPE/benchmarks/flux1/implementations/MI355X_release_6.0/experimental
SRC=${NAS}/sd-flux
ENERGON=${ENERGON:-/lvol0/mlperf/flux1/energon}
DST=${DST:-/lvol0/mlperf/flux1_mlc}
CONV=${CONV:-rocm/amd-mlperf:flux1_training_6.0}
CONT=${CONT:-zirui3/primus-v26.3-flux:v0.1}
WORKERS=${WORKERS:-16}
tarof() { echo "${NAS}/containers/$(echo "${1#rocm/}" | tr '/:' '__').tar"; }

echo "=== rebuild flux1 HF data $(date -u) on $(hostname): ${ENERGON} -> ${DST}"
for _ in $(seq 1 60); do [[ $(squeue -h -w "$(hostname -s)" -t CG | wc -l) -eq 0 ]] && break; sleep 10; done
docker info >/dev/null 2>&1 || { echo "FATAL: docker daemon not available"; exit 1; }
[[ $(ls "${ENERGON}"/train/shard_*.tar 2>/dev/null | wc -l) -eq 110 && $(ls "${ENERGON}"/val/shard_*.tar 2>/dev/null | wc -l) -eq 3 ]] \
    || { echo "FATAL: ${ENERGON} is not a complete energon tree (110 train + 3 val shards)"; exit 1; }
if ! docker image inspect "${CONV}" >/dev/null 2>&1; then
    [[ -f $(tarof "${CONV}") ]] || { echo "FATAL: no ${CONV} on node or /nas"; exit 1; }
    docker load -q -i "$(tarof "${CONV}")"
fi
sudo mkdir -p "${DST}" && sudo chown "$(id -u):$(id -g)" "${DST}"
have=$(du -s --block-size=1G "${DST}" | cut -f1)
avail=$(df -BG --output=avail /lvol0 | tail -1 | tr -dc 0-9)
[[ $(( avail + have )) -gt 2450 ]] || { echo "FATAL: ${avail} GB free + ${have} GB present < 2450 GB needed"; exit 1; }

(
    if ! docker image inspect "${CONT}" >/dev/null 2>&1; then
        if [[ -f $(tarof "${CONT}") ]]; then docker load -q -i "$(tarof "${CONT}")"; else docker pull -q "${CONT}" || { echo "IMAGE FATAL: pull failed"; exit 1; }; fi
    fi
    [[ -f $(tarof "${CONT}") ]] || { docker save -o "$(tarof "${CONT}").part" "${CONT}" && mv "$(tarof "${CONT}").part" "$(tarof "${CONT}")" && chmod 644 "$(tarof "${CONT}")"; }
    echo "IMAGE OK $(date -u +%H:%M) ${CONT}"
) &
img=$!

# Leftovers of an interrupted rsync, then the small files from /nas.
sudo find "${DST}" -maxdepth 2 -name '.*.arrow.*' -delete
mkdir -p "${DST}/cc12m_preprocessed" "${DST}/coco_preprocessed"
rsync -a "${SRC}/empty_encodings" "${DST}/"
for d in cc12m_preprocessed coco_preprocessed; do rsync -a "${SRC}/${d}/state.json" "${SRC}/${d}/dataset_info.json" "${DST}/${d}/"; done
( cd "${DST}/empty_encodings" && md5sum -c --quiet "${HERE}/flux_md5/flux-1-empty-encodings.md5" ) || rc=1
for d in cc12m_preprocessed coco_preprocessed; do
    ( cd "${DST}/${d}" && grep -E ' (state|dataset_info)\.json$' "${HERE}/flux_md5/flux-1-${d%_preprocessed}-preprocessed.md5" | md5sum -c --quiet ) || rc=1
done

conv() {  # <name> <energon subdir> <rows> <files>
    docker run --rm -v "${DST}:/out" -v "${ENERGON}:/energon:ro" -v "${SRC}:/src:ro" -v "${HERE}:/c:ro" "${CONV}" \
        python /c/energon_to_hf_arrow.py --energon "/energon/$2" --out "/out/$1_preprocessed" \
        --ref "/src/$1_preprocessed/data-00000-of-$(printf %05d "$4").arrow" \
        --md5 "/c/flux_md5/flux-1-$1-preprocessed.md5" --rows "$3" --files "$4" --workers "${WORKERS}"
}
s=$(date +%s)
conv coco val 29696 129 2>&1 | grep -E 'MISMATCH|RESULT|Error|Traceback' ; [[ ${PIPESTATUS[0]} -eq 0 ]] || rc=1
echo "coco done $(( ($(date +%s)-s)/60 )) min"
conv cc12m train 1099776 4762 2>&1 | grep -E 'MISMATCH|RESULT|00/4762|Error|Traceback' ; [[ ${PIPESTATUS[0]} -eq 0 ]] || rc=1
echo "cc12m done $(( ($(date +%s)-s)/60 )) min"
sudo chown -R "$(id -u):$(id -g)" "${DST}"

# MLCommons ships its own .md5 list inside coco_preprocessed (harmless): not counted.
n1=$(ls "${DST}/cc12m_preprocessed" | grep -vc '\.md5$'); n2=$(ls "${DST}/coco_preprocessed" | grep -vc '\.md5$')
echo "files: cc12m ${n1} (want 4764), coco ${n2} (want 131), $(du -sh "${DST}" | cut -f1)"
[[ ${n1} -eq 4764 && ${n2} -eq 131 ]] || rc=1
wait ${img} || rc=1
[[ ${rc:-0} -eq 0 ]] && echo "VERIFY OK: all files match the MLCommons md5 lists" || echo "FATAL: see above"
echo "=== rebuild done $(date -u)"
exit ${rc:-0}
