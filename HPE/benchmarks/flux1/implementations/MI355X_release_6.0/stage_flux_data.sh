#!/bin/bash
#
# Flux.1-schnell data staging.
#
# Data flow, chosen so no single filesystem ever holds both copies:
#
#   vader:/hpelustre/.../sd-flux   (MLCommons preprocessed, ~2.3 TiB)
#          |  pull
#          v
#   /nas/SHARED/MLPERF/mlperf_flux1   (shared, read-only during convert)
#          |  convert   (to_webdataset.py + energon prepare, run per node)
#          v
#   /lvol0/mlperf/flux1/energon       (node-local NVMe, ~2.3 TiB, read at ~1 GB/s)
#
# Flux streams ~2.2 MB/sample (the T5 256x4096 bf16 embedding dominates) and a
# scored run reads ~16 TB over ~6.6 epochs, so the training copy MUST be on
# node-local NVMe. This is unlike llama31, where NFS cost only 0.8%.
#
# `energon prepare` rewrites metadata in place (.nv-meta) rather than copying,
# so the to_webdataset.py output is already the final tree.
#
# Usage:
#   stage_flux_data.sh inspect     # read-only survey; ALWAYS run this first
#   stage_flux_data.sh pull        # vader -> NAS_STAGE
#   stage_flux_data.sh fetch-empty # empty_encodings/ (absent from the 5.1 copy)
#   stage_flux_data.sh convert     # NAS_STAGE -> LOCAL_STAGE, on the current node
#   stage_flux_data.sh verify      # post-conversion sanity check

set -euo pipefail

VADER_HOST=${VADER_HOST:-vader}
VADER_SRC=${VADER_SRC:-/hpelustre/SHARED/datasets/MLPERF/training5.1/sd-flux}
NAS_STAGE=${NAS_STAGE:-/nas/SHARED/MLPERF/sd-flux}
LOCAL_STAGE=${LOCAL_STAGE:-/lvol0/mlperf/flux1}
CONT=${CONT:-rocm/amd-mlperf:flux1_training_6.0}

# Reference sample counts, from text_to_image/README.md. Used only as a check.
TRAIN_SAMPLES=1099776
VAL_SAMPLES=29696

die() { echo "FATAL: $*" >&2; exit 1; }

free_gb() { df -BG --output=avail "$1" 2>/dev/null | tail -1 | tr -dc '0-9'; }

cmd_inspect() {
    echo "=== source: ${VADER_HOST}:${VADER_SRC} ==="
    ssh -o ConnectTimeout=20 "${VADER_HOST}" \
        "ls -la '${VADER_SRC}' && echo && du -sh '${VADER_SRC}'/* 2>/dev/null" \
        || echo "  (unreachable)"

    echo
    echo "=== which format is it? ==="
    echo "  cc12m_preprocessed/ + coco_preprocessed/ -> MLCommons form, conversion NEEDED"
    echo "  energon/{train,val}/                     -> already converted, skip to 'pull' into LOCAL_STAGE"

    echo
    echo "=== capacity ==="
    for p in "$(dirname "${NAS_STAGE}")" "$(dirname "${LOCAL_STAGE}")"; do
        [[ -d "$p" ]] && printf '  %-40s %s GB free\n' "$p" "$(free_gb "$p")" \
                      || printf '  %-40s MISSING\n' "$p"
    done
    echo "  need ~2400 GB free on each"

    echo
    echo "=== docker image ==="
    docker image inspect "${CONT}" >/dev/null 2>&1 \
        && echo "  ${CONT}: present" \
        || echo "  ${CONT}: ABSENT (docker pull ${CONT}; ~85 GB)"
    echo "  NOTE: / is tmpfs on these nodes, so /var/lib/docker consumes RAM."
    echo "        Consider moving docker data-root to /lvol0 before pulling a third image."
}

cmd_pull() {
    mkdir -p "${NAS_STAGE}"
    local avail; avail=$(free_gb "${NAS_STAGE}")
    [[ -n "${avail}" && "${avail}" -ge 2400 ]] \
        || die "${NAS_STAGE} has ${avail:-?} GB free, need ~2400"

    # -W: whole-file, correct for a first copy over a LAN; delta-xfer would only
    # burn CPU on 2.3 TiB of incompressible embeddings.
    rsync -aHW --info=progress2 --partial \
        "${VADER_HOST}:${VADER_SRC}/" "${NAS_STAGE}/"
}

cmd_convert() {
    [[ -d "${NAS_STAGE}" ]] || die "${NAS_STAGE} missing; run 'pull' first"
    docker image inspect "${CONT}" >/dev/null 2>&1 || die "image ${CONT} not present"

    # The vader 5.1 copy ships only cc12m_preprocessed/ and coco_preprocessed/.
    # empty_encodings/ is a separate MLCommons artifact and was NOT present; it
    # is fetched by 'fetch-empty'. Check up front rather than discovering it
    # after a multi-hour conversion.
    for d in cc12m_preprocessed coco_preprocessed empty_encodings; do
        [[ -d "${NAS_STAGE}/${d}" ]] || die "${NAS_STAGE}/${d} missing" \
            "$([[ ${d} == empty_encodings ]] && echo "; run '$0 fetch-empty'")"
        [[ -n "$(ls -A "${NAS_STAGE}/${d}" 2>/dev/null)" ]] || die "${NAS_STAGE}/${d} is empty"
    done

    mkdir -p "${LOCAL_STAGE}/energon"
    local avail; avail=$(free_gb "${LOCAL_STAGE}")
    [[ -n "${avail}" && "${avail}" -ge 2400 ]] \
        || die "${LOCAL_STAGE} has ${avail:-?} GB free, need ~2400"

    # Source is mounted read-only: the conversion reads from NFS and writes to
    # node-local NVMe, so the two 2.3 TiB copies never share a filesystem.
    # Run this on lodin1 and lodin2 concurrently; they write to their own /lvol0.
    docker run --rm -i --network=host --ipc=host \
        --volume "${NAS_STAGE}:/src:ro" \
        --volume "${LOCAL_STAGE}:/dataset" \
        "${CONT}" bash -s <<'INNER'
set -euo pipefail
mkdir -p /dataset/energon

python /workspace/code/scripts/to_webdataset.py \
    --input_path /src/cc12m_preprocessed \
    --output_path /dataset/energon/train \
    --num_workers 8

python /workspace/code/scripts/to_webdataset.py \
    --input_path /src/coco_preprocessed \
    --output_path /dataset/energon/val \
    --num_workers 8

cp -r /src/empty_encodings /dataset/energon/

# `energon prepare` is interactive: it asks about duplicate keys, whether to
# build dataset.yaml interactively, and for a sample-type index. AMD documents
# the answers as y / y / 11. If the prompt order ever changes this will desync,
# so the output is checked immediately below.
cd /dataset/energon
printf 'y\ny\n11\n' | energon prepare --split-parts 'train:train/.*' --split-parts 'val:val/.*' ./

test -d /dataset/energon/.nv-meta || { echo "FATAL: energon prepare produced no .nv-meta" >&2; exit 1; }
echo "OK: energon metadata written"
INNER
}

cmd_fetch_empty() {
    # empty_encodings is the T5/CLIP encoding of the empty prompt, used for
    # classifier-free guidance. Only 2 files (~2 MB), but the training loop
    # needs them and the vader 5.1 snapshot does not include them.
    local base=https://training.mlcommons-storage.org/flux_1/datasets/empty_encodings
    local dst="${NAS_STAGE}/empty_encodings"
    mkdir -p "${dst}"
    for f in t5_empty.npy clip_empty.npy; do
        curl -fsSL -o "${dst}/${f}" "${base}/${f}" || die "download of ${f} failed"
    done
    curl -fsSL https://training.mlcommons-storage.org/metadata/flux-1-empty-encodings.md5 \
        | (cd "${dst}" && md5sum -c -) || die "checksum mismatch in ${dst}"
    echo "OK: ${dst}"
}

cmd_verify() {
    local tree="${LOCAL_STAGE}/energon"
    [[ -d "${tree}" ]] || die "${tree} missing"

    echo "=== ${tree} ==="
    du -sh "${tree}"/* 2>/dev/null || true
    echo
    for d in train val empty_encodings .nv-meta; do
        [[ -e "${tree}/${d}" ]] && echo "  ${d}: present" || echo "  ${d}: MISSING"
    done
    echo
    echo "  expected train samples: ${TRAIN_SAMPLES}"
    echo "  expected val samples:   ${VAL_SAMPLES}"
    echo "  tar shards: train=$(find "${tree}/train" -name '*.tar' 2>/dev/null | wc -l)" \
         "val=$(find "${tree}/val" -name '*.tar' 2>/dev/null | wc -l)"
}

case "${1:-}" in
    inspect)     cmd_inspect ;;
    pull)        cmd_pull ;;
    fetch-empty) cmd_fetch_empty ;;
    convert)     cmd_convert ;;
    verify)      cmd_verify ;;
    *)           die "usage: $0 {inspect|pull|fetch-empty|convert|verify}" ;;
esac
