#!/bin/bash
#SBATCH --partition=def
#SBATCH --nodelist=lodin2
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --exclusive
#SBATCH --comment=docker
#SBATCH --time=24:00:00
#SBATCH --job-name=flux1-convert
#SBATCH --output=/nas/SHARED/MLPERF/results/flux1/convert-%j.out
#SBATCH --error=/nas/SHARED/MLPERF/results/flux1/convert-%j.err
#
# Flux.1 energon conversion, batch half.
#
# sbatch rather than salloc+tmux on purpose: this survives the submitter
# disconnecting by design, and queues automatically behind the current lodin2
# allocation instead of needing a human to start it at the handover.
#
# Runs ONLY to_webdataset.py, which is deterministic and safe unattended.
# `energon prepare` is deliberately NOT run here: it is interactive (y/y/11),
# that automation is untested, and a desynced prompt would hang and burn the
# whole window. It takes minutes, so it is cheap to do attended afterwards.
#
# GPUs are not needed for dataset preparation; the exclusive allocation is only
# to get the slurm-managed docker daemon (--comment=docker).

set -euo pipefail

NAS_STAGE=${NAS_STAGE:-/nas/SHARED/MLPERF/sd-flux}
LOCAL_STAGE=${LOCAL_STAGE:-/lvol0/mlperf/flux1}
CONT=${CONT:-rocm/amd-mlperf:flux1_training_6.0}

echo "=== flux1 convert: $(date -u) on $(hostname) ==="

# Idempotent: the image is normally already pulled, but /var/lib/containers is
# on the RAM-root tmpfs, so it does not survive a node reboot. Re-pull if gone.
if ! docker image inspect "${CONT}" >/dev/null 2>&1; then
    echo "image absent, pulling"
    docker pull "${CONT}"
fi

for d in cc12m_preprocessed coco_preprocessed empty_encodings; do
    [[ -d "${NAS_STAGE}/${d}" ]] || { echo "FATAL: ${NAS_STAGE}/${d} missing" >&2; exit 1; }
    [[ -n "$(ls -A "${NAS_STAGE}/${d}")" ]] || { echo "FATAL: ${NAS_STAGE}/${d} empty" >&2; exit 1; }
done

avail=$(df -BG --output=avail "${LOCAL_STAGE%/*}" | tail -1 | tr -dc '0-9')
[[ "${avail}" -ge 2400 ]] || { echo "FATAL: only ${avail} GB free, need ~2400" >&2; exit 1; }

mkdir -p "${LOCAL_STAGE}/energon"

docker run --rm -i --network=host --ipc=host \
    --volume "${NAS_STAGE}:/src:ro" \
    --volume "${LOCAL_STAGE}:/dataset" \
    "${CONT}" bash -s <<'INNER'
set -euo pipefail
mkdir -p /dataset/energon

# The argument names come from AMD's blog, not from the script itself. Verify
# before committing hours to a run that would otherwise fail on the first call.
help=$(python /workspace/code/scripts/to_webdataset.py --help 2>&1 || true)
for flag in --input_path --output_path --num_workers; do
    if ! grep -q -- "${flag}" <<<"${help}"; then
        echo "FATAL: to_webdataset.py does not accept ${flag}. Full --help below:" >&2
        echo "${help}" >&2
        exit 1
    fi
done
echo "preflight: to_webdataset.py arguments confirmed"

echo "--- train (cc12m, 2.2 TB) ---"
python /workspace/code/scripts/to_webdataset.py \
    --input_path /src/cc12m_preprocessed \
    --output_path /dataset/energon/train \
    --num_workers 8

echo "--- val (coco, 60 GB) ---"
python /workspace/code/scripts/to_webdataset.py \
    --input_path /src/coco_preprocessed \
    --output_path /dataset/energon/val \
    --num_workers 8

cp -r /src/empty_encodings /dataset/energon/
echo "CONVERT_OK"
INNER

echo "=== done: $(date -u) ==="
du -sh "${LOCAL_STAGE}/energon"/* 2>/dev/null || true
echo
echo "NEXT (attended): run 'energon prepare' inside the container:"
echo "  cd /dataset/energon && energon prepare --split-parts 'train:train/.*' --split-parts 'val:val/.*' ./"
