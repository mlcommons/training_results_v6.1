#!/bin/bash
#SBATCH --partition=def
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --exclusive
#SBATCH --comment=docker
#SBATCH --time=9:00:00
#SBATCH --job-name=bruno-flux1-v267
#SBATCH --output=/home/hpebench/slurm-logs/%x-%j.out
#
# AMD's Flux.1 MLPerf recipe from Primus v26.7.0 (examples/mlperf/flux1):
# in-tree "diffusion" backend, FSDP2, tensorwise FP8, compiled transformer
# blocks, AITER flash attention. Same docker command as AMD's
# run_with_docker.sh, with the changes a scored run needs:
#   - random seed per run (the recipe defaults to a fixed 10007);
#   - no checkpoints at all (default writes 93 GB every 100 steps, inside the
#     timed loop, and resumes from the latest one);
#   - a fresh output dir per run, compliance log under /nas results;
#   - submission fields filled in (defaults say "reference").
#   - open-file limit raised in the container: every data-loader process keeps
#     all 4,762 .arrow files open; the first eval died with 'Too many open files'.
# DATALOADER_NUM_WORKERS (per GPU) needs PRIMUS_HOST with
# primus_v26.7.0_flux_workers.patch; stock v26.7.0 always uses 4.
# Data: raw MLCommons preprocessed format on local disk
# (experimental/rebuild_flux_mlc.sbatch), not the v6.0 energon tree.
#
# Usage: sbatch -w <node> [--export=ALL,TAG=..,LR=..,MAX_STEPS=..] run_primus_flux1.sbatch
set -uo pipefail
if [[ -d /nas/data/SHARED/MLPERF/containers ]]; then NAS=/nas/data/SHARED/MLPERF; else NAS=/nas/SHARED/MLPERF; fi
CONT=${CONT:-zirui3/primus-v26.3-flux:v0.1}
PRIMUS_HOST=${PRIMUS_HOST:-/home/hpebench/primus/v26.7.0}
DATA_ROOT=${DATA_ROOT:-/lvol0/mlperf/flux1_mlc}
TAG=${TAG:-primus-v26.7}
LR=${LR:-2.0e-4}
WARMUP_STEPS=${WARMUP_STEPS:-1600}
MAX_STEPS=${MAX_STEPS:-30000}
OUT=${NAS}/results/flux1/${TAG}
SEED=$(( $(od -An -N4 -tu4 /dev/urandom) % 2147483647 ))
STAMP=$(date -u +%y%m%d%H%M%S)
NAME=bruno-flux1-${SLURM_JOB_ID:-local}

echo "=== flux1 v26.7 $(date -u) on $(hostname): CONT=${CONT} TAG=${TAG} LR=${LR} MAX_STEPS=${MAX_STEPS} SEED=${SEED} WORKERS=${DATALOADER_NUM_WORKERS:-4} PRIMUS_HOST=${PRIMUS_HOST}"
echo "Primus host: $(cd "${PRIMUS_HOST}" && git log -1 --format='%h %cs %D') + $(cd "${PRIMUS_HOST}" && git status --porcelain | wc -l) local changes"
for _ in $(seq 1 60); do [[ $(squeue -h -w "$(hostname -s)" -t CG | wc -l) -eq 0 ]] && break; sleep 10; done
docker info >/dev/null 2>&1 || { echo "FATAL: docker daemon not available"; exit 1; }
[[ $(ls "${DATA_ROOT}/cc12m_preprocessed" 2>/dev/null | grep -vc '\.md5$') -eq 4764 && $(ls "${DATA_ROOT}/coco_preprocessed" 2>/dev/null | grep -vc '\.md5$') -eq 131 \
   && $(ls "${DATA_ROOT}/empty_encodings" 2>/dev/null | wc -l) -eq 2 ]] \
    || { echo "FATAL: ${DATA_ROOT} incomplete (run rebuild_flux_mlc.sbatch)"; exit 1; }
if ! docker image inspect "${CONT}" >/dev/null 2>&1; then
    tar="${NAS}/containers/$(echo "${CONT#rocm/}" | tr '/:' '__').tar"
    [[ -f ${tar} ]] || { echo "FATAL: ${CONT} not on node and no ${tar}"; exit 1; }
    docker load -q -i "${tar}"
fi
mkdir -p "${OUT}/run-${SLURM_JOB_ID}"
docker rm -f "${NAME}" >/dev/null 2>&1

docker run --rm --init --privileged --name "${NAME}" \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --ipc=host --network=host --shm-size=20G --ulimit memlock=-1:-1 --ulimit nofile=1048576:1048576 \
  -v "${PRIMUS_HOST}:/workspace/Primus" -v "${DATA_ROOT}:/data:ro" -v "${OUT}:/output" \
  -w /workspace/Primus \
  -e NNODES=1 -e NODE_RANK=0 -e MASTER_ADDR=127.0.0.1 -e MASTER_PORT=29500 -e GPUS_PER_NODE=8 \
  -e CONFIG=examples/mlperf/flux1/flux.1_schnell_t2i-pretrain.yaml \
  -e PRIMUS_WORKSPACE=/output/run-${SLURM_JOB_ID}/primus \
  -e DATASET_PATH=/data/cc12m_preprocessed -e EVAL_DATASET_PATH=/data/coco_preprocessed \
  -e EMPTY_ENCODINGS_PATH=/data/empty_encodings \
  -e OUTPUT_DIR=/output/run-${SLURM_JOB_ID} \
  -e FLUX_FLOAT8_RECIPE=tensorwise -e ATTENTION_BACKEND=flash_attn_aiter \
  -e LOCAL_BATCH_SIZE=64 -e MAX_STEPS="${MAX_STEPS}" \
  -e LR="${LR}" -e WARMUP_STEPS="${WARMUP_STEPS}" \
  -e DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}" \
  -e GRADIENT_CHECKPOINTING_RATIO=0.25 -e COMPILE_TRANSFORMER_BLOCKS=true \
  -e FSDP2_RESHARD_AFTER_FORWARD=false -e FSDP2_REDUCE_DTYPE=fp32 \
  -e SAVE_STEPS=0 -e SAVE_STRATEGY=none -e CHECKPOINT_KEEP_LATEST=0 -e RESUME_FROM_CHECKPOINT=latest \
  -e MLPERF_ENABLE=true -e MLPERF_CLEAR_CACHES=true \
  -e MLLOG_OUTPUT_FILE=/output/${STAMP}_1.log \
  -e MLLOG_SUBMISSION_ORG=HPE -e MLLOG_SUBMISSION_PLATFORM=HPE-Compute-XD685-MI355X-288GBx8 \
  -e MLLOG_SUBMISSION_DIVISION=closed -e MLLOG_SUBMISSION_STATUS=onprem \
  -e TARGET_ACCURACY=0.586 -e VAL_CHECK_INTERVAL=262144 -e SEED="${SEED}" \
  "${CONT}" bash -c '
    set -euo pipefail
    mkdir -p "$OUTPUT_DIR"
    sync; echo 3 > /proc/sys/vm/drop_caches
    ./primus-cli direct -- train pretrain --config "$CONFIG"
  '
rc=$?
echo "=== done rc=${rc} $(date -u); log: ${OUT}/${STAMP}_1.log"
grep -h '"key": "run_stop"' "${OUT}/${STAMP}_1.log" 2>/dev/null | cut -c1-250
exit ${rc}
