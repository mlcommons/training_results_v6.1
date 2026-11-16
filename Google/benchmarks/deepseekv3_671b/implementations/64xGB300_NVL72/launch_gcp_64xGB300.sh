#!/bin/bash
# Google Cloud A4X Max: 64 nodes, 256 GPUs, 16 nodes per NVLink domain
set -euo pipefail
cd "$(dirname "$0")"

: "${LOGDIR:?Set LOGDIR to a shared output directory}"
: "${NODELIST:?Set NODELIST to the 64 compute nodes}"

# Generate a fresh random seed unless explicitly supplied.
DS_RUN_SEED="${1:-$(python3 -c 'import secrets; print(secrets.randbelow(2**31 - 1) + 1)')}"

export NEXP=1
source config_GB300_64x4x240xtp1pp4ep32cp1_mxfp8_full_cg.sh

# Override the defaults assigned by the sourced config.
export SEED="$DS_RUN_SEED"
export SEED_BASE="$DS_RUN_SEED"

STAGE="/mnt/localssd/$USER/mlperf-v6.1-deepseekv3"
export CONT="$STAGE/images/deepseekv3_671b-arm+20260921.sqsh"
export DATADIR="$STAGE/dataset"
export LOAD_CHECKPOINTS_DIR="$STAGE/checkpoints"
export LOAD_CHECKPOINT="/checkpoints/deepseekv3_checkpoint_bf16_megatron_bridge"
export LOAD_CHECKPOINT_STEP=0

export REMOUNT_WORKDIR="$PWD"
export LOGDIR
export NPY_INDEX_DIR="$LOGDIR/npy_index"
mkdir -p "$LOGDIR"
printf 'SEED=%s SEED_BASE=%s\n' "$SEED" "$SEED_BASE" |
  tee "$LOGDIR/launch-seed.txt"

export HF_HOME="/hf_home"
export EXTRA_MOUNTS="$DATADIR/config:/hf_home:ro"

export NCCL_SOCKET_IFNAME="=eth0"
export NCCL_SOCKET_FAMILY=AF_INET
export NCCL_DEBUG=WARN
export NCCL_TEST=1

export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
export BF16_EVAL=0
export WARMUP_MOCK_PIPELINE_COMPILE_MODE=off
export USE_SYNTHETIC_DATA=0
export FORCE_SUCCESS_STATUS=0

export MLPERF_SUBMISSION_ORG="Google"
export MLPERF_SUBMISSION_PLATFORM="GB300_NVL72"
export EXTRA_CONTAINER_ENV="NCCL_SOCKET_IFNAME,NCCL_SOCKET_FAMILY,NCCL_DEBUG,HF_HOME,MLPERF_SUBMISSION_ORG,MLPERF_SUBMISSION_PLATFORM,SEED,BF16_EVAL,WARMUP_MOCK_PIPELINE_COMPILE_MODE,NVTE_ALLOW_NONDETERMINISTIC_ALGO"

sbatch --parsable \
  --partition=a4xmaxpartition \
  --nodes=64 \
  --gres=gpu:4 \
  --distribution=block:block \
  --nodelist="$NODELIST" \
  --time="$WALLTIME" \
  --job-name=ds-v61-new \
  --output="$LOGDIR/slurm-%j.out" \
  --export=ALL \
  run.sub
