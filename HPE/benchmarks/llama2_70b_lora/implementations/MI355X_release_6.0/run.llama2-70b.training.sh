#!/bin/bash
#
# Llama 2 70B LoRA launcher.
#
# Mount contract enforced by code/run_with_docker.sh:
#   ${DATADIR}/data          -> /data   must hold train.npy, validation.npy, packed_metadata.jsonl
#   ${MODEL:-${DATADIR}/model} -> /ckpt  must hold megatron_checkpoints/Llama-2-70b-hf/iter_0000000
# Both point at the same directory here, which is the layout produced by
# scripts/prepare_data_and_model.sh.

set -euo pipefail

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
cd "${SCRIPT_DIR}"

# Node-local staged copy by default; override MLPERF_DATA_ROOT to use NFS.
export MLPERF_DATA_ROOT=${MLPERF_DATA_ROOT:-/lvol0/mlperf/llama2}
export DATADIR=${DATADIR:-${MLPERF_DATA_ROOT}}
export MODEL=${MODEL:-${MLPERF_DATA_ROOT}/data}
export LOGDIR=${LOGDIR:-/nas/SHARED/MLPERF/mlperf_llama2/results}
export CONT=${CONT:-rocm/amd-mlperf:llama2_70b_training_6.0}

for f in train.npy validation.npy packed_metadata.jsonl; do
  [[ -e "${DATADIR}/data/${f}" ]] || { echo "FATAL: missing ${DATADIR}/data/${f}" >&2; exit 1; }
done
[[ -d "${MODEL}/megatron_checkpoints/Llama-2-70b-hf/iter_0000000" ]] \
  || { echo "FATAL: missing ${MODEL}/megatron_checkpoints/Llama-2-70b-hf/iter_0000000" >&2; exit 1; }

mkdir -p "${LOGDIR}"
chmod -R 777 "${LOGDIR}" 2>/dev/null || true

export PYTHONPATH=""

# config_MI355X_1x8x1.sh resets NEXP, so any override must come after sourcing it.
# The same applies to LR: the config does `export LR=0.0006` unconditionally, so a
# host-side `LR=... bash run.llama2-70b.training.sh` is silently discarded. The
# active recipe reads `lr: ${LR:0.0006}` from conf/llama2_70b_lora_posttrain.yaml,
# so LR is the correct name to set -- it just has to be set last.
source config_MI355X_1x8x1.sh
export NEXP=${NEXP_OVERRIDE:-1}
export LR=${LR_OVERRIDE:-${LR}}
export PRIMUS_LR_WARMUP_ITERS=${PRIMUS_LR_WARMUP_ITERS:-0}
export CLEAR_CACHES=${CLEAR_CACHES:-1}
export MLLOG_VERBOSE_LOGS=${MLLOG_VERBOSE_LOGS:-0}
export CHECK_COMPLIANCE=${CHECK_COMPLIANCE:-1}

set -x
./run_with_docker.sh
