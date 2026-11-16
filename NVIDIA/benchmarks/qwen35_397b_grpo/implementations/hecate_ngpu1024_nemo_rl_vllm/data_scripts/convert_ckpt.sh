#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Pre-populate the mcore checkpoint cache: nemo-rl converts HF -> mcore on the
# first run against an empty NRL_MEGATRON_CHECKPOINT_DIR; this runs the
# reference config for one step so nightly runs start from a warm cache. The
# converted copy is keyed by the HF checkpoint path only (torch_dist, resharded
# on load), so every other profile reuses it. Pick another profile with
# CONVERT_CONFIG. Site options follow the launcher: --account, --partition and
# --slurm-extra=<sbatch-argument>.
# Usage (from pytorch/, on the cluster):
#   export LOGDIR=<scratch> CONT=<image>; source config_data_<cluster>.sh
#   bash data_scripts/convert_ckpt.sh [--account=<account>] [--partition=<partition>] [--slurm-extra=--gres=gpu:4]
# Wait for the job to finish; nemo-rl marks a complete conversion with
# iter_0000000/run_config.yaml under the cache's model_<HF_CKPT_PATH with / as _> directory.
set -euo pipefail

SCRIPT_DIR=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
cd "${SCRIPT_DIR}/.."

USAGE="usage: ${BASH_SOURCE[0]##*/} [--account=<account>] [--partition=<partition>] [--slurm-extra=<sbatch-argument>]*"
SBATCH_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -A|--account)    SBATCH_ARGS+=( "--account=$2" ); shift 2 ;;
        --account=*)     SBATCH_ARGS+=( "$1" ); shift ;;
        -p|--partition)  SBATCH_ARGS+=( "--partition=$2" ); shift 2 ;;
        --partition=*)   SBATCH_ARGS+=( "$1" ); shift ;;
        --slurm-extra)   SBATCH_ARGS+=( "$2" ); shift 2 ;;
        --slurm-extra=*) SBATCH_ARGS+=( "${1#--slurm-extra=}" ); shift ;;
        -h|--help)       echo "${USAGE}"; exit 0 ;;
        *) echo "ERROR: unrecognized option $1" >&2; echo "${USAGE}" >&2; exit 1 ;;
    esac
done

: "${NRL_MEGATRON_CHECKPOINT_DIR:?NRL_MEGATRON_CHECKPOINT_DIR not set}"
: "${HF_CKPT_PATH:?HF_CKPT_PATH not set}"
mkdir -p "${NRL_MEGATRON_CHECKPOINT_DIR}"

# The profile derives WALLTIME from NEXP, so set it first.
export NEXP=1
export OFFLINE_EVAL=0
source "${CONVERT_CONFIG:-config_GB300_64x4_t32g32_tp4pp2ep32gtp4_mxfp8.sh}"
# After the profile: its config_GBS*.sh sets MAX_STEPS unconditionally.
export MAX_STEPS=1

sbatch --no-requeue --segment="${SEGMENT}" -N "${DGXNNODES}" --time="${WALLTIME}" "${SBATCH_ARGS[@]}" run.sub
echo "Submitted 1-step run. The conversion is complete once iter_0000000/run_config.yaml exists under"
echo "${NRL_MEGATRON_CHECKPOINT_DIR}/model_${HF_CKPT_PATH//\//_}"
