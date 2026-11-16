#!/bin/bash
# Usage: ./launch.sh <config_file> <NEXP> [tag]
#   ./launch.sh config_GB300_16x4x1xtp1pp1cp2ep4_mxfp8.sh 1  test     # single test run
#   ./launch.sh config_GB300_16x4x1xtp1pp1cp2ep4_mxfp8.sh 10 final    # 10 consecutive runs in one job
# Extra sbatch args can be passed via SBATCH_EXTRA (e.g. SBATCH_EXTRA="--dependency=afterany:123").
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

CONFIG=${1:?config file}
export NEXP=${2:?NEXP}
TAG=${3:-run}

export CONT=/data/containers/gpt_oss_20b-arm_20261012.sqsh   # enroot import of nvcr.io/nvdlfwea/mlperftv61/gpt_oss_20b-arm:20261012
export DATADIR=/data/gpt_oss_20b/dataset
# mgx-00010 mlx5_2 IB port is DOWN; NCCL aborts on mismatched NIC counts. Exclude it everywhere
# (collectives inside the NVL72 go over MNNVL, not IB).
export NCCL_IB_HCA='^mlx5_2'
source "./${CONFIG}"
export LOGDIR=/data/gpt_oss_20b/results/${DGXSYSTEM}/${TAG}_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOGDIR"

echo "config=$CONFIG nodes=$DGXNNODES NEXP=$NEXP walltime=${WALLTIME}min logdir=$LOGDIR"
sbatch -N "${DGXNNODES}" --gres=gpu:${DGXNGPU} --time="${WALLTIME}" \
  -J "gptoss_${DGXNNODES}n_${TAG}" -o "${LOGDIR}/slurm-%j.out" ${SBATCH_EXTRA:-} run.sub
