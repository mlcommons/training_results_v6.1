#!/bin/bash
#SBATCH --partition=def
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --exclusive
#SBATCH --comment=docker
#SBATCH --time=4:00:00
#SBATCH --job-name=bruno-llama31
#SBATCH --output=/home/hpebench/slurm-logs/%x-%j.out
#
# One Llama 3.1 8B run with the settings of our scored set (tag softmax32_final:
# FP32 attention softmax YAML, micro-batch 4, NUMA binding), plus optional
# VAR=VAL overrides, e.g. PRIMUS_LR_WARMUP_ITERS=128. min_lr and lr decay are
# derived by run.llama31-8b.training.sh so they stay compliant.
#
# Usage: sbatch -w <node> --export=ALL,TAG=<tag> run_llama31.sbatch [VAR=VAL ...]
set -uo pipefail
IMPL=${IMPL:-${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "$0")")}}
if [[ -d /nas/data/SHARED/MLPERF/results ]]; then NAS=/nas/data/SHARED/MLPERF; else NAS=/nas/SHARED/MLPERF; fi
TAG=${TAG:?set TAG}
export LOGDIR=${NAS}/results/llama31_8b/${TAG}

echo "=== llama31 $(date -u) on $(hostname): TAG=${TAG} overrides: $*"
for _ in $(seq 1 60); do [[ $(squeue -h -w "$(hostname -s)" -t CG | wc -l) -eq 0 ]] && break; sleep 10; done
docker info >/dev/null 2>&1 || { echo "FATAL: docker daemon not available"; exit 1; }
cd "${IMPL}"
./run.llama31-8b.training.sh "${TAG}" 1 \
    EXP=/workspace/code/conf/llama3.1_8B-pretrain-fp4-softmax32.yaml \
    PRIMUS_MICRO_BATCH_SIZE=4 MLLOG_MICRO_BATCH_SIZE=4 MLPERF_NUMA_BIND=1 "$@"
rc=$?
echo "=== done rc=${rc} $(date -u)"
exit ${rc}
