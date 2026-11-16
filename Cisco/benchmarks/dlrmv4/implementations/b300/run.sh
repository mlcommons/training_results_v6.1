#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == --help ]]; then
    printf '%s\n' \
      'Set CONT, DATADIR, SBATCH_PARTITION, SBATCH_NODELIST; then bash run.sh.' \
      'Code, image and data paths must be accessible at the same path on every node.' \
      'Optional: DLRM_NEXP=1..10; DLRM_RUN_SEEDS=comma-separated independent seeds.' \
      'Optional: DLRM_HCA_MODE=full32 (default), exclude2, or exclude8.' \
      'Results 1-4 and 8-10 used full32; 5-6 used exclude2; 7 used exclude8.'
    exit 0
fi
(($# == 0)) || { echo 'Use --help for launch inputs' >&2; exit 2; }
export DLRM_IMPLEMENTATION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export SLURM_CONF="${SLURM_CONF:-/etc/slurm/slurm.conf}"
source "${DLRM_IMPLEMENTATION_DIR}/config_Zeus_8xB300_SXM_288GB_multi_planar.sh"
export EXTRA_MOUNTS="${EXTRA_MOUNTS},${DLRM_IMPLEMENTATION_DIR}/generative_recommenders/dlrm_v4/train/mlperf_logging_utils.py:/workspace/dlrmv4/generative_recommenders/dlrm_v4/train/mlperf_logging_utils.py:ro"
export NEXP="${DLRM_NEXP:-${NEXP}}"
export DLRM_HCA_MODE="${DLRM_HCA_MODE:-full32}"
source "${DLRM_IMPLEMENTATION_DIR}/select_rdma_ports.sh"
dlrm_select_rdma_ports "$(hostname -s)"
export EXTRA_CONTAINER_ENV="${EXTRA_CONTAINER_ENV},DLRM_HCA_MODE,MLPERF_SUBMISSION_ORG,MLPERF_SUBMISSION_PLATFORM"
export MLPERF_SUBMISSION_ORG=Cisco
export MLPERF_SUBMISSION_PLATFORM=Zeus_8xB300_SXM_288GB_multi_planar
if [[ -z "${DLRM_RUN_SEEDS:-}" ]]; then
    export DLRM_RUN_SEEDS="$(python3 -c 'import os,secrets; n=int(os.environ["NEXP"]); assert 1<=n<=10; seeds=set();
while len(seeds)<n: seeds.add(secrets.randbelow(2147483647)+1)
print(",".join(map(str,seeds)))')"
fi
source "${DLRM_IMPLEMENTATION_DIR}/submission_seeds.sh"
validate_submission_seeds
export SEED="${_submission_seeds[0]}"
export LOGDIR="${LOGDIR:-${DLRM_IMPLEMENTATION_DIR}/results/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${LOGDIR}"
printf '%s\n' "${DLRM_RUN_SEEDS}" > "${LOGDIR}/seeds.csv"
submit_without_allocation_env sbatch --parsable --export=ALL --no-requeue \
    --nodes="${DGXNNODES}" --gres="gpu:${DGXNGPU}" \
    --partition="${SBATCH_PARTITION:?Set SBATCH_PARTITION}" \
    --nodelist="${SBATCH_NODELIST:?Set SBATCH_NODELIST}" --time="${WALLTIME}" \
    --chdir="${DLRM_IMPLEMENTATION_DIR}" --job-name=dlrmv4-b300 \
    --output="${LOGDIR}/slurm-%j.out" "${DLRM_IMPLEMENTATION_DIR}/run.sub"
