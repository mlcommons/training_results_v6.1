#!/bin/bash
#SBATCH --partition=MDF-High
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --job-name=monnetb-flux61-full
#SBATCH --output=%x-%j.out
#
# HPE, lodin4: image (load or build+save), GPU GEMM check, then FLUX 6.1 runs
# on the full verified dataset. Images do not survive a job on lodin4.
# Usage (from lodi-login, in ~/slurm-logs): sbatch -w lodin4 --export=ALL,TAG=<tag>,NEXP=<n> flux61_lodin4_full.sbatch [VAR=VAL ...]
set -uo pipefail
IMPL=${HOME}/mlperf61/flux1
bash "${IMPL}/build_flux_lodin4.sh" || exit 1
docker run --rm --device /dev/kfd --device /dev/dri --group-add video --ipc=host \
    -v "${HOME}/mlperf61/tools:/t" rocm/amd-mlperf:flux1_training_6.1 python /t/gpu_bench.py
TAG=${TAG:-full} DATADIR=/lvol0/mlperf/flux1_mlc bash "${IMPL}/run_flux61.sbatch" FSDP_ALL_GATHER_BACKEND=nccl "$@"
