#!/bin/bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.
#
# Slurm job for the MLPerf flux1 submission runs. Set the variables below in this file
# (not in the submitting shell), then: sbatch job.sh
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --job-name=flux1_mlperf
#SBATCH --output=flux1_%j.out
#SBATCH --error=flux1_%j.err
#SBATCH --partition=YOUR_PARTITION

set -x
export CONT="<IMAGE>"                      # the image built from this directory's Dockerfile
export DATADIR=/path/to/dataset           # contains mlperf_flux1/ (scripts/prepare_data.sh)
export DATASET=${DATASET:-mlperf_flux1}   # the dataset directory inside DATADIR (scripts/verify_data.sh checks it)
export LOGDIR=/path/to/results            # shared by all nodes
export CACHEDIR=/path/to/node-local/cache # node-local NVMe for compile caches
export RCCL_TOPO_FILE=${RCCL_TOPO_FILE:-}  # optional: host path of the site's RCCL topology XML for these nodes
export MLPERF_CONFIG=${MLPERF_CONFIG:-MI355X_04x08x32_mxfp6heal_hb128}  # config_MI355X_04x08x32_mxfp6heal_hb128.sh, the submitted config
export NEXP=10
export ROOT_SEED=${ROOT_SEED:-$RANDOM}

srun bash run_with_docker_slurm.sh
