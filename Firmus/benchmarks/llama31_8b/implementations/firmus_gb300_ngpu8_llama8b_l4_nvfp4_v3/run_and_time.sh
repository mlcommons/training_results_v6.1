#!/bin/bash
# run_and_time.sh -- one result of firmus_gb300_ngpu8_llama8b_l4_nvfp4_v3: one training run (NEXP=1) of llama31_8b on
# 2 nodes x 4 GB300, submitted to Slurm exactly as the ten submitted runs were.
#
#   CONT=<squashfs from setup.sh> DATADIR=<as init_datasets.sh> LOGDIR=<results dir> SEED=<seed> bash run_and_time.sh
#
# What it does: sets the submission identity that run.sub writes into the MLLOG and the :::SYSJSON line, sources the L4
# config leaf (NVIDIA's GB300 2x4x2 leaf + the Firmus AK_* switches) from the implementation directory and submits
# run.sub, which runs NVIDIA's run_and_time.sh inside the container on every rank and the MLPerf compliance checker on
# the log.  The result is $LOGDIR/<DATESTAMP>_1.log (result_<i>.txt is that file after redact_logs.sh).
#
# Seeds: each submitted run used a fresh seed drawn from /dev/urandom before the set started (README.md, "Seeds").
set -euo pipefail
: "${CONT:?}" "${DATADIR:?}" "${LOGDIR:?}" "${SEED:?set SEED (a fresh random seed per run)}"
IMPL=$(cd "$(dirname "${BASH_SOURCE[0]}")/../nemo_l4_nvfp4_v3" && pwd)

# submission identity (as in the runs)
export MLPERF_RULESET=6.1.0 MLPERF_SUBMISSION_ORG=Firmus MLPERF_SUBMITTER=Firmus MLPERF_DIVISION=closed
export MLPERF_STATUS="Available on-premise" MLPERF_NUM_NODES=2
export MLPERF_SUBMISSION_PLATFORM=firmus_gb300_ngpu8_llama8b_l4_nvfp4_v3
export MLPERF_SYSTEM_NAME="Firmus GB300 NVL72"   # the ten runs set "Firmus GB300 NVL72, 2 trays (8x GB300), NGC 26.09"; the system json uses this name
export MLPERF_HOST_STORAGE_TYPE="NVMe SSD"
export MLPERF_HOST_STORAGE_CAPACITY="4x 3.84 TB NVMe (Solidigm SB5PHU7X038T) RAID0 14 TB + 1x 1.92 TB NVMe (Samsung MZ1L21T9HCLS) boot"
export MLPERF_HOST_NETWORKING="NVIDIA ConnectX-8 Ethernet/RoCE (not used for training collectives)"
export MLPERF_HOST_MEMORY_CONFIGURATION="LPDDR5X on-package, 2x ~490 GB"
export MLPERF_ACCELERATOR_INTERCONNECT="NVLink 5 (18 links x 53.125 GB/s per direction per GPU) + NVLink Switch"
export MLPERF_ACCELERATOR_INTERCONNECT_TOPOLOGY="Multi-node NVLink: both trays in one GB300 NVL72 NVLink/IMEX domain"
export MLPERF_COOLING=liquid
# as in the runs: no prolog NCCL test, one experiment per job
export NCCL_TEST=0 ATTEMPT_CUDA_GDB_CORE_DUMP=0 NEXP=1 SEED CONT DATADIR LOGDIR
mkdir -p "$LOGDIR"
cd "$IMPL"
source config_GB300_2x4x2xtp1pp1cp1_8b_l4.sh
sbatch -N "$DGXNNODES" --time="$WALLTIME" --exclusive --mem=0 --gpus-per-node="$DGXNGPU" \
       --job-name=mlperf-llama31_8b-2x4 --output="$LOGDIR/slurm-%j.out" ${SBATCH_EXTRA:-} run.sub
