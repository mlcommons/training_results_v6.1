#!/bin/bash
# run_and_time.sh -- one result of firmus_gb300_ngpu8_llama8b_l1_1012: one training run (NEXP=1) of llama31_8b on
# 2 nodes x 4 GB300, submitted to Slurm exactly as the ten submitted runs were.
#
#   CONT=<squashfs from setup.sh> DATADIR=<as init_datasets.sh> LOGDIR=<results dir> SEED=<seed> bash run_and_time.sh
#
# What it does: sets the submission identity that run.sub writes into the MLLOG and the :::SYSJSON line, sources NVIDIA's
# config config_GB300_2x4x2xtp1pp1cp1_8b_fp4.sh from the implementation directory and submits run.sub, which runs NVIDIA's
# run_and_time.sh inside the container on every rank and the MLPerf compliance checker on the log.  The result is
# $LOGDIR/<DATESTAMP>_1.log (result_<i>.txt is that file after redact_logs.sh).
#
# Seeds: each submitted run used a fresh seed drawn from /dev/urandom before the set started (README.md, "Seeds and the set").
set -euo pipefail
: "${CONT:?}" "${DATADIR:?}" "${LOGDIR:?}" "${SEED:?set SEED (a fresh random seed per run)}"
IMPL=$(cd "$(dirname "${BASH_SOURCE[0]}")/../nemo_l1_1012" && pwd)

# submission identity (as in the runs)
export MLPERF_RULESET=6.1.0 MLPERF_SUBMISSION_ORG=Firmus MLPERF_SUBMITTER=Firmus MLPERF_DIVISION=closed
export MLPERF_STATUS="Available on-premise" MLPERF_NUM_NODES=2
export MLPERF_CLUSTER_NAME=CBR01-DH01
export MLPERF_SUBMISSION_PLATFORM=firmus_ngpu8_ngc26.09_nemo   # logged with the node count as 2xfirmus_ngpu8_ngc26.09_nemo
export MLPERF_SYSTEM_NAME="Firmus GB300 NVL72, 2 trays (8x GB300), NGC 26.09"   # as in the ten runs; the system json names the system "Firmus Supermicro GB300 NVL72"
export MLPERF_HOST_STORAGE_TYPE="NVMe SSD"
export MLPERF_HOST_STORAGE_CAPACITY="4x 3.84 TB NVMe (Solidigm SB5PHU7X038T) RAID0 14 TB + 1x 1.92 TB NVMe (Samsung MZ1L21T9HCLS) boot"
export MLPERF_HOST_NETWORKING="NVIDIA ConnectX-8 Ethernet/RoCE (not used for training collectives)"
export MLPERF_HOST_NETWORKING_TOPOLOGY="The accelerators communicate over NVLink (both nodes in one GB300 NVL72 NVLink domain), not over the host network; the Ethernet/RoCE fabric carries management and storage traffic"
export MLPERF_HOST_MEMORY_CONFIGURATION="LPDDR5X on-package, 2x ~490 GB"
export MLPERF_ACCELERATOR_HOST_INTERCONNECT="NVLink-C2C"
export MLPERF_ACCELERATOR_FREQUENCY="2070 MHz max SM clock"
export MLPERF_ACCELERATOR_ON_CHIP_MEMORIES="L2 cache 135528448 bytes (129.25 MiB) per GPU, 152 SMs"
export MLPERF_ACCELERATOR_MEMORY_CONFIGURATION="HBM3e"
export MLPERF_ACCELERATOR_INTERCONNECT="NVLink 5 (18 links x 53.125 GB/s per direction per GPU) + NVLink Switch"
export MLPERF_ACCELERATOR_INTERCONNECT_TOPOLOGY="Multi-node NVLink: both trays in one GB300 NVL72 NVLink/IMEX domain"
export MLPERF_COOLING=liquid
export MLPERF_HW_NOTES="Nodes CBR01-DH01-SVR-CN010429 and CBR01-DH01-SVR-CN010434, 2 x NVIDIA Grace (Arm Neoverse-V2, 72 cores each) and 4 x NVIDIA GB300 per node, both nodes in one GB300 NVL72 NVLink / IMEX domain."
export MLPERF_FRAMEWORK_SHORT_NAME="NeMo Megatron-Bridge 26.09-alpha.rc1 on PyTorch"
# as in the runs: no prolog NCCL test, one experiment per job
export NCCL_TEST=0 ATTEMPT_CUDA_GDB_CORE_DUMP=0 NEXP=1 SEED CONT DATADIR LOGDIR
mkdir -p "$LOGDIR"
cd "$IMPL"
source config_GB300_2x4x2xtp1pp1cp1_8b_fp4.sh
sbatch -N "$DGXNNODES" --time="$WALLTIME" --exclusive --mem=0 --gpus-per-node="$DGXNGPU" \
       --job-name=mlperf-llama31_8b-2x4 --output="$LOGDIR/slurm-%j.out" ${SBATCH_EXTRA:-} run.sub
