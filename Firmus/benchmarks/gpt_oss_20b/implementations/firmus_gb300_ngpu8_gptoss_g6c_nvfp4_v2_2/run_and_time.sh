#!/bin/bash
# Submits the 10 timed runs as one Slurm job. run.sub flushes the page cache on every node before
# each run and aborts the job if the flush fails; the seed base is drawn once from /dev/urandom.
#   CONT=<image>.sqsh DATADIR=<dataset root> LOGDIR=<log dir> NODES=<node1>,<node2> ./run_and_time.sh
set -euo pipefail

: "${CONT:?set CONT to the squashfs built by setup.sh}"
: "${DATADIR:?set DATADIR to the dataset root}"
: "${LOGDIR:?set LOGDIR to the log directory}"
: "${NODES:?set NODES to the two nodes}"
CONFIG=config_GB300_2x4x3xtp1pp1cp1ep1_nvfp4_v7.sh

cd "$(dirname "$0")/../nemo_g6c_nvfp4_v2_2"
set -a; source "./${CONFIG}"; set +a
unset EXPERIMENT_PLAN_DIR
export CONT DATADIR LOGDIR NEXP=10 CLEAR_CACHES=1
export WALLTIME=$((5 + NEXP * (WALLTIME_RUNANDTIME + 5)))
export SEED_BASE=$(( $(od -An -N4 -tu4 /dev/urandom) % 2000000000 + 1 ))
mkdir -p "${LOGDIR}"

jobid=$(sbatch --parsable -N "${DGXNNODES}" -w "${NODES}" --exclusive --gpus-per-node="${DGXNGPU}" \
  -t "${WALLTIME}" -o "${LOGDIR}/slurm-%j.out" run.sub)
echo "seed_base=${SEED_BASE} seed_source=/dev/urandom nexp=${NEXP} walltime=${WALLTIME} image=${CONT} config=${CONFIG} nodes=${NODES} job=${jobid}" | tee "${LOGDIR}/LAUNCH.txt"
