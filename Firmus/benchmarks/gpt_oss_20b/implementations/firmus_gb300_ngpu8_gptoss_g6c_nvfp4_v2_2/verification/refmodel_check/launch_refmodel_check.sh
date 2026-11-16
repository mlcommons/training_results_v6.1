#!/bin/bash
# Classification: Internal
# One run of the submitted image and config through the unchanged run.sub / run_and_time.sh /
# pretrain.py, with refmodel_check.py hooked into the logging callback (via sitecustomize.py on
# PYTHONPATH). Only for this check, evaluation runs every REFCHECK_EVAL_INTERVAL steps (default 10,
# instead of 512) so that a short run passes through evaluation several times; the run stops after
# REFCHECK_STOP_STEP steps (default 30).
#   CONT=<image>.sqsh DATADIR=<dataset root> LOGDIR=<log dir> NODES=<node1>,<node2> ./launch_refmodel_check.sh
set -euo pipefail

: "${CONT:?set CONT to the squashfs built by setup.sh}"
: "${DATADIR:?set DATADIR to the dataset root}"
: "${LOGDIR:?set LOGDIR to the log directory}"
: "${NODES:?set NODES to the two nodes}"
CONFIG=config_GB300_2x4x3xtp1pp1cp1ep1_nvfp4_v7.sh
HERE=$(cd "$(dirname "$0")" && pwd)

cd "${HERE}/../../../nemo_g6c_nvfp4_v2_2"
set -a; source "./${CONFIG}"; set +a
unset EXPERIMENT_PLAN_DIR
export CONT DATADIR LOGDIR NEXP=1 CLEAR_CACHES=1
export EXTRA_MOUNTS="${HERE}:/refcheck:ro" PYTHONPATH=/refcheck REFCHECK=1
export REFCHECK_STOP_STEP=${REFCHECK_STOP_STEP:-30}
export VAL_CHECK_INTERVAL=${REFCHECK_EVAL_INTERVAL:-10}
export WALLTIME_RUNANDTIME=40 WALLTIME=50
export SEED_BASE=$(( $(od -An -N4 -tu4 /dev/urandom) % 2000000000 + 1 ))
mkdir -p "${LOGDIR}"

jobid=$(sbatch --parsable -N "${DGXNNODES}" -w "${NODES}" --exclusive --gpus-per-node="${DGXNGPU}" \
  -t "${WALLTIME}" -J g6c-refcheck -o "${LOGDIR}/slurm-%j.out" run.sub)
echo "refcheck stop_step=${REFCHECK_STOP_STEP} eval_interval=${VAL_CHECK_INTERVAL} seed_base=${SEED_BASE} seed_source=/dev/urandom image=${CONT} config=${CONFIG} nodes=${NODES} job=${jobid}" | tee "${LOGDIR}/LAUNCH.txt"
