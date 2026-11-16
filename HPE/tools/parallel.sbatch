#!/bin/bash
#SBATCH --partition=def
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --comment=docker
#SBATCH --time=8:00:00
#SBATCH --job-name=bruno-parallel
#SBATCH --output=/home/hpebench/slurm-logs/%x-%j.out
#
# Run independent tasks side by side in one allocation, e.g. a disk-bound data
# staging next to a CPU-bound image build. Each argument is one bash command
# line; its output goes to <slurm-logs>/<jobname>-<jobid>.<n>.log.
# Exit code: the highest of the tasks.
#
# Usage: sbatch -w <node> parallel.sbatch "VAR=x bash a.sbatch" "bash b.sbatch"
set -uo pipefail
base=/home/hpebench/slurm-logs/${SLURM_JOB_NAME:-parallel}-${SLURM_JOB_ID:-$$}
pids=()
n=0
for cmd in "$@"; do
    n=$((n + 1))
    echo "=== $(date -u +%H:%M:%S) task ${n}: ${cmd} -> ${base}.${n}.log"
    bash -c "${cmd}" > "${base}.${n}.log" 2>&1 &
    pids+=($!)
done
rc=0
n=0
for p in "${pids[@]}"; do
    n=$((n + 1))
    wait "$p"; r=$?
    echo "=== $(date -u +%H:%M:%S) task ${n} rc=${r}"
    (( r > rc )) && rc=$r
done
exit ${rc}
