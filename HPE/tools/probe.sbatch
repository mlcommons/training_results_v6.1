#!/bin/bash
#SBATCH --partition=def
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --exclusive
#SBATCH --comment=docker
#SBATCH --time=0:40:00
#SBATCH --job-name=probe
#SBATCH --output=/nas/SHARED/MLPERF/probes/probe-%j.out
# Usage: sbatch -w <node> probe.sbatch <label> [VAR=VAL ...]
set -uo pipefail
for _ in $(seq 1 60); do
  [[ $(squeue -h -w "$(hostname -s)" -t CG | wc -l) -eq 0 ]] && break; sleep 10; done
docker info >/dev/null 2>&1 || { echo "FATAL: no docker"; exit 1; }
REPO=${REPO:-$HOME/mlcommons_mlperf_training}
bash "${REPO}/HPE/benchmarks/llama31_8b/implementations/MI355X_release_6.0/probe.sh" "$@"
cat /nas/SHARED/MLPERF/probes/$1/result.json
