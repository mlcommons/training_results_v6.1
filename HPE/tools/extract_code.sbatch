#!/bin/bash
#SBATCH --partition=def
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --comment=docker
#SBATCH --time=0:20:00
#SBATCH --job-name=extract
#SBATCH --output=/nas/SHARED/MLPERF/results/extract-%j.out
# Copy /workspace/code out of a benchmark image. Usage: sbatch extract_code.sbatch <image> <dest>
set -uo pipefail
for _ in $(seq 1 60); do [[ $(squeue -h -w "$(hostname -s)" -t CG | wc -l) -eq 0 ]] && break; sleep 10; done
mkdir -p "$2"; c=$(docker create "$1") && docker cp "$c:/workspace/code/." "$2/" && docker rm "$c" >/dev/null
find "$2" -type f | wc -l
