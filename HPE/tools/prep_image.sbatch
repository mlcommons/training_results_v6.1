#!/bin/bash
#SBATCH --partition=def
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --comment=docker
#SBATCH --time=3:00:00
#SBATCH --job-name=bruno-prep-image
#SBATCH --output=/home/hpebench/slurm-logs/%x-%j.out
#
# Get a Docker image onto this node and keep a durable copy on /nas.
# Order: already present -> load from /nas tar -> pull from Docker Hub (slow,
# ~5 MB/s here) and then save the tar to /nas for the other nodes.
# Then print the versions that matter for MLPerf comparisons.
#
# Usage: sbatch -w <node> --export=ALL,CONT=rocm/primus:v26.7 prep_image.sbatch
set -uo pipefail
if [[ -d /nas/data/SHARED/MLPERF/containers ]]; then NAS=/nas/data/SHARED/MLPERF; else NAS=/nas/SHARED/MLPERF; fi
CONT=${CONT:?set CONT}
tar="${NAS}/containers/$(echo "${CONT#rocm/}" | tr '/:' '__').tar"

for _ in $(seq 1 60); do [[ $(squeue -h -w "$(hostname -s)" -t CG | wc -l) -eq 0 ]] && break; sleep 10; done
docker info >/dev/null 2>&1 || { echo "FATAL: docker daemon not available"; exit 1; }
echo "=== $(date -u +%H:%M) ${CONT} on $(hostname)"
if ! docker image inspect "${CONT}" >/dev/null 2>&1; then
    if [[ -f "${tar}" ]]; then docker load -q -i "${tar}"; else docker pull -q "${CONT}" || { echo "FATAL: pull failed"; exit 1; }; fi
fi
echo "=== $(date -u +%H:%M) image present"
if [[ ! -f "${tar}" ]]; then
    docker save -o "${tar}.part" "${CONT}" && mv "${tar}.part" "${tar}" && chmod 644 "${tar}" && echo "saved ${tar} ($(du -h "${tar}" | cut -f1))"
fi
docker run --rm "${CONT}" bash -c '
  echo "Primus: $(cd /workspace/Primus 2>/dev/null && git log -1 --format="%h %cs %D" 2>/dev/null)"
  echo "ROCm (image): $(cat /opt/rocm/.info/version 2>/dev/null)"
  python -c "import torch; print(\"torch\", torch.__version__)" 2>/dev/null
  pip list 2>/dev/null | grep -iE "^(primus|transformer.engine|aiter|flydsl|megatron|torch |triton|rccl)" 
  ls /workspace/Primus/examples/mlperf 2>/dev/null | tr "\n" " "; echo'
echo "=== $(date -u +%H:%M) done"
