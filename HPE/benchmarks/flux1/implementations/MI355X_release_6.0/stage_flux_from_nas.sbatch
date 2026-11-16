#!/bin/bash
#SBATCH --partition=def
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --comment=docker
#SBATCH --time=6:00:00
#SBATCH --job-name=bruno-flux1-stage
#SBATCH --output=/home/hpebench/slurm-logs/%x-%j.out
#
# Stage the prepared Flux.1 energon tree (2.3 TB) from durable /nas onto this
# node's /lvol0, and load the Flux.1 image from /nas, so run_flux1.sbatch can
# start on a node that has never run Flux.1. Restartable: rsync skips what is
# already complete. Exclusive, so it never shares a node with someone else's job.
#
# Usage: sbatch -p dmoise -w lodin3 stage_flux_from_nas.sbatch
set -uo pipefail
if [[ -d /nas/data/SHARED/MLPERF/flux1_energon ]]; then NAS=/nas/data/SHARED/MLPERF; else NAS=/nas/SHARED/MLPERF; fi
SRC=${NAS}/flux1_energon
DST=${DST:-/lvol0/mlperf/flux1/energon}
CONT=${CONT:-rocm/amd-mlperf:flux1_training_6.0}

echo "=== stage flux1 $(date -u) on $(hostname): ${SRC} -> ${DST}"
for _ in $(seq 1 60); do
    [[ $(squeue -h -w "$(hostname -s)" -t CG | wc -l) -eq 0 ]] && break; sleep 10; done
[[ -d ${SRC}/.nv-meta ]] || { echo "FATAL: ${SRC} is not a prepared energon tree"; exit 1; }
avail=$(df -BG --output=avail /lvol0 | tail -1 | tr -dc 0-9)
[[ ${avail} -gt 2500 ]] || { echo "FATAL: only ${avail} GB free on /lvol0"; exit 1; }
sudo mkdir -p "${DST}" && sudo chown -R "$(id -u):$(id -g)" "$(dirname "${DST}")"

# Small parts first, then the 110 x ~21 GB train shards, 8 at a time.
rsync -a "${SRC}/.nv-meta" "${SRC}/val" "${SRC}/empty_encodings" "${DST}/"
mkdir -p "${DST}/train"
s=$(date +%s)
ls "${SRC}/train" | xargs -P 8 -I{} rsync -a "${SRC}/train/{}" "${DST}/train/{}"
e=$(date +%s)
echo "train copied in $(( (e-s)/60 )) min"

# Verify: names and sizes must match exactly.
( cd "${SRC}" && find . -type f -printf '%P %s\n' | sort ) > /tmp/src.$$
( cd "${DST}" && find . -type f -printf '%P %s\n' | sort ) > /tmp/dst.$$
if diff -q /tmp/src.$$ /tmp/dst.$$ >/dev/null; then echo "VERIFY OK: $(wc -l < /tmp/dst.$$) files"; else echo "FATAL: mismatch"; diff /tmp/src.$$ /tmp/dst.$$ | head; exit 1; fi
rm -f /tmp/src.$$ /tmp/dst.$$

if ! docker image inspect "${CONT}" >/dev/null 2>&1; then
    tar="${NAS}/containers/$(echo "${CONT#rocm/}" | tr '/:' '__').tar"
    echo "loading ${CONT} from ${tar}"; docker load -q -i "${tar}"
fi
docker image inspect "${CONT}" >/dev/null 2>&1 && echo "image OK" || { echo "FATAL: no image"; exit 1; }
echo "=== stage done $(date -u)"
