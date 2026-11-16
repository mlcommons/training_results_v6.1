#!/bin/bash
#SBATCH --partition=def
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --comment=docker
#SBATCH --time=4:00:00
#SBATCH --job-name=bruno-docker-save
#SBATCH --output=/nas/SHARED/MLPERF/containers/save-%N-%j.out
#
# Save Docker images to durable /nas/SHARED/MLPERF/containers, in the same
# format as the files already there (plain `docker save` .tar, named
# <repo-without-rocm/>_<tag>.tar). On lodin the Docker data-root is on the
# RAM-backed tmpfs, so images are lost at reboot, and Docker Hub runs at only
# ~5 MB/s here. Restore with HPE/tools/docker_images_load.sh.
#
# Usage: sbatch -w <node> docker_images_save.sbatch <image> [<image> ...]
# Skips an image whose exact ID is already saved in any .tar there.
set -uo pipefail
DEST=${DEST:-/nas/SHARED/MLPERF/containers}
mkdir -p "${DEST}"

# A CG job's epilog runs disable-docker.sh: wait for the node to be clear.
for _ in $(seq 1 60); do
    [[ $(squeue -h -w "$(hostname -s)" -t CG | wc -l) -eq 0 ]] && break; sleep 10; done
docker info >/dev/null 2>&1 || { echo "FATAL: docker daemon not available"; exit 1; }

saved_ids() {   # image config IDs of every saved tar
    for t in "${DEST}"/*.tar; do
        tar -xOf "$t" manifest.json 2>/dev/null \
            | python3 -c 'import json,sys; print(json.load(sys.stdin)[0]["Config"].split("/")[-1].replace(".json",""))' 2>/dev/null
    done
}
have=$(saved_ids)

rc=0
for img in "$@"; do
    id=$(docker image inspect --format '{{.Id}}' "${img}" 2>/dev/null) || { echo "SKIP ${img}: not on $(hostname)"; continue; }
    if grep -q "${id#sha256:}" <<<"${have}"; then
        echo "SKIP ${img}: already saved (${id:7:12})"; continue
    fi
    file="${DEST}/$(echo "${img#rocm/}" | tr '/:' '__').tar"
    echo "--- ${img} (${id:7:12}) -> ${file}  $(date -u +%H:%M:%S)"
    # Temp name + rename, so a killed job never leaves a file that looks complete.
    if docker save -o "${file}.part" "${img}" && tar -tf "${file}.part" manifest.json >/dev/null; then
        mv "${file}.part" "${file}"
        echo "    ok $(du -h "${file}" | cut -f1)  $(date -u +%H:%M:%S)"
    else
        echo "    FAILED ${img}"; rm -f "${file}.part"; rc=1
    fi
done
ls -la "${DEST}"/*.tar
exit ${rc}
