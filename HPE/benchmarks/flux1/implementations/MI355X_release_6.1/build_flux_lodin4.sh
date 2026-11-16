#!/bin/bash
# HPE, lodin4 (user monnetb): build the FLUX.1 6.1 image locally.
# /nas is read-only for monnetb, so the durable tar comes from the lodin1 build
# (HPE/tools/build_image.sbatch). Base = rocm/primus:v26.7 from the /nas tar,
# checked to have the layers of the image the Dockerfile pins by digest.
#
# lodin4 deletes all Docker images at job end: load the tar saved in $HOME, else
# build and save it there (monnetb cannot write /nas/data/SHARED).
# Usage: srun --jobid <lodin4 job> --overlap bash build_flux_lodin4.sh
set -uo pipefail
CONT=${CONT:-rocm/amd-mlperf:flux1_training_6.1}
CTX=$(dirname "$(readlink -f "$0")")
echo "=== $(date -u +%H:%M:%S) $(hostname) ${CONT} from ${CTX}"
if docker image inspect "${CONT}" >/dev/null 2>&1; then echo "image present"; exit 0; fi
TAR=${HOME}/containers/amd-mlperf_flux1_training_6.1.tar
if [[ -f "${TAR}" ]]; then docker load -q -i "${TAR}" && exit 0; fi
# lodin4's Docker (containerd store) pulls the digest-pinned base during the
# build anyway, so loading the /nas tar first only costs time: skip it.
if [[ "${SKIP_BASE_LOAD:-1}" == 1 ]]; then
    DOCKER_BUILDKIT=1 docker build --network=host --progress=plain -t "${CONT}" \
        --build-arg MAX_JOBS=128 --build-arg RCCL_MAX_JOBS=128 "${CTX}" || exit 1
    mkdir -p "$(dirname "${TAR}")"
    docker save -o "${TAR}.part" "${CONT}" && mv "${TAR}.part" "${TAR}" && echo "saved ${TAR} ($(du -h "${TAR}" | cut -f1))"
    exit 0
fi
if ! docker image inspect rocm/primus:v26.7 >/dev/null 2>&1; then
    docker load -q -i /nas/data/SHARED/MLPERF/containers/primus_v26.7.tar || exit 1
fi
# Compare layer contents, not the image ID: with Docker's containerd image store
# (lodin4) the ID is a manifest digest, which a tar load regenerates.
# Reference = sha256 of the concatenated rootfs diff_ids of config 8087a2526fdc.
lay=$(docker image inspect --format '{{range .RootFS.Layers}}{{.}}{{end}}' rocm/primus:v26.7 | tr -d '\n' | sha256sum | cut -d' ' -f1)
[[ ${lay} == d696a5289da738e435a23ffbdb6ef443f8b195acdf91a7e5f556fd044fbcb453 ]] || { echo "FATAL: base layers ${lay}"; exit 1; }
echo "=== $(date -u +%H:%M:%S) base OK, build"
DOCKER_BUILDKIT=1 docker build --network=host --progress=plain -t "${CONT}" \
    --build-arg BASE_IMAGE=rocm/primus:v26.7 --build-arg MAX_JOBS=128 --build-arg RCCL_MAX_JOBS=128 "${CTX}" || exit 1
echo "=== $(date -u +%H:%M:%S) built $(docker image inspect --format '{{.Id}}' "${CONT}")"
