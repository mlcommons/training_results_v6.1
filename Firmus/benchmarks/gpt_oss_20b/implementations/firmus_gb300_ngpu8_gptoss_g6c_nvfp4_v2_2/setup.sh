#!/bin/bash
# One-time setup: build the container image from the Dockerfile in the implementation directory
# and convert it to an enroot squashfs that Pyxis can start.
#   IMAGE=<image>:<tag> SQSH=<path>.sqsh [BUILD_LOG_DIR=<dir>] [DOCKER_BUILD_ARGS=...] ./setup.sh
# BUILD_LOG_DIR receives the build log, image ID, pip list, commit-hash files and squashfs sha256.
# DOCKER_BUILD_ARGS e.g. "--build-arg BUILD_JOBS=48" (compile parallelism only).
set -euo pipefail

: "${IMAGE:?set IMAGE to the docker tag to build}"
: "${SQSH:?set SQSH to the output squashfs path}"
IMPL="$(cd "$(dirname "$0")/../nemo_g6c_nvfp4_v2_2" && pwd)"
LOGS="${BUILD_LOG_DIR:-.}"
mkdir -p "${LOGS}"

docker build --progress=plain ${DOCKER_BUILD_ARGS:-} -t "${IMAGE}" "${IMPL}" 2>&1 | tee "${LOGS}/docker_build.log"
docker image inspect --format '{{.Id}}' "${IMAGE}" | tee "${LOGS}/image_id.txt"
docker run --rm "${IMAGE}" bash -c 'pip list --format=freeze; for f in /*_COMMIT_HASH.env; do echo "$f $(cat "$f")"; done' \
  > "${LOGS}/image_contents.txt"
enroot import -o "${SQSH}" "dockerd://${IMAGE}"
sha256sum "${SQSH}" | tee "${LOGS}/squashfs_sha256.txt"
