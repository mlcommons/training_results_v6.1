#!/bin/bash
# setup.sh -- one-time configuration for firmus_gb300_ngpu8_llama8b_l4_nvfp4_v3 (llama31_8b, implementation nemo_l4_nvfp4_v3).
#
# Builds the container from the implementation directory (Dockerfile: NVIDIA's 20261006 llama31_8b recipe on the public
# nvcr.io/nvidia/pytorch:26.09-py3 base, pinned by digest, plus the Firmus patches and kernels compiled from source) and
# imports it for Slurm + Pyxis/Enroot.  Run on an aarch64 host with docker and enroot (the GB300 trays are aarch64).
#
#   TAG=llama31_8b_l4 SQSH=/path/to/llama31_8b_l4.sqsh bash setup.sh
#
# The results were produced by the image of 2026-10-09: docker image sha256:edde38cbf4c6..., squashfs sha256
# 23469bf756c0886ef3377d18badcbf6ef3843961005f6a11228d915b5795d519.  A rebuild compiles the CUDA extensions again, and
# compiled binaries are not byte-reproducible; /FIRMUS_AK_BUILD_SHA256.txt in the image records the bytes it built.
#
# Host prerequisites for run_and_time.sh (as on the systems that produced the results): Slurm with Pyxis and Enroot; the
# dataset prepared by init_datasets.sh on every node; passwordless `sudo /sbin/sysctl vm.drop_caches=3` for the job user
# (run.sub clears the page cache before each run and fails the job if it cannot).
set -euo pipefail
IMPL=$(cd "$(dirname "${BASH_SOURCE[0]}")/../nemo_l4_nvfp4_v3" && pwd)
TAG=${TAG:-llama31_8b_l4}
SQSH=${SQSH:-$PWD/llama31_8b_l4.sqsh}
docker build -t "$TAG" "$IMPL"
docker run --rm --entrypoint cat "$TAG" /FIRMUS_AK.env
enroot import -o "$SQSH" "dockerd://$TAG"
sha256sum "$SQSH"
echo "export CONT=$SQSH"
