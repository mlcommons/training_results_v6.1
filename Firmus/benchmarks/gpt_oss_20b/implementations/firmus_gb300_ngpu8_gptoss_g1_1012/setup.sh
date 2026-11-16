#!/bin/bash
# setup.sh -- one-time configuration for firmus_gb300_ngpu8_gptoss_g1_1012 (gpt_oss_20b, implementation nemo_g1_1012).
#
# Builds the container from NVIDIA's unmodified 20261012 gpt_oss_20b Dockerfile on the public base nvcr.io/nvidia/pytorch:26.09-py3,
# and imports it for Slurm + Pyxis/Enroot.  Run on an aarch64 host with docker and enroot (the GB300 trays are aarch64).
#
#   TAG=gpt_oss_20b_g1_1012 SQSH=/path/to/gpt_oss_20b_g1_1012.sqsh bash setup.sh
#
# Every input is pinned without editing the Dockerfile.  Its own build-args take the base by digest (FROM_IMAGE_NAME) and
# Megatron-Bridge and Transformer Engine by commit (MBRIDGE_REVISION, TE_REVISION; the Dockerfile's defaults are the tags
# 26.09-alpha.rc1 and v2.19, which gave these commits in the image that ran).  DeepEP is pinned by commit in the Dockerfile.
# requirements.txt stays NVIDIA's, so its two git lines keep their tags (6.1.0-rc2, training-v6.1-rc0); after the build the
# script fails unless the image holds exactly the commits of the image that ran: /MBRIDGE_COMMIT_HASH.env,
# /MCORE_COMMIT_HASH.env, /TE_COMMIT_HASH.env, DeepEP's checkout, and the commit_id in pip's direct_url.json of mlperf-logging
# and mlperf-common.
#
# The build context is the exact one in BUILD_CONTEXT.sha256: the implementation directory's files under their as-built names
# (README_NVIDIA.md -> README.md, run.sub.vendor1012 -> run.sub), without the Firmus README, the Firmus run.sub and
# stack_files.txt.  It is assembled in a temporary directory and checked before the build (README.md, "Rebuilding from the
# exact build context").
#
# The results were produced by NVIDIA's build of this Dockerfile, squashfs sha256
# 3b3a1c6669bb3cf25bc5027d01b101d9673c30c169d8d354f0653f38831397a6.  It has not been rebuilt and compared (README.md, "The
# image that produced the results"); docker image IDs and squashfs hashes are not reproducible from one build to the next.
#
# Host prerequisites for run_and_time.sh (as on the systems that produced the results): Slurm with Pyxis and Enroot; the
# dataset prepared by init_datasets.sh on every node; passwordless `sudo /sbin/sysctl vm.drop_caches=3` for the job user
# (run.sub clears the page cache before each run and fails the job if it cannot).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
IMPL=$(cd "$HERE/../nemo_g1_1012" && pwd)
TAG=${TAG:-gpt_oss_20b_g1_1012}
SQSH=${SQSH:-$PWD/gpt_oss_20b_g1_1012.sqsh}
BASE=nvcr.io/nvidia/pytorch:26.09-py3@sha256:6e8ccc607fc2a51e3741667b86316a0889418ba8b78ed0ba96e8d282b648a7a6
MBRIDGE_COMMIT=ea5dded83078cf7e61ef91960273053cf564d5ac        # NVIDIA-NeMo/Megatron-Bridge, tag 26.09-alpha.rc1
MCORE_COMMIT=f6c33bde4cdc10805ee7157db746d47b758f5f02          # NVIDIA/Megatron-LM, Megatron-Bridge's submodule at that commit
TE_COMMIT=5e52befd5262c06289106338c308079d6adb391f             # NVIDIA/TransformerEngine, tag v2.19
DEEPEP_COMMIT=58397eac493b075b0baea5861ac6e9793d8d7a21         # deepseek-ai/DeepEP, branch hybrid-ep (pinned in the Dockerfile)
MLPERF_LOGGING_COMMIT=8a0cc0558ab498b3b003b106d7eae0de19801876 # mlcommons/logging, tag 6.1.0-rc2
MLPERF_COMMON_COMMIT=b5014f3c6a2917771e5ddcb183ce46f4eda475dd  # NVIDIA/mlperf-common, tag training-v6.1-rc0

CTX=$(mktemp -d -t g1_1012ctx.XXXXXX)
while read -r _ path; do
  f=${path#./}
  case "$f" in
    README.md) src=README_NVIDIA.md ;;
    run.sub)   src=run.sub.vendor1012 ;;
    *)         src=$f ;;
  esac
  case "$f" in   # their modes in NVIDIA's image; every other file is 0755
    config_VR200_2x4x3xtp1pp1cp1ep1_mxfp8.sh|opt_kernels/__init__.py|opt_kernels/qknorm_rope.py|opt_kernels/qknorm_rope_patch.py) mode=0644 ;;
    *) mode=0755 ;;
  esac
  install -D -m "$mode" "$IMPL/$src" "$CTX/$f"
done < "$HERE/BUILD_CONTEXT.sha256"
( cd "$CTX" && sha256sum -c --quiet "$HERE/BUILD_CONTEXT.sha256" )
[ "$(find "$CTX" -type f | wc -l)" = "$(grep -c . "$HERE/BUILD_CONTEXT.sha256")" ]
echo "build context: $CTX ($(grep -c . "$HERE/BUILD_CONTEXT.sha256") files, matches BUILD_CONTEXT.sha256)"

docker build --build-arg FROM_IMAGE_NAME="$BASE" \
             --build-arg MBRIDGE_REVISION="$MBRIDGE_COMMIT" \
             --build-arg TE_REVISION="$TE_COMMIT" \
             -t "$TAG" "$CTX"

expect() {
  if [ "$2" = "$3" ]; then
    echo "OK    $1 $2"
  else
    echo "FAIL  $1 is '$2', expected $3" >&2
    exit 1
  fi
}
img() { docker run --rm --entrypoint "$1" "$TAG" "${@:2}"; }
expect /MBRIDGE_COMMIT_HASH.env "$(img cat /MBRIDGE_COMMIT_HASH.env)" "$MBRIDGE_COMMIT"
expect /MCORE_COMMIT_HASH.env "$(img cat /MCORE_COMMIT_HASH.env)" "$MCORE_COMMIT"
expect /TE_COMMIT_HASH.env "$(img cat /TE_COMMIT_HASH.env)" "$TE_COMMIT"
expect "DeepEP checkout" "$(img git -C /workspace/DeepEP rev-parse HEAD)" "$DEEPEP_COMMIT"
direct_url_commit() {
  img python3 -c 'import importlib.metadata as m, json, sys
print(json.loads(m.distribution(sys.argv[1]).read_text("direct_url.json"))["vcs_info"]["commit_id"])' "$1"
}
expect "mlperf-logging direct_url.json" "$(direct_url_commit mlperf-logging)" "$MLPERF_LOGGING_COMMIT"
expect "mlperf-common direct_url.json" "$(direct_url_commit mlperf-common)" "$MLPERF_COMMON_COMMIT"

enroot import -o "$SQSH" "dockerd://$TAG"
sha256sum "$SQSH"
echo "export CONT=$SQSH"
