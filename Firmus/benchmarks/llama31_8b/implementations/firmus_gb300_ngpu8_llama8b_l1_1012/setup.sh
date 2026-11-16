#!/bin/bash
# setup.sh -- one-time configuration for firmus_gb300_ngpu8_llama8b_l1_1012 (llama31_8b, implementation nemo_l1_1012).
#
# Builds the container from NVIDIA's unmodified 20261012 llama31_8b Dockerfile on the public base nvcr.io/nvidia/pytorch:26.09-py3,
# pinned by digest through the Dockerfile's own FROM_IMAGE_NAME build-arg, and imports it for Slurm + Pyxis/Enroot.  Run on an
# aarch64 host with docker and enroot (the GB300 trays are aarch64).
#
#   TAG=llama31_8b_l1_1012 SQSH=/path/to/llama31_8b_l1_1012.sqsh bash setup.sh
#
# The build context is the exact one in BUILD_CONTEXT.sha256: the implementation directory's files under their as-built names
# (README_NVIDIA.md -> README.md, run.sub.vendor1012 -> run.sub), without the Firmus README and the Firmus run.sub.  It is
# assembled in a temporary directory and checked before the build (README.md, "Rebuilding from the exact build context").
#
# The Dockerfile's two git revisions are passed as the commits the image that ran recorded, through the Dockerfile's own
# build-args, so the build cannot pick up a moved tag: MBRIDGE_REVISION (default 26.09-alpha.rc1) and TE_REVISION (default
# v2.20.0).  Megatron-Core is the Megatron-Bridge commit's submodule.  mlperf-logging and mlperf-common come from
# requirements.txt by tag; the Dockerfile and requirements.txt are used unchanged, so these two are checked after the build
# instead.  The build fails unless all five commits equal the ones recorded in the image that ran.  With the SHA build-arg the
# image's CUSTOM_TE_REVISION variable holds the commit instead of "v2.20.0"; nothing in the recipe reads it.
#
# The results were produced by NVIDIA's build of this Dockerfile, squashfs sha256
# 97a310af019f1006d9aa60196ce81122795852d26ba979d80bbe411e0b8e0c2b.  It has not been rebuilt and compared (README.md, "The
# image that produced the results"); docker image IDs and squashfs hashes are not reproducible from one build to the next.
#
# Host prerequisites for run_and_time.sh (as on the systems that produced the results): Slurm with Pyxis and Enroot; the
# dataset prepared by init_datasets.sh on every node; passwordless `sudo /sbin/sysctl vm.drop_caches=3` for the job user
# (run.sub clears the page cache before each run and fails the job if it cannot).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
IMPL=$(cd "$HERE/../nemo_l1_1012" && pwd)
TAG=${TAG:-llama31_8b_l1_1012}
SQSH=${SQSH:-$PWD/llama31_8b_l1_1012.sqsh}
BASE=nvcr.io/nvidia/pytorch:26.09-py3@sha256:6e8ccc607fc2a51e3741667b86316a0889418ba8b78ed0ba96e8d282b648a7a6
TE_SHA=5b28d7aa2bd639ff0541356cc025f62954f96ce3        # NVIDIA/TransformerEngine, tag v2.20.0
MBRIDGE_SHA=ea5dded83078cf7e61ef91960273053cf564d5ac   # NVIDIA-NeMo/Megatron-Bridge, tag 26.09-alpha.rc1
MCORE_SHA=f6c33bde4cdc10805ee7157db746d47b758f5f02     # NVIDIA/Megatron-LM, the Megatron-Bridge submodule
LOGGING_SHA=8a0cc0558ab498b3b003b106d7eae0de19801876   # mlcommons/logging, tag 6.1.0-rc2
COMMON_SHA=b5014f3c6a2917771e5ddcb183ce46f4eda475dd    # NVIDIA/mlperf-common, tag training-v6.1-rc0
CTX=$(mktemp -d "${TMPDIR:-/tmp}/l1_1012ctx.XXXXXX")
while read -r _ path; do
  f=${path#./}
  case "$f" in
    README.md) src=README_NVIDIA.md ;;
    run.sub)   src=run.sub.vendor1012 ;;
    *)         src=$f ;;
  esac
  install -D -m 0755 "$IMPL/$src" "$CTX/$f"
done < "$HERE/BUILD_CONTEXT.sha256"
( cd "$CTX" && sha256sum -c --quiet "$HERE/BUILD_CONTEXT.sha256" )
[ "$(find "$CTX" -type f | wc -l)" = "$(grep -c . "$HERE/BUILD_CONTEXT.sha256")" ]
echo "build context: $CTX ($(grep -c . "$HERE/BUILD_CONTEXT.sha256") files, matches BUILD_CONTEXT.sha256)"
docker build --build-arg FROM_IMAGE_NAME="$BASE" \
             --build-arg MBRIDGE_REVISION="$MBRIDGE_SHA" \
             --build-arg TE_REVISION="$TE_SHA" \
             -t "$TAG" "$CTX"
want="TE_COMMIT_HASH=$TE_SHA
MBRIDGE_COMMIT_HASH=$MBRIDGE_SHA
MCORE_COMMIT_HASH=$MCORE_SHA
mlperf_logging=$LOGGING_SHA
mlperf_common=$COMMON_SHA"
got=$(docker run --rm --entrypoint python3 "$TAG" -c '
import json, importlib.metadata as md
for n in ("TE", "MBRIDGE", "MCORE"):
    print("%s_COMMIT_HASH=%s" % (n, open("/%s_COMMIT_HASH.env" % n).read().strip()))
for d in ("mlperf_logging", "mlperf_common"):
    print("%s=%s" % (d, json.loads(md.distribution(d).read_text("direct_url.json"))["vcs_info"]["commit_id"]))
')
echo "$got"
if [ "$got" != "$want" ]; then
  echo "setup.sh: the built image's commits differ from the image that produced the results:" >&2
  diff <(echo "$want") <(echo "$got") >&2 || true
  exit 1
fi
echo "commits: all five equal the image that produced the results"
enroot import -o "$SQSH" "dockerd://$TAG"
sha256sum "$SQSH"
echo "export CONT=$SQSH"
