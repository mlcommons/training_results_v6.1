# Container images: how University of Florida built them

This file is the same in every benchmark dir of our submission.

NVIDIA ships the MLPerf Training v6.1 images for arm64 only (`nvcr.io/nvdlfwea/mlperftv61/<bench>-arm:20261006`). Each image holds its own Dockerfile and build context. We copied the build context out of each arm64 image and rebuilt it for x86_64 with `podman build`, then converted it to an Apptainer SIF. The benchmark files in each implementation dir are byte-identical to the build context of its image (`hf_hub/` of DeepSeek excepted, see its `README_UF.md`).

## Images

Every submitted result ran on the unpinned SIF in this table (HPG B200: a copy on `/blue` with the same md5; B300 nodes: the build output). No submitted run used `PINNED=1`.

| Benchmark | NGC image (arm64) | `GIT_COMMIT_ID` | Base | Workload dir in the image | x86 SIF | md5 |
|---|---|---|---|---|---|---|
| DLRMv4 | `dlrmv4-arm:20261006` | `2647c2168340a97ec5b94ae56eb539194bf8a961` | PyTorch 26.09 | `/workspace/dlrmv4` | `dlrmv4-amd_20261006.sif` | `8f1651c347f6fada2a1dfa60a910313b` |
| Flux.1 | `flux1-arm:20261006` | `2647c2168340a97ec5b94ae56eb539194bf8a961` | PyTorch 26.08 | `/workspace/flux` | `flux1-amd_20261006.sif` | `2c92dffa1d68c9a027bbd3a45e4fe7a1` |
| Llama 3.1 8B | `llama31_8b-arm:20261006` | `2647c2168340a97ec5b94ae56eb539194bf8a961` | PyTorch 26.09 | `/workspace/llm` | `llama31_8b-amd_20261006.sif` | `0edb48659139a0cc233017216974204d` |
| GPT-OSS 20B | `gpt_oss_20b-arm:20261006` | `2647c2168340a97ec5b94ae56eb539194bf8a961` | PyTorch 26.09 | `/workspace/llm` | `gpt_oss_20b-amd_20261006.sif` | `c3925cd83e819764ac10821301c3a541` |
| Llama 2 70B LoRA | `llama2_70b_lora-arm:20261006` | `2647c2168340a97ec5b94ae56eb539194bf8a961` | PyTorch 26.09 | `/workspace/ft-llm` | `llama2_70b_lora-amd_20261006.sif` | `89f4d50dc05963854fd8ad577c0e4317` |
| DeepSeek v3 671B | `deepseekv3_671b-arm:20261006` | `2647c2168340a97ec5b94ae56eb539194bf8a961` | PyTorch 26.09 | `/workspace/llm` | `deepseekv3_671b-amd_20261006.sif` | `0738faee0df023e9c02175f7c5c2d5c0` |
| Qwen3.5 397B GRPO | `qwen35_397b_grpo_trtllm-arm:20261006` | `7ad4b1129ea3df8341cae5f5006a51db5eabc399` | PyTorch 26.08 | `/workspace/optimized/qwen35_397b_grpo_trtllm/pytorch` | `qwen35_397b_grpo_trtllm-amd_20261006.sif` | `191670bec980063b9ac52f5e74de4643` |

## Steps

1. Pull the 7 arm64 images into rootless podman storage (`pull_all.sh`).
2. Copy the build context out of each image (`extract.sh <bench> <image> <workload dir>`). The context is the union of the COPY/ADD layers that the benchmark Dockerfile added on top of the NVIDIA base image (`copy_layers.py`). One layer is not enough: buildkit leaves a file out of the `COPY . .` layer when an earlier COPY layer has the same file (example: `requirements.txt`). `extract.sh` also saves the `GIT_COMMIT_ID` of the image.
3. Build each x86 image and its SIF (`build_amd.sh <bench> [podman build args]`, called by `run_all.sh`).

Build arguments and changes to NVIDIA's Dockerfiles:

- All builds: `--build-arg GIT_COMMIT_ID=<value in the arm64 image>`. Everything else is the Dockerfile default.
- DLRMv4: `--build-arg FBGEMM_CUDA_ARCH_LIST="10.0;10.3"` (B200 + B300). NVIDIA uses `10.0;10.7`.
- TE (`NVTE_CUDA_ARCHS="100a;103a;107a"`) and DeepEP (`10.0;10.3;12.0`) already cover B200 and B300. No change.
- Qwen: `build_amd.sh` uses `Dockerfile.x86`, made from NVIDIA's Dockerfile by `make_dockerfile_x86.sh` (3 edits for the x86 build host; the reasons are in the script). NVIDIA's `Dockerfile` in the context is unchanged, and `COPY . .` copies the same files as NVIDIA's build.
- No other Dockerfile ARG has the same name as a base-image ENV (checked for all 7).

## x86 build vs NVIDIA's arm64 image

We compared the Python packages and the image Env of each x86 build with its arm64 image (metadata only):

- Every image: the x86 images also have `mkl`, `mkl-devel`, `mkl-include`, `intel-openmp`, `tbb` (from the x86 base). Env differs only in base-image values (`CUDA_ARCH_LIST`, `TORCH_CUDA_ARCH_LIST`, `MAXSMVER`, `NVIDIA_BUILD_ID`).
- Flux.1 and Qwen: no other difference.
- Llama 3.1 8B, GPT-OSS, LoRA, DeepSeek: 3 unpinned dependencies resolved to newer versions at build time:

  | Package | arm64 (NVIDIA, Oct 6) | x86 (ours, Oct 7) |
  |---|---|---|
  | huggingface-hub | 1.31.0 | 1.32.0 |
  | multidict | 6.8.0 | 6.9.1 |
  | pandas | 3.0.5 | 3.0.6 |

- DLRMv4: `multidict` as above; the `fbgemm-gpu-nightly` version label is 2026.10.6 vs 2026.10.7 (the label is the build date; the FBGEMM commit is pinned: `a5941c1`).

## Scripts

### `pull_all.sh`

Step 1.

```bash
#!/bin/bash
# Pull all NVIDIA MLPerf Training v6.1 arm64 images (tag 20261006) into rootless podman storage.
# Low CPU and I/O priority: b300 is shared and may run timed benchmarks.
set -uo pipefail
TAG=20261006
for r in dlrmv4-arm flux1-arm llama31_8b-arm gpt_oss_20b-arm llama2_70b_lora-arm deepseekv3_671b-arm qwen35_397b_grpo_trtllm-arm; do
    echo "=== $(date) pull $r:$TAG"
    nice -n 19 ionice -c3 podman pull --arch arm64 "nvcr.io/nvdlfwea/mlperftv61/$r:$TAG" || echo "PULL_FAILED $r"
done
echo "=== $(date) PULL_DONE"
```

### `copy_layers.py`

Step 2: lists the COPY/ADD layers.

```python
#!/usr/bin/env python3
"""List the COPY/ADD layers that the benchmark Dockerfile added on top of the NVIDIA base image.

Usage: copy_layers.py <image>
Prints one line per COPY/ADD history entry after the base image:
<layer index> <diff dir> <created_by>.
The base image ends at the last history entry that sets NVIDIA_BUILD_ID or a com.nvidia.build label.
"""
import json
import os
import subprocess
import sys

img = sys.argv[1]
info = json.loads(subprocess.check_output(["podman", "image", "inspect", img]))[0]
diff_ids = info["RootFS"]["Layers"]
history = info["History"]
graph = info["GraphDriver"]["Data"]
store = os.path.expanduser("~/.local/share/containers/storage")
layers = json.load(open(os.path.join(store, "overlay-layers", "layers.json")))
by_diff = {}
for layer in layers:
    if layer.get("diff-digest"):
        by_diff.setdefault(layer["diff-digest"], layer["id"])

base_end = 0
for i, entry in enumerate(history):
    cb = entry.get("created_by", "")
    if "NVIDIA_BUILD_ID" in cb or "com.nvidia.build" in cb:
        base_end = i

# Each non-empty history entry has one layer, in order.
idx = 0
for i, entry in enumerate(history):
    if entry.get("empty_layer"):
        continue
    created_by = entry.get("created_by", "")
    if idx >= len(diff_ids):
        break
    is_copy = created_by.startswith(("COPY", "ADD")) or "#(nop) COPY" in created_by or "#(nop) ADD" in created_by
    if i > base_end and is_copy:
        lid = by_diff.get(diff_ids[idx], "?")
        print(idx, os.path.join(store, "overlay", lid, "diff"), created_by[:200].replace("\n", " "))
    idx += 1
if idx != len(diff_ids):
    print(f"WARNING: {idx} non-empty history entries, {len(diff_ids)} layers", file=sys.stderr)
```

### `extract.sh`

Step 2: copies the context out of the image.

```bash
#!/bin/bash
# Copy the build context of an NVIDIA v6.1 arm64 image out of podman storage.
# The context is the union of the COPY/ADD layers that the benchmark Dockerfile added.
# (Buildkit leaves a file out of the "COPY . ." layer when an earlier COPY layer has the same file.)
# Usage: ./extract.sh <bench> <image> <workdir in image>
# Output: <bench>/layers_root (union of the layers, as a root tree), <bench>/context, <bench>/GIT_COMMIT_ID
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
BENCH=$1 IMG=$2 WORKDIR=$3
mkdir -p "$BENCH"
rm -rf "$BENCH/layers_root" "$BENCH/context"
mkdir -p "$BENCH/layers_root"
python3 -I ./copy_layers.py "$IMG" | tee "$BENCH/copy_layers.txt" | while read -r idx dir rest; do
    rsync -a "$dir/" "$BENCH/layers_root/"
done
rsync -a "$BENCH/layers_root/${WORKDIR#/}/" "$BENCH/context/"
podman image inspect "$IMG" --format '{{range .Config.Env}}{{println .}}{{end}}' \
    | sed -n 's/^GIT_COMMIT_ID=//p' > "$BENCH/GIT_COMMIT_ID"
echo "GIT_COMMIT_ID=$(cat "$BENCH/GIT_COMMIT_ID")"
find "$BENCH/context" -maxdepth 1 | sort
```

### `build_amd.sh`

Step 3: one build.

```bash
#!/bin/bash
# Build an x86_64 image from the Dockerfile inside an NVIDIA v6.1 arm64 image, then convert it to a SIF.
# Same method as ../dlrmv4-build/build.sh (the 20260929 DLRMv4 build).
#
# Usage: ./build_amd.sh <bench> [extra podman build args]
#   <bench>/context        build context, copied out of the arm64 image (see extract.sh)
#   <bench>/GIT_COMMIT_ID  the GIT_COMMIT_ID value in the arm64 image
#   <bench>/Dockerfile.x86 optional: a patched copy of context/Dockerfile (see its make_dockerfile_x86.sh)
# Output: localhost/<bench>-amd:20261006 and <bench>/<bench>-amd_20261006.sif
set -euxo pipefail
cd "$(dirname "$(readlink -f "$0")")"
BENCH=$1
shift
TAG=20261006
IMG=localhost/${BENCH}-amd:${TAG}
SIF=${BENCH}/${BENCH}-amd_${TAG}.sif
TAR=${BENCH}/${BENCH}-amd_${TAG}.tar
# The shared rse-ce nvcr.io login cannot read public images. Pull the public base images without it.
AUTH=(--authfile "$PWD/anon-auth.json")
[[ -f anon-auth.json ]] || echo '{"auths":{}}' > anon-auth.json
GIT_COMMIT_ID=$(cat "${BENCH}/GIT_COMMIT_ID")
DOCKERFILE=${BENCH}/context/Dockerfile
[[ -f "${BENCH}/Dockerfile.x86" ]] && DOCKERFILE=${BENCH}/Dockerfile.x86
date
nice -n 10 podman build "${AUTH[@]}" --format docker \
    --build-arg GIT_COMMIT_ID="${GIT_COMMIT_ID}" "$@" \
    -t "$IMG" -f "${DOCKERFILE}" "${BENCH}/context"
date
podman image inspect "$IMG" --format '{{.Architecture}} {{.Size}}'
podman save --format docker-archive -o "$TAR" "$IMG"
apptainer build --force "$SIF" "docker-archive://$TAR"
rm -f "$TAR"
md5sum "$SIF" > "${SIF}.md5"
date
echo BUILD_DONE
```

### `run_all.sh`

Step 3: all 7 builds, with their build arguments.

```bash
#!/bin/bash
# Build the x86_64 images of all 7 NVIDIA v6.1 benchmarks in 3 parallel streams.
# Each stream runs its builds one after the other. Per-build log: <bench>/build.log.
# Stream 1 builds DLRMv4 first, then Qwen (the TRT-LLM build needs much CPU and RAM,
# so it starts when the FBGEMM build is done).
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")"
export APPTAINER_TMPDIR=$PWD/tmp
mkdir -p "$APPTAINER_TMPDIR"

build() {
    local bench=$1
    shift
    echo "=== $(date) start $bench"
    if ./build_amd.sh "$bench" "$@" > "$bench/build.log" 2>&1; then
        echo "=== $(date) done $bench"
    else
        echo "=== $(date) FAILED $bench"
    fi
}

# DLRMv4: same build args as ../dlrmv4-build/build.sh (10.0 = B200, 10.3 = B300).
( build dlrmv4 --build-arg FBGEMM_CUDA_ARCH_LIST="10.0;10.3"
  build qwen35_397b_grpo_trtllm --jobs 3 ) &
( build flux1
  build llama31_8b
  build llama2_70b_lora ) &
( build gpt_oss_20b
  build deepseekv3_671b ) &
wait
echo "=== $(date) ALL_DONE"
```

### `make_dockerfile_x86.sh`

Qwen only: makes `Dockerfile.x86` from NVIDIA's Dockerfile.

```bash
#!/bin/bash
# Make Dockerfile.x86 from NVIDIA's Qwen Dockerfile (context/Dockerfile stays unchanged).
# The patched file is outside the build context, so "COPY . ." copies the same files as NVIDIA's build.
#
# Edit 1 and 2 (ARG vs inherited ENV):
#   The 26.08 base image sets ENV NIXL_VERSION=1.3.0 and ENV CUTLASS_DSL_VERSION=4.6.2.
#   The Dockerfile declares ARG NIXL_VERSION=v1.4.0 and ARG CUTLASS_DSL_VERSION=4.8.0.dev0.
#   NVIDIA built with buildkit, which used the ARG values in RUN (the arm image has
#   /NIXL_VERSION.env = v1.4.0 and nvidia-cutlass-dsl 4.8.0.dev0). Buildah uses the inherited ENV
#   values instead (git clone -b 1.3.0 fails; pip keeps cutlass-dsl 4.6.2).
#   Fix: put the ARG defaults as literals into the RUN steps that read them. The image Env stays
#   as in NVIDIA's image (NIXL_VERSION=1.3.0, CUTLASS_DSL_VERSION=4.6.2 from the base).
#
# Edit 3 (build host kernel string):
#   OpenHands runs "poetry install". Its poetry.lock has markers 'platform_release >= "20.0"'
#   (macOS-only pyobjc packages). Poetry cannot parse the b300 kernel release
#   "5.14.0-687.48.1.el9_8.x86_64" (RHEL 9) as a version and stops ("Could not parse version
#   constraint"). NVIDIA's build host did not have this problem.
#   Fix: for the openhands.sh call only, LD_PRELOAD a small library that cuts the uname()
#   release string to its numeric part ("5.14.0"). The markers then evaluate to false, as on
#   NVIDIA's host, and the same packages get installed. The library is deleted after the call.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
python3 -I - context/Dockerfile Dockerfile.x86 <<'PY'
import sys

src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()


def sub(old, new):
    global s
    assert s.count(old) == 1, f"expected 1 match: {old!r}"
    s = s.replace(old, new)


# Edit 1
sub('"nvidia-cutlass-dsl[cu13]==${CUTLASS_DSL_VERSION}"', '"nvidia-cutlass-dsl[cu13]==4.8.0.dev0"')
# Edit 2
sub('git clone --depth 1 -b "${NIXL_VERSION}" ', 'git clone --depth 1 -b "v1.4.0" ')
sub('echo "${NIXL_VERSION}" > /NIXL_VERSION.env', 'echo "v1.4.0" > /NIXL_VERSION.env')
# Edit 3
sub(
    "SETUP_DIR=${S}/swe_openhands_setup MINIFORGE_DIR=${S}/swe_openhands_setup/miniforge3 \\\n",
    """# x86 build host: poetry cannot parse the RHEL kernel release; report only its numeric part.
cat > /tmp/uname_release.c <<'CSRC'
#define _GNU_SOURCE
#include <dlfcn.h>
#include <string.h>
#include <sys/utsname.h>
int uname(struct utsname *buf) {
    int (*real_uname)(struct utsname *) = (int (*)(struct utsname *))dlsym(RTLD_NEXT, "uname");
    int rc = real_uname(buf);
    if (rc == 0) buf->release[strspn(buf->release, "0123456789.")] = '\\0';
    return rc;
}
CSRC
gcc -shared -fPIC -O2 -o /tmp/uname_release.so /tmp/uname_release.c -ldl
LD_PRELOAD=/tmp/uname_release.so uname -r
LD_PRELOAD=/tmp/uname_release.so \\
SETUP_DIR=${S}/swe_openhands_setup MINIFORGE_DIR=${S}/swe_openhands_setup/miniforge3 \\
""",
)
sub(
    "    MAKE_BUILD_TIMEOUT_SECONDS=300 setup_scripts/openhands.sh\n",
    "    MAKE_BUILD_TIMEOUT_SECONDS=300 setup_scripts/openhands.sh\nrm -f /tmp/uname_release.so /tmp/uname_release.c\n",
)
open(dst, "w").write(s)
PY
diff context/Dockerfile Dockerfile.x86 || true
```
