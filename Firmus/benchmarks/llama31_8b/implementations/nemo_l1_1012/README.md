# nemo_l1_1012: Llama 3.1 8B, NVIDIA's NeMo / Megatron-Bridge recipe as released (Firmus, MLPerf Training v6.1)

System: `firmus_gb300_ngpu8_llama8b_l1_1012`, 2 nodes × 4 NVIDIA GB300 (one GB300 NVL72 NVLink domain).

## What this implementation is

This is NVIDIA's `llama31_8b` reference implementation from the MLPerf Training v6.1 release of 2026-10-12, with its code, Dockerfile and config unchanged. NVIDIA's original instructions are in [README_NVIDIA.md](README_NVIDIA.md).

**What Firmus changed: nothing that runs in the container.**
- **No code change.** Every file NVIDIA's Dockerfile copies into the image is byte-identical to NVIDIA's ([BUILD_CONTEXT.sha256](../firmus_gb300_ngpu8_llama8b_l1_1012/BUILD_CONTEXT.sha256)). The [Dockerfile](Dockerfile) is NVIDIA's, sha256 `f07ac6d2…`, byte-identical to the 20261006 release's, which Firmus's earlier L1 entry ran (`nemo_l1_base`, withdrawn from this submission).
- **The config leaf is NVIDIA's own GB300 leaf, unchanged.** [config_GB300_2x4x2xtp1pp1cp1_8b_fp4.sh](config_GB300_2x4x2xtp1pp1cp1_8b_fp4.sh) (sha256 `0e7d85ae…`) is in the image and in the build context.
  - 2 nodes × 4 GPUs, TP 1, PP 1, CP 1, micro-batch 2 (L7), global batch 16, LR 4e-4 (L15) with 16 warm-up steps (L16), evaluation every 768 steps (L17), NVIDIA's NVFP4 recipe with FP8 attention.
  - It sources NVIDIA's `config_common.sh`, `config_common_cg.sh`, `config_common_8b.sh`, `config_common_fp4.sh` and `config_common_fp8attn.sh` (L1-5), unchanged.
- **The Slurm launcher.** `run.sub` is the Firmus launcher that ran the jobs. It is host-side and not used inside the image. NVIDIA's is kept as `run.sub.vendor1012`, and every difference is listed in [../firmus_gb300_ngpu8_llama8b_l1_1012/run.sub.vs_nvidia1012.diff](../firmus_gb300_ngpu8_llama8b_l1_1012/run.sub.vs_nvidia1012.diff). The Firmus `run.sub` is byte-identical to L1's, NVIDIA's 20261012 `run.sub` is byte-identical to its 20261006 one, and so the diff is L1's.

**L1 and L1_1012.** Both are NVIDIA's llama31_8b recipe, unchanged, on the same two nodes, with the same Dockerfile, `requirements.txt`, launcher and recorded commits. The 20261012 recipe differs from the 20261006 one in `README.md`, `callback_warmup.py`, `pretrain.py` and `config_VR200_2x4x2xtp1pp1cp1_8b.sh`, and adds two GB300 leaves, one of which ran here (from the two `BUILD_CONTEXT.sha256` files; L1's was withdrawn with L1).

## Logged precision

The MLLOG keys `lowest_numerical_precision_in_linear = nvfp4`, `_in_attn = fp8` and `_in_comm = nvfp4` are NVIDIA's values. They come from NVIDIA's stock `config_common_fp4.sh` (L17-19), which this package does not change.
- `_in_linear = nvfp4` and `_in_attn = fp8` are true: every log shows `fp4='e2m1'` with `fp4_recipe='nvfp4'`, and `fp8_dot_product_attention=True`.
- **`_in_comm`: the logged value is `nvfp4`; the actual lowest precision in any communication is `bfloat16`.**
  - Every log shows `fp4_param_gather=False`, `fp8_param_gather=False` and `grad_reduce_in_fp32=False`, with TP, PP and CP all 1.
  - So the distributed optimizer's parameter all-gather and gradient reduce-scatter both run in BF16.
  - The value `nvfp4` comes from `MLPERF_COMM_PRECISION="nvfp4"` in NVIDIA's stock `config_common_fp4.sh` (L19). L1 has the same value for the same reason.
- The ten runs are filed as run, with this note.

## Build

```bash
docker build --build-arg FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:26.09-py3@sha256:6e8ccc607fc2a51e3741667b86316a0889418ba8b78ed0ba96e8d282b648a7a6 \
             --build-arg MBRIDGE_REVISION=ea5dded83078cf7e61ef91960273053cf564d5ac \
             --build-arg TE_REVISION=5b28d7aa2bd639ff0541356cc025f62954f96ce3 \
             -t llama31_8b_l1_1012 ctx
enroot import -o llama31_8b_l1_1012.sqsh dockerd://llama31_8b_l1_1012
```

The build-args pin the Dockerfile's own default base (`nvcr.io/nvidia/pytorch:26.09-py3`) by digest and its two git revisions by commit, without editing the Dockerfile. `ctx` is the build context, recreated from this directory by [../firmus_gb300_ngpu8_llama8b_l1_1012/setup.sh](../firmus_gb300_ngpu8_llama8b_l1_1012/setup.sh), or by hand as in the system README ("Rebuilding from the exact build context"). `setup.sh` also checks the built image's five commits against the table below before the Enroot import.

NVIDIA's Dockerfile and `requirements.txt` name four git dependencies by tag. The image that ran records the commit each tag gave (read from its squashfs), and every one is public at that tag in its official repository (SR-9, SR-15). They are the same commits as in L1's image:

| dependency | tag | commit |
|---|---|---|
| Transformer Engine (`TE_REVISION`) | `v2.20.0` | `5b28d7aa2bd639ff0541356cc025f62954f96ce3` |
| Megatron-Bridge (`MBRIDGE_REVISION`) | `26.09-alpha.rc1` | `ea5dded83078cf7e61ef91960273053cf564d5ac`, with Megatron-LM `f6c33bde4cdc10805ee7157db746d47b758f5f02` as its submodule |
| `mlcommons/logging` | `6.1.0-rc2` | `8a0cc0558ab498b3b003b106d7eae0de19801876` |
| `NVIDIA/mlperf-common` | `training-v6.1-rc0` | `b5014f3c6a2917771e5ddcb183ce46f4eda475dd` |

The Dockerfile clones each of the first two repositories and runs `git checkout` on the revision, so the commit build-args give the same sources as the tags. With them, the image's `CUSTOM_TE_REVISION` variable holds the commit instead of `v2.20.0`; nothing in the recipe reads it.

**The image that produced the results** is NVIDIA's build of this Dockerfile, squashfs sha256 `97a310af019f1006d9aa60196ce81122795852d26ba979d80bbe411e0b8e0c2b` (26,031,730,688 bytes). For provenance only: NVIDIA distributed it as the early-access image `nvcr.io/nvdlfwea/mlperftv61/llama31_8b-arm:20261012`. Firmus has **not** rebuilt this Dockerfile and compared the images (open by decision). L1's rebuild of the same Dockerfile, byte for byte, from the same pinned base matched NVIDIA's 20261006 build in every performance-relevant component; this image records the same commits. The system README ("The image that produced the results") says what was checked.

**This directory compared with the build context.** [../firmus_gb300_ngpu8_llama8b_l1_1012/BUILD_CONTEXT.sha256](../firmus_gb300_ngpu8_llama8b_l1_1012/BUILD_CONTEXT.sha256) lists the 48 files of the build context: the recipe directory in NVIDIA's image (`/workspace/llm`), read from the squashfs. All 48 are in this directory, byte-identical. The only differences:
- `README_NVIDIA.md` is the build's `README.md`, renamed. `README.md` (this file) is not in the build.
- `run.sub.vendor1012` is the build's `run.sub` (NVIDIA's). `run.sub` here is the Firmus launcher that ran the jobs; it is host-side and not used inside the image.
- **No `.dockerignore`.** The recipe directory inside NVIDIA's image has none, and none is needed.
  - The build context is that directory without the 12 files NVIDIA's own build wrote into it after the `COPY`: `embedding_lib/build/` and `embedding_lib/custom_embedding.egg-info/`, created by the Dockerfile's last step (`pip install` of `embedding_lib`).
  - This directory contains no build output, so nothing has to be excluded.
- Every file has mode 0755 in the image; `setup.sh` gives them that mode.

## Launch

See [../firmus_gb300_ngpu8_llama8b_l1_1012/README.md](../firmus_gb300_ngpu8_llama8b_l1_1012/README.md): `setup.sh`, `init_datasets.sh` and `run_and_time.sh`. In short:

```bash
export CONT=$PWD/llama31_8b_l1_1012.sqsh DATADIR=/path/to/data LOGDIR=/path/to/results   # DATADIR: README_NVIDIA.md 3.2
source config_GB300_2x4x2xtp1pp1cp1_8b_fp4.sh
sbatch -N ${DGXNNODES} --time=${WALLTIME} run.sub
```

**`config_filename`.** All ten logs record `config_filename = config_GB300_2x4x2xtp1pp1cp1_8b_fp4.sh`, the leaf in this directory that ran.
- NVIDIA's `pretrain.py` logs `config_${DGXSYSTEM}.sh` (L603).
- The leaf sets `DGXSYSTEM` from its own file name (L21), so the logged name is the leaf's.

## Data indices built before `run_start`

NVIDIA's `callback_logging.py` (`on_data_init_start`, L128-161; byte-identical to L1's) prepares the dataset indices before it logs `run_start`. In every log, `> building train, validation, and test datasets for GPT ...` and `> finished creating GPT datasets ...` come before `run_start`. In `result_0.txt` they are lines 867-871:

```
0: > building train, validation, and test datasets for GPT ...
0: > finished creating GPT datasets ...
0: :::MLLOG {... "key": "init_stop", ... "metadata": {"file": "/workspace/llm/callback_logging.py", "lineno": 161}}
0: :::MLLOG {... "key": "run_start", ... "metadata": {"file": "/workspace/llm/callback_logging.py", "lineno": 161}
0: > building train, validation, and test datasets ...
```

- Rank 0 builds Megatron's document, sample and shuffle indices (`.npy` files in a fresh per-job directory) from the `.idx` metadata: the document count and the document lengths.
- Every rank then reads the `.npy` files once, so they are in the page cache (`_warm_npy_page_cache`, L71-112).
- No training token file (`.bin`) is read. The real dataset setup follows `run_start` (line 871), and the first token read is in the timed run.

Training rules L256-258 allow metadata such as document counts and lengths to be used before the clock starts. L240 asks that preprocessing repeated on every run be timed. The shuffle index depends on the run's seed, but it only orders documents: no token is transformed, augmented or read. This is NVIDIA's code, unchanged; their entry builds its indices the same way.

## Optimizer (TR-9.1c)

The optimizer is NVIDIA's, unchanged: Megatron-Core's `DistributedOptimizer` with Transformer Engine's `FusedAdam` in AdamW mode.
- `pretrain.py` L340-349 sets Adam with β 0.9 / 0.95, ε 1e-5, weight decay 0.1, gradient clipping 1.0 and the distributed optimizer.
- Megatron-Core takes Transformer Engine's `FusedAdam` when it is installed (`megatron/core/optimizer/__init__.py` L14 at `f6c33bde`).

**Allowed implementation.** The rules' list of compliant implementations names `apex.optimizers.FusedAdam` for language models. The submitter has confirmed that Transformer Engine's `FusedAdam`, used here through Megatron-Core's `DistributedOptimizer` in NVIDIA's unchanged code, is an allowed optimizer implementation.

**Supporting evidence.** The L4 package's optimizer A/B ([../nemo_l4_nvfp4_v3/KERNELS/K06_x9step3_fused_adamw.md](../nemo_l4_nvfp4_v3/KERNELS/K06_x9step3_fused_adamw.md), "A/B against TE's and apex's `FusedAdam`", run on L4 v2's image) stepped Transformer Engine's and apex's `FusedAdam` from identical inputs on this recipe's values and found 0 differing bytes in its 64 TE-vs-apex comparisons (of K06's 192). That test ran on Transformer Engine `5b28d7aa`, the commit in this image.
