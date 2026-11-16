# nemo_g1_1012: gpt-oss-20b, NVIDIA's NeMo / Megatron-Bridge recipe, unchanged (Firmus, MLPerf Training v6.1)

System: `firmus_gb300_ngpu8_gptoss_g1_1012`, 2 nodes × 4 NVIDIA GB300 (one GB300 NVL72 NVLink domain).

## What this implementation is

This is NVIDIA's `gpt_oss_20b` reference implementation from the MLPerf Training v6.1 release of 2026-10-12, with its code, Dockerfile and config unchanged. NVIDIA's original instructions are in [README_NVIDIA.md](README_NVIDIA.md).

**What Firmus changed: nothing that runs in the container.**
- **No code change.** Every file NVIDIA's Dockerfile copies into the image is byte-identical to NVIDIA's ([BUILD_CONTEXT.sha256](../firmus_gb300_ngpu8_gptoss_g1_1012/BUILD_CONTEXT.sha256)). The [Dockerfile](Dockerfile) is NVIDIA's, sha256 `11665bc6…`, byte-identical to the 20261006 release's, which Firmus's earlier G1 entry ran (`nemo_g1_base`, withdrawn from this submission).
- **The config leaf is NVIDIA's own GB300 leaf, unchanged.** [config_GB300_2x4x3xtp1pp1cp1ep1_mxfp8.sh](config_GB300_2x4x3xtp1pp1cp1ep1_mxfp8.sh) (sha256 `f21c5265…`) is in the image and in the build context.
  - 2 nodes × 4 GPUs, TP 1, PP 1, CP 1, EP 1, micro-batch 3 (L15), global batch 24, LR 5e-4 (L35) with 256 warm-up steps (L37), evaluation every 512 steps (L36, 12,288 samples);
  - NVIDIA's MXFP8 recipe with FP8 attention, and the HybridEP MoE dispatcher.
  - It sources NVIDIA's `config_common.sh`, `config_common_mxfp8.sh` and `config_common_cg.sh` (L1-3), unchanged.
- **The Slurm launcher.** `run.sub` is the Firmus launcher that ran the jobs. It is host-side and not used inside the image. NVIDIA's is kept as `run.sub.vendor1012`, and every difference is listed in [../firmus_gb300_ngpu8_gptoss_g1_1012/run.sub.vs_nvidia1012.diff](../firmus_gb300_ngpu8_gptoss_g1_1012/run.sub.vs_nvidia1012.diff). The Firmus `run.sub` is byte-identical to G1's, NVIDIA's 20261012 `run.sub` is byte-identical to its 20261006 one, and so the diff is G1's. The changes: ranks are placed from a sorted host file (`--distribution=arbitrary`), and `--network=sharp` is passed; Enroot's data, cache and runtime paths go to node-local disk; the page-cache clear fails the job if the drop command fails or the cache does not shrink; the VBIOS and NVLink fabric state are printed; NVIDIA's NCCL Inspector preflight, peak-memory-clock option and container-image override are not in it, and `NCCL_NVLS_ENABLE` is passed into the container in place of `NCCL_COLLNET_ENABLE`. `run.sub` hashes the files listed in [stack_files.txt](stack_files.txt) inside the container before the run (`:::STACK_SHA256` lines); the list is G1's plus the three `opt_kernels/` files.

**G1 and G1_1012.** Both are NVIDIA's gpt_oss_20b recipe, unchanged, on the same two nodes, with the same Dockerfile, `requirements.txt`, launcher and recorded commits. The 20261012 recipe differs from the 20261006 one in `README.md`, `callback_warmup.py`, `conf/gpt_oss.yaml`, `pretrain.py` and the two VR200 leaves; it adds two GB300 leaves, one of which ran here, and `opt_kernels/` (from the two `BUILD_CONTEXT.sha256` files; G1's was withdrawn with G1).
- **Model definition.** Firmus compared the 20261012 recipe's model with the MLCommons reference `small_llm_moe_pretraining/primus` (mlcommons/training `7bf4fbd2`, resolved through AMD-AIG-AIMA/Primus `b369037b`): no linear bias, QK layernorm, vanilla softmax, SwiGLU, sliding window (128, 0) on alternating layers, RoPE base 150000 without YaRN, EOD mask loss, ε 1e-6, init std 0.008. They match ([evidence](../../../../../evidence/firmus_gb300_ngpu8_gptoss_g1_1012/README_EVIDENCE.md)). The 20261006 recipe that G1 ran did not match in these points: linear bias, learnable softmax, clamped quick-GELU, window 127, YaRN.
- **`opt_kernels/`** is NVIDIA's fused QK-norm + RoPE kernel (Triton), imported by `pretrain.py` (L48) and on by default (`OPT_QKNORM_ROPE`, default `1`, `opt_kernels/qknorm_rope_patch.py` L19). Every log shows `[opt qknorm_rope] installed`, and its three files' sha256 in the `:::STACK_SHA256` lines are the image's. It is part of NVIDIA's recipe; Firmus did not add or change it.

## Logged precision

The MLLOG keys `lowest_numerical_precision_in_linear = fp8`, `_in_attn = fp8` and `_in_comm = bfloat16` are NVIDIA's values, from the stock `config_common.sh` (L109-111), `config_common_mxfp8.sh` (L8) and the leaf (L12). All three are true:
- linear layers run in MXFP8: every log shows `fp8='e4m3'` with `fp8_recipe='mxfp8'`;
- attention runs in FP8: `fp8_dot_product_attention` is `True` in every log;
- communication is BF16: every log shows `fp8_param_gather=False`, `fp4_param_gather=False` and `grad_reduce_in_fp32=False`, with TP, PP, CP and EP all 1, so the distributed optimizer's parameter all-gather and gradient reduce-scatter both run in BF16.

## Build

```bash
docker build --build-arg FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:26.09-py3@sha256:6e8ccc607fc2a51e3741667b86316a0889418ba8b78ed0ba96e8d282b648a7a6 \
             --build-arg MBRIDGE_REVISION=ea5dded83078cf7e61ef91960273053cf564d5ac \
             --build-arg TE_REVISION=5e52befd5262c06289106338c308079d6adb391f -t gpt_oss_20b_g1_1012 ctx
enroot import -o gpt_oss_20b_g1_1012.sqsh dockerd://gpt_oss_20b_g1_1012
```

The build-args pin the Dockerfile's own defaults without editing it: the base `nvcr.io/nvidia/pytorch:26.09-py3` by digest, and the Megatron-Bridge and Transformer Engine tags by the commits they gave in the image that ran. `ctx` is the build context, recreated from this directory by [../firmus_gb300_ngpu8_gptoss_g1_1012/setup.sh](../firmus_gb300_ngpu8_gptoss_g1_1012/setup.sh), or by hand as in the system README ("Rebuilding from the exact build context"). `setup.sh` also fails unless the built image holds the commits below.

NVIDIA's Dockerfile and `requirements.txt` name five git dependencies, four of them by tag. The image that ran records the commit each gave (read from its squashfs), and every one is public in its official repository (SR-9, SR-15). They are the same commits as in G1's image:

| dependency | Dockerfile / requirements.txt | commit in the image that ran |
|---|---|---|
| Megatron-Bridge (`MBRIDGE_REVISION`) | tag `26.09-alpha.rc1` | `ea5dded83078cf7e61ef91960273053cf564d5ac`, with Megatron-LM `f6c33bde4cdc10805ee7157db746d47b758f5f02` as its submodule |
| Transformer Engine (`TE_REVISION`) | tag `v2.19` | `5e52befd5262c06289106338c308079d6adb391f` |
| DeepEP, `hybrid-ep` branch (HybridEP) | commit `58397eac493b075b0baea5861ac6e9793d8d7a21` | the same |
| `mlcommons/logging` | tag `6.1.0-rc2` | `8a0cc0558ab498b3b003b106d7eae0de19801876` |
| `NVIDIA/mlperf-common` | tag `training-v6.1-rc0` | `b5014f3c6a2917771e5ddcb183ce46f4eda475dd` |

`requirements.txt` stays NVIDIA's, byte for byte, so its two git lines keep their tags; `setup.sh` checks the installed commits instead (pip's `direct_url.json`). A rebuild with the commit build-args records `CUSTOM_TE_REVISION=5e52befd…` in the image environment where the image that ran has `v2.19`; it is the Dockerfile's label of the same revision.

**The image that produced the results** is NVIDIA's build of this Dockerfile, squashfs sha256 `3b3a1c6669bb3cf25bc5027d01b101d9673c30c169d8d354f0653f38831397a6` (25,965,891,584 bytes). For provenance only: NVIDIA distributed it as the early-access image `nvcr.io/nvdlfwea/mlperftv61/gpt_oss_20b-arm:20261012`. Firmus has **not** rebuilt this Dockerfile and compared the images (open by decision, as for G1). The system README ("The image that produced the results") says what was checked.

**This directory compared with the build context.** [../firmus_gb300_ngpu8_gptoss_g1_1012/BUILD_CONTEXT.sha256](../firmus_gb300_ngpu8_gptoss_g1_1012/BUILD_CONTEXT.sha256) lists the 44 files of the build context: the recipe directory in NVIDIA's image (`/workspace/llm`), read from the squashfs. All 44 are in this directory, byte-identical. The only differences:
- `README_NVIDIA.md` is the build's `README.md`, renamed. `README.md` (this file) is not in the build.
- `run.sub.vendor1012` is the build's `run.sub` (NVIDIA's). `run.sub` here is the Firmus launcher that ran the jobs; it is host-side and not used inside the image.
- `stack_files.txt`, the list `run.sub` hashes, is not in the build. It is read on the host.
- **No `.dockerignore`.** The recipe directory inside NVIDIA's image has none, and none is needed.
  - The build context is that directory without the 12 files NVIDIA's own build wrote into it after the `COPY`: `embedding_lib/build/` and `embedding_lib/custom_embedding.egg-info/`, created by the Dockerfile's last step (`pip install` of `embedding_lib`).
  - This directory contains no build output, so nothing has to be excluded.
- `config_VR200_2x4x3xtp1pp1cp1ep1_mxfp8.sh`, `opt_kernels/__init__.py`, `opt_kernels/qknorm_rope.py` and `opt_kernels/qknorm_rope_patch.py` have mode 0644 in the image and here; every other file is 0755. `setup.sh` keeps both modes.

## Launch

See [../firmus_gb300_ngpu8_gptoss_g1_1012/README.md](../firmus_gb300_ngpu8_gptoss_g1_1012/README.md): `setup.sh`, `init_datasets.sh` and `run_and_time.sh`. In short:

```bash
export CONT=$PWD/gpt_oss_20b_g1_1012.sqsh DATADIR=/path/to/data LOGDIR=/path/to/results   # DATADIR: README_NVIDIA.md 3.2
source config_GB300_2x4x3xtp1pp1cp1ep1_mxfp8.sh
sbatch -N ${DGXNNODES} --time=${WALLTIME} run.sub
```

**`config_filename`.** All ten logs record `config_filename = config_GB300_2x4x3xtp1pp1cp1ep1_mxfp8.sh`, the leaf in this directory that ran.
- NVIDIA's `pretrain.py` logs `config_${DGXSYSTEM}.sh` (L605).
- The leaf sets `DGXSYSTEM` from its own file name (L42), so the logged name is the leaf's.

## Data indices built before `run_start`

NVIDIA's `callback_logging.py` (`on_data_init_start`, L128-161; byte-identical to G1's) prepares the dataset indices before it logs `run_start`. In every log, `> building train, validation, and test datasets for GPT ...` and `> finished creating GPT datasets ...` come before `run_start`. In `result_0.txt` they are lines 934-938:

```
0: > building train, validation, and test datasets for GPT ...
0: > finished creating GPT datasets ...
0: :::MLLOG {... "key": "init_stop", ... "metadata": {"file": "/workspace/llm/callback_logging.py", "lineno": 161}}
0: :::MLLOG {... "key": "run_start", ... "metadata": {"file": "/workspace/llm/callback_logging.py", "lineno": 161}
0: > building train, validation, and test datasets ...
```

- Rank 0 builds Megatron's document, sample and shuffle indices (`.npy` files in a fresh per-job directory) from the `.idx` metadata: the document count and the document lengths.
- Every rank then reads the `.npy` files once, so they are in the page cache (`_warm_npy_page_cache`, L71-112).
- No training token file (`.bin`) is read. The real dataset setup follows `run_start` (line 938), and the first token read is in the timed run.

Training rules L256-258 allow metadata such as document counts and lengths to be used before the clock starts. L240 asks that preprocessing repeated on every run be timed. The shuffle index depends on the run's seed, but it only orders documents: no token is transformed, augmented or read. This is NVIDIA's code, unchanged; their entry builds its indices the same way.

## Optimizer (TR-9.1c)

The optimizer is NVIDIA's, unchanged: Megatron-Core's `DistributedOptimizer` with Transformer Engine's `FusedAdam` in AdamW mode.
- `pretrain.py` L357-366 sets Adam with β 0.9 / 0.95, weight decay 0.1, gradient clipping 1.0 and the distributed optimizer; ε comes from the recipe (1e-5, logged as `opt_adamw_epsilon`).
- Megatron-Core takes Transformer Engine's `FusedAdam` when it is installed (`megatron/core/optimizer/__init__.py` L14 at `f6c33bde`).

The rules' list of compliant implementations names `apex.optimizers.FusedAdam` for language models. The submitter has confirmed that Transformer Engine's `FusedAdam`, used through Megatron-Core's `DistributedOptimizer` as NVIDIA's unchanged code (Transformer Engine 2.19.0 in this image), is an allowed optimizer implementation under TR-9.1c. The supporting information was first given for G1, whose image has the same Transformer Engine commit `5e52befd` as this one. In the L4 package's optimizer A/B ([K06](../../../llama31_8b/implementations/nemo_l4_nvfp4_v3/KERNELS/K06_x9step3_fused_adamw.md), "A/B against TE's and apex's `FusedAdam`", run on L4 v2's image), Transformer Engine's and apex's `FusedAdam` (both `adam_w_mode=True`) were stepped from identical inputs with β 0.9 / 0.95, ε 1e-5 and weight decay 0.1, and gave 0 differing bytes in the fp32 master, both moments and the bf16 parameter after 1, 2, 16 and 100 steps. That test ran on Transformer Engine 2.20.0 (`5b28d7aa`); this image has 2.19.0 (`5e52befd`), and the test was not repeated on it. The two versions' `FusedAdam` sources differ in two places only: 2.20.0 splits a call with more than 1,024 tensors into several calls of the same element-wise update, and 2.20.0's CUDA-graph-capturable kernels accept a parameter dtype different from the gradient dtype (2.19.0 requires them to be equal, and for equal dtypes the kernels are the same code). The non-capturable kernels, the multi-tensor launcher and the C++ bindings are identical in both versions (record available on request).
