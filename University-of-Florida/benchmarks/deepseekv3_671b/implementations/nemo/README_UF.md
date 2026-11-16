# DeepSeek v3 671B: University of Florida files

This dir is NVIDIA's MLPerf Training v6.1 implementation, unchanged, from the image
`nvcr.io/nvdlfwea/mlperftv61/deepseekv3_671b-arm:20261006` (`GIT_COMMIT_ID=2647c2168340a97ec5b94ae56eb539194bf8a961`), rebuilt for x86 as `deepseekv3_671b-amd_20261006.sif` (see `BUILD_UF.md`).
NVIDIA's guide is `README.md`. Our files are the config below, the 2 patches in `patches/` and `uf_launcher/`.
NVIDIA's build context also has `hf_hub/` (a Hugging Face cache with `.locks` files and blobs). It is not in this dir: `repo_checker` rejects it, and the image has the same files.

## What to run

The config sources NVIDIA's `config_GB300_64x4x240xtp1pp4ep32cp1_mxfp8_full_cg.sh` and changes only the values below. LR and schedule are NVIDIA's.

| System | Config | GBS | Parallelism | LR, warmup |
|---|---|---|---|---|
| hpg.dgxb200x8.n32 | `config_DGXB200_32x8x1920xtp2pp16ep8cp1_mxfp8_full_cg_cf8_noopg.sh` | 15360 | TP 2, PP 16 x VPP 2, EP 8, DP 8 | 2.3238e-05, 4 steps |

The runs need the Megatron-Bridge checkpoint of DeepSeek v3 (`LOAD_CHECKPOINTS_DIR=<dir with iter_0000000/>`).

## How to run

`uf_launcher/` holds the launcher files that started the runs. Copy them to `<kit>/launcher/`, where `<kit>/University-of-Florida/` is this tree, then run from `<kit>`:

```bash
NEXP=3 LOAD_CHECKPOINTS_DIR=<dir> launcher/run_hpg.sh deepseekv3_671b config_DGXB200_32x8x1920xtp2pp16ep8cp1_mxfp8_full_cg_cf8_noopg.sh    # HPG B200 (Slurm); RESERVATION=<name> is optional
```

`kit_defaults.sh` sets the image, the data dir and this dir. The container's own code runs, with the 2 patches bound over the image files. `run_hpg.sub` does the steps of NVIDIA's `run.sub` with apptainer (cache drop per run, one rank per GPU through `srun`, compliance checker).

## Why we changed NVIDIA's settings

- **Layout.** NVIDIA runs EP 32 inside one GB300 NVL72 NVLink domain. On HGX B200 the NVLink domain is 1 node, and the image's HybridEP has no inter-node path. So each EP group of 8 and each TP pair stays inside one node, with PP 16 x VPP 2 (VPP > 1: NVIDIA's EP all-to-all overlap needs it). TP 2 with sequence parallel, as in NVIDIA's v6.0 GB200 config, fits the activations into 178 GiB. `USE_MNNVL=0`, `SEGMENT=1`.
- **`OVERLAP_PARAM_GATHER=False`.** With the overlapped MXFP8 param all-gather, the first all-gather after the checkpoint load puts stale buffer data into the weights: eval lm loss 15.62 on the loaded checkpoint. Without overlap the loaded weights are used: 5.155 (NVIDIA GB200 v6.0 start: 5.16).
- **HybridEP capacity factor 8** (= the EP group size): one rank can never receive more than the budget, so NVIDIA's dropless rerun (which needs more memory than is free) cannot start. Part of the paged stash is in pinned host memory (CUDA 1.0, CPU 2).
- **Shared expert at TP 2.** NVIDIA's fused shared expert fails with TP 2, so `USE_GROUPED_GEMM_FOR_SHARED_EXPERT=False`. `MOE_SHARED_EXPERT_GLU_INTERLEAVE_SIZE` is unset, because the standard MLP reads fc1 as contiguous.
- **Patch `patches/megatron/core/transformer/moe/paged_stash.py`** (the image's Megatron-LM file). The paged stash schedule key uses base 10000 instead of 1000: NVIDIA's key allows at most 999 micro-batches per step, and MINIBS 1920 needs more. The key is only compared, negated and used as a dict key, so the training math does not change. Also, a validation rerun does not zero the grad buffer: it has no optimizer, and with `REUSE_GRAD_BUF_FOR_MXFP8_PARAM_AG=True` the grad buffer is also the MXFP8 param buffer (zeroing it gave zero weights).
- **Patch `patches/llm/callback_warmup.py`** (NVIDIA's file). The untimed warmup steps (mock tokens, LR 0) zero the grad buffers before each optimizer step, so the update is exactly zero. Without it, an MoE overflow in a warmup step gave NaN grads, and 0 x NaN made the weights NaN.
- Mock pipeline compile on (`WARMUP_MOCK_PIPELINE_COMPILE_MODE=schedule`), recompute of the dense MLP layers, 60 min distributed timeout, and `LOAD_CHECKPOINT_STEP=0` (the tracker file of the MLCommons checkpoint points to a step that it does not have).
- The config was rewritten after the run as one file (before, a chain of 4 of our files). The shell variables after `source` are the same (checked with `declare -p`).

## Result files

3 runs (the 6.1 rule for this benchmark) from one job. `result_<k>.txt` = run k+1.
