# Krai llama2_70b_lora implementation v6 (AMD Instinct MI300X)

AMD's MLPerf Training 5.1 Llama 2 70B LoRA implementation (`rocm/amd-mlperf:llama2_70b_training_5.1`,
`/workspace/code`; its README is `README_upstream_AMD_v5.1.md`) with Krai's kernel fixes, no activation
recomputation with packed LoRA dropout, and eval trimming. `Dockerfile` builds it on AMD's container;
`../PowerEdge_XE9680_8xMI300X/` holds the setup and run scripts for the submitted
system. `README_kernel_changes.md` lists the changes by the version that introduced them (v3.1, v4.2, v6).

## What changed against AMD's 5.1 implementation

| Change | Version | Where |
|---|---|---|
| No activation recomputation (stock: 21 layers) | v4.2 | config (`ACG`, `ACM`, `ACL` null) |
| LoRA dropout with packed masks; the attention-output adapter keeps no dropped copy (rebuilt in the backward); bit-identical | v4.2 | `patches/lora_dropout_packed.py` |
| SwiGLU input stored in FP8 (e4m3) for the backward | v3.1 | config (`FP8_ACT_STORE`), `src/conf/megatron_gpt_peft_tuning_config.yaml`, `patches/hs_act_store.py` |
| Fixes 1, 2, 3a, 4: fused SwiGLU store, LoRA epilogue folds, e5m2 SwiGLU backward, fused qkv split and RoPE | v3.1 | `hs_swiglu_fp8_triton.py`, `hs_lora_epilogue.py`, `hs_dswiglu_e5m2.py`, `hs_rope_qkv.py` |
| Fixes 3b, 6b, 7, 8a–c, 11, 13, 14: SwiGLU forward writes fc2's e4m3 input, cached weight transposes, fc2 forward hipBLASLt solution, fused norm and gradient casts, fewer host syncs, RMSNorm grid, faster LoRA epilogues | v4.2 | `hs_swiglu_fwd_e4m3.py`, `hs_transpose_cache.py`, `te_algos_fix7.csv`, `hs_qkvnorm_e4m3.py`, `hs_resgrad_fold.py`, `hs_host_sync.py`, `hs_rmsnorm_grid.py`, `hs_lora_epilogue.py` |
| Packed-mask LoRA dgrad at fix 14's bandwidth | v6 | `patches/hs_lora_dgrad_packed.py` |
| **Eval trimming**: each validation row cut after its last loss token (512-token buckets), rows served longest first; TE's Triton kernels never autotune a new token count inside the eval | v6 | `patches/eval_trim.py` |
| Grad-output FP8 cast without the transposed copy (frozen base weights) | v3.1 | `te_patches/` (TE 2.2, one line each) |
| Round-toward-zero BF16 conversion in the v3 attention backward | v3.1 | config (`NVTE_CK_HOW_V3_BF16_CVT=2`) |
| Allocator cache released around evals | v3.1 | `src/callbacks/custom_callbacks.py` |
| Submitter Krai | — | config (`MLPERF_SUBMISSION_ORG`) |

The settings are the block appended to `config_MI300X_1x8x1.sh`, identical to what every run received.
`src/train.py` imports the modules in `TRAINLOOM_PY_PATCHES`, in order, before NeMo, Megatron and TE load;
`Dockerfile` puts `patches/` at `/workspace/trainloom_patches`, where the config and the logs refer to it.
No hyperparameter differs from the reference.

**Eval trimming.** Each validation row is one 8,192-token document whose loss covers only its summary (ending on
average at token 5,700). With causal attention, positions after a row's last loss token cannot change any logit the
loss reads, and with `sample_weight: constant` the validation loss is the mean of per-row means, so cutting those
positions and reordering rows leaves it mathematically unchanged: dead-code elimination, allowed in the closed
division. Evals take 22.0 s instead of 29.5 s (the first one about 25 s), 40–49 s per run; training is untouched.
The trimmed rows equal the stock rows on every position the loss reads (`check_eval_trim.py`, through NeMo's own
collate). TE's Triton cast/transpose and RMSNorm kernels autotune on the token count; benchmarking every config inside
the eval forward at a new length produced NaN in the last layers on random GPUs (480 new lengths: 30 non-finite events
with the autotuning, none without), so a new token count reuses the config tuned for the training shape
(`check_eval_trim_autotune.py`: no benchmark, byte-identical cast). In paired partial runs the trimmed evals' losses
matched the untrimmed ones within run-to-run noise.

**Inherited from AMD's 5.1 implementation:** the eval loss mask is off by one token (the collate scores the
predictions one position after the summary tokens the data marks). Eval trimming reproduces the stock eval exactly,
including this.

Checks on one GPU, inside the image (`patches/`): `check_eval_trim.py` (CPU), `check_eval_trim_autotune.py`,
`check_lora_epilogue_packed.py`, `check_lora_dropout_packed.py`, `check_swiglu_exact.py` and the `test_hs_*.py` tests.

## Running it

```bash
cd ../PowerEdge_XE9680_8xMI300X
./setup.sh                                   # builds krai4ai/mlperf:training_v6.1-llama2_70b_lora-mi300x, applies the host tuning (root)
HF_TOKEN=... ./init_datasets.sh              # once: dataset and model into $DATADIR
NEXP=10 DATADIR=/data/mlperf_llama2 ./run_and_time.sh
```

`run_with_docker.sh`, `run_and_time.sh`, `runtime_tunables.sh` and `scripts/` are AMD's, unchanged; AMD's own
Dockerfile and its configs for other GPUs are not included, as this submission does not use them.

## Results

`results/PowerEdge_XE9680_8xMI300X/llama2_70b_lora/result_0..9.txt`: 22.83 min (olympic mean 22.76 min × RCP
normalisation 1.0034). Twenty runs of this configuration were made on 15 October 2026, in two sets of ten declared
before their runs, the second declaration stating that both are reported and scored together; per the training rules
(Benchmark Results › Scoring) the submitted runs are the 10 consecutive runs closest to the median of the twenty,
runs 8–17. All twenty logs are kept by Krai and available on request.

## Provenance

- AMD MLPerf Training 5.1 implementation: the `rocm/amd-mlperf:llama2_70b_training_5.1` image's `/workspace/code`.
- `te_patches/`: TransformerEngine 2.2 (ROCm) files from that image, Apache-2.0, one line changed each.
- `patches/`, `Dockerfile` and the four changed recipe files: Krai Ltd. They are the code the results ran
  from, identical apart from comments.
