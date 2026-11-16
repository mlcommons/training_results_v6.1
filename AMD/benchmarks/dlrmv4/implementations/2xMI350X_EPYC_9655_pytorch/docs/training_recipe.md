# Training recipe (AMD MI350X, dlrmv4 closed division)

This file describes the configuration of the submitted runs: 1, 2 and 4 nodes of 8 AMD Instinct
MI350X GPUs, HSTU ranking model on Yambda-5b, MLPerf Training ruleset 6.1.0. The launch steps
are in [`../README.md`](../README.md). The RDMA setup of the multi-node systems is in
[`multi_node_config.md`](multi_node_config.md).

All model, numerics and kernel settings come from one gin file,
`generative_recommenders/dlrm_v4/train/gin/yambda_5b.gin`. The trainer logs it as the MLPerf
`config_filename`. The launch configuration (`config_*.sh`) sets only the topology, the dataset
path and the values in the tables below, and each of those equals the gin or trainer default.

## 1. Systems and batch size

| System | Nodes x GPUs | Per-GPU batch (`BATCH_SIZE`) | Global batch | Gradient accumulation |
|---|---|---|---|---|
| `MI350X_EPYC_9655_pytorch` | 1 x 8 | 1024 | 8192 | 1 |
| `2xMI350X_EPYC_9655_pytorch` | 2 x 8 | 1024 | 16384 | 1 |
| `4xMI350X_EPYC_9655_pytorch` | 4 x 8 | 1024 | 32768 | 1 |

Global batch = nodes x 8 x `BATCH_SIZE`. Evaluation also uses a per-GPU batch of 1024.
Data parallel only: tensor, pipeline, context and expert parallelism are all 1. TorchRec shards
the embedding tables across all ranks.

## 2. Optimizer and learning rate

| Setting | Value |
|---|---|
| Dense optimizer | Adam, beta1 0.95, beta2 0.999, eps 1e-8, weight decay 0 |
| Sparse (embedding) optimizer | RowWiseAdagrad |
| Gradient clipping | global norm 1.0 |
| Learning rate (dense = sparse) | 1e-6 x global batch / 8192 |
| Learning rate per system | 1e-6 (GBS 8192), 2e-6 (GBS 16384), 4e-6 (GBS 32768) |
| Warmup | linear from 0.0 to the base rate over the first 24000 steps, then constant |

When `DENSE_LR` and `SPARSE_LR` are unset, the trainer computes the rate from the global batch
with the formula above. The `config_*.sh` files set the same values explicitly. The sparse rate
always equals the dense rate.

## 3. Model and data

| Setting | Value |
|---|---|
| Model | HSTU ranker, 3 attention layers, `max_sequence_length` 4096 |
| History | `HISTORY_LENGTH` 4086: up to 1362 most recent events from each of three pools (listens, likes, skips), merged in time order (`history_strategy = "interleaved"`), plus 8 contextual positions and 1 candidate |
| Training data | Time-ordered stream over 299 one-day windows (0 to 298), every sample of each window (`START_TS=0`, `NUM_TRAIN_TS=299`, `TRAIN_SPLIT_PERCENTAGE=1.0`) |
| Data order | Fixed: no shuffling across or within windows (shuffle fraction 0.0, shuffle seed 0) |
| Evaluation data | The held-out window 299, scored in full: 2,058,204 samples |
| `train_samples` | 2,290,835,423 |
| Data loading | 6 workers per training data loader, prefetch factor 8, 8 evaluation workers |
| Time-window index | Offline window index from mlcommons/training PR 907 (`ts_indices_L4086_W86400.npy`, `ts_offsets_L4086_W86400.npy`), prebuilt with `scripts/ts_index_tool.py prebuild` or built during dataset initialization, before `run_start` |

The trainer reads only the reference dataset files: the flat-event cache
`processed_5b/hstu_cache_L4086/` (including the window index above),
`processed_5b/item_popularity.npy` and the two item mappings in `shared_metadata/`. The checksum
lists `md5sums_yambda_5b_processed.txt` and `sha256sums_yambda_5b_cache.txt` cover them.

The submitted runs copied exactly these files (229 GB) to node-local disk before each run
("reference-only staging", gin `DLRM_HSTU_FP8_STAGE_REF_ONLY = True`, done by
`scripts/stage_data_local.sh`). Training reads the same bytes from any location; staging only
moves them closer to the GPUs.

## 4. Evaluation and stopping

| Setting | Value |
|---|---|
| Metric | AUC of the `listen_plus` label on the held-out window |
| Target | 0.75 (`AUC_THRESHOLD`). The run stops at the first evaluation that reaches it |
| Evaluation interval | `EVAL_EVERY_DATA_PCT` = 0.001 of `train_samples`, rounded up to whole steps: 280 steps at GBS 8192, 140 at 16384, 70 at 32768 (2,293,760 samples in each case) |
| First evaluation | `SKIP_EVAL_EPOCH_PCT` = 0.025 (GBS 8192), 0.0303 (GBS 16384), 0.041 (GBS 32768) of the epoch |

Evaluations before the `SKIP_EVAL_EPOCH_PCT` point do not run. The values follow the MLPerf rule
for the first evaluation at each global batch size, with a margin of two evaluation intervals
(three at GBS 16384) before the earliest observed crossing of the target. The gin default (-1)
derives the same values from the global batch, so setting them explicitly changes nothing.
Skipping evaluations never changes the training trajectory. The final evaluation always runs.

## 5. Seeds

Every run draws a fresh random seed (gin default `SEED=-1`) and logs it as the MLPerf `seed`.
All ranks of a run use that seed. The data order does not depend on it (section 3).

## 6. Numerics

Training runs in bf16 mixed precision, with the low-precision paths below turned on in
`yambda_5b.gin`. Each one is a gin binding, and an environment variable of the same name turns it
off. The MLPerf log records:

| MLPerf key | Value |
|---|---|
| `lowest_numerical_precision_in_linear` | `mxfp6` |
| `lowest_numerical_precision_in_attn` | `fp8` |
| `lowest_numerical_precision_in_comm` | `fp16` |

| Area | Low-precision path | gin binding |
|---|---|---|
| HSTU linear layers, forward | FP8 e4m3 GEMMs with delayed per-tensor scales | `DLRM_HSTU_FP8_FWD` |
| HSTU K=512 forward GEMMs (U, V/Q/K, targets-only V/K) | MXFP6: E2M3 values with one E8M0 scale per 32 elements along K | `DLRM_HSTU_FP8_FWD_MXFP6` |
| HSTU linear layers, backward | FP8 output-projection gradients (e5m2 gradient, FP8 weight gradient), FP8 uvqk/vk input and weight gradients | `DLRM_HSTU_FP8_BWD`, `DLRM_HSTU_FP8_BWD_OUT_WGRAD`, `DLRM_HSTU_FP8_BWD_UVQK`, `DLRM_HSTU_FP8_BWD_UVQK_WGRAD = "fp8t"` |
| Attention forward | P@V on FP8: e4m3 P with a static scale, e4m3 V | `DLRM_HSTU_FP8_FWD_PV` |
| Attention backward | Full attention backward on the FP8 MFMA: e4m3 operands, e5m2 dO, dq, du and dk/dv, dS cast scale exponent -5 | `DLRM_HSTU_FP8_BWD_FULL`, `DLRM_HSTU_FP8_BWD_QK_MFMA = "fp8"`, `DLRM_HSTU_FP8_BWD_DO8`, `DLRM_HSTU_FP8_L11_DO8`, `DLRM_HSTU_FP8_BWD_DQ_E5M2`, `DLRM_HSTU_FP8_BWD_DU_E5M2`, `DLRM_HSTU_FP8_BWD_DKDV_E5M2`, `DLRM_HSTU_FP8_BWD_VF4`, `DLRM_HSTU_FP8_BWD_FULL_DSC_K = -5` |
| Preprocessor MLPs | FP8 L2 GEMMs. The indexed L1 forward uses FP8 operands in training and bf16 in evaluation | `DLRM_HSTU_FP8_PREPROC_L2`, `DLRM_HSTU_FP8_EC_INDEXED_L1_FP8` |
| Sparse embedding all-to-all | fp16 on the wire, forward and backward | `make_optimizer_and_shard.sparse_a2a_forward_precision`, `sparse_a2a_backward_precision` |

Several fused kernels also sum in a different fp32 order than the unfused code. The gin comments
list these as declared reduction-order changes; they are not precision changes. The evaluation
forward of the LN x U x dropout block uses the fused kernel the training forward uses
(`DLRM_HSTU_FP8_EVAL_LNMD_V3`).

## 7. Kernels and compilation

- HSTU attention forward and backward run FlyDSL kernels (`HSTU_ATTN_FWD = "flydsl"`,
  `HSTU_ATTN_BWD = "flydsl"`) built for gfx950. `FLYDSL_GPU_ARCH=gfx950` must be set.
  Other HSTU kernels use Triton (`HSTU_HAMMER_KERNEL=TRITON`).
- Before `run_start`, the trainer compiles the Triton kernel specializations listed in
  `generative_recommenders/ops/triton/triton_spec_table.jsonl` (gin `TRITON_SPEC_REPLAY`) and
  the HIP kernels the run needs, and loads the GPU code objects listed in
  `generative_recommenders/dlrm_v4/train/hip_preload_table.jsonl`. These steps change start-up
  time only, not the results.
- hipBLASLt solutions of the bf16 GEMMs are pinned by the gin (`GR_PIN_BF16_BLAS_SOLUTIONS`).

## 8. Environment

| Variable | Value | Purpose |
|---|---|---|
| `FLYDSL_GPU_ARCH` | `gfx950` | FlyDSL target |
| `HSTU_HAMMER_KERNEL` | `TRITON` | HSTU kernel family |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | allocator |
| `HIP_VISIBLE_DEVICES`, `CUDA_VISIBLE_DEVICES` | `0,1,2,3,4,5,6,7` | the 8 GPUs of a node |
| `NCCL_DEBUG` | `WARN` | log level |
| `DLRM_DATA_PATH` | dataset folder inside the container | dataset location |
| `CLEAR_CACHES` | `1` | page-cache flush before every run (MLPerf `cache_clear`) |

The multi-node settings are in [`multi_node_config.md`](multi_node_config.md).

## 9. Software

The container starts from `rocm/primus:v26.3`, pinned by digest
`sha256:1a02b74a94d82131f3f119a94ccbed45bb5b4cd77a0616cb4e6a29e64a048482` (ROCm 7.2.1,
Python 3.12, Triton 3.6.0, FlyDSL 0.1.1.dev409). The image adds:

| Package | Version |
|---|---|
| torch | 2.12.0+rocm7.2 |
| torchvision / torchaudio | 0.27.0+rocm7.2 / 2.11.0+rocm7.2 |
| fbgemm_gpu (`fbgemm_gpu_nightly_rocm`) | 2026.6.2, built from FBGEMM commit 10b775730212923f65f7b78f79b6a01d80cf3c29 for gfx950 |
| torchrec | 1.7.0a0+bf55480 (tag v2026.06.01.00) |
| polars-u64-idx | 1.33.1 (64-bit row index; Yambda-5b has more than 2^32 rows) |
| gin-config | 0.5.0 |
| numba / llvmlite | 0.68.0 / 0.50.0 |

All other Python packages, including torchmetrics 1.0.3 (which computes the evaluation AUC) and
mlperf-logging 4.1.52, are the base image's own. The trainer loads RCCL 2.27.7 and hipBLASLt
1.2.2, the libraries bundled with the torch wheel (not the base image's `/opt/rocm` copies). `requirements.txt` lists the versions;
`Dockerfile` is the build recipe.
