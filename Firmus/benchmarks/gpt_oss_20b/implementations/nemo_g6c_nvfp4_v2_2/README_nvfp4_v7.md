# GPT-OSS 20B, NVFP4 "V7" recipe (2 x 4 GB300)

This directory is the NeMo / Megatron-Bridge reference implementation (`README.md`) plus the
NVFP4 changes below.

| Item | Value |
|---|---|
| Topology | 2 nodes x 4 GB300, DP8, TP1 PP1 CP1 EP1 |
| Batch | MBS 3, GBS 24, sequence 8192 |
| Precision | NVFP4 linears (RHT, 2D weight quantization, stochastic rounding with 7 Philox rounds); last 2 layers MXFP8 instead of BF16; LM head forward MXFP8; FP8 (current scaling) dot-product attention; BF16 communication except the NVFP4 expert weight all-gather |
| Optimizer | AdamW, LR 5e-4, 256 warmup steps, cosine decay over 1200000 steps, clip 1.0 |
| Step time | 0.359 s/step |
| Evaluation | Full validation set every 512 steps (12288 sequences); each layer replays its captured forward CUDA graph (`NVFP4_EVAL_GRAPH=1`) |

## Layout

| Path | Contents |
|---|---|
| `Dockerfile` | Builds the complete image from `nvcr.io/nvidia/pytorch:26.08-py3` |
| `docker/patches/` | All Firmus changes to Transformer Engine, Megatron-Core and the cuDNN frontend |
| `docker/constraints/` | Pinned versions of every Python distribution in the image |
| `docker/checksums/` | cuDNN frontend files before and after its patch |
| `pretrain.py` | NVFP4 training entry point |
| `nvfp4_clip_adam.py`, `clip_adam_ext.cu` | Gradient clipping folded into the fused Adam step; the extension is compiled in the `Dockerfile` |
| `dglu_kernel_ep.py` | Grouped dGLU backward kernel loaded through `NVFP4_DGLU_KERNEL_FILE` |
| `config_GB300_2x4x3xtp1pp1cp1ep1_nvfp4_v7.sh` | Run configuration (sources `config_common.sh`) |

No source file is mounted into the container at run time; `run.sub` mounts only the dataset,
tokenizer and log directories.

## Container image

The `Dockerfile` starts from `nvcr.io/nvidia/pytorch:26.08-py3` (pinned by digest) and builds,
in order:

| Step | Source | Firmus patch |
|---|---|---|
| Megatron-Bridge, Megatron-Core | `72000e9d4306afa0354fd34e96743d46572b8d41`, `f6c33bde4cdc10805ee7157db746d47b758f5f02` | `megatron_lm_firmus_nvfp4.patch` |
| HybridEP (DeepEP) | `10d4dd7377d5bce900fbb4b80cce863764892b95` with Megatron-Bridge `docker/patches/deepep.patch` | none |
| Transformer Engine | `27486e03cfc1fa41f6932dcecdc47c71c47eac3e`, built for sm_103a with `NVTE_BUILD_NUM_PHILOX_ROUNDS=7` | `te_fused_permutation_2880_2944.patch`, `transformer_engine_firmus_nvfp4.patch`, `transformer_engine_firmus_v7.patch` |
| CuTe DSL, FlashInfer | `nvidia-cutlass-dsl[cu13]==4.5.0`, `flashinfer-python==0.6.18.post1` | none |
| cuDNN frontend | `35fd7b0d0e1d4952b904c79341c5e84e3af0a328` (shipped in the base image) | `cudnn_frontend_firmus_nvfp4.patch` |
| Benchmark code | this directory, `embedding_lib`; CUDA extensions in `/opt/nvfp4_ext`: clip+Adam, NVFP4 shard cast, dy amax+dbias | none |

`transformer_engine_firmus_v7.patch` adds to `transformer_engine/pytorch/ops/fused`:

| File | Switch | Purpose |
|---|---|---|
| `nvfp4_fp4_ag.py` | `NVFP4_FP4_AG=1` | Expert weights cast to NVFP4 per DP shard and all-gathered as NVFP4 |
| `nvfp4_shard_cast.py`, `nvfp4_shard_cast_cuda.py`, `shard_cast_ext.cu` | `NVFP4_SHARD_CAST_IMPL=cuda` | The shard cast in one CUDA launch per bucket |
| `nvfp4_resgrad.py` | `NVFP4_RESGRAD*=1` | Residual and router gradients summed inside the RMSNorm backward; MLP residual added in the FC2-bias unpermute |
| `nvfp4_amax_dbias.py`, `amax_dbias_ext.cu`, `nvfp4_dy_stats.py` | `NVFP4_DY_STATS=1`, `NVFP4_DY_STATS_IMPL=cuda` | Grouped dy amaxes, dbias and dscales in one pass |
| `nvfp4_qkv_packed.py` | `NVFP4_QKV_PACKED=1`, `NVFP4_QKV_PACKED_AMAX=1` | RoPE writes Q, K, V into one buffer and its current-scaling amax; FP8 attention quantizes it without a concat |
| `grouped_mlp.py` | | Hooks for the modules above |

and in `transformer_engine/pytorch/csrc/quantizer.cpp` lets the fused NVFP4 RHT cast take
2880-wide dense tensors when `NVTE_NVFP4_RHT_FUSION_COLS64=1`.

```bash
docker build -t <image> .
enroot import -o <image>.sqsh dockerd://<image>
```

## Running

```bash
export CONT=<image>
export DATADIR=<dataset root>
export LOGDIR=<log directory>
source config_GB300_2x4x3xtp1pp1cp1ep1_nvfp4_v7.sh
export NEXP=10 SEED_BASE=<seed>
export WALLTIME=$((5 + NEXP * (WALLTIME_RUNANDTIME + 5)))
sbatch -N "${DGXNNODES}" --ntasks-per-node=1 --gpus-per-node=4 --exclusive --time="${WALLTIME}" run.sub
```
