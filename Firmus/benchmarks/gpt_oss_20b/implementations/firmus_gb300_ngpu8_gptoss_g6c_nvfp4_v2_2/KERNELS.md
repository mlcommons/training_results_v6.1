# Kernels and numerical changes in the NVFP4 "V7" implementation

Classification: Internal (becomes public with the MLPerf Training v6.1 results)

This document lists every change this implementation makes to the computation of the
Megatron-Bridge / Megatron-Core / Transformer Engine (TE) stack it is built on, and for each one
states what it replaces, why the result is the same, and what it does numerically. Every item is
switched by an environment variable; the value used is the one in
`config_GB300_2x4x3xtp1pp1cp1ep1_nvfp4_v7.sh`. All code is built into the image by the
`Dockerfile`; nothing is mounted at run time. File names below refer to
`../nemo_g6c_nvfp4_v2_2/`, the Docker build context.

**G6c.** This implementation trains NVIDIA's 20261012 reference model (no attention or expert
biases, SiLU-gated experts, vanilla softmax without sinks, QK RMSNorm). The kernels that assumed
the earlier model were reworked for it instead of being switched off; they are listed in
section 10 and marked G6c in the summary. The patches (`docker/patches/`), the `Dockerfile`,
`nvfp4_clip_adam.py`, `clip_adam_ext.cu` and `dglu_kernel_ep.py` are byte-identical to the G6b
package's (`nemo_g6b_nvfp4_v2_1`; see "Earlier packages" below); `pretrain.py`, `conf/gpt_oss.yaml` and the config differ,
and `firmus_kernels/` and `opt_kernels/` are new.

**Earlier packages.** G3 (`nemo_g3_nvfp4_v1`), G6 (`nemo_g6_nvfp4_v2`) and G6b (`nemo_g6b_nvfp4_v2_1`) are Firmus's
earlier GPT-OSS NVFP4 packages. They were withdrawn from this submission and are no longer in this repository. This
document and `KERNELS/` still cite them where evidence was collected on their images or by their tests ("G6 image",
"G6b package tests"); those records are available on request.

One loader fallback in the code is never taken. `_v7_module` in
`docker/patches/transformer_engine_firmus_v7.patch` (lines 338–356) imports each V7 helper module
from the patched TE package and, only if it is missing there, from `NVFP4_V7_OV_DIR` (default
`/opt/v7ov`). The `Dockerfile` creates neither. The submitted switches load seven helpers through
`_v7_module`: `nvfp4_amax_dbias`, `nvfp4_dy_stats`, `nvfp4_fp4_ag`, `nvfp4_shard_cast`,
`nvfp4_shard_cast_cuda`, `nvfp4_qkv_packed` (`firmus_kernels/qknorm_rope_packed.py` line 162; the
`pretrain.py` call at line 87 is under the 5.2 switch, which is off) and `nvfp4_resgrad`
(`pretrain.py` line 113, `firmus_kernels/resgrad_fwd_nobias.py` line 71). Image verify jobs 22541
and 22589 found all seven installed and equal to the production overlay in the G6b image, which is
built from the same patches and `Dockerfile`; that check was not repeated on this package's image.
No run sets `NVFP4_V7_OV_DIR`, and the runs mount only the results, index cache, memory-dump,
tokenizer and data directories. The eighth helper, `nvfp4_act_quant`, is used only with
`NVFP4_TRITON_ACT_QUANT=1` (not enabled; see below) and is not installed. The fallback is accepted
for this set.

"Upstream" below means the same code path in the pinned upstream commits without our patches
(Megatron-LM `f6c33bde`, TE `27486e03`, cuDNN frontend `35fd7b0d`). Equivalence classes used in
the summary:

- **Bitwise**: the result is bit-identical to the upstream path.
- **Same function**: mathematically identical; floating-point results may differ by rounding
  (different summation order or fusion). This is the same function at infinite precision computed
  with a different floating-point order (E1). The tests (Section 9) count, per test, the cases that
  differ from the reference; no pass threshold is applied. Sizes quoted from the G6b package's
  tests are marked as such (that package ran the earlier model on its own test image).
- **Precision**: a choice of numeric format for a linear layer or the attention; Section 2 lists
  the formats and which of them the rule ("Numerical formats") names.
- **Not equivalent**: the change alters the training math. Stated explicitly.
- **Schedule**: only order, stream or memory placement changes; values are unchanged.
- **Fix**: removes non-finite values that the upstream path can read from uninitialised memory;
  otherwise values are unchanged.
- **not used**: the switch is set (or the code is present), but the path is never taken in the
  submitted configuration.

Each item below ends with four lines: **Gate** (inputs and switches the code accepts), **Falls
back** (what runs otherwise), **Assumes** (conditions the code does not check) and **Check** (how
the scored runs show the item was on). "config" is `config_GB300_2x4x3xtp1pp1cp1ep1_nvfp4_v7.sh`,
"TE patch" is `docker/patches/transformer_engine_firmus_nvfp4.patch`, "V7 patch" is
`docker/patches/transformer_engine_firmus_v7.patch`, "Megatron patch" is
`docker/patches/megatron_lm_firmus_nvfp4.patch` and "cuDNN FE patch" is
`docker/patches/cudnn_frontend_firmus_nvfp4.patch`; patch line numbers are lines of the patch file.
`result_0.txt` lines are in
`../../../../results/firmus_gb300_ngpu8_gptoss_g6c_nvfp4_v2_2/gpt_oss_20b/`; every line quoted
appears in all 10 result files, at most a few lines apart (the `run_stop` line moves with the
number of evaluations: lines 2732–2778).

## Summary

| # | Change | Switch | Source | Replaces | Class | Write-up |
|---|---|---|---|---|---|---|
| 1.1 | Gradient clip applied inside the Adam kernel | `NVFP4_CLIP_IN_ADAM`, `NVFP4_CLIP_ADAM_EXT` | `clip_adam_ext.cu`, `nvfp4_clip_adam.py`, `pretrain.py` | Megatron `clip_grad_by_total_norm_fp32` scale pass + TE `multi_tensor_adam(_param_remainder)` | Bitwise | [K1.1](KERNELS/K1.1_clip_in_adam.md) |
| 1.2 | Gradient norm in two launches | `NVFP4_FAST_L2NORM` | same | TE `multi_tensor_l2norm` | Bitwise | [K1.2](KERNELS/K1.2_fast_l2norm.md) |
| 1.3 | Expert weight amax folded into Adam | `NVFP4_ADAM_AMAX` | same | separate amax pass (1.3 / 6.2) | Bitwise | [K1.3](KERNELS/K1.3_adam_amax.md) |
| 1.4 | Adam in phases along the all-gather order | `NVFP4_PHASED_ADAM_AG=4` | `pretrain.py` | one `FusedAdam.step` | Bitwise (needs 1.5, not checked in code; 1.4) | [K1.4](KERNELS/K1.4_phased_adam.md) |
| 1.5 | Precision-aware optimizer on BF16 model params | `NVFP4_PAO_BF16_MODEL_PARAMS`, `NVFP4_PRECISION_AWARE_OPT` | `pretrain.py` | Megatron-managed FP32 master | Same function (FP32 master bitwise; BF16 rounding of ties) | [K1.5](KERNELS/K1.5_precision_aware_optimizer.md) |
| 2.1 | NVFP4 linear layers 0–21 | `FP4_RECIPE=nvfp4` | upstream TE / Megatron | BF16 linears | Precision | [K2.1](KERNELS/K2.1_nvfp4_linears.md) |
| 2.2 | Tail layers 22–23 in MXFP8 | `NVFP4_TAIL_MXFP8`, `NUM_LAYERS_AT_END_IN_BF16=2` | TE patch `nvfp4_tail_mxfp8.py` | BF16 tail layers of the NVFP4 recipe | Precision | [K2.2](KERNELS/K2.2_tail_mxfp8.md) |
| 2.3 | LM head forward in MXFP8 | `NVFP4_LMHEAD_MXFP8`, `_MODE=fprop` | TE patch `nvfp4_lmhead_mxfp8.py` | BF16 LM head forward | Precision | [K2.3](KERNELS/K2.3_lmhead_mxfp8_fprop.md) |
| 3.1 | Expert width padded 2880 → 2944; padded entries stay zero, so the padding is equivalent to not existing (run evidence: `verification/padding_zero/`) | `NVFP4_PADDED_EXPERT_HIDDEN=2944`, `NVFP4_ZERO_FC1_PAD_ROWS`, `NVFP4_ZERO_HIDDEN_PAD=1` | Megatron patch `experts.py`, `moe_layer.py` | 2880-wide expert GEMMs | Same function | [K3.1](KERNELS/K3.1_expert_padding.md) |
| 3.2 | FC1 bias folded into the FC1 weight: **off** (`=0`); the model has no FC1 bias | `NVFP4_FOLD_FC1_BIAS=0`, `NVFP4_FREEZE_FOLDED_BIAS=0` | Megatron patch `experts.py`, TE patch `grouped_mlp.py` | — | not used | [disabled](KERNELS/disabled.md) |
| 3.3 | Pad/crop fused into permute/unpermute | `NVFP4_FUSED_PERMUTE_PAD`, `NVFP4_PERMUTE_*`, `NVFP4_UNPERMUTE_*` | `te_fused_permutation_2880_2944.patch`, TE patch `triton/permutation.py` | permute + `F.pad` / crop copies | Bitwise | [K3.3](KERNELS/K3.3_fused_permute_pad.md) |
| 3.4 | FC2 expert bias added in unpermute: **off** (the reference model has no expert bias) | `NVFP4_DEFER_FC2_BIAS=0`, `NVFP4_DEFER_FC2_BIAS_BF16=0` | TE patch `grouped_mlp.py`, `triton/permutation.py`; `pretrain.py` | — | not used | [disabled](KERNELS/disabled.md) |
| 4.1 | Tail of fixed MoE buffers zeroed | (always with 4.3) | cuDNN FE patch `grouped_gemm_glu_hadamard/api.py` | uninitialised rows | Fix (removes NaN) | [K4.1](KERNELS/K4.1_moe_buffer_tail_zero.md) |
| 4.2 | Persistent zeroed grouped-GEMM outputs | `NVFP4_PERSISTENT_GEMM_OUT` | TE patch `grouped_mlp.py` | `torch.empty` outputs read as C with beta = 0 | Fix (removes NaN; up to the sign of exact zeros); test differences not yet explained, 4.2 | [K4.2](KERNELS/K4.2_persistent_gemm_out.md) |
| 4.3 | Fixed-size EP1 MoE buffers, CUDA-graph capturable MoE | `NVFP4_FIXED_EP1_MOE_BUFFERS`, `NVFP4_DEVICE_ONLY_EXPERT_METADATA`, `NVFP4_ALLOW_DROPLESS_MOE_GRAPH` | Megatron patch `token_dispatcher.py`, `moe_utils.py`, `transformer_config.py` | host-synchronised dispatcher | Same function | [K4.3](KERNELS/K4.3_fixed_ep1_moe_buffers.md) |
| 5.1 | Sliding-window attention (G6c: vanilla softmax, 10.2) | `NVFP4_SWA_TRITON`, `NVFP4_SWA_PDKDV` | `firmus_kernels/swa_attention.py`; `pretrain.py` | cuDNN fused attention (window layers) | Same function | [K5.1](KERNELS/K5.1_swa_triton.md), [K10.2](KERNELS/K10.2_swa_vanilla.md) |
| 5.2 | QKV + RoPE without QK norm: **off**; replaced by 10.1 | `NVFP4_TRITON_QKV_ROPE=0` | TE patch `nvfp4_triton_qkv_rope.py` | — | not used | [disabled](KERNELS/disabled.md) |
| 5.3 | RMSNorm backward | `NVFP4_TRITON_RMSNORM_BWD`, `NVFP4_RMSNORM_BWD_STAGES` | TE patch `nvfp4_triton_rmsnorm_bwd.py` | `tex.rmsnorm_bwd` | Same function | [K5.3](KERNELS/K5.3_triton_rmsnorm_bwd.md) |
| 5.4 | Attention bias + residual in one kernel: **off** (no attention bias) | `NVFP4_FUSED_BIAS_RESIDUAL=0` | TE patch `nvfp4_triton_bias_residual.py` | — | not used | [disabled](KERNELS/disabled.md) |
| 5.5 | Interleaved clamped SwiGLU (tail layers): the model builds no clamped GLU | `NVFP4_TRITON_INTERLEAVED_GLU` (default 1, not in the config) | TE patch `ops/basic/swiglu.py` | de-interleave copy + TE clamped SwiGLU | not used (the Triton GLU accepts only `ScaledClampedQGeGLU`; 5.5) | [disabled](KERNELS/disabled.md) |
| 5.6 | Cross-entropy gradient written in place in BF16 | `NVFP4_CE_BF16_INPLACE` | TE patch `triton/cross_entropy.py` | FP32 gradient buffer + cast | Same function (bitwise when every per-token `grad_output` is 0 or a power of two; 5.6) | [K5.6](KERNELS/K5.6_ce_bf16_inplace.md) |
| 5.7 | dGLU kernel tuning | `NVFP4_DGLU_KERNEL_FILE`, `DGLU_NO_AMAX`, `DGLU_C_STAGE=4` | `dglu_kernel_ep.py`, cuDNN FE patch | cuDNN FE dGLU kernel class | Bitwise (the two tuning options; the kernel file as a whole is not compared, 5.7) | [K5.7](KERNELS/K5.7_dglu_kernel_tuning.md) |
| 5.8 | FC1 dbias in a separate reduction: **off** (no FC1 bias; the dGLU epilogue computes no dbias either) | `NVFP4_DGLU_TRITON_DBIAS=0` | TE patch `grouped_mlp.py`, `triton/grouped_dbias_dscales.py` | — | not used | [disabled](KERNELS/disabled.md) |
| 5.9 | Expert wgrad written, not accumulated | `NVFP4_OVERWRITE_EXPERT_WGRAD` | TE patch `grouped_mlp.py`, Megatron patch DDP | zero fill + accumulate | Bitwise (one micro-batch per step, not guarded; 5.9) | [K5.9](KERNELS/K5.9_overwrite_expert_wgrad.md) |
| 6.1 | Cached quantized expert weights (does not run: 6.2 takes precedence) | `NVFP4_CACHED_WEIGHT_QUANT` | TE patch `grouped_mlp.py` | new quantized tensor per step | not used | [K6.1](KERNELS/K6.1_cached_weight_quant.md) |
| 6.2 | Expert weight amax from optimizer shards | `NVFP4_SHARD_WEIGHT_AMAX` | TE patch `nvfp4_shard_weight_amax.py`, `pretrain.py` | per-weight amax pass | Bitwise (the in-run check does not apply under 8.2; 6.2, 9.2) | [K6.2](KERNELS/K6.2_shard_weight_amax.md) |
| 6.3 | Expert weight quantization on a side stream | `NVFP4_OVERLAP_WEIGHT_QUANT`, `NVFP4_PREQUANT_WAIT_AG` | TE patch `grouped_mlp.py`, `pretrain.py` | quantization in the MoE forward | Schedule | [K6.3](KERNELS/K6.3_overlap_weight_quant.md) |
| 7.1 | Param all-gather prefetch | `NVFP4_PREFETCH_PARAM_AG` | `pretrain.py` | one bucket group at a time | Schedule | [K7.1](KERNELS/K7.1_prefetch_param_ag.md) |
| 7.2 | Gradient buffer zeroing on a side stream, batched grad accumulation | `NVFP4_ASYNC_GRAD_ZERO`, `NVFP4_BATCH_GRAD_ACC` | Megatron patch `distributed_data_parallel.py` | in-stream zero, per-param `add_` | Bitwise (one micro-batch per step, not guarded; 7.2) | [K7.2](KERNELS/K7.2_async_grad_zero_batch_acc.md) |
| 7.3 | CUDA-graph constant mask and chained layer IO | `NVFP4_GRAPH_CONST_MASK`, `NVFP4_GRAPH_CHAIN_IO` | TE patch `graph.py`, `pretrain.py` | per-replay input copies | Bitwise | [K7.3](KERNELS/K7.3_graph_const_mask_chain_io.md) |
| 7.4 | Evaluation replays the training forward graphs | `NVFP4_EVAL_GRAPH` | `pretrain.py` | eager evaluation forward | Bitwise | [K7.4](KERNELS/K7.4_eval_graph.md) |
| 8.1 | FP8 dot-product attention (full-attention layers) | `FP8_DPA=True`, `NVTE_DPA_FP8_RECIPE=Float8CurrentScaling`, `NVTE_DPA_FP8_FORMAT=HYBRID`, `NVTE_DPA_FP8CS_O_in_F16=1` | upstream TE / cuDNN | BF16 attention | Precision | [K8.1](KERNELS/K8.1_fp8_dpa.md) |
| 8.2 | Expert weights cast per optimizer shard and all-gathered as NVFP4 | `NVFP4_FP4_AG` | TE patch v7 `nvfp4_fp4_ag.py`, `grouped_mlp.py`; `pretrain.py` | BF16 all-gather + full NVFP4 cast on every rank | Bitwise | [K8.2](KERNELS/K8.2_fp4_expert_allgather.md) |
| 8.3 | Shard cast in one CUDA launch per bucket | `NVFP4_SHARD_CAST_IMPL=cuda`, `NVFP4_SHARD_CAST_EXT` | TE patch v7 `shard_cast_ext.cu`, `nvfp4_shard_cast_cuda.py` | per-piece TE 2D cast (8.2) | Bitwise | [K8.3](KERNELS/K8.3_shard_cast_cuda.md) |
| 8.4 | Residual / router gradients summed in the RMSNorm backward; MLP residual added in unpermute (G6c: without FC2 bias, 10.3) | `NVFP4_RESGRAD`, `NVFP4_RESGRAD_ROUTER`, `NVFP4_RESGRAD_FWD=0`, `FIRMUS_RESGRAD_FWD_NOBIAS` | TE patch v7 `nvfp4_resgrad.py`; `firmus_kernels/resgrad_fwd_nobias.py`; `pretrain.py` | separate BF16 adds | Bitwise (against 5.3 plus separate adds; 8.4) | [K8.4](KERNELS/K8.4_resgrad.md), [K10.3](KERNELS/K10.3_resgrad_fwd_nobias.md) |
| 8.5 | Grouped dy amax (+ dbias, FC2 dscales when there is a bias) in one pass (G6c: amax only, no bias) | `NVFP4_DY_STATS`, `NVFP4_DY_STATS_IMPL=cuda`, `NVFP4_DY_STATS_FC2`, `NVFP4_DY_STATS_CUDA_FC2`, `NVFP4_AMAX_DBIAS_EXT`, `NVFP4_AMAX_DBIAS_COLS=448` | TE patch v7 `amax_dbias_ext.cu`, `nvfp4_amax_dbias.py`, `nvfp4_dy_stats.py` | TE group RHT amax | Bitwise (amax; no dbias or dscales in this model, 8.5) | [K8.5](KERNELS/K8.5_dy_stats.md) |
| 8.6 | Q, K, V written packed, with the current-scaling amax (G6c: by the QK-norm + RoPE kernel, 10.1) | `NVFP4_QKV_PACKED`, `NVFP4_QKV_PACKED_AMAX` | TE patch v7 `nvfp4_qkv_packed.py`; `firmus_kernels/qknorm_rope_packed.py`; `pretrain.py` | concat + amax pass before the FP8 cast | Bitwise | [K8.6](KERNELS/K8.6_qkv_packed.md) |
| 8.7 | Fused NVFP4 RHT cast for 2880-wide dense tensors | `NVTE_NVFP4_RHT_FUSION_COLS64` | TE patch v7 `pytorch/csrc/quantizer.cpp` | unfused Hadamard + quantize kernels | Same function | [K8.7](KERNELS/K8.7_rht_fusion_cols64.md) |
| 8.8 | Philox4x32-7 for NVFP4 stochastic rounding | build-time `NVTE_BUILD_NUM_PHILOX_ROUNDS=7` (`Dockerfile` step 4) | TE CMake option | Philox4x32-10 | **Not equivalent** (different random stream) | [K8.8](KERNELS/K8.8_philox7.md) |
| 10.1 | Fused QK RMSNorm + RoPE (NVIDIA `opt_kernels`), with Q, K, V packed and the amax | `OPT_QKNORM_ROPE`, `OPT_QKNORM_ROPE_STRICT`, `FUSED_QKV_ROPE=True` | `opt_kernels/` (NVIDIA, unchanged), `firmus_kernels/qknorm_rope_packed.py`; `pretrain.py` | TE RMSNorm (q, k) + TE fused RoPE | Same function (as NVIDIA's reference implementation; the packed forward is bitwise to NVIDIA's kernel) | [K10.1](KERNELS/K10.1_qknorm_rope_packed.md) |

Per-item write-ups, generated from one source shared with the earlier G3, G6 and G6b packages (withdrawn; see "Earlier packages" above): `KERNELS/`; code present in
the image but not enabled: `KERNELS/disabled.md`.

Code present in the patches but **not enabled** in the submitted configuration: the items marked
off in the summary (3.2, 3.4, 5.2, 5.4, 5.8) and the MLP residual add of the FC2-bias unpermute
(`NVFP4_RESGRAD_FWD=0`, replaced by 10.3), prequant scale swizzles (`NVFP4_PREQUANT_SWIZZLE*`),
Triton Adam (`NVFP4_TRITON_ADAM`), grouped weight quantization (`NVFP4_GROUP_WEIGHT_QUANT`), direct
NVFP4 output of the dQGeGLU kernel (`NVTE_CUTEDSL_FUSED_DQGEGLU_NVFP4=0`), side-stream Adam
(`NVFP4_SIDE_ADAM`), early norm (`NVFP4_EARLY_L2NORM`), early dense all-gather
(`NVFP4_EARLY_DENSE_AG`), FP4 parameter gather (`NVFP4_FP4_PARAM_GATHER`, `pretrain.py` lines
1188 and 1211), a delayed-scaling FP8 recipe for the full-attention layers (`NVFP4_FULL_ATTN_FP8`,
`firmus_kernels/swa_attention.py` line 544; 8.1 uses TE's current scaling instead), BF16 main
gradients with the precision-aware optimizer (`NVFP4_PAO_BF16_GRADS`, `pretrain.py` line 1279), the
cuDNN alpha overrides (`NVFP4_CUDNN_ALPHA_OVERRIDE`, `NVFP4_CUDNN_ALPHA_ONE`, TE patch lines
2014–2018), a timing-only reuse of stale quantized expert weights (`NVFP4_ORACLE_REUSE_WQ`, TE
patch line 1516), Triton activation quantization (`NVFP4_TRITON_ACT_QUANT`, default 0; its module
`nvfp4_act_quant` is not installed in the image, so the switch would raise at its first use), the
alternative paths selected by `NVFP4_FP4_AG_KERNEL=0`, `NVFP4_DYSTATS_RHT_T=0` and
`NVFP4_DYSTATS_TRANS=0` (each defaults to 1, the path that ran), and all `*_DEBUG_*`, `*_CHECK`,
`*_DUMP_*` and `*_TRACE_*` switches.

Switches set in the config that have **no effect** in this configuration:
`NVFP4_DQGEGLU_DIRECT_MODE` and `NVFP4_DQGEGLU_FIXED_GLOBAL_SCALE` (config lines 100–101; read only
on the fused dQGeGLU path, which needs `NVTE_CUTEDSL_FUSED_DQGEGLU_NVFP4=1`, TE patch lines
2220–2222; the config sets 0, line 155); `NVFP4_MASK_PADDED_MAIN_GRAD` (line 119; the hook is
registered only when `NVFP4_ZERO_FC1_PAD_ROWS` is off, Megatron patch line 359);
`NVFP4_CUDA_GRAPH`, `NVFP4_CUDA_GRAPH_IMPLEMENTATION` and `NVFP4_CUDA_GRAPH_SCOPE` (lines 92–94;
read by no package file; the graphs are configured by `MCORE_CUDA_GRAPH`, `CUDA_GRAPH_*` and
`conf/gpt_oss.yaml` lines 71–73); `NVFP4_MBS` (line 120; read by no package file);
`NVFP4_DY_STATS_CUDA_FC2` (line 103; read only for a scaled FC2 bias, V7 patch lines 824–831, and
the model has no FC2 bias; 8.5).

## 1. Optimizer

The optimizer is AdamW with global-norm gradient clipping, as in the reference: \(\beta_1 = 0.9\),
\(\beta_2 = 0.95\), \(\epsilon = 10^{-5}\), weight decay \(\lambda = 0.1\) (0 for biases and 1-D
parameters, Megatron's default grouping; this model has no biases), clip norm 1.0, learning rate
\(\eta_t\) from the cosine schedule with linear warmup. For each parameter element at step \(t\):

\[
s = \min\!\left(1,\ \frac{1.0}{\lVert g \rVert_2 + 10^{-6}}\right),\qquad \hat g = s\,g
\]
\[
m_t = \beta_1 m_{t-1} + (1-\beta_1)\,\hat g,\qquad
v_t = \beta_2 v_{t-1} + (1-\beta_2)\,\hat g^2
\]
\[
p_t = p_{t-1} - \eta_t\left(\frac{m_t/(1-\beta_1^t)}{\sqrt{v_t/(1-\beta_2^t)} + \epsilon} + \lambda\,p_{t-1}\right)
\]

\(\lVert g\rVert_2\) is the global L2 norm over all gradients. \(\epsilon\) is added outside the
square root, as in `apex.optimizers.FusedAdam`; no epsilon exception is used. \(p\), \(m\), \(v\)
are FP32 (the FP32 master weight is stored as the BF16 model parameter plus an int16 remainder,
which together hold the exact FP32 bits). Gradients are FP32.

**Demonstration of equivalence:** [`verification/optimizer_ab/`](verification/optimizer_ab/README.md)
runs the submitted optimizer path (1.1–1.4 together), the upstream Megatron + TE path, and
`apex.optimizers.FusedAdam` on identical inputs for 20 steps, half of them clipped, on this
package's image. The submitted path is bit-identical to the upstream path (parameters, remainders,
both moments, gradient norm, amax) and its FP32 master weights and moments are bit-identical to
apex's (0 ulp).

### 1.1 Gradient clip inside the Adam kernel

- **Replaces:** Megatron's `clip_grad_by_total_norm_fp32`, which multiplies every main gradient
  by \(s\) with `multi_tensor_scale_tensor` and writes it back, after which TE `FusedAdam` reads
  the gradient again.
- **Change:** `pretrain.py` computes \(s\) exactly as Megatron does
  (`max_norm / (total_norm + 1e-6)`, clamped to 1) and records it with
  `nvfp4_clip_adam.defer_clip`. `nvfp4_clip_adam.install` wraps the TE `FusedAdam` multi-tensor
  ops; for the FP32 and BF16-param + int16-remainder ops it calls `clip_adam_ext.cu`, which
  multiplies each gradient by \(s\) as it loads it. Any other op, or any gradient no Adam op
  reads, is scaled with the original `multi_tensor_scale_tensor` (fallback counter is 0 in the
  A/B run and in the in-run check, 9.2).
- **Equivalence:** the kernel performs the same per-element operations as TE's
  `AdamFunctor` / `AdamFunctorMasterParamRemainder`. The scaled gradient is formed as
  `__fmul_rn(g, s)` (and rounded back to BF16 for BF16 gradients) exactly as the stored result of
  the reference scale pass. The update is written with explicit `__fmaf_rn`/`__fdividef`
  intrinsics in the same fused-multiply-add shape as TE's compiled kernel.
- **Numerics / compiler options:** built with `-O3 --use_fast_math` (`Dockerfile` step 7) to match
  TE's build of `multi_tensor_adam`. The A/B comparison shows the result is bit-identical to both
  TE and apex.
- **Gate:** `total_norm` is a tensor and no other clip is pending (`pretrain.py` lines 628, 640;
  `nvfp4_clip_adam.py` `defer_clip` lines 169–175); the op is `multi_tensor_adam` or
  `multi_tensor_adam_param_remainder` with 8 scalar arguments and contiguous CUDA FP32 or BF16 lists
  of equal sizes (`_fusable`, lines 103–115; line 193). Any tensor size (`clip_adam_ext.cu` lines
  19, 220–259).
- **Falls back:** to the original Megatron clip, or the wrapped op scales its deferred gradients
  with `multi_tensor_scale_tensor` and runs the original TE op (`nvfp4_clip_adam.py` lines
  193–199). Same result; counted in `counters["fallback"]`, printed only with
  `NVFP4_CLIP_IN_ADAM_CHECK` (not set in the scored runs; `pretrain.py` lines 624, 655–660).
- **Assumes:** nothing reads the main gradients between the step and the next zeroing (they are
  left unscaled); gradients are identified by `data_ptr` (lines 174, 190); the fused path does not
  read the op's `noop` flag (lines 149–158).
- **Check:** config lines 90–91. No log line in the scored runs; [`verification/optimizer_ab/`](verification/optimizer_ab/README.md)
  ran the extension of this image.

### 1.2 Gradient norm in two launches

- **Replaces:** TE `multi_tensor_l2norm` (per_tensor=False), which launches once per 320 chunks.
- **Change:** one launch computes every chunk's sum of squares; a second adds the chunk sums in
  the reference's launch/slot order and runs the same block reduction.
- **Equivalence:** each chunk's partial sum depends only on its data and is computed with the same
  per-thread order; the final accumulation order is the reference's. Bitwise equal (A/B:
  gradient norm bitwise equal every step). The in-training check `NVFP4_EARLY_L2NORM_CHECK` is
  read only together with early norm (`NVFP4_EARLY_L2NORM=1`, `pretrain.py` lines 677, 718), which
  is not enabled.
- **Gate:** `per_tensor=False`, one tensor list, no extra arguments, extension loaded; all tensors
  one dtype (FP32 or BF16), on CUDA, non-empty, fewer than 2³¹ elements (`nvfp4_clip_adam.py`
  lines 346–353).
- **Falls back:** to TE `multi_tensor_l2norm` (lines 347, 353), silently; only the fused path
  increments `counters["l2norm"]` (line 388).
- **Assumes:** TE's `multi_tensor_apply` limits of 110 tensors and 320 blocks per launch,
  hard-coded (lines 231–232); other limits would change the slot order and the rounding of the norm.
- **Check:** config line 107. No log line in the scored runs.

### 1.3 Expert weight amax folded into Adam

- **Change:** while writing back the updated BF16 expert weights, the remainder kernel takes the
  max of |param| per tensor and `atomicMax`es it into that weight's amax slot (6.2). Slots not
  covered by a fused launch are recomputed from the shards.
- **Equivalence:** max is exact and order-independent; the value equals `param.abs().amax()`
  (A/B: 0 of 9 slots differ every step).
- **Gate:** remainder kernel only (`clip_adam_ext.cu` `launch` line 304 rejects a slot table for the
  FP32 kernel); `pretrain.py` lines 455–461; not during CUDA-graph capture and only with registered
  slots (TE patch lines 3142–3143).
- **Falls back:** `pretrain.py` lines 483–486 and `finish_fold` recompute the slots from the shards
  when the fold is off or a slot was not covered (for example an Adam call that fell back in 1.1).
  Same values; counters not printed.
- **Assumes:** finite parameters (`fmaxf` ignores NaN, `clip_adam_ext.cu` lines 188, 242); the step
  runs on one stream between `begin_fold` and `finish_fold`.
- **Check:** config line 82, with lines 143 (6.2), 128 (1.4) and 91 (1.1). No log line in the
  scored runs.

### 1.4 Phased Adam

- **Change:** the optimizer step is run on the parameters of the first 4 bucket groups (in
  forward all-gather order) one group at a time, and each group's parameter all-gather is queued
  as soon as its update is done; then the rest is updated.
- **Equivalence:** Adam is element-wise; every phase uses the same step count, learning rate and
  hyperparameters (A/B runs the step in 4 phases). This holds only with the precision-aware
  optimizer (1.5), which writes the BF16 parameter inside the Adam kernel; the code does not check
  it. Without it the early all-gathers send stale parameters: in the DP test's fixture
  `phased_without_pao` every final output differs (G6b package tests, jobs 22355, 22477; the
  phased-Adam code of `pretrain.py` is the same in this package).
- **Gate:** no closure or grad scaler, one model chunk with forward pre-hooks,
  `overlap_param_gather` on, FusedAdam not `capturable`, bucket groups present (`pretrain.py`
  lines 421–430), none of the 4 groups already dispatched (lines 436–437).
- **Falls back:** to the original one-call `FusedAdam.step` (lines 430, 437), silently.
- **Assumes:** 1.5 is on (above); FusedAdam keeps `step` per parameter group; the bucket layout
  does not change after the first step (`_phase_param_sets` is cached, lines 362–364, 382).
- **Check:** config line 128 (and line 129 for 1.5). `result_0.txt` line 2321:
  `NVFP4_PHASED_ADAM_AG rank0: _nvfp4_phase_sets phases [8, 12, 5, 12] of 388 optimizer params`,
  printed only on the phased path, after the gate.

### 1.5 Precision-aware optimizer forced on

- **Change:** with `fp8_recipe="tensorwise"` (Bridge's default under FP4) Megatron does not let TE
  manage master weights; `NVFP4_PAO_BF16_MODEL_PARAMS` sets `use_precision_aware_optimizer_no_fp8_or_ds_fp8`
  so TE `FusedAdam` holds the FP32 master as BF16 param + int16 remainder and writes the BF16
  parameter inside the Adam kernel. Valid only without FP8/FP4 parameter gather, which is off.
- **Equivalence:** the master weight is FP32 in both cases and Adam math is unchanged (the FP32
  master is bit-identical to apex `FusedAdam`'s, Section 1). The BF16 model parameter is the
  master rounded to BF16; TE rounds exact ties away from zero where Megatron's cast rounds them to
  even, so the two can differ on exact ties (42–261 of 33.8 M elements per step in the A/B test).
- **vs reference / vs FP64:** G6b package tests: FP32 master, `exp_avg` and `exp_avg_sq` equal the
  upstream FP32 master path (0 of 95,668,277 and 95,566,890 elements differ, two seeds); the BF16
  parameter differs from round-to-nearest-even of the upstream master in 757 and 736 elements, all
  at exact BF16 ties (job 22240); over 2000 steps 1,463,641 elements in total, all at ties (job
  22322). On this image `k1_4_k1_5_optimizer` (job 22828) has 5 of 13 cases with a difference
  (Section 9). No FP64 comparison.
- **Gate:** any optimizer config with `use_precision_aware_optimizer=True` (`pretrain.py` line
  211); with `NVFP4_PRECISION_AWARE_OPT=0` the wrapper changes nothing.
- **Falls back:** none. FP8 or FP4 parameter gather with the switch raises `RuntimeError` (lines
  1187–1190).
- **Assumes:** BF16 model parameters (`params_dtype=torch.bfloat16`, line 1195), so the high half
  of the master is the parameter; other dtypes are not checked.
- **Check:** config lines 124 and 129. `result_0.txt` line 257
  `NVFP4 PAO on BF16 model params (fp8_recipe=tensorwise)` and line 273
  `use_precision_aware_optimizer: True  (Mcore default: False)`.

## 2. Precision recipe

| Component | Format |
|---|---|
| Linear and expert GEMMs, layers 0–21 | NVFP4: E2M1 values, E4M3 scale per 16 elements, FP32 per-tensor scale. Weights use 2D 16×16 block scaling. Gradients use a random Hadamard transform (RHT) and stochastic rounding (upstream TE NVFP4 recipe; `NVTE_NVFP4_DISABLE_RHT=0`, `NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING=0`, `NVTE_NVFP4_DISABLE_2D_QUANTIZATION=0`) |
| Linear and expert GEMMs, layers 22–23 | MXFP8 (E4M3 values, E8M0 scale per 32 elements) |
| LM head | forward MXFP8; dgrad and wgrad BF16 |
| Attention (SDPA) of the full-attention layers | FP8 (E4M3 forward / E5M2 backward, current scaling; output in BF16) (8.1) |
| Sliding-window attention, QK RMSNorm + RoPE, router, communication | BF16, except the NVFP4 expert weight all-gather (8.2) |
| Master weights, optimizer state | FP32 |

nvfp4, fp8, bfloat16 and fp32 are in the rule's pre-approved list ("Numerical formats"). `mxfp8`
is not named in the list. MXFP8 stores FP8 E4M3 values with a power-of-two (E8M0) scale per 32
elements; it is used for the tail layers (2.2) and the LM head forward (2.3). The MLPerf log has no
MXFP8 entry: it records `lowest_numerical_precision_in_linear: nvfp4`,
`lowest_numerical_precision_in_attn: fp8` and `lowest_numerical_precision_in_comm: nvfp4`
(`result_0.txt` lines 253–255), the values of `MLPERF_LINEAR_PRECISION`, `MLPERF_ATTN_PRECISION`
and `MLPERF_COMM_PRECISION` (config lines 210, 53 and 54; `pretrain.py` lines 1513–1515). The MXFP8
modules appear only in two Firmus log lines, `NVFP4 LM head MXFP8 (e4m3, mode=fprop) ...` and
`NVFP4 tail MXFP8 (e4m3) layers {22: [...], 23: [...]}` (`result_0.txt` lines 558–559). Scaling,
Hadamard transforms and stochastic rounding are the rule's listed numerically safe techniques.

The communication entry `nvfp4` is the expert weight all-gather (8.2); gradient reduce-scatter is
BF16.

Upstream TE options set differently from NVIDIA's recipe (options of the pinned TE, not Firmus
code):

| Option | NVIDIA's recipe | This implementation | Where | Effect |
|---|---|---|---|---|
| `NVTE_USE_FAST_MATH` | not set | `1` | config line 170 | fast-math arithmetic in TE's quantization kernels; its measured effect on the 2880-wide RHT cast is in 8.7 |
| `NVTE_BUILD_NUM_PHILOX_ROUNDS` | TE default, 10 | `7` | `Dockerfile` line 117 | random stream of the stochastic rounding (8.8) |
| `NVTE_CUDA_ARCHS` | `100a;103a;107a` | `103a` | `Dockerfile` line 116 | build target only: on GB300 the `sm_103a` code runs in both builds |

For 2.1 (NVFP4 linear layers 0–21):

- **Gate:** TE modules of the layers outside the tail range (2.2). The expert GEMM K dimension is
  padded 2880 → 2944 so that TE's RHT and grouped quantization kernels accept it (3.1).
- **Falls back:** none inside this item; modules that are not TE linears (router) stay BF16, and
  the recipe's BF16 tail is replaced by 2.2.
- **Assumes:** TE's NVFP4 shape requirements hold for every NVFP4 GEMM at the training shapes; the
  requirement is TE's and is not re-checked by Firmus code.
- **Check:** config lines 32–33. `result_0.txt` line 868 `fp4_recipe='nvfp4'`; line 253
  `lowest_numerical_precision_in_linear: nvfp4` is the declared value of `MLPERF_LINEAR_PRECISION`
  (config line 210; `pretrain.py` line 1513), not a measurement.

### 2.2 Tail layers in MXFP8

The NVFP4 recipe keeps the last 2 layers in BF16 (`NUM_LAYERS_AT_END_IN_BF16=2`). Their TE modules
(attention linears and the expert MLP) run inside an MXFP8 autocast instead. Attention itself and
the router stay BF16; parameters and optimizer are unchanged.

- **Gate:** layers in `tail_layer_indices`, which needs `first_last_layers_bf16` (TE patch lines
  848–849; config line 31).
- **Falls back:** none at run time. With `FIRST_LAST_LAYERS_BF16` off the set is empty; with
  `NVFP4_TAIL_MXFP8_ATTN=0` or `_MLP=0` (not set) those modules stay BF16. Both show in the log line
  below.
- **Assumes:** every tail GEMM meets TE's MXFP8 shape requirements (leading dimensions divisible by
  8); a violation raises in TE.
- **Check:** config lines 78 and 146. `result_0.txt` line 559:
  `NVFP4 tail MXFP8 (e4m3) layers {22: ['mlp.experts', 'self_attention.linear_proj', 'self_attention.linear_qkv'], 23: [...]}`.

### 2.3 LM head forward in MXFP8

The output layer's forward GEMM runs through a TE `Linear` in MXFP8 that shares the same weight
parameter; its backward GEMMs stay BF16. Parameters, optimizer and state-dict keys are unchanged.

- **Gate:** TP 1 without sequence parallel, no bias, both weight dimensions divisible by 32 (TE
  patch lines 149–154); mode `full`, `fprop`, `fwd` or `bwd` (lines 173–175). TE's MXFP8 cast also
  needs the token count divisible by 8.
- **Falls back:** none; a gate failure raises `RuntimeError`. With shared embedding and output
  weights the layer is not patched and no line is printed (line 230).
- **Assumes:** the BF16 `dw` set as `weight.grad` reaches `main_grad` through Megatron's DDP hook as
  the upstream wgrad does (not tested through Megatron's hook); the micro-batch token count
  (3 × 8192) stays divisible by 8.
- **Check:** config lines 117–118. `result_0.txt` line 558:
  `NVFP4 LM head MXFP8 (e4m3, mode=fprop) weight (128256, 2880) gaf=True`.

## 3. MoE shape adaptation

### 3.1 Expert width 2880 → 2944

- **Why:** TE's NVFP4 RHT and grouped quantization kernels require the K dimension to be a multiple
  of 128; GPT-OSS hidden and expert FFN width are 2880.
- **Change:** expert weights are allocated with physical width 2944 (FC1 `[2 × 2944, 2944]`,
  FC2 `[2944, 2944]`). Expert inputs are padded from 2880 to 2944 with zeros (column 2880 is set to
  1 only when 3.2 is on, which it is not) and expert outputs are cropped back to 2880. At
  initialization every padded weight entry is set to zero: the FC1 rows and FC2 columns of the
  padded FFN channels (`NVFP4_ZERO_FC1_PAD_ROWS`; GLU interleave 32 is respected), and the FC1
  input columns 2880–2943 and FC2 output rows 2880–2943 (`NVFP4_ZERO_HIDDEN_PAD=1`, config line 152;
  `megatron_lm_firmus_nvfp4.patch` lines 414–431). The experts have no biases
  (`add_bias_linear=False`).
- **Equivalence:** the padded FFN channels have zero pre-activation, so the SiLU-gated GLU output
  (no offset, \(\mathrm{silu}(0)\cdot 0=0\)) is zero; FC2's padded input columns are zero, so the
  2880 logical outputs equal the unpadded computation. Gradients of all padded entries are exactly
  zero (the corresponding activations or upstream gradients are zero, and zero stays zero under RHT
  and stochastic rounding). With zero gradient and zero value, Adam and weight decay leave them at
  zero.
- **Numerics:** `NVFP4_FOLD_FC1_BIAS=0` (config line 109). All padded entries stay exactly zero, so
  they do not change any tensor's FP32 per-tensor amax used for the NVFP4 global scale. 2880 is a
  multiple of 16, so no 16×16 weight block mixes padded and logical entries.
- **Padding stays zero, so it is equivalent to not existing.** Run evidence:
  [`verification/padding_zero/`](verification/padding_zero/README.md). One run of the submitted
  image and config (evaluation every 10 steps for this check only, stopped after step 30) checks
  every padded entry of the FC1/FC2 weights on all 8 ranks: the BF16 parameter, its gradient, and
  the distributed optimizer's FP32 master remainder, `exp_avg` and `exp_avg_sq`, before warmup, at
  train start, after steps 1, 2, 10, 11, 20, 21, 30, and around both evaluations. All checks report
  zero nonzero entries, zero negative zeros and maximum absolute value 0, while the logical FC1
  weights change during the same steps.
- **Gate:** `NVFP4_PADDED_EXPERT_HIDDEN > 0` with FP4, at least the hidden and FFN width, a multiple
  of 128 (Megatron patch lines 243–247); otherwise `AssertionError` at model build.
- **Falls back:** unset, the upstream 2880-wide experts are built (lines 250–252); TE's NVFP4
  grouped kernels would not accept them.
- **Assumes:** x and dy are zero in columns ≥ 2880 (3.3, or the `F.pad` fallback, lines 462–480);
  GLU interleave 32; the GLU maps a zero pre-activation pair to zero (true for SiLU without offset,
  `glu_linear_offset=0.0`).
- **Check:** config lines 123, 151 and 152; `result_0.txt` line 323
  `moe_mlp_glu_interleave_size: 32`, line 772 `glu_linear_offset=0.0`;
  [`verification/padding_zero/`](verification/padding_zero/README.md) shows every padded entry 0 in
  a run of the submitted image.

### 3.2 FC1 bias folded into the FC1 weight (off in this configuration)

`NVFP4_FOLD_FC1_BIAS=0`, and the reference model has no FC1 bias, so there is nothing to fold; input
column 2880 is zero padding. The description below applies only when it is on.

- **Why:** the fused NVFP4 grouped GEMM + GLU path (cuDNN FE) produces NaN when given an FC1
  bias.
- **Change:** at initialization each expert's FC1 bias is copied into FC1 weight column 2880 and
  the bias parameter is set to zero. Input column 2880 is the constant 1 (3.1), so FC1 computes
  \(W x + b\). The bias parameter stays in the model (TE requires bias and weight to share
  `requires_grad`); its gradient is marked as already accumulated so DDP skips it and it stays 0.
- **Forward:** identical function to the reference in exact arithmetic.
- **Impact (not equivalent):**
  - **Weight decay.** The reference applies no weight decay to biases (Megatron exempts `*.bias`
    and 1-D parameters). The folded bias lives in a 2-D weight and receives weight decay 0.1.
  - **Precision.** The folded bias is quantized with the weight (NVFP4 16×16 blocks) instead of
    being added in BF16, and its gradient comes from the NVFP4 wgrad GEMM (RHT, stochastic
    rounding) instead of a BF16 bias reduction.

- **Gate:** the fold runs only with `NVFP4_FOLD_FC1_BIAS=1`, padding on and an FC1 bias (Megatron
  patch lines 362–363).
- **Falls back:** otherwise `fold_fc1_bias = False` (lines 411–412) and the upstream path runs;
  nothing is logged.
- **Assumes (if on):** the permute writes 1 into column 2880 of every routed row (TE patch lines
  3722–3724); nothing checks it.
- **Check:** config lines 109 and 112. The fold code prints nothing; `result_0.txt` line 767
  `add_bias_linear=False` shows the experts have no bias to fold.

### 3.3 Pad and crop fused into permute / unpermute

The permute kernel writes 2944-wide rows (logical values, then zeros; column 2880 is 1 only with
3.2, which is off) directly into the expert buffer, and unpermute reads only the first 2880 columns
through the row stride. This replaces separate `F.pad` and crop copies. Values copied and written
are the same as the unfused pad + permute; bitwise. In training the forward unpermute of the MLP
output is the one of 10.3, which reads the rows through the same row stride.

- **Gate:** `NVFP4_FUSED_PERMUTE_PAD=1`, input narrower than `NVFP4_PADDED_EXPERT_HIDDEN`, no
  `pad_offsets`, no `scale` (TE patch lines 3712–3717); full-row launch only if the kernel is
  autotuned (line 3734).
- **Falls back:** to the upstream 2880-wide permute, and the experts pad with `F.pad` (Megatron
  patch lines 466–480), silently.
- **Assumes:** the unpermute's `hidden_size` is the logical width taken from the token shape (TE's
  upstream wrapper); with 2944 the output would keep the pad columns.
- **Check:** config lines 114, 125 and 126. No log line of its own; 8.4's router part raises at
  install unless `NVFP4_FUSED_PERMUTE_PAD=1` with padding (V7 patch lines 2317–2320), and
  `result_0.txt` line 219 shows `NVFP4_RESGRAD installed (router 1, fwd 0)`.

### 3.4 FC2 expert bias added in unpermute (off in this configuration)

`NVFP4_DEFER_FC2_BIAS=0` and `NVFP4_DEFER_FC2_BIAS_BF16=0`: the reference model has no expert
bias, so there is no bias to defer. In the G6b package this item applied the scaled FC2 bias in the
unpermute and was the prerequisite of `NVFP4_RESGRAD_FWD`; here 10.3 adds the MLP residual in the
unpermute without it.

- **Gate:** fused path: switch, FC2 bias and `bias_scale` present (TE patch lines 2123–2127). BF16
  tail wrapper: installed only with `NVFP4_DEFER_FC2_BIAS_BF16=1` (`pretrain.py` line 218), output
  size `bias_scale.numel() × NVFP4_PADDED_EXPERT_HIDDEN` (lines 230–235).
- **Falls back:** without a pending bias the plain unpermute runs (TE patch lines 3754–3755 take
  the pending bias only if one was recorded).
- **Assumes:** not applicable while off.
- **Check:** config lines 95 and 96; `result_0.txt` line 767 `add_bias_linear=False`. 10.3's install
  raises if `NVFP4_DEFER_FC2_BIAS=1` (`firmus_kernels/resgrad_fwd_nobias.py` lines 74–75), and
  `result_0.txt` line 220 shows it installed.

## 4. Correctness fixes and CUDA-graph capture

### 4.1 Tail of fixed MoE buffers zeroed

With fixed-size expert buffers (4.3), the cuDNN FE `glu_hadamard` grouped kernel launches no CTAs
for up to (experts − 1) alignment blocks past the last expert's rows, so those rows of the
`torch.empty_strided` output kept stale values (including NaN) that later reached the router's
zero-probability dummy rows. The wrapper now clears that maximum tail before the launch. Active
rows are overwritten by the kernel.

- **Gate:** `use_dynamic_sched and valid_m > 0 and l > 1` (cuDNN FE patch lines 439, 1505); TE path
  `out_shape[0] > 0 and num_groups > 1` (TE patch line 2486).
- **Falls back:** otherwise nothing is cleared (static schedule or one expert: every row is
  written). Silent.
- **Assumes:** the unwritten suffix is at most (experts − 1) alignment blocks (cuDNN FE patch lines
  1499–1503); the TE path assumes alignment 256 (TE patch line 2487).
- **Check:** no log line. Config line 108 (4.3); no run has a NaN loss, and every run ends with
  `run_stop` status `success` (`result_0.txt` line 2732).

### 4.2 Persistent zeroed grouped-GEMM outputs

TE calls cuBLASLt grouped GEMM with device-pointer beta, so with beta = 0 cuBLAS still reads C,
which aliases the `torch.empty` output; stale NaN produced `0 × NaN = NaN`. Each output shape now
uses one persistent buffer that is zeroed once and reused, so C never contains non-finite values.
Results equal the upstream path in the absence of stale NaN, up to the sign of exact zeros, which C
can set. The package tests do not show this in every arrangement: with two layers sharing the
buffers, under graph replay with new inputs, and for the graph backward of two captured training
layers, outputs differ from the zero-filled or eager path in ways not yet explained (on this
image `k4_2_cross_layer`, `k7_3_k4_2_graph` and `k4_2_bwd_graph_vs_eager`, Section 9; per-case
sizes are in the records).

- **Gate:** `buffer is None and _PERSISTENT_GEMM_OUT` (TE patch line 1602), read once at import
  (line 1510).
- **Falls back:** a caller-supplied buffer or the switch off gives a zero fill per call (lines
  1609–1610). Silent.
- **Assumes:** no consumer reads an FC2 output or FC1 dgrad after the next GEMM of the same role
  and shape; no consumer reads rows past the used rows; the buffers stay alive across graph capture
  and replay (line 1513).
- **Check:** config line 127; no log line. No run has a NaN loss; every run ends with `run_stop`
  status `success` (`result_0.txt` line 2732).

### 4.3 Fixed-size EP1 MoE buffers

So the MoE layer can be captured in CUDA graphs (`CUDA_GRAPH_SCOPE=[attn,moe]`):

- expert row counts use a static upper bound; the extra (dummy) rows get routing probability
  exactly 0, so they contribute nothing to outputs or gradients;
- token counts per expert stay on the device (no device-to-host copy);
- the chunk reordering that is an identity under EP1/TP1 is skipped;
- the TE fused router's routing map is cloned before downstream padding modifies it.

No token is dropped; routing and the routed computation are unchanged. The fixed buffer puts all
extra rows into one expert, so the row layout differs from the dynamic path's and dx, dprobs and
the weight gradients differ by summation order (G6b package tests: fixed vs dynamic out 0 elements
differ, dx and weight gradients differ in nearly every element, job 22405; the dynamic path given
the fixed path's row layout: 0 elements differ, job 22356).

- **Gate:** Megatron patch lines 747–754, inside the router-padding branch (lines 774–776);
  device-only metadata: switch, dropless, EP 1, TP 1 (lines 846–851).
- **Falls back:** dynamic sizes with the host sync (lines 786–790), silently; with `moe` in the
  graph scope that path cannot be captured.
- **Assumes:** tokens × top-k ≥ the bound (asserted; 1, 255, 256 and 257 tokens raise in
  `k4_3_pad_routing_map`); the token count is a multiple of the alignment (not checked); router
  probabilities are exactly 0 off the top-k.
- **Check:** config lines 83, 97 and 108. `result_0.txt` line 297 `cuda_graph_modules` with `attn`
  and `moe`; lines 912–913 `moe_expert_capacity_factor=None`,
  `moe_pad_expert_input_to_capacity=False` (a combination Megatron patch lines 879–890 reject unless
  the switch is on); line 584 `expert_model_parallel_size=1`.

## 5. Fused and replacement kernels

### 5.1 Sliding-window attention (Triton)

Replaces cuDNN fused attention for the sliding-window layers (layers 0, 2, …, 22; window 128,
causal, GQA 64 query heads / 8 KV groups). G6c: the kernel is `firmus_kernels/swa_attention.py`, a
copy of the TE-patch kernel with a compile-time `HAS_SINK` (10.2); this model has no sink. Same
semantics as TE `DotProductAttention(softmax_type="vanilla", window_size=(128, 0))`: query \(i\)
attends keys \(i-128 \ldots i\). BF16 inputs, FP32 softmax and accumulation, BF16 output. Backward
is deterministic (each dK/dV program owns a key block). Same function; rounding differs from cuDNN
through accumulation order. Full-attention layers still use cuDNN (8.1).

- **Gate:** `firmus_kernels/swa_attention.py` lines 556–567: window set with right size 0 and left
  size > 0, softmax `learnable` or `vanilla`, causal mask, no attention bias, no packed sequences,
  `sbhd`, BF16, head dim 64 or 128; `_check` (lines 421–430) raises on malformed tensors.
- **Falls back:** to the original TE forward (lines 587–588), silently; only the Triton path logs.
- **Assumes:** attention dropout 0 (not tested by the gate; `pretrain.py` line 1084); heads per KV
  group a multiple of the packed tile (4 heads; training: 8; not checked, 10.2); no context
  parallelism (`CONTEXT_PARALLEL=1`, config line 16).
- **Check:** config lines 144 and 145. `result_0.txt` line 1039: `NVFP4 SWA Triton attention
  window=(128, 0) q=(8192, 3, 64, 64) k=(8192, 3, 8, 64) packed=True ...`.

### 5.2 QKV + RoPE (Triton) (off in this configuration)

`NVFP4_TRITON_QKV_ROPE=0`: the Triton kernel replaces `tex.fused_qkv_rope_forward/backward` without
a QK norm; with QK RMSNorm the RoPE is computed by 10.1 instead. Not installed (`pretrain.py` line
71).

- **Gate (if on):** TE patch lines 1128–1141: `sbhd`, no `start_positions`, not interleaved, CP 1,
  4-D contiguous `qkv`, K and V split sizes equal to the head dim, head dim 32, 64 or 128.
- **Falls back:** to TE's `apply_fused_qkv_rotary_pos_emb` with the same arguments (lines
  1143–1147), silently. Here Megatron's call goes to 10.1's dispatcher instead
  (`opt_kernels/qknorm_rope_patch.py` line 260).
- **Assumes:** not applicable while off.
- **Check:** config line 147; `result_0.txt` line 1065 `[opt qknorm_rope] first forward:
  retyped=48 fused=24 fallback=0` (10.1).

### 5.3 RMSNorm backward (Triton)

Replaces `tex.rmsnorm_bwd` (both call sites). FP32 arithmetic per row; dgamma is reduced through
per-program partials, so its summation order differs from TE's two-kernel reduction. Same
function. G6b package tests, vs `tex.rmsnorm_bwd` on captured inputs of 7 layers: dx 276–341 of
70,781,760 elements differ, dgamma 0–3 of 2880 (job 22434); against FP64 the Triton and TE errors
are of the same size (job 22168). The QK norms of this model are not on this path (10.1).

- **Gate:** none. Every call of `tex.rmsnorm_bwd` and of the op module's `rmsnorm_bwd` goes to the
  Triton function (`pretrain.py` lines 59–60).
- **Falls back:** never while the switch is on.
- **Assumes:** TE's positional signature (TE patch line 1266); one row per program (4096 for 2880
  columns); one FP32 rsigma per row.
- **Check:** config lines 148 and 140. No log line of its own; 8.4's install raises unless
  `tex.rmsnorm_bwd` is this function (V7 patch lines 2276–2277), and `result_0.txt` line 219 shows
  `NVFP4_RESGRAD installed (router 1, fwd 0)`.

### 5.4 Attention output bias + residual (off in this configuration)

`NVFP4_FUSED_BIAS_RESIDUAL=0`: the kernel computes `bf16(residual + bf16(x + bias))` for the
attention output bias, which this model does not have. Not installed (`pretrain.py` line 89).

- **Gate (if on):** `eligible` (TE patch lines 942–955): training, dropout 0, a bias, equal BF16 or
  FP16 dtypes, equal shapes of x and residual, 1-D bias of length `x.shape[-1]`, all contiguous.
- **Falls back:** without a bias, 8.4's bias-dropout-add takes its residual-add path (V7 patch
  lines 2298–2299); 8.4 uses this kernel only when a bias is present (line 2296).
- **Assumes:** not applicable while off.
- **Check:** config line 113; `result_0.txt` line 767 `add_bias_linear=False`.

### 5.5 Interleaved clamped SwiGLU (tail layers; not used)

The kernel computes the GPT-OSS clamped SwiGLU (limit 7, alpha 1.702, offset 1) directly on the
GLU-interleaved (32) FC1 output instead of de-interleaving it first. It accepts only
`ScaledClampedQGeGLU` (TE patch line 1389). This model has SiLU-gated experts without clamp
(`pretrain.py` lines 1061–1064, `clamped_quick_gelu_activation` default False in
`conf/gpt_oss.yaml` line 38), so the path is never taken. The tail layers' FC1 + GLU + FC2 run in
the fused cuDNN grouped-MLP op as for the NVFP4 layers (TE patch lines 1903–1905): every result
file prints `NVFP4 grad zero: 1536 overwritten params skipped` (`result_0.txt` line 2294); the
count is set only inside the fused op's wgrad (TE patch lines 1842–1848), and 1536 = 24 layers × 64
expert weights.

- **Gate:** `_use_triton_interleaved_glu` (TE patch lines 1385–1395): switch `"1"` (line 1388,
  default 1, not in the config), `ScaledClampedQGeGLU` with an interleave size, BF16 CUDA input
  whose last dim is a multiple of 2 × interleave; BF16 compute dtype (line 1408).
- **Falls back:** to TE's GLU path, silently; with this model always.
- **Assumes:** not applicable (path not taken).
- **Check:** `result_0.txt` lines 770 `activation_func=<function silu ...>` and 773
  `activation_func_clamp_value=None`.

### 5.6 Cross-entropy gradient in place

TE's Triton cross entropy computes the gradient in FP32 and writes it to an FP32 buffer that is
then cast to BF16. With `NVFP4_CE_BF16_INPLACE`, each row block's gradient is written as BF16 into
the logits buffer after that block has been read (the loss uses the precomputed target logit).
TE's check whether `grad_output` equals 1.0 (which synchronizes the stream) now runs only when
`grad_output` has one element; a per-token `grad_output` is multiplied in as before. Same function.
The BF16 gradients are bitwise equal to the reference only when every per-token `grad_output` is 0
or a power of two; otherwise the product is rounded once more. This model uses EOD loss masking
(`eod_mask_loss=True`, `pretrain.py` line 1014).

G6b package tests: loss 0 elements differ in every case; dlogits on the real shape with a random
per-token `grad_output` 1,286,686,635 of 4,941,938,688 elements differ (max 2⁻⁸); relative L2 error
vs FP64, FP32 gradient / BF16 in place: constant 1/N 0.0019581 / 0.0019587, random per token
0.0014903 / 0.0014905 (job 22220).

- **Gate:** none besides the switch (TE patch line 3493); applies to every call, whatever the logits
  dtype.
- **Falls back:** never. FP32 logits get an FP32 gradient; FP16 logits an FP16 gradient.
- **Assumes:** the logits are not read after the cross-entropy forward (in training and evaluation
  only the loss is used); the code does not check it.
- **Check:** config line 89. No log line; the switch is read at import.

### 5.7 dGLU kernel tuning

`dglu_kernel_ep.py` is the cuDNN FE CuTe DSL grouped GEMM + dGLU kernel (NVIDIA, MIT license) with
two changes: `DGLU_NO_AMAX=1` skips a post-RHT amax that the wrapper always allocates for BF16
outputs but this path never reads, and `DGLU_C_STAGE=4` raises the C-operand pipeline stages for
NVFP4 inputs from 2 to 4. Neither changes the arithmetic of the outputs that are used. Without an
FC1 bias the epilogue writes no dbias (10.4).

- **Gate:** none of its own; every fused grouped-MLP backward uses the class from the file.
  `DGLU_C_STAGE` applies only to 4-bit A operands; the MXFP8 tail layers keep the default stages.
- **Falls back:** never; an empty `NVFP4_DGLU_KERNEL_FILE` raises (TE patch lines 2544–2546).
- **Assumes:** the file's class has the wrapper's constructor and call interface; a mismatch fails
  at compile time.
- **Check:** config lines 98, 24 and 25. No log line; a run without the file cannot start the
  backward.

### 5.8 FC1 dbias in a separate reduction (off in this configuration)

`NVFP4_DGLU_TRITON_DBIAS=0`, and the model has no FC1 bias, so there is no FC1 bias gradient: the
dGLU epilogue computes none, and 8.5's FC1 pass computes only the amaxes.

- **Gate (if on):** switch on, FC1 bias, NVFP4 layer, `NVTE_CUTEDSL_FUSED_DQGEGLU_NVFP4` off (TE
  patch lines 2228–2233).
- **Falls back:** without an FC1 bias nothing is computed; the FC1 dy statistics run with
  `dbias=fc1_op.has_bias`, i.e. amax only (V7 patch lines 728–735).
- **Assumes:** not applicable while off.
- **Check:** config line 99; `result_0.txt` line 767 `add_bias_linear=False`.

### 5.9 Expert wgrad written, not accumulated

Each step has one microbatch per rank (`MBS=3`, gradient accumulation 1), so the cuDNN expert
wgrad kernel writes the main gradient directly (`accumulate=False`) and these buffers are skipped
by the gradient-buffer zero fill. Experts that receive no tokens get 0 written by the kernel. The
resulting gradient equals `0 + wgrad`; bitwise.

- **Gate:** switch `"1"` and Megatron's fused wgrad accumulation, on the cuDNN wgrad path of the
  fused grouped MLP (TE patch lines 1837, 1842–1845); NVFP4 and MXFP8 layers alike.
- **Falls back:** without `accumulate_into_main_grad` the original accumulate path runs, silently.
- **Assumes:** one micro-batch per optimizer step; the code does not check it (`NVFP4_MBS` is not
  read). With gradient accumulation > 1 only the last micro-batch's expert gradients would reach
  the optimizer, silently.
- **Check:** config lines 122, 42 and 51. `result_0.txt` line 2294 `NVFP4 grad zero: 1536
  overwritten params skipped, 25 spans, 1378120512/21347201920 elements`; lines 228, 251 and 229:
  `global_batch_size` 24, `micro_batch_size` 3, `gradient_accumulation_steps` 1.0.

## 6. NVFP4 weight quantization pipeline

Values are unchanged by all three; only where and when the same quantization runs. 6.1 does not
run.

- **6.1 Cached quantized tensors:** each expert weight keeps one NVFP4 tensor that is re-quantized
  in place each step instead of allocating a new one. `NVFP4_CACHED_WEIGHT_QUANT=1` is set, but the
  expert weight quantization selects the 6.2 path first when `NVFP4_SHARD_WEIGHT_AMAX=1`, so this
  path does not run in the submitted configuration (in-run counters on the G6 image, built from the
  same TE patches: 0 calls, job 22329). Class: not used.
  - **Gate:** `NVFP4_CACHED_WEIGHT_QUANT=1` and neither 8.2's lookup nor
    `NVFP4_GROUP_WEIGHT_QUANT` nor `NVFP4_SHARD_WEIGHT_AMAX` returned first (TE patch lines
    1678–1681; V7 patch lines 461–463).
  - **Falls back:** in the submitted configuration the 6.2 path always runs instead; nothing is
    logged about 6.1.
  - **Assumes:** the cached tensor's shape and quantizer settings do not change per key; not
    relevant while the path is unreachable.
  - **Check:** config lines 88 and 143 (both 1). `result_0.txt` line 1052 is printed only inside
    the 6.2 function (TE patch lines 1802–1808), so the 6.2 path ran.
- **6.2 Amax from optimizer shards:** after the optimizer step each rank reduces |w| over its own
  shard of each expert weight and one MAX all-reduce combines them; quantization uses
  `tex.nvfp4_quantize_with_amax` with that value instead of recomputing the amax over the gathered
  weight. Max is exact, so quantized weights are bit-identical. The in-run check
  `NVFP4_SHARD_WEIGHT_AMAX_CHECK` (not set in the scored runs) compares, before each optimizer step,
  every slot with `p.abs().amax()` of the rank's BF16 parameter (`pretrain.py` lines 323–333). In
  job 22923 (9.2) rank 0 had 0 of 1,408 mismatched at all 16 checks; ranks 1–7 had 0 at the first
  11 checks, then 1, 2, 3, 4 and 5 of 1,408. With `NVFP4_FP4_AG=1` (8.2) this check does not test
  6.2: the BF16 rows outside a rank's own shard are not refreshed once a bucket is gathered in FP4
  (V7 patch L1080–1082), so the full-parameter `p.abs().amax()` it computes can be stale. Why rank
  0 shows no mismatch is not explained (open point). The evidence for 6.2 is the kernel
  comparison `k6_2_shard_amax` on this image has only its fixture differing (Section 9). A weight
  that fails the NVFP4-only gate is quantized by its own TE quantizer. Every scored run prints this
  once: `NVFP4_SHARD_WEIGHT_AMAX: 32 expert weights use the TE amax path (None)`. The line follows
  the first MXFP8 tail-layer expert forward in the log, so the 32 weights are one MXFP8 tail-layer
  expert op sent to TE's MXFP8 quantizer, the reference path. This is inferred from the log
  position and the gate code.
  - **Gate:** NVFP4 quantizer without RHT and without amax reduction (TE patch lines 1789–1797);
    BF16, contiguous, registered weight (lines 2952–2956); fewer than 8192 slots (lines
    2960–2962); weight storage inside its DDP bucket's param buffer (lines 2996–3006).
  - **Falls back:** a weight failing the gate is quantized by `quantizers[i](weights[i])` (lines
    1810–1811), printed once per op (lines 1802–1808); a failure in `register` or `_build_plan`
    disables the module and prints `NVFP4_SHARD_WEIGHT_AMAX disabled: <reason>` (lines 2931–2935).
  - **Assumes:** the bucket layout does not change after the plan is built, and each rank's shard
    is `numel // dp` of the bucket (line 3010); otherwise slots would hold wrong ranges, without an
    error.
  - **Check:** config line 143. `result_0.txt` line 1052 (above; `None` is the disabled reason, so
    the module was not disabled); no result file contains `NVFP4_SHARD_WEIGHT_AMAX disabled`.
- **6.3 Side-stream quantization:** each layer's expert weight quantization is issued on a side
  stream at the start of the layer and overlaps attention; the MoE GEMMs wait for it, and it waits
  for the layer's parameter all-gather (`NVFP4_PREQUANT_WAIT_AG=1`).
  - **Gate:** grad enabled and an FC1 op with a recorded peer (`pretrain.py` lines 791–800); the
    peer exists only for per-expert weights (TE patch lines 1939–1944); the stashed result is used
    only if its usage matches (lines 1773–1774).
  - **Falls back:** inline quantization in the MoE forward, silently (first eager forward;
    evaluation replays the captured launch, 7.4).
  - **Assumes:** with `NVFP4_PREQUANT_WAIT_AG=0` the cast could read weights not yet gathered (the
    switch is 1); the layer's attention runs before its MoE forward.
  - **Check:** config lines 121 and 131. No log line (`NVFP4_PREQUANT_WAIT_AG_LOG` is 0 by default,
    `pretrain.py` line 747).

## 7. Scheduling, communication and CUDA graphs

- **7.1 Param all-gather prefetch:** all bucket groups' parameter all-gathers are queued after the
  first one instead of one at a time. Each gather writes its own slice of the parameter buffer.
  - **Gate:** switch 1, `overlap_param_gather` on, not `force_sync` (`pretrain.py` lines 252, 265).
  - **Falls back:** the original one-at-a-time behaviour, silently.
  - **Assumes:** no code issues `start_param_sync` for a group whose parameters are not yet updated
    (1.4 uses the unwrapped function, lines 352–354).
  - **Check:** config line 130. No log line.
- **7.2 Gradient buffers:** gradient-buffer zeroing runs on a side stream that the next backward
  waits on; the per-parameter `main_grad.add_` calls in the DDP hook are deferred and applied with
  one `_foreach_add_` before the next bucket reduction is issued. Same values only with one
  micro-batch per step (global batch 24 = micro-batch 3 × 8 ranks, gradient accumulation 1, as
  here; 5.9); this is not guarded. The DP module test runs two micro-batches per step; there the
  deferred adds put the same `main_grad` twice in one `_foreach_add_`, which differs from two
  sequential adds, and the final parameters and optimizer state differ from the reference in every
  entry (G6b package tests, jobs 22355, 22477; the DDP patch is the same here). The one-micro-batch
  case is not covered by that test.
  - **Gate:** async zero when not capturing a graph (Megatron patch line 146); batched accumulation
    when `overlap_grad_reduce` is on (line 93; config line 172).
  - **Falls back:** zeroing on the current stream during capture (lines 160–161); immediate `add_`
    without `overlap_grad_reduce` (lines 96–97). Silent.
  - **Assumes:** at most one hook call per parameter between two flushes, i.e. one micro-batch per
    step; nothing checks it.
  - **Check:** config lines 86 and 87. `result_0.txt` line 2294 `NVFP4 grad zero: 1536 overwritten
    params skipped, 25 spans, ...` shows the span zeroing ran (printed in both the side-stream and
    the in-stream branch); no log line marks batched accumulation.
- **7.3 Graph inputs:** Megatron passes a fresh all-false attention mask on every graph replay
  (causal attention does not read it); one cached mask is passed instead and TE skips copying an
  unchanged input. Each layer's graph output buffer is used directly as the next layer's static
  input (and likewise for gradients); a mismatch raises an error.
  - **Gate:** mask: `attention_mask` None and `hidden_states` present (`pretrain.py` line 890).
    Chain: `NVFP4_GRAPH_CHAIN_IO=1`, training capture (TE patch lines 20–22), matching shape, dtype
    and stride, input not shared (lines 24–31, 43–44).
  - **Falls back:** non-matching layers keep their own static inputs and the copy; a changed mask is
    copied as upstream. Silent.
  - **Assumes:** nothing writes the cached mask without bumping its `_version`; replays run in
    capture order (checked at run time: error, not wrong values).
  - **Check:** config lines 116 and 115. `result_0.txt` line 297 `cuda_graph_modules` with `attn`
    and `moe`; no result file contains `NVFP4_GRAPH_CHAIN_IO:`, so the run-time check never fired.
- **7.4 Evaluation replays the training graphs:** Megatron replays the per-layer TE graphs only in
  training mode. With dropout 0, no forced load balancing and no expert bias, the training-mode
  forward computes the same function, so evaluation replays each layer's captured forward graph
  under `no_grad`. Evaluation data, frequency (every 512 steps), sample count and metric are
  unchanged. `NVFP4_EVAL_GRAPH_CHECK=<n>` compares graph and eager outputs for the first n layer
  calls (not enabled in the scored runs). Job 22923 ran this package's config with the check on
  this image (544 steps, one evaluation): 24 of 24 layer calls bitwise (9.2).
  - **Gate:** eval mode, graphs present, grad off, TE graph implementation, not capturing
    (`pretrain.py` lines 164–170).
  - **Falls back:** the eager evaluation forward (line 190), silently.
  - **Assumes:** parameters are gathered when evaluation starts (the replay skips the all-gather
    hooks, lines 154–155); evaluation micro-batches have the captured shape (3 × 8192 tokens).
  - **Check:** config line 106. No log line (it prints only in check mode); the result files show
    `eval_accuracy` every 512 steps.

## 8. V7 changes

All V7 code is in `transformer_engine_firmus_v7.patch` (TE `pytorch/ops/fused/` modules and
`pytorch/csrc/quantizer.cpp`), `pretrain.py` (install hooks) and the `Dockerfile` (TE build
options, CUDA extensions in `/opt/nvfp4_ext`). G6c adds `firmus_kernels/` (8.4, 8.6 with 10.1 and
10.3).

### 8.1 FP8 dot-product attention

The full-attention layers (layers 1, 3, …, 23) run TE `DotProductAttention` through cuDNN FP8
attention with current scaling (`Float8CurrentScaling`, HYBRID: E4M3 forward, E5M2 backward); the
attention output is kept in BF16 (`NVTE_DPA_FP8CS_O_in_F16=1`). The sliding-window layers keep the
BF16 Triton kernel (5.1). FP8 is a pre-approved format. The logged attention precision `fp8` is the
value of `MLPERF_ATTN_PRECISION` (config line 53; `pretrain.py` line 1514), not a measurement.
cuDNN FP8 attention is not run-to-run deterministic.

- **Gate:** `FP8_DPA` (config line 35) → `conf/gpt_oss.yaml` line 125 → Megatron → TE; applies to
  the layers 5.1 does not take.
- **Falls back:** the backend choice is TE's (upstream code); no Firmus log line.
- **Assumes:** TE `27486e03` applies `Float8CurrentScaling` to the attention under the NVFP4 and
  MXFP8 autocasts; not checked by Firmus code.
- **Check:** config lines 35 and 158–161. `result_0.txt` line 305
  `fp8_dot_product_attention: True`; line 2276 (8.6) shows `Float8CurrentScalingQuantizer` on the
  packed attention inputs.

### 8.2 NVFP4 expert weight all-gather

Every rank used to all-gather the BF16 expert weights and cast all of them to NVFP4. With
`NVFP4_FP4_AG=1` each rank casts only the part of each expert weight in its own
distributed-optimizer shard and the ranks all-gather the NVFP4 row-wise and column-wise data and
scales. The expert weights get their own DDP buckets, laid out so that every shard boundary inside
a weight falls on a 16-row boundary. NVFP4 weights use 16×16 blocks with the per-tensor amax of
6.2, so a 16-row slice cast with the full-tensor amax reproduces the full 2D cast bit for bit
(row data, scales and the slice's columns). Bitwise; `NVFP4_FP4_AG_CHECK=<n>` compares the
gathered tensors with a full cast (not enabled in the scored runs; in job 22923 every rank reports
`0 bad 0 diverged`, 9.2).

With 8.2 the NVFP4 expert weights arrive quantized from the gather. 6.1 never runs. 6.3's
side-stream launch is captured into the CUDA graphs and replayed every step; under 8.2 it returns
the gathered NVFP4 tensors (V7 patch lines 461–463). 6.2's slots supply the amax for the shard cast
(in-run counters on the G6 image, job 22329). For a weight cut by a shard boundary, `lookup` calls
`nvfp4_shard_cast.BatchCopy2D` (V7 patch line 1668), which is not defined in the image. With the
submitted FC1/FC2 shapes at DP4 and DP8 no weight is cut (`pairs` is empty), so the call is not
reached; if it were, it would raise `AttributeError`.

- **Gate:** 2-D BF16 expert weights of non-tail layers (V7 patch lines 1124–1138) with
  `numel % unit == 0` and rows a multiple of 16 (lines 1179–1181); bucket size a multiple of
  32 × DP (assert, line 1267); shard boundaries on 16 rows (assert, lines 1282–1284);
  `NVTE_NVFP4_4OVER6` must be `none` (lines 1511–1512).
- **Falls back:** weights failing line 1180 are untagged and use the BF16 gather and the 6.x cast,
  silently; a bucket without amax slots gathers BF16 (lines 1473–1479, 1576–1580); unknown weights
  return `None` from `lookup` (lines 1634–1636).
- **Assumes:** the BF16 rows outside the local shard are not read after an FP4 gather; every rank
  holds the same amax slots (6.2's MAX all-reduce); no weight is cut by a shard boundary (above).
- **Check:** config line 110. `result_0.txt` line 562 `NVFP4_FP4_AG layout: 47 buckets, 1408 FP4
  weights, unit 47104`, line 566 `NVFP4_FP4_AG: DDP has 47 buckets, 22 all-FP4`, line 2297
  `NVFP4_FP4_AG shard cast: ShardCastCUDA`; no result file contains a traceback.

### 8.3 Shard cast in CUDA

The per-shard cast of 8.2 for a whole bucket runs as one CUDA launch (`shard_cast_ext.cu`,
128×128 tiles, 16×16 block amax). Every output byte is equal to the one-launch Triton version
(`nvfp4_shard_cast.py`) and to a TE 2D cast of each piece with the same amax, for DP4 and DP8.
Piece tables that fail the kernel's alignment check fall back to the Triton version.

- **Gate:** `supported` (V7 patch lines 2539–2544): rows % 16, cols % 64, row-data offset % 16,
  column-data offset % 8, row-scale offset % 4, (cols / 16) % 4.
- **Falls back:** to the Triton `ShardCast` (line 1342), same outputs; the log line prints the
  class of the first bucket only.
- **Assumes:** `NVFP4_SHARD_CAST_EXT` names the built `.so` (no fallback, lines 2523–2531); the amax
  slot holds 6.2's full-tensor amax; finite weights (`fmaxf`, lines 2639–2643).
- **Check:** config lines 141–142. `result_0.txt` line 2297 `NVFP4_FP4_AG shard cast:
  ShardCastCUDA`.

### 8.4 Residual-stream gradient sums

- `NVFP4_RESGRAD`: the residual gradient of each bias-dropout-add is added by the Triton RMSNorm
  backward (5.3) of the same tensor while it writes dx, instead of a separate BF16 add. Without
  biases the bias-dropout-add takes the residual-add path (V7 patch lines 2298–2299).
- `NVFP4_RESGRAD_ROUTER`: the router input gradient is summed into the pre-MLP norm backward the
  same way (the two dz are added in BF16 before use, as the separate add did).
- `NVFP4_RESGRAD_FWD=0`: the MLP residual add of the FC2-bias unpermute (3.4) is off. G6c:
  `FIRMUS_RESGRAD_FWD_NOBIAS=1` adds the MLP residual in the unpermute without a bias and hands
  the result to this item's bias-dropout-add (10.3).

dx and dgamma are bitwise equal to 5.3's Triton backward plus separate adds (the kernel uses an
explicit FMA so the compiler contracts the same terms as the plain backward), also through the TE
LayerNormLinear and RMSNorm modules (`k8_4_resgrad_module`, job 22919, 0 of 8 cases differ; Section
9). Against TE's `rmsnorm_bwd` plus separate adds the results carry 5.3's rounding (G6b package
tests: 17–378 dx elements differ, jobs 22412, 22434). A gradient not consumed before the next
forward raises an error.

- **Gate:** `install` raises unless 5.3 is installed and `NVFP4_RMSNORM_BWD_UNROLL=1` (V7 patch
  lines 2276–2279); the router part needs 3.3 with padding (lines 2317–2320). Per call: training
  with grad and a residual that requires grad (lines 2294–2295); a pre-added unpermute output
  (10.3) must reach the add with the same residual, no bias and dropout 0 (lines 2289–2293).
- **Falls back:** to the original bias-dropout-add (line 2300) or 5.3's plain backward (lines
  2098–2099). Silent.
- **Assumes:** each handed-over gradient is consumed by the norm backward of the same tensor; this
  is checked (`check_empty` raises, lines 1994–2003, 2304–2311).
- **Check:** config lines 136, 137, 138 and 139. `result_0.txt` line 219 `NVFP4_RESGRAD installed
  (router 1, fwd 0)` and line 220 `FIRMUS resgrad fwd (no bias) installed (unroll 4)`; no result
  file contains a traceback.

### 8.5 Grouped dy statistics in one pass

For the grouped MoE dy, one CUDA pass computes the pre-RHT row amax and the post-RHT column amax
with the instruction sequence of TE's `GraphSafeGroupHadamardAmaxTmaKernel` (bitwise equal amax,
hence identical quantized bytes) and, from the same tile, the per-expert dbias (FC1) or scaled
dbias and dscales (FC2) when the experts have biases, accumulated in FP32. G6c: the experts have no
biases. FC1 runs the pass with `dbias=fc1_op.has_bias`, i.e. amax only (V7 patch lines 728–735).
FC2 takes the pass because it has no bias (`not output_fc2_dbias`, lines 644, 656) with
`dbias=scale_bias`; `scale_bias` is set by upstream TE code that is not in the package, and that it
is false without an FC2 bias is inferred (its true branch stacks the FC2 bias tensors, line 664),
not verified (open point). `NVFP4_AMAX_DBIAS_COLS=448` sets the columns per CTA of the unscaled pass
(line 834; default 1024, capped by the width, line 845): a launch shape, which does not change a
max.

- **Gate:** `_dy_stats_for` (V7 patch lines 361–370): switch on, more than one group, NVFP4
  quantizer with RHT and post-RHT amax, BF16 dy. CUDA path: rows and hidden size multiples of 64;
  with scales (FC2 bias) also `NVFP4_DY_STATS_CUDA_FC2=1`, FP32 contiguous scales, BF16 contiguous
  bias (lines 824–831).
- **Falls back:** to the Triton `nvfp4_dy_stats` (lines 377–380), or to TE's group quantize (FC1:
  line 759; FC2: the upstream branch after lines 676–677). Silent.
- **Assumes:** every expert's row offset is a multiple of 64 (not checked; the MoE layout aligns
  experts to 256 rows); `NVFP4_AMAX_DBIAS_EXT` is set (no fallback, line 811).
- **Check:** config lines 102–105, 84 and 85. No log line.

### 8.6 Packed Q, K, V for FP8 attention

G6c: the RoPE is computed by NVIDIA's QK-norm + RoPE kernel (10.1). Its packed form
(`firmus_kernels/qknorm_rope_packed.py`) writes Q, K and V back to back into one buffer and each
program's max |output| (lines 102–105, 113–116); TE's `nvfp4_qkv_packed.combine_and_quantize`
reduces the parts in one program and computes the FP8 scale as TE's `compute_scale_from_amax`. The
FP8 attention then casts the packed buffer with that scale instead of concatenating Q, K, V and
running a separate amax pass. FP8 data and `scale_inv` equal the unpacked path given the same amax
(`k8_6_qkv_packed`), and the packed kernel's amax equals `torch.max(|·|)` of its outputs
(`k10_1_qknorm_rope_packed`; Section 9).

- **Gate:** installed only when 10.1 is active and `NVFP4_QKV_PACKED=1` (`pretrain.py` lines
  64–69). `sbhd_sbhd_sbhd` or `bshd_bshd_bshd` without paged KV, an FP8 or FP8 current-scaling
  quantizer (V7 patch lines 1900–1907), Q, K, V contiguous and consecutive in one storage (lines
  1859–1871); amax shortcut only for the storage the RoPE kernel just wrote, at offset 0, with a
  current-scaling quantizer without amax reduction (lines 1913–1920).
- **Falls back:** to TE's `combine_and_quantize` (lines 1908–1911) or the normal quantizer on the
  packed view (line 1924); counted, printed once at the 64th call.
- **Assumes:** no other packed RoPE output with the same storage pointer is written between the
  RoPE forward and this quantize; head dim 64 (hard-coded in the packed kernel,
  `firmus_kernels/qknorm_rope_packed.py` lines 23, 49).
- **Check:** config lines 132–133. `result_0.txt` line 217 `NVFP4_QKV_PACKED installed (amax 1)`,
  line 218 `FIRMUS qknorm_rope packed installed (amax 1)` and line 2276 `NVFP4_QKV_PACKED stats
  {'packed': 64, 'fallback': 0, 'amax': 64} ...`.

### 8.7 Fused RHT cast for 2880-wide dense tensors

TE takes its fused single-tensor RHT + NVFP4 cast kernel (`row_col_rht_gemm`) only when the last
dimension is a multiple of 128; the kernel itself handles multiples of 64. With
`NVTE_NVFP4_RHT_FUSION_COLS64=1` the eligibility check accepts multiples of 64 for the
single-tensor kernel (the grouped kernel keeps 128), so the 2880-wide dense tensors (QKV input,
projection dy) use it. Same function.

Measured on the G6b image (same patches and build options; image verify job 22541 and kernel tests
22410, 22444): without fast math every output byte and both amaxes equal the unfused path on two
seeds and on the finite test inputs, except the column-wise scales of an input with every third
16-element block zero (89,120 elements); an input with Inf and NaN differs in 79,150,647 elements,
also with fast math. With `NVTE_USE_FAST_MATH=1` (used; Section 2) the fused and unfused paths
differ in 0.8–1.4% of the data bytes and in 0.7–1.8% of the column-wise scales; the row-wise scales
and both amaxes are equal. The 4096-wide tensors already take the fused kernel upstream, so the
switch does not change them. The verify output also has a bare `FAIL` line, printed by its test
script (`test_rht_c64.py`, not in the package) after the timing section; which of its comparisons
that line reports cannot be shown here. The verify job's two checks for this item are both OK. On
this image `k8_7_rht_cols64` (job 22898) has 54 of 136 cases with a difference (Section 9).

- **Gate:** single-tensor call, rows % 64 == 0, cols % 64 == 0 with the switch, SM arch 100–110
  (V7 patch lines 8–15).
- **Falls back:** to the unfused Hadamard + quantize kernels, silently.
- **Assumes:** `row_col_rht_gemm` is correct for column counts that are multiples of 64 but not of
  128; TE's own check excludes them, and the package relies on the tests above.
- **Check:** config lines 169 and 170. No log line.

### 8.8 Philox4x32-7 for stochastic rounding

TE draws the stochastic-rounding bits of the NVFP4 casts from Philox4x32 with
`NVTE_BUILD_NUM_PHILOX_ROUNDS` rounds (default 10); the image builds TE with 7. Scales, amaxes and
all round-to-nearest outputs are unchanged; the stochastically rounded data bytes come from a
different random stream, so training is not bit-identical to a 10-round build. Not equivalent.
The rounding rule is unchanged: each value rounds up with probability equal to its fractional
position between the two neighbouring FP4 values. Salmon et al. (SC 2011) report that
Philox4x32-7 passes the BigCrush test suite. The test `k8_8_sr_stats` measures the rounding
statistics of the build (G6b image, job 22235; rerun in job 22445): over 15,728,640 elements per
fraction f (9 values from 1/64 to 63/64), the share rounded up deviates from f by at most 1.3e-4
(at most 1.2 binomial standard deviations), and the largest |signed mean error| is 6.4e-5. On this
image the same test ran as job 22899 (Section 9). No test compares with a 10-round build. Open
point: stochastic rounding is a listed numerically safe technique and the rules do not name a
random number generator or round count; whether a reduced round count needs a statement to, or a
ruling from, the MLPerf Training working group is not decided.

- **Gate:** compile time; applies to every TE NVFP4 cast with stochastic rounding.
- **Falls back:** none; the image has no run-time way back to 10 rounds.
- **Assumes:** 7 rounds give random bits of sufficient quality for stochastic rounding; not
  checked at run time (evidence: the statistics above and the cited paper).
- **Check:** `Dockerfile` line 117 (step 4, lines 106–120); all 10 scored runs were made with this
  build.

## 9. Kernel tests

Each test runs the image's own code (`/workspace`, the installed Transformer Engine and
extensions) inside this package's image, on GB300 nodes, with the package's config sourced
(`config_GB300_2x4x3xtp1pp1cp1ep1_nvfp4_v7.sh`, sha256 `58153aac…`, the file in the image). A case
compares the outputs of the code used in training with the test's reference path (the stock or
unfused computation the item replaces, or an FP64 reference where the item says so), output by
output and byte by byte, and records the number of differing elements; nothing is graded against
a threshold. Inputs are seeded draws, shape neighbours of the training shapes (±1 and ± the
alignment), and tensors captured at iterations 4 and 303 of a training run of this package on this
image (job 22873; layers 0, 1, 11, 12, 21, 22, 23, forward and backward). Each test runs twice in
fresh processes (run a, run b) with the same seed; the last column compares every output's digest
between the two runs.

"Cases with a difference" counts cases in which at least one output differs from the reference.
It includes the cases built to differ: fixtures (an input the code must reject or that changes
the result by construction) and controls (a change the quantization does see). "Recorded errors"
are exceptions caught and recorded by a case, for example a shape the kernel rejects (a leading
dimension not divisible by 8) or a fixture that must raise. Per-case listings are in the records.
Some tests record their main comparisons as separate rows (the main 1.1 p, m, v and remainder rows
of `k1_optimizer`, the main 5.5 and 5.6 comparisons, the 6.2 comparisons on generated inputs);
those rows are not in the case and digest counts.

Image: `firmus+gpt-oss-20b-nvfp4-g6c-v2+v61.sqsh`, sqsh sha256
`fd815c589765300f2f559f6635c791b3a16f59aed319524abd64f789399aea7a` (image ID
`sha256:d69e3e51f4a18273d977c9c0767d7dae680f73c0bde166995ec0101120076869`).

**Model the test builds.** Tests marked "G6c model" build their modules as this package's model:
no linear bias, `te.ops.ScaledSwiGLU(glu_interleave_size=32)` between the expert GEMMs (what
Megatron's `_make_fused_ops` builds for `activation_func=F.silu` without clamp), vanilla softmax
without sinks, window (128, 0). For `k4_3_moe_dispatch` only the bias is switched off; its
activation is Megatron's default. The other tests do not build a bias, sink or GLU, or test a
kernel independent of them (optimizer, casts, quantization, routing).

| Items | Test | Model | Job | Runs | rc | Cases per run | Cases with a difference from the reference, per run | Recorded errors, per run | Run a vs run b output digests: equal / differ |
|---|---|---|---|---|---|---|---|---|---|
| 1.1, 1.2, 1.3 | `k1_optimizer` | — | 22878 | a, b | a:0, b:0 | 133 / 133 | 67 / 67 | 0 / 0 | 222 / 0 |
| 1.1 | `k1_state_steps` | — | 22827 | a, b | a:0, b:0 | 9 / 9 | 0 / 0 | 0 / 0 | 24 / 0 |
| 1.4, 1.5 | `k1_4_k1_5_optimizer` | — | 22828 | a, b | a:0, b:0 | 13 / 13 | 5 / 5 | 0 / 0 | 32 / 0 |
| 2.2 | `k2_2_tail_mxfp8` | G6c | 22918 | a, b | a:0, b:0 | 97 / 97 | 0 / 0 | 2 / 2 | 12 / 0 |
| 2.3 | `k2_3_lmhead` | — | 22894 | a, b | a:0, b:0 | 7 / 7 | 0 / 0 | 4 / 4 | 15 / 0 |
| 3.1 | `k3_1_padding` | G6c | 22911, 22912 | a, b | a:0, b:0 | 23 / 23 | 23 / 23 | 0 / 0 | 1,376 / 165 |
| 3.3 | `k3_3_permute_pad` | — | 22895 | a, b | a:0, b:0 | 33 / 33 | 1 / 1 | 0 / 0 | 150 / 0 |
| 4.1, 4.2, 5.9 | `k4_k5_9_grouped_mlp` | G6c | 22907, 22908 | a, b | a:0, b:0 | 140 / 140 | 106 / 106 | 0 / 0 | 7,060 / 374 |
| 4.2 | `k4_2_cross_layer` | G6c | 22913 | a, b | a:0, b:0 | 8 / 8 | 8 / 8 | 2 / 2 | 1,054 / 18 |
| 4.2 | `k4_2_replay_vs_probe` | — | 22891 | a, b | a:0, b:0 | 14 / 14 | 0 / 0 | 0 / 0 | 14 / 0 |
| 4.2 | `k4_2_bwd_graph_vs_eager` | — | 22887 | a, b | a:0, b:0 | 18 / 18 | 16 / 16 | 0 / 0 | 216 / 0 |
| 7.3, 4.2 | `k7_3_k4_2_graph` | G6c | 22920, 22921 | a, b | a:0, b:0 | 35 / 35 | 24 / 24 | 2 / 2 | 845 / 15 |
| 4.3 | `k4_3_moe_dispatch` | G6c (bias) | 22915, 22922 | a, b | a:0, b:0 | 81 / 81 | 79 / 78 | 0 / 0 | 5,085 / 342 |
| 4.3 | `k4_3_pad_routing_map` | — | 22892 | a, b | a:0, b:0 | buffer-size facts, no reference comparison | — | 4 / 4 | — |
| 4.3 | `k4_3_router_bwd` | — | 22889 | a, b | a:0, b:0 | 254 / 254 | 60 / 60 | 0 / 0 | 268 / 0 |
| 5.3, 8.4 | `k5_3_k8_4_rmsnorm_bwd` | — | 22884 | a, b | a:0, b:0 | 67 / 67 | 39 / 39 | 0 / 0 | 125 / 0 |
| 5.5, 5.6 | `k5_5_k5_6_glu_ce` | (below) | 22927, 22928 | a, b | a:0, b:0 | 53 / 53 | 3 / 3 | 2 / 2 | 29 / 0 |
| 5.7, 5.8 | `k5_7_k5_8_variants` | G6c | 22909, 22910 | a, b | a:0, b:0 | 64 / 64 | 59 / 60 | 0 / 0 | 422 / 90 |
| 5.7 | `k5_7_rht_memcheck` | — | 22853 | a | a:0 | one quantize under compute-sanitizer (below) | — | — | — |
| 5.8 | `k5_8_fc1_dbias_fp64` | G6c | 22914 | a | a:0 | 0 (no FC1 bias, so no FC1 bias gradient to compare) | — | 0 | — |
| 6 | `k6_2_shard_amax` | — | 22896 | a, b | a:0, b:0 | 112 / 112 | 1 / 1 | 0 / 0 | 110 / 0 |
| 8.1 | `k8_1_fp8_dpa` | G6c | 22917 | a, b | a:0, b:0 | 56 / 56 | 39 / 39 | 0 / 0 | 192 / 0 |
| 8.2 | `k8_2_slice_cast` | — | 22897 | a, b | a:0, b:0 | 7 / 7 | 1 / 1 | 0 / 0 | 20 / 0 |
| 8.3 | `k8_3_shard_cast` | — | 22850 | a, b | a:0, b:0 | 29 / 29 | 2 / 2 | 0 / 0 | 100 / 0 |
| 8.4 | `k8_4_resgrad_module` | G6c | 22919 | a, b | a:0, b:0 | 8 / 8 | 0 / 0 | 0 / 0 | 64 / 0 |
| 8.5 | `k8_5_dy_stats` | — | 22880 | a, b | a:0, b:0 | 60 / 60 | 2 / 2 | 0 / 0 | 200 / 0 |
| 8.6 | `k8_6_qkv_packed` | — | 22882 | a, b | a:0, b:0 | 35 / 35 | 1 / 1 | 0 / 0 | 302 / 0 |
| 8.7 | `k8_7_rht_cols64` | — | 22898 | a, b | a:0, b:0 | 136 / 136 | 54 / 54 | 0 / 0 | 816 / 0 |
| 8.8 | `k8_8_sr_stats` | — | 22899 | a, b | a:0, b:0 | 121 / 121 | 44 / 44 | 0 / 0 | 59 / 42 |
| 1.4, 7 | `k1_4_k7_dp` (DP8, 2 nodes) | — | 22902 | a, b | a:0, b:0 (8 ranks each) | 22 / 22 | 7 / 7 | 0 / 0 | 3,744 / 2,390 |
| 10.1 | `k10_1_qknorm_rope_packed` | G6c | 22900 | a, b | a:0, b:0 | 126 / 126 | 2 / 2 | 0 / 0 | 413 / 0 |
| 10.2 | `k10_2_swa_vanilla` | G6c | 22903 | a, b | a:0, b:0 | 49 / 49, plus 1 / 1 in the GQA 64/1 subprocess (the 64/64 subprocess records no case; 9.1) | 39 / 39, plus 1 / 1 | 0 / 0 | 182 / 0 |
| 10.3 | `k10_3_resgrad_fwd_nobias` | G6c | 22856 | a, b | a:0, b:0 | 23 / 23 | 0 / 0 | 0 / 0 | 21 / 0 |

`k4_3_moe_dispatch` run b first ran as job 22916 (rc 1: `EADDRINUSE` on the process-group port,
run a was on the same node at the same time) and was rerun alone as 22922.

`k10_1_qknorm_rope_packed`: the 2 cases with a difference are a fixture and `specials:amax`
(NaN inputs: the reference amax is NaN, the kernel's is not). Q, K, V, dX and dγ are bit-identical
to NVIDIA's kernel in every other case.

**5.5, 5.6.** 5.6 (cross-entropy in place, `NVFP4_CE_BF16_INPLACE=1`) is on this package's path;
its real cases use logits from the captured output-layer input and weight. 5.5 drives
`ScaledClampedQGeGLU` (limit 7.0, α 1.702), which this model does not build; its real cases use
the captured FC1 weights (no bias). The test records `NVFP4_TRITON_INTERLEAVED_GLU` as not set in
this config. The tail layers' `ScaledSwiGLU` cannot reach the interleaved Triton GLU:
`_use_triton_interleaved_glu` accepts only `ScaledClampedQGeGLU` (TE patch line 1389; 5.5).

**Items whose tests were not run on this image:** 3.4 (`k3_4_unpermute_bias`), 5.2
(`k5_2_qkv_rope`), 5.4 (`k5_4_bias_residual`), all three off in this configuration (summary);
5.1 with sinks (`k5_1_swa`), replaced by 10.2 (which compares its sink case with the TE-patch
kernel); `k1_4_k7_dp` on one node (DP4).

### 9.1 Sanitizer runs

All on this image, run a only, `compute-sanitizer` around the test process. Counts are the
`ERROR SUMMARY` / `RACECHECK SUMMARY` lines.

| Test | Tool | Mode | Job | rc | Result |
|---|---|---|---|---|---|
| `k5_7_rht_memcheck` | memcheck | one `tex.group_quantize` | 22853 | 0 | 168 errors: 152 `Invalid __shared__ write of size 4 bytes` in TE's `group_row_col_rht_gemm_device_graph_safe`, each with the note `Cluster needs to have at least 2 blocks`; 7 `CUDA API Error` (`cudaVersion` argument 13041 exceeds the driver version 13040); 7 `CUDA_ERROR_INVALID_VALUE` on `cuGetProcAddress_v2`; 2 `cudaErrorLaunchFailure` |
| `k10_1_qknorm_rope_packed` | memcheck | full | 22863 | 0 | 14 errors: 7 `CUDA API Error` (`cudaVersion` 13041 exceeds driver 13040), 7 `CUDA_ERROR_INVALID_VALUE` on `cuGetProcAddress_v2`; no memory access errors |
| `k10_1_qknorm_rope_packed` | initcheck | `KT_SMALL=1` | 22866 | 0 | 0 errors |
| `k10_1_qknorm_rope_packed` | racecheck | `KT_SMALL=1` | 22867 | 0 | 0 hazards |
| `k10_2_swa_vanilla` | memcheck | training GQA only (`KT_GQA=1`) | 22864 | 0 | 28 errors: 14 + 14 of the same two API messages; no memory access errors |
| `k10_2_swa_vanilla` | memcheck | GQA 64/64 only (`KT_GQA_ONLY=64`) | 22865 | 1 | 54,735 errors; the 2,000 printed are 1,986 `Invalid __global__ read of size 16 bytes` at `_swa_fwd_packed` (`firmus_kernels/swa_attention.py:261`) and 7 + 7 of the two API messages. The shape raises `misaligned address` without the sanitizer. Training uses 64 heads in 8 KV groups, not 64/64 (10.2) |
| `k10_2_swa_vanilla` | initcheck | `KT_SMALL=1` | 22868 | 0 | 6,815,744 errors `Uninitialized __global__ memory read`; the 200 printed are all in cuDNN `compute_dot_do_o_specialized<true, 64>` (the TE `DotProductAttention` reference path of the test); the other 6,815,544 are not printed |
| `k10_2_swa_vanilla` | racecheck | `KT_SMALL=1` | 22869 | 0 | 0 hazards |
| `k10_3_resgrad_fwd_nobias` | memcheck | full | 22859 | 0 | 14 errors: 7 + 7 of the two API messages; no memory access errors |
| `k10_3_resgrad_fwd_nobias` | initcheck | `KT_SMALL=1` | 22870 | 0 | 0 errors |
| `k10_3_resgrad_fwd_nobias` | racecheck | `KT_SMALL=1` | 22874 | 0 | 0 hazards |

The 152 `Invalid __shared__ write` errors and the 2 launch failures of `k5_7_rht_memcheck` also
occur on NVIDIA's image `nvdlfwea+mlperftv61+gpt_oss_20b-arm+20260929.sqsh` (TE 2.19.0+5e52befd,
no Firmus code), in TE's own `group_row_col_rht_gemm_device_graph_safe` kernel (job 22511, run for
the G6b package).

Line 261 of the package's `firmus_kernels/swa_attention.py` is in the `HAS_SINK` branch of
`_swa_fwd_packed`; whether the failing 64/64 case had a sink, and how Triton attributes the line,
was not checked (open point).

### 9.2 In-run checks

Job 22923 (this image and config, 544 steps, evaluation at step 512, `run_stop` status `aborted`
at `MAX_STEPS`), with the package's check switches `NVFP4_EVAL_GRAPH_CHECK=48`,
`NVFP4_FP4_AG_CHECK=8`, `NVFP4_CLIP_IN_ADAM_CHECK=16`, `NVFP4_SHARD_WEIGHT_AMAX_CHECK=16`,
`NVFP4_EARLY_L2NORM_CHECK=16`:

- `NVFP4_EVAL_GRAPH_CHECK` (7.4): 24 of 24 layers `bitwise True`, `max_abs 0`, `rel_l2 0`
  (rank 0, first evaluation).
- `NVFP4_FP4_AG_CHECK` (8.2): `0 bad 0 diverged` on every rank; layout 47 buckets, 22 all-FP4,
  1,408 FP4 weights.
- `NVFP4_CLIP_IN_ADAM_CHECK` (1.1): 128 lines (8 ranks × 16), all `leftover_before_finish=0`,
  `fallback: 0`.
- `NVFP4_SHARD_WEIGHT_AMAX_CHECK` (6.2): 16 checks per rank. Rank 0: 0/1408 mismatched at all 16.
  Ranks 1–7: 0/1408 at the first 11, then 1, 2, 3, 4, 5 /1408. Under `NVFP4_FP4_AG=1` the check
  reads BF16 rows that are stale outside the rank's shard (6.2), so it does not test 6.2; why rank 0
  shows none is not explained. The kernel comparison of 6.2 is `k6_2_shard_amax` above.
- `NVFP4_EARLY_L2NORM_CHECK` (1.2): read only with `NVFP4_EARLY_L2NORM=1` (`pretrain.py` lines 677,
  718), which is off, so it had no effect.

### 9.3 Records available on request

- the test scripts, the job scripts and, per test, the output directory: `env.txt` (image path
  and sha256, node, config path), `config.sha256`, `gpu.txt`, `rc.txt`, the per-run logs and the
  per-run JSONL records (every case, output, differing-element count and digest);
- the per-test statistics from which the tables above are taken;
- the capture of job 22873 (iterations 4 and 303) and the log of job 22923;
- the G6b package's test records cited above (jobs 22168, 22220, 22235, 22240, 22322, 22329,
  22355, 22356, 22405, 22410, 22412, 22434, 22444, 22445, 22477, 22511), and the image verify
  outputs of jobs 22541 and 22589 (G6b image).

## 10. G6c: kernels reworked for the reference model

The 20261012 reference model removes the attention and expert biases, the clamped quick-GELU and the
attention sinks, and adds QK RMSNorm. Each kernel that assumed the old model was reworked:

### 10.1 QK RMSNorm + RoPE with packed Q, K, V

NVIDIA's `opt_kernels/qknorm_rope.py` and `qknorm_rope_patch.py` are included unchanged (sha256
equal to the files of NVIDIA's 20261012 recipe in `../nemo_g1_1012/opt_kernels/`) and enabled the
same way (`fused_single_qkv_rope` and `apply_rope_fusion` as in NVIDIA's `pretrain.py`;
`pretrain.py` lines 1137–1139). `firmus_kernels/qknorm_rope_packed.py` replaces only the forward
launch: NVIDIA's kernel body, writing Q, K, V back to back into one buffer and, per program, the
max |output|. TE's `nvfp4_qkv_packed.combine_and_quantize` (8.6) then quantizes Q|K|V for the FP8
attention in place, without the concatenation copy or a separate amax pass. The backward is
NVIDIA's (`qknorm_rope_packed.py` line 147). Against TE's RMSNorm + fused RoPE (the path it
replaces) the result differs by rounding: NVIDIA's start-up self-check, which runs the packed
kernel after the install (line 164), logs max / mean relative differences of 5.85e-03 / 1.47e-03
for q and 6.62e-03 / 1.47e-03 for k, with v equal (`result_0.txt` line 1038; thresholds 2⁻⁶ and
2⁻⁹, `opt_kernels/qknorm_rope_patch.py` lines 135–140). Tests on this image: forward Q, K, V and
backward dX, dγ bit-identical to NVIDIA's kernel; amax equal to `torch.max(|·|)`
(`k10_1_qknorm_rope_packed`, Section 9). `OPT_QKNORM_ROPE_STRICT=1` makes any fallback fatal
(logged per run: `fused=24 fallback=0`).

- **Gate:** q and k norms are TE `RMSNorm` with 64 weights (`opt_kernels/qknorm_rope_patch.py`
  lines 43–51); per call (`_unsupported`, lines 87–101): BF16 contiguous 4-D qkv, split
  `[nq·64, 64, 64]`, power-of-two group and head counts, `sbhd`, no `start_positions`, not
  interleaved, CP 1, FP32 contiguous frequencies of width 64; the self-check passed (lines
  127–164). The packed form is installed only with `NVFP4_QKV_PACKED=1` (`pretrain.py` lines
  64–69).
- **Falls back:** to TE RMSNorm + TE fused RoPE (`_composed`, lines 104–117, 214–216); with
  `OPT_QKNORM_ROPE_STRICT=1` a module that cannot be retyped, a failed self-check or an unsupported
  call raises instead (lines 66–67, 161–163, 212–213).
- **Assumes:** head dim 64 (hard-coded, `firmus_kernels/qknorm_rope_packed.py` lines 23, 49);
  NVIDIA's backward is valid for the packed forward's saved tensors (the same five tensors, eps and
  zero-centring flag as NVIDIA's forward, lines 142–144; `opt_kernels/qknorm_rope.py` lines
  425–427); the eps is `layernorm_epsilon` (1e-6, `opt_kernels/qknorm_rope_patch.py` line 81).
- **Check:** config lines 39, 132–135. `result_0.txt` line 216 `[opt qknorm_rope] installed`, line
  218 `FIRMUS qknorm_rope packed installed (amax 1)`, line 1038 `... selfcheck ... ok`, line 1065
  `[opt qknorm_rope] first forward: retyped=48 fused=24 fallback=0`, line 340
  `qk_layernorm: True`.

### 10.2 Sliding-window attention without sinks

`firmus_kernels/swa_attention.py` is the TE-patch kernel (5.1) with a compile-time `HAS_SINK`.
Without a sink the online softmax starts from m = −∞, l = 0 (fully masked key blocks are guarded,
lines 80–84 and 279–283) and no sink gradient is produced. With a sink it is bit-identical to the
TE-patch kernel. The vanilla forward and backward match a PyTorch reference (window (128, 0),
causal) within BF16 rounding (`k10_2_swa_vanilla`, Section 9). `NVFP4_SWA_PDKDV=4,16,64,4,3` is the
dK/dV tile for this shape (B = 3; default `2,32,64,4,2`, line 33), chosen by a launch-configuration
sweep; the tile sets the order in which dK and dV are accumulated over query rows (lines 329–349),
not the function.

The packed kernels have no guard for a KV group with fewer query heads than the packed tile: they
compute `NP = (H // G) // HP` (lines 244, 310) and divide by it (lines 248–250), and `_check`
(lines 421–430) only requires the query heads to be a multiple of the KV heads. With 64 query
heads in 64 KV groups (`k10_2_swa_vanilla` GQA 64/64 case, job 22865) `NP` is 0 and the forward
raises `misaligned address` (9.1). Training uses 64 query heads in 8 KV groups (8 heads per group,
tile 4 heads: `NP` = 2; `result_0.txt` line 1039).

- **Gate:** as 5.1 (lines 556–567); no sink when `softmax_type` is `vanilla` (line 576).
- **Falls back:** to the original TE forward (lines 587–588), silently.
- **Assumes:** heads per KV group a multiple of the packed tile's head count (4 for `PFWD`,
  `PDKDV`, `PDQ` here; not checked, above); every query row has at least one key in its window
  (causal, so key i itself), so l > 0.
- **Check:** config lines 144–145. `result_0.txt` line 1039 (`window=(128, 0)`, `'DKDV': (4, 16,
  64, 4, 3)`), line 346 `window_size: (128, 0)`, line 756 `softmax_type='vanilla'`.

### 10.3 MLP residual added in the unpermute without FC2 bias

`NVFP4_RESGRAD_FWD` required the deferred FC2 bias (3.4). `firmus_kernels/resgrad_fwd_nobias.py`
sums the routed rows as TE's `_unpermute_kernel` (FP32, routing order, one BF16 rounding; lines
36–48) and adds the residual rounded as the PyTorch BF16 add (lines 50–51); the result is handed to
8.4's bias-dropout-add through its `_preadded` map (line 98), which routes the residual gradient
as before. Bit-identical to TE unpermute followed by the add (tested with padded 2944-wide rows,
`k10_3_resgrad_fwd_nobias`, Section 9). That TE's `_unpermute_kernel` sums in the same order is
the test's result; the upstream kernel is not quoted here.

- **Gate:** installed inside 8.4's install block (`pretrain.py` lines 108–119); `install` raises
  unless `NVFP4_RESGRAD_FWD=0`, `NVFP4_DEFER_FC2_BIAS` is not 1 and `NVFP4_UNPERMUTE_BIAS_UNROLL` >
  0 (`resgrad_fwd_nobias.py` lines 72–77). Per call (lines 83–92): a residual recorded by the MLP
  sublayer in training with grad (lines 105–106), no merging or permuted probs, no pad offsets, no
  pending expert bias, same dtype, `numel` equal to tokens × hidden, contiguous residual, unit
  column stride.
- **Falls back:** to the original unpermute (lines 93–95), silently; the `fused` counter (line 99)
  is not printed.
- **Assumes:** the output reaches the MLP bias-dropout-add with the same residual, no bias and
  dropout 0 (checked: V7 patch lines 2289–2293 raise); an unconsumed pre-added output raises at
  the next forward (V7 patch lines 1996–1998).
- **Check:** config lines 137–138 (with 149–150 for the unroll and warps). `result_0.txt` line 219
  `NVFP4_RESGRAD installed (router 1, fwd 0)` and line 220 `FIRMUS resgrad fwd (no bias) installed
  (unroll 4)`.

### 10.4 Kernels checked, no change needed

- dGLU (5.7): without an FC1 bias no dbias is computed. The flag `generate_dbias` is set from
  `fc1_op.has_bias` by upstream TE code that is not in the package; the patch only forces it off
  for 5.8 (TE patch line 2235).
- Grouped dy statistics (8.5): without bias the amax-only kernel variant runs (FC1 verified from
  the patch; FC2 see 8.5).
- Hadamard GLU fusion: the SwiGLU path runs the fused kernel (22 calls per step, the NVFP4 layers;
  development measurement, record on request).
- RMSNorm backward (5.3, 8.4): 49 calls per step (2 per layer + final norm; development
  measurement, record on request); the QK norms are inside 10.1, not in this path.
- Evaluation graphs (7.4): with `NVFP4_EVAL_GRAPH_CHECK` on, graph and eager evaluation are
  bit-identical for all 24 layers (job 22923, 9.2).
