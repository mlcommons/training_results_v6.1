# Fused kernels (`opt_kernels/`)

With `OPT_STACK=rubin` (set by the shipped VR config) `pretrain.py` installs seven
optional modules that replace stock Transformer Engine / Megatron paths on NVIDIA VR
with fused equivalents. Each one computes the same math as the stock path and differs
only in how the work is fused or scheduled: same inputs, same amax sources, same
scale formulas, same rounding modes, nothing dropped or approximated.

| module | replaces | what it fuses |
|---|---|---|
| `qkv_rope_fp8amax_gemm.py` | QKV GEMM, fused RoPE, FP8 cast of q/k/v | one NVFP4 GEMM whose epilogue dequantizes, rounds to bf16, applies RoPE and records the amax for the FP8 attention input |
| `weight_quant_once_per_step.py` | per-forward NVFP4 weight quantization | quantizes each weight once per optimizer step; the amax is taken during the master-to-model parameter copy |
| `attn_fwd_fp8_amax.py` | TE FP8 fused-attention forward | same cuDNN FP8 attention graph; additionally records the output amax so the projection cast does not re-read the tensor |
| `fc1_swiglu_epilogue.py` | fc1 GEMM and SwiGLU | one NVFP4 GEMM with SwiGLU and both quantization amaxes in the epilogue |
| `rht_cast_nvfp4.py` | TE NVFP4 cast with random Hadamard transform | row-wise and column-wise cast in a single pass, same arithmetic as the TE cast |
| `lm_head_ce_fusion.py` | output-layer GEMM and TE cross-entropy | one autograd function; the logits gradient is written in place of the logits |
| `skip_grad_buffer_zero.py` | per-step gradient-buffer zeroing | every gradient slice is written exactly once per step, so the zeroing is dropped (requires one micro-batch) |

Every module checks the shapes and quantizer settings it supports and falls back to the
stock path otherwise. Each one can be turned off with its `OPT_<NAME>_DISABLE=1`
variable; with `OPT_STACK` unset none of them is installed and the stock path runs.

The remaining files in this directory (`dp_engine.py`, `fused_adam*.py`,
`fp8_attn_bwd*.py`, `rope_fp8_qkv*.py`, `norm_bwd_add_patch.py`, `rmsnorm_bwd_add.py`,
`nvfp4_producer_quant*`, `quant_bookkeeping*`, `grad_clip_norm.py`) belong to the
`OPT_STACK=blackwell` variant of the same stack. No shipped config selects it, so they
are not used by the submitted runs. `arch.py` and `_philox_state.cpp` are helpers.

## Equivalence with the stock path

The modules were developed against the stock kernels of the Transformer Engine build in
the submitted container: `lm_head_ce_fusion.py --selftest` and
`skip_grad_buffer_zero.py --selftest` assert bit-identical results, and every module has
an `OPT_<NAME>_DEBUG=1` mode that compares its output against the stock kernel at run
time. Where a fused GEMM or cast is not bit-identical to the cuBLAS / TE kernel it
replaces, the difference is floating-point accumulation order, not a change of
algorithm. The submitted runs converge within the reference convergence points (RCP
check).
