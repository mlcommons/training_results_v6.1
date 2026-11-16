# LoRA NVFP4 kernels (`lora_kernels/`) and TE patches (`te_patches/`)

The shipped VR config turns on a set of optional kernel patches for the NVFP4 LoRA recipe.
Each `LORA_PATCH_*` switch enables one patch; all are off by default, and
`callback_nvfp4.py` applies the enabled ones before the model is built. The Dockerfile builds
the CUDA extensions once (`lora_kernels/build.py`) and applies `te_patches/` to the installed
Transformer Engine (TE). Every patch checks the shapes, dtypes and quantizer settings it
supports and otherwise falls back to the stock path.

| switch | replaces | what it does |
|---|---|---|
| `LORA_PATCH_FAST_ROPE` | TE fused QKV RoPE | same per-element math with vectorised memory access |
| `LORA_PATCH_FAST_KERNELS` | TE NVFP4 activation quantize, RMSNorm output quantize, SwiGLU forward/backward | quantize without a separate amax pass or memset, scales written in the GEMM layout; SwiGLU fused with its amax and quantize |
| `LORA_SWIGLU_SR_TWO_PASS` | SwiGLU backward + quantize | recomputes the activation gradient in two passes (amax, then quantize) so the bf16 intermediate is never written |
| `LORA_PATCH_FAST_DROPOUT` | TE dropout kernels | faster dropout and fused NVFP4 dequantize + dropout; installed, but not reached in this configuration |
| `LORA_PATCH_FP8_ROPE` | RoPE followed by the attention's FP8 cast of q/k/v | one kernel for RoPE and the FP8 cast (forward) and for the FP8 dequantize and RoPE backward |
| `LORA_PATCH_AMAX_PRODUCERS` | the amax pass before an NVFP4 quantize | producer kernels (dequantize, residual-gradient add, RoPE backward) also emit the amax of their output, which the next quantize reuses |
| `LORA_PATCH_FUSED_LORA` | the LoRA branch (dropout, A and B GEMMs, scale, add) | three kernels for the branch forward and backward; the B GEMM stays in cuBLAS and accumulates into the base output |
| `LORA_PATCH_RESNORM` | residual add, RMSNorm and NVFP4 quantize of the next sublayer input | one forward kernel for add, norm statistics and quantize; one backward kernel for the norm backward and the residual gradient add |
| `LORA_PATCH_FP8_O` | bf16 dequantize of the attention output | the attention returns its FP8 output; the projection's NVFP4 quantize and the LoRA branch read the FP8 bytes directly |
| `LORA_PATCH_GEMM_ALPHA` | TE's per-GEMM kernel computing the NVFP4 alpha | the quantize kernels write the alpha of the following GEMM, which runs without the extra kernel |
| `NVTE_LORA_PATCH_BATCH_DPA_AMAX` (`te_patches/batch_dpa_amax.patch`) | TE's per-layer delayed-scaling amax reduce and scale update | batches the reductions of one pass into one call; each scale is still updated before it is next read |

The other files in `lora_kernels/` are the extension loaders (`*_ext.py`, `swiglu_fused.py`),
the shared device math headers (`nvfp4q_math.cuh`, `swiglu_math.cuh`), the sub-modules of
the composite switches (`nvfp4q_*.py`, `swiglu_*.py`, `amax_channel.py`, `dequant_patch.py`,
`fork_patch.py`, `rope_patch.py`, `fast_quant_ap.py`, `nvte_gemm.py`) and `build.py`.

## Numerical equivalence

All patches except two are designed to reproduce the arithmetic of the stock TE path bit for
bit: same amax sources, same scale formulas, same rounding points, and for stochastic
rounding the same random stream. This holds for the TE build in the submitted container
(TE v2.20.0 as pinned in the Dockerfile), whose math the kernels copy.

The two exceptions change the order of floating-point accumulation, not the algorithm:

- `LORA_PATCH_FUSED_LORA` sums the LoRA dot products in a different fp32 order and applies
  the dropout and NVFP4 scales once in the reduction instead of per element.
- `LORA_PATCH_RESNORM` computes the RMSNorm statistics in a different reduction order and
  adds the two gradients of the norm output in fp32 before a single bf16 rounding.

One further deviation from the reference, disclosed during review: the fused LoRA branch
uses TE's dropout mask convention, which drops with probability 25/256 (9.77%) and rescales
by 1/0.9, instead of the exact 0.1 of the reference `nn.Dropout`. With `LORA_PATCH_FUSED_LORA=1`
this applies to every training step of the submitted runs, in both the NVFP4 and the FP8
phase; the exact-probability dropout shipped in `lora_exact_dropout.py` is bypassed on that
path. Dropout is off in evaluation.
