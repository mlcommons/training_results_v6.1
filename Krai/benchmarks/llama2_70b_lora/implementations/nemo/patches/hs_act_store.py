# Copyright (c) 2026 Krai Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SwiGLUFunctionTE follows the structure of SwiGLUFunction in Megatron-LM megatron/core/fusions/fused_bias_swiglu.py,
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved (Megatron-LM license, BSD-3-Clause; see NOTICE).

"""Cheaper FP8 storage of the SwiGLU input (Megatron activation_func_fp8_input_store, env FP8_ACT_STORE=1).

Loaded through the recipe's patch hook (HS_MOUNT_CODE=1 HS_PATCH_DIR=<this dir> HS_PATCH_MODULES=hs_act_store),
before NeMo/Megatron are imported. Megatron's fused_bias_swiglu.SwiGLUFunction stores the fc1 output
(896 MiB BF16 per layer) as unscaled torch.float8_e4m3fn and dequantizes it in backward. Measured on the
ROCm 10.1 stack: +4.2% step time at equal ACL (kb rocm101-screen2-results), ~4x the memory-traffic cost,
because (code reading 2026-10-08):
  1. checkpointed layers pay the cast in their no-grad forward (result discarded), again in the recompute
     forward, and the dequant in backward: 2 casts + 1 dequant for no memory benefit;
  2. torch's fp8 -> bf16 copy on ROCm goes through the generic dynamic-cast elementwise kernel (scalar
     1-byte loads), ~2-3x slower than a vectorized dequantize.

Three levels, all off unless the module is loaded:
  default           skip the store when grad is disabled (checkpoint forward) or in TE's recompute phase.
                    Checkpointed layers become exactly the baseline (BF16, transient); no numerics change
                    for the others.
  HS_ACT_STORE_TE=1 also replace torch's cast/dequant by TE's vectorized Float8CurrentScalingQuantizer and
                    Float8Tensor.dequantize (what te.ops.SwiGLU(cache_quantized_input=True) does).
                    Numerics change vs the unscaled cast: per-tensor current scaling, e4m3 fnuz storage
                    (max 240, hardware saturating conversion), i.e. scaled rather than clipped at 448.
  HS_ACT_STORE_FUSED=1
                    fuse the store into SwiGLU with two Triton kernels (hs_swiglu_fp8_triton.py): the forward
                    writes the SwiGLU output and the e4m3fn copy of its input in one pass, the backward
                    dequantizes in registers. Same numerics as the default level (the submission reference):
                    the stored bytes are bit-identical to x.to(torch.float8_e4m3fn), the forward is TE's
                    gated_act formula, the backward TE's dgated_act formula on the e4m3-rounded input.
                    Replaces per layer: cast 0.36 + gated_act 0.49 ms -> 0.65 ms, dequant 1.47 + dswiglu
                    0.70 ms -> 0.64 ms (profile 2026-10-09 vs microbenchmark, one MI300X). Takes precedence
                    over HS_ACT_STORE_TE. Where the store is skipped (no-grad / recompute) the path is
                    unchanged (Megatron's SwiGLUFunction, BF16 kept transiently).
  HS_SWIGLU_BWD_FP8=1 (fix 3a, needs HS_ACT_STORE_FUSED=1)
                    the fused backward writes the fc1-output gradient as e5m2 with fc1's own delayed-scaling
                    grad_output quantizer (hs_dswiglu_e5m2.py) and returns it as a Float8Tensor, so fc1's
                    _LayerNormLinear.backward skips its cast (grad_output_preprocess). Bit-exact with the BF16
                    dx + TE cast it replaces (bytes, amax, scale_inv). Falls back to the BF16 path per call unless:
                    fc1 is a TE node found through views, FP8 delayed scaling (Float8Quantizer), no weight
                    gradient, and TE's rowwise-only grad cast (HS_GRAD_CAST_ONLY=1, the recipe's setting).
  HS_SWIGLU_FWD_FP8=1 (fix 3b, needs HS_ACT_STORE_FUSED=1)
                    the fused forward writes fc2's input as e4m3 with fc2's own delayed-scaling input quantizer
                    (hs_swiglu_fwd_e4m3.py) and returns it as a Float8Tensor, so fc2's _Linear.forward skips its
                    cast; the BF16 SwiGLU output is never materialised. Bit-exact with fix #1's BF16 out + TE cast
                    (bytes, amax, scale_inv; the act-store bytes are unchanged). The MLP forward (Megatron MLP and
                    NeMo MCoreMLPMixin) publishes its linear_fc2 in a context variable; falls back to the BF16 path
                    per call unless: fc2 is FP8-initialised under the current recipe, delayed scaling
                    (Float8Quantizer, e4m3), no userbuffer overlap, no MLP adapter consuming the BF16 output.
"""

import contextvars
import os

import torch

import megatron.core.fusions.fused_bias_swiglu as fbs
import transformer_engine_torch as tex
from transformer_engine.pytorch.distributed import in_fp8_activation_recompute_phase
from transformer_engine.pytorch.tensor.float8_tensor import Float8CurrentScalingQuantizer

USE_TE_QUANTIZER = os.getenv("HS_ACT_STORE_TE", "0") == "1"
USE_FUSED = os.getenv("HS_ACT_STORE_FUSED", "0") == "1"
BWD_FP8 = USE_FUSED and os.getenv("HS_SWIGLU_BWD_FP8", "0") == "1"
if USE_FUSED:
    # same directory as this module (HS_PATCH_PATH is on sys.path when loaded through the recipe hook)
    from hs_swiglu_fp8_triton import dswiglu_fp8in, swiglu_fwd_store
FWD_FP8 = USE_FUSED and os.getenv("HS_SWIGLU_FWD_FP8", "0") == "1"
if BWD_FP8:
    from hs_dswiglu_e5m2 import dswiglu_e5m2_into
if BWD_FP8 or FWD_FP8:
    from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer, Float8Tensor
if FWD_FP8:
    from hs_swiglu_fwd_e4m3 import swiglu_fwd_e4m3_into
    from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

_VIEW_NODES = ("ViewBackward0", "UnsafeViewBackward0", "ReshapeAliasBackward0", "AliasBackward0")
_FC1_NODES = ("_LayerNormLinearBackward", "_LinearBackward")
STATS = {"fp8_bwd": 0, "bf16_bwd": 0, "fp8_fwd": 0, "bf16_fwd": 0}   # per-process counters (tests, logs)
_FC2 = contextvars.ContextVar("hs_fc2", default=None)   # the linear_fc2 of the MLP forward in progress (fix 3b)
_MLP_ADAPTERS = ("mlp_infused_adapter", "lora_4htoh_adapter", "lora_moe_4htoh_adapter")


def _te_node(t, max_depth=4):
    """The TE autograd node (ctx of _LayerNormLinear / _Linear) that produced t, through view nodes; else None."""
    node = t.grad_fn
    for _ in range(max_depth):
        if node is None:
            return None
        name = type(node).__name__
        if name in _FC1_NODES:
            return node
        if name not in _VIEW_NODES or not node.next_functions:
            return None
        node = node.next_functions[0][0]
    return None


def _fp8_grad_quantizer(node):
    """fc1's grad_output quantizer if the e5m2 path is valid for this backward, else None."""
    if node is None or not getattr(node, "fp8", False) or getattr(node, "requires_wgrad", True):
        return None
    if os.environ.get("HS_GRAD_CAST_ONLY", "0") != "1" or getattr(node, "ub_overlap_ag", False):
        return None
    if getattr(node, "sequence_parallel", False) and getattr(node, "parallel_mode", None) == "row":
        return None
    q = getattr(node, "grad_output_quantizer", None)
    if type(q) is not Float8Quantizer or q.dtype != tex.DType.kFloat8E5M2:
        return None
    return q


def _fc2_input_quantizer(fc2):
    """fc2's input quantizer if the e4m3 forward path is valid for this call, else None. Mirrors the state
    _Linear.forward will see: prepare_forward leaves an initialised module's quantizers untouched while the
    recipe is unchanged (TE 2.2 base.py init_fp8_metadata)."""
    if fc2 is None or not FP8GlobalStateManager.is_fp8_enabled() or not getattr(fc2, "fp8_initialized", False):
        return None
    recipe = FP8GlobalStateManager.get_fp8_recipe()
    if recipe != fc2.fp8_meta.get("recipe") or not recipe.delayed():
        return None
    if any(getattr(fc2, a, False) for a in ("ub_overlap_rs_fprop", "ub_overlap_ag_fprop", "fp8_calibration")):
        return None
    if getattr(fc2, "parallel_mode", None) == "column" and getattr(fc2, "sequence_parallel", False):
        return None
    if fc2.weight.requires_grad:    # an input saved for wgrad would need the columnwise copy too
        return None
    try:
        q = fc2.quantizers["scaling_fwd"][tex.FP8FwdTensors.GEMM1_INPUT]
    except (KeyError, IndexError, TypeError):
        return None
    if type(q) is not Float8Quantizer or q.dtype != tex.DType.kFloat8E4M3:
        return None
    return q


def _mlp_fc2(mlp):
    """mlp.linear_fc2 unless an adapter of this MLP also consumes the SwiGLU output (then it must stay BF16)."""
    cfg = getattr(mlp, "adapter_cfg", None) or {}
    get = getattr(mlp, "get_adapter_module", None)
    if get is not None:
        for name in _MLP_ADAPTERS:
            if get(name) is not None and cfg.get(name, {}).get("enabled", True):
                return None
    return getattr(mlp, "linear_fc2", None)


def _wrap_mlp_forward(cls):
    orig = cls.forward
    if getattr(orig, "_hs_fc2_wrapped", False):
        return

    def forward(self, *args, **kwargs):
        token = _FC2.set(_mlp_fc2(self))
        try:
            return orig(self, *args, **kwargs)
        finally:
            _FC2.reset(token)

    forward._hs_fc2_wrapped = True
    forward.__wrapped__ = orig
    cls.forward = forward


def _store_now(fp8_input_store: bool) -> bool:
    """Store in FP8 only where the stored tensor outlives the layer's forward (not in a no-grad
    checkpoint forward, not in the recompute phase: there the input is consumed by the backward at once)."""
    return bool(fp8_input_store) and torch.is_grad_enabled() and not in_fp8_activation_recompute_phase()


class SwiGLUFunctionTE(torch.autograd.Function):
    """Megatron's SwiGLUFunction with TE's quantize/dequantize kernels for the stored input."""

    @staticmethod
    def forward(ctx, input, fp8_input_store):
        ctx.fp8_input_store = fp8_input_store
        ctx.ori_input_dtype = input.dtype
        if fp8_input_store:
            q = Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, input.device)
            q.set_usage(rowwise=True, columnwise=False)
            ctx.fp8_input = q(input)  # Float8Tensor: keeps only the 1-byte data (+ scale)
        else:
            ctx.save_for_backward(input)
        if fbs.use_te_swiglu:
            return fbs.te.ops.SwiGLU()(input)
        return fbs.swiglu(input)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.fp8_input_store:
            input = ctx.fp8_input.dequantize(dtype=ctx.ori_input_dtype)
            ctx.fp8_input = None
        else:
            input = ctx.saved_tensors[0]
        if fbs.use_te_swiglu:
            return tex.dswiglu(grad_output, input, None), None
        return fbs.swiglu_back(grad_output, input), None


class SwiGLUFunctionFused(torch.autograd.Function):
    """SwiGLU forward fused with the e4m3fn store; backward from the stored bytes. Only used when the store
    happens (_store_now); the stored tensor lives on ctx as a 1-byte-per-element float8_e4m3fn view."""

    @staticmethod
    def forward(ctx, input, fc1_node=None, fc2_quantizer=None, out_shape=None):
        ctx.fc1_node = fc1_node
        if fc2_quantizer is None:
            output, ctx.x_fp8 = swiglu_fwd_store(input)
            return output
        # fix 3b: rowwise-only Float8Tensor in fc2's quantizer (fc2 needs no columnwise copy: frozen weight)
        M, H2 = input.shape
        q = torch.empty((M, H2), dtype=torch.uint8, device=input.device)
        data = torch.empty((M, H2 // 2), dtype=torch.uint8, device=input.device)
        scale_inv = torch.empty(1, dtype=torch.float32, device=input.device)
        swiglu_fwd_e4m3_into(input, q, data, fc2_quantizer.scale, fc2_quantizer.amax, scale_inv)
        ctx.x_fp8 = q.view(torch.float8_e4m3fn)
        ctx.grad_shape = (M, H2 // 2)
        shape = tuple(out_shape) if out_shape is not None else (M, H2 // 2)
        return Float8Tensor(shape=shape, dtype=input.dtype, data=data.view(shape), fp8_scale_inv=scale_inv,
                            fp8_dtype=fc2_quantizer.dtype, requires_grad=False, data_transpose=None,
                            quantizer=fc2_quantizer)

    @staticmethod
    def backward(ctx, grad_output):
        x_fp8, ctx.x_fp8 = ctx.x_fp8, None
        node, ctx.fc1_node = ctx.fc1_node, None
        if getattr(ctx, "grad_shape", None) is not None:
            grad_output = grad_output.reshape(ctx.grad_shape)
        q = _fp8_grad_quantizer(node) if BWD_FP8 else None
        if q is None:
            STATS["bf16_bwd"] += 1
            return dswiglu_fp8in(grad_output, x_fp8), None, None, None
        # rowwise-only Float8Tensor (no transpose buffer: fc1 needs no wgrad), filled by the fused kernel
        data = torch.empty(x_fp8.shape, dtype=torch.uint8, device=x_fp8.device)
        scale_inv = torch.empty(1, dtype=torch.float32, device=x_fp8.device)
        dswiglu_e5m2_into(grad_output, x_fp8, data, q.scale, q.amax, scale_inv)
        STATS["fp8_bwd"] += 1
        return Float8Tensor(shape=x_fp8.shape, dtype=grad_output.dtype, data=data, fp8_scale_inv=scale_inv,
                            fp8_dtype=q.dtype, requires_grad=False, data_transpose=None, quantizer=q), None, None, None


_orig_bias_swiglu_impl = fbs.bias_swiglu_impl


def bias_swiglu_impl(input, bias, fp8_input_store=False):
    fp8_input_store = _store_now(fp8_input_store)
    if USE_FUSED and fp8_input_store and bias is None:
        ori_shape = input.shape
        assert len(ori_shape) in (2, 3)
        node = _te_node(input) if BWD_FP8 else None
        qz = _fc2_input_quantizer(_FC2.get()) if FWD_FP8 else None
        if FWD_FP8:
            STATS["fp8_fwd" if qz is not None else "bf16_fwd"] += 1
            n = STATS["fp8_fwd"] + STATS["bf16_fwd"]
            if n in (1, 1000) and torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                print(f"[hyperspindle] fix 3b after {n} stored SwiGLU forwards: fused e4m3 {STATS['fp8_fwd']}, "
                      f"BF16 fallback {STATS['bf16_fwd']}", flush=True)
        if qz is not None:   # the Float8Tensor is built in its final shape (no view through TE's _ViewFunc)
            return SwiGLUFunctionFused.apply(input.view(-1, ori_shape[-1]).contiguous(), node, qz,
                                             (*ori_shape[:-1], ori_shape[-1] // 2))
        out = SwiGLUFunctionFused.apply(input.view(-1, ori_shape[-1]).contiguous(), node)
        return out if len(ori_shape) == 2 else out.view(ori_shape[0], ori_shape[1], -1)
    if USE_TE_QUANTIZER and bias is None:
        ori_shape = input.shape
        assert len(ori_shape) in (2, 3)
        out = SwiGLUFunctionTE.apply(input.view(-1, ori_shape[-1]), fp8_input_store)
        return out if len(ori_shape) == 2 else out.view(ori_shape[0], ori_shape[1], -1)
    return _orig_bias_swiglu_impl(input, bias, fp8_input_store)


# mlp.py and NeMo's mcore_mixins.py import bias_swiglu_impl by name after this hook has run
fbs.bias_swiglu_impl = bias_swiglu_impl
if FWD_FP8:
    from megatron.core.transformer.mlp import MLP
    from nemo.collections.nlp.modules.common.megatron.adapters import mcore_mixins as _mm
    _wrap_mlp_forward(MLP)
    _wrap_mlp_forward(_mm.MCoreMLPMixin)
print("[hyperspindle] hs_act_store: skip store in no-grad/recompute; "
      f"fused Triton store={'on' if USE_FUSED else 'off'}; TE quantizer={'on' if USE_TE_QUANTIZER else 'off'}; "
      f"e5m2 bwd (fix 3a)={'on' if BWD_FP8 else 'off'}; e4m3 fc2-input fwd (fix 3b)={'on' if FWD_FP8 else 'off'}",
      flush=True)
