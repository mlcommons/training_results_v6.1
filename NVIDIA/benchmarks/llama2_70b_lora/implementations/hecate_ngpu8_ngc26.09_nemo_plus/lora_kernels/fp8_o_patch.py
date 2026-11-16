# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""FP8 attention output straight into the proj branch (switch LORA_PATCH_FP8_O).

TE path: cuDNN writes O in FP8 (E4M3, one scale). TE dequantizes O to bf16. The proj input
quantize reads the bf16 O, and the LoRA branch reads it again. This path: the DPA returns the
Float8Tensor O (TE fp8_output). The proj quantize and the fused LoRA kernels read the FP8 bytes
and dequantize in registers, with the TE math. The bytes of the NVFP4 input and of the LoRA
outputs stay the same, and so does the RNG state.

apply() installs three hooks:
- DotProductAttention.forward: fp8_output=True in the NVFP4 training phase.
- te.ops BasicLinear.op_forward: a Float8Tensor input goes to the FP8 NVFP4 quantize.
- NVFP4Quantizer.quantize_impl: a Float8Tensor input goes to the FP8 NVFP4 quantize.
The fused LoRA patch (lora_fused_patch.input_parts) takes the Float8Tensor O as it is.
"""

import os

import torch
import transformer_engine.pytorch  # noqa: F401  (loads transformer_engine_torch)
import transformer_engine_torch as tex
from transformer_engine.common.recipe import NVFP4BlockScaling
from transformer_engine.pytorch import DotProductAttention
from transformer_engine.pytorch.ops.basic.basic_linear import BasicLinear
from transformer_engine.pytorch.quantization import FP8GlobalStateManager
from transformer_engine.pytorch.tensor.float8_tensor import Float8Tensor
from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer, NVFP4Tensor
from transformer_engine.pytorch.tensor.storage.float8_tensor_storage import Float8TensorStorage
from transformer_engine.pytorch.tensor.storage.nvfp4_tensor_storage import NVFP4TensorStorage

import fp8_o_ext
import gemm_alpha

_FP8_TYPES = {tex.DType.kFloat8E4M3: False, tex.DType.kFloat8E5M2: True}
_FORMATS = ("sbhd", "bshd")
_state = {
    "dpa_forward": None,
    "linear_forward": None,
    "quantize_impl": None,
    "stats": {"fp8_outputs": 0, "quantize": 0, "dequantized": 0},
}


def stats() -> dict:
    """Return the counters of the fast paths."""
    return dict(_state["stats"])


def plain_fp8(tensor: object) -> bool:
    """Return True for a contiguous, aligned FP8 tensor with one float scale_inv."""
    if not isinstance(tensor, Float8TensorStorage) or tensor._fp8_dtype not in _FP8_TYPES:
        return False
    data, scale_inv = tensor._data, tensor._scale_inv
    if data is None or scale_inv is None:
        return False
    plain_data = data.is_cuda and data.dtype == torch.uint8 and data.is_contiguous()
    plain_data = plain_data and data.numel() % 16 == 0 and data.data_ptr() % 16 == 0
    return plain_data and scale_inv.numel() == 1 and scale_inv.dtype == torch.float32


def nvfp4_phase() -> bool:
    """Return True in the NVFP4 training phase (an NVFP4 autocast recipe)."""
    if not FP8GlobalStateManager.is_fp8_enabled():
        return False
    return isinstance(FP8GlobalStateManager.get_fp8_recipe(), NVFP4BlockScaling)


def _quantize_shape(data: torch.Tensor) -> bool:
    """Return True if the quantize kernel handles this shape (cols % 256, rows % 128)."""
    if data.dim() < 2 or data.numel() >= 2**31:
        return False
    cols = data.shape[-1]
    if cols == 0 or cols % 256 != 0:
        return False
    rows = data.numel() // cols
    return rows > 0 and rows % 128 == 0


def _plain_quantizer(quantizer: object) -> bool:
    """Return True for a 1D, rowwise-only NVFP4 quantizer with no RHT, 4over6 or reduction."""
    if not isinstance(quantizer, NVFP4Quantizer):
        return False
    special = quantizer.with_rht or quantizer.with_amax_reduction or quantizer.row_scaled_nvfp4
    special = special or quantizer.nvfp4_use_4over6 or quantizer.with_2d_quantization
    usage = quantizer.rowwise_usage and not quantizer.columnwise_usage
    return usage and not special and quantizer.dtype == tex.DType.kFloat4E2M1


def _make_output(quantizer: NVFP4Quantizer, tensor: Float8TensorStorage, parts: list) -> object:
    """Wrap (data, scales, amax) in the same Python class that TE makes."""
    data, scales, amax = parts
    kwargs = {
        "rowwise_data": data,
        "rowwise_scale_inv": scales,
        "columnwise_data": None,
        "columnwise_scale_inv": None,
        "amax_rowwise": amax,
        "amax_columnwise": None,
        "fp4_dtype": quantizer.dtype,
        "quantizer": quantizer,
        "with_gemm_swizzled_scales": True,
        "row_scaled_nvfp4": False,
        "nvfp4_use_4over6": False,
        "nvfp4_e4m3_max": 448,
    }
    if quantizer.internal:
        return NVFP4TensorStorage(fake_dtype=tensor.dtype, **kwargs)
    return NVFP4Tensor(shape=tensor._data.shape, dtype=tensor.dtype, device=tensor._data.device,
                       **kwargs)


def quantize_fp8(quantizer: NVFP4Quantizer, tensor: Float8TensorStorage) -> object:
    """NVFP4 quantize of an FP8 tensor: amax pass over FP8, quantize pass from FP8."""
    module = fp8_o_ext.get_extension()
    e5m2 = _FP8_TYPES[tensor._fp8_dtype]
    partials = module.fp8_amax_partials(tensor._data, tensor._scale_inv, e5m2)
    weight_amax, alpha = gemm_alpha.kernel_tensors(quantizer)
    parts = module.quantize_fp8_from_partials(tensor._data, tensor._scale_inv, e5m2, partials,
                                              bool(quantizer.stochastic_rounding),
                                              alpha_weight_amax=weight_amax, alpha_out=alpha)
    gemm_alpha.record(parts[2], quantizer, alpha is not None)
    _state["stats"]["quantize"] += 1
    return _make_output(quantizer, tensor, parts)


def _quantize_supported(quantizer: object, tensor: object) -> bool:
    """Return True if quantize_fp8 handles this call."""
    if not _plain_quantizer(quantizer) or not plain_fp8(tensor):
        return False
    return _quantize_shape(tensor._data)


def quantize_impl_fp8(self: NVFP4Quantizer, tensor: torch.Tensor) -> object:
    """Replacement of NVFP4Quantizer.quantize_impl."""
    if isinstance(tensor, Float8Tensor) and _quantize_supported(self, tensor):
        return quantize_fp8(self, tensor)
    return _state["quantize_impl"](self, tensor)


# te.ops BasicLinear: TE hands a quantized input to the GEMM as it is. Thus the hook quantizes
# the FP8 input to NVFP4 before the TE forward.
def _nvfp4_input(op: BasicLinear, tensor: Float8Tensor) -> object:
    """Return the NVFP4 input for the GEMM, or None if the fast path does not apply."""
    if not FP8GlobalStateManager.is_fp8_enabled() or op.weight.requires_grad:
        return None
    if FP8GlobalStateManager.get_fp8_recipe().backward_override is not None:
        return None
    quantizer = op.get_quantizer("forward", 0)
    if isinstance(quantizer, NVFP4Quantizer):
        quantizer.set_usage(rowwise=True, columnwise=False)
    if not _quantize_supported(quantizer, tensor):
        return None
    return quantize_fp8(quantizer, tensor)


def linear_op_forward(self: BasicLinear, ctx: object, input_: torch.Tensor,
                      prev_op_grad_output_quantizer: object,
                      next_op_input_quantizer: object) -> torch.Tensor:
    """Replacement of te.ops BasicLinear.op_forward."""
    quantizers = (prev_op_grad_output_quantizer, next_op_input_quantizer)
    if not isinstance(input_, Float8Tensor):
        return _state["linear_forward"](self, ctx, input_, *quantizers)
    nvfp4 = _nvfp4_input(self, input_)
    if nvfp4 is not None:
        return _state["linear_forward"](self, ctx, nvfp4, *quantizers)
    if getattr(input_, "_lora_fp8_o", False):
        # The TE path for this O: a bf16 input. TE must not hand the FP8 O to an NVFP4 GEMM.
        _state["stats"]["dequantized"] += 1
        input_ = input_.dequantize()
    return _state["linear_forward"](self, ctx, input_, *quantizers)


def _fp8_attention_recipe(dpa: DotProductAttention) -> bool:
    """Return True if the DPA runs FP8 attention with delayed scaling in forward and backward."""
    recipe = dpa.fp8_meta.get("recipe")
    if not dpa.fp8 or recipe is None or not recipe.delayed():
        return False
    fp8_backward = int(os.getenv("NVTE_FP8_DPA_BWD", "1")) != 0
    return recipe.fp8_dpa and not recipe.fp8_mha and fp8_backward


def _fp8_output_wanted(dpa: DotProductAttention, kwargs: dict) -> bool:
    """Return True if this forward should return the FP8 O."""
    simple_call = not kwargs.get("fp8_output") and not kwargs.get("checkpoint_core_attention")
    simple_call = simple_call and kwargs.get("inference_params") is None
    simple_call = simple_call and kwargs.get("cu_seqlens_q") is None
    if not simple_call or (kwargs.get("qkv_format") or dpa.qkv_format) not in _FORMATS:
        return False
    if not dpa.training or not torch.is_grad_enabled() or not nvfp4_phase():
        return False
    # Same call as the DPA forward. It makes the quantizers when the recipe changed.
    dpa.init_fp8_metadata(num_gemms=3)
    return _fp8_attention_recipe(dpa)


def dpa_forward(self: DotProductAttention, *args: object, **kwargs: object) -> object:
    """Replacement of DotProductAttention.forward."""
    wanted = _fp8_output_wanted(self, kwargs)
    if wanted:
        kwargs["fp8_output"] = True
    out = _state["dpa_forward"](self, *args, **kwargs)
    if wanted and isinstance(out, Float8Tensor):
        out._lora_fp8_o = True
        _state["stats"]["fp8_outputs"] += 1
    return out


def apply() -> None:
    """Install the hooks."""
    if _state["dpa_forward"] is not None:
        return
    fp8_o_ext.get_extension()
    _state["dpa_forward"] = DotProductAttention.forward
    _state["linear_forward"] = BasicLinear.op_forward
    _state["quantize_impl"] = NVFP4Quantizer.quantize_impl
    DotProductAttention.forward = dpa_forward
    BasicLinear.op_forward = linear_op_forward
    NVFP4Quantizer.quantize_impl = quantize_impl_fp8
