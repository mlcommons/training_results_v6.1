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

"""FP8 output for the fused QKV RoPE in MCore attention, with the same bytes as TE.

TE path: the RoPE kernel writes bf16 q, k and v. The DPA casts them to FP8 with its QKV
quantizer (delayed scaling). In the backward, the DPA dequantizes the FP8 dQ, dK and dV, and
then the RoPE backward runs.
This path: one kernel does the RoPE and the FP8 cast with the same DPA quantizer. The DPA gets
Float8Tensor inputs, thus it skips the cast and returns FP8 dQ, dK and dV. One kernel does the
dequantize and the RoPE backward.

apply() wraps MCore Attention.forward_pre_attn_and_core_attn (Attention.forward in older MCore)
to find the DPA module of the layer. It also replaces apply_fused_qkv_rotary_pos_emb in
megatron.core.transformer.attention.
"""

import os

import torch
import transformer_engine.pytorch  # noqa: F401  (loads transformer_engine_torch)
import transformer_engine_torch as tex
from transformer_engine.pytorch import DotProductAttention
from transformer_engine.pytorch.attention.dot_product_attention import (
    utils as dpa_utils,
)
from transformer_engine.pytorch.quantization import FP8GlobalStateManager
from transformer_engine.pytorch.tensor.float8_tensor import (
    Float8Quantizer,
    Float8Tensor,
)
from transformer_engine.pytorch.tensor.storage.float8_tensor_storage import (
    Float8TensorStorage,
)

import lora_attn_ext

_HEAD_DIM = 128
# The kernels copy one token row into shared memory (lora_attn.cu kMaxRowBytes).
_MAX_ROW_BYTES = 48 * 1024
_FORMATS = {
    "sbhd": tex.NVTE_QKV_Format.NVTE_SBHD,
    "bshd": tex.NVTE_QKV_Format.NVTE_BSHD,
}

_state = {"attention": [], "core_forward": None, "rope": None}


def _fp8_attention_recipe(dpa: DotProductAttention) -> bool:
    """Return True if the DPA runs FP8 attention with delayed scaling in forward and backward."""
    recipe = dpa.fp8_meta.get("recipe")
    if not dpa.fp8 or recipe is None or not recipe.delayed():
        return False
    fp8_backward = int(os.getenv("NVTE_FP8_DPA_BWD", "1")) != 0
    return recipe.fp8_dpa and not recipe.fp8_mha and fp8_backward


def qkv_quantizer(dpa: object) -> Float8Quantizer | None:
    """Return the QKV quantizer that the DPA uses in this forward, or None."""
    if not isinstance(dpa, DotProductAttention) or not FP8GlobalStateManager.is_fp8_enabled():
        return None
    # Same call as DotProductAttention.get_qkv_quantization_capabilities. It returns at once
    # if the recipe did not change, and it makes the new quantizers if the recipe changed.
    dpa.init_fp8_metadata(num_gemms=3)
    if not _fp8_attention_recipe(dpa):
        return None
    quantizer = dpa.quantizers["scaling_fwd"][dpa_utils.META_QKV]
    if type(quantizer) is not Float8Quantizer or quantizer.dtype != tex.DType.kFloat8E4M3:
        return None
    return quantizer


def _split_supported(qkv: torch.Tensor, split: list) -> bool:
    """Return True for the split [k * 128, 128, 128] that the kernel handles."""
    if len(split) != 3 or split[1] != _HEAD_DIM or split[2] != _HEAD_DIM:
        return False
    return split[0] % _HEAD_DIM == 0 and sum(split) == qkv.size(3)


def _qkv_supported(qkv: object) -> bool:
    """Return True for a contiguous, aligned 4D bf16 CUDA tensor (not a subclass)."""
    if type(qkv) is not torch.Tensor or not qkv.is_cuda or qkv.dim() != 4:
        return False
    if qkv.dtype != torch.bfloat16 or not qkv.is_contiguous() or qkv.data_ptr() % 16 != 0:
        return False
    # The kernels use bulk copies (TMA), thus they need sm_90 or later.
    small_row = qkv.size(2) * qkv.size(3) * qkv.element_size() <= _MAX_ROW_BYTES
    return small_row and torch.cuda.get_device_capability(qkv.device)[0] >= 9


def _freq_supported(freq: torch.Tensor, positions: int) -> bool:
    """Return True for contiguous fp32 freqs of shape [s2, 1, 1, 128] with s2 >= positions."""
    if freq.dtype != torch.float32 or not freq.is_contiguous() or freq.dim() != 4:
        return False
    if freq.size(1) != 1 or freq.size(2) != 1:
        return False
    return freq.size(3) == _HEAD_DIM and freq.size(0) >= positions


def _inputs_supported(qkv: object, freqs: tuple, split: list, sbhd: bool) -> bool:
    """Return True if the kernel handles these RoPE inputs."""
    if not _qkv_supported(qkv) or not _split_supported(qkv, split):
        return False
    positions = qkv.size(0) if sbhd else qkv.size(1)
    return all(_freq_supported(freq, positions) for freq in freqs)


def _output_shapes(qkv: torch.Tensor, split: list) -> tuple:
    """Return the q shape and the k/v shape, as in TE fused_qkv_rope_forward."""
    q_shape = [qkv.size(0), qkv.size(1), qkv.size(2) * split[0] // _HEAD_DIM, _HEAD_DIM]
    kv_shape = [qkv.size(0), qkv.size(1), qkv.size(2), _HEAD_DIM]
    return q_shape, kv_shape


class _FP8QKVRoPE(torch.autograd.Function):
    """RoPE with Float8Tensor outputs. The backward takes FP8 or bf16 grads."""

    @staticmethod
    def forward(
        ctx: object, qkv: torch.Tensor, freqs: tuple, layout: tuple, quantizer: Float8Quantizer
    ) -> tuple:
        """Return q, k and v as Float8Tensors that share one scale_inv.

        The freqs argument holds (q_freqs, k_freqs). The layout argument holds
        (split, tensor_format).
        """
        q_freqs, k_freqs = freqs
        split, tensor_format = layout
        q_shape, kv_shape = _output_shapes(qkv, split)
        quantizer.internal = False
        quantizer.set_usage(rowwise=True, columnwise=False)
        q = quantizer.make_empty(q_shape, dtype=qkv.dtype, device=qkv.device)
        k_data = torch.empty(kv_shape, dtype=torch.uint8, device=qkv.device)
        v_data = torch.empty(kv_shape, dtype=torch.uint8, device=qkv.device)
        lora_attn_ext.get_module().rope_forward_fp8(
            qkv, q_freqs, k_freqs, split, tensor_format == "sbhd",
            quantizer.scale, quantizer.amax, q._scale_inv, q._data, k_data, v_data,
        )
        k = Float8Tensor.make_like(q, data=k_data, dtype=qkv.dtype)
        v = Float8Tensor.make_like(q, data=v_data, dtype=qkv.dtype)
        ctx.save_for_backward(q_freqs, k_freqs)
        ctx.split = split
        ctx.tensor_format = tensor_format
        return q, k, v

    @staticmethod
    def backward(ctx: object, dq: torch.Tensor, dk: torch.Tensor, dv: torch.Tensor) -> tuple:
        """Return the qkv grad. FP8 grads use dq's scale_inv, as TE combine_and_dequantize."""
        q_freqs, k_freqs = ctx.saved_tensors
        grads = (dq, dk, dv)
        data = None
        if all(isinstance(grad, Float8TensorStorage) for grad in grads):
            data = [grad._data.contiguous() for grad in grads]
        if data is not None and all(item.data_ptr() % 16 == 0 for item in data):
            e5m2 = dq._fp8_dtype == tex.DType.kFloat8E5M2
            dqkv = lora_attn_ext.get_module().rope_backward_fp8(
                *data, dq._scale_inv, q_freqs, k_freqs, ctx.split,
                ctx.tensor_format == "sbhd", e5m2,
            )
        else:
            plain = _plain_grads(grads)
            dqkv = tex.fused_qkv_rope_backward(
                *plain, q_freqs, k_freqs, ctx.split, _FORMATS[ctx.tensor_format], False, 1, 0
            )
        return dqkv, None, None, None


def _plain_grads(grads: tuple) -> list:
    """Return contiguous bf16 grads. FP8 grads use dq's scale_inv, as TE combine_and_dequantize."""
    first = grads[0]
    if all(isinstance(grad, Float8TensorStorage) for grad in grads):
        return [
            Float8Tensor.make_like(first, data=grad._data.contiguous(), dtype=first.dtype)
            .dequantize(dtype=first.dtype)
            .contiguous()
            for grad in grads
        ]
    return [_dequantize(grad).contiguous() for grad in grads]


def _dequantize(tensor: torch.Tensor) -> torch.Tensor:
    """Return a plain tensor."""
    if isinstance(tensor, Float8TensorStorage):
        return tensor.dequantize()
    return tensor


# Parameters and defaults of TE apply_fused_qkv_rotary_pos_emb, in order.
_ROPE_PARAMS = ("qkv", "q_freqs", "k_freqs", "qkv_split_arg_list", "tensor_format",
                "start_positions", "interleaved", "cu_seqlens", "cp_size", "cp_rank")
_ROPE_DEFAULTS = {"tensor_format": "sbhd", "start_positions": None, "interleaved": False,
                  "cu_seqlens": None, "cp_size": 1, "cp_rank": 0}


def _rope_arguments(args: tuple, kwargs: dict) -> dict:
    """Return the RoPE arguments by name."""
    values = dict(_ROPE_DEFAULTS)
    values.update(zip(_ROPE_PARAMS, args, strict=False))
    values.update(kwargs)
    return values


def _fp8_plan(values: dict) -> tuple:
    """Return (quantizer, freqs, layout) for the FP8 path, or Nones for the TE path."""
    none = (None, None, None)
    tensor_format = values["tensor_format"]
    simple = values["start_positions"] is None and not values["interleaved"]
    dpa = _state["attention"][-1] if _state["attention"] else None
    if not simple or values["cp_size"] != 1 or tensor_format not in _FORMATS or dpa is None:
        return none
    split = list(values["qkv_split_arg_list"])
    freqs = tuple(f if f.dtype == torch.float32 else f.float()
                  for f in (values["q_freqs"], values["k_freqs"]))
    if not _inputs_supported(values["qkv"], freqs, split, tensor_format == "sbhd"):
        return none
    quantizer = qkv_quantizer(dpa)
    if quantizer is None:
        return none
    return quantizer, freqs, (split, tensor_format)


def fused_qkv_rope(*args: object, **kwargs: object) -> tuple:
    """Same arguments and results as TE apply_fused_qkv_rotary_pos_emb."""
    values = _rope_arguments(args, kwargs)
    quantizer, freqs, layout = _fp8_plan(values)
    if quantizer is None:
        return _state["rope"](*args, **kwargs)
    return _FP8QKVRoPE.apply(values["qkv"], freqs, layout, quantizer)


def _forward_pre_attn_and_core_attn(self: object, *args: object, **kwargs: object) -> object:
    """Remember the DPA module of this attention layer while the original method runs."""
    _state["attention"].append(getattr(self, "core_attention", None))
    try:
        return _state["core_forward"](self, *args, **kwargs)
    finally:
        _state["attention"].pop()


def apply() -> None:
    """Install the FP8 RoPE in MCore attention."""
    if _state["rope"] is not None:
        return
    import megatron.core.transformer.attention as mcore_attention

    if not getattr(mcore_attention, "HAVE_FUSED_QKV_ROPE", False):
        return
    lora_attn_ext.get_module()
    # Older MCore has no forward_pre_attn_and_core_attn: its forward runs the RoPE and the DPA.
    name = "forward_pre_attn_and_core_attn"
    if not hasattr(mcore_attention.Attention, name):
        name = "forward"
    _state["rope"] = mcore_attention.apply_fused_qkv_rotary_pos_emb
    _state["core_forward"] = getattr(mcore_attention.Attention, name)
    mcore_attention.apply_fused_qkv_rotary_pos_emb = fused_qkv_rope
    setattr(mcore_attention.Attention, name, _forward_pre_attn_and_core_attn)
