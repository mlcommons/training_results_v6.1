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

"""Fuse the residual add with the RMSNorm and the NVFP4 quantize of the next sublayer.

Switch LORA_PATCH_RESNORM=1. MCore adds each sublayer output to the residual stream. The next
sublayer starts with an RMSNorm and an NVFP4 quantize inside te.ops. That sublayer is the MLP
of the same layer, or the qkv LoRA linear of the next layer.

Here two kernels (resnorm.cu) make the sum, the norm, and the NVFP4 tensor. For the qkv LoRA
linear, the quantize kernel also writes the norm output in bf16 for the LoRA branch (Bridge #6342:
the LoRA branch reads the unquantized norm output). The add hook puts the outputs in a side
channel under the sum. The consumer takes them and runs its te.ops pipeline without the norm and
the quantize. In the backward, one kernel makes the norm backward, the residual grad sum, and the
amax partials. The bits differ from TE: the rstd sum
order differs, and the grad sum rounds one time less.
"""

import os

import megatron.core.fusions.fused_bias_dropout as fbd
import torch
import transformer_engine.pytorch as te
from megatron.core.extensions.transformer_engine import TEFusedMLP
from megatron.core.transformer.transformer_layer import TransformerLayer
from transformer_engine.pytorch.ops.basic import BasicLinear, RMSNorm
from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer, NVFP4Tensor

import amax_channel
import gemm_alpha
import nvfp4q_patch
import resnorm_ext

COLS = 8192
_entries: dict = {}
_reduced: dict = {}  # id(full te.ops pipeline) -> (full pipeline, pipeline without the norm)
_state: dict = {"consumer": None, "originals": {}, "reported": set()}
counts: dict = {"fused": 0, "fallback": 0, "taken_mlp": 0, "taken_qkv": 0}


def _key(tensor: torch.Tensor) -> tuple:
    return (tensor.data_ptr(), tensor.numel(), tensor.dtype)


def _report(key: str, message: str) -> None:
    if key not in _state["reported"]:
        _state["reported"].add(key)
        print(f"[resnorm] {message}", flush=True)


def clear() -> None:
    """Drop entries that no consumer took. Call at each step start."""
    _entries.clear()


def stats() -> dict:
    """Return the call counters."""
    return dict(counts, entries=len(_entries))


def _take(tensor: object) -> tuple | None:
    """Return the NVFP4 tensors of a sum that the add hook made, or None."""
    if not isinstance(tensor, torch.Tensor) or not _entries:
        return None
    entry = _entries.pop(_key(tensor), None)
    if entry is None:
        return None
    source, outputs = entry
    if source.shape != tensor.shape:
        return None
    return outputs


def _plain_rows(tensor: torch.Tensor) -> bool:
    if type(tensor) is not torch.Tensor or not tensor.is_cuda or tensor.dtype != torch.bfloat16:
        return False
    if not tensor.is_contiguous() or tensor.shape[-1] != COLS or tensor.data_ptr() % 16 != 0:
        return False
    return (tensor.numel() // COLS) % 128 == 0


def _plain_grad(grad: torch.Tensor | None) -> torch.Tensor | None:
    if grad is None:
        return None
    if type(grad) is not torch.Tensor:
        grad = grad.dequantize()
    return grad.contiguous()


def _make_tensor(quantizer: NVFP4Quantizer, like: torch.Tensor, parts: list) -> NVFP4Tensor:
    """Wrap (data, swizzled scales, amax) in an NVFP4Tensor.

    Not the storage class, also for an internal quantizer: the te.ops fuser reads requires_grad
    of its input.
    """
    data, scales, amax = parts
    return NVFP4Tensor(shape=like.shape, dtype=like.dtype, device=like.device,
                       rowwise_data=data, rowwise_scale_inv=scales, columnwise_data=None,
                       columnwise_scale_inv=None, amax_rowwise=amax, amax_columnwise=None,
                       fp4_dtype=quantizer.dtype, quantizer=quantizer,
                       with_gemm_swizzled_scales=True, row_scaled_nvfp4=False,
                       nvfp4_use_4over6=False, nvfp4_e4m3_max=448)


class ResNormFused(torch.autograd.Function):
    """(residual, x) -> (h, NVFP4(norm(h))[, norm(h) in bf16 for the LoRA branch])."""

    @staticmethod
    def forward(ctx: object, residual: torch.Tensor, x: torch.Tensor,
                weight: torch.Tensor, options: tuple) -> tuple:
        # The signature has the five arguments of autograd.Function.apply.
        eps, quantizer, zero_centered, dual = options
        ext = resnorm_ext.get_extension()
        h, rstd, partial = ext.forward_stats(residual, x, weight, eps, zero_centered)
        gemm_weight_amax, alpha = gemm_alpha.kernel_tensors(quantizer)
        parts = list(ext.quantize_norm(h, rstd, weight, partial, zero_centered,
                                       alpha_weight_amax=gemm_weight_amax, alpha_out=alpha,
                                       with_y=dual))
        gemm_alpha.record(parts[2], quantizer, alpha is not None)
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(h, rstd, weight)
        ctx.zero_centered = zero_centered
        ctx.channel = amax_channel.active()
        y = _make_tensor(quantizer, h, parts[:3])
        if dual:
            return h, y, parts[3]
        return h, y

    @staticmethod
    def backward(ctx: object, grad_h: torch.Tensor | None, grad_y: torch.Tensor | None,
                 grad_y2: torch.Tensor | None = None) -> tuple:
        h, rstd, weight = ctx.saved_tensors
        ext = resnorm_ext.get_extension()
        out, partial = ext.backward(_plain_grad(grad_h), _plain_grad(grad_y), _plain_grad(grad_y2),
                                    h, rstd, weight, ctx.zero_centered)
        if ctx.channel:
            amax_channel.register(out, partial)
        return out, out, None, None


def _input_quantizer(linear: BasicLinear) -> NVFP4Quantizer | None:
    quantizer = linear.get_quantizer("forward", 0)
    if not isinstance(quantizer, NVFP4Quantizer) or not nvfp4q_patch._plain_quantizer(quantizer):
        return None
    return quantizer


def _fused_impl(mlp: TEFusedMLP) -> object:
    """The te.ops pipeline of the MLP. Older MCore has no _get_fused_impl: forward makes it."""
    if hasattr(mlp, "_get_fused_impl"):
        return mlp._get_fused_impl()
    if mlp._fused_impl is None:
        mlp._fused_impl = (mlp._make_fused_impl(),)
    return mlp._fused_impl[0]


def _mlp_parts(mlp: TEFusedMLP) -> tuple | None:
    """(norm op, fc1 BasicLinear) of the MLP pipeline [RMSNorm, fc1, SwiGLU, fc2], or None."""
    ops = list(_fused_impl(mlp))
    if len(ops) != 4 or not isinstance(ops[0], RMSNorm) or not isinstance(ops[1], BasicLinear):
        return None
    return ops[0], ops[1]


def _qkv_parts(linear: object) -> tuple | None:
    """(norm op, BasicLinear) of the LoRA main branch.

    The branch is [RMSNorm, MakeExtraOutput, Linear] (Bridge #6342 removed the Quantize op).
    """
    if not getattr(linear, "_adapter_enabled", False):
        return None
    if linear._fused_branches is None:
        linear._fused_branches = linear._make_fused_branches()
    ops = list(linear._fused_branches[0])
    if len(ops) != 3 or not isinstance(ops[0], RMSNorm):
        return None
    basic = [op for op in ops[2].basic_ops if isinstance(op, BasicLinear)]
    return (ops[0], basic[0]) if len(basic) == 1 else None


def _consumer_params(consumer: tuple) -> tuple | None:
    """(weight, eps, zero_centered, quantizer, dual) of the norm that reads the sum, or None."""
    module, kind = consumer
    parts = _mlp_parts(module) if kind == "mlp" else _qkv_parts(module)
    if parts is None:
        return None
    norm, linear = parts
    quantizer = _input_quantizer(linear)
    weight = norm.weight
    if quantizer is None or weight.requires_grad or not _plain_weight(weight):
        return None
    return weight, norm.eps, norm.zero_centered_gamma, quantizer, kind == "qkv"


def _plain_weight(weight: torch.Tensor) -> bool:
    return (weight.is_cuda and weight.dtype == torch.bfloat16 and weight.is_contiguous()
            and weight.numel() == COLS and weight.data_ptr() % 16 == 0)


def _fused_add(original: object, x_with_bias: tuple, residual: torch.Tensor,
               prob: float) -> torch.Tensor:
    """The residual add: fused with the next norm when the consumer allows it."""
    consumer = _state["consumer"]
    x, bias = x_with_bias
    if consumer is None or bias is not None or prob != 0.0:
        return original(x_with_bias, residual, prob)
    usable = amax_channel.nvfp4_recipe_active() and _plain_rows(x) and _plain_rows(residual)
    params = _consumer_params(consumer) if usable and x.shape == residual.shape else None
    if params is None:
        counts["fallback"] += 1
        return original(x_with_bias, residual, prob)
    weight, eps, zero_centered, quantizer, dual = params
    outputs = ResNormFused.apply(residual, x, weight, (eps, quantizer, zero_centered, dual))
    h = outputs[0]
    _entries[_key(h)] = (h, outputs[1:])
    counts["fused"] += 1
    _report("fused", "first fused residual add + RMSNorm + NVFP4 quantize")
    return h


def _bda_train(x_with_bias: tuple, residual: torch.Tensor, prob: float) -> torch.Tensor:
    return _fused_add(_state["originals"]["train"], x_with_bias, residual, prob)


def _bda_inference(x_with_bias: tuple, residual: torch.Tensor, prob: float) -> torch.Tensor:
    return _fused_add(_state["originals"]["inference"], x_with_bias, residual, prob)


def _with_consumer(name: str, find_consumer: object) -> None:
    """Wrap a TransformerLayer bda step: the add inside it knows which norm reads the sum."""
    original = getattr(TransformerLayer, name)
    _state["originals"][name] = original

    def wrapped(self: TransformerLayer, *args: object, **kwargs: object) -> object:
        previous = _state["consumer"]
        _state["consumer"] = find_consumer(self)
        try:
            return original(self, *args, **kwargs)
        finally:
            _state["consumer"] = previous

    setattr(TransformerLayer, name, wrapped)


def _mlp_consumer(layer: TransformerLayer) -> tuple | None:
    return (layer.mlp, "mlp") if isinstance(layer.mlp, TEFusedMLP) else None


def _qkv_consumer(layer: TransformerLayer) -> tuple | None:
    following = getattr(layer, "_resnorm_next", None)
    if following is None:
        return None
    return (following.self_attention.linear_qkv, "qkv")


def _reduced_pipeline(full: object, first: int) -> object:
    entry = _reduced.get(id(full))
    if entry is None or entry[0] is not full:
        entry = (full, te.ops.Sequential(*list(full)[first:]))
        _reduced[id(full)] = entry
    return entry[1]


def _mlp_forward(self: TEFusedMLP, hidden_states: torch.Tensor, **kwargs: object) -> tuple:
    """Replacement of TEFusedMLP.forward: start after the norm when the add made its output."""
    outputs = _take(hidden_states)
    if outputs is None:
        return _state["originals"]["mlp_forward"](self, hidden_states, **kwargs)
    counts["taken_mlp"] += 1
    out = _reduced_pipeline(_fused_impl(self), 1)(outputs[0])
    bias = None
    if self.linear_fc2.te_return_bias:
        bias = self.linear_fc2.bias
        if isinstance(bias, torch.Tensor) and bias.numel() == 0:
            bias = None
    return out, bias


class MainBranch:
    """The te.ops main branch of a LoRA linear.

    It starts at the linear when the add hook made its NVFP4 input and the bf16 LoRA input.
    """

    def __init__(self, main: object) -> None:
        self._main = main

    def __call__(self, x: torch.Tensor) -> tuple:
        outputs = _take(x)
        if outputs is None:
            return self._main(x)
        counts["taken_qkv"] += 1
        for hook in self._main._forward_pre_hooks.values():
            hook(self._main, (outputs[0],))
        y, y_lora = outputs
        return _reduced_pipeline(self._main, 2)(y), y_lora

    def __getattr__(self, name: str) -> object:
        return getattr(self._main, name)

    def __getitem__(self, index: int) -> object:
        return self._main[index]

    def __iter__(self) -> object:
        return iter(self._main)

    def __len__(self) -> int:
        return len(self._main)


def _make_fused_branches(self: object) -> tuple:
    main, lora = _state["originals"]["make_branches"](self)
    return MainBranch(main), lora


def mark_model(model: object) -> int:
    """Give each transformer layer a link to the next one. Return the number of links."""
    from megatron.core.transformer.transformer_block import TransformerBlock

    chunks = list(model) if isinstance(model, (list, tuple)) else [model]
    links = 0
    for chunk in chunks:
        for module in chunk.modules():
            if isinstance(module, TransformerBlock):
                layers = list(module.layers)
                for index, layer in enumerate(layers):
                    following = layers[index + 1] if index + 1 < len(layers) else None
                    object.__setattr__(layer, "_resnorm_next", following)
                    links += following is not None
    _report("mark", f"linked {links} layers")
    return links


def apply() -> None:
    """Install the hooks when LORA_PATCH_RESNORM=1. Apply after the fused LoRA patch."""
    if _state["originals"] or os.environ.get("LORA_PATCH_RESNORM", "0") != "1":
        return
    from megatron.bridge.peft.lora_layers import TEFusedLoRALinear

    resnorm_ext.get_extension()
    _state["originals"]["train"] = fbd.bias_dropout_add_fused_train
    _state["originals"]["inference"] = fbd.bias_dropout_add_fused_inference
    fbd.bias_dropout_add_fused_train = _bda_train
    fbd.bias_dropout_add_fused_inference = _bda_inference
    if hasattr(TransformerLayer, "_apply_self_attn_bda_step"):
        _with_consumer("_apply_self_attn_bda_step", _mlp_consumer)
        _with_consumer("_apply_mlp_bda_step", _qkv_consumer)
    else:
        # Older MCore: the self-attention add runs in _forward_attention and the MLP add in
        # _forward_post_mlp. No other bias-dropout-add runs in these two methods.
        _with_consumer("_forward_attention", _mlp_consumer)
        _with_consumer("_forward_post_mlp", _qkv_consumer)
    _state["originals"]["mlp_forward"] = TEFusedMLP.forward
    TEFusedMLP.forward = _mlp_forward
    _state["originals"]["make_branches"] = TEFusedLoRALinear._make_fused_branches
    TEFusedLoRALinear._make_fused_branches = _make_fused_branches
