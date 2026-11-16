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

"""Gradient accumulation of the residual stream with amax partials.

In each transformer layer, the residual stream tensor has two consumers: the norm (inside the
te.ops fuser) and the residual add. Autograd sums their two gradients with a bf16 add before
it passes the sum on. That sum is the grad_output of the proj dgrad or of the fc2 dgrad, and the
NVFP4 quantize of it runs an amax pass.

Fork gives the two consumers two distinct views of the tensor. Then its backward gets the two
gradients and adds them with the PyTorch math, bf16(float(a) + float(b)). The kernel also
writes the amax partials, and the backward registers them in the side channel.

Older MCore (the pin of Bridge v0.6.0) has no _run_input_layernorm and no
_pre_mlp_layernorm_and_residual. There, the layer takes a norm output (output, residual) as the
norm output and the residual. Thus Fork goes into the input norm module of each layer and into
_forward_pre_mlp_layernorm.
"""

import functools

import torch

import amax_channel
import amax_prod_ext

_state = {"wrapped": {}, "forks": 0}


def _plain(x: torch.Tensor) -> bool:
    return (type(x) is torch.Tensor and x.is_cuda and x.dtype == torch.bfloat16
            and x.is_contiguous() and x.data_ptr() % 16 == 0 and x.numel() % 8 == 0)


class Fork(torch.autograd.Function):
    """Two views of one tensor, with a fused gradient accumulation."""

    @staticmethod
    def forward(ctx: object, x: torch.Tensor) -> tuple:
        ctx.set_materialize_grads(False)
        ctx.channel_active = amax_channel.active()
        return x.view_as(x), x.view_as(x)

    @staticmethod
    def backward(ctx: object, grad_a: torch.Tensor | None, grad_b: torch.Tensor | None) -> torch.Tensor | None:
        if grad_a is None:
            return grad_b
        if grad_b is None:
            return grad_a
        same = grad_a.shape == grad_b.shape and grad_a.dtype == grad_b.dtype
        if not (same and _plain(grad_a) and _plain(grad_b)):
            return grad_a + grad_b
        out, partials = amax_prod_ext.get_module().add_partials(grad_a, grad_b)
        if ctx.channel_active:
            amax_channel.register(out, partials)
        return out


def fork_if_needed(hidden_states: object) -> tuple:
    """Return (view for the norm, view for the residual), or the tensor twice."""
    if torch.is_grad_enabled() and isinstance(hidden_states, torch.Tensor) \
            and hidden_states.requires_grad and _plain(hidden_states):
        _state["forks"] += 1
        return Fork.apply(hidden_states)
    return hidden_states, hidden_states


def wrap_norm_method(cls: type, name: str) -> None:
    """Wrap a method (hidden_states, ...) -> (norm output, residual, state) with Fork."""
    if (cls, name) in _state["wrapped"]:
        return
    original = getattr(cls, name)

    def wrapper(self: object, hidden_states: object, *args: object, **kwargs: object) -> tuple:
        norm_input, residual_view = fork_if_needed(hidden_states)
        result = original(self, norm_input, *args, **kwargs)
        output, residual, *rest = result
        if residual is norm_input:
            residual = residual_view
        return (output, residual, *rest)

    wrapper.__name__ = name
    setattr(cls, name, wrapper)
    _state["wrapped"][(cls, name)] = original


def forked_norm(norm: object, hidden_states: object) -> object:
    """Older MCore: run the norm on the norm view. Return (norm output, residual view).

    If Fork does not apply, return the norm output unchanged.
    """
    norm_input, residual_view = fork_if_needed(hidden_states)
    output = norm(norm_input)
    if norm_input is residual_view:
        return output
    residual = residual_view
    if isinstance(output, tuple):
        output, residual = output
        if residual is norm_input:
            residual = residual_view
    return output, residual


def fork_input_norm(layer: object) -> None:
    """Older MCore: give the input norm module of the layer a forward with Fork."""
    norm = layer.input_layernorm
    # The recompute checkpoint frees all norm outputs, thus also the residual view.
    if layer.recompute_input_layernorm or "forward" in vars(norm):
        return
    norm.forward = functools.partial(forked_norm, norm.forward)


def wrap_old_layer(cls: type) -> None:
    """Older MCore: install Fork in the input norm and in _forward_pre_mlp_layernorm."""
    if (cls, "_forward_attention") in _state["wrapped"]:
        return
    original_attention = cls._forward_attention
    original_pre_mlp = cls._forward_pre_mlp_layernorm

    @functools.wraps(original_attention)
    def forward_attention(self: object, *args: object, **kwargs: object) -> object:
        fork_input_norm(self)
        return original_attention(self, *args, **kwargs)

    @functools.wraps(original_pre_mlp)
    def pre_mlp_layernorm(self: object, hidden_states: object) -> object:
        return forked_norm(functools.partial(original_pre_mlp, self), hidden_states)

    cls._forward_attention = forward_attention
    cls._forward_pre_mlp_layernorm = pre_mlp_layernorm
    _state["wrapped"][(cls, "_forward_attention")] = original_attention
    _state["wrapped"][(cls, "_forward_pre_mlp_layernorm")] = original_pre_mlp


def apply() -> None:
    """Install Fork in the two norm-and-residual steps of the MCore transformer layer."""
    from megatron.core.transformer.transformer_layer import TransformerLayer

    amax_prod_ext.get_module()
    if not hasattr(TransformerLayer, "_run_input_layernorm"):
        wrap_old_layer(TransformerLayer)
        return
    wrap_norm_method(TransformerLayer, "_run_input_layernorm")
    wrap_norm_method(TransformerLayer, "_pre_mlp_layernorm_and_residual")
