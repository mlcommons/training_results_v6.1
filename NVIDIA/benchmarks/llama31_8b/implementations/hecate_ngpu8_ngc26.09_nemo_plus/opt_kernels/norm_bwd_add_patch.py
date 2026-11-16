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

import os

import torch

_ENABLED = os.environ.get("OPT_NORM_BWD_ADD", "1") == "1"


class _Slot:
    __slots__ = ("residual", "armed", "grad")

    def __init__(self, residual):
        self.residual = residual
        self.armed = False
        self.grad = None


_pending = None
_open = None
_active = None
_counts = {"claimed": 0, "fused": 0, "fallback": 0}
_announced = False
_orig = {}


def _announce():
    global _announced
    if not _announced:
        _announced = True
        if int(os.environ.get("RANK", "0")) == 0:
            print("[opt norm_bwd_add] RMSNorm backward accumulates the residual gradient in place",
                  flush=True)
    if _counts["fused"] in (64, 1024) and int(os.environ.get("RANK", "0")) == 0:
        print("[opt norm_bwd_add] counts %s" % (_counts,), flush=True)


def open_slot(residual):
    global _pending, _open
    if not (torch.is_grad_enabled() and isinstance(residual, torch.Tensor) and residual.requires_grad):
        return None
    if residual.dtype != torch.bfloat16 or not residual.is_cuda or not residual.is_contiguous():
        return None
    _pending = _open = _Slot(residual)
    return _pending


def close_slot():
    global _pending, _open
    _pending = _open = None


class _ResidualGradToNorm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, out, residual, slot):
        ctx.slot = slot
        return out

    @staticmethod
    def backward(ctx, grad):
        slot, ctx.slot = ctx.slot, None
        r = slot.residual
        if (
            grad.dtype == torch.bfloat16
            and grad.is_contiguous()
            and grad.numel() == r.numel()
            and grad.device == r.device
        ):
            slot.grad = grad
            slot.residual = None
            return grad, None, None
        slot.residual = None
        return grad, grad, None


def residual_grad_to_norm(out, residual, slot):
    return _ResidualGradToNorm.apply(out, residual, slot)


def _claim(inp):
    global _pending
    s = _pending
    if s is None:
        return None
    _pending = None
    r = s.residual
    if (
        isinstance(inp, torch.Tensor)
        and inp.dtype == torch.bfloat16
        and inp.device == r.device
        and inp.numel() == r.numel()
        and inp.data_ptr() == r.data_ptr()
        and inp.is_contiguous()
    ):
        s.armed = True
        _counts["claimed"] += 1
        return s
    return None


def _rmsnorm_bwd(orig):
    def rmsnorm_bwd(dz, x, rsigma, gamma, sm_margin, zero_centered_gamma):
        s = _active
        if s is None or s.grad is None:
            return orig(dz, x, rsigma, gamma, sm_margin, zero_centered_gamma)
        rg, s.grad = s.grad, None
        N = x.shape[-1]
        if (
            isinstance(dz, torch.Tensor)
            and dz.dtype == torch.bfloat16
            and x.dtype == torch.bfloat16
            and x.is_contiguous()
            and rsigma.dtype == torch.float32
            and rsigma.is_contiguous()
            and rsigma.numel() * N == x.numel()
            and gamma.numel() == N
            and rg.numel() == x.numel()
        ):
            dgamma = torch.ops.optimized.rmsnorm_bwd_add_(
                dz.contiguous(), x, rsigma, gamma, rg, bool(zero_centered_gamma), int(sm_margin)
            )
            _counts["fused"] += 1
            _announce()
            return rg.view(dz.shape), dgamma.view(gamma.shape)
        dx, dgamma = orig(dz, x, rsigma, gamma, sm_margin, zero_centered_gamma)
        _counts["fallback"] += 1
        _announce()
        return dx + rg.view(dx.shape), dgamma

    rmsnorm_bwd._opt_orig = orig
    return rmsnorm_bwd


def _run_backward(slot, fn, *args):
    global _active
    prev, _active = _active, slot
    try:
        res = fn(*args)
    finally:
        _active = prev
    left, slot.grad = slot.grad, None
    return res, left


def _wrap_lnl(cls):
    orig_fwd, orig_bwd = cls.forward, cls.backward

    def forward(ctx, inp, *args, **kwargs):
        if _pending is not None and ctx is not None:
            s = _claim(inp)
            if s is not None:
                ctx._opt_slot = s
        return orig_fwd(ctx, inp, *args, **kwargs)

    def backward(ctx, *grads):
        s = getattr(ctx, "_opt_slot", None)
        if s is None:
            return orig_bwd(ctx, *grads)
        ctx._opt_slot = None
        res, left = _run_backward(s, orig_bwd, ctx, *grads)
        if left is not None:
            res = list(res)
            res[0] = left.view(res[0].shape) if res[0] is None else res[0] + left.view(res[0].shape)
            res = tuple(res)
        return res

    forward._opt_orig = orig_fwd
    backward._opt_orig = orig_bwd
    cls.forward = staticmethod(forward)
    cls.backward = staticmethod(backward)


def _wrap_rmsnorm_op(cls):
    orig_fwd, orig_bwd = cls.op_forward, cls.op_backward

    def op_forward(self, ctx, input_, *args, **kwargs):
        if _pending is not None:
            s = _claim(input_)
            if s is not None:
                ctx._opt_slot = s
        return orig_fwd(self, ctx, input_, *args, **kwargs)

    def op_backward(self, ctx, grad_output):
        s = getattr(ctx, "_opt_slot", None)
        if s is None:
            return orig_bwd(self, ctx, grad_output)
        ctx._opt_slot = None
        (grad_input, grad_params), left = _run_backward(s, orig_bwd, self, ctx, grad_output)
        if left is not None:
            grad_input = grad_input + left.view(grad_input.shape)
        return grad_input, grad_params

    op_forward._opt_orig = orig_fwd
    op_backward._opt_orig = orig_bwd
    cls.op_forward = op_forward
    cls.op_backward = op_backward


def _branch(orig):
    def wrapped(self, hidden_states, *args, **kwargs):
        if _open is not None or open_slot(hidden_states) is None:
            return orig(self, hidden_states, *args, **kwargs)
        try:
            return orig(self, hidden_states, *args, **kwargs)
        finally:
            close_slot()

    wrapped._opt_orig = orig
    return wrapped


def _residual_aware_bda(orig_getter):
    def getter(training, fused):
        orig_fn = orig_getter(training, fused)

        def bda(x_with_bias, residual, prob):
            s = _open
            if (
                s is not None
                and s.armed
                and isinstance(residual, torch.Tensor)
                and residual.dtype == torch.bfloat16
                and residual.data_ptr() == s.residual.data_ptr()
                and residual.numel() == s.residual.numel()
                and isinstance(x_with_bias, tuple)
                and x_with_bias[1] is None
                and float(prob) == 0.0
            ):
                out = orig_fn(x_with_bias, residual.detach(), prob)
                return residual_grad_to_norm(out, residual, s)
            return orig_fn(x_with_bias, residual, prob)

        return bda

    getter._opt_orig = orig_getter
    return getter


def _install_layer_hooks():
    from megatron.core.transformer.transformer_layer import TransformerLayer

    for name in ("_forward_attention", "_forward_mlp"):
        fn = getattr(TransformerLayer, name, None)
        if fn is None:
            raise RuntimeError(f"opt norm_bwd_add: TransformerLayer has no {name}")
        if not hasattr(fn, "_opt_orig"):
            setattr(TransformerLayer, name, _branch(fn))
    if hasattr(TransformerLayer.__init__, "_opt_orig"):
        return
    orig_init = TransformerLayer.__init__

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        for attr in ("self_attn_bda", "mlp_bda"):
            fn = getattr(self, attr, None)
            if callable(fn) and not hasattr(fn, "_opt_orig"):
                setattr(self, attr, _residual_aware_bda(fn))

    __init__._opt_orig = orig_init
    TransformerLayer.__init__ = __init__


def install():
    import transformer_engine.pytorch  # noqa: F401
    import transformer_engine_torch as tex
    import transformer_engine.pytorch.module.layernorm_linear as te_lnl
    import transformer_engine.pytorch.ops.basic.rmsnorm as te_rmsnorm_op

    from . import rmsnorm_bwd_add  # noqa: F401  (registers torch.ops.optimized.rmsnorm_bwd_add_)

    if not hasattr(tex.rmsnorm_bwd, "_opt_orig"):
        _orig["rmsnorm_bwd"] = tex.rmsnorm_bwd
        tex.rmsnorm_bwd = _rmsnorm_bwd(tex.rmsnorm_bwd)
    if not hasattr(te_rmsnorm_op.rmsnorm_bwd, "_opt_orig"):
        te_rmsnorm_op.rmsnorm_bwd = _rmsnorm_bwd(te_rmsnorm_op.rmsnorm_bwd)
    if not hasattr(te_lnl._LayerNormLinear.forward, "_opt_orig"):
        _wrap_lnl(te_lnl._LayerNormLinear)
    if not hasattr(te_rmsnorm_op.RMSNorm.op_forward, "_opt_orig"):
        _wrap_rmsnorm_op(te_rmsnorm_op.RMSNorm)
    _install_layer_hooks()


if _ENABLED:
    install()
