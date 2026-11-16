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
# Portions of this file (the Triton RMSNorm forward kernel, reproduced with an added epilogue) are adapted from ROCm
# Transformer Engine, transformer_engine/pytorch/triton_kernels/rmsnorm.py, Copyright (c) 2025, Advanced Micro
# Devices, Inc. All rights reserved. AMD's contributions to Transformer Engine are licensed under the MIT License
# (see NOTICE).

"""Fix 8a: the qkv RMSNorm writes its BF16 output (for the LoRA adapter) and the e4m3 GEMM input (+ amax) in one
pass (env HS_QKVNORM_E4M3=1; needs HS_RMSNORM_GRID=1, fix 13, whose pinned launch this kernel copies).

Today the qkv TE LayerNormLinear has return_layernorm_output=True (NeMo's LoRA reads the norm output), so TE
cannot fuse the quantizer into the norm: the Triton RMSNorm forward writes BF16 ln_out, then `input_quantizer(ln_out)`
runs TE's unary_kernel cast to e4m3 (68 us at [8192, 8192], 80 per step). The fc1 norm (no returned output)
already writes FP8 directly. Here a copy of TE's `_rmsnorm_fwd_triton_impl` (non-blocked path, same tree: fix 13's
grid and 4 warps) stores the BF16 row and, from the same BF16-rounded value, the e4m3fnuz bytes with the qkv input
quantizer's scale (TE unary_kernel semantics via hs_swiglu_fwd_e4m3._fnuz_te: x scale, clamp finite to +-240,
hardware RNE), max-accumulates |BF16 value| into the quantizer's amax and writes scale_inv. The Float8Tensor is
handed to TE through the quantizer call the forward makes on that very BF16 tensor, so the cast kernel never runs.
Bit-exact with today's chain by construction (rowwise only: the frozen weight needs no columnwise copy).

Applies when: FP8 delayed scaling, RMSNorm without bias, return_layernorm_output (not gathered), no sequence
parallelism / comm overlap, BF16 activations, frozen weight, non-blocked row length (N <= 32768).
"""

import inspect
import os
import threading

import torch
import triton
import triton.language as tl

from transformer_engine.pytorch.module import layernorm_linear as lnl_mod
from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer, Float8Tensor
from transformer_engine.pytorch.triton_kernels.norm_common import block_size, use_blocked
import transformer_engine_torch as tex

import hs_rmsnorm_grid as _grid
from hs_swiglu_fwd_e4m3 import _fnuz_te

_ENABLED = os.getenv("HS_QKVNORM_E4M3", "0") == "1"
STATS = {"dual": 0, "handed": 0, "skipped": 0}
_tls = threading.local()   # dual: the input quantizer for the norm call in progress; pre: (bf16 ln_out, Float8Tensor)


def set_enabled(on: bool):
    global _ENABLED
    _ENABLED = bool(on)


@triton.jit
def _rmsnorm_fwd_dual_kernel(output_ptr, input_ptr, g_ptr, rsigma_ptr, f8_ptr, q_scale_ptr, q_amax_ptr,
                             scale_inv_ptr, input_row_stride, output_row_stride, n_rows, n_cols, epsilon,
                             ZERO_CENTERED_GAMMA: tl.constexpr, BLOCK_SIZE: tl.constexpr, NUM_PRGMS: tl.constexpr):
    """TE's _rmsnorm_fwd_triton_impl (non-blocked, BF16 output) verbatim + e4m3 bytes of the stored BF16 value."""
    row_start = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    output_type = output_ptr.type.element_ty
    scale = tl.load(q_scale_ptr)
    amax = 0.0
    mask = col_offsets < n_cols
    for row_idx in tl.range(row_start, n_rows, NUM_PRGMS, num_stages=2):
        input_ptrs = input_ptr + row_idx * input_row_stride + col_offsets
        input_ptrs = tl.multiple_of(input_ptrs, (16, ))
        row = tl.load(input_ptrs, mask=mask, other=0.0, cache_modifier=".cg").to(tl.float32)
        g = tl.load(g_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
        row_norm = row * row
        row_norm = tl.sum(row_norm, axis=-1)
        norm_factor = tl.math.rsqrt((row_norm / n_cols) + epsilon)

        # Store rsigma (norm_factor)
        rsigma_output_ptr = rsigma_ptr + row_idx
        tl.store(rsigma_output_ptr, norm_factor)

        if (ZERO_CENTERED_GAMMA):
            g += 1
        rms_norm = row * norm_factor * g

        output_ptrs = output_ptr + row_idx * output_row_stride + col_offsets
        output_ptrs = tl.multiple_of(output_ptrs, (16, ))
        out_bf16 = rms_norm.to(output_type)
        tl.store(output_ptrs, out_bf16, mask=mask)
        # --- fix 8a: TE's unary_kernel cast of that BF16 value, bytes packed in pairs behind a reshape (a 1-byte
        # store would change the row layout and with it the row_norm reduction tree)
        v = out_bf16.to(tl.float32)
        amax_temp = tl.max(tl.abs(v), axis=-1)
        amax = tl.maximum(amax, amax_temp)
        b2 = tl.reshape(_fnuz_te(v, scale), [BLOCK_SIZE // 2, 2])
        lo, hi = tl.split(b2)
        packed = lo.to(tl.uint16) | (hi.to(tl.uint16) << 8)
        cols2 = tl.arange(0, BLOCK_SIZE // 2)
        tl.store(f8_ptr.to(tl.pointer_type(tl.uint16)) + (row_idx * n_cols) // 2 + cols2, packed,
                 mask=cols2 < n_cols // 2)
    mx = tl.maximum(amax, 0.0)
    mi = mx.to(tl.int32, bitcast=True)
    cur = tl.load(q_amax_ptr, volatile=True)
    tl.atomic_max(q_amax_ptr, mi, mask=mi > cur, sem="relaxed")
    tl.store(scale_inv_ptr, 1.0 / scale, mask=row_start == 0)


def rmsnorm_fwd_dual(inp, weight, eps, ln_out, quantizer, zero_centered_gamma, num_programs):
    """TE te_rmsnorm_fwd_triton(inp, weight, eps, ln_out, None, bf16, 0, zcg, autotune=False) at fix 13's grid,
    plus the e4m3 bytes / scale_inv of `quantizer` applied to ln_out (amax max-accumulated in place).
    Returns (ln_out, rsigma, f8 uint8 [M, N], scale_inv [1])."""
    assert inp.dim() == 2 and inp.dtype == torch.bfloat16 and inp.is_contiguous()
    M, N = inp.shape
    assert ln_out.shape == (M, N) and ln_out.dtype == torch.bfloat16 and ln_out.is_contiguous()
    assert not use_blocked(inp) and N % 2 == 0
    rsigma = torch.empty((M,), dtype=torch.float32, device=inp.device)
    f8 = torch.empty((M, N), dtype=torch.uint8, device=inp.device)
    sinv = torch.empty(1, dtype=torch.float32, device=inp.device)
    NUM_PRGMS = min(M, num_programs)
    _rmsnorm_fwd_dual_kernel[(NUM_PRGMS,)](
        ln_out, inp, weight, rsigma, f8, quantizer.scale, quantizer.amax.view(torch.int32), sinv,
        inp.stride(0), ln_out.stride(0), M, N, eps, zero_centered_gamma, block_size(inp), NUM_PRGMS)
    return ln_out, rsigma, f8, sinv


def _eligible(b):
    """b: the bound arguments of _LayerNormLinear.forward."""
    a = b.arguments
    q = a.get("input_quantizer")
    if not a.get("fp8") or a.get("fp8_calibration") or type(q) is not Float8Quantizer:
        return None
    if q.dtype != tex.DType.kFloat8E4M3 or a.get("normalization") != "RMSNorm" or a.get("ln_bias") is not None:
        return None
    if not a.get("return_layernorm_output") or a.get("return_layernorm_output_gathered"):
        return None
    if a.get("parallel_mode") == "column" and a.get("sequence_parallel"):
        return None
    if a.get("ub_overlap_ag_fprop") or a.get("ub_bulk_dgrad") or a.get("ub_overlap_rs_dgrad"):
        return None
    if a.get("activation_dtype") != torch.bfloat16 or a["weight"].requires_grad:
        return None
    return q


def _install():
    assert _grid.ENABLED, "fix 8a needs HS_RMSNORM_GRID=1 (fix 13): it copies that pinned norm launch"
    _LNL = lnl_mod._LayerNormLinear
    orig_fwd = _LNL.forward
    sig = inspect.signature(orig_fwd)

    def lnl_forward(ctx, *args, **kwargs):
        _tls.dual = None
        if _ENABLED:
            try:
                b = sig.bind(ctx, *args, **kwargs)
            except TypeError:
                b = None
            if b is not None:
                _tls.dual = _eligible(b)
        try:
            return orig_fwd(ctx, *args, **kwargs)
        finally:
            _tls.dual = None
            _tls.pre = None

    _LNL.forward = staticmethod(lnl_forward)

    orig_apply_norm = lnl_mod.apply_normalization

    def apply_normalization(inputmat, ln_out, ln_weight, ln_bias, eps, output_quantizer, output_dtype,
                            normalization, fwd_ln_sm_margin, zero_centered_gamma):
        q = getattr(_tls, "dual", None)
        if (q is None or output_quantizer is not None or normalization != "RMSNorm" or ln_bias is not None
                or not isinstance(ln_out, torch.Tensor) or isinstance(ln_out, Float8Tensor)
                or ln_out.dtype != torch.bfloat16 or inputmat.dtype != torch.bfloat16
                or not inputmat.is_contiguous() or use_blocked(inputmat) or inputmat.shape[1] % 2):
            if q is not None:
                STATS["skipped"] += 1
            return orig_apply_norm(inputmat, ln_out, ln_weight, ln_bias, eps, output_quantizer, output_dtype,
                                   normalization, fwd_ln_sm_margin, zero_centered_gamma)
        _tls.dual = None
        ln_out, rsigma, f8, sinv = rmsnorm_fwd_dual(inputmat, ln_weight, eps, ln_out, q, zero_centered_gamma,
                                                    _grid.GRID_BF16)
        f8t = Float8Tensor(shape=tuple(ln_out.shape), dtype=ln_out.dtype, data=f8, fp8_scale_inv=sinv,
                           fp8_dtype=q.dtype, requires_grad=False, data_transpose=None, quantizer=q)
        _tls.pre = (ln_out, f8t)
        STATS["dual"] += 1
        return ln_out, None, rsigma

    lnl_mod.apply_normalization = apply_normalization

    orig_quantize = Float8Quantizer.quantize   # inherited from Quantizer

    def quantize(self, tensor, *, out=None):
        pre = getattr(_tls, "pre", None)
        if pre is not None and out is None and tensor is pre[0] and pre[1]._quantizer is self:
            _tls.pre = None
            STATS["handed"] += 1
            return pre[1]
        return orig_quantize(self, tensor, out=out)

    Float8Quantizer.quantize = quantize


_install()
print(f"[hyperspindle] hs_qkvnorm_e4m3: {'on' if _ENABLED else 'off'} (qkv RMSNorm writes BF16 + e4m3 + amax in "
      f"one pass; HS_QKVNORM_E4M3=1 enables)", flush=True)


def report():
    return f"[hyperspindle] hs_qkvnorm_e4m3: {STATS}"
