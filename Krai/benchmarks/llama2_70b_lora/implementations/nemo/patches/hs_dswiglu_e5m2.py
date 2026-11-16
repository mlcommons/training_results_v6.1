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

"""Fix 3a: fused SwiGLU backward that writes the fc1-output gradient directly as FP8 e5m2 (FNUZ on gfx942) with the
fc1 module's delayed-scaling grad_output quantizer, plus its amax and scale_inv. Replaces, per non-recomputed MLP
backward, fix #1's BF16 dx store (`_dswiglu_fp8in_kernel`) AND TE's `unary_kernel<..., te_hip_fp8_e5m2>` cast that
`_LayerNormLinear.backward` runs on it (grad_output_preprocess skips the cast for a QuantizedTensor).

Bit-exact with that chain by construction (microbenchmark lane l0_swiglu_bwd, campaign kfix_261010; exhaustive over
all 65,536 BF16 patterns x 11 scales and the full-shape exactness sweep): fp32 math exactly as fix #1 -> RNE to BF16
-> fp32 -> x scale -> clamp to +-57344 if finite -> hardware v_cvt_pk_bf8_f32 (Triton's fp8e5b16 cast, same
instruction); amax = max |BF16 dx| (fmaxf semantics: NaN ignored, all-NaN -> 0) folded into quantizer.amax with an
atomic max on the fp32 bits, as TE's atomicMaxFloat; scale_inv = 1/scale correctly rounded (TE: __frcp_rn).

VALU cut that keeps exactness: y1 is decoded from an e4m3fn byte, so dsilu(y1) and silu(y1) take 256 values: a
256-entry table built on the GPU by fix #1's own jit helpers (same compile flags), gathered through LDS (tl.gather).

Configuration = the lane's sweet spot (BLOCK 4096, 4 warps, LUT via tl.gather, check-then-atomic amax, Triton
native fp8 cast), except BF16 rounding uses Triton's conversion (RND 0) instead of the integer shortcut (RND 1):
~2% slower, no assumption on NaN payloads.
"""

import torch
import triton
import triton.language as tl

from hs_swiglu_fp8_triton import _e4m3fn_bits_to_fp32, _sigmoid

BLOCK = 4096
NUM_WARPS = 4
E5M2_MAX = 57344.0


@triton.jit
def _lut_build_kernel(lut_d_ptr, lut_s_ptr):
    """256 e4m3fn codes -> dsilu, silu with fix #1's exact math (fix #1: dsilu = fma(y1, s(1-s), s), silu = y1*s)."""
    c = tl.arange(0, 256)
    y1 = _e4m3fn_bits_to_fp32(c.to(tl.uint8))
    s = _sigmoid(y1)
    tl.store(lut_d_ptr + c, tl.math.fma(y1, s * (1.0 - s), s))
    tl.store(lut_s_ptr + c, y1 * s)


@triton.jit
def _e5m2_bits(f, scale):
    """fp32 (holding a BF16 value) -> e5m2fnuz byte exactly as TE's unary_kernel stores te_hip_fp8_e5m2(f * scale):
    finite values clamped to +-57344 (TE: fmed3), inf/NaN passed to the hardware convert (gives 0x80)."""
    v = f * scale
    u = v.to(tl.int32, bitcast=True)
    finite = (u & 0x7F800000) != 0x7F800000
    vc = tl.where(finite, tl.minimum(tl.maximum(v, -57344.0), 57344.0), v)
    return vc.to(tl.float8e5b16).to(tl.uint8, bitcast=True)


@triton.jit
def _dswiglu_e5m2_kernel(g_ptr, q_ptr, lut_d_ptr, lut_s_ptr, f8_ptr, scale_ptr, amax_ptr, sinv_ptr, H,
                         BLOCK: tl.constexpr):
    """One program: row pid0, columns [pid1*BLOCK, +BLOCK) of both halves. g: BF16 [M, H]; q: e4m3fn bytes of the
    SwiGLU input [M, 2H] (fix #1's act-store); f8: e5m2fnuz bytes of dx [M, 2H]."""
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    row = pid0.to(tl.int64)
    col = pid1 * BLOCK + tl.arange(0, BLOCK)
    mask = col < H
    q_row = q_ptr + row * (2 * H)
    b1 = tl.load(q_row + col, mask=mask, other=0)
    y2 = _e4m3fn_bits_to_fp32(tl.load(q_row + H + col, mask=mask, other=0))
    g = tl.load(g_ptr + row * H + col, mask=mask, other=0.0).to(tl.float32)

    c = tl.arange(0, 256)
    idx = b1.to(tl.int32)
    dsilu = tl.gather(tl.load(lut_d_ptr + c), idx, 0)
    silu = tl.gather(tl.load(lut_s_ptr + c), idx, 0)
    # fix #1's arithmetic, same order: dx1 = (dsilu * g) * y2, dx2 = g * silu
    f1 = (dsilu * g * y2).to(tl.bfloat16).to(tl.float32)
    f2 = (g * silu).to(tl.bfloat16).to(tl.float32)

    scale = tl.load(scale_ptr)
    row2 = row * (2 * H)
    tl.store(f8_ptr + row2 + col, _e5m2_bits(f1, scale), mask=mask)
    tl.store(f8_ptr + row2 + H + col, _e5m2_bits(f2, scale), mask=mask)

    # amax of |BF16 dx|: masked lanes are 0; maximum ignores NaN like fmaxf; the final max with 0 maps an all-NaN
    # program to 0 as TE's `max = 0; fmaxf(...)` does. Skip the atomic when it cannot raise the current value.
    m = tl.maximum(tl.maximum(tl.max(tl.abs(f1), axis=0), tl.max(tl.abs(f2), axis=0)), 0.0)
    mi = m.to(tl.int32, bitcast=True)
    cur = tl.load(amax_ptr, volatile=True)
    tl.atomic_max(amax_ptr, mi, mask=mi > cur, sem="relaxed")
    tl.store(sinv_ptr, 1.0 / scale, mask=(pid0 == 0) & (pid1 == 0))


_LUT = {}


def _lut(device):
    key = str(device)
    if key not in _LUT:
        d = torch.empty(256, dtype=torch.float32, device=device)
        s = torch.empty(256, dtype=torch.float32, device=device)
        _lut_build_kernel[(1,)](d, s, num_warps=4, enable_fp_fusion=False)
        _LUT[key] = (d, s)
    return _LUT[key]


def dswiglu_e5m2_into(grad_out: torch.Tensor, x_fp8: torch.Tensor, data: torch.Tensor, scale: torch.Tensor,
                      amax: torch.Tensor, scale_inv: torch.Tensor) -> None:
    """grad_out [M, H] BF16, x_fp8 [M, 2H] float8_e4m3fn (fix #1's act-store) -> data [M, 2H] uint8 e5m2fnuz bytes
    of dswiglu (== TE Float8Quantizer(scale, amax, kFloat8E5M2) applied to fix #1's BF16 dx), amax (float32 [1],
    max-accumulated in place), scale_inv (float32 [1]) = 1/scale."""
    assert x_fp8.dtype == torch.float8_e4m3fn and x_fp8.dim() == 2 and x_fp8.is_contiguous()
    M, H2 = x_fp8.shape
    H = H2 // 2
    g = grad_out.contiguous()
    assert g.dtype == torch.bfloat16 and g.shape == (M, H), (g.dtype, g.shape)
    assert data.dtype == torch.uint8 and data.shape == (M, H2) and data.is_contiguous()
    assert scale.dtype == amax.dtype == scale_inv.dtype == torch.float32
    if M == 0 or H == 0:
        return
    lut_d, lut_s = _lut(g.device)
    _dswiglu_e5m2_kernel[(M, triton.cdiv(H, BLOCK))](
        g, x_fp8.view(torch.uint8), lut_d, lut_s, data, scale, amax.view(torch.int32), scale_inv, H,
        BLOCK=BLOCK, num_warps=NUM_WARPS, enable_fp_fusion=False)
