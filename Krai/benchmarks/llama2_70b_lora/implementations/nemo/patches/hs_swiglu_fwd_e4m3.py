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

"""Fix 3b: fused SwiGLU forward that writes fc2's input directly as FP8 e4m3 (FNUZ on gfx942) with the fc2 module's
delayed-scaling input quantizer, plus its amax and scale_inv, and fix #1's e4m3fn act-store bytes. Replaces, per
stored MLP forward, fix #1's `_swiglu_fwd_store_kernel` (BF16 out + act-store) AND TE's `unary_kernel<..., e4m3>` cast
that `_Linear.forward` runs on that BF16 out (skipped for a QuantizedTensor input). The BF16 out is never written: its
only consumer is fc2 (LoRA targets attention; no MLP adapter; fc2's frozen weight saves no input for wgrad).

Bit-exact with that chain by construction (microbenchmark lane l1_swiglu_fwd, candidate F5, campaign kfix_261010;
exhaustive over all 65,536 BF16 patterns for the conversions and the sigmoid, full-shape exactness sweep):
  out    = (y1 * sigmoid(y1)) * y2 in fp32 as fix #1 / TE gated_act, RNE to BF16 (never stored)
  f8     = TE's quantize of that BF16 value: v * scale, clamp finite values to +-240 (SATFINITE, v_med3), hardware
           v_cvt_pk_fp8_f32 (Triton's fp32 -> float8e4b8, same instruction)
  amax   = max |BF16 out| (NaN ignored, as fmaxf), atomic max on the fp32 bits into quantizer.amax (TE atomicMaxFloat)
  sinv   = 1/scale correctly rounded (TE: __frcp_rn), written by program 0
  q      = x.to(torch.float8_e4m3fn) bytes (fix #1's act-store) through the FNUZ converter: the OCP e4m3fn grid is
           the FNUZ grid times 2, so bits_ocp(x) = cvt_fnuz(x / 2) for |x| < 480, 0x7f beyond and for inf/NaN, sign ORed
  sigmoid without the IEEE division sequence: ds = fma(e, 2^-32, 2^-32) = 2^-32 * (1 + e) rounded once; hardware
           rcp + one Newton step + Markstein correction; * 2^-32; the exp overflow clamp applied to the result
           (identical to 1.f / (1.f + expf(-x)) on every BF16 input; plain rcp + Newton fails on 3, no clamps on 8,281)

Configuration = the lane's sweet spot: BLOCK 2048 x ROWS 4 per program, 4 warps, waves_per_eu 2, one atomic per
program. Measured 577-596 us vs fix #1 alone 677-681 us and vs fix #1 + TE cast 874-883 us at [8192, 57344].
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

BLOCK, ROWS, NUM_WARPS, WAVES_PER_EU = 2048, 4, 4, 2


@triton.jit
def _sigmoid_scaled(x):
    a = -x
    ph = a * 1.4426950216293335
    pl = tl.math.fma(a, 1.4426950216293335, -ph)
    pl = tl.math.fma(a, 1.925962855864327e-08, pl)
    e = libdevice.nearbyint(ph)
    r = (ph - e) + pl
    v = tl.exp2(r)
    v = libdevice.ldexp(v, e.to(tl.int32))
    v = tl.where(a < -103.2789306640625, 0.0, v)
    ds = tl.math.fma(v, 2.3283064365386963e-10, 2.3283064365386963e-10)
    y = tl.inline_asm_elementwise(asm="v_rcp_f32 $0, $1", constraints="=v,v", args=[ds], dtype=tl.float32,
                                  is_pure=True, pack=1)
    rr = tl.math.fma(-ds, y, 1.0)
    y = tl.math.fma(rr, y, y)
    rr = tl.math.fma(-ds, y, 1.0)
    q = tl.math.fma(rr, y, y)
    return tl.where(a > 88.72283935546875, 0.0, q * 2.3283064365386963e-10)


@triton.jit
def _fnuz_te(v, s):
    """TE's static_cast<te_hip_fp8_e4m3>(v * s) on gfx942: SATFINITE clamp (v_med3_f32) then hardware RNE."""
    t = v * s
    lo = tl.full(t.shape, -240.0, tl.float32)
    hi = tl.full(t.shape, 240.0, tl.float32)
    c = tl.inline_asm_elementwise(asm="v_med3_f32 $0, $1, $2, $3", constraints="=v,v,v,v", args=[t, lo, hi],
                                  dtype=tl.float32, is_pure=True, pack=1)
    t = tl.where(tl.abs(t) < float("inf"), c, t)
    return t.to(tl.float8e4b8).to(tl.uint8, bitcast=True)


@triton.jit
def _ocp_bits(x):
    """torch x.to(float8_e4m3fn) byte from the FNUZ hardware conversion of x / 2."""
    b = (x * 0.5).to(tl.float8e4b8).to(tl.uint8, bitcast=True).to(tl.int32)
    sign = (x.to(tl.int32, bitcast=True) >> 24) & 0x80
    b = tl.where(tl.abs(x) < 480.0, b, 0x7F)
    return (b | sign).to(tl.uint8)


@triton.jit
def _swiglu_fwd_e4m3_kernel(x_ptr, q_ptr, f8_ptr, scale_ptr, amax_ptr, sinv_ptr, M, H,
                            BLOCK: tl.constexpr, ROWS: tl.constexpr):
    """Program (pid0, pid1): rows [pid0*ROWS, +ROWS), columns [pid1*BLOCK, +BLOCK) of y1 / y2. x, q rows are 2H,
    f8 rows H."""
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    col = pid1 * BLOCK + tl.arange(0, BLOCK)
    s = tl.load(scale_ptr)
    m = tl.zeros([BLOCK], dtype=tl.float32)
    for i in tl.static_range(ROWS):
        row = (pid0 * ROWS + i).to(tl.int64)
        mask = (col < H) & (row < M)
        x_row = x_ptr + row * (2 * H)
        y1 = tl.load(x_row + col, mask=mask, other=0.0).to(tl.float32)
        y2 = tl.load(x_row + H + col, mask=mask, other=0.0).to(tl.float32)
        v = ((y1 * _sigmoid_scaled(y1)) * y2).to(tl.bfloat16).to(tl.float32)
        m = tl.maximum(m, tl.abs(v))
        tl.store(f8_ptr + row * H + col, _fnuz_te(v, s), mask=mask)
        q_row = q_ptr + row * (2 * H)
        tl.store(q_row + col, _ocp_bits(y1), mask=mask)
        tl.store(q_row + H + col, _ocp_bits(y2), mask=mask)
    mx = tl.maximum(tl.max(m, axis=0), 0.0)
    tl.atomic_max(amax_ptr, mx.to(tl.int32, bitcast=True), sem="relaxed")
    tl.store(sinv_ptr, 1.0 / s, mask=(pid0 == 0) & (pid1 == 0))


def swiglu_fwd_e4m3_into(x: torch.Tensor, q: torch.Tensor, f8: torch.Tensor, scale: torch.Tensor,
                         amax: torch.Tensor, scale_inv: torch.Tensor) -> None:
    """x [M, 2H] BF16 -> q [M, 2H] uint8 (== x.to(float8_e4m3fn) bytes), f8 [M, H] uint8 e4m3fnuz bytes of
    swiglu(x) (== TE Float8Quantizer(scale, amax, kFloat8E4M3) applied to fix #1's BF16 out), amax (float32 [1],
    max-accumulated in place), scale_inv (float32 [1]) = 1/scale."""
    assert x.dtype == torch.bfloat16 and x.dim() == 2 and x.is_contiguous(), (x.dtype, x.shape, x.stride())
    M, H2 = x.shape
    H = H2 // 2
    assert q.dtype == torch.uint8 and q.shape == (M, H2) and q.is_contiguous()
    assert f8.dtype == torch.uint8 and f8.shape == (M, H) and f8.is_contiguous()
    assert scale.dtype == amax.dtype == scale_inv.dtype == torch.float32
    if M == 0 or H == 0:
        return
    _swiglu_fwd_e4m3_kernel[(triton.cdiv(M, ROWS), triton.cdiv(H, BLOCK))](
        x, q, f8, scale, amax.view(torch.int32), scale_inv, M, H, BLOCK=BLOCK, ROWS=ROWS, num_warps=NUM_WARPS,
        waves_per_eu=WAVES_PER_EU, enable_fp_fusion=False)
