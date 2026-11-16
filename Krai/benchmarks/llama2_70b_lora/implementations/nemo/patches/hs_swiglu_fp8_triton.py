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

"""Triton kernels fusing the FP8 (e4m3fn) activation store into SwiGLU (MI300X / gfx942).

Used by hs_act_store.py level 3 (HS_ACT_STORE_FUSED=1). Replaces, per MLP layer, the four kernels of the
level-1 path with two:

  forward   x.to(float8_e4m3fn) + TE gated_act     ->  swiglu_fwd_store(x)        (one pass over x)
  backward  x_fp8.to(bf16)      + tex.dswiglu      ->  dswiglu_fp8in(g, x_fp8)    (dequantize in registers)

Conventions (Megatron fused_bias_swiglu with USE_TE_SWIGLU=1): x is [M, 2H] BF16, y1, y2 = x.chunk(2, -1),
out = silu(y1) * y2 is [M, H] BF16, fp32 math inside the kernels with TE's exact formula and operation order
(transformer_engine/common/util/math.h and vectorized_pointwise.h):

  s      = 1 / (1 + exp(-y1))
  silu   = y1 * s                        out  = (y1 * s) * y2
  dsilu  = y1 * (s * (1 - s)) + s        dx1  = dsilu * g * y2        dx2 = g * (y1 * s)

The stored tensor is bit-identical to torch's `x.to(torch.float8_e4m3fn)` (c10 Float8_e4m3fn.h), which is
NOT a saturating conversion: |x| <= 464 rounds (RNE) to at most 448, |x| > 464 and +-inf give NaN (0x7f / 0xff
with the sign). Triton's own `.to(tl.float8e4nv)` saturates differently on gfx942, so the conversion is done
on the fp32 bit pattern (see _fp32_to_e4m3fn_bits). The dequantize goes through fp16 bits and the hardware
fp16 -> fp32 conversion (see _e4m3fn_bits_to_fp32): Triton's fp8 -> fp32 upcast is exact too but lowers to
~40 instructions per element on gfx942, which made the backward ALU-bound (0.80 ms vs 0.64 ms per call).

Measured on one MI300X, [8192, 57344] (see test_hs_swiglu_fp8.py --bench): forward 0.65 ms vs cast 0.35 +
gated_act 0.45 ms; backward 0.64 ms vs dequant 1.07 + dswiglu 0.69 ms. Both kernels are at ~2.9 TB/s of
HBM traffic (memory floor for the same tile shape ~0.55 ms); the remaining gap is the e4m3 conversions and
the precise exp/div, kept for bit-fidelity with torch and TE.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice  # __ocml_* device-library functions on AMD

# Tile: one program handles BLOCK columns of y1 and the matching BLOCK columns of y2 of one row.
# Tuned on MI300X for [8192, 57344] (H = 28672 = 7 x 4096); masks handle any H.
FWD_BLOCK, FWD_WARPS = 2048, 4
BWD_BLOCK, BWD_WARPS = 2048, 4

# fp32 bit-pattern constants of the c10 conversion (all fit in int32; Triton integer math wraps like uint32).
# tl.constexpr instances so that the jit'ed helpers can read them as globals.
_E4M3_NAN_THRESHOLD = tl.constexpr(0x43F00000)  # bits of 480.0: |x| >= 480 -> NaN in c10 (464 < |x| < 480 too, via rounding)
_E4M3_MIN_NORMAL = tl.constexpr(0x3C800000)  # bits of 2^-6, the smallest e4m3 normal; below it the subnormal path
# normal path: f_bits + ((7 - 127) << 23) + 0x7FFFF (+ mant_odd) then >> 20 : rebias the exponent to the
# e4m3 bias 7 and round-to-nearest-even the 20 dropped mantissa bits (as uint32: 0xC407FFFF)
_E4M3_NORMAL_BIAS = tl.constexpr(((7 - 127) << 23) + 0x7FFFF)
# subnormal path: |x| + 2^14 in fp32 has ulp 2^-9 = the e4m3 subnormal step, so the fp32 adder does the RNE;
# the low bits of the sum minus bits(2^14) are the e4m3 code.
_E4M3_DENORM_MAGIC = tl.constexpr(141 << 23)  # bits of 2^14 = (127 + 14) << 23


@triton.jit
def _fp32_to_e4m3fn_bits(x):
    """fp32 tensor -> int32 tensor holding the e4m3fn byte, bit-identical to torch's Float8_e4m3fn(float)."""
    u = x.to(tl.int32, bitcast=True)
    neg = u < 0
    a = u & 0x7FFFFFFF  # |x| bits
    # normal range (|x| >= 2^-6), also used for the overflow range where the result lands on 0x7f
    mant_odd = (a >> 20) & 1
    n = ((a + (_E4M3_NORMAL_BIAS + mant_odd)) >> 20) & 0xFF
    # subnormal range (|x| < 2^-6), including zero
    x_abs = a.to(tl.float32, bitcast=True)
    s = ((x_abs + 16384.0).to(tl.int32, bitcast=True) - _E4M3_DENORM_MAGIC) & 0xFF
    r = tl.where(a < _E4M3_MIN_NORMAL, s, n)
    r = tl.where(a >= _E4M3_NAN_THRESHOLD, 0x7F, r)  # inf, nan and |x| >= 480
    return r | tl.where(neg, 0x80, 0)


@triton.jit
def _e4m3fn_bits_to_fp32(b):
    """uint8 e4m3fn code -> fp32, exact (the unit test checks all 256 codes).

    Placing the 4 exponent + 3 mantissa bits at the top of an fp16's exponent/mantissa fields (code << 7) gives
    an fp16 whose value is exactly value_e4m3 * 2^-8 for normals (fp16 bias 15 vs e4m3 bias 7) AND for
    subnormals (fp16 subnormal step 2^-24 = 2^-9 * 2^-8 / 2^7 per shifted mantissa bit); the fp16 -> fp32
    conversion is a hardware instruction and the * 256 is exact. Only the NaN code (exponent and mantissa all
    ones, which fp16 reads as 1.875) needs a select. The sign moves separately to fp16 bit 15."""
    bi = b.to(tl.int32)
    h = ((bi & 0x80) << 8) | ((bi & 0x7F) << 7)
    v = h.to(tl.int16).to(tl.float16, bitcast=True).to(tl.float32) * 256.0
    return tl.where((bi & 0x7F) == 0x7F, float("nan"), v)


@triton.jit
def _exp_hip(a):
    """expf exactly as hipcc / LLVM lower it on gfx942 (what TE's gated_act and torch.exp run): two-constant
    range reduction x*log2e with the product error recovered by FMAs, round-to-nearest-even, hardware exp2,
    ldexp, then the under/overflow clamps. Neither tl.exp (exp2(x*log2e), 1-2 ulp off) nor the __ocml_exp_f32
    that Triton links reproduces it. The kernels are launched with enable_fp_fusion=False: LLVM would otherwise
    contract `ph - e` into fma(a, log2e, -e), which hipcc's lowering does not do (verified bit-for-bit against
    torch.exp on 4 x 16M inputs incl. the clamp ranges)."""
    ph = a * 1.4426950216293335                      # 0x3fb8aa3b, log2(e) high part
    pl = tl.math.fma(a, 1.4426950216293335, -ph)      # exact error of the product
    pl = tl.math.fma(a, 1.925962855864327e-08, pl)    # 0x32a5705f, log2(e) low part
    e = libdevice.nearbyint(ph)                      # v_rndne_f32
    r = (ph - e) + pl
    v = tl.exp2(r)                                   # v_exp_f32
    v = libdevice.ldexp(v, e.to(tl.int32))
    v = tl.where(a < -103.2789306640625, 0.0, v)     # 0x42ce8ed0: exp underflows to 0
    v = tl.where(a > 88.72283935546875, float("inf"), v)   # 0x42b17218: overflow
    return v


@triton.jit
def _sigmoid(x):
    """TE's sigmoid<float,float>: 1.f / (1.f + expf(-x)); the division is correctly rounded in both (hipcc default
    -fhip-fp32-correctly-rounded-divide-sqrt; Triton's fp32 `/` emits the same v_div_scale/fmas/fixup sequence)."""
    return 1.0 / (1.0 + _exp_hip(-x))


@triton.jit
def _swiglu_fwd_store_kernel(x_ptr, out_ptr, q_ptr, H, BLOCK: tl.constexpr):
    """One program: row `pid0`, columns [pid1*BLOCK, +BLOCK) of y1 and y2. Writes out (bf16) and the e4m3
    bytes of both halves (uint8). x and q rows are 2H contiguous, out rows H contiguous."""
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = col < H
    x_row = x_ptr + row * (2 * H)
    y1 = tl.load(x_row + col, mask=mask, other=0.0).to(tl.float32)
    y2 = tl.load(x_row + H + col, mask=mask, other=0.0).to(tl.float32)

    out = (y1 * _sigmoid(y1)) * y2
    tl.store(out_ptr + row * H + col, out.to(tl.bfloat16), mask=mask)

    q_row = q_ptr + row * (2 * H)
    tl.store(q_row + col, _fp32_to_e4m3fn_bits(y1).to(tl.uint8), mask=mask)
    tl.store(q_row + H + col, _fp32_to_e4m3fn_bits(y2).to(tl.uint8), mask=mask)


@triton.jit
def _dswiglu_fp8in_kernel(g_ptr, q_ptr, dx_ptr, H, BLOCK: tl.constexpr):
    """One program: row `pid0`, columns [pid1*BLOCK, +BLOCK). Reads g (bf16, rows of H) and the e4m3 bytes of
    y1 / y2 (rows of 2H), writes dx (bf16, rows of 2H) = [dx1 | dx2]."""
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = col < H
    q_row = q_ptr + row * (2 * H)
    y1 = _e4m3fn_bits_to_fp32(tl.load(q_row + col, mask=mask, other=0))
    y2 = _e4m3fn_bits_to_fp32(tl.load(q_row + H + col, mask=mask, other=0))
    g = tl.load(g_ptr + row * H + col, mask=mask, other=0.0).to(tl.float32)

    s = _sigmoid(y1)
    # TE: dsilu = cval * dsigmoid(cval) + sigmoid(cval); hipcc contracts this one multiply-add into an FMA
    # (fp-contract=fast). With fusion disabled for the kernel, write that FMA explicitly; nothing else in these
    # kernels is contractible (products only).
    dsilu = tl.math.fma(y1, s * (1.0 - s), s)
    dx1 = dsilu * g * y2
    dx2 = g * (y1 * s)

    dx_row = dx_ptr + row * (2 * H)
    tl.store(dx_row + col, dx1.to(tl.bfloat16), mask=mask)
    tl.store(dx_row + H + col, dx2.to(tl.bfloat16), mask=mask)


def swiglu_fwd_store(x: torch.Tensor):
    """x [M, 2H] BF16 contiguous -> (out [M, H] BF16, x_fp8 [M, 2H] torch.float8_e4m3fn == x.to(float8_e4m3fn))."""
    assert x.dtype == torch.bfloat16 and x.dim() == 2 and x.is_contiguous(), (x.dtype, x.shape, x.stride())
    M, H2 = x.shape
    assert H2 % 2 == 0
    H = H2 // 2
    out = torch.empty((M, H), dtype=torch.bfloat16, device=x.device)
    q = torch.empty((M, H2), dtype=torch.uint8, device=x.device)
    if M > 0 and H > 0:
        grid = (M, triton.cdiv(H, FWD_BLOCK))
        _swiglu_fwd_store_kernel[grid](x, out, q, H, BLOCK=FWD_BLOCK, num_warps=FWD_WARPS, enable_fp_fusion=False)
    return out, q.view(torch.float8_e4m3fn)


def dswiglu_fp8in(grad_out: torch.Tensor, x_fp8: torch.Tensor):
    """grad_out [M, H] BF16, x_fp8 [M, 2H] float8_e4m3fn -> dx [M, 2H] BF16 (== tex.dswiglu(g, x_fp8.to(bf16)))."""
    assert x_fp8.dtype == torch.float8_e4m3fn and x_fp8.dim() == 2 and x_fp8.is_contiguous()
    M, H2 = x_fp8.shape
    H = H2 // 2
    assert grad_out.dtype == torch.bfloat16 and grad_out.shape == (M, H), (grad_out.dtype, grad_out.shape)
    g = grad_out.contiguous()
    dx = torch.empty((M, H2), dtype=torch.bfloat16, device=g.device)
    if M > 0 and H > 0:
        grid = (M, triton.cdiv(H, BWD_BLOCK))
        _dswiglu_fp8in_kernel[grid](g, x_fp8.view(torch.uint8), dx, H, BLOCK=BWD_BLOCK, num_warps=BWD_WARPS, enable_fp_fusion=False)
    return dx
