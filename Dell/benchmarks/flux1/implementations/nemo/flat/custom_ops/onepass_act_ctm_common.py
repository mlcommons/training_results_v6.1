# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Device helpers shared by the B4 chain-1 DSL kernels.

The bitwise-contract helpers (proved 45/45 on two nodes) plus the two staging helpers the ``stsm`` variants need:
``stsm_x2`` (``stmatrix.sync.aligned.x2.m8n8.shared.b16``) and ``sts32``.

Every instruction whose identity is the bitwise contract is emitted as the
exact PTX through ``cute.arch.inline_ptx`` (or ``llvm.inline_asm`` for the
``mma`` whose zero accumulators LLVM would otherwise fold into immediates).
TE citations for each numeric step are in
``flat/custom_ops/onepass_act_tma_kernel.cu`` and
``flat/custom_ops/dual_cast_one_read_kernel.cu`` and are not repeated.

These helpers are *traced*, not preprocessed: plain Python loops only, no
``cutlass.range_constexpr``, no dynamic ``if`` (use ``cutlass.select_``).
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cutlass_dsl import T

TILE_COLS = 128
BOX_COLS = 64                          # 128 B of bf16: the SWIZZLE_128B span
_BF16_HI_MASK = -65536                 # 0xFFFF0000 as a signed Int32
FLT_MAX = 3.4028234663852886e38
E4M3_MAX_NORM_RCP = 1.0 / 448.0
FP4_MAX_INV = 1.0 / 6.0
INT32_MAX = 2**31 - 1

# --- PTX, verbatim from the CUDA kernels --------------------------------------

_LDMATRIX_X4_TRANS = (
    "ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 "
    "{ {$w0}, {$w1}, {$w2}, {$w3} }, [{$r0}];"
)
_MMA_16816_BF16 = (
    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
    "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};"
)
_MMA_CONSTRAINTS = "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f"
_MAX_XORSIGN_ABS_BF16X2 = "max.xorsign.abs.bf16x2 {$w0}, {$r0}, {$r1};"
_PRMT = "prmt.b32 {$w0}, {$r0}, {$r1}, {$r2};"
_PACK_BF16X2 = "cvt.rn.bf16x2.f32 {$w0}, {$r0}, {$r1};"          # r0 = hi, r1 = lo
_DIV_RN = "div.rn.f32 {$w0}, {$r0}, {$r1};"
# Constant numerators live INSIDE the asm: a Float32 constant passed as an
# inline_ptx operand is folded to an f32 immediate that libNVVM refuses to print
# on 4.6.2; fixed in 4.8.0a0, written the safe way anyway.
_DIV_CONST_RN = "{ .reg .f32 c; mov.f32 c, 0f%08X; div.rn.f32 {$w0}, c, {$r0}; }"
_LDS128 = "ld.shared.v4.b32 { {$w0}, {$w1}, {$w2}, {$w3} }, [{$r0}];"
_STS128 = "st.shared.v4.b32 [{$r0}], { {$r1}, {$r2}, {$r3}, {$r4} };"
_LDS32 = "ld.shared.b32 {$w0}, [{$r0}];"
_STS32 = "st.shared.b32 [{$r0}], {$r1};"
_STS16 = "{ .reg .b16 h; cvt.u16.u32 h, {$r1}; st.shared.b16 [{$r0}], h; }"
_STS8 = "{ .reg .b16 h; cvt.u16.u32 h, {$r1}; st.shared.b8 [{$r0}], h; }"
# B4: two 8x8 b16 matrices; lanes 0-7 / 8-15 supply the 16 B row addresses of
# matrix 0 / 1, lane (gi, q) supplies element (gi, 2q..2q+1) of each as one b32.
_STSM_X2 = "stmatrix.sync.aligned.x2.m8n8.shared.b16 [{$r0}], { {$r1}, {$r2} };"
_STG256 = (
    "st.global.v8.b32 [{$r0}], "
    "{ {$r1}, {$r2}, {$r3}, {$r4}, {$r5}, {$r6}, {$r7}, {$r8} };"
)
# ptx.cuh:386-396 float_to_e8m0 fused with ptx.cuh:356-365 exp2f_rcp (076, verbatim).
_E8M0_AND_RCP = "\n".join([
    "{",
    ".reg .b16 e16;",
    ".reg .b32 e, sh;",
    ".reg .pred p;",
    "cvt.rp.satfinite.ue8m0x2.f32 e16, 0f00000000, {$r0};",
    "cvt.u32.u16 e, e16;",
    "and.b32 e, e, 255;",
    "mov.b32 {$w0}, e;",
    "mov.b32 sh, 254;",
    "sub.u32 sh, sh, e;",
    "shl.b32 sh, sh, 23;",
    "setp.eq.u32 p, e, 254;",
    "selp.b32 sh, 0x00400000, sh, p;",
    "setp.eq.u32 p, e, 255;",
    "selp.b32 sh, 0x7fffffff, sh, p;",
    "mov.b32 {$w1}, sh;",
    "}",
])
# ptx.cuh:1003-1023 mul_cvt_2x on two bf16x2 words -> four E4M3 bytes (076, verbatim).
_MUL_CVT_2X = "\n".join([
    "{",
    ".reg .b16 h0, h1, h2, h3, o0, o1;",
    ".reg .b32 f0, f1, f2, f3;",
    ".reg .b64 s2, p0, p1;",
    "mov.b64 s2, { {$r2}, {$r2} };",
    "mov.b32 {h0, h1}, {$r0};",
    "mov.b32 {h2, h3}, {$r1};",
    "cvt.f32.bf16 f0, h0;",
    "cvt.f32.bf16 f1, h1;",
    "cvt.f32.bf16 f2, h2;",
    "cvt.f32.bf16 f3, h3;",
    "mov.b64 p0, {f0, f1};",
    "mov.b64 p1, {f2, f3};",
    "mul.f32x2 p0, p0, s2;",
    "mul.f32x2 p1, p1, s2;",
    "mov.b64 {f0, f1}, p0;",
    "mov.b64 {f2, f3}, p1;",
    "cvt.rn.satfinite.e4m3x2.f32 o0, f1, f0;",
    "cvt.rn.satfinite.e4m3x2.f32 o1, f3, f2;",
    "mov.b32 {$w0}, {o0, o1};",
    "}",
])
# dual_cast_one_read_kernel.cu f32x2_to_e4m3x2: byte 0 = e4m3(lo), byte 1 = e4m3(hi).
_F32X2_TO_E4M3X2 = "\n".join([
    "{",
    ".reg .b16 h;",
    "cvt.rn.satfinite.e4m3x2.f32 h, {$r0}, {$r1};",     # r0 = hi, r1 = lo
    "cvt.u32.u16 {$w0}, h;",
    "}",
])
# dual_cast_one_read_kernel.cu e4m3_to_f32: cvt.rn.f16x2.e4m3x2 then the low half -> f32.
_E4M3_TO_F32 = "\n".join([
    "{",
    ".reg .b16 h, lo, hi;",
    ".reg .b32 p;",
    "cvt.u16.u32 h, {$r0};",
    "cvt.rn.f16x2.e4m3x2 p, h;",
    "mov.b32 {lo, hi}, p;",
    "cvt.f32.f16 {$w0}, lo;",
    "}",
])
# dual_cast_one_read_kernel.cu mul_cvt_bf16x2pairs_to_e2m1x4: two packed words ->
# four E2M1 codes in the low 16 bits ({b0, b1} = positions 0-1, 2-3). NOTE the
# high 16 bits DUPLICATE the low 16 ({b0, b1, b0, b1}); mask before packing.
_MUL_CVT_E2M1X4 = "\n".join([
    "{",
    ".reg .b16 h0, h1, h2, h3;",
    ".reg .b32 f0, f1, f2, f3;",
    ".reg .b64 A, B, S;",
    ".reg .b8 b0, b1;",
    "mov.b64 S, { {$r2}, {$r2} };",
    "mov.b32 {h0, h1}, {$r0};",
    "mov.b32 {h2, h3}, {$r1};",
    "cvt.f32.bf16 f0, h0;",
    "cvt.f32.bf16 f1, h1;",
    "cvt.f32.bf16 f2, h2;",
    "cvt.f32.bf16 f3, h3;",
    "mov.b64 A, {f0, f1};",
    "mov.b64 B, {f2, f3};",
    "mul.f32x2 A, A, S;",
    "mul.f32x2 B, B, S;",
    "mov.b64 {f0, f1}, A;",
    "mov.b64 {f2, f3}, B;",
    "cvt.rn.satfinite.e2m1x2.f32 b0, f1, f0;",
    "cvt.rn.satfinite.e2m1x2.f32 b1, f3, f2;",
    "mov.b32 {$w0}, {b0, b1, b0, b1};",
    "}",
])


# --- traced device helpers -------------------------------------------------------


def _ptx(code, outs, ins):
    return cute.arch.inline_ptx(code, write_only_types=outs, read_only_args=ins)


def ldmatrix_x4_trans(addr):
    """``ldmatrix.x4.trans`` from a 32-bit shared address: four Int32 fragments."""
    return _ptx(_LDMATRIX_X4_TRANS, [cutlass.Int32] * 4, [addr])


def mma_16816(a, b0, b1, c):
    """D = A(16x16 bf16, row) * B(16x8 bf16, col) + C, fp32 accumulate -- TE's instruction."""
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [v.ir_value() for v in (a[0], a[1], a[2], a[3], b0, b1, c[0], c[1], c[2], c[3])],
        _MMA_16816_BF16, _MMA_CONSTRAINTS,
        has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return tuple(cutlass.Float32(llvm.extractvalue(T.f32(), res, [i])) for i in range(4))


def xorsign_abs(a, b):
    return _ptx(_MAX_XORSIGN_ABS_BF16X2, [cutlass.Int32], [a, b])


def prmt(a, b, sel):
    return _ptx(_PRMT, [cutlass.Int32], [a, b, cutlass.Int32(sel)])


def pack_bf16x2(lo, hi):
    """``cvt.rn.bf16x2.f32``: lo in the low half, hi in the high half (RN, no satfinite)."""
    return _ptx(_PACK_BF16X2, [cutlass.Int32], [hi, lo])


def div_rn(a, b):
    """IEEE ``div.rn.f32`` of two RUNTIME values. Never pass a constant here; use :func:`div_const_rn`."""
    return _ptx(_DIV_RN, [cutlass.Float32], [a, b])


def div_const_rn(numerator: float, b):
    """IEEE ``div.rn.f32`` of a compile-time constant numerator by a runtime value."""
    import struct
    bits = struct.unpack("<I", struct.pack("<f", numerator))[0]
    return _ptx(_DIV_CONST_RN % bits, [cutlass.Float32], [b])


def lds128(addr):
    return _ptx(_LDS128, [cutlass.Int32] * 4, [addr])


def sts128(addr, w0, w1, w2, w3):
    _ptx(_STS128, [], [addr, w0, w1, w2, w3])


def lds32(addr):
    return _ptx(_LDS32, [cutlass.Int32], [addr])


def sts32(addr, w):
    _ptx(_STS32, [], [addr, w])


def sts16(addr, w):
    _ptx(_STS16, [], [addr, w])


def sts8(addr, w):
    _ptx(_STS8, [], [addr, w])


def stsm_x2(addr, w0, w1):
    """``stmatrix.x2`` of two 8x8 b16 matrices: ``addr`` is this lane's row address
    (lanes 0-7 matrix 0, 8-15 matrix 1; other lanes' addresses are ignored), ``w0`` /
    ``w1`` the lane's b32 of matrix 0 / 1 (row ``lane/4``, b16 slots ``2q, 2q+1``)."""
    _ptx(_STSM_X2, [], [addr, w0, w1])


def stg256(addr, words):
    """One 256-bit ``st.global.v8.b32`` of eight Int32 words at a 64-bit address."""
    _ptx(_STG256, [], [addr] + list(words))


def e8m0_and_rcp(scaled_amax):
    """(biased exponent as Int32, 1/scale as Float32) -- TE's float_to_e8m0 + exp2f_rcp."""
    return _ptx(_E8M0_AND_RCP, [cutlass.Int32, cutlass.Float32], [scaled_amax])


def mul_cvt_2x(w0, w1, inv):
    return _ptx(_MUL_CVT_2X, [cutlass.Int32], [w0, w1, inv])


def f32x2_to_e4m3x2(lo, hi):
    """Int32 whose byte 0 = e4m3(lo), byte 1 = e4m3(hi)."""
    return _ptx(_F32X2_TO_E4M3X2, [cutlass.Int32], [hi, lo])


def e4m3_to_f32(code):
    return _ptx(_E4M3_TO_F32, [cutlass.Float32], [code])


def mul_cvt_e2m1x4(p01, p23, acc):
    """Four E2M1 codes (low 16 bits of the Int32; high 16 duplicate them) of two packed-bf16x2 words scaled by ``acc``."""
    return _ptx(_MUL_CVT_E2M1X4, [cutlass.Int32], [p01, p23, acc])


def bf16_lo(w):
    return (w << 16).bitcast(cutlass.Float32)


def bf16_hi(w):
    return (w & cutlass.Int32(_BF16_HI_MASK)).bitcast(cutlass.Float32)


def round_bf16(x):
    return x.to(cutlass.BFloat16).to(cutlass.Float32)


def warp_max(v):
    """Butterfly max over the warp (plain loop: traced helper, see module docstring)."""
    for off in (16, 8, 4, 2, 1):
        v = cute.arch.fmax(v, cute.arch.shuffle_sync_bfly(v, offset=off))
    return v


def swizzled_scale_idx(i, j, n_tiles_x):
    """mxfp8/swizzle.cuh:31-43 -- cuBLASLt's 128 x 4 scale atom, as Int32 math."""
    idx = ((i >> 7) * n_tiles_x + (j >> 2)) * 512
    return idx + (i & 31) * 16 + ((i >> 5) & 3) * 4 + (j & 3)


def global_encode_scale(amax):
    """dual_cast_one_read_kernel.cu global_encode_scale: min(2688/amax, FLT_MAX), 1 if amax or s is 0."""
    s = cute.arch.fmin(div_const_rn(2688.0, amax), cutlass.Float32(FLT_MAX))
    one = cutlass.Float32(1.0)
    s = cutlass.Float32(cutlass.select_(s == cutlass.Float32(0.0), one, s))
    return cutlass.Float32(cutlass.select_(amax == cutlass.Float32(0.0), one, s))


# --- TMA geometry (called from the @cute.jit host, traced) ------------------------


def ring_layout(tile_rows: int, slots: int):
    """``slots`` SWIZZLE_128B boxes of ``tile_rows`` x 64 bf16, box-major (K_SW128 = S<3,4,3>).

    The kernels hard-code the CUDA kernels' addressing of this layout: box ``k``
    at byte ``k * tile_rows * 128``, row ``r`` at ``r * 128``, 16 B chunk
    ``c ^ (r & 7)``.
    """
    atom = tcgen05.make_smem_layout_atom(tcgen05.SmemLayoutAtomKind.K_SW128, cutlass.BFloat16)
    return cute.coalesce(
        cute.tile_to_shape(atom, (tile_rows, BOX_COLS, slots), order=(0, 1, 2)),
        target_profile=(1, 1, 1),
    )


def make_tma(x, tile_rows: int, smem_layout):
    """TMA G2S atom for ``tile_rows`` x 64 boxes of a row-major bf16 ``[M, N]``."""
    tma_atom, tma_tensor = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(), x, smem_layout, (tile_rows, BOX_COLS),
    )
    return tma_atom, tma_tensor


# --- host-side grid rules (the CUDA hosts', 1:1) ------------------------------------

_sm_count: dict[int, int] = {}


def sm_count(device_index: int) -> int:
    if device_index not in _sm_count:
        import torch
        _sm_count[device_index] = torch.cuda.get_device_properties(
            device_index).multi_processor_count
    return _sm_count[device_index]


def geometry_windows(
    rows: int, cols: int, tile_rows: int, window: int, ctas_per_sm: int, sms: int, factor: int,
) -> tuple[int, int, int, int]:
    """``onepass_act_grid``: (col_strips, row_splits, tiles_per_cta, resident) with whole
    flush windows per CTA. ``factor`` oversubscribes the resident-CTA count in the grid
    rule (1 = one CTA per resident slot; 2 = the CUDA kernel's ``+o2``; 4 = the
    Rubin ``x4`` grid, which trades a 1.96-wave tail for a 3.9-wave one at ``[16384]``)."""
    col_strips = cols // TILE_COLS
    flush_tiles = window // tile_rows
    windows = rows // window
    resident = ctas_per_sm * sms * factor
    row_splits = max(1, min(windows, resident // col_strips))
    wpc = -(-windows // row_splits)
    row_splits = -(-windows // wpc)
    return col_strips, row_splits, wpc * flush_tiles, resident


def geometry_tiles(
    rows: int, cols: int, tile_rows: int, ctas_per_sm: int, sms: int, oversub: bool,
) -> tuple[int, int, int, int]:
    """``gelu_tanh_amax_grid``: (col_strips, row_splits, tiles_per_cta, resident)."""
    col_strips = cols // TILE_COLS
    row_tiles = rows // tile_rows
    resident = ctas_per_sm * sms * (2 if oversub else 1)
    row_splits = max(1, min(row_tiles, resident // col_strips))
    tpc = -(-row_tiles // row_splits)
    row_splits = -(-row_tiles // tpc)
    return col_strips, row_splits, tpc, resident
