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

"""CuTe DSL kernels for the gpt-oss-20b FP8 attention input/output chain (torch.library namespace `gpt_oss`). They replace TE's
fused_qkv_rope_forward_kernel + Float8CurrentScaling amax chain (zero_amax, amax_kernel, compute_scale_from_amax, at::reciprocal) in
the forward and fused_qkv_rope_backward_kernel + bgrad_quantize + reduce_dbias in the backward; TE's cast_fp8_1D (bf16 -> e4m3) stays.
  qkv_rope_fwd_amax(x [s,b,ng*(dq+dk+dv)] bf16 (qkv GEMM output, bias applied), cos, sin [s,d] fp32, ng, dq, dk, dv, fma_order)
      -> flat [nq+nk+nv] bf16 (q rotated | k rotated | v, each contiguous in sbhd order), scale fp32[1] (e4m3_max / amax, TE
         compute_scale_from_amax semantics), amax fp32[1]
  qkv_rope_bwd_mxfp8(gq [s,b,hq,d], gk, gv [s,b,ng,d] bf16, cos, sin, ng, dq, dk, dv, fp8_dtype_code, swizzled, fma_order)
      -> rowwise_data u8 [M,N], rowwise_scale_inv u8 [M,N/32], columnwise_data u8 [M,N], columnwise_scale_inv u8 [M/32,N],
         dbias_partial fp32 [M/32,N]  (M = s*b, N = ng*(dq+dk+dv); TE MXFP8Tensor layout, scales compact or cuBLAS GEMM-swizzled)
  dbias_reduce(partial, out_dtype) -> [N], the sum over the M/32 partial rows
Workload shape: s=8192, b=3, ng=8, dq=512 (8 q heads x 64), dk=dv=64, N=5120, M=24576.
Numerics knobs (env, read at import): QKV_ROPE_FMA_ORDER 1 fma(x,cos,rot*sin) / 2 fma(rot,sin,x*cos) / 0 x*cos + rot*sin (TE's nvcc
contraction); QKV_ROPE_FMA_IMPL 0 MLIR math.fma / 1 cute.math.fma; QKV_ROPE_E8M0_MODE 0 DSL conversion / 1 TE float_to_e8m0 bit
arithmetic; QKV_ROPE_FENCE_MODE 0 cute.arch.fence_acq_rel_gpu / 1 inline PTX fence.acq_rel.gpu.
"""
import os
from typing import Tuple

import torch

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32, BFloat16
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T
from cutlass._mlir.dialects import llvm
from cutlass._mlir.dialects import math as mlir_math

VEC = 8                      # bf16 elements per 16-byte vector
HEAD_D = 64                  # head dim == rotary dim (asserted on the torch side)
NVEC_D = HEAD_D // VEC       # 8 vectors per head
HALF = NVEC_D // 2           # the RoPE partner of vector i is vector i +- 4
FP8_E4M3_MAX = 448.0
FP8_E5M2_MAX = 57344.0
FLT_MAX = 3.4028234663852886e38
F32_INF = float("inf")
F32_MIN_NORMAL = 1.1754943508222875e-38
TE_FP8_CODE = {7: ("e4m3", cutlass.Float8E4M3FN, torch.float8_e4m3fn), 8: ("e5m2", cutlass.Float8E5M2, torch.float8_e5m2)}
# TE's `1.0f / Quantized_Limits<OType>::max_norm` rounded to fp32, once at import
MAX_NORM_RCP = {name: float(torch.tensor(1.0 / mx, dtype=torch.float32).item()) for name, mx in (("e4m3", FP8_E4M3_MAX), ("e5m2", FP8_E5M2_MAX))}

_DEFAULT_FMA_ORDER = int(os.environ.get("QKV_ROPE_FMA_ORDER", "1"))
_DEFAULT_FMA_IMPL = int(os.environ.get("QKV_ROPE_FMA_IMPL", "0"))
_DEFAULT_E8M0_MODE = int(os.environ.get("QKV_ROPE_E8M0_MODE", "0"))
_DEFAULT_FENCE_MODE = int(os.environ.get("QKV_ROPE_FENCE_MODE", "0"))


# ---------------------------------------------------------------------------------------------- device helpers
def _fma_vec(a, b, c, fma_impl):
    """a*b + c with one rounding, on 8-wide fp32 TensorSSA values."""
    if cutlass.const_expr(fma_impl == 1):
        return cute.math.fma(a, b, c)
    r = mlir_math.fma(a.ir_value(), b.ir_value(), c.ir_value())
    return type(a)(r, a.shape, a.dtype)


def _rope_combine(x, cos_v, rot, sin_v, fma_order, fma_impl):
    """TE: `v_src * v_cos + v_src_rotate * v_sin` (fp32); fma_order picks the operand order of nvcc's FMA contraction."""
    if cutlass.const_expr(fma_order == 1):
        return _fma_vec(x, cos_v, rot * sin_v, fma_impl)
    if cutlass.const_expr(fma_order == 2):
        return _fma_vec(rot, sin_v, x * cos_v, fma_impl)
    return x * cos_v + rot * sin_v


def _f32_bits(v: Float32) -> Int32:
    return Int32(llvm.bitcast(T.i32(), v.ir_value()))


def _bits_f32(b: Int32) -> Float32:
    return Float32(llvm.bitcast(T.f32(), b.ir_value()))


def _fence_gpu(fence_mode):
    if cutlass.const_expr(fence_mode == 1):
        llvm.inline_asm(None, [], "fence.acq_rel.gpu;", "", has_side_effects=True, asm_dialect=0)
    else:
        cute.arch.fence_acq_rel_gpu()


def _e8m0_bits_te(v: Float32) -> Int32:
    """TE float_to_e8m0 (non-PTX reference): biased exponent of the power of two >= v, satfinite, NaN -> 255."""
    bits = _f32_bits(v)
    exp = (bits >> 23) & 0xFF
    mant = bits & 0x7FFFFF
    nz = (mant + 0x7FFFFF) >> 23                     # 1 iff mant != 0
    e = cutlass.min(exp + nz, 254)                   # exp < 255: round the exponent up when the mantissa is non-zero
    one_if_exp0 = (0xFF - exp + 1) >> 8              # 1 iff exp == 0 (zero / denormal input)
    e_den = (mant + 0x3FFFFF) >> 23                  # exp == 0: 0 unless v > 2^-127 (mant > 0x400000) -> 1
    e = e * (1 - one_if_exp0) + e_den * one_if_exp0
    one_if_exp255 = (exp + 1) >> 8                   # 1 iff exp == 255: inf -> 254, nan -> 255
    e = e * (1 - one_if_exp255) + (254 + nz) * one_if_exp255
    return e


def _block_scale(amax: Float32, max_norm_rcp, e8m0_mode):
    """MXFP8 block scale from the block amax (TE cast_mxfp8 semantics):
    e8m0 = float_to_e8m0(amax * max_norm_rcp) -> (scale as Float8E8M0FNU, scale_inv fp32 = 2^-(e-127))."""
    v = amax * Float32(max_norm_rcp)
    if cutlass.const_expr(e8m0_mode == 1):
        e = _e8m0_bits_te(v)
        sf = _bits_f32(e << 23)                      # 2^(e-127) exactly (e == 0 -> 0.0, e == 255 -> inf)
    else:
        sf = v.to(cutlass.Float8E8M0FNU).to(Float32)
    e8 = sf.to(cutlass.Float8E8M0FNU)
    # 1/2^k is exact; the fmax guards the e == 0 block (amax == 0: every element is 0, so scale_inv is irrelevant,
    # TE returns 1 there) against a 1/0.
    sinv = Float32(1.0) / cute.arch.fmax(sf, Float32(F32_MIN_NORMAL))
    return e8, sinv


def _swz_idx(i: Int32, j: Int32, ntx: Int32) -> Int32:
    """TE gemm_swizzled_scale_idx(i, j, num_tiles_X): cuBLAS 2D block-scaling layout (128 x 4 tiles)."""
    tile_y = i // 128
    tile_x = j // 4
    iy = i % 128
    ix = j % 4
    return (tile_y * ntx + tile_x) * 512 + (iy % 32) * 16 + (iy // 32) * 4 + ix


# =============================================================================================
# Kernel 1: forward RoPE (q, k) + relocation into one flat q|k|v buffer + per-tensor amax, scale
#   thread <-> (token, head slot, quarter): the thread owns the 16-byte vectors `quarter` and `quarter + 4` of one
#   64-wide head (the RoPE pair), so the 4 lanes of a slot cover 64 contiguous bytes per load and a warp's 8 slots
#   cover 8 full 128-byte lines: every 32-byte sector fetched is fully used.
# =============================================================================================
FWD_TOKENS_PER_BLOCK = 2
FWD_LANES_PER_SLOT = 4


@cute.kernel
def _rope_fwd_kernel(
    gX: cute.Tensor, gOut: cute.Tensor, gCos: cute.Tensor, gSin: cute.Tensor,
    gScale: cute.Tensor, gAmax: cute.Tensor, gWsF: cute.Tensor, gWsI: cute.Tensor,
    b: Int32, m_tokens: Int32, nblocks: Int32,
    ng: cutlass.Constexpr, nqpg: cutlass.Constexpr, nslots: cutlass.Constexpr, nwarps: cutlass.Constexpr,
    nq_elems: cutlass.Constexpr, nk_elems: cutlass.Constexpr,
    fma_order: cutlass.Constexpr, fma_impl: cutlass.Constexpr, fence_mode: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    hidden = ng * (nqpg + 2) * HEAD_D
    gsz = (nqpg + 2) * HEAD_D
    nqh = ng * nqpg
    tpt = nslots * FWD_LANES_PER_SLOT                 # threads per token

    smem_ptr = cute.arch.alloc_smem(Float32, nwarps)
    smem = cute.make_tensor(smem_ptr, cute.make_layout((nwarps,)))

    tok_in_blk = tidx // tpt
    rem = tidx % tpt
    slot = rem // FWD_LANES_PER_SLOT
    quarter = rem % FWD_LANES_PER_SLOT
    t = bidx * FWD_TOKENS_PER_BLOCK + tok_in_blk
    g = slot // (nqpg + 2)
    r = slot % (nqpg + 2)
    kv_r = cutlass.max(r - nqpg, -1) + 1          # 0: q head, 1: k, 2: v
    is_kv = cutlass.min(kv_r, 1)                  # 0 for q, 1 for k/v
    in_vec = (t * hidden + g * gsz + r * HEAD_D) // VEC + quarter
    out_q = (t * nqh + g * nqpg + r) * HEAD_D
    out_kv = nq_elems + (kv_r - 1) * nk_elems + (t * ng + g) * HEAD_D
    out_vec = ((1 - is_kv) * out_q + is_kv * out_kv) // VEC + quarter
    cs_vec = (t // b) * NVEC_D + quarter

    local_amax = Float32(0.0)
    if t < m_tokens:
        x_lo = gX[(None, in_vec)].load().to(Float32)
        x_hi = gX[(None, in_vec + HALF)].load().to(Float32)
        out_lo = x_lo
        out_hi = x_hi
        if kv_r < 2:
            cos_lo = gCos[(None, cs_vec)].load()
            cos_hi = gCos[(None, cs_vec + HALF)].load()
            sin_lo = gSin[(None, cs_vec)].load()
            sin_hi = gSin[(None, cs_vec + HALF)].load()
            # TE (non-interleaved): rotate = -x[d + d2/2] for d < d2/2, x[d - d2/2] otherwise
            out_lo = _rope_combine(x_lo, cos_lo, x_hi * Float32(-1.0), sin_lo, fma_order, fma_impl)
            out_hi = _rope_combine(x_hi, cos_hi, x_lo, sin_hi, fma_order, fma_impl)
        o_lo = out_lo.to(BFloat16)
        o_hi = out_hi.to(BFloat16)
        gOut[(None, out_vec)] = o_lo
        gOut[(None, out_vec + HALF)] = o_hi
        a_lo = cute.math.absf(o_lo.to(Float32)).reduce(cute.ReductionOp.MAX, Float32(0.0), 0)
        a_hi = cute.math.absf(o_hi.to(Float32)).reduce(cute.ReductionOp.MAX, Float32(0.0), 0)
        local_amax = cute.arch.fmax(a_lo, a_hi)

    # block amax -> one atomic per CTA; the last CTA (atomic counter) derives the scale and resets the workspace
    warp_id = tidx // 32
    lane_id = tidx % 32
    local_amax = cute.arch.warp_reduction_max(local_amax)
    if lane_id == 0:
        smem[warp_id] = local_amax
    cute.arch.sync_threads()
    if tidx == 0:
        blk = smem[0]
        for w in cutlass.range_constexpr(1, nwarps):
            blk = cute.arch.fmax(blk, smem[w])
        cute.arch.atomic_fmax(gWsF.iterator, blk, sign_bit=False)
        _fence_gpu(fence_mode)
        old = Int32(cute.arch.atomic_add(gWsI.iterator.llvm_ptr, Int32(1)))
        if old == nblocks - 1:
            _fence_gpu(fence_mode)
            amax = gWsF[0]
            gWsF[0] = Float32(0.0)
            gWsI[0] = Int32(0)
            # TE compute_scale_from_amax: 1 for amax == 0 or inf, else fp8_max / amax, inf -> FLT_MAX
            scale = Float32(1.0)
            if amax > Float32(0.0):
                if amax < Float32(F32_INF):
                    scale = Float32(FP8_E4M3_MAX) / amax
            if scale == Float32(F32_INF):
                scale = Float32(FLT_MAX)
            gScale[0] = scale
            gAmax[0] = amax


@cute.jit
def _rope_fwd_host(
    mX: cute.Tensor, mOut: cute.Tensor, mCos: cute.Tensor, mSin: cute.Tensor,
    mScale: cute.Tensor, mAmax: cute.Tensor, mWsF: cute.Tensor, mWsI: cute.Tensor,
    b: Int32, m_tokens: Int32, nblocks: Int32,
    ng: cutlass.Constexpr, nqpg: cutlass.Constexpr, nq_elems: cutlass.Constexpr, nk_elems: cutlass.Constexpr,
    fma_order: cutlass.Constexpr, fma_impl: cutlass.Constexpr, fence_mode: cutlass.Constexpr, stream: cuda.CUstream,
):
    nslots = ng * (nqpg + 2)
    nthreads = FWD_TOKENS_PER_BLOCK * nslots * FWD_LANES_PER_SLOT
    nwarps = (nthreads + 31) // 32
    gX = cute.zipped_divide(mX, (VEC,))
    gOut = cute.zipped_divide(mOut, (VEC,))
    gCos = cute.zipped_divide(mCos, (VEC,))
    gSin = cute.zipped_divide(mSin, (VEC,))
    _rope_fwd_kernel(
        gX, gOut, gCos, gSin, mScale, mAmax, mWsF, mWsI, b, m_tokens, nblocks,
        ng, nqpg, nslots, nwarps, nq_elems, nk_elems, fma_order, fma_impl, fence_mode,
    ).launch(grid=(nblocks, 1, 1), block=(nthreads, 1, 1), stream=stream)


# =============================================================================================
# Kernel 2: backward RoPE + pack into d(qkv) + MXFP8 rowwise & columnwise quantization + dbias partials
# =============================================================================================
BWD_ROWS = 32                 # tokens per CTA tile (one MXFP8 columnwise block)
BWD_THREADS = 256
BWD_Q_HEADS_PER_CTA = 4       # q CTA tile: 32 tokens x 4 heads (256 cols); kv CTA tile: 32 tokens x (k head, v head)


@cute.kernel
def _rope_bwd_kernel(
    gSrcA: cute.Tensor, gSrcB: cute.Tensor, gCos: cute.Tensor, gSin: cute.Tensor,
    gRowData: cute.Tensor, gRowScale: cute.Tensor, gColData: cute.Tensor, gColScale: cute.Tensor, gPartial: cute.Tensor,
    b: Int32, m_tokens: Int32,
    ng: cutlass.Constexpr, nqpg: cutlass.Constexpr, is_kv: cutlass.Constexpr, npass: cutlass.Constexpr,
    swizzled: cutlass.Constexpr, max_norm_rcp: cutlass.Constexpr,
    fma_order: cutlass.Constexpr, fma_impl: cutlass.Constexpr, e8m0_mode: cutlass.Constexpr,
):
    """grid = (M/32, ng, nz) ; q kernel (is_kv=False): z selects a group of BWD_Q_HEADS_PER_CTA q heads, npass = 4,
    source gSrcA = dq (sbhd, hq heads); kv kernel (is_kv=True): npass = 2, pass 0 = k (gSrcA), pass 1 = v (gSrcB)."""
    tidx, _, _ = cute.arch.thread_idx()
    bx, g, bz = cute.arch.block_idx()
    hidden = ng * (nqpg + 2) * HEAD_D
    gsz = (nqpg + 2) * HEAD_D
    nqh = ng * nqpg
    width = npass * HEAD_D                        # tile width in columns (256 or 128)
    fp8_t = gRowData.element_type

    s_bf = cute.arch.alloc_smem(BFloat16, BWD_ROWS * width)
    smem_bf = cute.make_tensor(s_bf, cute.make_layout((BWD_ROWS * width,)))
    smem_bf8 = cute.zipped_divide(smem_bf, (VEC,))
    s_f8 = cute.arch.alloc_smem(fp8_t, BWD_ROWS * width)
    smem_f8 = cute.make_tensor(s_f8, cute.make_layout((BWD_ROWS * width,)))
    smem_f88 = cute.zipped_divide(smem_f8, (VEC,))

    row = tidx // NVEC_D                          # 0..31
    c = tidx % NVEC_D                             # vector (8 dims) within the head: dims 8c..8c+7
    cp = c ^ HALF                                 # partner vector (dims +- 32)
    sgn = Float32(1 - 2 * (c // HALF))            # +sin[d+32] for d < 32, -sin[d-32] for d >= 32 (TE backward)
    t = bx * BWD_ROWS + row
    s_id = t // b
    cs_vec = s_id * NVEC_D
    t_row_base = t * hidden

    # ---------------- phase A: row-mapped: inverse RoPE, pack, rowwise MXFP8, stage the bf16 tile
    for p in cutlass.range_constexpr(npass):
        if cutlass.const_expr(is_kv):
            head_col = (nqpg + p) * HEAD_D                       # k: nqpg*D, v: (nqpg+1)*D within the group
            src_vec = ((t * ng + g) * HEAD_D) // VEC
        else:
            hq = g * nqpg + bz * BWD_Q_HEADS_PER_CTA + p
            head_col = (bz * BWD_Q_HEADS_PER_CTA + p) * HEAD_D
            src_vec = ((t * nqh + hq) * HEAD_D) // VEC
        col0 = g * gsz + head_col                                # packed column of dim 0 of this head
        if cutlass.const_expr(is_kv and p == 1):
            f = gSrcB[(None, src_vec + c)].load().to(Float32)    # v: copy
        else:
            x = gSrcA[(None, src_vec + c)].load().to(Float32)
            xp = gSrcA[(None, src_vec + cp)].load().to(Float32)
            cos_v = gCos[(None, cs_vec + c)].load()
            sin_p = gSin[(None, cs_vec + cp)].load() * sgn
            f = _rope_combine(x, cos_v, xp, sin_p, fma_order, fma_impl).to(BFloat16).to(Float32)
        # stage the bf16-rounded packed gradient (TE's bgrad_quantize reads the bf16 d(qkv))
        smem_bf8[(None, row * (width // VEC) + p * NVEC_D + c)] = f.to(BFloat16)
        # rowwise MXFP8 block = 32 columns = 4 lanes (c // 4 selects the block within the head)
        amax = cute.math.absf(f).reduce(cute.ReductionOp.MAX, Float32(0.0), 0)
        amax = cute.arch.fmax(amax, cute.arch.shuffle_sync_bfly(amax, offset=1))
        amax = cute.arch.fmax(amax, cute.arch.shuffle_sync_bfly(amax, offset=2))
        e8, sinv = _block_scale(amax, max_norm_rcp, e8m0_mode)
        gRowData[(None, (t_row_base + col0) // VEC + c)] = (f * sinv).to(fp8_t)
        if c % 4 == 0:
            cb = (col0 // 32) + c // 4
            if cutlass.const_expr(swizzled):
                sidx = _swz_idx(t, cb, hidden // 128)
            else:
                sidx = t * (hidden // 32) + cb
            gRowScale[sidx] = e8
    cute.arch.sync_threads()

    # ---------------- phase B: column-mapped: columnwise MXFP8 (32 tokens per block) + dbias partials
    for j in cutlass.range_constexpr((width + BWD_THREADS - 1) // BWD_THREADS):
        col = tidx + j * BWD_THREADS
        if col < width:
            camax = Float32(0.0)
            csum = Float32(0.0)
            for rr in cutlass.range_constexpr(BWD_ROWS):
                v = smem_bf[rr * width + col].to(Float32)
                camax = cute.arch.fmax(camax, cute.arch.fmax(v, Float32(0.0) - v))
                csum = csum + v
            e8, sinv = _block_scale(camax, max_norm_rcp, e8m0_mode)
            for rr in cutlass.range_constexpr(BWD_ROWS):
                v = smem_bf[rr * width + col].to(Float32)
                smem_f8[rr * width + col] = (v * sinv).to(fp8_t)
            # packed column of this tile column
            if cutlass.const_expr(is_kv):
                gcol = g * gsz + nqpg * HEAD_D + col
            else:
                gcol = g * gsz + bz * BWD_Q_HEADS_PER_CTA * HEAD_D + col
            rb = bx
            if cutlass.const_expr(swizzled):
                sidx = _swz_idx(gcol, rb, m_tokens // 128)
            else:
                sidx = rb * hidden + gcol
            gColScale[sidx] = e8
            gPartial[rb * hidden + gcol] = csum
    cute.arch.sync_threads()

    # ---------------- phase C: row-mapped copy-out of the columnwise fp8 tile (8-byte vectors)
    for p in cutlass.range_constexpr(npass):
        if cutlass.const_expr(is_kv):
            head_col = (nqpg + p) * HEAD_D
        else:
            head_col = (bz * BWD_Q_HEADS_PER_CTA + p) * HEAD_D
        col0 = g * gsz + head_col
        v8 = smem_f88[(None, row * (width // VEC) + p * NVEC_D + c)].load()
        gColData[(None, (t_row_base + col0) // VEC + c)] = v8


@cute.jit
def _rope_bwd_host(
    mSrcA: cute.Tensor, mSrcB: cute.Tensor, mCos: cute.Tensor, mSin: cute.Tensor,
    mRowData: cute.Tensor, mRowScale: cute.Tensor, mColData: cute.Tensor, mColScale: cute.Tensor, mPartial: cute.Tensor,
    b: Int32, m_tokens: Int32, nrow_blk: Int32,
    ng: cutlass.Constexpr, nqpg: cutlass.Constexpr, is_kv: cutlass.Constexpr, npass: cutlass.Constexpr,
    nz: cutlass.Constexpr, swizzled: cutlass.Constexpr, max_norm_rcp: cutlass.Constexpr,
    fma_order: cutlass.Constexpr, fma_impl: cutlass.Constexpr, e8m0_mode: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    gA = cute.zipped_divide(mSrcA, (VEC,))
    gB = cute.zipped_divide(mSrcB, (VEC,))
    gCos = cute.zipped_divide(mCos, (VEC,))
    gSin = cute.zipped_divide(mSin, (VEC,))
    gRow = cute.zipped_divide(mRowData, (VEC,))
    gCol = cute.zipped_divide(mColData, (VEC,))
    _rope_bwd_kernel(
        gA, gB, gCos, gSin, gRow, mRowScale, gCol, mColScale, mPartial, b, m_tokens,
        ng, nqpg, is_kv, npass, swizzled, max_norm_rcp, fma_order, fma_impl, e8m0_mode,
    ).launch(grid=(nrow_blk, ng, nz), block=(BWD_THREADS, 1, 1), stream=stream)


# =============================================================================================
# Kernel 3: dbias partial reduction  [R, N] fp32 -> [N] (bf16 / fp32)
# =============================================================================================
RED_THREADS = 256
RED_COLS = 32


@cute.kernel
def _dbias_reduce_kernel(gP: cute.Tensor, gOut: cute.Tensor, nrows: Int32, ncols: Int32):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    nwarps = RED_THREADS // 32
    smem_ptr = cute.arch.alloc_smem(Float32, RED_THREADS)
    smem = cute.make_tensor(smem_ptr, cute.make_layout((RED_THREADS,)))
    warp_id = tidx // 32
    lane_id = tidx % 32
    col = bidx * RED_COLS + lane_id
    acc = Float32(0.0)
    if col < ncols:
        for r in cutlass.range(warp_id, nrows, nwarps):
            acc = acc + gP[r * ncols + col]
    smem[tidx] = acc
    cute.arch.sync_threads()
    if tidx < RED_COLS:
        tot = Float32(0.0)
        for w in cutlass.range_constexpr(nwarps):
            tot = tot + smem[w * 32 + tidx]
        if bidx * RED_COLS + tidx < ncols:
            gOut[bidx * RED_COLS + tidx] = tot.to(gOut.element_type)


@cute.jit
def _dbias_reduce_host(mP: cute.Tensor, mOut: cute.Tensor, nrows: Int32, ncols: Int32, nblocks: Int32, stream: cuda.CUstream):
    _dbias_reduce_kernel(mP, mOut, nrows, ncols).launch(grid=(nblocks, 1, 1), block=(RED_THREADS, 1, 1), stream=stream)


# =============================================================================================
# torch integration
# =============================================================================================
_COMPILED = {}
_WS = {}


def _ws(device):
    """Persistent amax/counter workspace (zero at every kernel entry: the last CTA resets it)."""
    key = str(device)
    w = _WS.get(key)
    if w is None:
        w = (torch.zeros(1, dtype=torch.float32, device=device), torch.zeros(1, dtype=torch.int32, device=device))
        _WS[key] = w
    return w


def _flat(t: torch.Tensor, align=16):
    return from_dlpack(t.detach().reshape(-1), assumed_align=align)


def _flat_as(t: torch.Tensor, cute_dtype, align=16):
    """uint8 storage viewed with a CuTe element type the torch tensor cannot carry (Float8E8M0FNU scales)."""
    ct = from_dlpack(t.detach().reshape(-1), assumed_align=align)
    ct.element_type = cute_dtype
    return ct


def _stream():
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


def _check_tables(cos, sin, s, d):
    assert cos.dtype == torch.float32 and sin.dtype == torch.float32
    assert cos.shape == (s, d) and sin.shape == (s, d), (cos.shape, sin.shape, s, d)
    assert cos.is_contiguous() and sin.is_contiguous()


def _variants(fma_order):
    if fma_order < 0:
        fma_order = _DEFAULT_FMA_ORDER
    return int(fma_order), int(_DEFAULT_FMA_IMPL), int(_DEFAULT_E8M0_MODE), int(_DEFAULT_FENCE_MODE)


@torch.library.custom_op("gpt_oss::qkv_rope_fwd_amax", mutates_args=())
def qkv_rope_fwd_amax(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, ng: int, dq: int, dk: int, dv: int, fma_order: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """x [s, b, ng*(dq+dk+dv)] bf16 -> (flat [nq+nk+nv] bf16 : q_rot | k_rot | v in sbhd order, scale[1], amax[1])."""
    assert x.dim() == 3 and x.dtype == torch.bfloat16 and x.is_contiguous(), (x.shape, x.dtype)
    s, b, hidden = x.shape
    assert dk == HEAD_D and dv == HEAD_D and dq % HEAD_D == 0 and hidden == ng * (dq + dk + dv), (s, b, hidden, ng, dq, dk, dv)
    nqpg = dq // HEAD_D
    _check_tables(cos, sin, s, HEAD_D)
    m = s * b
    assert m % FWD_TOKENS_PER_BLOCK == 0
    fma_order, fma_impl, _, fence_mode = _variants(fma_order)
    numel = m * hidden
    out = torch.empty(numel, dtype=torch.bfloat16, device=x.device)
    scale = torch.empty(1, dtype=torch.float32, device=x.device)
    amax = torch.empty(1, dtype=torch.float32, device=x.device)
    ws_f, ws_i = _ws(x.device)
    nblocks = m // FWD_TOKENS_PER_BLOCK
    key = ("fwd", s, b, ng, nqpg, fma_order, fma_impl, fence_mode, str(x.device))
    # The compiled JitExecutor's argument list EXCLUDES the Constexpr parameters (baked in at compile time): they go
    # to cute.compile only; the compiled function is called with the runtime arguments + stream alone.
    stream = _stream()
    rt_args = (
        _flat(x), _flat(out), _flat(cos), _flat(sin), _flat(scale, 4), _flat(amax, 4), _flat(ws_f, 4), _flat(ws_i, 4),
        Int32(b), Int32(m), Int32(nblocks),
    )
    cexpr = (ng, nqpg, m * ng * nqpg * HEAD_D, m * ng * HEAD_D, fma_order, fma_impl, fence_mode)
    fn = _COMPILED.get(key)
    if fn is None:
        fn = cute.compile(_rope_fwd_host, *rt_args, *cexpr, stream)
        _COMPILED[key] = fn
    fn(*rt_args, stream)
    return out, scale, amax


@qkv_rope_fwd_amax.register_fake
def _(x, cos, sin, ng, dq, dk, dv, fma_order=-1):
    return (torch.empty(x.numel(), dtype=x.dtype, device=x.device),
            torch.empty(1, dtype=torch.float32, device=x.device),
            torch.empty(1, dtype=torch.float32, device=x.device))


@torch.library.custom_op("gpt_oss::qkv_rope_bwd_mxfp8", mutates_args=())
def qkv_rope_bwd_mxfp8(
    gq: torch.Tensor, gk: torch.Tensor, gv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
    ng: int, dq: int, dk: int, dv: int, fp8_dtype_code: int = 7, swizzled: bool = True, fma_order: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """-> (rowwise_data u8 [M,N], rowwise_scale_inv u8 [M,N/32], columnwise_data u8 [M,N],
           columnwise_scale_inv u8 [M/32,N], dbias_partial f32 [M/32,N]) in TE's MXFP8Tensor layout."""
    s, b, hq, d = gq.shape
    assert d == HEAD_D and dk == HEAD_D and dv == HEAD_D and dq % HEAD_D == 0
    nqpg = dq // HEAD_D
    assert hq == ng * nqpg and gk.shape == (s, b, ng, d) and gv.shape == (s, b, ng, d), (gq.shape, gk.shape, gv.shape)
    assert gq.dtype == torch.bfloat16 and gk.dtype == torch.bfloat16 and gv.dtype == torch.bfloat16
    assert gq.is_contiguous() and gk.is_contiguous() and gv.is_contiguous()
    assert nqpg % BWD_Q_HEADS_PER_CTA == 0, nqpg
    _check_tables(cos, sin, s, HEAD_D)
    m = s * b
    hidden = ng * (dq + dk + dv)
    assert m % 128 == 0 and hidden % 128 == 0, (m, hidden)   # TE MXFP8 scale padding: none needed
    fma_order, fma_impl, e8m0_mode, _ = _variants(fma_order)
    name, cute_fp8, torch_fp8 = TE_FP8_CODE[int(fp8_dtype_code)]
    max_norm_rcp = MAX_NORM_RCP[name]
    dev = gq.device
    row_data = torch.empty(m, hidden, dtype=torch.uint8, device=dev)
    col_data = torch.empty(m, hidden, dtype=torch.uint8, device=dev)
    row_scale = torch.empty(m, hidden // 32, dtype=torch.uint8, device=dev)
    col_scale = torch.empty(m // 32, hidden, dtype=torch.uint8, device=dev)
    partial = torch.empty(m // 32, hidden, dtype=torch.float32, device=dev)
    nrow_blk = m // BWD_ROWS
    stream = _stream()
    common = (_flat(cos), _flat(sin), _flat_as(row_data.view(torch_fp8), cute_fp8), _flat_as(row_scale, cutlass.Float8E8M0FNU, 1),
              _flat_as(col_data.view(torch_fp8), cute_fp8), _flat_as(col_scale, cutlass.Float8E8M0FNU, 1), _flat(partial, 4),
              Int32(b), Int32(m), Int32(nrow_blk))
    # q heads: grid (M/32, ng, nqpg/4), 4 heads per CTA.  Constexpr parameters go to cute.compile only (see fwd).
    nz = nqpg // BWD_Q_HEADS_PER_CTA
    key = ("bwd_q", s, b, ng, nqpg, name, bool(swizzled), fma_order, fma_impl, e8m0_mode, str(dev))
    rt_args = (_flat(gq), _flat(gq), *common)
    cexpr = (ng, nqpg, False, BWD_Q_HEADS_PER_CTA, nz, bool(swizzled), max_norm_rcp, fma_order, fma_impl, e8m0_mode)
    fn = _COMPILED.get(key)
    if fn is None:
        fn = cute.compile(_rope_bwd_host, *rt_args, *cexpr, stream)
        _COMPILED[key] = fn
    fn(*rt_args, stream)
    # k and v heads: grid (M/32, ng, 1), pass 0 = k (rope), pass 1 = v (copy)
    key = ("bwd_kv", s, b, ng, nqpg, name, bool(swizzled), fma_order, fma_impl, e8m0_mode, str(dev))
    rt_args = (_flat(gk), _flat(gv), *common)
    cexpr = (ng, nqpg, True, 2, 1, bool(swizzled), max_norm_rcp, fma_order, fma_impl, e8m0_mode)
    fn = _COMPILED.get(key)
    if fn is None:
        fn = cute.compile(_rope_bwd_host, *rt_args, *cexpr, stream)
        _COMPILED[key] = fn
    fn(*rt_args, stream)
    return row_data, row_scale, col_data, col_scale, partial


@qkv_rope_bwd_mxfp8.register_fake
def _(gq, gk, gv, cos, sin, ng, dq, dk, dv, fp8_dtype_code=7, swizzled=True, fma_order=-1):
    s, b, hq, d = gq.shape
    m = s * b
    hidden = ng * (dq + dk + dv)
    dev = gq.device
    return (torch.empty(m, hidden, dtype=torch.uint8, device=dev), torch.empty(m, hidden // 32, dtype=torch.uint8, device=dev),
            torch.empty(m, hidden, dtype=torch.uint8, device=dev), torch.empty(m // 32, hidden, dtype=torch.uint8, device=dev),
            torch.empty(m // 32, hidden, dtype=torch.float32, device=dev))


@torch.library.custom_op("gpt_oss::dbias_reduce", mutates_args=())
def dbias_reduce(partial: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    assert partial.dim() == 2 and partial.dtype == torch.float32 and partial.is_contiguous()
    nrows, ncols = partial.shape
    out = torch.empty(ncols, dtype=out_dtype, device=partial.device)
    nblocks = (ncols + RED_COLS - 1) // RED_COLS
    key = ("red", nrows, ncols, str(out_dtype), str(partial.device))
    args = (_flat(partial, 4), _flat(out, 2), Int32(nrows), Int32(ncols), Int32(nblocks), _stream())
    fn = _COMPILED.get(key)
    if fn is None:
        fn = cute.compile(_dbias_reduce_host, *args)
        _COMPILED[key] = fn
    fn(*args)
    return out


@dbias_reduce.register_fake
def _(partial, out_dtype):
    return torch.empty(partial.shape[1], dtype=out_dtype, device=partial.device)


# =============================================================================================
# References (pure torch) and the get_inputs / kernel_fn / reference_fn test entry points
# =============================================================================================
def _ref_rope_fwd(x, cos, sin, ng, dq, dk, dv):
    """fp32 RoPE (TE non-interleaved convention) of q and k, v copied; -> (q [s,b,hq,d], k [s,b,ng,d], v, amax)."""
    s, b, hidden = x.shape
    d = dk
    nqpg = dq // d
    xg = x.view(s, b, ng, dq + dk + dv).float()
    q = xg[..., :dq].reshape(s, b, ng * nqpg, d)
    k = xg[..., dq:dq + dk].reshape(s, b, ng, d)
    v = xg[..., dq + dk:].reshape(s, b, ng, d)
    c = cos.view(s, 1, 1, d)
    sn = sin.view(s, 1, 1, d)

    def rot(t):
        h = d // 2
        return torch.cat([-t[..., h:], t[..., :h]], dim=-1)

    q = (q * c + rot(q) * sn).to(torch.bfloat16)
    k = (k * c + rot(k) * sn).to(torch.bfloat16)
    v = v.to(torch.bfloat16)
    amax = torch.stack([q.float().abs().amax(), k.float().abs().amax(), v.float().abs().amax()]).amax()
    return q, k, v, amax


def _ref_rope_bwd_packed(gq, gk, gv, cos, sin, ng, dq, dk, dv):
    """bf16 packed d(qkv) [s, b, ng*(dq+dk+dv)] with the TE backward RoPE (fp32 math, bf16 store)."""
    s, b, hq, d = gq.shape
    nqpg = dq // d
    c = cos.view(s, 1, 1, d)
    sn = sin.view(s, 1, 1, d)

    def bwd(gr):
        h = d // 2
        g = gr.float()
        lo = g[..., :h] * c[..., :h] + g[..., h:] * sn[..., h:]
        hi = g[..., h:] * c[..., h:] - g[..., :h] * sn[..., :h]
        return torch.cat([lo, hi], dim=-1).to(torch.bfloat16)

    dqp = bwd(gq).view(s, b, ng, dq)
    dkp = bwd(gk).view(s, b, ng, dk)
    dvp = gv.view(s, b, ng, dv)
    return torch.cat([dqp, dkp, dvp], dim=-1).reshape(s, b, ng * (dq + dk + dv))


def _ref_mxfp8(packed_bf16_2d, fp8_max, torch_fp8):
    """Reference MXFP8 (TE semantics: e8m0 = ceil(log2(amax/fp8_max)), x * 2^-e, satfinite cast) rowwise + columnwise."""
    x = packed_bf16_2d.float()
    M, N = x.shape

    def quant(blocks):  # blocks: [..., 32]
        amax = blocks.abs().amax(dim=-1, keepdim=True)
        v = amax * torch.tensor(1.0 / fp8_max, dtype=torch.float32, device=x.device)
        e = torch.where(v > 0, torch.ceil(torch.log2(v)), torch.full_like(v, -127.0)).clamp(-127, 127)
        sinv = torch.exp2(-e)
        q = (blocks * sinv).to(torch_fp8)
        return q, (e + 127).to(torch.uint8)

    qr, er = quant(x.view(M, N // 32, 32))
    qc, ec = quant(x.view(M // 32, 32, N).transpose(1, 2))     # [M/32, N, 32]
    return (qr.view(M, N).view(torch.uint8), er.view(M, N // 32), qc.transpose(1, 2).reshape(M, N).view(torch.uint8),
            ec.view(M // 32, N), x.view(M // 32, 32, N).sum(dim=1))


def get_inputs():
    s, b, ng, dq, dk, dv = 256, 2, 8, 512, 64, 64
    torch.manual_seed(0)
    x = torch.randn(s, b, ng * (dq + dk + dv), dtype=torch.bfloat16, device="cuda")
    freqs = torch.outer(torch.arange(s, device="cuda", dtype=torch.float32), 1.0 / (10000 ** (torch.arange(0, 64, 2, device="cuda").float() / 64)))
    freqs = torch.cat([freqs, freqs], dim=-1)
    return [x, torch.cos(freqs), torch.sin(freqs)]


def kernel_fn(x, cos, sin):
    flat, scale, amax = torch.ops.gpt_oss.qkv_rope_fwd_amax(x, cos, sin, 8, 512, 64, 64)
    s, b, _ = x.shape
    nq = s * b * 64 * 64
    nk = s * b * 8 * 64
    return flat[:nq].view(s, b, 64, 64), flat[nq:nq + nk].view(s, b, 8, 64), flat[nq + nk:].view(s, b, 8, 64), amax


def reference_fn(x, cos, sin):
    return _ref_rope_fwd(x, cos, sin, 8, 512, 64, 64)


if __name__ == "__main__":
    xs = get_inputs()
    outs = kernel_fn(*xs)
    refs = reference_fn(*xs)
    torch.cuda.synchronize()
    for n, a, r in zip(("q", "k", "v", "amax"), outs, refs):
        print(n, "max abs diff", (a.float() - r.float()).abs().max().item())
