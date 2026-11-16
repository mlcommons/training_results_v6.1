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

"""Triton kernels: single-pass fused RMSNorm backward for gpt-oss-20b (hidden 2880).

Replaces TransformerEngine's two-kernel cuDNN RMSNorm backward (ln_tma_bwd_kernel + ln_bwd_finalize_kernel) and TE's
native kernel alike.  One streaming pass over dy / x / rstd writes dx (optionally accumulated into the residual-stream
gradient: dx + dres), optionally the MXFP8 rowwise + columnwise copies of that gradient in TE's MXFP8Tensor layout (fp8
data [M, N], e8m0 scales in the cuBLAS GEMM-swizzled 128x4 layout with TE's zero padding) -- the proj dgrad/wgrad operand
the following TE Linear backward would otherwise re-cast -- and per-CTA (32-row block) fp32 partial sums of dgamma and
optionally dbias (column sums of the emitted gradient, i.e. the proj bias gradient).  A second tiny kernel (colsum) reduces
the [M/32, N or 2N] partials to dgamma (and dbias) in the weight dtype, turning the ~137 us/launch ln_bwd_finalize_kernel
pass into a ~17.7 MB read.  Both are torch.library ops: torch.ops.gpt_oss.rmsnorm_bwd_fused and torch.ops.gpt_oss.colsum.
Math (fp32 inside; rstd = 1/sqrt(mean(x^2) + eps) from the forward): g = gamma (+1 when zero_centered_gamma),
dx_i = rstd * dy_i * g_i - x_i * rstd^3 / N * sum_j(dy_j g_j x_j), dgamma = sum_rows dy * x * rstd.  MXFP8 follows TE's
MXFP8Quantizer exactly (verified against it) on the bf16-rounded gradient: per 32-element block e8m0 = biased exponent of
the power of two >= amax / fp8_max (round up, satfinite; 0 -> scale 1), data = satfinite_cast_fp8(x * exp2(127 - e8m0)).
"""
import os
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

MX_BLOCK = 32            # MXFP8 block: 32 elements along the quantized dimension
ROWS = 32                # rows per CTA = one columnwise MXFP8 block
FP8_MAX = {7: 448.0, 8: 57344.0}          # TE DType codes: 7 = kFloat8E4M3, 8 = kFloat8E5M2
TORCH_FP8 = {7: torch.float8_e4m3fn, 8: torch.float8_e5m2}
# Default tile BLOCK_N=256 / 4 warps.  At M=24576, N=2880: full variant 160.2 us at 256/4 vs 206.1 at 128/4; plain variant
# 115.7 vs 101.2.  One default serves both variants (a per-call-site tile has not been measured end to end).
DEFAULT_BLOCK_N = int(os.environ.get("NORM_BWD_BLOCK_N", "256"))
DEFAULT_NUM_WARPS = int(os.environ.get("NORM_BWD_NUM_WARPS", "4"))


def _round_up(a: int, b: int) -> int:
    return (a + b - 1) // b * b


# =============================================================================================
# Triton device helpers
# =============================================================================================
@triton.jit
def _e8m0_from_amax(amax, FP8_MAX_RCP: tl.constexpr):
    """TE float_to_e8m0(amax * (1/fp8_max)): biased exponent of the power of two >= v (round up), satfinite."""
    v = amax * FP8_MAX_RCP
    bits = v.to(tl.int32, bitcast=True)
    exp = (bits >> 23) & 0xFF
    mant = bits & 0x7FFFFF
    # if ((mantissa > 0 && exponent != 0xFE) && !(exponent == 0 && mantissa <= 0x400000)) exponent++
    inc = (mant > 0) & (exp != 0xFE) & ((exp != 0) | (mant > 0x400000))
    e = exp + inc.to(tl.int32)
    e = tl.where(v == 0.0, 0, e)
    return e


@triton.jit
def _scale_rcp(e):
    """TE exp2f_rcp(e8m0): 1 for e == 0, else 2^(127 - e) (the multiplier that quantizes x)."""
    r = ((254 - e) << 23).to(tl.float32, bitcast=True)          # exact for 1 <= e <= 253
    r = tl.where(e == 0, 1.0, r)
    r = tl.where(e >= 254, 5.877471754111438e-39, r)              # 2^-127
    return r


@triton.jit
def _swz(i, j, ntx):
    """TE gemm_swizzled_scale_idx(i, j, num_tiles_X): flat index in the cuBLAS 2D block-scaling (128 x 4 tiles) layout."""
    iy = i % 128
    return ((i // 128) * ntx + (j // 4)) * 512 + (iy % 32) * 16 + (iy // 32) * 4 + (j % 4)


# =============================================================================================
# Kernel 1: fused RMSNorm backward, one CTA per 32-row block, two sweeps over N in BLOCK_N chunks
#   sweep 1: per-row sum_j dy*g*x (the second sweep re-reads dy/x; the 32-row working set stays in L2)
#   sweep 2: dx (+dres) -> bf16 store, dgamma/dbias partials, MXFP8 rowwise (32 cols) + columnwise (32 rows)
# =============================================================================================
@triton.jit
def _rmsnorm_bwd_fused_kernel(
    dy_ptr, x_ptr, rstd_ptr, g_ptr, dres_ptr, out_ptr, part_ptr,
    rd_ptr, rs_ptr, cd_ptr, cs_ptr,
    M, N, PSTRIDE, NPAD_C, NBLK_PAD, NTX_R, NTX_C, NCHUNK,
    ZERO_CENTERED: tl.constexpr, HAS_RES: tl.constexpr, OUT_MXFP8: tl.constexpr, OUT_DBIAS: tl.constexpr,
    ROWS: tl.constexpr, BLOCK_N: tl.constexpr, FP8_MAX_RCP: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * ROWS + tl.arange(0, ROWS)
    rmask = rows < M
    rstd = tl.load(rstd_ptr + rows, mask=rmask, other=0.0)
    acc = tl.zeros([ROWS], dtype=tl.float32)
    for ch in range(0, NCHUNK):
        cols = ch * BLOCK_N + tl.arange(0, BLOCK_N)
        cm = cols < N
        m2 = rmask[:, None] & cm[None, :]
        offs = rows[:, None] * N + cols[None, :]
        dy = tl.load(dy_ptr + offs, mask=m2, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + offs, mask=m2, other=0.0).to(tl.float32)
        g = tl.load(g_ptr + cols, mask=cm, other=0.0).to(tl.float32)
        if ZERO_CENTERED:
            g = g + 1.0
        acc += tl.sum(dy * g[None, :] * x, axis=1)
    k = acc * (rstd * rstd * rstd) / N              # x_i * k_i is the mean-correction term of dx

    for ch in range(0, NCHUNK):
        cols = ch * BLOCK_N + tl.arange(0, BLOCK_N)
        cm = cols < N
        m2 = rmask[:, None] & cm[None, :]
        offs = rows[:, None] * N + cols[None, :]
        dy = tl.load(dy_ptr + offs, mask=m2, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + offs, mask=m2, other=0.0).to(tl.float32)
        g = tl.load(g_ptr + cols, mask=cm, other=0.0).to(tl.float32)
        if ZERO_CENTERED:
            g = g + 1.0
        dx = rstd[:, None] * (dy * g[None, :]) - x * k[:, None]
        if HAS_RES:
            dx += tl.load(dres_ptr + offs, mask=m2, other=0.0).to(tl.float32)
        o16 = dx.to(tl.bfloat16)
        tl.store(out_ptr + offs, o16, mask=m2)
        # dgamma partial of this 32-row block: sum_rows dy * x * rstd
        dg = tl.sum(dy * x * rstd[:, None], axis=0)
        tl.store(part_ptr + pid * PSTRIDE + cols, dg, mask=cm)
        of = o16.to(tl.float32)                       # the bf16-rounded gradient TE would quantize / sum
        if OUT_DBIAS:
            db = tl.sum(of, axis=0)
            tl.store(part_ptr + pid * PSTRIDE + N + cols, db, mask=cm)
        if OUT_MXFP8:
            # rowwise: blocks of 32 columns
            a3 = tl.reshape(tl.abs(of), [ROWS, BLOCK_N // 32, 32])
            amax_r = tl.max(a3, axis=2)                                    # [ROWS, BLOCK_N // 32]
            e_r = _e8m0_from_amax(amax_r, FP8_MAX_RCP)
            rcp_r = _scale_rcp(e_r)
            rcp_r3 = tl.broadcast_to(tl.expand_dims(rcp_r, 2), [ROWS, BLOCK_N // 32, 32])
            rcp_rf = tl.reshape(rcp_r3, [ROWS, BLOCK_N])
            qr = (of * rcp_rf).to(rd_ptr.dtype.element_ty)
            tl.store(rd_ptr + offs, qr, mask=m2)
            blk = ch * (BLOCK_N // 32) + tl.arange(0, BLOCK_N // 32)
            sidx_r = _swz(rows[:, None], blk[None, :], NTX_R)
            smask_r = rmask[:, None] & (blk < NBLK_PAD)[None, :]           # zero padding blocks too (amax = 0 -> e = 0)
            tl.store(rs_ptr + sidx_r, e_r.to(tl.uint8), mask=smask_r)
            # columnwise: this CTA's 32 rows are one block per column
            amax_c = tl.max(tl.abs(of), axis=0)                            # [BLOCK_N]
            e_c = _e8m0_from_amax(amax_c, FP8_MAX_RCP)
            rcp_c = _scale_rcp(e_c)
            qc = (of * rcp_c[None, :]).to(cd_ptr.dtype.element_ty)
            tl.store(cd_ptr + offs, qc, mask=m2)
            sidx_c = _swz(cols, pid, NTX_C)                                # transposed convention: i = col, j = row block
            tl.store(cs_ptr + sidx_c, e_c.to(tl.uint8), mask=cols < NPAD_C)  # zero padding columns too


# =============================================================================================
# Kernel 2: column sums of the fp32 partials -> weight dtype (dgamma | dbias)
# =============================================================================================
@triton.jit
def _colsum_kernel(part_ptr, out_ptr, NROWS, NCOLS, BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr):
    pid = tl.program_id(0)
    cols = pid * BLOCK_C + tl.arange(0, BLOCK_C)
    cm = cols < NCOLS
    acc = tl.zeros([BLOCK_C], dtype=tl.float32)
    for r0 in range(0, NROWS, BLOCK_R):
        rows = r0 + tl.arange(0, BLOCK_R)
        m = (rows < NROWS)[:, None] & cm[None, :]
        v = tl.load(part_ptr + rows[:, None] * NCOLS + cols[None, :], mask=m, other=0.0)
        acc += tl.sum(v, axis=0)
    tl.store(out_ptr + cols, acc.to(out_ptr.dtype.element_ty), mask=cm)


# =============================================================================================
# torch.library ops
# =============================================================================================
def _check_inputs(dy, x, rstd, gamma, dres):
    assert dy.dim() == 2 and x.shape == dy.shape, (dy.shape, x.shape)
    M, N = dy.shape
    assert dy.dtype == torch.bfloat16 and x.dtype == torch.bfloat16, (dy.dtype, x.dtype)
    assert dy.is_contiguous() and x.is_contiguous()
    assert rstd.dtype == torch.float32 and rstd.numel() == M and rstd.is_contiguous(), (rstd.dtype, rstd.shape)
    assert gamma.numel() == N and gamma.is_contiguous(), gamma.shape
    assert N % MX_BLOCK == 0 and M % ROWS == 0, (M, N)
    if dres is not None:
        assert dres.shape == dy.shape and dres.dtype == torch.bfloat16 and dres.is_contiguous()
    return M, N


@torch.library.custom_op("gpt_oss::rmsnorm_bwd_fused", mutates_args=())
def rmsnorm_bwd_fused(
    dy: torch.Tensor, x: torch.Tensor, rstd: torch.Tensor, gamma: torch.Tensor, dres: Optional[torch.Tensor],
    zero_centered_gamma: bool, out_mxfp8: bool, out_dbias: bool, fp8_code: int = 7,
    block_n: int = -1, num_warps: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    M, N = _check_inputs(dy, x, rstd, gamma, dres)
    dev = dy.device
    block_n = DEFAULT_BLOCK_N if block_n <= 0 else block_n
    num_warps = DEFAULT_NUM_WARPS if num_warps <= 0 else num_warps
    assert block_n % MX_BLOCK == 0 and (block_n & (block_n - 1)) == 0, block_n
    pstride = 2 * N if out_dbias else N
    out = torch.empty(M, N, dtype=torch.bfloat16, device=dev)
    part = torch.empty(M // ROWS, pstride, dtype=torch.float32, device=dev)
    if out_mxfp8:
        nblk_pad = _round_up(N // MX_BLOCK, 4)
        npad_c = _round_up(N, 128)          # == nblk_pad * 32
        # sweep over the padded width: the fully-masked tail chunks write TE's zero padding of both scale matrices
        nchunk = triton.cdiv(npad_c, block_n)
        rd = torch.empty(M, N, dtype=torch.uint8, device=dev)
        cd = torch.empty(M, N, dtype=torch.uint8, device=dev)
        # Row padding (M % 128 != 0 / (M/32) % 4 != 0) interleaves inside the swizzled 128x4 tiles, so those buffers
        # start zeroed; the workload's M = 24576 needs none and gets plain empties (no memset launch).
        rs_alloc = torch.zeros if _round_up(M, 128) != M else torch.empty
        cs_alloc = torch.zeros if _round_up(M // MX_BLOCK, 4) != M // MX_BLOCK else torch.empty
        rs = rs_alloc(_round_up(M, 128), nblk_pad, dtype=torch.uint8, device=dev)
        cs = cs_alloc(_round_up(M // MX_BLOCK, 4), npad_c, dtype=torch.uint8, device=dev)
        ntx_r = nblk_pad // 4
        ntx_c = _round_up(M // MX_BLOCK, 4) // 4
        rd_t = rd.view(TORCH_FP8[int(fp8_code)])
        cd_t = cd.view(TORCH_FP8[int(fp8_code)])
    else:
        nblk_pad = npad_c = ntx_r = ntx_c = 1
        nchunk = triton.cdiv(N, block_n)
        rd, rs, cd, cs = (torch.empty(0, dtype=torch.uint8, device=dev) for _ in range(4))   # distinct: no aliasing outputs
        rd_t = rd.view(torch.float8_e4m3fn)
        cd_t = cd.view(torch.float8_e4m3fn)
    grid = (M // ROWS,)
    _rmsnorm_bwd_fused_kernel[grid](
        dy, x, rstd, gamma, dres if dres is not None else dy, out, part,
        rd_t, rs, cd_t, cs,
        M, N, pstride, npad_c, nblk_pad, ntx_r, ntx_c, nchunk,
        ZERO_CENTERED=bool(zero_centered_gamma), HAS_RES=dres is not None, OUT_MXFP8=bool(out_mxfp8),
        OUT_DBIAS=bool(out_dbias), ROWS=ROWS, BLOCK_N=block_n, FP8_MAX_RCP=1.0 / FP8_MAX[int(fp8_code)],
        num_warps=num_warps,
    )
    return out, part, rd, rs, cd, cs


@rmsnorm_bwd_fused.register_fake
def _(dy, x, rstd, gamma, dres, zero_centered_gamma, out_mxfp8, out_dbias, fp8_code=7, block_n=-1, num_warps=-1):
    M, N = dy.shape
    dev = dy.device
    pstride = 2 * N if out_dbias else N
    out = torch.empty(M, N, dtype=torch.bfloat16, device=dev)
    part = torch.empty(M // ROWS, pstride, dtype=torch.float32, device=dev)
    if out_mxfp8:
        return (out, part, torch.empty(M, N, dtype=torch.uint8, device=dev),
                torch.empty(_round_up(M, 128), _round_up(N // MX_BLOCK, 4), dtype=torch.uint8, device=dev),
                torch.empty(M, N, dtype=torch.uint8, device=dev),
                torch.empty(_round_up(M // MX_BLOCK, 4), _round_up(N, 128), dtype=torch.uint8, device=dev))
    return (out, part) + tuple(torch.empty(0, dtype=torch.uint8, device=dev) for _ in range(4))


@torch.library.custom_op("gpt_oss::colsum", mutates_args=())
def colsum(part: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    assert part.dim() == 2 and part.dtype == torch.float32 and part.is_contiguous()
    nrows, ncols = part.shape
    out = torch.empty(ncols, dtype=out_dtype, device=part.device)
    BLOCK_C = 32
    _colsum_kernel[(triton.cdiv(ncols, BLOCK_C),)](part, out, nrows, ncols, BLOCK_R=32, BLOCK_C=BLOCK_C, num_warps=4)
    return out


@colsum.register_fake
def _(part, out_dtype):
    return torch.empty(part.shape[1], dtype=out_dtype, device=part.device)


# =============================================================================================
# Python-side helpers used by the hook
# =============================================================================================
def rmsnorm_bwd_plain(dy2d, x2d, rstd, gamma, dres=None, zero_centered_gamma=False, block_n=-1, num_warps=-1):
    """dx (+dres) and dgamma (weight dtype), nothing else -- the drop-in for transformer_engine_torch.rmsnorm_bwd."""
    out, part, _, _, _, _ = torch.ops.gpt_oss.rmsnorm_bwd_fused(
        dy2d, x2d, rstd, gamma, dres, zero_centered_gamma, False, False, 7, block_n, num_warps)
    dgamma = torch.ops.gpt_oss.colsum(part, gamma.dtype)
    return out, dgamma


# =============================================================================================
# References (pure torch) and a standalone self-test (get_inputs / kernel_fn / reference_fn)
# =============================================================================================
def ref_rmsnorm_bwd(dy, x, rstd, gamma, dres=None, zero_centered_gamma=False):
    """fp32 reference: (dx (+dres) bf16, dgamma fp32 [N])."""
    dyf, xf, rs = dy.float(), x.float(), rstd.float()[:, None]
    g = gamma.float() + (1.0 if zero_centered_gamma else 0.0)
    dyg = dyf * g
    N = x.shape[1]
    c = (dyg * xf).sum(dim=1, keepdim=True) * rs * rs * rs / N
    dx = rs * dyg - xf * c
    if dres is not None:
        dx = dx + dres.float()
    dgamma = (dyf * xf * rs).sum(dim=0)
    return dx.to(torch.bfloat16), dgamma


def ref_mxfp8(x_bf16_2d, fp8_code=7):
    """Reference MXFP8 (TE semantics) of a bf16 [M, N] tensor: rowwise/columnwise fp8 data (u8 view) and e8m0 scales
    in the COMPACT layouts [M, N/32] and [M/32, N]; see swizzle_ref() for the GEMM layout."""
    xf = x_bf16_2d.float()
    M, N = xf.shape
    fp8_max = FP8_MAX[fp8_code]
    tfp8 = TORCH_FP8[fp8_code]

    def quant(blocks):   # [..., 32]
        amax = blocks.abs().amax(dim=-1, keepdim=True)
        v = amax * torch.tensor(1.0 / fp8_max, dtype=torch.float32, device=xf.device)
        e = torch.where(v > 0, torch.ceil(torch.log2(v)), torch.full_like(v, -127.0)).clamp(-127, 127)
        sinv = torch.exp2(-e)
        return (blocks * sinv).to(tfp8), (e + 127).to(torch.uint8)

    qr, er = quant(xf.view(M, N // 32, 32))
    qc, ec = quant(xf.view(M // 32, 32, N).transpose(1, 2))
    return (qr.view(M, N).view(torch.uint8), er.view(M, N // 32),
            qc.transpose(1, 2).reshape(M, N).view(torch.uint8), ec.view(M // 32, N))


def swizzle_ref(compact, transposed, pad_rows, pad_cols):
    """GEMM-swizzle a compact [R, C] e8m0 matrix into the padded flat layout the kernel writes.
    transposed=False: rowwise scales (i = row, j = col block); True: columnwise (i = col, j = row block)."""
    R, C = compact.shape
    dev = compact.device
    if transposed:
        src = compact.t().contiguous()        # [C(=N), R(=M/32)]  i = col, j = row block
        I, J = C, R
        Ipad, Jpad = pad_cols, pad_rows      # buffer is [pad_rows, pad_cols] = [round_up(M/32,4), round_up(N,128)]
    else:
        src = compact
        I, J = R, C
        Ipad, Jpad = pad_rows, pad_cols
    out = torch.zeros(Ipad * Jpad, dtype=torch.uint8, device=dev)
    ntx = Jpad // 4
    ii = torch.arange(I, device=dev)[:, None].expand(I, J)
    jj = torch.arange(J, device=dev)[None, :].expand(I, J)
    idx = ((ii // 128) * ntx + (jj // 4)) * 512 + ((ii % 128) % 32) * 16 + ((ii % 128) // 32) * 4 + (jj % 4)
    out[idx.reshape(-1)] = src.reshape(-1)
    return out.view(pad_rows, pad_cols)


def get_inputs():
    M, N = 4096, 2880
    torch.manual_seed(0)
    dy = torch.randn(M, N, device="cuda").to(torch.bfloat16)
    x = (torch.randn(M, N, device="cuda") * 2.0).to(torch.bfloat16)
    rstd = torch.rsqrt(x.float().pow(2).mean(dim=1) + 1e-5).contiguous()
    gamma = (1.0 + 0.1 * torch.randn(N, device="cuda")).to(torch.bfloat16)
    dres = torch.randn(M, N, device="cuda").to(torch.bfloat16)
    return [dy, x, rstd, gamma, dres]


def kernel_fn(dy, x, rstd, gamma, dres):
    out, part, rd, rs, cd, cs = torch.ops.gpt_oss.rmsnorm_bwd_fused(dy, x, rstd, gamma, dres, False, True, True, 7, -1, -1)
    red = torch.ops.gpt_oss.colsum(part, torch.float32)
    N = x.shape[1]
    return out, red[:N], red[N:]


def reference_fn(dy, x, rstd, gamma, dres):
    out, dgamma = ref_rmsnorm_bwd(dy, x, rstd, gamma, dres, False)
    return out, dgamma, out.float().sum(dim=0)


if __name__ == "__main__":
    xs = get_inputs()
    outs = kernel_fn(*xs)
    refs = reference_fn(*xs)
    torch.cuda.synchronize()
    for n, a, r in zip(("dx+dres", "dgamma", "dbias"), outs, refs):
        print(n, "max abs diff", (a.float() - r.float()).abs().max().item(), "ref max", r.float().abs().max().item())
