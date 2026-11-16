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

"""Triton kernels: MoE token permute fused with MXFP8 quantization in HybridEP's dense layout (EP=1).

Replaces HybridEP's dispatch copy + permute_kernel and TE's group_quantize_mxfp8_kernel re-read of the permuted bf16
rows with one gather pass that reads each token row once per expert slot and writes the MXFP8 operands directly.
gpt_oss::moe_qperm_inverse_map builds row_to_token [P] (-1 on padding) and permuted_probs from HybridEP's
dense_to_expert_map; gpt_oss::moe_qperm_gather_quant writes the rowwise and columnwise e4m3 data with their e8m0 scales
and, on request, the bf16 rows; gpt_oss::moe_qperm_gather_quant_scaled does the same for the combine backward's
dz = bf16(p * dres[t]) and also emits dprob[p] = dres[t] . z[p], the bf16 dz rows (write_bf16) and the per-expert fp32
column sum of dz (colsum, the fc2 bias gradient).  Numerics follow TE's MXFP8 cast: data and scales are bitwise
tex.group_quantize, padding rows are all zero.  Scales use TE's GEMM-swizzled layouts (rowwise over (row/128, H-block/4),
columnwise one block per expert group = cw_layout 3); MOE_QPERM_RW_LAYOUT / MOE_QPERM_CW_LAYOUT / MOE_QPERM_DIV select
the variants.  MOE_QPERM_CUDA=1 (default) sends the gather ops to the CUDA C++ kernels of moe_qperm_cuda for
rw 0 / cw 3 / div 0, pad_multiple % 128 == 0 and E <= 256; MOE_QPERM_CUDA=0 or any other request stays on the Triton
kernels below.  No autotune (CUDA-graph captured); kernel_fn / reference_fn / get_inputs are the standalone check entry points.
"""
import os
import sys
from typing import List, Tuple

import torch
import triton
import triton.language as tl

try:
    import custom_kernels.moe_qperm_cuda as _CUDA   # CUDA C++ backend of the two gather ops
except Exception as _ex:  # noqa
    _CUDA = None
    sys.__stdout__.write(f"MOE_QPERM_CUDA rank={os.getenv('RANK', os.getenv('SLURM_PROCID', '0'))} backend module import failed "
                         f"{type(_ex).__name__}: {_ex}; Triton kernels\n")

# Tile: BLOCK_M rows (a multiple of the 32-row columnwise block, <= 128 so one program stays inside one 128-row
# scale tile and one expert group) x BLOCK_K columns (a multiple of 128 = one scale tile of 4 rowwise blocks).
# Defaults 32 x 256 x 4 warps measured fastest at the workload shape: fwd (fp8 rowwise+columnwise) 183 us, bwd (+bf16
# rows) 234 us; 32x128x4 237/274, 32x512x4 193/255, 64x256x8 376/434, 128x128x8 481/532.
BLOCK_M = int(os.environ.get("MOE_QPERM_BM", "32"))
# Combine-backward kernel only: 32-row blocks handled by one program (consecutive, inside one expert group since
# PAD % (NRB * BLOCK_M) == 0); the fused fc2-dbias column sum is accumulated over them in registers and reaches the
# [E, H] accumulator through one atomic per program instead of one per 32-row block.  Measured at the workload shape
# (fused mode): NRB=1 0.346 ms, 2 0.480, 4 0.431, 8 0.427 -- fewer programs cost more than the atomics save, so 1 is
# the default.
NRB = int(os.environ.get("MOE_QPERM_NRB", "1"))
BLOCK_K = int(os.environ.get("MOE_QPERM_BK", "256"))
NUM_WARPS = int(os.environ.get("MOE_QPERM_NUM_WARPS", "4"))
E4M3_MAX_RCP = 1.0 / 448.0


# =============================================================================================
# device helpers
# =============================================================================================
@triton.jit
def _e8m0_from_amax(amax, rcp, DIV_MODE: tl.constexpr):
    """TE's float_to_e8m0(amax * max_norm_rcp): biased exponent, rounded up to the next power of two."""
    if DIV_MODE == 0:
        v = amax * rcp
    else:
        v = amax / 448.0
    bits = v.to(tl.int32, bitcast=True)
    exp = (bits >> 23) & 0xFF
    man = bits & 0x7FFFFF
    abits = bits & 0x7FFFFFFF
    e = exp + tl.where((man > 0) & (exp != 254), 1, 0)
    e = tl.where(v == 0.0, 0, e)
    e = tl.where(abits == 0x7F800000, 254, e)      # inf  -> 0xFE
    e = tl.where(abits > 0x7F800000, 255, e)       # nan  -> 0xFF
    return e


@triton.jit
def _scale_inv_from_e8m0(e):
    """TE's exp2f_rcp(e): 2^(127 - e) as fp32, 1.0 for e == 0; denormal for e >= 254."""
    bits_n = (254 - e) << 23                                  # exact for 1 <= e <= 253
    bits_d = tl.where(e == 254, 0x00400000, 0x00200000)       # 2^-127, 2^-128
    bits = tl.where(e >= 254, bits_d, bits_n)
    s = bits.to(tl.float32, bitcast=True)
    return tl.where(e == 0, 1.0, s)


@triton.jit
def _cvt_e4m3_u8(v):
    """fp32 -> e4m3 bytes (round-to-nearest-even, saturate to the finite max); four elements per asm block,
    two per cvt.e4m3x2 instruction (element i lands in byte i)."""
    return tl.inline_asm_elementwise(
        asm="""{
        .reg .b16 lo, hi;
        cvt.rn.satfinite.e4m3x2.f32 lo, $2, $1;
        cvt.rn.satfinite.e4m3x2.f32 hi, $4, $3;
        mov.b32 $0, {lo, hi};
        }""",
        constraints="=r,f,f,f,f",
        args=[v],
        dtype=tl.int8,
        is_pure=True,
        pack=4,
    ).to(tl.uint8, bitcast=True)


# =============================================================================================
# kernels
# =============================================================================================
@triton.jit
def _qperm_inverse_map_kernel(map_ptr, probs_ptr, layout_ptr, r2t_ptr, pprob_ptr, n_chunks, T_rows, P,
                              E: tl.constexpr, HAS_PROBS: tl.constexpr, BLOCK_T: tl.constexpr):
    pid = tl.program_id(0)
    n_dense = tl.load(layout_ptr + n_chunks - 1)
    rows = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    cols = tl.arange(0, E)
    m2 = (rows[:, None] < T_rows) & (cols[None, :] < E)
    idx = rows[:, None].to(tl.int64) * E + cols[None, :]
    dest = tl.load(map_ptr + idx, mask=m2, other=-1)
    ok = (dest >= 0) & (dest < P) & (rows[:, None] < n_dense) & m2
    dest64 = tl.maximum(dest, 0).to(tl.int64)
    tok = tl.broadcast_to(rows[:, None], (BLOCK_T, E))
    tl.store(r2t_ptr + dest64, tok, mask=ok)
    if HAS_PROBS:
        pr = tl.load(probs_ptr + idx, mask=ok, other=0.0)
        tl.store(pprob_ptr + dest64, pr, mask=ok)


@triton.jit
def _qperm_gather_quant_kernel(src_ptr, r2t_ptr, tpe_ptr, rw_ptr, rws_ptr, cw_ptr, cws_ptr, out_ptr,
                               T, H, KT, MB4, KB_PAD, K_PAD, rcp,
                               rk0, rk1, sk0, sk1, sk2, rm0, rm1, sm0, sm1, sm2,
                               E: tl.constexpr, PAD: tl.constexpr, WRITE_BF16: tl.constexpr,
                               RW_LAYOUT: tl.constexpr, CW_LAYOUT: tl.constexpr, DIV_MODE: tl.constexpr,
                               BM: tl.constexpr, BK: tl.constexpr):
    kt = tl.program_id(0)
    pm = tl.program_id(1)
    # rows in use = sum_e round_up(count_e, PAD); blocks beyond are left untouched (HybridEP leaves them too)
    eidx = tl.arange(0, E)
    cnt = tl.load(tpe_ptr + eidx)
    pc = ((cnt + (PAD - 1)) // PAD) * PAD                          # padded rows per expert [E]
    n_used = tl.sum(pc, axis=0)
    p0 = pm * BM
    if p0 < n_used:
        rows = p0 + tl.arange(0, BM)
        rows64 = rows.to(tl.int64)
        tok = tl.load(r2t_ptr + rows)                              # -1 on padding rows
        rvalid = (tok >= 0) & (tok < T)
        col0 = kt * BK
        cols = col0 + tl.arange(0, BK)
        cvalid = cols < H
        m2 = rvalid[:, None] & cvalid[None, :]
        src = tl.load(src_ptr + tl.maximum(tok, 0).to(tl.int64)[:, None] * H + cols[None, :], mask=m2, other=0.0)
        if WRITE_BF16:
            tl.store(out_ptr + rows64[:, None] * H + cols[None, :], src, mask=cvalid[None, :])
        x = src.to(tl.float32)
        ax = tl.abs(x)

        # ---- rowwise: NB blocks of 32 along H
        NB: tl.constexpr = BK // 32
        ax3 = tl.reshape(ax, (BM, NB, 32))
        amax_r = tl.max(ax3, axis=2)                               # [BM, NB]
        e_r = _e8m0_from_amax(amax_r, rcp, DIV_MODE)
        sinv_r = _scale_inv_from_e8m0(e_r)
        sinv_r_full = tl.reshape(tl.broadcast_to(sinv_r[:, :, None], (BM, NB, 32)), (BM, BK))
        q_r = _cvt_e4m3_u8(x * sinv_r_full)
        tl.store(rw_ptr + rows64[:, None] * H + cols[None, :], q_r, mask=cvalid[None, :])
        jb = tl.arange(0, NB)
        kb = (col0 // 32) + jb                                     # absolute H-block index [NB]
        kb_valid = (kb * 32) < H
        if RW_LAYOUT == 0:
            pos_r = (((p0 // 128) * KT + kb // 4)[None, :] * 512 + (rows % 32)[:, None] * 16
                     + ((rows // 32) % 4)[:, None] * 4 + (kb % 4)[None, :])
        else:
            pos_r = rows[:, None] * KB_PAD + kb[None, :]
        tl.store(rws_ptr + pos_r.to(tl.int64), e_r.to(tl.uint8), mask=kb_valid[None, :])

        # ---- columnwise: NMB blocks of 32 rows per column
        NMB: tl.constexpr = BM // 32
        ax3c = tl.reshape(ax, (NMB, 32, BK))
        amax_c = tl.max(ax3c, axis=1)                              # [NMB, BK]
        e_c = _e8m0_from_amax(amax_c, rcp, DIV_MODE)
        sinv_c = _scale_inv_from_e8m0(e_c)
        sinv_c_full = tl.reshape(tl.broadcast_to(sinv_c[:, None, :], (NMB, 32, BK)), (BM, BK))
        q_c = _cvt_e4m3_u8(x * sinv_c_full)
        tl.store(cw_ptr + rows64[:, None] * H + cols[None, :], q_c, mask=cvalid[None, :])
        mb = (p0 // 32) + tl.arange(0, NMB)                        # 32-row block index [NMB]
        ktc = cols // 128                                          # scale tile column index per column [BK]
        inner = (cols % 32) * 16 + ((cols // 32) % 4) * 4          # [BK]
        if CW_LAYOUT == 3:
            # TE grouped storage: group g's columnwise scales are one contiguous block (round_up(M_g/32, 4) x
            # round_up(K, 128) bytes) holding the single-tensor GEMM-swizzled layout of its [M_g, K] slice:
            # tiles row-major over (k/128, mb_local/4), inner (k%32)*16 + ((k//32)%4)*4 + mb_local%4
            starts = tl.cumsum(pc, axis=0) - pc                        # exclusive prefix: first row of each group
            gsel = (starts <= p0) & (eidx < E)
            gidx = tl.sum(gsel.to(tl.int32), axis=0) - 1
            m_g = tl.sum(tl.where(eidx == gidx, pc, 0), axis=0)
            start_g = tl.sum(tl.where(eidx == gidx, starts, 0), axis=0)
            mb_l = mb - start_g // 32                                  # [NMB]
            base_g = (start_g // 32).to(tl.int64) * K_PAD
            pos_c = base_g + ((ktc[None, :] * (m_g // 128) + (mb_l // 4)[:, None]) * 512 + inner[None, :] + (mb_l % 4)[:, None]).to(tl.int64)
        elif CW_LAYOUT == 4:
            # generic mixed-radix layout: digits of k = (k%rk0, (k//rk0)%rk1, k//(rk0*rk1)), digits of mb likewise,
            # each with its own byte stride (the ten cw_params)
            kk = cols.to(tl.int64)[None, :]
            mb64 = mb.to(tl.int64)[:, None]
            pos_c = ((kk % rk0) * sk0 + ((kk // rk0) % rk1) * sk1 + (kk // (rk0 * rk1)) * sk2
                     + (mb64 % rm0) * sm0 + ((mb64 // rm0) % rm1) * sm1 + (mb64 // (rm0 * rm1)) * sm2)
        elif CW_LAYOUT == 0:
            pos_c = ((ktc[None, :] * MB4 + (mb // 4)[:, None]) * 512 + inner[None, :] + (mb % 4)[:, None]).to(tl.int64)
        elif CW_LAYOUT == 1:
            pos_c = (((mb // 4)[:, None] * KT + ktc[None, :]) * 512 + inner[None, :] + (mb % 4)[:, None]).to(tl.int64)
        else:
            pos_c = (mb[:, None] * K_PAD + cols[None, :]).to(tl.int64)
        tl.store(cws_ptr + pos_c, e_c.to(tl.uint8), mask=cvalid[None, :])


@triton.jit
def _qperm_gather_quant_scaled_kernel(src_ptr, r2t_ptr, pprob_ptr, z_ptr, tpe_ptr, rw_ptr, rws_ptr, cw_ptr, cws_ptr,
                                      dprob_ptr, out_ptr, dbias_ptr, T, H, KT, MB4, KB_PAD, K_PAD, rcp,
                                      rk0, rk1, sk0, sk1, sk2, rm0, rm1, sm0, sm1, sm2,
                                      E: tl.constexpr, PAD: tl.constexpr, WRITE_BF16: tl.constexpr, COLSUM: tl.constexpr,
                                      RW_LAYOUT: tl.constexpr, CW_LAYOUT: tl.constexpr, DIV_MODE: tl.constexpr,
                                      BM: tl.constexpr, BK: tl.constexpr, NRB: tl.constexpr):
    """_qperm_gather_quant_kernel for the combine backward: the gathered row is scaled by its routing probability (rounded
    to bf16, as autograd's `x * p` would) before quantization, and dres[t] . z[p] is accumulated into dprob[p] (fp32 atomics
    over the BK tiles).  WRITE_BF16 also stores the bf16 dz rows (out_ptr [P, H]); COLSUM accumulates their fp32 column sum
    over the program's NRB blocks (one expert group, since PAD % (NRB * BM) == 0) into dbias_ptr [E, H], one atomic each."""
    kt = tl.program_id(0)
    pm = tl.program_id(1)
    eidx = tl.arange(0, E)
    cnt = tl.load(tpe_ptr + eidx)
    pc = ((cnt + (PAD - 1)) // PAD) * PAD
    n_used = tl.sum(pc, axis=0)
    pb = pm * (NRB * BM)                                               # first row of this program
    col0 = kt * BK
    cols = col0 + tl.arange(0, BK)
    cvalid = cols < H
    # the expert group of this program's rows (first row pb): exclusive prefix of the padded counts
    starts = tl.cumsum(pc, axis=0) - pc
    gsel = (starts <= pb) & (eidx < E)
    gidx = tl.sum(gsel.to(tl.int32), axis=0) - 1
    csum = tl.zeros([BK], dtype=tl.float32)
    for rb in range(NRB):
        p0 = pb + rb * BM
        if p0 < n_used:
            rows = p0 + tl.arange(0, BM)
            rows64 = rows.to(tl.int64)
            tok = tl.load(r2t_ptr + rows)
            rvalid = (tok >= 0) & (tok < T)
            m2 = rvalid[:, None] & cvalid[None, :]
            src = tl.load(src_ptr + tl.maximum(tok, 0).to(tl.int64)[:, None] * H + cols[None, :], mask=m2, other=0.0)
            v = src.to(tl.float32)
            # dprob[p] += dres[tok] . z[p] over this tile's columns
            zt = tl.load(z_ptr + rows64[:, None] * H + cols[None, :], mask=m2, other=0.0).to(tl.float32)
            part = tl.sum(v * zt, axis=1)
            tl.atomic_add(dprob_ptr + rows, part, mask=rvalid)
            # dz = bf16(p * dres[tok]) : the fc2 output gradient of the unscaled expert output
            p = tl.load(pprob_ptr + rows, mask=rvalid, other=0.0)
            x = (v * p[:, None]).to(tl.bfloat16).to(tl.float32)
            if WRITE_BF16:
                tl.store(out_ptr + rows64[:, None] * H + cols[None, :], x.to(tl.bfloat16), mask=cvalid[None, :])
            if COLSUM:
                csum += tl.sum(x, axis=0)                                    # [BK] fp32 partial over this block's rows
            ax = tl.abs(x)

            # ---- rowwise: NB blocks of 32 along H
            NB: tl.constexpr = BK // 32
            ax3 = tl.reshape(ax, (BM, NB, 32))
            amax_r = tl.max(ax3, axis=2)
            e_r = _e8m0_from_amax(amax_r, rcp, DIV_MODE)
            sinv_r = _scale_inv_from_e8m0(e_r)
            sinv_r_full = tl.reshape(tl.broadcast_to(sinv_r[:, :, None], (BM, NB, 32)), (BM, BK))
            q_r = _cvt_e4m3_u8(x * sinv_r_full)
            tl.store(rw_ptr + rows64[:, None] * H + cols[None, :], q_r, mask=cvalid[None, :])
            jb = tl.arange(0, NB)
            kb = (col0 // 32) + jb
            kb_valid = (kb * 32) < H
            if RW_LAYOUT == 0:
                pos_r = (((p0 // 128) * KT + kb // 4)[None, :] * 512 + (rows % 32)[:, None] * 16
                         + ((rows // 32) % 4)[:, None] * 4 + (kb % 4)[None, :])
            else:
                pos_r = rows[:, None] * KB_PAD + kb[None, :]
            tl.store(rws_ptr + pos_r.to(tl.int64), e_r.to(tl.uint8), mask=kb_valid[None, :])

            # ---- columnwise: NMB blocks of 32 rows per column
            NMB: tl.constexpr = BM // 32
            ax3c = tl.reshape(ax, (NMB, 32, BK))
            amax_c = tl.max(ax3c, axis=1)
            e_c = _e8m0_from_amax(amax_c, rcp, DIV_MODE)
            sinv_c = _scale_inv_from_e8m0(e_c)
            sinv_c_full = tl.reshape(tl.broadcast_to(sinv_c[:, None, :], (NMB, 32, BK)), (BM, BK))
            q_c = _cvt_e4m3_u8(x * sinv_c_full)
            tl.store(cw_ptr + rows64[:, None] * H + cols[None, :], q_c, mask=cvalid[None, :])
            mb = (p0 // 32) + tl.arange(0, NMB)
            ktc = cols // 128
            inner = (cols % 32) * 16 + ((cols // 32) % 4) * 4
            if CW_LAYOUT == 3:
                m_g = tl.sum(tl.where(eidx == gidx, pc, 0), axis=0)
                start_g = tl.sum(tl.where(eidx == gidx, starts, 0), axis=0)
                mb_l = mb - start_g // 32
                base_g = (start_g // 32).to(tl.int64) * K_PAD
                pos_c = base_g + ((ktc[None, :] * (m_g // 128) + (mb_l // 4)[:, None]) * 512 + inner[None, :] + (mb_l % 4)[:, None]).to(tl.int64)
            elif CW_LAYOUT == 4:
                kk = cols.to(tl.int64)[None, :]
                mb64 = mb.to(tl.int64)[:, None]
                pos_c = ((kk % rk0) * sk0 + ((kk // rk0) % rk1) * sk1 + (kk // (rk0 * rk1)) * sk2
                         + (mb64 % rm0) * sm0 + ((mb64 // rm0) % rm1) * sm1 + (mb64 // (rm0 * rm1)) * sm2)
            elif CW_LAYOUT == 0:
                pos_c = ((ktc[None, :] * MB4 + (mb // 4)[:, None]) * 512 + inner[None, :] + (mb % 4)[:, None]).to(tl.int64)
            elif CW_LAYOUT == 1:
                pos_c = (((mb // 4)[:, None] * KT + ktc[None, :]) * 512 + inner[None, :] + (mb % 4)[:, None]).to(tl.int64)
            else:
                pos_c = (mb[:, None] * K_PAD + cols[None, :]).to(tl.int64)
            tl.store(cws_ptr + pos_c, e_c.to(tl.uint8), mask=cvalid[None, :])
    if COLSUM:
        tl.atomic_add(dbias_ptr + gidx.to(tl.int64) * H + cols, csum, mask=cvalid)


# =============================================================================================
# host side
# =============================================================================================
def scale_buffer_numel(P: int, H: int) -> int:
    """Bytes of one GEMM-swizzled MXFP8 scale tensor for a [P, H] operand (TE: ceil(P/128) * ceil(H/128) * 512;
    the columnwise one, round_up(H, 128) * (P / 32), is the same number when P % 128 == 0)."""
    return ((P + 127) // 128) * ((H + 127) // 128) * 512


def _check_shapes(P: int, H: int, E: int, pad_multiple: int):
    assert P % 128 == 0 and P % pad_multiple == 0, (P, pad_multiple)
    assert H % 32 == 0, H
    assert BLOCK_M % 32 == 0 and 128 % BLOCK_M == 0 and BLOCK_K % 128 == 0, (BLOCK_M, BLOCK_K)
    assert pad_multiple % BLOCK_M == 0, pad_multiple
    assert E & (E - 1) == 0, E                       # tl.arange over the experts needs a power of two


@torch.library.custom_op("gpt_oss::moe_qperm_inverse_map", mutates_args=())
def moe_qperm_inverse_map(dense_to_expert_map: torch.Tensor, probs: torch.Tensor, dense_chunk_layout: torch.Tensor,
                          P: int, has_probs: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    assert dense_to_expert_map.dim() == 2 and dense_to_expert_map.dtype == torch.int32 and dense_to_expert_map.is_contiguous()
    assert dense_chunk_layout.dtype == torch.int32 and dense_chunk_layout.numel() >= 1 and dense_chunk_layout.is_contiguous()
    T_rows, E = dense_to_expert_map.shape
    assert E & (E - 1) == 0, E
    dev = dense_to_expert_map.device
    row_to_token = torch.full((P,), -1, dtype=torch.int32, device=dev)
    if has_probs:
        assert probs.shape == (T_rows, E) and probs.dtype == torch.float32 and probs.is_contiguous(), (probs.shape, probs.dtype)
        permuted_probs = torch.zeros((P,), dtype=torch.float32, device=dev)
    else:
        permuted_probs = torch.empty((0,), dtype=torch.float32, device=dev)
    if T_rows == 0 or P == 0:
        return row_to_token, permuted_probs
    BLOCK_T = max(1, 1024 // E)
    _qperm_inverse_map_kernel[(triton.cdiv(T_rows, BLOCK_T),)](
        dense_to_expert_map, probs if has_probs else dense_to_expert_map, dense_chunk_layout, row_to_token,
        permuted_probs if has_probs else row_to_token, dense_chunk_layout.numel(), T_rows, P,
        E=E, HAS_PROBS=has_probs, BLOCK_T=BLOCK_T, num_warps=4)
    return row_to_token, permuted_probs


@moe_qperm_inverse_map.register_fake
def _(dense_to_expert_map, probs, dense_chunk_layout, P, has_probs):
    dev = dense_to_expert_map.device
    return (torch.empty((P,), dtype=torch.int32, device=dev),
            torch.empty((P if has_probs else 0,), dtype=torch.float32, device=dev))


@torch.library.custom_op("gpt_oss::moe_qperm_gather_quant", mutates_args=())
def moe_qperm_gather_quant(src: torch.Tensor, row_to_token: torch.Tensor, tokens_per_expert: torch.Tensor, P: int,
                           pad_multiple: int, write_bf16: bool, rw_layout: int, cw_layout: int, div_mode: int,
                           cw_params: List[int]
                           ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert src.dim() == 2 and src.dtype == torch.bfloat16 and src.is_contiguous(), (src.shape, src.dtype)
    T, H = src.shape
    E = tokens_per_expert.numel()
    assert tokens_per_expert.dtype == torch.int32 and tokens_per_expert.is_contiguous()
    assert row_to_token.shape == (P,) and row_to_token.dtype == torch.int32 and row_to_token.is_contiguous()
    _check_shapes(P, H, E, pad_multiple)
    if _CUDA is not None and _CUDA.enabled_for(rw_layout, cw_layout, div_mode, pad_multiple, H, E):
        try:
            return _CUDA.gather_quant(src, row_to_token, tokens_per_expert, P, pad_multiple, write_bf16)
        except Exception as ex:  # noqa
            _CUDA.disable(f"gather_quant raised {type(ex).__name__}: {str(ex)[:400]}; Triton kernels for the rest of the run")
    dev = src.device
    n_scale = scale_buffer_numel(P, H)
    rw = torch.empty((P * H,), dtype=torch.uint8, device=dev)
    cw = torch.empty((P * H,), dtype=torch.uint8, device=dev)
    rws = torch.zeros((n_scale,), dtype=torch.uint8, device=dev)    # padding entries must read as zero
    cws = torch.zeros((n_scale,), dtype=torch.uint8, device=dev)
    out = torch.empty((P, H) if write_bf16 else (0, H), dtype=torch.bfloat16, device=dev)
    if P == 0:
        return rw, rws, cw, cws, out
    KT = (H + 127) // 128                            # 128-column scale tiles
    MB4 = P // 128
    KB_PAD = ((H // 32 + 3) // 4) * 4
    K_PAD = ((H + 127) // 128) * 128
    grid = ((H + BLOCK_K - 1) // BLOCK_K, P // BLOCK_M)
    gp = list(cw_params) if cw_params else []
    if cw_layout == 4:
        assert len(gp) == 10, gp
    else:
        gp = [1, 1, 0, 0, 0, 1, 1, 0, 0, 0]
    _qperm_gather_quant_kernel[grid](
        src, row_to_token, tokens_per_expert, rw, rws, cw, cws, out if write_bf16 else src,
        T, H, KT, MB4, KB_PAD, K_PAD, E4M3_MAX_RCP, *gp,
        E=E, PAD=pad_multiple, WRITE_BF16=write_bf16, RW_LAYOUT=rw_layout, CW_LAYOUT=cw_layout, DIV_MODE=div_mode,
        BM=BLOCK_M, BK=BLOCK_K, num_warps=NUM_WARPS)
    return rw, rws, cw, cws, out


@moe_qperm_gather_quant.register_fake
def _(src, row_to_token, tokens_per_expert, P, pad_multiple, write_bf16, rw_layout, cw_layout, div_mode, cw_params):
    T, H = src.shape
    dev = src.device
    n_scale = scale_buffer_numel(P, H)
    return (torch.empty((P * H,), dtype=torch.uint8, device=dev), torch.empty((n_scale,), dtype=torch.uint8, device=dev),
            torch.empty((P * H,), dtype=torch.uint8, device=dev), torch.empty((n_scale,), dtype=torch.uint8, device=dev),
            torch.empty((P, H) if write_bf16 else (0, H), dtype=torch.bfloat16, device=dev))


@torch.library.custom_op("gpt_oss::moe_qperm_gather_quant_scaled", mutates_args=())
def moe_qperm_gather_quant_scaled(src: torch.Tensor, row_to_token: torch.Tensor, permuted_probs: torch.Tensor,
                                  z: torch.Tensor, tokens_per_expert: torch.Tensor, P: int, pad_multiple: int,
                                  write_bf16: bool, colsum: bool, rw_layout: int, cw_layout: int, div_mode: int,
                                  cw_params: List[int]
                                  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                             torch.Tensor, torch.Tensor]:
    """Combine backward: MXFP8 rowwise/columnwise of dz = bf16(p * dres[row]), dprob = dres[row] . z, and optionally
    the bf16 dz rows (write_bf16) and/or their fp32 per-expert column sum (colsum) = the fc2 bias gradient."""
    assert src.dim() == 2 and src.dtype == torch.bfloat16 and src.is_contiguous(), (src.shape, src.dtype)
    T, H = src.shape
    E = tokens_per_expert.numel()
    assert tokens_per_expert.dtype == torch.int32 and tokens_per_expert.is_contiguous()
    assert row_to_token.shape == (P,) and row_to_token.dtype == torch.int32 and row_to_token.is_contiguous()
    assert permuted_probs.shape == (P,) and permuted_probs.dtype == torch.float32 and permuted_probs.is_contiguous(), \
        (permuted_probs.shape, permuted_probs.dtype)
    assert z.shape == (P, H) and z.dtype == torch.bfloat16 and z.is_contiguous(), (z.shape, z.dtype)
    _check_shapes(P, H, E, pad_multiple)
    if _CUDA is not None and _CUDA.enabled_for(rw_layout, cw_layout, div_mode, pad_multiple, H, E):
        try:
            return _CUDA.gather_quant_scaled(src, row_to_token, permuted_probs, z, tokens_per_expert, P, pad_multiple,
                                             write_bf16, colsum)
        except Exception as ex:  # noqa
            _CUDA.disable(f"gather_quant_scaled raised {type(ex).__name__}: {str(ex)[:400]}; Triton kernels for the rest of the run")
    dev = src.device
    n_scale = scale_buffer_numel(P, H)
    rw = torch.empty((P * H,), dtype=torch.uint8, device=dev)
    cw = torch.empty((P * H,), dtype=torch.uint8, device=dev)
    rws = torch.zeros((n_scale,), dtype=torch.uint8, device=dev)
    cws = torch.zeros((n_scale,), dtype=torch.uint8, device=dev)
    dprob = torch.zeros((P,), dtype=torch.float32, device=dev)
    out = torch.empty((P, H) if write_bf16 else (0, H), dtype=torch.bfloat16, device=dev)
    dbias = torch.zeros((E, H) if colsum else (0, H), dtype=torch.float32, device=dev)
    if P == 0:
        return rw, rws, cw, cws, dprob, out, dbias
    assert NRB >= 1 and pad_multiple % (NRB * BLOCK_M) == 0 and P % (NRB * BLOCK_M) == 0, (NRB, BLOCK_M, pad_multiple, P)
    KT = (H + 127) // 128
    MB4 = P // 128
    KB_PAD = ((H // 32 + 3) // 4) * 4
    K_PAD = ((H + 127) // 128) * 128
    grid = ((H + BLOCK_K - 1) // BLOCK_K, P // (BLOCK_M * NRB))
    gp = list(cw_params) if cw_params else []
    if cw_layout == 4:
        assert len(gp) == 10, gp
    else:
        gp = [1, 1, 0, 0, 0, 1, 1, 0, 0, 0]
    _qperm_gather_quant_scaled_kernel[grid](
        src, row_to_token, permuted_probs, z, tokens_per_expert, rw, rws, cw, cws, dprob,
        out if write_bf16 else src, dbias if colsum else dprob,
        T, H, KT, MB4, KB_PAD, K_PAD, E4M3_MAX_RCP, *gp,
        E=E, PAD=pad_multiple, WRITE_BF16=write_bf16, COLSUM=colsum, RW_LAYOUT=rw_layout, CW_LAYOUT=cw_layout,
        DIV_MODE=div_mode, BM=BLOCK_M, BK=BLOCK_K, NRB=NRB, num_warps=NUM_WARPS)
    return rw, rws, cw, cws, dprob, out, dbias


@moe_qperm_gather_quant_scaled.register_fake
def _(src, row_to_token, permuted_probs, z, tokens_per_expert, P, pad_multiple, write_bf16, colsum, rw_layout, cw_layout,
      div_mode, cw_params):
    T, H = src.shape
    E = tokens_per_expert.numel()
    dev = src.device
    n_scale = scale_buffer_numel(P, H)
    return (torch.empty((P * H,), dtype=torch.uint8, device=dev), torch.empty((n_scale,), dtype=torch.uint8, device=dev),
            torch.empty((P * H,), dtype=torch.uint8, device=dev), torch.empty((n_scale,), dtype=torch.uint8, device=dev),
            torch.empty((P,), dtype=torch.float32, device=dev),
            torch.empty((P, H) if write_bf16 else (0, H), dtype=torch.bfloat16, device=dev),
            torch.empty((E, H) if colsum else (0, H), dtype=torch.float32, device=dev))


# =============================================================================================
# torch references (TE's MXFP8 math in torch) and the fixed-name contract
# =============================================================================================
def e8m0_reference(amax: torch.Tensor, div_mode: int = 0) -> torch.Tensor:
    v = amax * torch.tensor(E4M3_MAX_RCP, dtype=torch.float32, device=amax.device) if div_mode == 0 else amax / 448.0
    bits = v.view(torch.int32)
    exp = (bits >> 23) & 0xFF
    man = bits & 0x7FFFFF
    e = exp + ((man > 0) & (exp != 254)).to(torch.int32)
    e = torch.where(v == 0.0, torch.zeros_like(e), e)
    return e


def scale_inv_reference(e: torch.Tensor) -> torch.Tensor:
    s = torch.ldexp(torch.ones_like(e, dtype=torch.float32), (127 - e).to(torch.int32))
    return torch.where(e == 0, torch.ones_like(s), s)


def e4m3_reference(v: torch.Tensor) -> torch.Tensor:
    """fp32 -> e4m3 bytes with saturation to +-448 (torch's cast is round-to-nearest-even)."""
    v = v.clamp(-448.0, 448.0)
    return v.to(torch.float8_e4m3fn).view(torch.uint8)


def swizzled_positions(rows: torch.Tensor, cols: torch.Tensor, n_col_tiles: int) -> torch.Tensor:
    """byte offset of scale (row r, block column c) in the GEMM-swizzled layout (tiles row-major over (r/128, c/4))."""
    return ((rows // 128) * n_col_tiles + cols // 4) * 512 + (rows % 32) * 16 + ((rows // 32) % 4) * 4 + (cols % 4)


def grouped_columnwise_positions(k: torch.Tensor, mb: torch.Tensor, padded_counts: torch.Tensor, H: int) -> torch.Tensor:
    """byte offset of columnwise scale (column k, 32-row block mb) in TE's grouped storage (layout 3): group g's
    block starts at (mb_start_g) * round_up(H, 128) and holds its slice's swizzled layout, tiles row-major over
    (k/128, mb_local/4)."""
    K_PAD = ((H + 127) // 128) * 128
    pc = padded_counts.to(torch.int64)
    starts = torch.cumsum(pc, 0) - pc                      # first row of each group
    row = mb * 32
    g = (starts[None, :] <= row.reshape(-1)[:, None]).sum(dim=1) - 1
    g = g.reshape(mb.shape)
    m_g = pc[g]
    start_g = starts[g]
    mb_l = (row - start_g) // 32
    return (start_g // 32) * K_PAD + ((k // 128) * (m_g // 128) + mb_l // 4) * 512 + (k % 32) * 16 + ((k // 32) % 4) * 4 + (mb_l % 4)


def gather_quant_reference(src, row_to_token, tokens_per_expert, P, pad_multiple, div_mode=0):
    """Dense-layout reference: bf16 gather, rowwise/columnwise MXFP8 bytes [P, H] and swizzled scale buffers (layout 0)."""
    T, H = src.shape
    dev = src.device
    counts = tokens_per_expert.to(torch.int64)
    n_used = int((((counts + pad_multiple - 1) // pad_multiple) * pad_multiple).sum().item())
    tok = row_to_token[:n_used].to(torch.int64)
    valid = (tok >= 0) & (tok < T)
    rows = torch.where(valid[:, None], src[tok.clamp(min=0)], torch.zeros((), dtype=src.dtype, device=dev))   # [n_used, H] bf16
    x = rows.float()
    # rowwise
    amax_r = x.abs().view(n_used, H // 32, 32).amax(dim=2)
    e_r = e8m0_reference(amax_r, div_mode)
    q_r = e4m3_reference(x.view(n_used, H // 32, 32) * scale_inv_reference(e_r)[:, :, None]).view(n_used, H)
    # columnwise
    amax_c = x.abs().view(n_used // 32, 32, H).amax(dim=1)
    e_c = e8m0_reference(amax_c, div_mode)
    q_c = e4m3_reference(x.view(n_used // 32, 32, H) * scale_inv_reference(e_c)[:, None, :]).view(n_used, H)
    KT = (H + 127) // 128
    MB4 = P // 128
    rws = torch.zeros(scale_buffer_numel(P, H), dtype=torch.uint8, device=dev)
    cws = torch.zeros_like(rws)
    r = torch.arange(n_used, device=dev)[:, None].expand(n_used, H // 32)
    c = torch.arange(H // 32, device=dev)[None, :].expand(n_used, H // 32)
    rws[swizzled_positions(r, c, KT).reshape(-1)] = e_r.to(torch.uint8).reshape(-1)
    k = torch.arange(H, device=dev)[:, None].expand(H, n_used // 32)
    mb = torch.arange(n_used // 32, device=dev)[None, :].expand(H, n_used // 32)
    padded = ((counts + pad_multiple - 1) // pad_multiple) * pad_multiple
    cws[grouped_columnwise_positions(k, mb, padded, H).reshape(-1)] = e_c.t().to(torch.uint8).reshape(-1)
    return rows, q_r, q_c, rws, cws, n_used


def make_routing(num_tokens, E, topk, pad_multiple, device="cuda", seed=0, drop_frac=0.0):
    """Synthetic HybridEP dense layout: per-expert chunks aligned to pad_multiple; map[t, e] = row or -1."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    experts = torch.stack([torch.randperm(E, generator=g)[:topk] for _ in range(num_tokens)])      # [T, topk]
    if drop_frac > 0:
        keep = torch.rand(num_tokens, topk, generator=g) >= drop_frac
        experts = torch.where(keep, experts, torch.full_like(experts, -1))
    counts = torch.zeros(E, dtype=torch.long)
    for e in range(E):
        counts[e] = (experts == e).sum()
    padded = (counts + pad_multiple - 1) // pad_multiple * pad_multiple
    starts = torch.cumsum(padded, 0) - padded
    m = torch.full((num_tokens, E), -1, dtype=torch.int32)
    fill = starts.clone()
    for t in range(num_tokens):
        for e in experts[t].tolist():
            if e >= 0:
                m[t, e] = int(fill[e]); fill[e] += 1
    P = int(padded.sum())
    layout = torch.tensor([num_tokens], dtype=torch.int32)
    probs = torch.rand(num_tokens, E, generator=g)
    probs = torch.where(m >= 0, probs, torch.zeros_like(probs)).float()
    return m.to(device), layout.to(device), counts.to(torch.int32).to(device), probs.to(device), P


def kernel_fn(src, dense_to_expert_map, probs, dense_chunk_layout, tokens_per_expert, P, pad_multiple):
    r2t, pp = torch.ops.gpt_oss.moe_qperm_inverse_map(dense_to_expert_map, probs, dense_chunk_layout, P, True)
    rw, rws, cw, cws, out = torch.ops.gpt_oss.moe_qperm_gather_quant(src, r2t, tokens_per_expert, P, pad_multiple, True, 0, 3, 0, [])
    return out, rw, cw, rws, cws, pp


def reference_fn(src, dense_to_expert_map, probs, dense_chunk_layout, tokens_per_expert, P, pad_multiple):
    T, H = src.shape
    n_dense = int(dense_chunk_layout[-1].item())
    m = dense_to_expert_map[:n_dense].long()
    r2t = torch.full((P,), -1, dtype=torch.int64, device=src.device)
    pp = torch.zeros((P,), dtype=torch.float32, device=src.device)
    t_idx = torch.arange(n_dense, device=src.device)[:, None].expand_as(m)
    ok = m >= 0
    r2t[m[ok]] = t_idx[ok]
    pp[m[ok]] = probs[:n_dense][ok]
    rows, q_r, q_c, rws, cws, n_used = gather_quant_reference(src, r2t.to(torch.int32), tokens_per_expert, P, pad_multiple)
    out = torch.zeros((P, H), dtype=src.dtype, device=src.device)
    out[:n_used] = rows
    rw = torch.zeros((P, H), dtype=torch.uint8, device=src.device); rw[:n_used] = q_r
    cw = torch.zeros((P, H), dtype=torch.uint8, device=src.device); cw[:n_used] = q_c
    return out, rw.reshape(-1), cw.reshape(-1), rws, cws, pp


def get_inputs(num_tokens=1024, H=2880, E=32, topk=4, pad_multiple=256, P=None):
    m, layout, counts, probs, P_min = make_routing(num_tokens, E, topk, pad_multiple, drop_frac=0.05)
    if P is None:
        P = ((P_min + 127) // 128 + 2) * 128
        P += -P % pad_multiple
    src = (torch.randn(num_tokens, H, device="cuda") * 3.0).to(torch.bfloat16)
    return [src, m, probs, layout, counts, P, pad_multiple]


def grouped_colsum_reference(rows_bf16, tokens_per_expert, pad_multiple):
    """[E, H] fp32: per-expert column sums of the bf16 rows (group g = padded rows [start_g, start_g + m_g)), fp32."""
    E = tokens_per_expert.numel()
    padded = ((tokens_per_expert.to(torch.int64) + pad_multiple - 1) // pad_multiple) * pad_multiple
    starts = torch.cumsum(padded, 0) - padded
    out = torch.zeros((E, rows_bf16.shape[1]), dtype=torch.float32, device=rows_bf16.device)
    for g in range(E):
        s0, m = int(starts[g].item()), int(padded[g].item())
        if m > 0:
            out[g] = rows_bf16[s0:s0 + m].float().sum(dim=0)
    return out


def gather_quant_scaled_reference(dres, row_to_token, permuted_probs, z, tokens_per_expert, P, pad_multiple):
    """dz_bf16 = bf16(p * dres[row]) rows (zero on padding), its MXFP8 quantization (via gather_quant_reference of a
    pre-gathered tensor) and dprob = dres[row] . z (fp32)."""
    T, H = dres.shape
    dev = dres.device
    tok = row_to_token.to(torch.int64)
    valid = (tok >= 0) & (tok < T)
    rows = torch.where(valid[:, None], dres[tok.clamp(min=0)].float(), torch.zeros((), device=dev))
    dz = (rows * permuted_probs[:, None]).to(torch.bfloat16)
    dprob = torch.where(valid, (rows * z.float()).sum(dim=1), torch.zeros((), device=dev))
    ident = torch.arange(P, device=dev, dtype=torch.int32)
    _, q_r, q_c, rws, cws, n_used = gather_quant_reference(dz, ident, tokens_per_expert, P, pad_multiple)
    return dz, q_r, q_c, rws, cws, dprob, n_used


if __name__ == "__main__":
    inp = get_inputs()
    got = kernel_fn(*inp)
    ref = reference_fn(*inp)
    # combine-backward variant
    src, m, probs, layout, counts, P, pad = inp
    r2t, pp = torch.ops.gpt_oss.moe_qperm_inverse_map(m, probs, layout, P, True)
    z = (torch.randn(P, src.shape[1], device="cuda") * 2.0).to(torch.bfloat16)
    rw2, rws2, cw2, cws2, dprob2, _, _ = torch.ops.gpt_oss.moe_qperm_gather_quant_scaled(src, r2t, pp, z, counts, P, pad, False, False, 0, 3, 0, [])
    dz_ref, qr_ref, qc_ref, rws_ref, cws_ref, dprob_ref, nu = gather_quant_scaled_reference(src, r2t, pp, z, counts, P, pad)
    Hh = src.shape[1]
    print("scaled: rowwise data bitwise:", torch.equal(rw2[: nu * Hh], qr_ref.reshape(-1)[: nu * Hh]),
          "columnwise data bitwise:", torch.equal(cw2[: nu * Hh], qc_ref.reshape(-1)[: nu * Hh]),
          "rws bitwise:", torch.equal(rws2, rws_ref), "cws bitwise:", torch.equal(cws2, cws_ref))
    dd = (dprob2 - dprob_ref).abs().max().item()
    print("scaled: dprob max|diff|:", dd, "max|ref|:", dprob_ref.abs().max().item())
    rw3, rws3, cw3, cws3, dprob3, dz3, db3 = torch.ops.gpt_oss.moe_qperm_gather_quant_scaled(src, r2t, pp, z, counts, P, pad, True, True, 0, 3, 0, [])
    db_ref = grouped_colsum_reference(dz_ref, counts, pad)
    print("scaled(bf16+colsum): quantized outputs bitwise the plain call:",
          torch.equal(rw3, rw2) and torch.equal(cw3, cw2) and torch.equal(rws3, rws2) and torch.equal(cws3, cws2),
          "dz rows bitwise:", torch.equal(dz3[:nu], dz_ref[:nu]),
          "dbias max|diff|:", (db3 - db_ref).abs().max().item(), "max|ref|:", db_ref.abs().max().item())
    n_used = int(((inp[4].long() + inp[6] - 1) // inp[6] * inp[6]).sum().item())
    H = inp[0].shape[1]
    print("bf16 rows bitwise:", torch.equal(got[0][:n_used], ref[0][:n_used]))
    print("rowwise data bitwise:", torch.equal(got[1][: n_used * H], ref[1][: n_used * H]))
    print("columnwise data bitwise:", torch.equal(got[2][: n_used * H], ref[2][: n_used * H]))
    print("rowwise scales bitwise:", torch.equal(got[3], ref[3]), " columnwise scales bitwise:", torch.equal(got[4], ref[4]))
    print("permuted probs bitwise:", torch.equal(got[5], ref[5]))
