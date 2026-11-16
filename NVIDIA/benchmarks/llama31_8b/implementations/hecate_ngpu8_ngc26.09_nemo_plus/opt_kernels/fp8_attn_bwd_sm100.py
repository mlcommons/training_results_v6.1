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
#
# Adapted from the FlashAttention-4 SM100 backward kernel
# (flash_attn/cute/flash_bwd_sm100.py) and its helpers (flash_attn/cute: mma_sm100_desc.py,
# blackwell_helpers.py, copy_utils.py, pipeline.py, named_barrier.py, utils.py), and from
# quack (quack/layout_utils.py). Those portions are subject to the following license:
#
# BSD 3-Clause License
#
# Copyright (c) 2025, Ted Zadouri, Markus Hoehnerbach, Jay Shah, Tri Dao (flash_bwd_sm100.py)
# Copyright (c) 2025, Tri Dao (mma_sm100_desc.py, blackwell_helpers.py, pipeline.py, utils.py)
# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao (copy_utils.py, quack/layout_utils.py)
# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani,
#     Tri Dao (named_barrier.py)
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import enum
import math
import sys
from dataclasses import dataclass
from enum import IntEnum
from functools import partial
from types import SimpleNamespace
from typing import Callable, Optional, Tuple, Type

import torch

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline
import cutlass.utils.blackwell_helpers as sm100_utils_basic
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import Boolean, Float32, Int32, const_expr
from cutlass._mlir.dialects import llvm, nvvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op, if_generate
from cutlass.pipeline import PipelineState, PipelineUserType
from cutlass.pipeline import PipelineTmaUmma as PipelineTmaUmmaOg
from cutlass.utils import LayoutEnum

class Major(IntEnum):
    K = 0
    MN = 1


class ScaleIn(IntEnum):
    One = 0
    Neg = 1


class Saturate(IntEnum):
    False_ = 0
    True_ = 1


class CFormat(IntEnum):
    F16 = 0
    F32 = 1
    S32 = 2


class F16F32Format(IntEnum):
    F16 = 0
    BF16 = 1
    TF32 = 2


class S8Format(IntEnum):
    UINT8 = 0
    INT8 = 1


class MXF8F6F4Format(IntEnum):
    E4M3 = 0
    E5M2 = 1
    E2M3 = 3
    E3M2 = 4
    E2M1 = 5


class MaxShift(IntEnum):
    NoShift = 0
    MaxShift8 = 1
    MaxShift16 = 2
    MaxShift32 = 3


def to_UMMA_format(cutlass_type) -> int:
    if cutlass_type is cutlass.Int8:
        return S8Format.INT8
    if cutlass_type is cutlass.Uint8:
        return S8Format.UINT8
    if cutlass_type is cutlass.Float16:
        return F16F32Format.F16
    if cutlass_type is cutlass.BFloat16:
        return F16F32Format.BF16
    if cutlass_type is cutlass.TFloat32:
        return F16F32Format.TF32
    if cutlass_type is cutlass.Float8E4M3FN:
        return MXF8F6F4Format.E4M3
    if cutlass_type is cutlass.Float8E5M2:
        return MXF8F6F4Format.E5M2
    raise TypeError(f"Unsupported CUTLASS scalar type for A/B: {cutlass_type!r}")


def to_C_format(cutlass_type) -> int:
    if cutlass_type is cutlass.Float16:
        return CFormat.F16
    if cutlass_type is cutlass.Float32:
        return CFormat.F32
    if cutlass_type is cutlass.Int32:
        return CFormat.S32
    raise TypeError(f"Unsupported CUTLASS scalar type for accumulator: {cutlass_type!r}")


def make_instr_desc(
    a_type,
    b_type,
    c_type,
    M: int,
    N: int,
    a_major: Major,
    b_major: Major,
    a_neg: ScaleIn = ScaleIn.One,
    b_neg: ScaleIn = ScaleIn.One,
    c_sat: Saturate = Saturate.False_,
    is_sparse: bool = False,
    max_shift: MaxShift = MaxShift.NoShift,
) -> int:
    a_fmt = int(to_UMMA_format(a_type))
    b_fmt = int(to_UMMA_format(b_type))
    c_fmt = int(to_C_format(c_type))

    if M not in (64, 128, 256):
        raise ValueError("M must be 64, 128 or 256")
    if N < 8 or N > 256 or (N & 7):
        raise ValueError("N must be a multiple of 8 in the range 8…256")

    m_dim = M >> 4
    n_dim = N >> 3

    # fmt: off
    desc = 0
    desc |= (0                 & 0x3) << 0
    desc |= (int(is_sparse)    & 0x1) << 2
    desc |= (int(c_sat)        & 0x1) << 3
    desc |= (c_fmt             & 0x3) << 4
    desc |= (a_fmt             & 0x7) << 7
    desc |= (b_fmt             & 0x7) << 10
    desc |= (int(a_neg)        & 0x1) << 13
    desc |= (int(b_neg)        & 0x1) << 14
    desc |= (int(a_major)      & 0x1) << 15
    desc |= (int(b_major)      & 0x1) << 16
    desc |= (n_dim             & 0x3F) << 17
    desc |= (m_dim             & 0x1F) << 24
    desc |= (int(max_shift)    & 0x3) << 30
    # fmt: on

    return desc & 0xFFFF_FFFF


def mma_op_to_idesc(op: cute.nvgpu.tcgen05.mma.MmaOp):
    return make_instr_desc(
        op.a_dtype,
        op.b_dtype,
        op.acc_dtype,
        op.shape_mnk[0],
        op.shape_mnk[1],
        Major.K if op.a_major_mode == cute.nvgpu.tcgen05.mma.OperandMajorMode.K else Major.MN,
        Major.K if op.b_major_mode == cute.nvgpu.tcgen05.mma.OperandMajorMode.K else Major.MN,
    )


class LayoutType(IntEnum):
    SWIZZLE_NONE = 0
    SWIZZLE_128B_BASE32B = 1
    SWIZZLE_128B = 2
    SWIZZLE_64B = 4
    SWIZZLE_32B = 6


def _layout_type(swizzle: cute.Swizzle) -> LayoutType:
    B, M, S = swizzle.num_bits, swizzle.num_base, swizzle.num_shift

    if M == 4:
        if S != 3:
            raise ValueError("Unexpected swizzle shift – want S==3 for M==4")
        return {
            0: LayoutType.SWIZZLE_NONE,
            1: LayoutType.SWIZZLE_32B,
            2: LayoutType.SWIZZLE_64B,
            3: LayoutType.SWIZZLE_128B,
        }[B]
    if M == 5:
        if (B, S) != (2, 2):
            raise ValueError("Only Swizzle<2,5,2> supported for 128B_BASE32B")
        return LayoutType.SWIZZLE_128B_BASE32B

    raise ValueError("Unsupported swizzle triple for UMMA smem descriptor")


def make_smem_desc_base(layout: cute.Layout, swizzle: cute.Swizzle, major: Major) -> int:
    layout_type = _layout_type(swizzle)

    VERSION = 1
    LBO_MODE = 0
    BASE_OFFSET = 0

    swizzle_atom_mn_size = {
        LayoutType.SWIZZLE_NONE: 1,
        LayoutType.SWIZZLE_32B: 2,
        LayoutType.SWIZZLE_64B: 4,
        LayoutType.SWIZZLE_128B: 8,
        LayoutType.SWIZZLE_128B_BASE32B: 8,
    }[layout_type]

    if major is Major.MN:
        swizzle_atom_k_size = 4 if layout_type is LayoutType.SWIZZLE_128B_BASE32B else 8
        canonical_layout = cute.logical_divide(layout, (swizzle_atom_mn_size, swizzle_atom_k_size))
        if not cute.is_congruent(canonical_layout, ((1, 1), (1, 1))):
            raise ValueError("Not a canonical UMMA_MN Layout: Expected profile failure.")
        stride_00 = canonical_layout.stride[0][0]
        if layout_type is not LayoutType.SWIZZLE_NONE and stride_00 != 1:
            raise ValueError("Not a canonical UMMA_MN Layout: Expected stride failure.")
        stride_10 = canonical_layout.stride[1][0]
        if stride_10 != swizzle_atom_mn_size:
            raise ValueError("Not a canonical UMMA_MN Layout: Expected stride failure.")
        stride_01, stride_11 = canonical_layout.stride[0][1], canonical_layout.stride[1][1]
        if layout_type is LayoutType.SWIZZLE_NONE:
            stride_byte_offset, leading_byte_offset = stride_01, stride_11
        else:
            stride_byte_offset, leading_byte_offset = stride_11, stride_01
    else:
        if layout_type == LayoutType.SWIZZLE_128B_BASE32B:
            raise ValueError("SWIZZLE_128B_BASE32B is invalid for Major-K")
        if not cute.size(layout.shape[0]) % 8 == 0:
            raise ValueError("Not a canonical UMMA_K Layout: Expected MN-size multiple of 8.")
        canonical_layout = cute.logical_divide(layout, (8, 2))
        if not cute.is_congruent(canonical_layout, ((1, 1), (1, 1))):
            raise ValueError("Not a canonical UMMA_K Layout: Expected profile failure.")
        stride_00 = canonical_layout.stride[0][0]
        if stride_00 != swizzle_atom_mn_size:
            raise ValueError("Not a canonical UMMA_K Layout: Expected stride failure.")
        stride_10 = canonical_layout.stride[1][0]
        if layout_type is not LayoutType.SWIZZLE_NONE and stride_10 != 1:
            raise ValueError("Not a canonical UMMA_K Layout: Expected stride failure.")
        stride_01 = canonical_layout.stride[0][1]
        stride_byte_offset, leading_byte_offset = stride_01, stride_10

    desc = 0
    desc |= (leading_byte_offset & 0x3FFF) << 16
    desc |= (stride_byte_offset & 0x3FFF) << 32
    desc |= (VERSION & 0x3) << 46
    desc |= (BASE_OFFSET & 0x7) << 49
    desc |= (LBO_MODE & 0x1) << 52
    desc |= (int(layout_type) & 0x7) << 61

    return desc & 0xFFFF_FFFF_FFFF_FFFF


def make_smem_desc_start_addr(start_addr: cute.Pointer) -> cutlass.Int32:
    return (start_addr.toint() & 0x3FFFF) >> 4


sm100_desc = sys.modules[__name__]


def _tcgen05_mma_kind(op: cute.nvgpu.tcgen05.mma.MmaOp) -> str:
    if isinstance(op, tcgen05.mma.MmaF16BF16Op):
        return "f16"
    if isinstance(op, tcgen05.mma.MmaTF32Op):
        return "tf32"
    if isinstance(op, tcgen05.mma.MmaI8Op):
        return "i8"
    if isinstance(op, (tcgen05.mma.MmaFP8Op, tcgen05.mma.MmaF8F6F4Op)):
        return "f8f6f4"
    if isinstance(op, tcgen05.mma.MmaMXF8Op):
        return "mxf8f6f4"
    if isinstance(op, tcgen05.mma.MmaMXF4Op):
        return "mxf4"
    if isinstance(op, tcgen05.mma.MmaMXF4NVF4Op):
        return "mxf4nvf4"
    raise TypeError(f"Unsupported tcgen05 MMA op kind: {type(op).__name__}")


@cute.jit
def gemm_w_idx(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    A_idx: Optional[Int32] = None,
    B_idx: Optional[Int32] = None,
    zero_init: bool | Boolean = False,
    swap_AB: bool = False,
    num_unroll_groups: int = 1,
) -> None:
    if const_expr(swap_AB):
        return gemm_w_idx(
            tiled_mma, acc, tCrB, tCrA, B_idx, A_idx, zero_init=zero_init, swap_AB=False
        )
    else:
        rA = tCrA if const_expr(A_idx is None) else tCrA[None, None, None, A_idx]
        rB = tCrB if const_expr(B_idx is None) else tCrB[None, None, None, B_idx]

        mma_atom = cute.make_mma_atom(tiled_mma.op)
        for k in cutlass.range(
            cute.size(tCrA.shape[2]), unroll=cute.size(tCrA.shape[2]) // num_unroll_groups
        ):
            mma_atom.set(tcgen05.Field.ACCUMULATE, not zero_init or k != 0)
            cute.gemm(mma_atom, acc, rA[None, None, k], rB[None, None, k], acc)


@cute.jit
def gemm_ptx_w_idx(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    sA: Optional[cute.Tensor],
    sB: cute.Tensor,
    A_idx: Optional[Int32] = None,
    B_idx: Optional[Int32] = None,
    zero_init: bool | Boolean = False,
    cta_group: int = 1,
    **kwargs,
) -> None:
    rA = tCrA if const_expr(A_idx is None) else tCrA[None, None, None, A_idx]
    rB = tCrB if const_expr(B_idx is None) else tCrB[None, None, None, B_idx]
    sA_cur = None
    if const_expr(sA is not None):
        sA_cur = sA if const_expr(A_idx is None) else sA[None, None, None, A_idx]
    sB_cur = sB if const_expr(B_idx is None) else sB[None, None, None, B_idx]
    mma_atom = cute.make_mma_atom(tiled_mma.op)
    acc_tmem_addr = acc.iterator.toint()
    gemm_ptx_partial(
        mma_atom.op,
        acc_tmem_addr,
        rA,
        rB,
        sA_cur,
        sB_cur,
        zero_init=zero_init,
        cta_group=cta_group,
        **kwargs,
    )


def i64_to_i32x2(i: int) -> Tuple[int, int]:
    return i & 0xFFFF_FFFF, (i >> 32) & 0xFFFF_FFFF


@cute.jit
def gemm_ptx_partial(
    op: cute.nvgpu.tcgen05.mma.MmaOp,
    acc_tmem_addr: Int32,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    sA: Optional[cute.Tensor],
    sB: cute.Tensor,
    mbar_ptr: Optional[cutlass.Pointer] = None,
    mbar_phase: Optional[Int32] = None,
    split_arrive: Optional[int] = None,
    zero_init: bool | Boolean = False,
    tA_addr: Optional[Int32] = None,
    cta_group: int = 1,
    tA_offsets: Optional[tuple] = None,
) -> None:
    is_ts = op.a_src == cute.nvgpu.tcgen05.OperandSource.TMEM
    if const_expr(not is_ts):
        assert sA is not None, "sA must be provided when a_src is not TMEM"
    sA_layout = sA.layout if sA is not None else tCrA.layout
    sB_layout = sB.layout
    idesc: int = const_expr(sm100_desc.mma_op_to_idesc(op))
    kind = _tcgen05_mma_kind(op)
    if const_expr(not is_ts):
        sA_swizzle = sA.iterator.type.swizzle_type
        smem_desc_base_a: int = const_expr(
            sm100_desc.make_smem_desc_base(
                cute.recast_layout(128, op.a_dtype.width, sA_layout[0]),
                sA_swizzle,
                sm100_desc.Major.K
                if const_expr(op.a_major_mode == cute.nvgpu.tcgen05.mma.OperandMajorMode.K)
                else sm100_desc.Major.MN,
            )
        )
        smem_desc_base_a_lo, smem_desc_a_hi = i64_to_i32x2(smem_desc_base_a)
        smem_desc_base_a_lo = const_expr(smem_desc_base_a_lo)
        smem_desc_a_hi = const_expr(smem_desc_a_hi)
    else:
        smem_desc_base_a = None
        smem_desc_base_a_lo, smem_desc_a_hi = None, None
    sB_swizzle = sB.iterator.type.swizzle_type
    smem_desc_base_b: int = const_expr(
        sm100_desc.make_smem_desc_base(
            cute.recast_layout(128, op.b_dtype.width, sB_layout[0]),
            sB_swizzle,
            sm100_desc.Major.K
            if const_expr(op.b_major_mode == cute.nvgpu.tcgen05.mma.OperandMajorMode.K)
            else sm100_desc.Major.MN,
        )
    )
    smem_desc_base_b_lo, smem_desc_b_hi = i64_to_i32x2(smem_desc_base_b)
    smem_desc_base_b_lo = const_expr(smem_desc_base_b_lo)
    smem_desc_b_hi = const_expr(smem_desc_b_hi)

    tCrA_layout = (
        tCrA.layout
        if const_expr(not is_ts)
        else cute.recast_layout(32, tCrA.element_type.width, tCrA.layout)
    )
    offset_a = [cute.crd2idx((0, 0, k), tCrA_layout) for k in range(cute.size(tCrA.shape[2]))]
    if const_expr(tA_offsets is not None):
        assert is_ts and len(tA_offsets) == len(offset_a)
        offset_a = list(tA_offsets)
    offset_a_diff = [offset_a[k] - offset_a[k - 1] for k in range(1, cute.size(tCrA.shape[2]))]
    offset_b = [cute.crd2idx((0, 0, k), tCrB.layout) for k in range(cute.size(tCrB.shape[2]))]
    offset_b_diff = [offset_b[k] - offset_b[k - 1] for k in range(1, cute.size(tCrB.shape[2]))]

    if const_expr(not is_ts):
        smem_desc_start_a_lo = Int32(
            smem_desc_base_a_lo | sm100_desc.make_smem_desc_start_addr(sA[None, None, 0].iterator)
        )
    else:
        smem_desc_start_a_lo = None
    smem_desc_start_b_lo = Int32(
        smem_desc_base_b_lo | sm100_desc.make_smem_desc_start_addr(sB[None, None, 0].iterator)
    )
    pred_str = "p" if isinstance(zero_init, Boolean) else "0" if zero_init else "1"
    if const_expr(not is_ts):
        assert mbar_ptr is None, "mbar_ptr must be None when a_src is not TMEM"
        llvm.inline_asm(
            None,
            [
                Int32(cute.arch.make_warp_uniform(smem_desc_start_a_lo)).ir_value(),
                Int32(cute.arch.make_warp_uniform(smem_desc_start_b_lo)).ir_value(),
                Int32(not zero_init).ir_value(),
                Int32(cute.arch.make_warp_uniform(acc_tmem_addr)).ir_value(),
            ],
            "{\n\t"
            ".reg .pred leader_thread;\n\t"
            ".reg .pred p;\n\t"
            ".reg .b32 idesc;\n\t"
            ".reg .b32 tmem_acc;\n\t"
            ".reg .b32 smem_desc_a_lo_start, smem_desc_b_lo_start;\n\t"
            ".reg .b32 smem_desc_a_lo, smem_desc_b_lo;\n\t"
            ".reg .b32 smem_desc_a_hi, smem_desc_b_hi;\n\t"
            ".reg .b64 smem_desc_a, smem_desc_b;\n\t"
            "elect.sync _|leader_thread, -1;\n\t"
            f"mov.b32 idesc, {hex(idesc)};\n\t"
            f"mov.b32 tmem_acc, $3;\n\t"
            "mov.b32 smem_desc_a_lo_start, $0;\n\t"
            "mov.b32 smem_desc_b_lo_start, $1;\n\t"
            f"mov.b32 smem_desc_a_hi, {hex(smem_desc_a_hi)};\n\t"
            f"mov.b32 smem_desc_b_hi, {hex(smem_desc_b_hi)};\n\t"
            f"mov.b64 smem_desc_a, {{smem_desc_a_lo_start, smem_desc_a_hi}};\n\t"
            f"mov.b64 smem_desc_b, {{smem_desc_b_lo_start, smem_desc_b_hi}};\n\t"
            "setp.ne.b32 p, $2, 0;\n\t"
            f"@leader_thread tcgen05.mma.cta_group::{cta_group}.kind::{kind} [tmem_acc], smem_desc_a, smem_desc_b, idesc, {pred_str};\n\t"
            + "".join(
                (
                    f"add.u32 smem_desc_a_lo, smem_desc_a_lo_start, {hex(offset_a[k])};\n\t"
                    f"add.u32 smem_desc_b_lo, smem_desc_b_lo_start, {hex(offset_b[k])};\n\t"
                    f"mov.b64 smem_desc_a, {{smem_desc_a_lo, smem_desc_a_hi}};\n\t"
                    f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
                    f"@leader_thread tcgen05.mma.cta_group::{cta_group}.kind::{kind} [tmem_acc], smem_desc_a, smem_desc_b, idesc, 1;\n\t"
                )
                for k in range(1, cute.size(tCrA.shape[2]))
            )
            + "}\n",
            "r,r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    else:
        tA_addr = tCrA[None, None, 0].iterator.toint() if tA_addr is None else tA_addr
        input_args = [
            Int32(cute.arch.make_warp_uniform(tA_addr)).ir_value(),
            Int32(cute.arch.make_warp_uniform(smem_desc_start_b_lo)).ir_value(),
            Int32(not zero_init).ir_value(),
            Int32(cute.arch.make_warp_uniform(acc_tmem_addr)).ir_value(),
        ]
        if const_expr(mbar_ptr is not None):
            assert mbar_phase is not None, "mbar_phase must be provided when mbar_ptr is not None"
            assert split_arrive is not None, (
                "split_arrive must be provided when mbar_ptr is not None"
            )
            split_arrive_idx = split_arrive // op.shape_mnk[2]
            input_args.append(mbar_ptr.toint().ir_value())
            input_args.append(Int32(mbar_phase).ir_value())
            mbar_wait_str = (
                ".reg .pred P1; \n\t"
                "LAB_WAIT: \n\t"
                "mbarrier.try_wait.parity.shared::cta.b64 P1, [$4], $5, 10000000; \n\t"
                "@P1 bra DONE; \n\t"
                "bra     LAB_WAIT; \n\t"
                "DONE: \n\t"
            )
        else:
            mbar_wait_str = ""
        llvm.inline_asm(
            None,
            input_args,
            "{\n\t"
            ".reg .pred leader_thread;\n\t"
            ".reg .pred p;\n\t"
            ".reg .b32 idesc;\n\t"
            ".reg .b32 tmem_acc;\n\t"
            ".reg .b32 tmem_a;\n\t"
            ".reg .b32 smem_desc_b_lo_start;\n\t"
            ".reg .b32 smem_desc_b_lo;\n\t"
            ".reg .b32 smem_desc_b_hi;\n\t"
            ".reg .b64 smem_desc_b;\n\t"
            "elect.sync _|leader_thread, -1;\n\t"
            f"mov.b32 idesc, {hex(idesc)};\n\t"
            f"mov.b32 tmem_acc, $3;\n\t"
            f"mov.b32 tmem_a, $0;\n\t"
            f"mov.b32 smem_desc_b_lo_start, $1;\n\t"
            f"mov.b32 smem_desc_b_hi, {hex(smem_desc_b_hi)};\n\t"
            f"mov.b64 smem_desc_b, {{smem_desc_b_lo_start, smem_desc_b_hi}};\n\t"
            "setp.ne.b32 p, $2, 0;\n\t"
            f"@leader_thread tcgen05.mma.cta_group::{cta_group}.kind::{kind} [tmem_acc], [tmem_a], smem_desc_b, idesc, {pred_str};\n\t"
            + "".join(
                (
                    f"add.u32 smem_desc_b_lo, smem_desc_b_lo_start, {hex(offset_b[k])};\n\t"
                    f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
                    f"@leader_thread tcgen05.mma.cta_group::{cta_group}.kind::{kind} [tmem_acc], [tmem_a + {hex(offset_a[k])}], smem_desc_b, idesc, 1;\n\t"
                )
                for k in range(
                    1,
                    cute.size(tCrA.shape[2]) if const_expr(mbar_ptr is None) else split_arrive_idx,
                )
            )
            + mbar_wait_str
            + (
                "".join(
                    (
                        f"add.u32 smem_desc_b_lo, smem_desc_b_lo, {hex(offset_b_diff[k - 1])};\n\t"
                        f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
                        f"@leader_thread tcgen05.mma.cta_group::{cta_group}.kind::{kind} [tmem_acc], [tmem_a + {hex(offset_a[k])}], smem_desc_b, idesc, 1;\n\t"
                    )
                    for k in range(split_arrive_idx, cute.size(tCrA.shape[2]))
                )
                if const_expr(mbar_ptr is not None)
                else ""
            )
            + "}\n",
            "r,r,r,r" if const_expr(mbar_ptr is None) else "r,r,r,r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )


@dsl_user_op
def make_tmem_copy(
    tmem_copy_atom: cute.CopyAtom, num_wg: int = 1, *, loc=None, ip=None
) -> cute.CopyAtom:
    num_dp, num_bits, num_rep, _ = sm100_utils.get_tmem_copy_properties(tmem_copy_atom)
    assert num_dp == 32
    assert num_bits == 32
    tiler_mn = (cute.make_layout((128 * num_rep * num_wg // 32, 32), stride=(32, 1)),)
    layout_tv = cute.make_layout(
        ((32, 4, num_wg), (num_rep, 32)), stride=((0, 1, 4 * num_rep), (4, 4 * num_rep * num_wg))
    )
    return cute.make_tiled_copy(tmem_copy_atom, layout_tv, tiler_mn)


def tiled_copy_1d(
    dtype: Type[cutlass.Numeric], num_threads: int, num_copy_elems: int = 1, is_async: bool = False
) -> cute.TiledCopy:
    num_copy_bits = num_copy_elems * dtype.width
    copy_op = cpasync.CopyG2SOp() if is_async else cute.nvgpu.CopyUniversalOp()
    copy_atom = cute.make_copy_atom(copy_op, dtype, num_bits_per_copy=num_copy_bits)
    thr_layout = cute.make_layout(num_threads)
    val_layout = cute.make_layout(num_copy_elems)
    return cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)


@dsl_user_op
def set_block_rank(
    smem_ptr: cute.Pointer, peer_cta_rank_in_cluster: Int32, *, loc=None, ip=None
) -> Int32:
    smem_ptr_i32 = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [smem_ptr_i32, peer_cta_rank_in_cluster.ir_value()],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def cpasync_bulk_s2cluster(
    smem_src_ptr: cute.Pointer,
    smem_dst_ptr: cute.Pointer,
    mbar_ptr: cute.Pointer,
    size: int | Int32,
    peer_cta_rank_in_cluster: Int32,
    *,
    loc=None,
    ip=None,
):
    smem_src_ptr_i32 = smem_src_ptr.toint(loc=loc, ip=ip).ir_value()
    smem_dst_ptr_i32 = set_block_rank(
        smem_dst_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
    ).ir_value()
    mbar_ptr_i32 = set_block_rank(mbar_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [
            smem_dst_ptr_i32,
            smem_src_ptr_i32,
            mbar_ptr_i32,
            Int32(size).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [$0], [$1], $3, [$2];",
        "r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def cpasync_reduce_bulk_add_f32(
    smem_ptr: cute.Pointer,
    gmem_ptr: cute.Pointer,
    store_bytes: int | Int32,
    *,
    loc=None,
    ip=None,
):
    smem_ptr_i32 = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [gmem_ptr.llvm_ptr, smem_ptr_i32, Int32(store_bytes).ir_value()],
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [$0], [$1], $2;",
        "l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


def tma_get_copy_fn(
    atom: cute.CopyAtom,
    cta_coord: cute.Coord,
    cta_layout: cute.Layout,
    src_tensor: cute.Tensor,
    dst_tensor: cute.Tensor,
    filter_zeros: bool = False,
    single_stage: bool = False,
    **kwargs,
) -> Callable:
    src_is_smem = const_expr(
        isinstance(src_tensor.iterator, cute.Pointer)
        and src_tensor.memspace == cute.AddressSpace.smem
    )
    smem_tensor, gmem_tensor = (src_tensor, dst_tensor) if src_is_smem else (dst_tensor, src_tensor)
    group_rank_smem = const_expr(cute.rank(smem_tensor) - (1 if not single_stage else 0))
    group_rank_gmem = const_expr(cute.rank(gmem_tensor) - (1 if not single_stage else 0))
    s, g = cpasync.tma_partition(
        atom,
        cta_coord,
        cta_layout,
        cute.group_modes(smem_tensor, 0, group_rank_smem),
        cute.group_modes(gmem_tensor, 0, group_rank_gmem),
    )
    if const_expr(filter_zeros):
        s = cute.filter_zeros(s)
        g = cute.filter_zeros(g)
    src, dst = (s, g) if src_is_smem else (g, s)

    def copy_tma(src_idx, dst_idx, **new_kwargs):
        cute.copy(atom, src[None, src_idx], dst[None, dst_idx], **new_kwargs, **kwargs)

    def copy_tma_single_stage(**new_kwargs):
        cute.copy(atom, src, dst, **new_kwargs, **kwargs)

    return (copy_tma if const_expr(not single_stage) else copy_tma_single_stage), s, g


def tma_producer_copy_fn(copy: Callable, pipeline: cutlass.pipeline.PipelineAsync):
    def copy_fn(src_idx, producer_state: cutlass.pipeline.PipelineState, **new_kwargs):
        copy(
            src_idx=src_idx,
            dst_idx=producer_state.index,
            tma_bar_ptr=pipeline.producer_get_barrier(producer_state),
            **new_kwargs,
        )

    return copy_fn


def _override_create(parent_cls, child_cls):

    @staticmethod
    def create(*args, **kwargs):
        obj = parent_cls.create(*args, **kwargs)
        object.__setattr__(obj, "__class__", child_cls)
        return obj

    return create


class PipelineStateSimple:

    def __init__(self, stages: int, phase_index: Int32):
        self._stages = stages
        self._phase_index = phase_index

    def clone(self) -> "PipelineStateSimple":
        return PipelineStateSimple(self.stages, self._phase_index)

    @property
    def stages(self) -> int:
        return self._stages

    @property
    def index(self) -> Int32:
        if const_expr(self._stages == 1):
            return Int32(0)
        else:
            return self._phase_index % self._stages

    @property
    def phase(self) -> Int32:
        if const_expr(self._stages == 1):
            return self._phase_index
        else:
            return self._phase_index // self._stages

    def advance(self):
        if const_expr(self._stages == 1):
            self._phase_index ^= 1
        else:
            self._phase_index += 1

    def __extract_mlir_values__(self):
        phase_index = self._phase_index
        return [phase_index.ir_value()]

    def __new_from_mlir_values__(self, values):
        return PipelineStateSimple(self.stages, Int32(values[0]))


def make_pipeline_state(type: PipelineUserType, stages: int):
    if type is PipelineUserType.Producer:
        return PipelineStateSimple(stages, Int32(stages))
    elif type is PipelineUserType.Consumer:
        return PipelineStateSimple(stages, Int32(0))
    else:
        assert False, "Error: invalid PipelineUserType specified for make_pipeline_state."


@dataclass(frozen=True)
class PipelineTmaUmma(PipelineTmaUmmaOg):

    @dsl_user_op
    def producer_acquire(
        self,
        state: PipelineState,
        try_acquire_token: Optional[Boolean] = None,
        extra_tx_count: int = 0,
        *,
        loc=None,
        ip=None,
    ):
        if_generate(
            try_acquire_token is None or try_acquire_token == 0,
            lambda: self.sync_object_empty.wait(state.index, state.phase, loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
        if const_expr(extra_tx_count == 0):
            if_generate(
                self.is_leader_cta,
                lambda: self.sync_object_full.arrive(
                    state.index, self.producer_mask, loc=loc, ip=ip
                ),
                loc=loc,
                ip=ip,
            )
        else:
            tx_count = self.sync_object_full.tx_count + extra_tx_count
            if_generate(
                self.is_leader_cta,
                lambda: self.sync_object_full.arrive_and_expect_tx(
                    state.index, tx_count, loc=loc, ip=ip
                ),
                loc=loc,
                ip=ip,
            )


PipelineTmaUmma.create = _override_create(PipelineTmaUmmaOg, PipelineTmaUmma)


class NamedBarrierBwdSm100(enum.IntEnum):
    EpilogueWG1 = enum.auto()
    EpilogueWG2 = enum.auto()
    Compute = enum.auto()
    dQaccReduce = enum.auto()
    TmemPtr = enum.auto()


def _select(a, mode):
    return cute.make_tensor(a.iterator, cute.select(a.layout, mode))


def _transpose_view(a):
    shape = (a.shape[1], a.shape[0], *a.shape[2:])
    order = (1, 0, *range(2, cute.rank(a)))
    return cute.composition(a, cute.make_ordered_layout(shape, order=order))


layout_utils = SimpleNamespace(select=_select, transpose_view=_transpose_view)
copy_utils = SimpleNamespace(
    make_tmem_copy=make_tmem_copy,
    tiled_copy_1d=tiled_copy_1d,
    cpasync_bulk_s2cluster=cpasync_bulk_s2cluster,
    cpasync_reduce_bulk_add_f32=cpasync_reduce_bulk_add_f32,
    tma_get_copy_fn=tma_get_copy_fn,
    tma_producer_copy_fn=tma_producer_copy_fn,
)
pipeline = SimpleNamespace(
    PipelineTmaUmma=PipelineTmaUmma,
    make_pipeline_state=make_pipeline_state,
    PipelineUserType=PipelineUserType,
)


def assume_tensor_aligned(t):
    import cutlass.cute as cute
    divby = 128 // t.element_type.width
    strides = tuple(s if isinstance(s, int) else cute.assume(s, divby=divby) for s in t.stride[:-1])
    return cute.make_tensor(t.iterator, cute.make_layout(t.shape, stride=(*strides, t.stride[-1])))


@dsl_user_op
def fmax3(a, b, c=None, *, loc=None, ip=None) -> Float32:
    return Float32(
        nvvm.fmax(
            Float32(a).ir_value(loc=loc, ip=ip),
            Float32(b).ir_value(loc=loc, ip=ip),
            c=Float32(c).ir_value(loc=loc, ip=ip) if c is not None else None,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def atomic_max_nonneg_f32(gmem_ptr, val, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [gmem_ptr.llvm_ptr, Float32(val).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .b32 vbits;\n\tmov.b32 vbits, $1;\n\tred.global.max.u32 [$0], vbits;\n\t}",
        "l,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


LOG2_E = math.log2(math.e)


class FP8FlashAttnBwdSm100:
    arch = 100

    def __init__(
        self,
        head_dim: int = 128,
        qhead_per_kvhead: int = 4,
        is_causal: bool = True,
        use_2cta: bool = True,
        Q_stage: int = 2,
        dO_stage: int = 2,
    ):
        self.pds_kblock_cols = (0, 32, 8, 40)
        assert head_dim == 128
        self.tile_hdim = head_dim
        self.tile_hdimv = head_dim
        self.tile_m = 128
        self.tile_n = 128
        tile_m, tile_n = self.tile_m, self.tile_n
        self.use_2cta_instrs = use_2cta
        self.cta_group_size = 2 if use_2cta else 1
        cg = self.cta_group_size
        self.cta_tiler = (tile_n, tile_m, self.tile_hdim)
        self.mma_tiler_kq = (cg * tile_n, tile_m, self.tile_hdim)
        self.mma_tiler_vdo = (cg * tile_n, tile_m, self.tile_hdimv)
        self.mma_tiler_pdo = (cg * tile_n, self.tile_hdimv, tile_m)
        self.mma_tiler_dsq = (cg * tile_n, self.tile_hdim, tile_m)
        self.mma_tiler_dsk = (tile_m, self.tile_hdim, tile_n * cg)
        self.acc_dtype = Float32
        self.cluster_shape_mn = (cg, 1)
        self.is_causal = is_causal
        self.qhead_per_kvhead = qhead_per_kvhead
        self.Q_stage = Q_stage
        self.dO_stage = dO_stage

        self.reduce_warp_ids = (0, 1, 2, 3)
        self.compute_warp_ids = (4, 5, 6, 7, 8, 9, 10, 11)
        self.mma_warp_id = 12
        self.load_warp_id = 13
        self.relay_warp_id = 14
        self.empty_warp_id = 15
        self.threads_per_cta = cute.arch.WARP_SIZE * 16

        self.compute_sync_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierBwdSm100.Compute),
            num_threads=len(self.compute_warp_ids) * cute.arch.WARP_SIZE,
        )
        self.reduce_sync_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierBwdSm100.dQaccReduce),
            num_threads=len(self.reduce_warp_ids) * cute.arch.WARP_SIZE,
        )
        self.tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols("sm_100")
        self.tmem_S_offset = 0
        self.tmem_P_offset = 0
        self.tmem_dV_offset = self.tmem_S_offset + self.tile_n
        self.tmem_dP_offset = self.tmem_dV_offset + self.tile_hdimv
        self.tmem_dQ_offset = (
            self.tmem_S_offset + self.tile_hdim // 2 if use_2cta else self.tmem_dP_offset
        )
        self.tmem_dK_offset = self.tmem_dP_offset + self.tile_m
        self.tmem_dS_offset = self.tmem_dP_offset

        if use_2cta:
            self.num_regs_reduce = 136
            self.num_regs_compute = 136
            self.num_regs_load = 104
            self.num_regs_mma = 104
        else:
            self.num_regs_reduce = 136 if is_causal else 152
            self.num_regs_compute = 144 if is_causal else 136
            self.num_regs_load = 88
            self.num_regs_mma = 88
        self.num_regs_empty = 24
        assert self.num_regs_reduce + 2 * self.num_regs_compute + self.num_regs_load <= 512
        self.buffer_align_bytes = 1024

    def _setup_attributes(self):
        self.single_stage = 1
        self.sdKVaccum_stage = 2
        if self.use_2cta_instrs:
            self.dQ_reduce_ncol = 8
            self.sdQaccum_stage = 4
            self.dQ_reduce_ncol_t2r = 32
        else:
            self.dQ_reduce_ncol = 32
            self.sdQaccum_stage = (
                64 // self.dQ_reduce_ncol
            )
            self.dQ_reduce_ncol_t2r = self.dQ_reduce_ncol
        assert (self.tile_hdim // self.cta_group_size) % self.dQ_reduce_ncol == 0
        self.dQaccum_reduce_stage = self.tile_hdim // self.dQ_reduce_ncol
        self.dQaccum_reduce_stage_t2r = self.tile_hdim // self.dQ_reduce_ncol_t2r
        self.cta_group = tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE

    def _get_tiled_mma(self):
        tiled_mma_S = sm100_utils_basic.make_trivial_tiled_mma(
            self.k_dtype, self.q_dtype,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K,
            self.acc_dtype, self.cta_group, self.mma_tiler_kq[:2],
        )
        tiled_mma_dP = sm100_utils_basic.make_trivial_tiled_mma(
            self.v_dtype, self.do_dtype,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K,
            self.acc_dtype, self.cta_group, self.mma_tiler_vdo[:2],
        )
        tiled_mma_dV = sm100_utils_basic.make_trivial_tiled_mma(
            self.p_dtype, self.do_dtype,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.MN,
            self.acc_dtype, self.cta_group, self.mma_tiler_pdo[:2],
            a_source=tcgen05.OperandSource.TMEM,
        )
        tiled_mma_dK = sm100_utils_basic.make_trivial_tiled_mma(
            self.ds_dtype, self.q_dtype,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.MN,
            self.acc_dtype, self.cta_group, self.mma_tiler_dsq[:2],
            a_source=tcgen05.OperandSource.TMEM,
        )
        tiled_mma_dQ = sm100_utils_basic.make_trivial_tiled_mma(
            self.ds_dtype, self.k_dtype,
            tcgen05.OperandMajorMode.MN, tcgen05.OperandMajorMode.MN,
            self.acc_dtype, self.cta_group, self.mma_tiler_dsk[:2],
        )
        return tiled_mma_S, tiled_mma_dP, tiled_mma_dK, tiled_mma_dV, tiled_mma_dQ

    def _setup_smem_layout(self):
        sK_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_S, self.mma_tiler_kq, self.k_dtype, 1
        )
        self.sK_layout = cute.slice_(sK_layout, (None, None, None, 0))
        self.sQ_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_S, self.mma_tiler_kq, self.q_dtype, self.Q_stage
        )
        sV_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dP, self.mma_tiler_vdo, self.v_dtype, 1
        )
        self.sV_layout = cute.slice_(sV_layout, (None, None, None, 0))
        self.sdOt_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_dP, self.mma_tiler_vdo, self.do_dtype, self.dO_stage
        )
        tP_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dV, self.mma_tiler_pdo, self.p_dtype, 1
        )
        self.tP_layout = cute.slice_(tP_layout, (None, None, None, 0))
        self.sdO_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_dV, self.mma_tiler_pdo, self.do_dtype, self.dO_stage
        )
        sdSt_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dK, self.mma_tiler_dsq, self.ds_dtype, 1
        )
        self.sdSt_layout = cute.slice_(sdSt_layout, (None, None, None, 0))
        tdS_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dK, self.mma_tiler_dsq, self.ds_dtype, 1
        )
        self.tdS_layout = cute.slice_(tdS_layout, (None, None, None, 0))
        self.sQt_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_dK, self.mma_tiler_dsq, self.q_dtype, self.Q_stage
        )
        sdS_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dQ, self.mma_tiler_dsk, self.ds_dtype, 1
        )
        self.sdS_layout = cute.slice_(sdS_layout, (None, None, None, 0))
        sKt_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_dQ, self.mma_tiler_dsk, self.k_dtype, 1
        )
        self.sKt_layout = cute.slice_(sKt_layout, (None, None, None, 0))
        self.sdS_xchg_layout = cute.make_layout(shape=(self.tile_n, self.tile_m // 2))

        self.sdQaccum_layout = cute.make_layout(
            (self.tile_m * self.dQ_reduce_ncol, self.sdQaccum_stage)
        )
        self.sLSE_layout = cute.make_layout(
            shape=(self.tile_m, self.Q_stage), stride=(1, cute.round_up(self.tile_m, 64))
        )
        self.sdPsum_layout = cute.make_layout(
            shape=(self.tile_m, self.dO_stage), stride=(1, cute.round_up(self.tile_m, 64))
        )
        self.sdK_epi_tile = (
            self.tile_n,
            math.gcd(128 // (self.dk_dtype.width // 8), self.tile_hdim // 2),
        )
        self.sdV_epi_tile = (
            self.tile_n,
            math.gcd(128 // (self.dk_dtype.width // 8), self.tile_hdimv // 2),
        )
        self.sdK_layout = sm100_utils_basic.make_smem_layout_epi(
            self.dk_dtype, LayoutEnum.ROW_MAJOR, self.sdK_epi_tile, 2
        )
        self.sdV_layout = sm100_utils_basic.make_smem_layout_epi(
            self.dv_dtype, LayoutEnum.ROW_MAJOR, self.sdV_epi_tile, 2
        )

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdQaccum: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        mScales: cute.Tensor,
        mAmax: cute.Tensor,
        stream: cuda.CUstream = None,
    ):
        self.q_dtype = mQ.element_type
        self.k_dtype = mK.element_type
        self.v_dtype = mV.element_type
        self.do_dtype = mdO.element_type
        self.lse_dtype = mLSE.element_type
        self.dpsum_dtype = mdPsum.element_type
        self.dqaccum_dtype = mdQaccum.element_type
        self.dk_dtype = mdK.element_type
        self.dv_dtype = mdV.element_type
        self.p_dtype = cutlass.Float8E4M3FN
        self.ds_dtype = cutlass.Float8E5M2

        mdQaccum, mdK, mdV, mLSE, mdPsum = [
            assume_tensor_aligned(t) for t in (mdQaccum, mdK, mdV, mLSE, mdPsum)
        ]
        mQ, mdO = [layout_utils.select(t, mode=[1, 3, 2, 0]) for t in (mQ, mdO)]
        mK, mV = [layout_utils.select(t, mode=[1, 3, 2, 0]) for t in (mK, mV)]
        mLSE, mdPsum, mdQaccum = [
            layout_utils.select(t, mode=[2, 1, 0]) for t in (mLSE, mdPsum, mdQaccum)
        ]
        mdK, mdV = [layout_utils.select(t, mode=[1, 3, 2, 0]) for t in (mdK, mdV)]
        mdO = layout_utils.select(mdO, mode=[1, 0, 2, 3])

        self._setup_attributes()
        (
            self.tiled_mma_S,
            self.tiled_mma_dP,
            self.tiled_mma_dK,
            self.tiled_mma_dV,
            self.tiled_mma_dQ,
        ) = self._get_tiled_mma()
        self._setup_smem_layout()

        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk), (self.tiled_mma_S.thr_id.shape,)
        )

        tma_copy_op_dKV = cpasync.CopyBulkTensorTileS2GOp()
        tma_atom_dK, mdK_tma_tensor = cpasync.make_tiled_tma_atom(
            tma_copy_op_dKV, mdK, cute.select(self.sdK_layout, mode=[0, 1]), self.sdK_epi_tile, 1
        )
        tma_atom_dV, mdV_tma_tensor = cpasync.make_tiled_tma_atom(
            tma_copy_op_dKV, mdV, cute.select(self.sdV_layout, mode=[0, 1]), self.sdV_epi_tile, 1
        )
        thr_layout_r2s_dKV = cute.make_ordered_layout((128, 1), order=(1, 0))
        val_layout_r2s_dKV = cute.make_ordered_layout(
            (1, 128 // self.dk_dtype.width), order=(1, 0)
        )
        copy_atom_r2s_dKV = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), self.dk_dtype, num_bits_per_copy=128
        )
        tiled_copy_r2s_dKV = cute.make_tiled_copy_tv(
            copy_atom_r2s_dKV, thr_layout_r2s_dKV, val_layout_r2s_dKV
        )

        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(self.cta_group)
        tma_atom_K, tma_tensor_K = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op, mK, cute.select(self.sK_layout, mode=[0, 1, 2]),
            self.mma_tiler_kq, self.tiled_mma_S, self.cluster_layout_vmnk.shape,
        )
        Q_tma_op = sm100_utils_basic.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mnk, self.tiled_mma_S.thr_id
        )
        tma_atom_Q, tma_tensor_Q = cute.nvgpu.make_tiled_tma_atom_B(
            Q_tma_op, mQ, cute.select(self.sQ_layout, mode=[0, 1, 2]),
            self.mma_tiler_kq, self.tiled_mma_S, self.cluster_layout_vmnk.shape,
        )
        tma_atom_V, tma_tensor_V = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op, mV, cute.select(self.sV_layout, mode=[0, 1, 2]),
            self.mma_tiler_vdo, self.tiled_mma_dP, self.cluster_layout_vmnk.shape,
        )
        dO_tma_op = sm100_utils_basic.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mnk, self.tiled_mma_dV.thr_id
        )
        tma_atom_dO, tma_tensor_dO = cute.nvgpu.make_tiled_tma_atom_B(
            dO_tma_op, mdO, cute.select(self.sdO_layout, mode=[0, 1, 2]),
            self.mma_tiler_pdo, self.tiled_mma_dV, self.cluster_layout_vmnk.shape,
        )
        tma_atom_dOt = tma_tensor_dOt = None
        tma_atom_Qt = tma_tensor_Qt = None
        tma_atom_Kt = tma_tensor_Kt = None
        if const_expr(self.use_2cta_instrs):
            tma_atom_dOt, tma_tensor_dOt = cute.nvgpu.make_tiled_tma_atom_B(
                dO_tma_op, layout_utils.select(mdO, mode=[1, 0, 2, 3]),
                cute.select(self.sdOt_layout, mode=[0, 1, 2]),
                self.mma_tiler_vdo, self.tiled_mma_dP, self.cluster_layout_vmnk.shape,
            )
            tma_atom_Qt, tma_tensor_Qt = cute.nvgpu.make_tiled_tma_atom_B(
                Q_tma_op, layout_utils.select(mQ, mode=[1, 0, 2, 3]),
                cute.select(self.sQt_layout, mode=[0, 1, 2]),
                self.mma_tiler_dsq, self.tiled_mma_dK, self.cluster_layout_vmnk.shape,
            )
            Kt_tma_op = sm100_utils_basic.cluster_shape_to_tma_atom_B(
                self.cluster_shape_mnk, self.tiled_mma_dQ.thr_id
            )
            tma_atom_Kt, tma_tensor_Kt = cute.nvgpu.make_tiled_tma_atom_B(
                Kt_tma_op, layout_utils.select(mK, mode=[1, 0, 2, 3]),
                cute.select(self.sKt_layout, mode=[0, 1, 2]),
                self.mma_tiler_dsk, self.tiled_mma_dQ, self.cluster_layout_vmnk.shape,
            )

        self.tma_copy_bytes = {
            name: self.cta_group_size
            * cute.size_in_bytes(mX.element_type, cute.select(layout, mode=[0, 1, 2]))
            for name, mX, layout in [
                ("Q", mQ, self.sQ_layout),
                ("K", mK, self.sK_layout),
                ("V", mV, self.sV_layout),
                ("dO", mdO, self.sdO_layout),
            ]
        }
        self.tma_copy_bytes["LSE"] = self.tile_m * Float32.width // 8
        self.tma_copy_bytes["dPsum"] = self.tile_m * Float32.width // 8
        self.tma_copy_bytes["dQ"] = self.tile_m * self.dQ_reduce_ncol * Float32.width // 8
        self.tma_copy_bytes["dS"] = cute.size_in_bytes(self.ds_dtype, self.sdS_layout)

        num_kv_heads = cute.size(mK.shape[2])
        num_batch = cute.size(mK.shape[3])
        num_n_blocks = cute.ceil_div(cute.size(mK.shape[0]), self.tile_n)
        grid_dim = (num_kv_heads * num_batch * num_n_blocks, 1, 1)

        bytes_Q = cute.size_in_bytes(self.q_dtype, self.sQ_layout)
        bytes_Qt = (
            cute.size_in_bytes(self.q_dtype, self.sQt_layout) if const_expr(self.use_2cta_instrs)
            else 0
        )
        bytes_dO = cute.size_in_bytes(self.do_dtype, self.sdO_layout)
        bytes_dOt = (
            cute.size_in_bytes(self.do_dtype, self.sdOt_layout) if const_expr(self.use_2cta_instrs)
            else 0
        )
        sQQt_bytes = max(bytes_Q + bytes_Qt, cute.size_in_bytes(self.dk_dtype, self.sdK_layout))
        sdOdOt_bytes = max(
            bytes_dO + bytes_dOt, cute.size_in_bytes(self.dv_dtype, self.sdV_layout)
        )
        self.sQt_byte_offset = bytes_Q
        self.sdOt_byte_offset = bytes_dO
        sKt_size = cute.cosize(self.sKt_layout) if const_expr(self.use_2cta_instrs) else 0
        sdS_xchg_size = cute.cosize(self.sdS_xchg_layout) if const_expr(self.use_2cta_instrs) else 0

        @cute.struct
        class SharedStorage:
            Q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.Q_stage]
            dO_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.dO_stage]
            LSE_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.Q_stage]
            dPsum_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.dO_stage]
            S_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.single_stage]
            dP_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.single_stage]
            dS_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.single_stage]
            dKV_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.sdKVaccum_stage]
            dQ_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            Qt_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.Q_stage]
            Kt_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.single_stage]
            dS_cluster_empty_mbar_ptr: cutlass.Int64
            dS_cluster_full_mbar_ptr: cutlass.Int64
            dS_cluster_leader_mbar_ptr: cutlass.Int64
            xchg_ready_mbar_ptr: cutlass.Int64
            tmem_holding_buf: Int32
            tmem_dealloc_mbar: cutlass.Int64
            sQQt: cute.struct.Align[
                cute.struct.MemRange[cute.Uint8, sQQt_bytes], self.buffer_align_bytes
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self.k_dtype, cute.cosize(self.sK_layout)],
                self.buffer_align_bytes,
            ]
            sV: cute.struct.Align[
                cute.struct.MemRange[self.v_dtype, cute.cosize(self.sV_layout)],
                self.buffer_align_bytes,
            ]
            sdOdOt: cute.struct.Align[
                cute.struct.MemRange[cute.Uint8, sdOdOt_bytes], self.buffer_align_bytes
            ]
            sKt: cute.struct.Align[
                cute.struct.MemRange[self.k_dtype, sKt_size], self.buffer_align_bytes
            ]
            sdS: cute.struct.Align[
                cute.struct.MemRange[self.ds_dtype, cute.cosize(self.sdSt_layout)],
                self.buffer_align_bytes,
            ]
            sdS_xchg: cute.struct.Align[
                cute.struct.MemRange[self.ds_dtype, sdS_xchg_size], self.buffer_align_bytes
            ]
            sLSE: cute.struct.Align[
                cute.struct.MemRange[self.lse_dtype, cute.cosize(self.sLSE_layout)], 128
            ]
            sdPsum: cute.struct.Align[
                cute.struct.MemRange[self.dpsum_dtype, cute.cosize(self.sdPsum_layout)], 128
            ]
            sdQaccum: cute.struct.Align[
                cute.struct.MemRange[self.dqaccum_dtype, cute.cosize(self.sdQaccum_layout)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        self.kernel(
            tma_tensor_Q,
            tma_tensor_Qt,
            tma_tensor_K,
            tma_tensor_Kt,
            tma_tensor_V,
            mLSE,
            mdPsum,
            tma_tensor_dO,
            tma_tensor_dOt,
            mdQaccum,
            mdV_tma_tensor,
            mdK_tma_tensor,
            mScales,
            mAmax,
            tma_atom_Q,
            tma_atom_Qt,
            tma_atom_K,
            tma_atom_Kt,
            tma_atom_V,
            tma_atom_dO,
            tma_atom_dOt,
            tma_atom_dV,
            tma_atom_dK,
            self.sQ_layout,
            self.sQt_layout,
            self.sK_layout,
            self.sKt_layout,
            self.sV_layout,
            self.sLSE_layout,
            self.sdPsum_layout,
            self.sdO_layout,
            self.sdOt_layout,
            self.sdSt_layout,
            self.sdS_layout,
            self.sdS_xchg_layout,
            self.sdQaccum_layout,
            self.sdK_layout,
            self.sdV_layout,
            self.tP_layout,
            self.tdS_layout,
            self.tiled_mma_S,
            self.tiled_mma_dP,
            self.tiled_mma_dV,
            self.tiled_mma_dK,
            self.tiled_mma_dQ,
            tiled_copy_r2s_dKV,
            num_kv_heads * num_batch,
            num_kv_heads,
        ).launch(
            grid=grid_dim,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk if const_expr(self.use_2cta_instrs) else None,
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.jit
    def get_tile(self, num_hb: Int32, num_kv_heads: Int32, seqlen: Int32):
        bidx, _, _ = cute.arch.block_idx()
        cg = self.cta_group_size
        rank = bidx % cg
        c = bidx // cg
        hb = c % num_hb
        n_pair = c // num_hb
        head_kv = hb % num_kv_heads
        batch_idx = hb // num_kv_heads
        n_block = n_pair * cg + rank
        m_block_max = seqlen // self.tile_m
        m_block_min = Int32(0)
        if const_expr(self.is_causal):
            m_block_min = n_pair * cg * self.tile_n // self.tile_m
        return n_block, head_kv, batch_idx, m_block_min, m_block_max

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mQt,
        mK: cute.Tensor,
        mKt,
        mV: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdO: cute.Tensor,
        mdOt,
        mdQaccum: cute.Tensor,
        mdV_tma_tensor: cute.Tensor,
        mdK_tma_tensor: cute.Tensor,
        mScales: cute.Tensor,
        mAmax: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_Qt,
        tma_atom_K: cute.CopyAtom,
        tma_atom_Kt,
        tma_atom_V: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        tma_atom_dOt,
        tma_atom_dV: cute.CopyAtom,
        tma_atom_dK: cute.CopyAtom,
        sQ_layout: cute.ComposedLayout,
        sQt_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sKt_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sLSE_layout: cute.Layout,
        sdPsum_layout: cute.Layout,
        sdO_layout: cute.ComposedLayout,
        sdOt_layout: cute.ComposedLayout,
        sdSt_layout: cute.ComposedLayout,
        sdS_layout: cute.ComposedLayout,
        sdS_xchg_layout: cute.Layout,
        sdQaccum_layout: cute.Layout,
        sdK_layout: cute.ComposedLayout,
        sdV_layout: cute.ComposedLayout,
        tP_layout: cute.ComposedLayout,
        tdS_layout: cute.ComposedLayout,
        tiled_mma_S: cute.TiledMma,
        tiled_mma_dP: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dQ: cute.TiledMma,
        tiled_copy_r2s_dKV: cute.TiledCopy,
        num_hb: Int32,
        num_kv_heads: Int32,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bidx, _, _ = cute.arch.block_idx()
        mma_tile_coord_v = bidx % self.cta_group_size
        is_leader_cta = mma_tile_coord_v == 0

        if warp_idx == self.load_warp_id:
            with cute.arch.elect_one():
                cpasync.prefetch_descriptor(tma_atom_Q)
                cpasync.prefetch_descriptor(tma_atom_K)
                cpasync.prefetch_descriptor(tma_atom_V)
                cpasync.prefetch_descriptor(tma_atom_dO)
                cpasync.prefetch_descriptor(tma_atom_dV)
                cpasync.prefetch_descriptor(tma_atom_dK)
                if const_expr(self.use_2cta_instrs):
                    cpasync.prefetch_descriptor(tma_atom_Qt)
                    cpasync.prefetch_descriptor(tma_atom_Kt)
                    cpasync.prefetch_descriptor(tma_atom_dOt)

        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk), (tiled_mma_S.thr_id.shape,)
        )

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        dS_cluster_full_mbar_ptr = None
        dS_cluster_empty_mbar_ptr = None
        dS_cluster_leader_mbar_ptr = None
        if const_expr(self.use_2cta_instrs):
            dS_cluster_full_mbar_ptr = storage.dS_cluster_full_mbar_ptr
            dS_cluster_empty_mbar_ptr = storage.dS_cluster_empty_mbar_ptr
            dS_cluster_leader_mbar_ptr = storage.dS_cluster_leader_mbar_ptr
            if warp_idx == 4:
                cute.arch.mbarrier_init(dS_cluster_full_mbar_ptr, 1)
                cute.arch.mbarrier_init(dS_cluster_empty_mbar_ptr, 1)
                cute.arch.mbarrier_init(dS_cluster_leader_mbar_ptr, 2)
                cute.arch.mbarrier_init(storage.xchg_ready_mbar_ptr, len(self.compute_warp_ids))

        tmem_alloc_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierBwdSm100.TmemPtr),
            num_threads=cute.arch.WARP_SIZE
            * len((self.mma_warp_id, *self.compute_warp_ids, *self.reduce_warp_ids)),
        )
        tmem = cutlass.utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.mma_warp_id,
            is_two_cta=self.use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )
        cg = self.cta_group_size
        pipeline_producer_group_MMA_AsyncThread = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, 1
        )
        pipeline_consumer_group_MMA_AsyncThread = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, len(self.compute_warp_ids) * cg
        )
        pipeline_S_P = cutlass.pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=pipeline_producer_group_MMA_AsyncThread,
            consumer_group=pipeline_consumer_group_MMA_AsyncThread,
            barrier_storage=storage.S_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        pipeline_dP = cutlass.pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=pipeline_producer_group_MMA_AsyncThread,
            consumer_group=pipeline_consumer_group_MMA_AsyncThread,
            barrier_storage=storage.dP_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        pipeline_dKV = cutlass.pipeline.PipelineUmmaAsync.create(
            num_stages=2,
            producer_group=pipeline_producer_group_MMA_AsyncThread,
            consumer_group=pipeline_consumer_group_MMA_AsyncThread,
            barrier_storage=storage.dKV_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        pipeline_consumer_group_MMA_AsyncThread_dQ = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, len(self.reduce_warp_ids) * cg
        )
        pipeline_dQ = cutlass.pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=pipeline_producer_group_MMA_AsyncThread,
            consumer_group=pipeline_consumer_group_MMA_AsyncThread_dQ,
            barrier_storage=storage.dQ_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        pipeline_PdS_producer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, len(self.compute_warp_ids) * cg
        )
        pipeline_PdS_consumer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, 1
        )
        pipeline_dS = cutlass.pipeline.PipelineAsyncUmma.create(
            num_stages=1,
            producer_group=pipeline_PdS_producer_group,
            consumer_group=pipeline_PdS_consumer_group,
            barrier_storage=storage.dS_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        pipeline_producer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, 1
        )
        pipeline_consumer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, 1
        )
        pipeline_consumer_group_compute = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, len(self.compute_warp_ids)
        )
        pipeline_LSE = cutlass.pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.LSE_mbar_ptr.data_ptr(),
            num_stages=self.Q_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group_compute,
            tx_count=self.tma_copy_bytes["LSE"],
            defer_sync=True,
        )
        pipeline_dPsum = cutlass.pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.dPsum_mbar_ptr.data_ptr(),
            num_stages=self.dO_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group_compute,
            tx_count=self.tma_copy_bytes["dPsum"],
            defer_sync=True,
        )
        pipeline_Q = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.Q_mbar_ptr.data_ptr(),
            num_stages=self.Q_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group,
            tx_count=self.tma_copy_bytes["Q"],
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        if const_expr(self.use_2cta_instrs):
            pipeline_Qt = pipeline.PipelineTmaUmma.create(
                barrier_storage=storage.Qt_mbar_ptr.data_ptr(),
                num_stages=self.Q_stage,
                producer_group=pipeline_producer_group,
                consumer_group=pipeline_consumer_group,
                tx_count=self.tma_copy_bytes["Q"],
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            )
            pipeline_Kt = pipeline.PipelineTmaUmma.create(
                barrier_storage=storage.Kt_mbar_ptr.data_ptr(),
                num_stages=self.single_stage,
                producer_group=pipeline_producer_group,
                consumer_group=pipeline_consumer_group,
                tx_count=self.tma_copy_bytes["K"],
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            )
        else:
            pipeline_Qt = pipeline_Kt = pipeline_Q
        pipeline_dO = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.dO_mbar_ptr.data_ptr(),
            num_stages=self.dO_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group,
            tx_count=self.tma_copy_bytes["dO"],
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=False,
        )

        sQ = storage.sQQt.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner, dtype=self.q_dtype)
        if const_expr(self.use_2cta_instrs):
            sQt_ptr = cute.recast_ptr(
                storage.sQQt.data_ptr() + self.sQt_byte_offset, sQt_layout.inner,
                dtype=self.q_dtype,
            )
            sQt = cute.make_tensor(sQt_ptr, sQt_layout.outer)
        else:
            sQt = cute.make_tensor(
                cute.recast_ptr(sQ.iterator, sQt_layout.inner, dtype=self.q_dtype),
                sQt_layout.outer,
            )
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        if const_expr(self.use_2cta_instrs):
            sKt = storage.sKt.get_tensor(sKt_layout.outer, swizzle=sKt_layout.inner)
        else:
            sKt = cute.make_tensor(cute.recast_ptr(sK.iterator, sKt_layout.inner), sKt_layout.outer)
        sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        sdSt = storage.sdS.get_tensor(sdSt_layout.outer, swizzle=sdSt_layout.inner)
        sdS = cute.make_tensor(cute.recast_ptr(sdSt.iterator, sdS_layout.inner), sdS_layout.outer)
        sdS_xchg = None
        if const_expr(self.use_2cta_instrs):
            sdS_xchg = storage.sdS_xchg.get_tensor(sdS_xchg_layout)
        sdO = storage.sdOdOt.get_tensor(
            sdO_layout.outer, swizzle=sdO_layout.inner, dtype=self.do_dtype
        )
        if const_expr(self.use_2cta_instrs):
            sdOt_ptr = cute.recast_ptr(
                storage.sdOdOt.data_ptr() + self.sdOt_byte_offset, sdOt_layout.inner,
                dtype=self.do_dtype,
            )
            sdOt = cute.make_tensor(sdOt_ptr, sdOt_layout.outer)
        else:
            sdOt = cute.make_tensor(
                cute.recast_ptr(sdO.iterator, sdOt_layout.inner, dtype=self.do_dtype),
                sdOt_layout.outer,
            )
        sLSE = storage.sLSE.get_tensor(sLSE_layout)
        sdPsum = storage.sdPsum.get_tensor(sdPsum_layout)
        sdV = storage.sdOdOt.get_tensor(
            sdV_layout.outer, swizzle=sdV_layout.inner, dtype=self.dv_dtype
        )
        sdK = storage.sQQt.get_tensor(
            sdK_layout.outer, swizzle=sdK_layout.inner, dtype=self.dk_dtype
        )
        sdQaccum = storage.sdQaccum.get_tensor(sdQaccum_layout)

        tmem_ptr = cute.make_ptr(Float32, 0, mem_space=cute.AddressSpace.tmem, assumed_align=16)
        thr_mma_S = tiled_mma_S.get_slice(mma_tile_coord_v)
        Sacc_shape = thr_mma_S.partition_shape_C(self.mma_tiler_kq[:2])
        tStS = thr_mma_S.make_fragment_C(Sacc_shape)
        tStS = cute.make_tensor(tmem_ptr + self.tmem_S_offset, tStS.layout)
        thr_mma_dP = tiled_mma_dP.get_slice(mma_tile_coord_v)
        dPacc_shape = thr_mma_dP.partition_shape_C(self.mma_tiler_vdo[:2])
        tdPtdP = thr_mma_dP.make_fragment_C(dPacc_shape)
        tdPtdP = cute.make_tensor(tmem_ptr + self.tmem_dP_offset, tdPtdP.layout)
        thr_mma_dV = tiled_mma_dV.get_slice(mma_tile_coord_v)
        dvacc_shape = thr_mma_dV.partition_shape_C(self.mma_tiler_pdo[:2])
        tdVtdV = thr_mma_dV.make_fragment_C(dvacc_shape)
        tdVtdV = cute.make_tensor(tmem_ptr + self.tmem_dV_offset, tdVtdV.layout)
        tP = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + self.tmem_P_offset, dtype=self.p_dtype), tP_layout.outer
        )
        thr_mma_dK = tiled_mma_dK.get_slice(mma_tile_coord_v)
        dkacc_shape = thr_mma_dK.partition_shape_C(self.mma_tiler_dsq[:2])
        tdKtdK = thr_mma_dK.make_fragment_C(dkacc_shape)
        tdKtdK = cute.make_tensor(tmem_ptr + self.tmem_dK_offset, tdKtdK.layout)
        tdS = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + self.tmem_dS_offset, dtype=self.ds_dtype), tdS_layout.outer
        )
        thr_mma_dQ = tiled_mma_dQ.get_slice(mma_tile_coord_v)
        dQacc_shape = thr_mma_dQ.partition_shape_C(self.mma_tiler_dsk[:2])
        tdQtdQ = thr_mma_dQ.make_fragment_C(dQacc_shape)
        tdQtdQ = cute.make_tensor(tmem_ptr + self.tmem_dQ_offset, tdQtdQ.layout)

        seqlen = mQ.shape[0]
        tile = self.get_tile(num_hb, num_kv_heads, seqlen)

        if warp_idx == self.empty_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_empty)

        if warp_idx == self.relay_warp_id:
            cute.arch.setmaxregister_decrease(
                self.num_regs_mma if const_expr(self.use_2cta_instrs) else self.num_regs_empty
            )
            if const_expr(self.use_2cta_instrs):
                self.relay(
                    dS_cluster_full_mbar_ptr, dS_cluster_leader_mbar_ptr,
                    storage.xchg_ready_mbar_ptr, sdS, sdS_xchg, tile,
                )

        if warp_idx == self.load_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_load)
            self.load(
                thr_mma_S, thr_mma_dP, thr_mma_dV, thr_mma_dK, thr_mma_dQ,
                mQ, mK, mKt, mV, mdO, mQt, mdOt, mLSE, mdPsum,
                sQ, sK, sKt, sV, sdO, sQt, sdOt, sLSE, sdPsum,
                tma_atom_Q, tma_atom_K, tma_atom_Kt, tma_atom_V, tma_atom_dO, tma_atom_Qt,
                tma_atom_dOt,
                pipeline_Q, pipeline_Qt, pipeline_Kt, pipeline_dO, pipeline_LSE, pipeline_dPsum,
                cluster_layout_vmnk,
                tile,
            )

        if warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_mma)
            tmem.allocate(self.tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)
            self.mma(
                tiled_mma_S, tiled_mma_dP, tiled_mma_dV, tiled_mma_dK, tiled_mma_dQ,
                sQ, sQt, sK, sKt, sV, sdO, sdOt, tP, sdSt, sdS, tdS,
                tStS, tdPtdP, tdVtdV, tdKtdK, tdQtdQ,
                dS_cluster_leader_mbar_ptr,
                pipeline_Q, pipeline_Qt, pipeline_Kt, pipeline_dO, pipeline_S_P, pipeline_dS,
                pipeline_dKV, pipeline_dP, pipeline_dQ,
                is_leader_cta,
                tile,
            )
            tmem.relinquish_alloc_permit()
            tmem_alloc_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)

        if warp_idx >= self.compute_warp_ids[0] and warp_idx <= self.compute_warp_ids[-1]:
            cute.arch.setmaxregister_increase(self.num_regs_compute)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)
            self.compute_loop(
                thr_mma_S, thr_mma_dP, thr_mma_dV, thr_mma_dK,
                tStS, tdPtdP, tdVtdV, tdKtdK,
                sLSE, sdPsum, sdS, sdS_xchg,
                pipeline_LSE, pipeline_dPsum, pipeline_S_P, pipeline_dS, pipeline_dKV,
                pipeline_dP,
                dS_cluster_full_mbar_ptr,
                mScales, mAmax,
                sdV, sdK, mdV_tma_tensor, mdK_tma_tensor, tma_atom_dV, tma_atom_dK,
                tiled_copy_r2s_dKV,
                tile,
                storage.xchg_ready_mbar_ptr,
            )
            tmem_alloc_barrier.arrive()

        if warp_idx >= self.reduce_warp_ids[0] and warp_idx <= self.reduce_warp_ids[-1]:
            cute.arch.setmaxregister_increase(self.num_regs_reduce)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)
            self.dQacc_reduce(mdQaccum, sdQaccum, thr_mma_dQ, tdQtdQ, pipeline_dQ, tile)
            tmem_alloc_barrier.arrive()
        return

    @cute.jit
    def relay(
        self, dS_cluster_full_mbar_ptr, dS_cluster_leader_mbar_ptr, xchg_ready_mbar_ptr,
        sdS, sdS_xchg, tile,
    ):
        n_block, head_kv, batch_idx, m_block_min, m_block_max = tile
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        peer = cta_rank_in_cluster ^ 1
        stage_copy_bytes = const_expr(self.tma_copy_bytes["dS"] // 2)
        stage_copy_elems = const_expr(stage_copy_bytes // (self.ds_dtype.width // 8))
        dS_cluster_phase = Int32(0)
        total = (m_block_max - m_block_min) * self.qhead_per_kvhead
        for _ in cutlass.range(total, unroll=1):
            cute.arch.mbarrier_wait(xchg_ready_mbar_ptr, phase=dS_cluster_phase)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    dS_cluster_full_mbar_ptr, stage_copy_bytes, peer_cta_rank_in_cluster=peer
                )
                copy_utils.cpasync_bulk_s2cluster(
                    sdS_xchg.iterator,
                    sdS.iterator + cta_rank_in_cluster * stage_copy_elems,
                    dS_cluster_full_mbar_ptr,
                    stage_copy_bytes,
                    peer_cta_rank_in_cluster=peer,
                )
            cute.arch.mbarrier_wait(dS_cluster_full_mbar_ptr, phase=dS_cluster_phase)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(dS_cluster_leader_mbar_ptr, Int32(0))
            dS_cluster_phase ^= 1

    @cute.jit
    def load(
        self,
        thr_mma_S, thr_mma_dP, thr_mma_dV, thr_mma_dK, thr_mma_dQ,
        mQ, mK, mKt, mV, mdO, mQt, mdOt, mLSE, mdPsum,
        sQ, sK, sKt, sV, sdO, sQt, sdOt, sLSE, sdPsum,
        tma_atom_Q, tma_atom_K, tma_atom_Kt, tma_atom_V, tma_atom_dO, tma_atom_Qt, tma_atom_dOt,
        pipeline_Q, pipeline_Qt, pipeline_Kt, pipeline_dO, pipeline_LSE, pipeline_dPsum,
        cluster_layout_vmnk,
        tile,
    ):
        n_block, head_kv, batch_idx, m_block_min, m_block_max = tile
        G = self.qhead_per_kvhead
        cg = self.cta_group_size
        n_block_cta_group = n_block // cg
        producer_state_Q_LSE = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.Q_stage
        )
        producer_state_Qt = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.Q_stage
        )
        producer_state_Kt = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.single_stage
        )
        producer_state_dO_dPsum = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.dO_stage
        )
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)

        mK_cur = mK[None, None, head_kv, batch_idx]
        mV_cur = mV[None, None, head_kv, batch_idx]
        gK = cute.local_tile(
            mK_cur, cute.select(self.mma_tiler_kq, mode=[0, 2]), (n_block_cta_group, 0)
        )
        tSgK = thr_mma_S.partition_A(gK)
        load_K, _, _ = copy_utils.tma_get_copy_fn(
            tma_atom_K, block_in_cluster_coord_vmnk[2], a_cta_layout, tSgK, sK, single_stage=True
        )
        gV = cute.local_tile(
            mV_cur, cute.select(self.mma_tiler_vdo, mode=[0, 2]), (n_block_cta_group, 0)
        )
        tdPgV = thr_mma_dP.partition_A(gV)
        load_V, _, _ = copy_utils.tma_get_copy_fn(
            tma_atom_V, 0, cute.make_layout(1), tdPgV, sV, single_stage=True
        )
        load_Kt = None
        if const_expr(self.use_2cta_instrs):
            mKt_cur = mKt[None, None, head_kv, batch_idx]
            gKt = cute.local_tile(
                mKt_cur, cute.select(self.mma_tiler_dsk, mode=[1, 2]), (0, n_block_cta_group)
            )
            tdQgK = thr_mma_dQ.partition_B(gKt)
            load_Kt, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_Kt, block_in_cluster_coord_vmnk[1], b_cta_layout, tdQgK, sKt,
                single_stage=True,
            )
        copy_atom_stats = cute.make_copy_atom(cpasync.CopyBulkG2SOp(), Float32)
        copy_stats = partial(cute.copy, copy_atom_stats)

        nm = m_block_max - m_block_min
        total = nm * G
        qh = Int32(0)
        m_block = m_block_min
        qh_prev = Int32(0)
        m_block_prev = m_block_min
        for it in cutlass.range(total, unroll=1):
            head_idx = head_kv * G + qh
            mQ_cur = mQ[None, None, head_idx, batch_idx]
            mdO_cur = mdO[None, None, head_idx, batch_idx]
            mLSE_cur = mLSE[None, head_idx, batch_idx]
            mdPsum_cur = mdPsum[None, head_idx, batch_idx]
            gQ = cute.local_tile(mQ_cur, cute.select(self.mma_tiler_kq, mode=[1, 2]), (None, 0))
            tSgQ = thr_mma_S.partition_B(gQ)
            gdO = cute.local_tile(mdO_cur, cute.select(self.mma_tiler_pdo, mode=[1, 2]), (0, None))
            tdVgdO = thr_mma_dV.partition_B(gdO)
            gLSE = cute.local_tile(mLSE_cur, (self.tile_m,), (None,))
            gdPsum = cute.local_tile(mdPsum_cur, (self.tile_m,), (None,))
            load_Q, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_Q, block_in_cluster_coord_vmnk[1], b_cta_layout, tSgQ, sQ
            )
            load_Q = copy_utils.tma_producer_copy_fn(load_Q, pipeline_Q)
            load_dO, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_dO, block_in_cluster_coord_vmnk[1], b_cta_layout, tdVgdO, sdO
            )
            load_dO = copy_utils.tma_producer_copy_fn(load_dO, pipeline_dO)
            load_dOt = None
            if const_expr(self.use_2cta_instrs):
                mdOt_cur = mdOt[None, None, head_idx, batch_idx]
                gdOt = cute.local_tile(
                    mdOt_cur, cute.select(self.mma_tiler_vdo, mode=[1, 2]), (None, 0)
                )
                tdPgdOt = thr_mma_dP.partition_B(gdOt)
                load_dOt, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_dOt, block_in_cluster_coord_vmnk[1], b_cta_layout, tdPgdOt, sdOt
                )
                load_dOt = copy_utils.tma_producer_copy_fn(load_dOt, pipeline_dO)
                if it > 0:
                    head_prev = head_kv * G + qh_prev
                    mQt_prev = mQt[None, None, head_prev, batch_idx]
                    gQt = cute.local_tile(
                        mQt_prev, cute.select(self.mma_tiler_dsq, mode=[1, 2]), (0, None)
                    )
                    tdKgQt = thr_mma_dK.partition_B(gQt)
                    load_Qt, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_Qt, block_in_cluster_coord_vmnk[1], b_cta_layout, tdKgQt, sQt
                    )
                    load_Qt = copy_utils.tma_producer_copy_fn(load_Qt, pipeline_Qt)
                    pipeline_Qt.producer_acquire(producer_state_Qt)
                    load_Qt(m_block_prev, producer_state=producer_state_Qt)
                    pipeline_Qt.producer_commit(producer_state_Qt)
                    producer_state_Qt.advance()

            if it == 0:
                pipeline_Q.producer_acquire(
                    producer_state_Q_LSE, extra_tx_count=self.tma_copy_bytes["K"]
                )
                load_K(tma_bar_ptr=pipeline_Q.producer_get_barrier(producer_state_Q_LSE))
            else:
                pipeline_Q.producer_acquire(producer_state_Q_LSE)
            load_Q(m_block, producer_state=producer_state_Q_LSE)
            pipeline_Q.producer_commit(producer_state_Q_LSE)
            pipeline_LSE.producer_acquire(producer_state_Q_LSE)
            with cute.arch.elect_one():
                copy_stats(
                    gLSE[None, m_block],
                    sLSE[None, producer_state_Q_LSE.index],
                    mbar_ptr=pipeline_LSE.producer_get_barrier(producer_state_Q_LSE),
                )
            producer_state_Q_LSE.advance()

            extra_dOt = self.tma_copy_bytes["dO"] if const_expr(self.use_2cta_instrs) else 0
            if it == 0:
                pipeline_dO.producer_acquire(
                    producer_state_dO_dPsum, extra_tx_count=self.tma_copy_bytes["V"] + extra_dOt
                )
                load_V(tma_bar_ptr=pipeline_dO.producer_get_barrier(producer_state_dO_dPsum))
            else:
                pipeline_dO.producer_acquire(producer_state_dO_dPsum, extra_tx_count=extra_dOt)
            load_dO(m_block, producer_state=producer_state_dO_dPsum)
            if const_expr(self.use_2cta_instrs):
                load_dOt(m_block, producer_state=producer_state_dO_dPsum)
            pipeline_dO.producer_commit(producer_state_dO_dPsum)
            pipeline_dPsum.producer_acquire(producer_state_dO_dPsum)
            with cute.arch.elect_one():
                copy_stats(
                    gdPsum[None, m_block],
                    sdPsum[None, producer_state_dO_dPsum.index],
                    mbar_ptr=pipeline_dPsum.producer_get_barrier(producer_state_dO_dPsum),
                )
            producer_state_dO_dPsum.advance()

            if const_expr(self.use_2cta_instrs):
                if it == 0:
                    pipeline_Kt.producer_acquire(producer_state_Kt)
                    load_Kt(tma_bar_ptr=pipeline_Kt.producer_get_barrier(producer_state_Kt))
                    pipeline_Kt.producer_commit(producer_state_Kt)
                    producer_state_Kt.advance()

            qh_prev = qh
            m_block_prev = m_block
            m_block += 1
            if m_block == m_block_max:
                m_block = m_block_min
                qh += 1

        if const_expr(self.use_2cta_instrs):
            head_prev = head_kv * G + qh_prev
            mQt_prev = mQt[None, None, head_prev, batch_idx]
            gQt = cute.local_tile(mQt_prev, cute.select(self.mma_tiler_dsq, mode=[1, 2]), (0, None))
            tdKgQt = thr_mma_dK.partition_B(gQt)
            load_Qt, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_Qt, block_in_cluster_coord_vmnk[1], b_cta_layout, tdKgQt, sQt
            )
            load_Qt = copy_utils.tma_producer_copy_fn(load_Qt, pipeline_Qt)
            pipeline_Qt.producer_acquire(producer_state_Qt)
            load_Qt(m_block_prev, producer_state=producer_state_Qt)
            pipeline_Qt.producer_commit(producer_state_Qt)
            producer_state_Qt.advance()

        pipeline_Q.producer_tail(producer_state_Q_LSE.clone())
        pipeline_LSE.producer_tail(producer_state_Q_LSE)
        if const_expr(self.use_2cta_instrs):
            pipeline_Qt.producer_tail(producer_state_Qt)
        pipeline_dO.producer_tail(producer_state_dO_dPsum.clone())
        pipeline_dPsum.producer_tail(producer_state_dO_dPsum)

    @cute.jit
    def mma(
        self,
        tiled_mma_S, tiled_mma_dP, tiled_mma_dV, tiled_mma_dK, tiled_mma_dQ,
        sQ, sQt, sK, sKt, sV, sdO, sdOt, tP, sdSt, sdS, tdS,
        tStS, tdPtdP, tdVtdV, tdKtdK, tdQtdQ,
        dS_cluster_leader_mbar_ptr,
        pipeline_Q, pipeline_Qt, pipeline_Kt, pipeline_dO, pipeline_S_P, pipeline_dS,
        pipeline_dKV, pipeline_dP, pipeline_dQ,
        is_leader_cta,
        tile,
    ):
        n_block, head_kv, batch_idx, m_block_min, m_block_max = tile
        cg = self.cta_group_size
        tSrK = tiled_mma_S.make_fragment_A(sK)
        tSrQ = tiled_mma_S.make_fragment_B(sQ)
        tdPrV = tiled_mma_dP.make_fragment_A(sV)
        tdPrdOt = tiled_mma_dP.make_fragment_B(sdOt)
        tdKrdS = tiled_mma_dK.make_fragment_A(tdS)
        tdKrQ = tiled_mma_dK.make_fragment_B(sQt)
        tdQrdS = tiled_mma_dQ.make_fragment_A(sdS)
        tdQrK = tiled_mma_dQ.make_fragment_B(sKt)
        tdVrdO = tiled_mma_dV.make_fragment_B(sdO)
        tdVrP = tiled_mma_dV.make_fragment_A(tP)

        mma_qk_fn = partial(
            gemm_ptx_w_idx, tiled_mma_S, tStS, tSrK, tSrQ, sA=sK, sB=sQ, zero_init=True,
            cta_group=cg,
        )
        mma_dov_fn = partial(
            gemm_ptx_w_idx, tiled_mma_dP, tdPtdP, tdPrV, tdPrdOt, sA=sV, sB=sdOt,
            zero_init=True, cta_group=cg,
        )
        mma_pdo_fn = partial(
            gemm_ptx_w_idx, tiled_mma_dV, tdVtdV, tdVrP, tdVrdO, sA=None, sB=sdO,
            tA_addr=self.tmem_P_offset, cta_group=cg, tA_offsets=self.pds_kblock_cols,
        )
        mma_dsk_fn = partial(
            gemm_w_idx, tiled_mma_dQ, tdQtdQ, tdQrdS, tdQrK, zero_init=True,
            num_unroll_groups=2 if const_expr(self.use_2cta_instrs) else 1,
        )
        mma_dsq_fn = partial(
            gemm_ptx_w_idx, tiled_mma_dK, tdKtdK, tdKrdS, tdKrQ, sA=None, sB=sQt,
            tA_addr=self.tmem_dS_offset, cta_group=cg, tA_offsets=self.pds_kblock_cols,
        )

        pipeline_Q_consumer = pipeline_Q.make_consumer()
        consumer_state_Q = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.Q_stage
        )
        consumer_state_Qt = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.Q_stage
        )
        consumer_state_Kt = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.single_stage
        )
        consumer_state_dO = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.dO_stage
        )
        producer_phase_acc = Int32(1)
        producer_phase_dQ = Int32(1)
        consumer_state_dS = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, 1
        )
        producer_phase_dKV = Int32(1)
        cta_group = pipeline_S_P.cta_group
        dS_cluster_phase = Int32(0)

        total = (m_block_max - m_block_min) * self.qhead_per_kvhead
        if const_expr(self.use_2cta_instrs):
            if is_leader_cta:
                accumulate_dK = False
                pipeline_Q.consumer_wait(consumer_state_Q)
                pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc)
                mma_qk_fn(B_idx=consumer_state_Q.index)
                pipeline_S_P.sync_object_full.arrive(0, pipeline_S_P.producer_mask, cta_group)
                pipeline_Q.consumer_release(consumer_state_Q)
                consumer_state_Q.advance()

                pipeline_dO.consumer_wait(consumer_state_dO)
                pipeline_dP.sync_object_empty.wait(0, producer_phase_acc)
                mma_dov_fn(B_idx=consumer_state_dO.index)
                pipeline_dP.sync_object_full.arrive(0, pipeline_dP.producer_mask, cta_group)

                producer_phase_acc ^= 1
                pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc)
                mma_pdo_fn(B_idx=consumer_state_dO.index, zero_init=True)
                pipeline_dO.consumer_release(consumer_state_dO)
                consumer_state_dO.advance()

                pipeline_Kt.consumer_wait(consumer_state_Kt)
                for _ in cutlass.range(total - 1, unroll=1):
                    pipeline_Q.consumer_wait(consumer_state_Q)
                    pipeline_dQ.sync_object_empty.wait(0, producer_phase_dQ)
                    mma_qk_fn(B_idx=consumer_state_Q.index)
                    pipeline_S_P.sync_object_full.arrive(0, pipeline_S_P.producer_mask, cta_group)
                    pipeline_Q.consumer_release(consumer_state_Q)
                    consumer_state_Q.advance()
                    pipeline_Qt.consumer_wait(consumer_state_Qt)
                    pipeline_dP.sync_object_empty.wait(0, producer_phase_acc)
                    mma_dsq_fn(B_idx=consumer_state_Qt.index, zero_init=not accumulate_dK)
                    accumulate_dK = True
                    pipeline_Qt.consumer_release(consumer_state_Qt)
                    consumer_state_Qt.advance()
                    pipeline_dO.consumer_wait(consumer_state_dO)
                    mma_dov_fn(B_idx=consumer_state_dO.index)
                    pipeline_dP.sync_object_full.arrive(0, pipeline_dP.producer_mask, cta_group)
                    pipeline_dS.consumer_wait(consumer_state_dS)
                    cute.arch.mbarrier_wait(dS_cluster_leader_mbar_ptr, phase=dS_cluster_phase)
                    mma_dsk_fn()
                    pipeline_dQ.sync_object_full.arrive(0, pipeline_dQ.producer_mask, cta_group)
                    pipeline_dS.consumer_release(consumer_state_dS)
                    consumer_state_dS.advance()
                    dS_cluster_phase ^= 1
                    producer_phase_dQ ^= 1
                    producer_phase_acc ^= 1
                    pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc)
                    mma_pdo_fn(B_idx=consumer_state_dO.index, zero_init=False)
                    pipeline_dO.consumer_release(consumer_state_dO)
                    consumer_state_dO.advance()

                pipeline_S_P.sync_object_full.arrive(0, pipeline_S_P.producer_mask, cta_group)
                pipeline_dKV.sync_object_empty.wait(0, producer_phase_dKV)
                pipeline_dKV.sync_object_full.arrive(0, pipeline_dKV.producer_mask, cta_group)
                pipeline_dKV.sync_object_empty.wait(1, producer_phase_dKV)
                pipeline_Qt.consumer_wait(consumer_state_Qt)
                pipeline_dP.sync_object_empty.wait(0, producer_phase_acc)
                mma_dsq_fn(B_idx=consumer_state_Qt.index, zero_init=not accumulate_dK)
                pipeline_Qt.consumer_release(consumer_state_Qt)
                consumer_state_Qt.advance()
                pipeline_dKV.sync_object_full.arrive(1, pipeline_dKV.producer_mask, cta_group)
                producer_phase_dKV ^= 1
                pipeline_dS.consumer_wait(consumer_state_dS)
                cute.arch.mbarrier_wait(dS_cluster_leader_mbar_ptr, phase=dS_cluster_phase)
                pipeline_dQ.sync_object_empty.wait(0, producer_phase_dQ)
                mma_dsk_fn()
                pipeline_dQ.sync_object_full.arrive(0, pipeline_dQ.producer_mask, cta_group)
                pipeline_dS.consumer_release(consumer_state_dS)
                pipeline_Kt.consumer_release(consumer_state_Kt)
                consumer_state_dS.advance()
                consumer_state_Kt.advance()
        else:
            accumulate_dK = False
            handle_Q = pipeline_Q_consumer.wait_and_advance()
            pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc)
            mma_qk_fn(B_idx=handle_Q.index)
            pipeline_S_P.sync_object_full.arrive(0, pipeline_S_P.producer_mask, cta_group)
            pipeline_dO.consumer_wait(consumer_state_dO)
            pipeline_dP.sync_object_empty.wait(0, producer_phase_acc)
            pipeline_dQ.sync_object_empty.wait(0, producer_phase_acc)
            mma_dov_fn(B_idx=consumer_state_dO.index)
            pipeline_dP.sync_object_full.arrive(0, pipeline_dP.producer_mask, cta_group)
            producer_phase_acc ^= 1
            pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc)
            mma_pdo_fn(B_idx=consumer_state_dO.index, zero_init=True)
            pipeline_dO.consumer_release(consumer_state_dO)
            consumer_state_dO.advance()
            handle_Q_next = handle_Q
            for _ in cutlass.range(total - 1, unroll=1):
                handle_Q_next = pipeline_Q_consumer.wait_and_advance()
                mma_qk_fn(B_idx=handle_Q_next.index)
                pipeline_S_P.sync_object_full.arrive(0, pipeline_S_P.producer_mask, cta_group)
                pipeline_dS.consumer_wait(consumer_state_dS)
                mma_dsq_fn(B_idx=handle_Q.index, zero_init=not accumulate_dK)
                accumulate_dK = True
                handle_Q.release()
                mma_dsk_fn()
                pipeline_dQ.sync_object_full.arrive(0, pipeline_dQ.producer_mask, cta_group)
                pipeline_dS.consumer_release(consumer_state_dS)
                consumer_state_dS.advance()
                pipeline_dO.consumer_wait(consumer_state_dO)
                pipeline_dQ.sync_object_empty.wait(0, producer_phase_acc)
                mma_dov_fn(B_idx=consumer_state_dO.index)
                pipeline_dP.sync_object_full.arrive(0, pipeline_dP.producer_mask, cta_group)
                producer_phase_acc ^= 1
                pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc)
                mma_pdo_fn(B_idx=consumer_state_dO.index, zero_init=False)
                pipeline_dO.consumer_release(consumer_state_dO)
                consumer_state_dO.advance()
                handle_Q = handle_Q_next
            pipeline_S_P.sync_object_full.arrive(0, pipeline_S_P.producer_mask, cta_group)
            pipeline_dKV.sync_object_empty.wait(0, producer_phase_dKV)
            pipeline_dKV.sync_object_full.arrive(0, pipeline_dKV.producer_mask, cta_group)
            pipeline_dKV.sync_object_empty.wait(1, producer_phase_dKV)
            pipeline_dS.consumer_wait(consumer_state_dS)
            mma_dsq_fn(B_idx=handle_Q.index, zero_init=not accumulate_dK)
            pipeline_dKV.sync_object_full.arrive(1, pipeline_dKV.producer_mask, cta_group)
            producer_phase_dKV ^= 1
            mma_dsk_fn()
            pipeline_dQ.sync_object_full.arrive(0, pipeline_dQ.producer_mask, cta_group)
            handle_Q.release()
            pipeline_dS.consumer_release(consumer_state_dS)
            consumer_state_dS.advance()

    @cute.jit
    def split_wg(self, t: cute.Tensor, wg_idx: Int32, num_wg: cutlass.Constexpr[int]):
        reduced_shape = cute.product_each(t.shape)
        rank = len(reduced_shape)
        if const_expr(reduced_shape[1] > 1):
            t = cute.logical_divide(t, (reduced_shape[0], reduced_shape[1] // num_wg))
            coord = (None, (None, wg_idx)) + (None,) * (rank - 2)
        else:
            if const_expr(rank == 3):
                t = cute.logical_divide(
                    t, (reduced_shape[0], reduced_shape[1], reduced_shape[2] // num_wg)
                )
                coord = (None, None, (None, wg_idx))
            else:
                t = cute.logical_divide(
                    t,
                    (reduced_shape[0], reduced_shape[1], reduced_shape[2],
                     reduced_shape[3] // num_wg),
                )
                coord = (None, None, None, (None, wg_idx)) + (None,) * (rank - 4)
        return t[coord]

    @cute.jit
    def store_dS_sw64(self, base_addr, k, m_local, regs):
        regs_i32 = cute.recast_tensor(regs, Int32)
        row_off = 64 * (k % 32) + 2048 * (k // 32)
        sw = ((k % 32) // 2) % 4
        for c in cutlass.range_constexpr(2):
            chunk = (m_local // 16 + c) ^ sw
            addr = base_addr + row_off + chunk * 16
            ptr = cute.make_ptr(Int32, addr, cute.AddressSpace.smem, assumed_align=16)
            dst = cute.make_tensor(ptr, cute.make_layout(4))
            cute.autovec_copy(cute.local_tile(regs_i32, (4,), (c,)), dst)

    @cute.jit
    def compute_loop(
        self,
        thr_mma_S, thr_mma_dP, thr_mma_dV, thr_mma_dK,
        tStS, tdPtdP, tdVtdV, tdKtdK,
        sLSE, sdPsum, sdS, sdS_xchg,
        pipeline_LSE, pipeline_dPsum, pipeline_S_P, pipeline_dS, pipeline_dKV, pipeline_dP,
        dS_cluster_full_mbar_ptr,
        mScales, mAmax,
        sdV, sdK, mdV_tma_tensor, mdK_tma_tensor, tma_atom_dV, tma_atom_dK,
        tiled_copy_r2s_dKV,
        tile,
        xchg_ready_mbar_ptr=None,
    ):
        n_block, head_kv, batch_idx, m_block_min, m_block_max = tile
        scale_log2 = mScales[0]
        ddk = mScales[1]
        dV_scale = mScales[2]
        dK_scale = mScales[3]
        amax_descale = mScales[5]

        sLSE_2D = cute.make_tensor(
            sLSE.iterator,
            cute.make_layout(
                (self.tile_m, self.tile_n, self.Q_stage),
                stride=(1, 0, cute.round_up(self.tile_m, 64)),
            ),
        )
        sdPsum_2D = cute.make_tensor(
            sdPsum.iterator,
            cute.make_layout(
                (self.tile_m, self.tile_n, self.dO_stage),
                stride=(1, 0, cute.round_up(self.tile_m, 64)),
            ),
        )
        sLSE_2D = layout_utils.transpose_view(sLSE_2D)
        sdPsum_2D = layout_utils.transpose_view(sdPsum_2D)

        tidx = cute.arch.thread_idx()[0] % (cute.arch.WARP_SIZE * len(self.compute_warp_ids))
        dp_idx = tidx % 128
        num_wg = len(self.compute_warp_ids) // 4

        tileP_f32_like = self.cta_tiler[1] // 32 * self.p_dtype.width
        tStP = cute.composition(tStS, (cute.make_layout((self.tile_n, tileP_f32_like)), 1, 1))
        tStP = cute.make_tensor(tStS.iterator, tStP.layout)
        tScS = thr_mma_S.partition_C(cute.make_identity_tensor(self.mma_tiler_kq[:2]))
        tScP = cute.composition(tScS, (cute.make_layout((self.tile_n, tileP_f32_like)), 1, 1))
        tdPtdS = cute.composition(tdPtdP, (cute.make_layout((self.tile_n, tileP_f32_like)), 1, 1))

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), Float32
        )
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(8)), Float32
        )
        thr_copy_t2r = copy_utils.make_tmem_copy(tmem_load_atom, num_wg).get_slice(tidx)
        tStS_t2r = thr_copy_t2r.partition_S(tStS)
        tdPtdP_t2r = thr_copy_t2r.partition_S(tdPtdP)
        tScS_t2r = thr_copy_t2r.partition_D(tScS)
        tSsLSE = thr_copy_t2r.partition_D(thr_mma_S.partition_C(sLSE_2D))
        tSsdPsum = thr_copy_t2r.partition_D(thr_mma_dP.partition_C(sdPsum_2D))
        thr_copy_r2t = copy_utils.make_tmem_copy(tmem_store_atom, num_wg).get_slice(tidx)
        tScP_r2t = thr_copy_r2t.partition_S(tScP)
        tStP_r2t = thr_copy_r2t.partition_D(tStP)
        tdPtdS_r2t = thr_copy_r2t.partition_D(tdPtdS)

        wg_idx = tidx // 128


        copy_atom_r2s = sm100_utils_basic.get_smem_store_op(
            LayoutEnum.ROW_MAJOR, self.ds_dtype, Float32, thr_copy_t2r
        )
        thr_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, thr_copy_t2r).get_slice(tidx)
        sdS_epi_layout = sm100_utils_basic.make_smem_layout_epi(
            self.ds_dtype, LayoutEnum.ROW_MAJOR, (self.tile_n, self.tile_m), 1
        )
        sdS_layout = cute.slice_(sdS_epi_layout.outer, (None, None, 0))
        sdS_layout = cute.make_layout((sdS_layout.shape,), stride=(sdS_layout.stride,))
        sdS_epi = cute.make_tensor(sdS.iterator, sdS_layout)
        tRS_sdS = thr_copy_r2s.partition_D(sdS_epi)
        if const_expr(self.use_2cta_instrs):
            sdS_addr = sdS.iterator.toint()
            sdS_xchg_addr = sdS_xchg.iterator.toint()

        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        exchange_stage = cta_rank_in_cluster ^ 1 if const_expr(self.use_2cta_instrs) else Int32(0)

        consumer_state_S_P_dP = pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, 1
        )
        producer_state_dS = pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, 1
        )
        consumer_state_dKV = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, 2
        )
        consumer_state_LSE = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.Q_stage
        )
        consumer_state_dPsum = pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.dO_stage
        )

        amax_run = Float32(0.0)
        total = (m_block_max - m_block_min) * self.qhead_per_kvhead
        m_block = m_block_min
        for iter_idx in cutlass.range(total, unroll=1):
            pipeline_LSE.consumer_wait(consumer_state_LSE)
            tSrLSE_s2r = cute.make_rmem_tensor(tScS_t2r[None, 0, 0, 0].shape, Float32)
            pipeline_S_P.consumer_wait(consumer_state_S_P_dP)
            tSrS_t2r = cute.make_rmem_tensor(tScS_t2r.shape, Float32)
            cute.copy(thr_copy_t2r, tStS_t2r, tSrS_t2r)
            if const_expr(self.use_2cta_instrs):
                if iter_idx > 0:
                    cute.arch.fence_view_async_tmem_load()
                    with cute.arch.elect_one():
                        pipeline_dS.producer_commit(producer_state_dS)
                    producer_state_dS.advance()

            if const_expr(self.is_causal):
                if m_block <= n_block:
                    row_off = (n_block - m_block) * self.tile_n
                    for i in cutlass.range_constexpr(cute.size(tSrS_t2r.shape)):
                        tSrS_t2r[i] = (
                            -Float32.inf
                            if tScS_t2r[i][1] < tScS_t2r[i][0] % self.tile_n + row_off
                            else tSrS_t2r[i]
                        )

            num_stages = cute.size(tScS_t2r, mode=[1])
            tSrP_r2t_f32 = cute.make_rmem_tensor(tScP_r2t.shape, Float32)
            tSrP_r2t = cute.recast_tensor(tSrP_r2t_f32, self.p_dtype)
            for stage in cutlass.range_constexpr(num_stages):
                tSrS_cur = tSrS_t2r[None, stage, 0, 0]
                tSsLSE_cur = tSsLSE[None, stage, 0, 0, consumer_state_LSE.index]
                cute.autovec_copy(tSsLSE_cur, tSrLSE_s2r)
                tSrLSE = tSrLSE_s2r
                for v in cutlass.range_constexpr(cute.size(tSrS_t2r, mode=[0]) // 2):
                    tSrS_cur[2 * v], tSrS_cur[2 * v + 1] = cute.arch.fma_packed_f32x2(
                        (tSrS_cur[2 * v], tSrS_cur[2 * v + 1]),
                        (scale_log2, scale_log2),
                        (-tSrLSE[2 * v], -tSrLSE[2 * v + 1]),
                    )
                    tSrS_cur[2 * v] = cute.math.exp2(tSrS_cur[2 * v], fastmath=True)
                    tSrS_cur[2 * v + 1] = cute.math.exp2(tSrS_cur[2 * v + 1], fastmath=True)
                tSrP_cur = tSrP_r2t[None, stage, 0, 0]
                tSrP_cur.store(tSrS_cur.load().to(self.p_dtype))
                if const_expr(stage == 0):
                    cute.arch.fence_view_async_tmem_load()
                dstP = tStP_r2t[None, stage, None, None]
                dstP = cute.make_tensor(dstP.iterator + (wg_idx * 24 - stage * 8), dstP.layout)
                cute.copy(thr_copy_r2t, tSrP_r2t_f32[None, stage, None, None], dstP)
            cute.arch.fence_view_async_tmem_store()
            cute.arch.sync_warp()
            with cute.arch.elect_one():
                pipeline_S_P.consumer_release(consumer_state_S_P_dP)
            pipeline_LSE.consumer_release(consumer_state_LSE)
            consumer_state_LSE.advance()

            pipeline_dPsum.consumer_wait(consumer_state_dPsum)
            pipeline_dP.consumer_wait(consumer_state_S_P_dP)
            for stage in cutlass.range_constexpr(num_stages):
                tdPrdP_t2r = cute.make_rmem_tensor(tScS_t2r[None, 0, None, None].shape, Float32)
                cute.copy(thr_copy_t2r, tdPtdP_t2r[None, stage, None, None], tdPrdP_t2r)
                cute.arch.fence_view_async_tmem_load()
                tdPrdP_cur = tdPrdP_t2r[None, 0, 0]
                tSrS_cur = tSrS_t2r[None, stage, 0, 0]
                tSsdPsum_cur = tSsdPsum[None, stage, 0, 0, consumer_state_dPsum.index]
                tSrdPsum = cute.make_fragment_like(tSsdPsum_cur, Float32)
                cute.autovec_copy(tSsdPsum_cur, tSrdPsum)
                for v in cutlass.range_constexpr(cute.size(tdPrdP_t2r, mode=[0]) // 2):
                    tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1] = cute.arch.fma_packed_f32x2(
                        (tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1]),
                        (ddk, ddk),
                        (-tSrdPsum[2 * v], -tSrdPsum[2 * v + 1]),
                    )
                    tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1] = cute.arch.mul_packed_f32x2(
                        (tSrS_cur[2 * v], tSrS_cur[2 * v + 1]),
                        (tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1]),
                    )
                for v in cutlass.range_constexpr(cute.size(tdPrdP_t2r, mode=[0]) // 2):
                    amax_run = fmax3(
                        amax_run,
                        cute.math.absf(tdPrdP_cur[2 * v]),
                        cute.math.absf(tdPrdP_cur[2 * v + 1]),
                    )
                tdPrdS_cvt = cute.make_fragment_like(tdPrdP_cur, self.ds_dtype)
                tdPrdS_cvt.store(tdPrdP_cur.load().to(self.ds_dtype))
                if const_expr(stage == 0):
                    pipeline_dS.producer_acquire(producer_state_dS)
                tdPrdS_r2t_f32 = cute.recast_tensor(tdPrdS_cvt, Float32)
                dstS = tdPtdS_r2t[None, stage, 0, 0]
                dstS = cute.make_tensor(dstS.iterator + (wg_idx * 24 - stage * 8), dstS.layout)
                cute.copy(thr_copy_r2t, tdPrdS_r2t_f32, dstS)
                if const_expr(self.use_2cta_instrs):
                    n_local = tScS_t2r[0, stage, 0, 0][0] % self.tile_n
                    m_local = tScS_t2r[0, stage, 0, 0][1] % (self.tile_m // 2)
                    if exchange_stage == stage:
                        self.store_dS_sw64(sdS_xchg_addr, n_local, m_local, tdPrdS_cvt)
                    else:
                        self.store_dS_sw64(
                            sdS_addr, cta_rank_in_cluster * self.tile_n + n_local, m_local,
                            tdPrdS_cvt,
                        )
                else:
                    cute.autovec_copy(tdPrdS_cvt, tRS_sdS[None, stage])

            cute.arch.fence_view_async_tmem_store()
            if const_expr(self.use_2cta_instrs):
                cute.arch.sync_warp()
                with cute.arch.elect_one():
                    pipeline_dP.consumer_release(consumer_state_S_P_dP)
            consumer_state_S_P_dP.advance()
            cute.arch.fence_view_async_shared()
            cute.arch.sync_warp()
            if const_expr(self.use_2cta_instrs):
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(xchg_ready_mbar_ptr)
            pipeline_dPsum.consumer_release(consumer_state_dPsum)
            consumer_state_dPsum.advance()
            if const_expr(not self.use_2cta_instrs):
                with cute.arch.elect_one():
                    pipeline_dS.producer_commit(producer_state_dS)
                producer_state_dS.advance()
            m_block += 1
            if m_block == m_block_max:
                m_block = m_block_min

        if const_expr(self.use_2cta_instrs):
            with cute.arch.elect_one():
                pipeline_dS.producer_commit(producer_state_dS)
            producer_state_dS.advance()

        for i in cutlass.range_constexpr(5):
            amax_run = fmax3(amax_run, cute.arch.shuffle_sync_bfly(amax_run, offset=1 << i))
        if cute.arch.lane_idx() == 0:
            atomic_max_nonneg_f32(mAmax.iterator, amax_run * amax_descale)

        thr_copy_r2s_dKV = tiled_copy_r2s_dKV.get_slice(dp_idx)
        consumer_state_dKV = self.epilogue_dK_or_dV_tma(
            dp_idx, batch_idx, head_kv, n_block, thr_mma_dV, tdVtdV, mdV_tma_tensor, sdV,
            tma_atom_dV, thr_copy_r2s_dKV, pipeline_dKV, consumer_state_dKV, dV_scale,
            int(NamedBarrierBwdSm100.EpilogueWG1), "V",
        )
        consumer_state_dKV = self.epilogue_dK_or_dV_tma(
            dp_idx, batch_idx, head_kv, n_block, thr_mma_dK, tdKtdK, mdK_tma_tensor, sdK,
            tma_atom_dK, thr_copy_r2s_dKV, pipeline_dKV, consumer_state_dKV, dK_scale,
            int(NamedBarrierBwdSm100.EpilogueWG1), "K",
        )

    @cute.jit
    def dQacc_reduce(self, mdQaccum, sdQaccum, thr_mma_dQ, tdQtdQ, pipeline_dQ, tile):
        n_block, head_kv, batch_idx, m_block_min, m_block_max = tile
        G = self.qhead_per_kvhead
        num_reduce_threads = cute.arch.WARP_SIZE * len(self.reduce_warp_ids)
        tidx = cute.arch.thread_idx()[0] % num_reduce_threads
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx() % len(self.reduce_warp_ids))
        is_tma_warp = warp_idx == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(self.dQ_reduce_ncol_t2r)), Float32
        )
        thr_copy_t2r = tcgen05.make_tmem_copy(tmem_load_atom, tdQtdQ).get_slice(tidx)
        tdQtdQ_t2r = thr_copy_t2r.partition_S(tdQtdQ)
        tdQcdQ = thr_mma_dQ.partition_C(cute.make_identity_tensor(self.mma_tiler_dsk[:2]))
        tdQrdQ_t2r_shape = thr_copy_t2r.partition_D(tdQcdQ).shape
        expected_reduce_stages_t2r = self.dQaccum_reduce_stage_t2r // self.cta_group_size
        assert cute.size(tdQrdQ_t2r_shape, mode=[1]) == expected_reduce_stages_t2r
        expected_reduce_stages = self.dQaccum_reduce_stage // self.cta_group_size
        stage_offset = (
            expected_reduce_stages * cta_rank_in_cluster if const_expr(self.use_2cta_instrs) else 0
        )
        thr_copy_dQaccum_r2s = copy_utils.tiled_copy_1d(
            self.dqaccum_dtype, num_reduce_threads, num_copy_elems=128 // self.dqaccum_dtype.width
        ).get_slice(tidx)
        tdQsdQ = thr_copy_dQaccum_r2s.partition_D(sdQaccum)

        dQ_consumer_state = pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, 1
        )
        dQ_tma_store_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.sdQaccum_stage
        )
        total = (m_block_max - m_block_min) * G
        qh = Int32(0)
        m_block = m_block_min
        for _ in cutlass.range(total, unroll=1):
            head_idx = head_kv * G + qh
            mdQaccum_cur = mdQaccum[None, head_idx, batch_idx]
            gdQaccum_ = cute.local_tile(mdQaccum_cur, (self.tile_m * self.tile_hdim,), (None,))
            gdQaccum = cute.flat_divide(
                gdQaccum_, (self.tile_m * self.tile_hdim // self.dQaccum_reduce_stage,)
            )
            pipeline_dQ.consumer_wait(dQ_consumer_state)
            tdQrdQ_t2r = cute.make_rmem_tensor(tdQrdQ_t2r_shape, Float32)
            cute.copy(thr_copy_t2r, tdQtdQ_t2r, tdQrdQ_t2r)
            cute.arch.fence_view_async_tmem_load()
            cute.arch.sync_warp()
            with cute.arch.elect_one():
                pipeline_dQ.consumer_release(dQ_consumer_state)
            dQ_consumer_state.advance()

            gdQaccum_cur = gdQaccum[None, None, m_block]
            tdQrdQ_shape = (
                self.dQ_reduce_ncol,
                self.tile_hdim // self.cta_group_size // self.dQ_reduce_ncol,
            )
            tdQrdQ = cute.make_tensor(tdQrdQ_t2r.iterator, tdQrdQ_shape)
            for stage in cutlass.range_constexpr(cute.size(tdQrdQ, mode=[1])):
                smem_idx = dQ_tma_store_producer_state.index
                tdQsdQ_r2s = tdQsdQ[None, None, smem_idx]
                tdQrdQ_r2s = cute.make_tensor(tdQrdQ[None, stage].iterator, tdQsdQ_r2s.shape)
                cute.copy(thr_copy_dQaccum_r2s, tdQrdQ_r2s, tdQsdQ_r2s)
                cute.arch.fence_view_async_shared()
                self.reduce_sync_barrier.arrive_and_wait()
                if is_tma_warp:
                    with cute.arch.elect_one():
                        copy_utils.cpasync_reduce_bulk_add_f32(
                            sdQaccum[None, smem_idx].iterator,
                            gdQaccum_cur[None, stage + stage_offset].iterator,
                            self.tma_copy_bytes["dQ"],
                        )
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(self.sdQaccum_stage - 1, read=True)
                self.reduce_sync_barrier.arrive_and_wait()
                dQ_tma_store_producer_state.advance()

            m_block += 1
            if m_block == m_block_max:
                m_block = m_block_min
                qh += 1

        if is_tma_warp:
            cute.arch.cp_async_bulk_wait_group(0, read=True)
        self.reduce_sync_barrier.arrive_and_wait()
        cute.arch.cp_async_bulk_wait_group(0, read=True)

    @cute.jit
    def epilogue_dK_or_dV_tma(
        self,
        tidx: Int32,
        batch_idx: Int32,
        head_kv: Int32,
        n_block: Int32,
        thr_mma: cute.ThrMma,
        tdKVtdKV: cute.Tensor,
        mdKV: cute.Tensor,
        sdKV: cute.Tensor,
        tma_atom_dKV: cute.CopyAtom,
        thr_copy_r2s_dKV: cute.TiledCopy,
        pipeline_dKV,
        consumer_state_dKV,
        scale: Float32,
        barrier_id: Int32,
        K_or_V: cutlass.Constexpr[str],
    ):
        tile_hdim = self.tile_hdim if const_expr(K_or_V == "K") else self.tile_hdimv
        dtype = self.dk_dtype if const_expr(K_or_V == "K") else self.dv_dtype
        epi_tile = self.sdK_epi_tile if const_expr(K_or_V == "K") else self.sdV_epi_tile
        num_compute_threads = cute.arch.WARP_SIZE * len(self.compute_warp_ids)
        wg_idx = (cute.arch.thread_idx()[0] % num_compute_threads) // 128
        num_wg = num_compute_threads // 128
        leader_warp = (cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4) == 0
        cta_group_tile_n = const_expr(self.tile_n * self.cta_group_size)

        sdKV = sdKV[None, None, wg_idx]
        tdKVsdKV_r2s = thr_copy_r2s_dKV.partition_D(sdKV)
        mdKV_cur = mdKV[None, None, head_kv, batch_idx]
        gdKV_p = cute.local_tile(mdKV_cur, (self.tile_n, tile_hdim), (n_block, 0))
        gdKV = self.split_wg(gdKV_p, wg_idx, num_wg)
        gdKV_epi = cute.local_tile(gdKV, epi_tile, (0, None))
        tdKVsdKV, tdKVgdKV = cpasync.tma_partition(
            tma_atom_dKV, 0, cute.make_layout(1),
            cute.group_modes(sdKV, 0, 2), cute.group_modes(gdKV_epi, 0, 2),
        )
        num_epi_stages = cute.size(tdKVgdKV.shape[1])
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(math.gcd(32, self.tile_hdim // 2))),
            Float32,
        )
        pipeline_dKV.consumer_wait(consumer_state_dKV)
        for epi_stage in cutlass.range_constexpr(num_epi_stages):
            thr_copy_t2r = tcgen05.make_tmem_copy(tmem_load_atom, tdKVtdKV).get_slice(tidx)
            tdKVtdKV_t2r_p = thr_copy_t2r.partition_S(tdKVtdKV)
            tdKVtdKV_t2r = self.split_wg(tdKVtdKV_t2r_p, wg_idx, num_wg)[None, None, 0, 0]
            if const_expr(num_epi_stages > 1):
                tdKVtdKV_t2r = tdKVtdKV_t2r[None, epi_stage]
            cdKV = cute.make_identity_tensor((cta_group_tile_n, tile_hdim))
            tdKVcdKV = thr_mma.partition_C(cdKV)
            tdKVcdKV_t2r_p = thr_copy_t2r.partition_D(tdKVcdKV)
            tdKVcdKV_t2r = self.split_wg(tdKVcdKV_t2r_p, wg_idx, num_wg)[None, None, 0, 0]
            if const_expr(num_epi_stages > 1):
                tdKVcdKV_t2r = tdKVcdKV_t2r[None, epi_stage]
            tdKVrdKV_t2r = cute.make_rmem_tensor(tdKVcdKV_t2r.shape, Float32)
            cute.copy(thr_copy_t2r, tdKVtdKV_t2r, tdKVrdKV_t2r)
            cute.arch.fence_view_async_tmem_load()
            for i in cutlass.range(cute.size(tdKVrdKV_t2r.shape) // 2, unroll_full=True):
                tdKVrdKV_t2r[2 * i], tdKVrdKV_t2r[2 * i + 1] = cute.arch.mul_packed_f32x2(
                    (tdKVrdKV_t2r[2 * i], tdKVrdKV_t2r[2 * i + 1]), (scale, scale)
                )
            tdKVrdKV = cute.make_rmem_tensor(tdKVrdKV_t2r.shape, dtype)
            tdKVrdKV.store(tdKVrdKV_t2r.load().to(dtype))
            tdKVrdKV_r2s = cute.make_tensor(tdKVrdKV.iterator, tdKVsdKV_r2s.shape)
            cute.copy(thr_copy_r2s_dKV, tdKVrdKV_r2s, tdKVsdKV_r2s)
            cute.arch.fence_view_async_shared()
            cute.arch.barrier(barrier_id=barrier_id + wg_idx, number_of_threads=128)
            if leader_warp:
                cute.copy(tma_atom_dKV, tdKVsdKV, tdKVgdKV[None, epi_stage])
                if const_expr(epi_stage < num_epi_stages - 1):
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0, read=True)
                cute.arch.barrier_arrive(
                    barrier_id=barrier_id + wg_idx, number_of_threads=128 + cute.arch.WARP_SIZE
                )
            cute.arch.fence_view_async_shared()
            cute.arch.barrier(
                barrier_id=barrier_id + wg_idx, number_of_threads=128 + cute.arch.WARP_SIZE
            )
        if leader_warp:
            cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(0, read=True)
        cute.arch.sync_warp()
        with cute.arch.elect_one():
            pipeline_dKV.consumer_release(consumer_state_dKV)
        consumer_state_dKV.advance()
        return consumer_state_dKV


class FP8AttnBwdPreprocess:
    rows_per_cta = 32
    threads_per_row = 8

    def __init__(self, head_dim: int = 128, zero_acc: bool = True):
        assert head_dim == 128
        self.head_dim = head_dim
        self.elems_per_thread = head_dim // self.threads_per_row
        self.zero_acc = zero_acc

    @cute.jit
    def __call__(
        self,
        mO: cute.Tensor,
        mdO: cute.Tensor,
        mStats: cute.Tensor,
        mdPsum: cute.Tensor,
        mLSE: cute.Tensor,
        mdQaccum: cute.Tensor,
        mQsi: cute.Tensor,
        mKsi: cute.Tensor,
        mVsi: cute.Tensor,
        mdOsi: cute.Tensor,
        mSs: cute.Tensor,
        mdPs: cute.Tensor,
        mScales: cute.Tensor,
        mAmax: cute.Tensor,
        attn_scale: Float32,
        stream: cuda.CUstream = None,
    ):
        mO, mdO, mdQaccum = [assume_tensor_aligned(t) for t in (mO, mdO, mdQaccum)]
        b = cute.size(mO.shape[0])
        s = cute.size(mO.shape[1])
        h = cute.size(mO.shape[2])
        grid = (cute.ceil_div(s, self.rows_per_cta), h, b)
        self.kernel(
            mO, mdO, mStats, mdPsum, mLSE, mdQaccum, mQsi, mKsi, mVsi, mdOsi, mSs, mdPs,
            mScales, mAmax, attn_scale,
        ).launch(
            grid=grid,
            block=[self.rows_per_cta * self.threads_per_row, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mO: cute.Tensor,
        mdO: cute.Tensor,
        mStats: cute.Tensor,
        mdPsum: cute.Tensor,
        mLSE: cute.Tensor,
        mdQaccum: cute.Tensor,
        mQsi: cute.Tensor,
        mKsi: cute.Tensor,
        mVsi: cute.Tensor,
        mdOsi: cute.Tensor,
        mSs: cute.Tensor,
        mdPs: cute.Tensor,
        mScales: cute.Tensor,
        mAmax: cute.Tensor,
        attn_scale: Float32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bx, head, batch = cute.arch.block_idx()
        row = tidx // self.threads_per_row
        part = tidx % self.threads_per_row
        s_idx = bx * self.rows_per_cta + row

        descale_dO = mdOsi[0]
        scale_s = mSs[0]
        scale_dP = mdPs[0]
        k_fac = scale_dP * attn_scale / scale_s

        E = self.elems_per_thread
        gO = cute.local_tile(mO[batch, s_idx, head, None], (E,), (part,))
        gdO = cute.local_tile(mdO[batch, s_idx, head, None], (E,), (part,))
        rO = cute.make_rmem_tensor((E,), mO.element_type)
        rdO = cute.make_rmem_tensor((E,), mdO.element_type)
        cute.autovec_copy(gO, rO)
        cute.autovec_copy(gdO, rdO)
        o = rO.load().to(Float32)
        do = rdO.load().to(Float32)
        prod = o * do
        acc = prod.reduce(cute.ReductionOp.ADD, Float32(0.0), 0)
        for i in cutlass.range_constexpr(3):
            acc = acc + cute.arch.shuffle_sync_bfly(acc, offset=1 << i)
        if part == 0:
            mdPsum[batch, head, s_idx] = acc * descale_dO * k_fac
            mLSE[batch, head, s_idx] = mStats[batch, head, s_idx] * LOG2_E - cute.math.log2(
                scale_s, fastmath=False
            )

        if const_expr(self.zero_acc):
            n_zero = self.rows_per_cta * self.head_dim // (self.rows_per_cta * self.threads_per_row)
            base = bx * self.rows_per_cta * self.head_dim + tidx * n_zero
            gAcc = cute.local_tile(mdQaccum[batch, head, None], (n_zero,), (base // n_zero,))
            rZ = cute.make_rmem_tensor((n_zero,), Float32)
            rZ.fill(0.0)
            cute.autovec_copy(rZ, gAcc)

        if bx == 0 and head == 0 and batch == 0 and tidx == 0:
            dq = mQsi[0]
            dk = mKsi[0]
            dv = mVsi[0]
            mScales[0] = attn_scale * LOG2_E * dq * dk
            mScales[1] = descale_dO * dv * k_fac
            mScales[2] = descale_dO / scale_s
            mScales[3] = dq / scale_dP
            mScales[4] = dk / scale_dP
            mScales[5] = Float32(1.0) / scale_dP
            mScales[6] = Float32(0.0)
            mScales[7] = Float32(0.0)
            mAmax[0] = Float32(0.0)


class FP8AttnBwdPostprocess:

    tile_m = 128

    def __init__(self, head_dim: int = 128, two_cta: bool = True, rezero: bool = False):
        assert head_dim == 128
        self.head_dim = head_dim
        self.two_cta = two_cta
        self.rezero = rezero

    @cute.jit
    def __call__(
        self,
        mdQaccum: cute.Tensor,
        mdQ: cute.Tensor,
        mScales: cute.Tensor,
        stream: cuda.CUstream = None,
    ):
        mdQaccum, mdQ = [assume_tensor_aligned(t) for t in (mdQaccum, mdQ)]
        b = cute.size(mdQ.shape[0])
        s = cute.size(mdQ.shape[1])
        h = cute.size(mdQ.shape[2])
        grid = (s // self.tile_m, h, b)
        self.kernel(mdQaccum, mdQ, mScales).launch(
            grid=grid, block=[self.tile_m, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(self, mdQaccum: cute.Tensor, mdQ: cute.Tensor, mScales: cute.Tensor):
        m, _, _ = cute.arch.thread_idx()
        m_block, head, batch = cute.arch.block_idx()
        scale = mScales[4]
        tile_elems = self.tile_m * self.head_dim
        gAcc = cute.local_tile(mdQaccum[batch, head, None], (tile_elems,), (m_block,))
        row = m_block * self.tile_m + m
        for blk in cutlass.range_constexpr(self.head_dim // 32):
            r = cute.make_rmem_tensor((32,), Float32)
            for g in cutlass.range_constexpr(8):
                d0 = blk * 32 + g * 4
                if const_expr(self.two_cta):
                    rank = m // 64
                    t = m % 64 + 64 * (d0 // 64)
                    off4 = (rank * 8 + (d0 % 64) // 8) * 256 + ((d0 % 8) // 4) * 128 + t
                else:
                    off4 = (d0 // 32) * 1024 + ((d0 % 32) // 4) * 128 + m
                src = cute.local_tile(gAcc, (4,), (off4,))
                cute.autovec_copy(src, cute.local_tile(r, (4,), (g,)))
            v = r.load() * scale
            rb = cute.make_rmem_tensor((32,), mdQ.element_type)
            rb.store(v.to(mdQ.element_type))
            dst = cute.local_tile(mdQ[batch, row, head, None], (32,), (blk,))
            cute.autovec_copy(rb, dst)
        if const_expr(self.rezero):
            cute.arch.barrier()
            z = cute.make_rmem_tensor((4,), Float32)
            z.fill(0.0)
            for j in cutlass.range_constexpr(tile_elems // (4 * self.tile_m)):
                cute.autovec_copy(z, cute.local_tile(gAcc, (4,), (j * self.tile_m + m,)))


_T2C = {torch.float8_e4m3fn: cutlass.Float8E4M3FN, torch.float8_e5m2: cutlass.Float8E5M2}
_compiled = {}


def _ct(t, align=16):
    if t.dtype in _T2C:
        ct = from_dlpack(t.view(torch.uint8).detach(), assumed_align=align, enable_tvm_ffi=True)
        ct.element_type = _T2C[t.dtype]
    else:
        ct = from_dlpack(t.detach(), assumed_align=align, enable_tvm_ffi=True)
    return ct.mark_layout_dynamic(leading_dim=t.ndim - 1)


def _arg(t):
    return t.view(torch.uint8) if t.dtype in _T2C else t


def supported(q, k, v, causal_or_none: bool = True) -> bool:
    if q.dim() != 4 or k.dim() != 4:
        return False
    b, s, h, d = q.shape
    return (
        d == 128 and v.shape[-1] == 128 and k.shape[1] == s and h % k.shape[2] == 0
        and s % 256 == 0 and causal_or_none
        and q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1
        and torch.cuda.get_device_capability(q.device)[0] == 10
    )


def _fp8_attn_bwd_impl(q, k, v, o, do, stats, q_si, k_si, v_si, do_si, s_scale, dp_scale,
                       dp_amax, attn_scale: float, causal: bool, out_sbhd: bool,
                       dk_out=None, dv_out=None, post=None):
    b, s, h, d = q.shape
    hkv = k.shape[2]
    dev = q.device
    if dk_out is not None:
        dq, dk, dv = None, dk_out, dv_out
    elif out_sbhd:
        dq = torch.empty(s, b, h, d, dtype=torch.bfloat16, device=dev).permute(1, 0, 2, 3)
        dk = torch.empty(s, b, hkv, d, dtype=torch.bfloat16, device=dev).permute(1, 0, 2, 3)
        dv = torch.empty(s, b, hkv, d, dtype=torch.bfloat16, device=dev).permute(1, 0, 2, 3)
    else:
        dq = torch.empty(b, s, h, d, dtype=torch.bfloat16, device=dev)
        dk = torch.empty(b, s, hkv, d, dtype=torch.bfloat16, device=dev)
        dv = torch.empty(b, s, hkv, d, dtype=torch.bfloat16, device=dev)
    dpsum = torch.empty(b, h, s, dtype=torch.float32, device=dev)
    lse2 = torch.empty(b, h, s, dtype=torch.float32, device=dev)
    dqacc = torch.empty(b, h, s * d, dtype=torch.float32, device=dev)
    scales = torch.empty(8, dtype=torch.float32, device=dev)
    scal = [t.reshape(1) for t in (q_si, k_si, v_si, do_si, s_scale, dp_scale)]
    amax = dp_amax.reshape(1)

    key = ("pre", d)
    if key not in _compiled:
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        args = [_ct(o), _ct(do), _ct(stats), _ct(dpsum), _ct(lse2), _ct(dqacc)]
        args += [_ct(t, align=4) for t in scal] + [_ct(scales), _ct(amax, align=4)]
        _compiled[key] = cute.compile(FP8AttnBwdPreprocess(d), *args, Float32(attn_scale),
                                      stream, options="--enable-tvm-ffi")
    _compiled[key](o, _arg(do), stats, dpsum, lse2, dqacc, *scal, scales, amax, attn_scale)

    key = ("main", d, h // hkv, causal)
    if key not in _compiled:
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        args = [_ct(q), _ct(k), _ct(v), _ct(do), _ct(lse2), _ct(dpsum), _ct(dqacc), _ct(dk),
                _ct(dv), _ct(scales), _ct(amax, align=4)]
        _compiled[key] = cute.compile(
            FP8FlashAttnBwdSm100(d, h // hkv, causal, use_2cta=True, Q_stage=3, dO_stage=2),
            *args, stream, options="--enable-tvm-ffi",
        )
    _compiled[key](_arg(q), _arg(k), _arg(v), _arg(do), lse2, dpsum, dqacc, dk, dv, scales, amax)

    if post is not None:
        post(dqacc, scales)
        return dq, dk, dv
    if dq is None:
        dq = (torch.empty(s, b, h, d, dtype=torch.bfloat16, device=dev).permute(1, 0, 2, 3)
              if out_sbhd else torch.empty(b, s, h, d, dtype=torch.bfloat16, device=dev))
    key = ("post", d)
    if key not in _compiled:
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        _compiled[key] = cute.compile(FP8AttnBwdPostprocess(d, True), _ct(dqacc), _ct(dq),
                                      _ct(scales), stream, options="--enable-tvm-ffi")
    _compiled[key](dqacc, dq, scales)
    return dq, dk, dv


@torch.library.custom_op("optimized::fp8_attn_bwd", mutates_args=("dp_amax",), device_types="cuda")
def fp8_attn_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    do: torch.Tensor,
    stats: torch.Tensor,
    q_si: torch.Tensor,
    k_si: torch.Tensor,
    v_si: torch.Tensor,
    do_si: torch.Tensor,
    s_scale: torch.Tensor,
    dp_scale: torch.Tensor,
    dp_amax: torch.Tensor,
    attn_scale: float,
    causal: bool,
    out_sbhd: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dq, dk, dv = _fp8_attn_bwd_impl(q, k, v, o, do, stats, q_si, k_si, v_si, do_si, s_scale,
                                     dp_scale, dp_amax, attn_scale, causal, out_sbhd)
    return dq, dk, dv


@fp8_attn_bwd.register_fake
def _(q, k, v, o, do, stats, q_si, k_si, v_si, do_si, s_scale, dp_scale, dp_amax, attn_scale,
      causal, out_sbhd):
    b, s, h, d = q.shape
    hkv = k.shape[2]
    mk = lambda hh: (torch.empty(s, b, hh, d, dtype=torch.bfloat16, device=q.device).permute(1, 0, 2, 3)
                     if out_sbhd else torch.empty(b, s, hh, d, dtype=torch.bfloat16, device=q.device))
    return mk(h), mk(hkv), mk(hkv)
