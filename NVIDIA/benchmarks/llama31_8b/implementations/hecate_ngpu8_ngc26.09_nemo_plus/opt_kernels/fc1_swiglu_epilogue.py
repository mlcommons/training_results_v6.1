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
# Portions adapted from the CUTLASS CuTe DSL block-scaled persistent dense GEMM
# (examples/python/CuTeDSL/cute/blackwell/kernel/blockscaled_gemm/dense_blockscaled_gemm_persistent.py
# and cutlass.kernels.rubin.gemm.base.dense_blockscaled_gemm_persistent) and the CuTe DSL GEMM
# tutorial's CLC scheduler warp. Those portions are subject to the following license:
#
# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
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
from __future__ import annotations

import os
import sys
from typing import Optional, Tuple, Type

import cuda.bindings.driver as cuda
import numpy as np
import torch

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import cutlass.utils.rubin_helpers as sm107_utils
from cutlass.cute.nvgpu import cpasync, tcgen05, warp, OperandMajorMode
from cutlass.cute.nvgpu.tcgen05.mma import CollectorOp
from cutlass.cute.runtime import from_dlpack, make_ptr
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.utils.gemm.sm100 import (
    epilogue_smem_copy_and_partition,
    epilogue_tmem_copy_and_partition,
    transform_partitioned_tensor_layout,
)
from cutlass.kernels.rubin.gemm.base.dense_blockscaled_gemm_persistent import (
    Sm107BlockScaledPersistentDenseGemmKernel,
)

_NVFP4_FACTOR_INV = float(np.float32(1.0) / np.float32(6.0 * 6.0 * 448.0 * 448.0))
_RHT_DIM = 16


class Fc1SwigluAmaxGemmKernel(Sm107BlockScaledPersistentDenseGemmKernel):

    def __init__(
        self,
        ffn: int,
        mma_inst_shape: Tuple[int, int, int] = (256, 256, 128),
        mma_tiler: Tuple[int, int, int] = (256, 256, 256),
        cluster_shape_mn: Tuple[int, int] = (2, 1),
        swizzle_size: int = 1,
        raster_order: str = "m",
        prefetch_dist: Optional[int] = None,
        b_interleave: str = "tma",
        rht_mode: str = "ldsm",
        with_rht_amax: bool = True,
        with_swiglu: bool = True,
        split_cols: int = 4,
        swiglu_math: str = "te_fast",
        epi_warps: int = 8,
        max_ab_stages: int = 6,
        store_mode: str = "tma",
        pairs_in_flight: int = 2,
        simt_store_bits: int = 128,
        skip_swiglu_store: bool = False,
        scheduler: str = "clc",
        num_clc_stage: int = 2,
    ):
        if scheduler not in ("clc", "static"):
            raise ValueError("scheduler must be clc or static")
        if num_clc_stage < 1:
            raise ValueError("num_clc_stage must be >= 1")
        super().__init__(
            16, mma_inst_shape, mma_tiler, cluster_shape_mn,
            prefetch_dist=prefetch_dist, swizzle_size=swizzle_size, raster_order=raster_order,
            scheduler_type=(
                utils.ClcDynamicPersistentTileScheduler if scheduler == "clc"
                else utils.StaticPersistentTileScheduler
            ),
        )
        self.scheduler = scheduler
        self.num_clc_stage = num_clc_stage if self.use_clc_dynamic_scheduler else 0
        self.threads_per_cta = self.threads_per_warp * (
            len(self.epilog_warp_id) + 2 + (1 if self.use_clc_dynamic_scheduler else 0)
        )
        if epi_warps not in (4, 8):
            raise ValueError("epi_warps must be 4 or 8")
        if store_mode not in ("simt", "tma"):
            raise ValueError("store_mode must be simt or tma")
        if store_mode == "tma" and pairs_in_flight < 2:
            raise ValueError("pairs_in_flight must be >= 2")
        self.store_mode = store_mode
        self.pairs_in_flight = pairs_in_flight
        self.skip_swiglu_store = skip_swiglu_store
        self.c_stages_per_pair = 3 if store_mode == "tma" else 1
        if simt_store_bits not in (128, 256):
            raise ValueError("simt_store_bits must be 128 or 256")
        self.simt_store_bits = simt_store_bits
        self.c_stages_per_group = self.c_stages_per_pair * (pairs_in_flight if store_mode == "tma" else 2)
        self.epi_warps = epi_warps
        self.epi_groups = epi_warps // 4
        self.max_ab_stages = max_ab_stages
        if epi_warps == 8:
            self.epilog_warp_id = tuple(range(8))
            self.mma_warp_id = 8
            self.tma_warp_id = 9
            self.sched_warp_id = 10
            self.threads_per_cta = self.threads_per_warp * len(
                (self.mma_warp_id, self.tma_warp_id, *self.epilog_warp_id)
            ) + (self.threads_per_warp if self.use_clc_dynamic_scheduler else 0)
            self.epilog_sync_barrier = pipeline.NamedBarrier(
                barrier_id=1, num_threads=self.threads_per_warp * len(self.epilog_warp_id),
            )
            self.tmem_alloc_barrier = pipeline.NamedBarrier(
                barrier_id=2,
                num_threads=self.threads_per_warp * len((self.mma_warp_id, *self.epilog_warp_id)),
            )
            self.epilogue_warp_id = self.epilog_warp_id
            self.epilog_sync_bar_id = self.epilog_sync_barrier.barrier_id
        if b_interleave not in ("tma", "perm"):
            raise ValueError(f"unknown b_interleave {b_interleave}")
        if rht_mode not in ("ldsm", "simt"):
            raise ValueError(f"unknown rht_mode {rht_mode}")
        self.b_interleave = b_interleave
        self.rht_mode = rht_mode
        if swiglu_math not in ("exact", "fastexp", "rcp", "fast", "te_fast"):
            raise ValueError(f"unknown swiglu_math {swiglu_math}")
        self.with_rht_amax = with_rht_amax
        self.with_swiglu = with_swiglu
        self.split_cols = split_cols
        self.swiglu_math = swiglu_math
        self.ffn = ffn
        self.tile_n = mma_tiler[1]
        self.half_n = self.tile_n // 2
        if ffn % self.half_n != 0:
            raise ValueError("ffn must be a multiple of half the N tile")
        self.n_blocks = ffn // self.half_n
        self.n_total = 2 * ffn

    def _compute_stages(
        self, tiled_mma, mma_tiler_mnk, a_dtype, b_dtype, epi_tile, c_dtype, c_layout, sf_dtype,
        sf_vec_size, smem_capacity, occupancy, with_breuse,
    ):
        num_acc_stage = 1 if (with_breuse and mma_tiler_mnk[1] in {192, 256}) else 2
        num_c_stage = 2
        a_one = sm100_utils.make_smem_layout_a(tiled_mma, mma_tiler_mnk, a_dtype, 1)
        b_one = sm100_utils.make_smem_layout_b(tiled_mma, mma_tiler_mnk, b_dtype, 1)
        sfa_one = blockscaled_utils.make_smem_layout_sfa(tiled_mma, mma_tiler_mnk, sf_vec_size, 1)
        sfb_one = blockscaled_utils.make_smem_layout_sfb(tiled_mma, mma_tiler_mnk, sf_vec_size, 1)
        c_one = sm100_utils.make_smem_layout_epi(c_dtype, c_layout, epi_tile, 1)
        ab_bytes = (
            cute.size_in_bytes(a_dtype, a_one) + cute.size_in_bytes(b_dtype, b_one)
            + cute.size_in_bytes(sf_dtype, sfa_one) + cute.size_in_bytes(sf_dtype, sfb_one)
        )
        mbar_helpers_bytes = 1024
        c_bytes_per_stage = cute.size_in_bytes(c_dtype, c_one)
        c_bytes = c_bytes_per_stage * num_c_stage
        num_ab_stage = (smem_capacity // occupancy - (mbar_helpers_bytes + c_bytes)) // ab_bytes
        num_ab_stage = min(num_ab_stage, self.max_ab_stages)
        num_c_stage += (
            smem_capacity - occupancy * ab_bytes * num_ab_stage - occupancy * (mbar_helpers_bytes + c_bytes)
        ) // (occupancy * c_bytes_per_stage)
        need = self.c_stages_per_group * self.epi_groups
        num_c_stage -= num_c_stage % need
        if num_c_stage < need:
            raise ValueError(f"need {need} epilogue smem stages, have {num_c_stage}: lower max_ab_stages")
        return num_acc_stage, num_ab_stage, num_c_stage

    @cute.jit
    def __call__(
        self,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer,
        sfb_ptr: cute.Pointer,
        gu_ptr: cute.Pointer,
        sw_ptr: cute.Pointer,
        rht: cute.Tensor,
        amax: cute.Tensor,
        amax_a: cute.Tensor,
        amax_b: cute.Tensor,
        problem_mnkl: Tuple[int, int, int, int],
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        self.a_dtype: Type[cutlass.Numeric] = a_ptr.value_type
        self.b_dtype: Type[cutlass.Numeric] = b_ptr.value_type
        self.sf_dtype: Type[cutlass.Numeric] = sfa_ptr.value_type
        self.c_dtype: Type[cutlass.Numeric] = gu_ptr.value_type
        self.a_major_mode = OperandMajorMode.K
        self.b_major_mode = OperandMajorMode.K
        self.c_layout = cutlass.tensor_utils.LayoutEnum.ROW_MAJOR

        m, n, k, l = problem_mnkl
        self._setup_attributes()

        m_ = cute.assume(m, 32)
        k_ = cute.assume(k, 256)
        ffn = self.ffn
        hn = self.half_n
        nb = self.n_blocks
        a_layout = cute.make_ordered_layout((m_, k_, l), order=(1, 0, 2))
        a_tensor = cute.make_tensor(a_ptr, a_layout)
        if cutlass.const_expr(self.b_interleave == "tma"):
            b_layout = cute.make_layout(
                ((hn, 2, nb), k_, l), stride=((k_, ffn * k_, hn * k_), 1, 2 * ffn * k_)
            )
        else:
            b_layout = cute.make_ordered_layout((cute.assume(n, 32), k_, l), order=(1, 0, 2))
        b_tensor = cute.make_tensor(b_ptr, b_layout)
        c_layout = cute.make_ordered_layout((m_, cute.assume(n, 32), l), order=(1, 0, 2))
        c_tensor = cute.make_tensor(gu_ptr, c_layout)
        gu_tensor = cute.make_tensor(gu_ptr, cute.make_ordered_layout((m_, 2 * ffn, l), order=(1, 0, 2)))
        sw_tensor = cute.make_tensor(sw_ptr, cute.make_ordered_layout((m_, ffn, l), order=(1, 0, 2)))

        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(a_tensor.shape, self.sf_vec_size)
        sfa_tensor = cute.make_tensor(sfa_ptr, sfa_layout)
        if cutlass.const_expr(self.b_interleave == "tma"):
            kb = k_ // 64
            sfb_layout = cute.make_layout(
                ((32, 4, (2, nb)), (16, 4, kb), l),
                stride=((16, 4, (nb * 512 * kb, 512 * kb)), (0, 1, 512), 2 * nb * 512 * kb),
            )
        else:
            sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(b_tensor.shape, self.sf_vec_size)
        sfb_tensor = cute.make_tensor(sfb_ptr, sfb_layout)

        atom_layout_mnk = (1, 1, 1)
        permutation_mnk = self._get_mma_permutation_mnk()
        tiled_mma = sm107_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype, self.b_dtype, self.a_major_mode, self.b_major_mode,
            self.sf_dtype, self.sf_vec_size, self.cta_group, self.mma_inst_shape,
            a_collector_op=CollectorOp.DISCARD, b_collector_op=CollectorOp.DISCARD,
            atom_layout_mnk=atom_layout_mnk, permutation_mnk=permutation_mnk,
        )
        tiled_mma.set(tcgen05.Field.NEGATE_A, False)
        tiled_mma.set(tcgen05.Field.NEGATE_B, False)
        tiled_mma_sfb = sm107_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype, self.b_dtype, self.a_major_mode, self.b_major_mode,
            self.sf_dtype, self.sf_vec_size, cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_sfb,
            a_collector_op=CollectorOp.DISCARD, b_collector_op=CollectorOp.DISCARD,
        )
        tiled_mma_sfb.set(tcgen05.Field.NEGATE_A, False)
        tiled_mma_sfb.set(tcgen05.Field.NEGATE_B, False)

        tiled_mma_bkeep = None
        tiled_mma_breuse = None
        if cutlass.const_expr(self.enable_breuse):
            tiled_mma_bkeep = sm107_utils.make_blockscaled_trivial_tiled_mma(
                self.a_dtype, self.b_dtype, self.a_major_mode, self.b_major_mode,
                self.sf_dtype, self.sf_vec_size, self.cta_group, self.mma_inst_shape,
                a_collector_op=CollectorOp.DISCARD, b_collector_op=CollectorOp.FILL,
                atom_layout_mnk=atom_layout_mnk, permutation_mnk=permutation_mnk,
            )
            tiled_mma_bkeep.set(tcgen05.Field.NEGATE_A, False)
            tiled_mma_bkeep.set(tcgen05.Field.NEGATE_B, False)
            tiled_mma_breuse = sm107_utils.make_blockscaled_trivial_tiled_mma(
                self.a_dtype, self.b_dtype, self.a_major_mode, self.b_major_mode,
                self.sf_dtype, self.sf_vec_size, self.cta_group, self.mma_inst_shape,
                a_collector_op=CollectorOp.DISCARD, b_collector_op=CollectorOp.LASTUSE,
                atom_layout_mnk=atom_layout_mnk, permutation_mnk=permutation_mnk,
            )
            tiled_mma_breuse.set(tcgen05.Field.NEGATE_A, False)
            tiled_mma_breuse.set(tcgen05.Field.NEGATE_B, False)

        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        a_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op, a_tensor, a_smem_layout, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape,
        )
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id)
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op, b_tensor, b_smem_layout, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape,
        )
        sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        sfa_smem_layout = cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0))
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            sfa_op, sfa_tensor, sfa_smem_layout, self.mma_tiler, tiled_mma,
            self.cluster_layout_vmnk.shape, internal_type=cutlass.Int16,
        )
        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(self.cluster_shape_mn, tiled_mma.thr_id)
        sfb_smem_layout = cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0))
        tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op, sfb_tensor, sfb_smem_layout, self.mma_tiler_sfb, tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk.shape, internal_type=cutlass.Int16,
        )

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        sfa_copy_size = cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
        sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        self.num_tma_load_bytes = (a_copy_size + b_copy_size + sfa_copy_size + sfb_copy_size) * atom_thr_size

        epi_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
        tma_atom_gu, tma_tensor_gu = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), gu_tensor, epi_smem_layout, self.epi_tile,
        )
        tma_atom_sw, tma_tensor_sw = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), sw_tensor, epi_smem_layout, self.epi_tile,
        )

        self.tile_sched_params, grid = self._compute_grid(
            c_tensor, self.cta_tile_shape_mnk, self.cluster_shape_mn, max_active_clusters,
            self.swizzle_size, self.raster_order == "m",
        )

        self.buffer_align_bytes = 1024

        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            ab_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            acc_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            clc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.num_clc_stage]
            clc_response: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, (self.num_clc_response_bytes // 4) * max(1, self.num_clc_stage)],
                16,
            ]
            sC: cute.struct.Align[
                cute.struct.MemRange[self.c_dtype, cute.cosize(self.c_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        self.kernel(
            tiled_mma, tiled_mma_bkeep, tiled_mma_breuse, tiled_mma_sfb,
            tma_atom_a, tma_tensor_a, tma_atom_b, tma_tensor_b,
            tma_atom_sfa, tma_tensor_sfa, tma_atom_sfb, tma_tensor_sfb,
            c_tensor,
            tma_atom_gu, tma_tensor_gu, tma_atom_sw, tma_tensor_sw,
            gu_tensor, sw_tensor,
            rht, amax, amax_a, amax_b,
            self.cluster_layout_vmnk, self.cluster_layout_sfb_vmnk,
            self.a_smem_layout_staged, self.b_smem_layout_staged,
            self.sfa_smem_layout_staged, self.sfb_smem_layout_staged,
            self.tCtSFA_layout, self.tCtSFB_layout, self.c_smem_layout_staged,
            self.epi_tile, self.tile_sched_params,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
            min_blocks_per_mp=1,
        )
        return

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_bkeep: Optional[cute.TiledMma],
        tiled_mma_breuse: Optional[cute.TiledMma],
        tiled_mma_sfb: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA_mkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,
        mC_mnl: cute.Tensor,
        tma_atom_gu: cute.CopyAtom,
        mGU_mnl: cute.Tensor,
        tma_atom_sw: cute.CopyAtom,
        mSW_mnl: cute.Tensor,
        mGU_gmem: cute.Tensor,
        mSW_gmem: cute.Tensor,
        mRht: cute.Tensor,
        mAmax: cute.Tensor,
        mAmaxA: cute.Tensor,
        mAmaxB: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        tCtSFA_layout: cute.Layout,
        tCtSFB_layout: cute.Layout,
        c_smem_layout_staged: cute.ComposedLayout,
        epi_tile: cute.Tile,
        tile_sched_params,
    ):
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_sfa)
            cpasync.prefetch_descriptor(tma_atom_sfb)
            cpasync.prefetch_descriptor(tma_atom_gu)
            cpasync.prefetch_descriptor(tma_atom_sw)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2
        cluster_shape_mn = self.cluster_shape_mn

        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(cta_rank_in_cluster)
        tidx, _, _ = cute.arch.thread_idx()

        smem = cutlass.memory.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Warp)
        ab_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            enable_multicast_signaling=True,
            defer_sync=True,
        )
        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.epilog_warp_id) * (2 if use_2cta_instrs else 1)
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_acc_consumer_threads)
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        clc_pipeline = None
        clc_response_ptr = None
        clc_consumer_state = None
        if cutlass.const_expr(self.use_clc_dynamic_scheduler):
            clc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
            num_clc_consumer_threads = self.threads_per_warp * (
                1 + cute.size(self.cluster_shape_mn) * (2 + len(self.epilog_warp_id))
            )
            clc_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_clc_consumer_threads)
            clc_pipeline = pipeline.PipelineClcFetchAsync.create(
                barrier_storage=storage.clc_mbar_ptr.data_ptr(),
                num_stages=self.num_clc_stage,
                producer_group=clc_pipeline_producer_group,
                consumer_group=clc_pipeline_consumer_group,
                tx_count=self.num_clc_response_bytes,
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            )
            clc_response_ptr = storage.clc_response.data_ptr()
            clc_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_clc_stage)
        clc_resp_words = self.num_clc_response_bytes // 4
        tmem = cutlass.memory.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.epilog_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
            arch=self.arch,
        )
        pipeline_init_arrive(cluster_shape_mn=cluster_shape_mn, is_relaxed=True)

        sC = storage.sC.get_tensor(c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner)
        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)

        a_full_mcast_mask = None
        b_full_mcast_mask = None
        sfa_full_mcast_mask = None
        sfb_full_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2)
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1)
            sfa_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2)
            sfb_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_sfb_vmnk, block_in_cluster_coord_sfb_vmnk, mcast_mode=1)

        gA_mkl = cute.local_tile(mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None))
        gB_nkl = cute.local_tile(mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None))
        gSFA_mkl = cute.local_tile(mSFA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None))
        gSFB_nkl = cute.local_tile(mSFB_nkl, cute.slice_(self.mma_tiler_sfb, (0, None, None)), (None, None, None))
        gC_mnl = cute.local_tile(mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None))
        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgB = thr_mma.partition_B(gB_nkl)
        tCgSFA = thr_mma.partition_A(gSFA_mkl)
        tCgSFB = thr_mma_sfb.partition_B(gSFB_nkl)
        tCgC = thr_mma.partition_C(gC_mnl)

        a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a, block_in_cluster_coord_vmnk[2], a_cta_layout,
            cute.group_modes(sA, 0, 3), cute.group_modes(tCgA, 0, 3),
        )
        b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b, block_in_cluster_coord_vmnk[1], b_cta_layout,
            cute.group_modes(sB, 0, 3), cute.group_modes(tCgB, 0, 3),
        )
        sfa_cta_layout = a_cta_layout
        tAsSFA, tAgSFA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfa, block_in_cluster_coord_vmnk[2], sfa_cta_layout,
            cute.group_modes(sSFA, 0, 3), cute.group_modes(tCgSFA, 0, 3),
        )
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)
        sfb_cta_layout = cute.make_layout(cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape)
        tBsSFB, tBgSFB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfb, block_in_cluster_coord_sfb_vmnk[1], sfb_cta_layout,
            cute.group_modes(sSFB, 0, 3), cute.group_modes(tCgSFB, 0, 3),
        )
        tBsSFB = cute.filter_zeros(tBsSFB)
        tBgSFB = cute.filter_zeros(tBgSFB)

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))

        pipeline_init_wait(cluster_shape_mn=cluster_shape_mn)

        if cutlass.const_expr(self.use_clc_dynamic_scheduler):
            tile_sched = utils.ClcDynamicPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim(), clc_response_ptr
            )
        else:
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
        work_tile = tile_sched.initial_work_tile_info()

        if warp_idx == self.tma_warp_id:
            ab_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_ab_stage)
            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )
                tAgA_slice = tAgA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
                tBgB_slice = tBgB[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]
                tAgSFA_slice = tAgSFA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
                slice_n = mma_tile_coord_mnl[1]
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                    slice_n = mma_tile_coord_mnl[1] // 2
                tBgSFB_slice = tBgSFB[(None, slice_n, None, mma_tile_coord_mnl[2])]

                if self.prefetch_enabled:
                    for pf_k_tile in cutlass.range(0, min(self.prefetch_dist, k_tile_cnt), unroll=1):
                        cute.prefetch(tma_atom_a, tAgA_slice[(None, pf_k_tile)])
                        cute.prefetch(tma_atom_b, tBgB_slice[(None, pf_k_tile)])
                        cute.prefetch(tma_atom_sfa, tAgSFA_slice[(None, pf_k_tile)])
                        cute.prefetch(tma_atom_sfb, tBgSFB_slice[(None, pf_k_tile)])

                ab_producer_state.reset_count()
                peek_ab_empty_status = cutlass.Boolean(1)
                if ab_producer_state.count < k_tile_cnt:
                    peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    ab_pipeline.producer_acquire(ab_producer_state, peek_ab_empty_status)
                    cute.copy(
                        tma_atom_a, tAgA_slice[(None, ab_producer_state.count)], tAsA[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state), mcast_mask=a_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b, tBgB_slice[(None, ab_producer_state.count)], tBsB[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state), mcast_mask=b_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_sfa, tAgSFA_slice[(None, ab_producer_state.count)], tAsSFA[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state), mcast_mask=sfa_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_sfb, tBgSFB_slice[(None, ab_producer_state.count)], tBsSFB[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state), mcast_mask=sfb_full_mcast_mask,
                    )
                    if self.prefetch_enabled:
                        if k_tile < k_tile_cnt - self.prefetch_dist:
                            future_k_tile = ab_producer_state.count + self.prefetch_dist
                            cute.prefetch(tma_atom_a, tAgA_slice[(None, future_k_tile)])
                            cute.prefetch(tma_atom_b, tBgB_slice[(None, future_k_tile)])
                            cute.prefetch(tma_atom_sfa, tAgSFA_slice[(None, future_k_tile)])
                            cute.prefetch(tma_atom_sfb, tBgSFB_slice[(None, future_k_tile)])
                    ab_producer_state.advance()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if ab_producer_state.count < k_tile_cnt:
                        peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)

                if cutlass.const_expr(self.use_clc_dynamic_scheduler):
                    clc_pipeline.consumer_wait(clc_consumer_state)
                    work_tile = tile_sched.work_tile_info_from_clc_response(
                        (clc_response_ptr + clc_consumer_state.index * clc_resp_words).align(16)
                    )
                    clc_pipeline.consumer_release(clc_consumer_state)
                    clc_consumer_state.advance()
                else:
                    tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()
            ab_pipeline.producer_tail(ab_producer_state)

        if warp_idx == self.mma_warp_id:
            if cutlass.const_expr(self.use_clc_dynamic_scheduler):
                tile_sched = utils.ClcDynamicPersistentTileScheduler.create(
                    tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim(), clc_response_ptr
                )
            tmem.wait_for_alloc()
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)
            sfa_tmem_ptr = cute.recast_ptr(acc_tmem_ptr + self.num_accumulator_tmem_cols, dtype=self.sf_dtype)
            tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)
            sfb_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols, dtype=self.sf_dtype
            )
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)
            sfa_s2t_bundle = self._mainloop_s2t_copy_and_partition(sSFA, tCtSFA)
            sfb_s2t_bundle = self._mainloop_s2t_copy_and_partition(sSFB, tCtSFB)

            ab_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_ab_stage)
            acc_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_acc_stage)

            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )
                acc_stage_index = acc_producer_state.index
                tCtAcc = tCtAcc_base[(None, None, None, acc_stage_index)]
                ab_consumer_state.reset_count()
                peek_ab_full_status = cutlass.Boolean(1)
                if ab_consumer_state.count < k_tile_cnt and is_leader_cta:
                    peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)
                if is_leader_cta:
                    acc_pipeline.producer_acquire(acc_producer_state)

                tCtSFB_mma = tCtSFB
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] in {64, 192}):
                    offset = cutlass.Int32((mma_tile_coord_mnl[1] % 2) * 2)
                    shifted_ptr = cute.recast_ptr(
                        acc_tmem_ptr + self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols + offset,
                        dtype=self.sf_dtype,
                    )
                    tCtSFB_mma = cute.make_tensor(shifted_ptr, tCtSFB_layout)

                for k_tile in range(k_tile_cnt):
                    if is_leader_cta:
                        ab_pipeline.consumer_wait(ab_consumer_state, peek_ab_full_status)
                        if cutlass.const_expr(not self._is_interleaved_utccp()):
                            self._mainloop_s2t_copies(ab_consumer_state.index, sfa_s2t_bundle, sfb_s2t_bundle)
                        num_kblocks = cute.size(tCrA, mode=[2])
                        for k_block in cutlass.range(num_kblocks, unroll_full=True):
                            if cutlass.const_expr(
                                self.enable_breuse
                                and cute.size(tCtAcc.layout, mode=[1]) == 2
                                and cute.size(tCtAcc.layout, mode=[2]) == 1
                            ):
                                tCtAcc_bkeep = tCtAcc[(None, 0, 0)]
                                tCtAcc_breuse = tCtAcc[(None, 1, 0)]
                                a_kblk_crd_keep = (None, 0, k_block, ab_consumer_state.index)
                                a_kblk_crd_reuse = (None, 1, k_block, ab_consumer_state.index)
                                b_kblk_crd = (None, 0, k_block, ab_consumer_state.index)
                                sfa_kblk_crd_keep = (None, 0, k_block)
                                sfa_kblk_crd_reuse = (None, 1, k_block)
                                sfb_kblk_crd = (None, 0, k_block)
                                if cutlass.const_expr(self._is_interleaved_utccp()):
                                    self._mainloop_s2t_interleaved_copies(
                                        k_block, ab_consumer_state.index, sfa_s2t_bundle, sfb_s2t_bundle
                                    )
                                tiled_mma_bkeep.set(tcgen05.Field.ACCUMULATE, k_tile != 0 or k_block != 0)
                                cute.gemm(
                                    tiled_mma_bkeep, tCtAcc_bkeep,
                                    [tCrA[a_kblk_crd_keep], tCtSFA[sfa_kblk_crd_keep]],
                                    [tCrB[b_kblk_crd], tCtSFB_mma[sfb_kblk_crd]],
                                    tCtAcc_bkeep,
                                )
                                tiled_mma_breuse.set(tcgen05.Field.ACCUMULATE, k_tile != 0 or k_block != 0)
                                cute.gemm(
                                    tiled_mma_breuse, tCtAcc_breuse,
                                    [tCrA[a_kblk_crd_reuse], tCtSFA[sfa_kblk_crd_reuse]],
                                    [tCrB[b_kblk_crd], tCtSFB_mma[sfb_kblk_crd]],
                                    tCtAcc_breuse,
                                )
                            else:
                                kblk_crd = (None, None, k_block, ab_consumer_state.index)
                                sf_kblk_crd = (None, None, k_block)
                                tiled_mma.set(tcgen05.Field.ACCUMULATE, k_tile != 0 or k_block != 0)
                                cute.gemm(
                                    tiled_mma, tCtAcc,
                                    [tCrA[kblk_crd], tCtSFA[sf_kblk_crd]],
                                    [tCrB[kblk_crd], tCtSFB_mma[sf_kblk_crd]],
                                    tCtAcc,
                                )
                        ab_pipeline.consumer_release(ab_consumer_state)
                    ab_consumer_state.advance()
                    peek_ab_full_status = cutlass.Boolean(1)
                    if ab_consumer_state.count < k_tile_cnt:
                        if is_leader_cta:
                            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)

                if is_leader_cta:
                    acc_pipeline.producer_commit(acc_producer_state)
                acc_producer_state.advance()
                if cutlass.const_expr(self.use_clc_dynamic_scheduler):
                    clc_pipeline.consumer_wait(clc_consumer_state)
                    work_tile = tile_sched.work_tile_info_from_clc_response(
                        (clc_response_ptr + clc_consumer_state.index * clc_resp_words).align(16)
                    )
                    clc_pipeline.consumer_release(clc_consumer_state)
                    clc_consumer_state.advance()
                else:
                    tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()
            acc_pipeline.producer_tail(acc_producer_state)

        if warp_idx < self.mma_warp_id:
            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            n_groups = self.epi_groups
            epi_group = warp_idx // 4
            quad_warp = warp_idx % 4
            tidx_q = tidx % 128
            is_leader_warp = quad_warp == 0
            c_stages_g = self.num_c_stage // n_groups

            acc_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_acc_stage)
            c_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 32 * 4)
            c_pipeline = pipeline.PipelineTmaStore.create(num_stages=2, producer_group=c_producer_group)

            alpha = (mAmaxA[0] * mAmaxB[0]) * cutlass.Float32(_NVFP4_FACTOR_INV)
            amax_row = cutlass.Float32(0.0)
            amax_col = cutlass.Float32(0.0)

            tCgC_t = transform_partitioned_tensor_layout(tCgC)
            tCtAcc_t = transform_partitioned_tensor_layout(tCtAcc_base)
            tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc = epilogue_tmem_copy_and_partition(
                self, tidx_q, tCtAcc_t, tCgC_t, epi_tile, use_2cta_instrs
            )
            tTR_rAcc2 = cute.make_rmem_tensor(tTR_rAcc.shape, self.acc_dtype)
            tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
            tTR_rC2 = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
            tTR_rC3 = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
            tiled_copy_r2s, tRS_rC, tRS_sC = epilogue_smem_copy_and_partition(
                self, tiled_copy_t2r, tTR_rC, tidx_q, sC
            )
            tRS_rC2 = tiled_copy_r2s.retile(tTR_rC2)
            tRS_rC3 = tiled_copy_r2s.retile(tTR_rC3)
            tTR_rD = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
            tTR_rD2 = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
            tTR_rD3 = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
            tRS_rD = tiled_copy_r2s.retile(tTR_rD)
            tRS_rD2 = tiled_copy_r2s.retile(tTR_rD2)
            tRS_rD3 = tiled_copy_r2s.retile(tTR_rD3)
            thr_copy_t2r = tiled_copy_t2r.get_slice(tidx_q)
            simt_store = cute.make_copy_atom(
                cute.nvgpu.CopyR2GOp(), self.c_dtype, num_bits_per_copy=self.simt_store_bits,
                l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.NO_ALLOCATE,
            )

            gGU_epi = cute.flat_divide(mGU_mnl, epi_tile)
            gSW_epi = cute.flat_divide(mSW_mnl, epi_tile)
            gGU_epi_g = cute.flat_divide(mGU_gmem, epi_tile)
            gSW_epi_g = cute.flat_divide(mSW_gmem, epi_tile)
            bSG_sC, bSG_gGU = cpasync.tma_partition(
                tma_atom_gu, 0, cute.make_layout(1), cute.group_modes(sC, 0, 2), cute.group_modes(gGU_epi, 0, 2)
            )
            _, bSG_gSW = cpasync.tma_partition(
                tma_atom_sw, 0, cute.make_layout(1), cute.group_modes(sC, 0, 2), cute.group_modes(gSW_epi, 0, 2)
            )
            all_barrier = pipeline.NamedBarrier(
                barrier_id=self.epilog_sync_bar_id, num_threads=32 * len(self.epilogue_warp_id)
            )
            if cutlass.const_expr(n_groups == 1):
                group_barrier = all_barrier
            else:
                group_barrier = pipeline.NamedBarrier(barrier_id=cutlass.Int32(3) + epi_group, num_threads=128)

            lane = tidx % 32
            mma_op_rht = warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16))
            tiled_mma_rht = cute.make_tiled_mma(mma_op_rht, (1, 1, 1), (_RHT_DIM, _RHT_DIM, _RHT_DIM))
            thr_mma_rht = tiled_mma_rht.get_slice(lane)
            tCgB_rht = thr_mma_rht.partition_B(mRht)
            tCrB_rht = tiled_mma_rht.make_fragment_B(tCgB_rht)
            cute.autovec_copy(tCgB_rht, tCrB_rht)
            ldsm_atom = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), cutlass.BFloat16
            )
            tiled_copy_rht_a = cute.make_tiled_copy_A(ldsm_atom, tiled_mma_rht)
            thr_copy_rht_a = tiled_copy_rht_a.get_slice(lane)

            nsh = self.half_n // self.epi_tile_n
            pairs_g = nsh // n_groups
            sw_cols = self.ffn // self.epi_tile_n

            epi_tiles_executed = cutlass.Int32(0)
            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                row_blk = cur_tile_coord[0]
                tile_n = cur_tile_coord[1]
                if cutlass.const_expr(self.use_clc_dynamic_scheduler):
                    clc_pipeline.consumer_wait(clc_consumer_state)
                    work_tile = tile_sched.work_tile_info_from_clc_response(
                        (clc_response_ptr + clc_consumer_state.index * clc_resp_words).align(16)
                    )
                    clc_pipeline.consumer_release(clc_consumer_state)
                    clc_consumer_state.advance()
                else:
                    tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()
                epi_tiles_executed += 1

                tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_consumer_state.index)]
                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))

                acc_pipeline.consumer_wait(acc_consumer_state)

                if cutlass.const_expr(pairs_g == 2 and self.store_mode == "tma" and self.pairs_in_flight >= 3):
                    j0 = epi_group
                    j1 = epi_group + n_groups
                    cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, j0)], tTR_rAcc)
                    cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, nsh + j0)], tTR_rAcc2)
                    amax_row = self._epi_swiglu_math(tTR_rAcc, tTR_rAcc2, tTR_rC, tTR_rC2, tTR_rC3, alpha, amax_row)
                    cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, j1)], tTR_rAcc)
                    cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, nsh + j1)], tTR_rAcc2)
                    amax_row = self._epi_swiglu_math(tTR_rAcc, tTR_rAcc2, tTR_rD, tTR_rD2, tTR_rD3, alpha, amax_row)
                    all_barrier.arrive_and_wait()
                    with cute.arch.elect_one():
                        acc_pipeline.consumer_release(acc_consumer_state)
                    acc_consumer_state.advance()
                    pair_idx = (epi_tiles_executed - 1) * pairs_g
                    col_g = tile_n * nsh + j0
                    col_u = sw_cols + tile_n * nsh + j0
                    amax_col = self._epi_store_pair(
                        is_leader_warp, quad_warp, tiled_copy_r2s, tRS_rC, tRS_rC2, tRS_rC3, tRS_sC, sC, bSG_sC,
                        bSG_gGU[(None, row_blk, col_g, 0)], bSG_gGU[(None, row_blk, col_u, 0)],
                        bSG_gSW[(None, row_blk, col_g, 0)], tma_atom_gu, tma_atom_sw, epi_group, c_stages_g, pair_idx,
                        group_barrier, c_pipeline, tiled_mma_rht, thr_mma_rht, thr_copy_rht_a, ldsm_atom, tCrB_rht,
                        amax_col,
                    )
                    col_g = tile_n * nsh + j1
                    col_u = sw_cols + tile_n * nsh + j1
                    amax_col = self._epi_store_pair(
                        is_leader_warp, quad_warp, tiled_copy_r2s, tRS_rD, tRS_rD2, tRS_rD3, tRS_sC, sC, bSG_sC,
                        bSG_gGU[(None, row_blk, col_g, 0)], bSG_gGU[(None, row_blk, col_u, 0)],
                        bSG_gSW[(None, row_blk, col_g, 0)], tma_atom_gu, tma_atom_sw, epi_group, c_stages_g, pair_idx + 1,
                        group_barrier, c_pipeline, tiled_mma_rht, thr_mma_rht, thr_copy_rht_a, ldsm_atom, tCrB_rht,
                        amax_col,
                    )
                else:
                    for jj in cutlass.range_constexpr(pairs_g):
                        j = epi_group + n_groups * jj
                        cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, j)], tTR_rAcc)
                        cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, nsh + j)], tTR_rAcc2)
                        amax_row = self._epi_swiglu_math(tTR_rAcc, tTR_rAcc2, tTR_rC, tTR_rC2, tTR_rC3, alpha, amax_row)

                        col_g = tile_n * nsh + j
                        col_u = sw_cols + tile_n * nsh + j
                        bSG_gG = bSG_gGU[(None, row_blk, col_g, 0)]
                        bSG_gU = bSG_gGU[(None, row_blk, col_u, 0)]
                        bSG_gS = bSG_gSW[(None, row_blk, col_g, 0)]
                        pair_idx = (epi_tiles_executed - 1) * pairs_g + jj
                        if cutlass.const_expr(self.store_mode == "tma"):
                            amax_col = self._epi_store_pair(
                                is_leader_warp, quad_warp, tiled_copy_r2s, tRS_rC, tRS_rC2, tRS_rC3, tRS_sC, sC, bSG_sC,
                                bSG_gG, bSG_gU, bSG_gS, tma_atom_gu, tma_atom_sw, epi_group, c_stages_g, pair_idx,
                                group_barrier, c_pipeline, tiled_mma_rht, thr_mma_rht, thr_copy_rht_a, ldsm_atom, tCrB_rht,
                                amax_col,
                            )
                        else:
                            tTR_gG = thr_copy_t2r.partition_D(gGU_epi_g[(None, None, row_blk, col_g, 0)])
                            tTR_gU = thr_copy_t2r.partition_D(gGU_epi_g[(None, None, row_blk, col_u, 0)])
                            tTR_gS = thr_copy_t2r.partition_D(gSW_epi_g[(None, None, row_blk, col_g, 0)])
                            amax_col = self._epi_store_pair_simt(
                                quad_warp, simt_store, tTR_rC, tTR_rC2, tTR_rC3, tTR_gG, tTR_gU, tTR_gS,
                                tiled_copy_r2s, tRS_rC3, tRS_sC, sC, epi_group, c_stages_g, pair_idx, group_barrier,
                                tiled_mma_rht, thr_mma_rht, thr_copy_rht_a, ldsm_atom, tCrB_rht, amax_col,
                            )
                    all_barrier.arrive_and_wait()
                    with cute.arch.elect_one():
                        acc_pipeline.consumer_release(acc_consumer_state)
                    acc_consumer_state.advance()

            if cutlass.const_expr(self.store_mode == "tma"):
                c_pipeline.producer_tail()

            amax_row = cute.arch.warp_reduction(amax_row, cute.arch.fmax)
            amax_col = cute.arch.warp_reduction(amax_col, cute.arch.fmax)
            if tidx % 32 == 0:
                cute.arch.atomic_fmax(mAmax.iterator + 0, amax_row, sign_bit=False)
                cute.arch.atomic_fmax(mAmax.iterator + 1, amax_col, sign_bit=False)

            tmem.relinquish_alloc_permit()
            tmem.free(acc_tmem_ptr)

        if cutlass.const_expr(self.use_clc_dynamic_scheduler):
            is_first_cta_in_cluster = cta_rank_in_cluster == 0
            if warp_idx == self.sched_warp_id and is_first_cta_in_cluster:
                clc_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.ProducerConsumer, self.num_clc_stage
                )
                while work_tile.is_valid_tile:
                    clc_pipeline.producer_acquire(clc_producer_state)
                    clc_mbar = clc_pipeline.producer_get_barrier(clc_producer_state)
                    clc_slot = (clc_response_ptr + clc_producer_state.index * clc_resp_words).align(16)
                    with cute.arch.elect_one():
                        cute.arch.issue_clc_query(clc_mbar, clc_slot)
                    clc_producer_state.advance()
                    clc_pipeline.consumer_wait(clc_consumer_state)
                    work_tile = tile_sched.work_tile_info_from_clc_response(
                        (clc_response_ptr + clc_consumer_state.index * clc_resp_words).align(16)
                    )
                    clc_pipeline.consumer_release(clc_consumer_state)
                    clc_consumer_state.advance()
                clc_pipeline.producer_tail(clc_producer_state)

    @cute.jit
    def _epi_swiglu_math(self, tTR_rAcc, tTR_rAcc2, tTR_rC, tTR_rC2, tTR_rC3, alpha, amax_row):
        n = cute.size(tTR_rAcc)
        ns = self.split_cols
        w = n // ns
        for h in cutlass.range_constexpr(ns):
            va = cute.make_tensor(tTR_rAcc.iterator + h * w, cute.make_layout((w,))).load()
            vb = cute.make_tensor(tTR_rAcc2.iterator + h * w, cute.make_layout((w,))).load()
            g = (va * alpha).to(cutlass.BFloat16)
            u = (vb * alpha).to(cutlass.BFloat16)
            cute.make_tensor(tTR_rC.iterator + h * w, cute.make_layout((w,))).store(g)
            cute.make_tensor(tTR_rC2.iterator + h * w, cute.make_layout((w,))).store(u)
            if cutlass.const_expr(self.with_swiglu and self.swiglu_math == "te_fast"):
                gf = g.to(cutlass.Float32)
                uf = u.to(cutlass.Float32)
                r_g = cute.make_rmem_tensor((w,), cutlass.Float32)
                r_u = cute.make_rmem_tensor((w,), cutlass.Float32)
                r_s = cute.make_rmem_tensor((w,), cutlass.Float32)
                r_g.store(gf)
                r_u.store(uf)
                for i in cutlass.range_constexpr(w):
                    x_i = r_g[i]
                    e_i = cute.math.exp2(x_i * cutlass.Float32(-1.4426950216293334961), approx=True, ftz=True)
                    s_i = cute.math.rcp(cutlass.Float32(1.0) + e_i, approx=True, ftz=True)
                    r_s[i] = (x_i * s_i) * r_u[i]
                sw = r_s.load().to(cutlass.BFloat16)
            elif cutlass.const_expr(self.with_swiglu):
                gf = g.to(cutlass.Float32)
                uf = u.to(cutlass.Float32)
                one = cute.full_like(gf, 1.0)
                if cutlass.const_expr(self.swiglu_math in ("fastexp", "fast")):
                    e = cute.math.exp(gf * cutlass.Float32(-1.0), fastmath=True)
                else:
                    e = cute.math.exp(gf * cutlass.Float32(-1.0))
                den = one + e
                if cutlass.const_expr(self.swiglu_math in ("rcp", "fast")):
                    sig = cute.math.rcp(den, fastmath=True)
                else:
                    sig = one / den
                sw = ((gf * sig) * uf).to(cutlass.BFloat16)
            else:
                sw = g
            cute.make_tensor(tTR_rC3.iterator + h * w, cute.make_layout((w,))).store(sw)
            ma = abs(sw.to(cutlass.Float32)).reduce(cute.ReductionOp.MAX, cutlass.Float32(0.0), 0)
            amax_row = cute.arch.fmax(amax_row, cutlass.Float32(ma))
        return amax_row

    @cute.jit
    def _epi_store_pair(
        self, is_leader_warp, quad_warp, tiled_copy_r2s, tRS_rC, tRS_rC2, tRS_rC3, tRS_sC, sC, bSG_sC,
        bSG_gG, bSG_gU, bSG_gS, tma_atom_gu, tma_atom_sw, epi_group, c_stages_g, pair_idx,
        group_barrier, c_pipeline, tiled_mma_rht, thr_mma_rht, thr_copy_rht_a, ldsm_atom, tCrB_rht, amax_col,
    ):
        sbase = epi_group * c_stages_g + (pair_idx % self.pairs_in_flight) * 3
        cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, sbase)])
        cute.copy(tiled_copy_r2s, tRS_rC2, tRS_sC[(None, None, None, sbase + 1)])
        cute.copy(tiled_copy_r2s, tRS_rC3, tRS_sC[(None, None, None, sbase + 2)])
        cute.arch.fence_proxy("async.shared", space="cta")
        group_barrier.arrive_and_wait()
        if is_leader_warp:
            cute.copy(tma_atom_gu, bSG_sC[(None, sbase)], bSG_gG)
            cute.copy(tma_atom_gu, bSG_sC[(None, sbase + 1)], bSG_gU)
            if cutlass.const_expr(not self.skip_swiglu_store):
                cute.copy(tma_atom_sw, bSG_sC[(None, sbase + 2)], bSG_gS)
            c_pipeline.producer_commit()
        if cutlass.const_expr(self.with_rht_amax):
            amax_col = self._rht_amax_stage(
                sC[(None, None, sbase + 2)], quad_warp, tiled_mma_rht, thr_mma_rht, thr_copy_rht_a,
                ldsm_atom, tCrB_rht, amax_col,
            )
        if is_leader_warp:
            c_pipeline.producer_acquire()
        if cutlass.const_expr(self.pairs_in_flight == 2):
            group_barrier.arrive_and_wait()
        return amax_col

    @cute.jit
    def _epi_store_pair_simt(
        self, quad_warp, simt_store, tTR_rC, tTR_rC2, tTR_rC3, tTR_gG, tTR_gU, tTR_gS,
        tiled_copy_r2s, tRS_rC3, tRS_sC, sC, epi_group, c_stages_g, pair_idx, group_barrier,
        tiled_mma_rht, thr_mma_rht, thr_copy_rht_a, ldsm_atom, tCrB_rht, amax_col,
    ):
        if cutlass.const_expr(self.with_rht_amax):
            stage = epi_group * c_stages_g + pair_idx % 2
            cute.copy(tiled_copy_r2s, tRS_rC3, tRS_sC[(None, None, None, stage)])
        cute.copy(simt_store, tTR_rC, tTR_gG)
        cute.copy(simt_store, tTR_rC2, tTR_gU)
        cute.copy(simt_store, tTR_rC3, tTR_gS)
        if cutlass.const_expr(self.with_rht_amax):
            group_barrier.arrive_and_wait()
            amax_col = self._rht_amax_stage(
                sC[(None, None, stage)], quad_warp, tiled_mma_rht, thr_mma_rht, thr_copy_rht_a,
                ldsm_atom, tCrB_rht, amax_col,
            )
        return amax_col

    @cute.jit
    def _rht_amax_stage(
        self, sX, epi_warp, tiled_mma_rht, thr_mma_rht, thr_copy_rht_a, ldsm_atom, tCrB_rht, amax_col,
    ):
        lay = sX.layout
        if cutlass.const_expr(isinstance(lay, cute.ComposedLayout)):
            lay_t = cute.make_composed_layout(lay.inner, lay.offset, cute.select(lay.outer, mode=[1, 0]))
        else:
            lay_t = cute.select(lay, mode=[1, 0])
        sXT = cute.make_tensor(sX.iterator, lay_t)
        n_cb = self.epi_tile_n // _RHT_DIM
        n_rb = self.cta_tile_shape_mnk[0] // _RHT_DIM
        n_blocks = n_cb * n_rb
        n_warps = 4
        per_warp = n_blocks // n_warps
        for i in cutlass.range_constexpr(per_warp):
            b = epi_warp * per_warp + i
            cb = b % n_cb
            rb = b // n_cb
            blk = cute.local_tile(sXT, (_RHT_DIM, _RHT_DIM), (cb, rb))
            tCsA = thr_mma_rht.partition_A(blk)
            tCrA = tiled_mma_rht.make_fragment_A(tCsA)
            if cutlass.const_expr(self.rht_mode == "ldsm"):
                tCsA_c = thr_copy_rht_a.partition_S(blk)
                tCrA_c = thr_copy_rht_a.retile(tCrA)
                cute.copy(ldsm_atom, tCsA_c, tCrA_c)
            else:
                cute.autovec_copy(tCsA, tCrA)
            tCrC = tiled_mma_rht.make_fragment_C(thr_mma_rht.partition_shape_C((_RHT_DIM, _RHT_DIM)))
            tCrC.fill(0.0)
            cute.gemm(tiled_mma_rht, tCrC, tCrA, tCrB_rht, tCrC)
            v = tCrC.load().to(cutlass.BFloat16).to(cutlass.Float32)
            mb = abs(v).reduce(cute.ReductionOp.MAX, cutlass.Float32(0.0), 0)
            amax_col = cute.arch.fmax(amax_col, cutlass.Float32(mb))
        return amax_col


class FusedFc1Gemm:

    def __init__(self, ffn: int, hidden: int, device: torch.device, max_active_clusters: Optional[int] = None,
                 **kernel_kwargs):
        self.ffn, self.hidden = ffn, hidden
        self.kernel = Fc1SwigluAmaxGemmKernel(ffn, **kernel_kwargs)
        cluster = self.kernel.cluster_shape_mn
        if max_active_clusters is None:
            hw = utils.HardwareInfo(device.index if device.index is not None else torch.cuda.current_device())
            max_active_clusters = hw.get_max_active_clusters(cluster[0] * cluster[1])
        self.max_active_clusters = max_active_clusters
        self.compiled = None
        self.b_interleave = self.kernel.b_interleave

    @staticmethod
    def _ptr(dtype, t: torch.Tensor, align: int):
        return make_ptr(dtype, t.data_ptr(), cutlass.AddressSpace.gmem, assumed_align=align)

    @staticmethod
    def _vec(t: torch.Tensor):
        return from_dlpack(t, assumed_align=16).mark_layout_dynamic()

    def permute_weight(self, w_data: torch.Tensor, w_sf: torch.Tensor):
        nb = self.kernel.n_blocks
        hn = self.kernel.half_n
        n, kh = w_data.shape
        wd = w_data.view(2, nb, hn, kh).transpose(0, 1).contiguous().view(n, kh)
        sf = w_sf.view(2, nb, -1).transpose(0, 1).contiguous().view_as(w_sf)
        return wd, sf

    def __call__(self, x_data, w_data, x_sf, w_sf, x_amax, w_amax, rht, gate_up, swiglu, amax, M, K):
        if self.b_interleave == "perm":
            w_data, w_sf = self.permute_weight(w_data, w_sf)
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        args = (
            self._ptr(cutlass.Float4E2M1FN, x_data, 16),
            self._ptr(cutlass.Float4E2M1FN, w_data, 16),
            self._ptr(cutlass.Float8E4M3FN, x_sf, 32),
            self._ptr(cutlass.Float8E4M3FN, w_sf, 32),
            self._ptr(cutlass.BFloat16, gate_up, self.kernel.simt_store_bits // 8),
            self._ptr(cutlass.BFloat16, swiglu, self.kernel.simt_store_bits // 8),
            from_dlpack(rht, assumed_align=16),
            self._vec(amax),
            self._vec(x_amax),
            self._vec(w_amax),
        )
        if self.compiled is None:
            self.compiled = cute.compile(
                self.kernel, *args,
                (cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(0)),
                self.max_active_clusters, stream,
            )
        self.compiled(*args, (M, 2 * self.ffn, K, 1), stream)


def _zeros_amax(n: int, device) -> torch.Tensor:
    return torch.zeros(n, dtype=torch.float32, device=device)


def fc1_swiglu_amax_gemm(x_q, w_q, rht: torch.Tensor, gemm: FusedFc1Gemm):
    K = x_q._rowwise_data.shape[-1] * 2
    M = x_q._rowwise_data.numel() // x_q._rowwise_data.shape[-1]
    dev = x_q._rowwise_data.device
    gate_up = torch.empty(M, 2 * gemm.ffn, dtype=torch.bfloat16, device=dev)
    swiglu = torch.empty(M, gemm.ffn, dtype=torch.bfloat16, device=dev)
    amax = _zeros_amax(2, dev)
    gemm(
        x_q._rowwise_data, w_q._rowwise_data, x_q._rowwise_scale_inv, w_q._rowwise_scale_inv,
        x_q._amax_rowwise, w_q._amax_rowwise, rht, gate_up, swiglu, amax, M, K,
    )
    return gate_up, swiglu, amax


_DEBUG = int(os.environ.get("OPT_FC1_SWIGLU_DEBUG", "0"))
KERNEL_CONFIG = dict(
    mma_inst_shape=(256, 256, 128),
    mma_tiler=(256, 256, 256),
    cluster_shape_mn=(2, 1),
    swizzle_size=1,
    raster_order="m",
    prefetch_dist=0,
    b_interleave="tma",
    rht_mode="ldsm",
    epi_warps=8,
    swiglu_math="te_fast",
    scheduler="clc",
    num_clc_stage=2,
)


def _dbg(msg: str) -> None:
    rank = os.environ.get("RANK", os.environ.get("SLURM_PROCID", "?"))
    sys.stderr.write(f"[opt fc1-swiglu r{rank}] {msg}\n")
    sys.stderr.flush()


class _State:
    enabled = True
    installed = False
    ffn = None
    hidden = None
    gemm = None
    rht = None
    sign_mask = None
    record = None
    orig_general_gemm = None
    orig_swiglu_fwd = None
    n_fused = 0
    n_swizzled = 0
    n_fused_cast = 0
    n_fallback_cast = 0
    debug_calls = 0
    reported = False


def _rht_for(device: torch.device):
    if _State.rht is None or _State.rht.device != device:
        from transformer_engine.pytorch.tensor.nvfp4_tensor import (
            get_random_sign_mask_for_rht, get_rht_matrix,
        )
        idx = device.index if device.index is not None else torch.cuda.current_device()
        _State.rht = get_rht_matrix(True, idx).t().contiguous()
        _State.sign_mask = get_random_sign_mask_for_rht(True, idx)
    return _State.rht


def _is_fc1_gemm(A, B, kwargs) -> bool:
    from transformer_engine.pytorch.tensor.storage.nvfp4_tensor_storage import NVFP4TensorStorage

    if not _State.enabled or _State.ffn is None:
        return False
    if not (isinstance(A, NVFP4TensorStorage) and isinstance(B, NVFP4TensorStorage)):
        return False
    if kwargs.get("layout", "TN") != "TN" or kwargs.get("grad", False) or kwargs.get("accumulate", False):
        return False
    if kwargs.get("quantization_params") is not None or kwargs.get("bias") is not None:
        return False
    if kwargs.get("ub") is not None or kwargs.get("extra_output") is not None or kwargs.get("out") is not None:
        return False
    if kwargs.get("alpha", 1.0) not in (None, 1.0) or kwargs.get("beta") not in (None, 0.0):
        return False
    if kwargs.get("out_dtype") not in (None, torch.bfloat16):
        return False
    if tuple(A.shape) != (2 * _State.ffn, _State.hidden) or B.shape[-1] != _State.hidden:
        return False
    if A._rowwise_data is None or B._rowwise_data is None or A._amax_rowwise is None or B._amax_rowwise is None:
        return False
    if not B._with_gemm_swizzled_scales:
        return False
    if getattr(A, "_row_scaled_nvfp4", False) or getattr(B, "_row_scaled_nvfp4", False):
        return False
    M = B._rowwise_data.numel() // B._rowwise_data.shape[-1]
    if M % 128 != 0:
        return False
    return True


def _general_gemm_patched(A, B, *args, **kwargs):
    if args or not _is_fc1_gemm(A, B, kwargs):
        return _State.orig_general_gemm(A, B, *args, **kwargs)
    dev = B._rowwise_data.device
    rht = _rht_for(dev)
    if _State.gemm is None:
        _State.gemm = FusedFc1Gemm(_State.ffn, _State.hidden, dev, **KERNEL_CONFIG)
    if not A._with_gemm_swizzled_scales:
        import transformer_engine_torch as tex
        tex.swizzle_scales_for_gemm_(A)
        _State.n_swizzled += 1
    gate_up, swiglu, amax = fc1_swiglu_amax_gemm(B, A, rht, _State.gemm)
    out_shape = tuple(B.shape[:-1]) + (2 * _State.ffn,)
    gate_up = gate_up.view(out_shape)
    _State.record = {"gu_ptr": gate_up.data_ptr(), "gu_numel": gate_up.numel(), "swiglu": swiglu, "amax": amax}
    _State.n_fused += 1
    if _DEBUG and _State.debug_calls < _DEBUG * 4 and not torch.cuda.is_current_stream_capturing():
        _State.debug_calls += 1
        try:
            import transformer_engine_torch as tex
            torch.cuda.synchronize()
            ref = _State.orig_general_gemm(A, B, *args, **kwargs)[0]
            ref_sw = tex.swiglu(ref, None)
            d_gu = (gate_up.view(torch.int16) != ref.view(torch.int16)).sum().item()
            d_sw = (swiglu.view(-1).view(torch.int16) != ref_sw.reshape(-1).view(torch.int16)).sum().item()
            _dbg(f"gemm call {_State.debug_calls}: M={out_shape[:-1]} gate_up mismatches={d_gu}/{gate_up.numel()} "
                 f"swiglu mismatches={d_sw}/{swiglu.numel()} amax={amax.tolist()} ref_sw_absmax={float(ref_sw.float().abs().max())} "
                 f"nan={int(torch.isnan(gate_up).sum())}")
        except Exception as exc:
            _dbg(f"  debug compare failed: {exc!r}")
    return gate_up, None, None, None


def _quantizer_ok(q) -> bool:
    from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer

    if not isinstance(q, NVFP4Quantizer):
        return False
    if not (q.with_rht and q.with_post_rht_amax):
        return False
    if getattr(q, "row_scaled_nvfp4", False):
        return False
    if q.rht_matrix_random_sign_mask_t != _State.sign_mask:
        return False
    return True


def _swiglu_fwd_patched(self, input_, quantizer):
    rec = _State.record
    _State.record = None
    if (
        rec is not None and _State.enabled and getattr(self, "glu_interleave_size", None) is None
        and input_.data_ptr() == rec["gu_ptr"] and input_.numel() == rec["gu_numel"] and _quantizer_ok(quantizer)
    ):
        import transformer_engine_torch as tex

        amax = rec["amax"]
        sw = rec["swiglu"].view(tuple(input_.shape[:-1]) + (_State.ffn,))
        out = tex.nvfp4_quantize_with_amax(sw, quantizer, amax[0:1], amax[1:2])
        _State.n_fused_cast += 1
        if _DEBUG and _State.debug_calls <= _DEBUG * 4 and not torch.cuda.is_current_stream_capturing():
            try:
                torch.cuda.synchronize()
                ref = _State.orig_swiglu_fwd(self, input_, quantizer)

                def cmp(name, a, b):
                    if a is None or b is None:
                        return f"{name}={'both-none' if a is b else 'ONE-NONE'}"
                    return f"{name}={'ok' if torch.equal(a, b) else 'BAD:' + str(int((a != b).sum()))}"

                _dbg("cast: " + " ".join([
                    cmp("rowwise_data", out._rowwise_data, ref._rowwise_data),
                    cmp("rowwise_sf", out._rowwise_scale_inv, ref._rowwise_scale_inv),
                    cmp("columnwise_data", out._columnwise_data, ref._columnwise_data),
                    cmp("columnwise_sf", out._columnwise_scale_inv, ref._columnwise_scale_inv),
                ]) + f" amax_mine={amax.tolist()} amax_te=({ref._amax_rowwise.item() if ref._amax_rowwise is not None else None}, "
                     f"{ref._amax_columnwise.item() if ref._amax_columnwise is not None else None}) "
                     f"usage=({quantizer.rowwise_usage},{quantizer.columnwise_usage}) type={type(out).__name__}")
            except Exception as exc:
                _dbg(f"  cast debug compare failed: {exc!r}")
        if not _State.reported and _State.n_fused_cast >= 64:
            _State.reported = True
            if int(os.environ.get("SLURM_PROCID", "0")) == 0:
                _dbg(f"fused fc1 GEMM+SwiGLU+amax active: gemms={_State.n_fused} weight_swizzles={_State.n_swizzled} "
                     f"fused_casts={_State.n_fused_cast} fallback_casts={_State.n_fallback_cast} config={KERNEL_CONFIG}")
        return out
    if rec is not None:
        _State.n_fallback_cast += 1
        if _State.n_fallback_cast <= 3:
            _dbg(f"swiglu fallback (quantizer={type(quantizer).__name__} rht={getattr(quantizer, 'with_rht', None)} "
                 f"post={getattr(quantizer, 'with_post_rht_amax', None)} ptr_match={input_.data_ptr() == rec['gu_ptr']})")
    return _State.orig_swiglu_fwd(self, input_, quantizer)


def install(ffn_hidden_size: int, hidden_size: int) -> bool:
    if _State.installed:
        return _State.enabled
    _State.installed = True
    if os.environ.get("OPT_FC1_SWIGLU_DISABLE", "0") == "1":
        _State.enabled = False
        return False
    from opt_kernels.arch import rubin_only
    if rubin_only("fused fc1 GEMM + SwiGLU + RHT-amax epilogue (CuTe DSL Sm107 block-scaled GEMM)"):
        _State.enabled = False
        return False
    import transformer_engine.pytorch.ops.basic.basic_linear as bl
    import transformer_engine.pytorch.ops.basic.swiglu as sg

    _State.ffn, _State.hidden = ffn_hidden_size, hidden_size
    _State.orig_general_gemm = bl.general_gemm
    bl.general_gemm = _general_gemm_patched
    _State.orig_swiglu_fwd = sg.SwiGLU._tex_swiglu_forward
    sg.SwiGLU._tex_swiglu_forward = _swiglu_fwd_patched
    if int(os.environ.get("SLURM_PROCID", "0")) == 0:
        print("[opt] fused fc1 GEMM + SwiGLU + RHT-amax epilogue installed", flush=True)
    return True
