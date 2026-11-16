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
# and cutlass.kernels.rubin.gemm.base.dense_blockscaled_gemm_persistent). Those portions are
# subject to the following license:
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
from cutlass.cute.nvgpu import cpasync, tcgen05, OperandMajorMode
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
from cutlass._mlir.dialects import math as math_ir

_NVFP4_FACTOR_INV = float(np.float32(1.0) / np.float32(6.0 * 6.0 * 448.0 * 448.0))
_E4M3_MAX = 448.0
_F32_MAX = 3.4028234663852886e38


class QkvRopeAmaxGemmKernel(Sm107BlockScaledPersistentDenseGemmKernel):

    def __init__(
        self,
        hq: int,
        hkv: int,
        d: int,
        mma_inst_shape: Tuple[int, int, int] = (256, 256, 128),
        mma_tiler: Tuple[int, int, int] = (256, 256, 256),
        cluster_shape_mn: Tuple[int, int] = (2, 1),
        swizzle_size: int = 1,
        raster_order: str = "m",
        epi_variant: str = "lane",
        with_amax: bool = True,
        rope_fma: int = 1,
        split_cols: int = 2,
    ):
        super().__init__(
            16, mma_inst_shape, mma_tiler, cluster_shape_mn,
            swizzle_size=swizzle_size, raster_order=raster_order,
        )
        if epi_variant not in ("lane", "pre", "pre2", "v2", "noreuse", "passthrough", "notable"):
            raise ValueError(f"unknown epilogue variant {epi_variant}")
        self.epi_variant = epi_variant
        self.with_amax = with_amax
        self.rope_fma = rope_fma
        self.split_cols = split_cols
        self.hq, self.hkv, self.d = hq, hkv, d
        self.q_per_group = hq // hkv
        self.group_cols = (self.q_per_group + 2) * d
        self.tile_n = mma_tiler[1]
        if d != 128 or self.tile_n != 256:
            raise ValueError("the fused epilogue assumes head_dim 128 and a 256-wide N tile")
        if self.group_cols % self.tile_n != 0 or (self.q_per_group * d) % self.tile_n != 0:
            raise ValueError("q heads of a GQA group must fill whole 256-wide N tiles")
        self.tiles_per_group = self.group_cols // self.tile_n
        self.q_tiles_per_group = (self.q_per_group * d) // self.tile_n
        self.heads_per_tile = self.tile_n // d
        self.n_total = (hq + 2 * hkv) * d

    @cute.jit
    def __call__(
        self,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer,
        sfb_ptr: cute.Pointer,
        q_ptr: cute.Pointer,
        k_ptr: cute.Pointer,
        v_ptr: cute.Pointer,
        tab_q: cute.Tensor,
        tab_k: cute.Tensor,
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
        self.c_dtype: Type[cutlass.Numeric] = q_ptr.value_type
        self.a_major_mode = OperandMajorMode.K
        self.b_major_mode = OperandMajorMode.K
        self.c_layout = cutlass.tensor_utils.LayoutEnum.ROW_MAJOR

        m, n, k, l = problem_mnkl
        self._setup_attributes()

        a_layout = cute.make_ordered_layout((cute.assume(m, 32), k, l), order=(1, 0, 2))
        b_layout = cute.make_ordered_layout((cute.assume(n, 32), k, l), order=(1, 0, 2))
        c_layout = cute.make_ordered_layout((m, cute.assume(n, 32), l), order=(1, 0, 2))
        a_tensor = cute.make_tensor(a_ptr, a_layout)
        b_tensor = cute.make_tensor(b_ptr, b_layout)
        c_tensor = cute.make_tensor(q_ptr, c_layout)
        q_tensor = cute.make_tensor(
            q_ptr, cute.make_ordered_layout((m, self.hq * self.d, l), order=(1, 0, 2))
        )
        k_tensor = cute.make_tensor(
            k_ptr, cute.make_ordered_layout((m, self.hkv * self.d, l), order=(1, 0, 2))
        )
        v_tensor = cute.make_tensor(
            v_ptr, cute.make_ordered_layout((m, self.hkv * self.d, l), order=(1, 0, 2))
        )

        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(a_tensor.shape, self.sf_vec_size)
        sfa_tensor = cute.make_tensor(sfa_ptr, sfa_layout)
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
        tma_atom_q, tma_tensor_q = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), q_tensor, epi_smem_layout, self.epi_tile,
        )
        tma_atom_k, tma_tensor_k = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), k_tensor, epi_smem_layout, self.epi_tile,
        )
        tma_atom_v, tma_tensor_v = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), v_tensor, epi_smem_layout, self.epi_tile,
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
            tma_atom_q, tma_tensor_q, tma_atom_k, tma_tensor_k, tma_atom_v, tma_tensor_v,
            tab_q, tab_k, amax, amax_a, amax_b,
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
        tma_atom_q: cute.CopyAtom,
        mQ_mnl: cute.Tensor,
        tma_atom_k: cute.CopyAtom,
        mK_mnl: cute.Tensor,
        tma_atom_v: cute.CopyAtom,
        mV_mnl: cute.Tensor,
        mTabQ: cute.Tensor,
        mTabK: cute.Tensor,
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
            cpasync.prefetch_descriptor(tma_atom_q)
            cpasync.prefetch_descriptor(tma_atom_k)
            cpasync.prefetch_descriptor(tma_atom_v)

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

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            ab_pipeline.producer_tail(ab_producer_state)

        if warp_idx == self.mma_warp_id:
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
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            acc_pipeline.producer_tail(acc_producer_state)

        if warp_idx < self.mma_warp_id:
            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            acc_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_acc_stage)
            c_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 32 * len(self.epilogue_warp_id))
            c_pipeline = pipeline.PipelineTmaStore.create(num_stages=self.num_c_stage, producer_group=c_producer_group)

            alpha = (mAmaxA[0] * mAmaxB[0]) * cutlass.Float32(_NVFP4_FACTOR_INV)
            amax_q = cutlass.Float32(0.0)
            amax_k = cutlass.Float32(0.0)
            amax_v = cutlass.Float32(0.0)

            tCgC_t = transform_partitioned_tensor_layout(tCgC)
            tCtAcc_t = transform_partitioned_tensor_layout(tCtAcc_base)
            tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc = epilogue_tmem_copy_and_partition(
                self, tidx, tCtAcc_t, tCgC_t, epi_tile, use_2cta_instrs
            )
            tTR_rAcc2 = cute.make_rmem_tensor(tTR_rAcc.shape, self.acc_dtype)
            tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
            tTR_rC2 = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
            tiled_copy_r2s, tRS_rC, tRS_sC = epilogue_smem_copy_and_partition(
                self, tiled_copy_t2r, tTR_rC, tidx, sC
            )
            tRS_rC2 = tiled_copy_r2s.retile(tTR_rC2)

            head_tiler = (self.cta_tile_shape_mnk[0], self.d)
            gQ_epi = cute.flat_divide(cute.local_tile(mQ_mnl, head_tiler, (None, None, None)), epi_tile)
            gK_epi = cute.flat_divide(cute.local_tile(mK_mnl, head_tiler, (None, None, None)), epi_tile)
            gV_epi = cute.flat_divide(cute.local_tile(mV_mnl, head_tiler, (None, None, None)), epi_tile)
            bSG_sC, bSG_gQ_p = cpasync.tma_partition(
                tma_atom_q, 0, cute.make_layout(1), cute.group_modes(sC, 0, 2), cute.group_modes(gQ_epi, 0, 2)
            )
            _, bSG_gK_p = cpasync.tma_partition(
                tma_atom_k, 0, cute.make_layout(1), cute.group_modes(sC, 0, 2), cute.group_modes(gK_epi, 0, 2)
            )
            _, bSG_gV_p = cpasync.tma_partition(
                tma_atom_v, 0, cute.make_layout(1), cute.group_modes(sC, 0, 2), cute.group_modes(gV_epi, 0, 2)
            )
            epilog_sync_barrier = pipeline.NamedBarrier(
                barrier_id=self.epilog_sync_bar_id, num_threads=32 * len(self.epilogue_warp_id)
            )

            epi_tiles_executed = cutlass.Int32(0)
            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                row_blk = cur_tile_coord[0]
                tile_n = cur_tile_coord[1]
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
                num_prev_subtiles = epi_tiles_executed * (2 * self.heads_per_tile * 2)
                epi_tiles_executed += 1

                group = tile_n // self.tiles_per_group
                p_in_group = tile_n % self.tiles_per_group
                m_row = row_blk * self.cta_tile_shape_mnk[0] + tidx

                tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_consumer_state.index)]
                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))

                like = tTR_rAcc.load()
                cos0 = cute.full_like(like, 0.75)
                sin0 = cute.full_like(like, 0.5)
                if cutlass.const_expr(self.epi_variant in ("pre", "pre2")):
                    if p_in_group < self.q_tiles_per_group:
                        self._prefetch_tab_row(mTabQ, m_row)
                        cos0 = self._load_tab_chunk(mTabQ, m_row, 0, like)
                        sin0 = self._load_tab_chunk(mTabQ, m_row, 2, like)
                    else:
                        self._prefetch_tab_row(mTabK, m_row)
                        cos0 = self._load_tab_chunk(mTabK, m_row, 0, like)
                        sin0 = self._load_tab_chunk(mTabK, m_row, 2, like)
                acc_pipeline.consumer_wait(acc_consumer_state)

                if p_in_group < self.q_tiles_per_group:
                    head0 = group * self.q_per_group + p_in_group * self.heads_per_tile
                    bSG_gX0 = bSG_gQ_p[(None, None, None, row_blk, head0, 0)]
                    bSG_gX0 = cute.group_modes(bSG_gX0, 1, cute.rank(bSG_gX0))
                    bSG_gX1 = bSG_gQ_p[(None, None, None, row_blk, head0 + 1, 0)]
                    bSG_gX1 = cute.group_modes(bSG_gX1, 1, cute.rank(bSG_gX1))
                    amax_q = self._epi_tile_q(
                        tidx, warp_idx, tiled_copy_t2r, tTR_tAcc, tTR_rAcc, tTR_rAcc2, tTR_rC, tTR_rC2,
                        tiled_copy_r2s, tRS_rC, tRS_rC2, tRS_sC, bSG_sC, bSG_gX0, bSG_gX1, tma_atom_q, mTabQ,
                        m_row, cos0, sin0, num_prev_subtiles, epilog_sync_barrier, c_pipeline, alpha, amax_q,
                    )
                    amax_k = amax_k
                    amax_v = amax_v
                else:
                    bSG_gXk = bSG_gK_p[(None, None, None, row_blk, group, 0)]
                    bSG_gXk = cute.group_modes(bSG_gXk, 1, cute.rank(bSG_gXk))
                    bSG_gXv = bSG_gV_p[(None, None, None, row_blk, group, 0)]
                    bSG_gXv = cute.group_modes(bSG_gXv, 1, cute.rank(bSG_gXv))
                    amax_k, amax_v = self._epi_tile_kv(
                        tidx, warp_idx, tiled_copy_t2r, tTR_tAcc, tTR_rAcc, tTR_rAcc2, tTR_rC, tTR_rC2,
                        tiled_copy_r2s, tRS_rC, tRS_rC2, tRS_sC, bSG_sC, bSG_gXk, bSG_gXv, tma_atom_k, tma_atom_v,
                        mTabK, m_row, cos0, sin0, num_prev_subtiles, epilog_sync_barrier, c_pipeline, alpha,
                        amax_k, amax_v,
                    )
                    amax_q = amax_q

                epilog_sync_barrier.arrive_and_wait()
                with cute.arch.elect_one():
                    acc_pipeline.consumer_release(acc_consumer_state)
                acc_consumer_state.advance()

            c_pipeline.producer_tail()

            amax_q = cute.arch.warp_reduction(amax_q, cute.arch.fmax)
            amax_k = cute.arch.warp_reduction(amax_k, cute.arch.fmax)
            amax_v = cute.arch.warp_reduction(amax_v, cute.arch.fmax)
            if tidx % 32 == 0:
                cute.arch.atomic_fmax(mAmax.iterator + 0, amax_q, sign_bit=False)
                cute.arch.atomic_fmax(mAmax.iterator + 1, amax_k, sign_bit=False)
                cute.arch.atomic_fmax(mAmax.iterator + 2, amax_v, sign_bit=False)

            tmem.relinquish_alloc_permit()
            tmem.free(acc_tmem_ptr)

    @cute.jit
    def _epi_store_subtile(
        self, warp_idx, tiled_copy_r2s, tRS_rX, tRS_sC, bSG_sC, bSG_gX, tma_atom_x,
        subtile, order, num_prev_subtiles, epilog_sync_barrier, c_pipeline,
    ):
        c_buffer = (num_prev_subtiles + order) % self.num_c_stage
        cute.copy(tiled_copy_r2s, tRS_rX, tRS_sC[(None, None, None, c_buffer)])
        cute.arch.fence_proxy("async.shared", space="cta")
        epilog_sync_barrier.arrive_and_wait()
        if warp_idx == self.epilogue_warp_id[0]:
            cute.copy(tma_atom_x, bSG_sC[(None, c_buffer)], bSG_gX[(None, subtile)])
            c_pipeline.producer_commit()
            c_pipeline.producer_acquire()
        epilog_sync_barrier.arrive_and_wait()

    @cute.jit
    def _prefetch_tab_row(self, mTab, m_row):
        n = self.epi_tile_n
        base = cute.assume(m_row * (4 * n), 4 * n)
        for c in cutlass.range_constexpr(4):
            ptr = mTab.iterator + (base + c * n)
            cute.arch.prefetch(ptr.llvm_ptr, cache_level="L2")

    @cute.jit
    def _tab_chunks(self, mTab, m_row, c: cutlass.Constexpr, like, cos0, sin0):
        n_pairs = 2
        if cutlass.const_expr(self.epi_variant == "notable"):
            return cute.full_like(like, 0.75), cute.full_like(like, 0.5)
        if cutlass.const_expr(c == 0 and self.epi_variant in ("pre", "pre2")):
            return cos0, sin0
        if cutlass.const_expr(self.epi_variant == "lane"):
            return self._load_tab_chunk_rmem(mTab, m_row, c), self._load_tab_chunk_rmem(mTab, m_row, n_pairs + c)
        return self._load_tab_chunk(mTab, m_row, c, like), self._load_tab_chunk(mTab, m_row, n_pairs + c, like)

    @cute.jit
    def _load_tab_chunk_rmem(self, mTab, m_row, chunk: cutlass.Constexpr):
        n = self.epi_tile_n
        dst = cute.make_rmem_tensor((n,), cutlass.Float32)
        if cutlass.const_expr(self.epi_variant == "lane"):
            blk = m_row // 32
            lane = m_row % 32
            for k in cutlass.range_constexpr(n // 4):
                piece = chunk * (n // 4) + k
                off = cute.assume((blk * 32 + piece) * 128 + lane * 4, 4)
                src = cute.make_tensor(mTab.iterator + off, cute.make_layout((4,)))
                dst_k = cute.make_tensor(dst.iterator + 4 * k, cute.make_layout((4,)))
                cute.autovec_copy(src, dst_k)
        else:
            off = cute.assume(m_row * (4 * n) + chunk * n, n)
            src = cute.make_tensor(mTab.iterator + off, cute.make_layout((n,)))
            cute.autovec_copy(src, dst)
        return dst

    @cute.jit
    def _load_tab_chunk(self, mTab, m_row, chunk: cutlass.Constexpr, like):
        n = self.epi_tile_n
        dst = cute.make_rmem_tensor((n,), cutlass.Float32)
        if cutlass.const_expr(self.epi_variant == "lane"):
            blk = m_row // 32
            lane = m_row % 32
            for k in cutlass.range_constexpr(n // 4):
                piece = chunk * (n // 4) + k
                off = cute.assume((blk * 32 + piece) * 128 + lane * 4, 4)
                src = cute.make_tensor(mTab.iterator + off, cute.make_layout((4,)))
                dst_k = cute.make_tensor(dst.iterator + 4 * k, cute.make_layout((4,)))
                cute.autovec_copy(src, dst_k)
        else:
            off = cute.assume(m_row * (4 * n) + chunk * n, n)
            src = cute.make_tensor(mTab.iterator + off, cute.make_layout((n,)))
            cute.autovec_copy(src, dst)
        return dst.load().reshape(like.shape)

    @cute.jit
    def _epi_rope_pair(
        self, warp_idx, tiled_copy_t2r, tTR_tAcc, tTR_rAcc, tTR_rAcc2, tTR_rC, tTR_rC2,
        tiled_copy_r2s, tRS_rC, tRS_rC2, tRS_sC, bSG_sC, bSG_gX, tma_atom_x, cos_v, sin_v,
        ja: cutlass.Constexpr, jb: cutlass.Constexpr, dst_a: cutlass.Constexpr, dst_b: cutlass.Constexpr,
        order: cutlass.Constexpr, num_prev_subtiles, epilog_sync_barrier, c_pipeline, alpha, amax_acc,
    ):
        cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, ja)], tTR_rAcc)
        cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, jb)], tTR_rAcc2)
        n = cute.size(tTR_rAcc)
        ns = self.split_cols
        w = n // ns
        for h in cutlass.range_constexpr(ns):
            va = cute.make_tensor(tTR_rAcc.iterator + h * w, cute.make_layout((w,))).load()
            vb = cute.make_tensor(tTR_rAcc2.iterator + h * w, cute.make_layout((w,))).load()
            if cutlass.const_expr(isinstance(cos_v, cute.Tensor)):
                cs = cute.make_tensor(cos_v.iterator + h * w, cute.make_layout((w,))).load()
                sn = cute.make_tensor(sin_v.iterator + h * w, cute.make_layout((w,))).load()
            else:
                cs = cos_v
                sn = sin_v
            a = (va * alpha).to(cutlass.BFloat16).to(cutlass.Float32)
            b = (vb * alpha).to(cutlass.BFloat16).to(cutlass.Float32)
            oa = (a * cs - b * sn).to(cutlass.BFloat16)
            if cutlass.const_expr(self.rope_fma == 1):
                t = a * sn
                ob_f = type(a)(math_ir.fma(b.ir_value(), cs.ir_value(), t.ir_value()), a.shape, a.dtype)
            elif cutlass.const_expr(self.rope_fma == 2):
                t = b * cs
                ob_f = type(a)(math_ir.fma(a.ir_value(), sn.ir_value(), t.ir_value()), a.shape, a.dtype)
            else:
                ob_f = b * cs + a * sn
            ob = ob_f.to(cutlass.BFloat16)
            cute.make_tensor(tTR_rC.iterator + h * w, cute.make_layout((w,))).store(oa)
            cute.make_tensor(tTR_rC2.iterator + h * w, cute.make_layout((w,))).store(ob)
            if cutlass.const_expr(self.with_amax):
                ma = abs(oa.to(cutlass.Float32)).reduce(cute.ReductionOp.MAX, cutlass.Float32(0.0), 0)
                mb = abs(ob.to(cutlass.Float32)).reduce(cute.ReductionOp.MAX, cutlass.Float32(0.0), 0)
                amax_acc = cute.arch.fmax(amax_acc, cute.arch.fmax(cutlass.Float32(ma), cutlass.Float32(mb)))
        self._epi_store_subtile(
            warp_idx, tiled_copy_r2s, tRS_rC, tRS_sC, bSG_sC, bSG_gX, tma_atom_x,
            dst_a, order, num_prev_subtiles, epilog_sync_barrier, c_pipeline,
        )
        self._epi_store_subtile(
            warp_idx, tiled_copy_r2s, tRS_rC2, tRS_sC, bSG_sC, bSG_gX, tma_atom_x,
            dst_b, order + 1, num_prev_subtiles, epilog_sync_barrier, c_pipeline,
        )
        return amax_acc

    @cute.jit
    def _epi_plain_subtile(
        self, warp_idx, tiled_copy_t2r, tTR_tAcc, tTR_rAcc, tTR_rC, tiled_copy_r2s, tRS_rC, tRS_sC,
        bSG_sC, bSG_gX, tma_atom_x, j: cutlass.Constexpr, dst: cutlass.Constexpr, order: cutlass.Constexpr,
        num_prev_subtiles, epilog_sync_barrier, c_pipeline, alpha, amax_acc,
    ):
        cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, j)], tTR_rAcc)
        o = (tTR_rAcc.load() * alpha).to(cutlass.BFloat16)
        tTR_rC.store(o)
        if cutlass.const_expr(self.with_amax):
            mo = abs(o.to(cutlass.Float32)).reduce(cute.ReductionOp.MAX, cutlass.Float32(0.0), 0)
            amax_acc = cute.arch.fmax(amax_acc, cutlass.Float32(mo))
        self._epi_store_subtile(
            warp_idx, tiled_copy_r2s, tRS_rC, tRS_sC, bSG_sC, bSG_gX, tma_atom_x,
            dst, order, num_prev_subtiles, epilog_sync_barrier, c_pipeline,
        )
        return amax_acc

    @cute.jit
    def _epi_tile_q(
        self, tidx, warp_idx, tiled_copy_t2r, tTR_tAcc, tTR_rAcc, tTR_rAcc2, tTR_rC, tTR_rC2,
        tiled_copy_r2s, tRS_rC, tRS_rC2, tRS_sC, bSG_sC, bSG_gX0, bSG_gX1, tma_atom_x, mTab,
        m_row, cos0, sin0, num_prev_subtiles, epilog_sync_barrier, c_pipeline, alpha, amax_acc,
    ):
        n_sub = self.d // self.epi_tile_n
        n_pairs = n_sub // 2
        if cutlass.const_expr(self.epi_variant == "passthrough"):
            for half in cutlass.range_constexpr(2):
                for c in cutlass.range_constexpr(n_sub):
                    if cutlass.const_expr(half == 0):
                        amax_acc = self._epi_plain_subtile(
                            warp_idx, tiled_copy_t2r, tTR_tAcc, tTR_rAcc, tTR_rC, tiled_copy_r2s, tRS_rC, tRS_sC,
                            bSG_sC, bSG_gX0, tma_atom_x, half * n_sub + c, c, half * n_sub + c,
                            num_prev_subtiles, epilog_sync_barrier, c_pipeline, alpha, amax_acc)
                    else:
                        amax_acc = self._epi_plain_subtile(
                            warp_idx, tiled_copy_t2r, tTR_tAcc, tTR_rAcc, tTR_rC, tiled_copy_r2s, tRS_rC, tRS_sC,
                            bSG_sC, bSG_gX1, tma_atom_x, half * n_sub + c, c, half * n_sub + c,
                            num_prev_subtiles, epilog_sync_barrier, c_pipeline, alpha, amax_acc)
        else:
            like = tTR_rAcc.load()
            cos1 = cos0
            sin1 = sin0
            if cutlass.const_expr(self.epi_variant == "pre2"):
                cos1 = self._load_tab_chunk(mTab, m_row, 1, like)
                sin1 = self._load_tab_chunk(mTab, m_row, n_pairs + 1, like)
            for c in cutlass.range_constexpr(n_pairs):
                if cutlass.const_expr(self.epi_variant == "pre2" and c == 1):
                    cos_v, sin_v = cos1, sin1
                else:
                    cos_v, sin_v = self._tab_chunks(mTab, m_row, c, like, cos0, sin0)
                for half in cutlass.range_constexpr(2):
                    if cutlass.const_expr(self.epi_variant == "noreuse" and half == 1):
                        cos_v = self._load_tab_chunk(mTab, m_row, c, like)
                        sin_v = self._load_tab_chunk(mTab, m_row, n_pairs + c, like)
                    ja = half * n_sub + c
                    order = c * 4 + half * 2
                    if cutlass.const_expr(half == 0):
                        amax_acc = self._epi_rope_pair(
                            warp_idx, tiled_copy_t2r, tTR_tAcc, tTR_rAcc, tTR_rAcc2, tTR_rC, tTR_rC2,
                            tiled_copy_r2s, tRS_rC, tRS_rC2, tRS_sC, bSG_sC, bSG_gX0, tma_atom_x, cos_v, sin_v,
                            ja, ja + n_pairs, c, c + n_pairs, order, num_prev_subtiles, epilog_sync_barrier,
                            c_pipeline, alpha, amax_acc)
                    else:
                        amax_acc = self._epi_rope_pair(
                            warp_idx, tiled_copy_t2r, tTR_tAcc, tTR_rAcc, tTR_rAcc2, tTR_rC, tTR_rC2,
                            tiled_copy_r2s, tRS_rC, tRS_rC2, tRS_sC, bSG_sC, bSG_gX1, tma_atom_x, cos_v, sin_v,
                            ja, ja + n_pairs, c, c + n_pairs, order, num_prev_subtiles, epilog_sync_barrier,
                            c_pipeline, alpha, amax_acc)
        return amax_acc

    @cute.jit
    def _epi_tile_kv(
        self, tidx, warp_idx, tiled_copy_t2r, tTR_tAcc, tTR_rAcc, tTR_rAcc2, tTR_rC, tTR_rC2,
        tiled_copy_r2s, tRS_rC, tRS_rC2, tRS_sC, bSG_sC, bSG_gXk, bSG_gXv, tma_atom_k, tma_atom_v,
        mTab, m_row, cos0, sin0, num_prev_subtiles, epilog_sync_barrier, c_pipeline, alpha, amax_k, amax_v,
    ):
        n_sub = self.d // self.epi_tile_n
        n_pairs = n_sub // 2
        if cutlass.const_expr(self.epi_variant == "passthrough"):
            for c in cutlass.range_constexpr(n_sub):
                amax_k = self._epi_plain_subtile(
                    warp_idx, tiled_copy_t2r, tTR_tAcc, tTR_rAcc, tTR_rC, tiled_copy_r2s, tRS_rC, tRS_sC,
                    bSG_sC, bSG_gXk, tma_atom_k, c, c, c, num_prev_subtiles, epilog_sync_barrier, c_pipeline,
                    alpha, amax_k)
        else:
            like = tTR_rAcc.load()
            cos1 = cos0
            sin1 = sin0
            if cutlass.const_expr(self.epi_variant == "pre2"):
                cos1 = self._load_tab_chunk(mTab, m_row, 1, like)
                sin1 = self._load_tab_chunk(mTab, m_row, n_pairs + 1, like)
            for c in cutlass.range_constexpr(n_pairs):
                if cutlass.const_expr(self.epi_variant == "pre2" and c == 1):
                    cos_v, sin_v = cos1, sin1
                else:
                    cos_v, sin_v = self._tab_chunks(mTab, m_row, c, like, cos0, sin0)
                amax_k = self._epi_rope_pair(
                    warp_idx, tiled_copy_t2r, tTR_tAcc, tTR_rAcc, tTR_rAcc2, tTR_rC, tTR_rC2,
                    tiled_copy_r2s, tRS_rC, tRS_rC2, tRS_sC, bSG_sC, bSG_gXk, tma_atom_k, cos_v, sin_v,
                    c, c + n_pairs, c, c + n_pairs, c * 2, num_prev_subtiles, epilog_sync_barrier,
                    c_pipeline, alpha, amax_k)
        for c in cutlass.range_constexpr(n_sub):
            amax_v = self._epi_plain_subtile(
                warp_idx, tiled_copy_t2r, tTR_tAcc, tTR_rAcc, tTR_rC, tiled_copy_r2s, tRS_rC, tRS_sC,
                bSG_sC, bSG_gXv, tma_atom_v, n_sub + c, c, n_sub + c, num_prev_subtiles, epilog_sync_barrier,
                c_pipeline, alpha, amax_v)
        return amax_k, amax_v


@cute.jit
def _scale_from_amax(amax, eps):
    a = amax
    if a < eps:
        a = eps
    scale = cutlass.Float32(_E4M3_MAX) / a
    if abs(scale) > cutlass.Float32(_F32_MAX):
        scale = cutlass.Float32(_F32_MAX)
    if a == cutlass.Float32(0.0):
        scale = cutlass.Float32(1.0)
    if a != a:
        scale = cutlass.Float32(1.0)
    if abs(a) > cutlass.Float32(_F32_MAX):
        scale = cutlass.Float32(1.0)
    return scale


@cute.kernel
def _qkv_fp8_scale_kernel(
    mAmax: cute.Tensor, mScale: cute.Tensor, mScaleInv: cute.Tensor, eps: cutlass.Float32,
    shared: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    if tidx < 3:
        a = mAmax[tidx]
        if cutlass.const_expr(shared):
            a = cute.arch.fmax(cute.arch.fmax(mAmax[0], mAmax[1]), mAmax[2])
        sc = _scale_from_amax(a, eps)
        mScale[tidx] = sc
        mScaleInv[tidx] = cutlass.Float32(1.0) / sc


@cute.kernel
def _qkv_fp8_cast_kernel(
    gX: cute.Tensor, gO: cute.Tensor, mScale: cute.Tensor,
    n_q_vec: cutlass.Int32, n_qk_vec: cutlass.Int32,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    idx = bidx * bdim + tidx
    total = cute.size(gX.shape[1])
    if cutlass.dynamic_expr(idx < total):
        region = cutlass.Int32(0)
        if idx >= n_qk_vec:
            region = cutlass.Int32(2)
        elif idx >= n_q_vec:
            region = cutlass.Int32(1)
        scale = mScale[region]
        x = gX[(None, idx)].load().to(cutlass.Float32)
        gO[(None, idx)] = (x * scale).to(cutlass.Float8E4M3FN)


@cute.jit
def _qkv_fp8_cast_host(
    mX: cute.Tensor, mO: cute.Tensor, mAmax: cute.Tensor, mScale: cute.Tensor, mScaleInv: cute.Tensor,
    n_q: cutlass.Int32, n_qk: cutlass.Int32, eps: cutlass.Float32, stream: cuda.CUstream,
    vec: cutlass.Constexpr = 16, shared: cutlass.Constexpr = True,
):
    _qkv_fp8_scale_kernel(mAmax, mScale, mScaleInv, eps, shared).launch(
        grid=(1, 1, 1), block=(32, 1, 1), stream=stream
    )
    gX = cute.zipped_divide(mX, (vec,))
    gO = cute.zipped_divide(mO, (vec,))
    threads = 256
    num_vec = cute.size(gX.shape[1])
    _qkv_fp8_cast_kernel(gX, gO, mScale, n_q // vec, n_qk // vec).launch(
        grid=((num_vec + threads - 1) // threads, 1, 1), block=(threads, 1, 1), stream=stream
    )


class _FusedQkvGemm:

    def __init__(self, hq: int, hkv: int, d: int, device: torch.device, **kernel_kwargs):
        self.hq, self.hkv, self.d = hq, hkv, d
        self.kernel = QkvRopeAmaxGemmKernel(hq, hkv, d, **kernel_kwargs)
        cluster = self.kernel.cluster_shape_mn
        hw = utils.HardwareInfo(device.index if device.index is not None else torch.cuda.current_device())
        self.max_active_clusters = hw.get_max_active_clusters(cluster[0] * cluster[1])
        self.compiled = None

    @staticmethod
    def _ptr(dtype, t: torch.Tensor, align: int):
        return make_ptr(dtype, t.data_ptr(), cutlass.AddressSpace.gmem, assumed_align=align)

    @staticmethod
    def _tab(t: torch.Tensor):
        order = tuple(range(t.dim()))
        return from_dlpack(t, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=order)

    @staticmethod
    def _vec(t: torch.Tensor):
        return from_dlpack(t, assumed_align=16).mark_layout_dynamic()

    def __call__(self, a_data, b_data, a_sf, b_sf, a_amax, b_amax, q, k, v, tab_q, tab_k, amax, M, N, K):
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        args = (
            self._ptr(cutlass.Float4E2M1FN, a_data, 16),
            self._ptr(cutlass.Float4E2M1FN, b_data, 16),
            self._ptr(cutlass.Float8E4M3FN, a_sf, 32),
            self._ptr(cutlass.Float8E4M3FN, b_sf, 32),
            self._ptr(cutlass.BFloat16, q, 16),
            self._ptr(cutlass.BFloat16, k, 16),
            self._ptr(cutlass.BFloat16, v, 16),
            self._tab(tab_q),
            self._tab(tab_k),
            self._vec(amax),
            self._vec(a_amax),
            self._vec(b_amax),
        )
        if self.compiled is None:
            self.compiled = cute.compile(
                self.kernel, *args,
                (cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(0)),
                self.max_active_clusters, stream,
            )
        self.compiled(*args, (M, N, K, 1), stream)


_compiled_cast = {}
CAST_VEC = 16
SCALE_MODE = os.environ.get("OPT_QKV_FP8_SCALE", "shared")


def qkv_fp8_cast(buf: torch.Tensor, amax: torch.Tensor, n_q: int, n_k: int, eps: float, vec: int = None,
                 shared: bool = None):
    vec = CAST_VEC if vec is None else vec
    shared = (SCALE_MODE == "shared") if shared is None else shared
    align = 2 * vec
    out = torch.empty(buf.numel(), dtype=torch.uint8, device=buf.device)
    scales = torch.empty(6, dtype=torch.float32, device=buf.device)
    scale, scale_inv = scales[:3], scales[3:]
    mX = from_dlpack(buf, assumed_align=align).mark_layout_dynamic()
    mO = from_dlpack(out, assumed_align=align).mark_layout_dynamic()
    mO.element_type = cutlass.Float8E4M3FN
    mA = from_dlpack(amax, assumed_align=16).mark_layout_dynamic()
    mSc = from_dlpack(scale, assumed_align=4).mark_layout_dynamic()
    mS = from_dlpack(scale_inv, assumed_align=4).mark_layout_dynamic()
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    args = (mX, mO, mA, mSc, mS, cutlass.Int32(n_q), cutlass.Int32(n_q + n_k), cutlass.Float32(eps), stream)
    key = (vec, shared)
    if key not in _compiled_cast:
        _compiled_cast[key] = cute.compile(_qkv_fp8_cast_host, *args, vec, shared)
    _compiled_cast[key](*args)
    return out, scale_inv


_DEBUG = int(os.environ.get("OPT_QKV_FUSED_DEBUG", "0"))


def _dbg(msg: str) -> None:
    import sys
    rank = os.environ.get("RANK", os.environ.get("SLURM_PROCID", "?"))
    sys.stderr.write(f"[opt qkv-dbg r{rank}] {msg}\n")
    sys.stderr.flush()


class _State:
    enabled = True
    debug_calls = 0
    debug_bwd = 0
    debug_attn = 0
    debug_lnl = 0
    debug_freqs = None
    tables = {}
    last_tables = None
    gemm = None
    record = None
    orig_general_gemm = None
    orig_apply_rope = None
    orig_combine_and_quantize = None
    hq = hkv = d = None


def _rope_table(freqs: torch.Tensor, s: int, b: int) -> torch.Tensor:
    d = freqs.shape[-1]
    f = freqs.reshape(s, d).float()
    half = d // 2
    if not torch.cuda.is_current_stream_capturing():
        if not torch.equal(f[:, :half], f[:, half:]):
            raise RuntimeError("fused qkv RoPE expects rotate-half freqs duplicated over the two halves")
    cs = torch.cat([torch.cos(f[:, :half]), torch.sin(f[:, :half])], dim=1)
    tab = cs[:, None, :].expand(s, b, d).reshape(s * b, 4, half // 2).contiguous()
    return tab


def _lane_table(tab: torch.Tensor) -> torch.Tensor:
    t = tab.shape[0]
    if t % 32 != 0:
        raise RuntimeError("fused qkv RoPE table needs the token count to be a multiple of 32")
    return tab.reshape(t // 32, 32, 32, 4).permute(0, 2, 1, 3).contiguous()


def _get_tables(q_freqs: torch.Tensor, k_freqs: torch.Tensor, s: int, b: int):
    key = (q_freqs.data_ptr(), tuple(q_freqs.shape), k_freqs.data_ptr(), tuple(k_freqs.shape), s, b)
    tabs = _State.tables.get(key)
    _State.debug_freqs = (q_freqs, k_freqs)
    if tabs is None:
        tab_q = _lane_table(_rope_table(q_freqs, s, b))
        same = k_freqs is q_freqs or k_freqs.data_ptr() == q_freqs.data_ptr()
        tab_k = tab_q if same else _lane_table(_rope_table(k_freqs, s, b))
        tabs = (tab_q, tab_k)
        _State.tables[key] = tabs
    _State.last_tables = (tabs[0], tabs[1], s * b)
    return tabs


def _is_qkv_gemm(A, B, kwargs) -> bool:
    from transformer_engine.pytorch.tensor.storage.nvfp4_tensor_storage import NVFP4TensorStorage

    if _State.last_tables is None or not _State.enabled:
        return False
    if not (isinstance(A, NVFP4TensorStorage) and isinstance(B, NVFP4TensorStorage)):
        return False
    if kwargs.get("layout", "TN") != "TN" or kwargs.get("grad", False) or kwargs.get("accumulate", False):
        return False
    if kwargs.get("quantization_params") is not None or kwargs.get("bias") is not None:
        return False
    if kwargs.get("ub") is not None or kwargs.get("extra_output") is not None or kwargs.get("out") is not None:
        return False
    n_total = (_State.hq + 2 * _State.hkv) * _State.d
    if tuple(A.shape) != (n_total, B.shape[-1]) or B.shape[0] != _State.last_tables[2]:
        return False
    if A._rowwise_data is None or B._rowwise_data is None or A._amax_rowwise is None or B._amax_rowwise is None:
        return False
    if not (A._with_gemm_swizzled_scales and B._with_gemm_swizzled_scales):
        return False
    if kwargs.get("out_dtype") not in (None, torch.bfloat16):
        return False
    return True


def _zeros_amax(n: int, device) -> torch.Tensor:
    return torch.zeros(n, dtype=torch.float32, device=device)


def _general_gemm_patched(A, B, *args, **kwargs):
    if args or not _is_qkv_gemm(A, B, kwargs):
        return _State.orig_general_gemm(A, B, *args, **kwargs)
    tab_q, tab_k, M = _State.last_tables
    hq, hkv, d = _State.hq, _State.hkv, _State.d
    N = (hq + 2 * hkv) * d
    K = B.shape[-1]
    dev = B._rowwise_data.device
    buf = torch.empty(M * N, dtype=torch.bfloat16, device=dev)
    nq, nk = M * hq * d, M * hkv * d
    q = buf[:nq].view(M, hq * d)
    k = buf[nq:nq + nk].view(M, hkv * d)
    v = buf[nq + nk:].view(M, hkv * d)
    amax = _zeros_amax(3, dev)
    if _State.gemm is None:
        _State.gemm = _FusedQkvGemm(hq, hkv, d, dev)
    _State.gemm(
        B._rowwise_data, A._rowwise_data, B._rowwise_scale_inv, A._rowwise_scale_inv,
        B._amax_rowwise, A._amax_rowwise, q, k, v, tab_q, tab_k, amax, M, N, K,
    )
    _State.record = {"buf_ptr": buf.data_ptr(), "numel": buf.numel(), "amax": amax, "M": M}
    if _DEBUG and _State.debug_calls < _DEBUG * 4 and not torch.cuda.is_current_stream_capturing():
        _State.debug_calls += 1
        try:
            torch.cuda.synchronize()
            def al(t):
                return f"ptr%32={t.data_ptr() % 32} shape={tuple(t.shape)} dtype={t.dtype} off={t.storage_offset()} contig={t.is_contiguous()}"
            _dbg(f"gemm call {_State.debug_calls}: X data {al(B._rowwise_data)} | X sf {al(B._rowwise_scale_inv)} | "
                 f"W data {al(A._rowwise_data)} | W sf {al(A._rowwise_scale_inv)} | amax_x={B._amax_rowwise.float().tolist()} "
                 f"amax_w={A._amax_rowwise.float().tolist()} swz={B._with_gemm_swizzled_scales},{A._with_gemm_swizzled_scales} "
                 f"kwargs={ {kk: (type(vv).__name__ if vv is not None else None) for kk, vv in kwargs.items()} }")
            _dbg(f"  fused buf: nan={int(torch.isnan(buf).sum())} inf={int(torch.isinf(buf).sum())} absmax={float(buf.float().abs().max())} amax={amax.tolist()}")
            ref = _State.orig_general_gemm(A, B, *args, **kwargs)[0]
            _dbg(f"  ref gemm: nan={int(torch.isnan(ref).sum())} absmax={float(ref.float().abs().max())}")
            if _State.debug_freqs is not None:
                qf, kf = _State.debug_freqs
                s_, b_ = qf.shape[0], M // qf.shape[0]
                rq, rk, rv = _State.orig_apply_rope(ref.view(s_, b_, hkv, (hq // hkv + 2) * d), qf, kf, [hq // hkv * d, d, d])
                for nm, got, rr in (("q", q, rq), ("k", k, rk), ("v", v, rv)):
                    dd = (got.float().reshape(-1) - rr.float().reshape(-1)).abs()
                    _dbg(f"  {nm}: maxabs={float(dd.max())} mismatch_frac={float((dd > 0).float().mean())} ref_absmax={float(rr.float().abs().max())} got_nan={int(torch.isnan(got).sum())}")
        except Exception as exc:
            _dbg(f"  debug compare failed: {exc!r}")
    return buf.view(M, N), None, None, None


class _FusedQkvSplitRope(torch.autograd.Function):

    @staticmethod
    def forward(ctx, mixed_qkv, q_freqs, k_freqs, split, s, b, hq, hkv, d):
        import transformer_engine_torch as tex

        flat = mixed_qkv.view(-1)
        nq, nk = s * b * hq * d, s * b * hkv * d
        q = flat[:nq].view(s, b, hq, d)
        k = flat[nq:nq + nk].view(s, b, hkv, d)
        v = flat[nq + nk:].view(s, b, hkv, d)
        ctx.save_for_backward(q_freqs.float(), k_freqs.float())
        ctx.split = split
        return q, k, v

    @staticmethod
    def backward(ctx, gq, gk, gv):
        import transformer_engine_torch as tex
        from transformer_engine.pytorch.attention.rope import QKVFormat

        q_freqs, k_freqs = ctx.saved_tensors
        grad = tex.fused_qkv_rope_backward(
            gq.contiguous(), gk.contiguous(), gv.contiguous(), q_freqs, k_freqs, ctx.split,
            QKVFormat["sbhd"], False, 1, 0,
        )
        if _DEBUG and _State.debug_bwd < _DEBUG * 4 and not torch.cuda.is_current_stream_capturing():
            _State.debug_bwd += 1
            try:
                torch.cuda.synchronize()
                def st(t):
                    return f"shape={tuple(t.shape)} dtype={t.dtype} contig={t.is_contiguous()} nan={int(torch.isnan(t).sum())} inf={int(torch.isinf(t).sum())} absmax={float(t.float().abs().max())}"
                _dbg(f"rope bwd {_State.debug_bwd}: gq {st(gq)} | gk {st(gk)} | gv {st(gv)} -> grad {st(grad)}")
            except Exception as exc:
                _dbg(f"  bwd debug failed: {exc!r}")
        return grad, None, None, None, None, None, None, None, None


def _apply_fused_qkv_rotary_pos_emb_patched(qkv, q_freqs, k_freqs, qkv_split_arg_list, *args, **kwargs):
    s, b = qkv.shape[0], qkv.shape[1]
    rec = _State.record
    fused = (
        rec is not None and _State.enabled and not args and not kwargs
        and qkv.is_contiguous() and qkv.data_ptr() == rec["buf_ptr"] and qkv.numel() == rec["numel"]
        and list(qkv_split_arg_list) == [_State.hq // _State.hkv * _State.d, _State.d, _State.d]
    )
    if not fused:
        _State.record = None
        _get_tables(q_freqs, k_freqs, s, b)
        return _State.orig_apply_rope(qkv, q_freqs, k_freqs, qkv_split_arg_list, *args, **kwargs)
    _State.record = None
    q, k, v = _FusedQkvSplitRope.apply(
        qkv, q_freqs, k_freqs, list(qkv_split_arg_list), s, b, _State.hq, _State.hkv, _State.d
    )
    _State.record = {
        "ptrs": (q.data_ptr(), k.data_ptr(), v.data_ptr()),
        "shapes": (tuple(q.shape), tuple(k.shape), tuple(v.shape)),
        "amax": rec["amax"],
    }
    return q, k, v


def _combine_and_quantize_patched(qkv_layout, q, k, v, qkv_quantizer, *args, **kwargs):
    from transformer_engine.pytorch.tensor.float8_tensor import Float8CurrentScalingQuantizer, Float8Tensor

    rec = _State.record
    fused = (
        rec is not None and _State.enabled and "ptrs" in rec
        and isinstance(qkv_quantizer, Float8CurrentScalingQuantizer)
        and qkv_layout == "sbhd_sbhd_sbhd"
        and (q.data_ptr(), k.data_ptr(), v.data_ptr()) == rec["ptrs"]
        and (tuple(q.shape), tuple(k.shape), tuple(v.shape)) == rec["shapes"]
        and not getattr(qkv_quantizer, "force_pow_2_scales", False)
    )
    if not fused:
        return _State.orig_combine_and_quantize(qkv_layout, q, k, v, qkv_quantizer, *args, **kwargs)
    _State.record = None
    amax = rec["amax"]
    group = getattr(qkv_quantizer, "amax_reduction_group", None)
    if getattr(qkv_quantizer, "with_amax_reduction", False) and group is not None:
        torch.distributed.all_reduce(amax, op=torch.distributed.ReduceOp.MAX, group=group)
    nq, nk = q.numel(), k.numel()
    base = torch.empty(0, dtype=q.dtype, device=q.device).set_(q.untyped_storage())
    buf = base[q.storage_offset(): q.storage_offset() + nq + nk + v.numel()]
    fp8, scale_inv = qkv_fp8_cast(buf, amax, nq, nk, float(getattr(qkv_quantizer, "amax_epsilon", 0.0)))
    if _DEBUG and _State.debug_calls <= _DEBUG * 4 and not torch.cuda.is_current_stream_capturing():
        try:
            torch.cuda.synchronize()
            _dbg(f"cast: layout={qkv_layout} amax={amax.tolist()} scale_inv={scale_inv.tolist()} fp8_nonzero={int((fp8 != 0).sum())}/{fp8.numel()} "
                 f"quantizer dtype={qkv_quantizer.dtype} eps={getattr(qkv_quantizer, 'amax_epsilon', None)} pow2={getattr(qkv_quantizer, 'force_pow_2_scales', None)} "
                 f"args={args} kwargs={list(kwargs)}")
            oq, ok, ov, lay, _ = _State.orig_combine_and_quantize(qkv_layout, q, k, v, qkv_quantizer, *args, **kwargs)
            _dbg(f"  orig: layout={lay} scale_inv={oq._scale_inv.tolist()} data_dtype={oq._data.dtype} data_mismatch_q={int((oq._data.reshape(-1).view(torch.uint8) != fp8[:nq]).sum())} "
                 f"k={int((ok._data.reshape(-1).view(torch.uint8) != fp8[nq:nq + nk]).sum())} v={int((ov._data.reshape(-1).view(torch.uint8) != fp8[nq + nk:]).sum())}")
        except Exception as exc:
            _dbg(f"  cast debug compare failed: {exc!r}")
    outs = []
    for i, (t, off, n) in enumerate(((q, 0, nq), (k, nq, nk), (v, nq + nk, v.numel()))):
        outs.append(
            Float8Tensor(
                shape=t.shape,
                dtype=t.dtype,
                data=fp8[off:off + n].view(t.shape),
                fp8_scale_inv=scale_inv[i:i + 1],
                fp8_dtype=qkv_quantizer.dtype,
                requires_grad=False,
                data_transpose=None,
                quantizer=qkv_quantizer,
                device=t.device,
            )
        )
    if _DEBUG and _State.debug_calls <= _DEBUG * 4 and not torch.cuda.is_current_stream_capturing():
        try:
            torch.cuda.synchronize()
            for nm, t in zip("qkv", outs):
                _dbg(f"  fp8 {nm}: shape={tuple(t.shape)} dtype={t.dtype} data={t._data.dtype}{tuple(t._data.shape)} contig={t._data.is_contiguous()} "
                     f"scale_inv={t._scale_inv.tolist()} fp8_dtype={t._fp8_dtype} transpose_invalid={t._transpose_invalid} quantizer={type(t._quantizer).__name__} "
                     f"dequant_absmax={float(t.dequantize().float().abs().max())} dequant_nan={int(torch.isnan(t.dequantize()).sum())}")
        except Exception as exc:
            _dbg(f"  fp8 debug failed: {exc!r}")
    return outs[0], outs[1], outs[2], qkv_layout, None


def install(num_attention_heads: int = 32, num_query_groups: int = 8, kv_channels: int = 128) -> None:
    if _State.orig_general_gemm is not None:
        return
    from opt_kernels.arch import rubin_only
    if rubin_only("fused qkv GEMM + RoPE + FP8-amax epilogue (CuTe DSL Sm107 block-scaled GEMM)"):
        return
    import transformer_engine.pytorch.module.layernorm_linear as lnl
    import transformer_engine.pytorch.attention.dot_product_attention.backends as dpa_backends
    import megatron.core.transformer.attention as mcore_attention

    _State.hq, _State.hkv, _State.d = num_attention_heads, num_query_groups, kv_channels
    _State.orig_general_gemm = lnl.general_gemm
    lnl.general_gemm = _general_gemm_patched
    _State.orig_apply_rope = mcore_attention.apply_fused_qkv_rotary_pos_emb
    mcore_attention.apply_fused_qkv_rotary_pos_emb = _apply_fused_qkv_rotary_pos_emb_patched
    _State.orig_combine_and_quantize = dpa_backends.combine_and_quantize
    dpa_backends.combine_and_quantize = _combine_and_quantize_patched
    if _DEBUG:
        import megatron.core.extensions.transformer_engine as mcore_te
        _orig_dpa_fwd = mcore_te.TEDotProductAttention.forward
        _orig_lnl_fwd = lnl.LayerNormLinear.forward

        def _dpa_fwd_dbg(self, query, key, value, *a, **kw):
            out = _orig_dpa_fwd(self, query, key, value, *a, **kw)
            if _State.debug_attn < _DEBUG * 8 and not torch.cuda.is_current_stream_capturing():
                _State.debug_attn += 1
                try:
                    torch.cuda.synchronize()
                    o = out[0] if isinstance(out, (tuple, list)) else out
                    _dbg(f"attn out {_State.debug_attn}: layer={getattr(self, 'layer_number', '?')} q_ptr_match={_State.record is None} "
                         f"shape={tuple(o.shape)} nan={int(torch.isnan(o).sum())} inf={int(torch.isinf(o).sum())} absmax={float(o.float().abs().max())} "
                         f"q nan={int(torch.isnan(query).sum())} absmax={float(query.float().abs().max())}")
                except Exception as exc:
                    _dbg(f"  attn debug failed: {exc!r}")
            return out

        def _lnl_fwd_dbg(self, inp, *a, **kw):
            out = _orig_lnl_fwd(self, inp, *a, **kw)
            if _State.debug_lnl < _DEBUG * 8 and not torch.cuda.is_current_stream_capturing():
                _State.debug_lnl += 1
                try:
                    torch.cuda.synchronize()
                    o = out[0] if isinstance(out, (tuple, list)) else out
                    _dbg(f"lnl {_State.debug_lnl}: in shape={tuple(inp.shape)} nan={int(torch.isnan(inp).sum())} absmax={float(inp.float().abs().max())} | "
                         f"out shape={tuple(o.shape)} nan={int(torch.isnan(o).sum())} absmax={float(o.float().abs().max())} out_features={self.out_features}")
                except Exception as exc:
                    _dbg(f"  lnl debug failed: {exc!r}")
            return out

        mcore_te.TEDotProductAttention.forward = _dpa_fwd_dbg
        lnl.LayerNormLinear.forward = _lnl_fwd_dbg
    if int(os.environ.get("SLURM_PROCID", "0")) == 0:
        print("[opt] fused qkv GEMM + RoPE + FP8-amax epilogue installed", flush=True)
