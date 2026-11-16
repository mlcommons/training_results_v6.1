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
# The RHT + NVFP4 cast arithmetic (global scale, block scale, amax, e2m1 rounding, Philox
# constants and draw schedule) mirrors NVIDIA Transformer Engine (Apache-2.0, Copyright (c)
# 2022-2026 NVIDIA CORPORATION & AFFILIATES): common/cast/nvfp4/core_nvfp4.cuh,
# common/hadamard_transform/*, common/util/ptx.cuh, common/util/curanddx.hpp; the NVTE struct
# and enum mirrors follow the public transformer_engine.h. The CuTe kernel is written from scratch.
from __future__ import annotations

import ctypes
import os
import sys
from typing import Optional

import numpy as np
import torch

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import Float32, Int32, Int64, Uint32, Uint64
from cutlass.cute.nvgpu import cpasync, tcgen05, OperandMajorMode
from cutlass.cute.runtime import from_dlpack, make_ptr
from cutlass.utils.gemm.sm100 import transform_partitioned_tensor_layout

_ARCH = "sm_107"
_RHT_DIM = 16
_RNG_ELTS_PER_THREAD = 1024
_NVFP4_FACTOR_INV = float(np.float32(1.0) / np.float32(6.0 * 6.0 * 448.0 * 448.0))

_TE_VERIFIED_VERSIONS = ("2.20.0+5b28d7aa",)

_PHILOX_W0 = 0x9E3779B9
_PHILOX_W1 = 0xBB67AE85
_PHILOX_M0 = 0xD2511F53
_PHILOX_M1 = 0xCD9E8D57
_PHILOX_ROUNDS = 3


def _ptx_amax16() -> str:
    return (
        "{\n"
        ".reg .b32 m0, m1, m2, m3, lo, hi;\n"
        ".reg .f32 r;\n"
        "max.xorsign.abs.bf16x2 m0, {$r0}, {$r1};\n"
        "max.xorsign.abs.bf16x2 m1, {$r2}, {$r3};\n"
        "max.xorsign.abs.bf16x2 m2, {$r4}, {$r5};\n"
        "max.xorsign.abs.bf16x2 m3, {$r6}, {$r7};\n"
        "max.xorsign.abs.bf16x2 m0, m0, m1;\n"
        "max.xorsign.abs.bf16x2 m2, m2, m3;\n"
        "max.xorsign.abs.bf16x2 m0, m0, m2;\n"
        "shl.b32 lo, m0, 16;\n"
        "and.b32 hi, m0, 0xffff0000;\n"
        "max.ftz.xorsign.abs.f32 r, lo, hi;\n"
        "abs.ftz.f32 {$w0}, r;\n"
        "}"
    )


def _ptx_blockscale() -> str:
    return (
        "{\n"
        ".reg .f32 p, u, q, r;\n"
        ".reg .b16 h, ulo, uhi;\n"
        ".reg .b32 uu;\n"
        "mul.ftz.f32 p, {$r0}, {$r1};\n"
        "cvt.rn.satfinite.e4m3x2.f32 h, 0f00000000, p;\n"
        "cvt.u32.u16 {$w0}, h;\n"
        "cvt.rn.f16x2.e4m3x2 uu, h;\n"
        "mov.b32 {ulo, uhi}, uu;\n"
        "cvt.f32.f16 u, ulo;\n"
        "mul.ftz.f32 q, u, {$r2};\n"
        "div.rn.f32 r, 0f3F800000, q;\n"
        "min.ftz.f32 {$w1}, r, 0f7F7FFFFF;\n"
        "}"
    )


def _ptx_global_scales() -> str:
    return (
        "{\n"
        ".reg .pred p;\n"
        ".reg .f32 t, e;\n"
        "setp.gt.ftz.f32 p, {$r0}, 0f00000000;\n"
        "div.rn.f32 t, 0f45280000, {$r0};\n"
        "min.ftz.f32 t, t, 0f7F7FFFFF;\n"
        "selp.f32 e, t, 0f3F800000, p;\n"
        "div.rn.f32 {$w1}, 0f3F800000, e;\n"
        "mul.ftz.f32 {$w2}, e, 0f3E2AAAAB;\n"
        "mov.b32 {$w0}, e;\n"
        "}"
    )


def _ptx_unpack_mul() -> str:
    s = ".reg .b32 f<16>;\n"
    for j in range(8):
        s += f"shl.b32 f{2 * j}, {{$r{j}}}, 16;\n"
        s += f"and.b32 f{2 * j + 1}, {{$r{j}}}, 0xffff0000;\n"
    for i in range(16):
        s += f"mul.ftz.f32 f{i}, f{i}, {{$r8}};\n"
    return s


def _ptx_quant16_sr() -> str:
    return (
        "{\n"
        ".reg .b16 h0, h1, h2, h3;\n"
        + _ptx_unpack_mul()
        + "cvt.rs.satfinite.e2m1x4.f32 h0, {f3, f2, f1, f0}, {$r9};\n"
        "cvt.rs.satfinite.e2m1x4.f32 h1, {f7, f6, f5, f4}, {$r10};\n"
        "cvt.rs.satfinite.e2m1x4.f32 h2, {f11, f10, f9, f8}, {$r11};\n"
        "cvt.rs.satfinite.e2m1x4.f32 h3, {f15, f14, f13, f12}, {$r12};\n"
        "mov.b32 {$w0}, {h0, h1};\n"
        "mov.b32 {$w1}, {h2, h3};\n"
        "}"
    )


def _ptx_quant16_rn() -> str:
    s = "{\n.reg .b8 b0, b1, b2, b3, b4, b5, b6, b7;\n" + _ptx_unpack_mul()
    for j in range(8):
        s += f"cvt.rn.satfinite.e2m1x2.f32 b{j}, f{2 * j + 1}, f{2 * j};\n"
    s += "mov.b32 {$w0}, {b0, b1, b2, b3};\nmov.b32 {$w1}, {b4, b5, b6, b7};\n}"
    return s


_PTX_CVT_BF16X2 = "cvt.rn.bf16x2.f32 {$w0}, {$r0}, {$r1};"
_PTX_ST_B32 = "st.global.L1::no_allocate.b32 [{$r0}], {$r1};"
_PTX_ST_V8 = "st.global.L1::no_allocate.v8.b32 [{$r0}], {{$r1}, {$r2}, {$r3}, {$r4}, {$r5}, {$r6}, {$r7}, {$r8}};"


def _philox_keys(seed: Uint64):
    k0 = seed.to(Uint32)
    k1 = (seed >> Uint64(32)).to(Uint32)
    keys = []
    for r in range(_PHILOX_ROUNDS):
        a = Uint32((r * _PHILOX_W0) & 0xFFFFFFFF)
        b = Uint32((r * _PHILOX_W1) & 0xFFFFFFFF)
        keys.append((k0 + a, k1 + b))
    return keys


def _philox4(ctr: Uint64, seq: Uint64, keys):
    c0 = ctr.to(Uint32)
    c1 = (ctr >> Uint64(32)).to(Uint32)
    c2 = seq.to(Uint32)
    c3 = (seq >> Uint64(32)).to(Uint32)
    m0 = Uint32(_PHILOX_M0)
    m1 = Uint32(_PHILOX_M1)
    for r in range(len(keys)):
        k0, k1 = keys[r]
        lo0 = c0 * m0
        hi0 = cute.arch.mul_hi(c0, m0)
        lo1 = c2 * m1
        hi1 = cute.arch.mul_hi(c2, m1)
        c0, c1, c2, c3 = (hi1 ^ c1) ^ k0, lo1, (hi0 ^ c3) ^ k1, lo0
    return c0, c1, c2, c3


def _rng_state(pSeed, pOff, seed_val: Int64, offset_val: Int64, captured: Int32):
    mSeed = cute.make_tensor(pSeed, cute.make_layout((1,)))
    mOff = cute.make_tensor(pOff, cute.make_layout((1,)))
    cap = captured.to(Uint64)
    ncap = Uint64(1) - cap
    seed_u = cap * mSeed[0].bitcast(Uint64) + ncap * seed_val.bitcast(Uint64)
    off_u = cap * (mOff[0].bitcast(Uint64) + offset_val.bitcast(Uint64)) + ncap * offset_val.bitcast(Uint64)
    return seed_u, off_u


class RhtCastNvfp4Kernel:

    TILE_H = 128
    TILE_T = 128
    NUM_STAGES = 6
    NUM_ACC = 4
    NUM_CLC_STAGES = 2
    CLC_RESPONSE_BYTES = 16
    TMEM_COLS = 512
    COL_WARPS = tuple(range(0, 8))
    ROW_WARPS = tuple(range(8, 16))
    TMA_WARP = 16
    MMA_WARP = 17
    SCHED_WARP = 18
    NUM_THREADS = 32 * 19
    END = -1

    def __init__(self, stochastic_rounding: bool, swizzled_sf: bool):
        self.sr = bool(stochastic_rounding)
        self.swizzled_sf = bool(swizzled_sf)
        self.mma_tiler = (self.TILE_H, _RHT_DIM, self.TILE_T)
        self.tma_bytes = self.TILE_H * self.TILE_T * 2
        self.ptx_amax = _ptx_amax16()
        self.ptx_blockscale = _ptx_blockscale()
        self.ptx_gscale = _ptx_global_scales()
        self.ptx_quant = _ptx_quant16_sr() if self.sr else _ptx_quant16_rn()
        self.shared_storage = None

    @staticmethod
    def _row_view_layout(a_layout):
        outer = a_layout.outer
        shp, strd = outer.shape, outer.stride
        assert isinstance(shp, tuple) and len(shp) == 4, f"unexpected A smem layout shape {shp}"
        assert isinstance(shp[0], tuple) and len(shp[0]) == 2, f"unexpected A smem atom shape {shp[0]}"
        assert shp[1] == 1, f"unexpected MMA_M {shp[1]}"
        new_shape = (shp[0][0], (shp[0][1], shp[2]), shp[3])
        new_stride = (strd[0][0], (strd[0][1], strd[2]), strd[3])
        new_outer = cute.make_layout(new_shape, stride=new_stride)
        return cute.make_composed_layout(a_layout.inner, 0, new_outer)

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mRht: cute.Tensor,
        d_base: Int64,
        sfd_base: Int64,
        qa_base: Int64,
        sfa_base: Int64,
        pAmaxA: cute.Pointer,
        pAmaxC: cute.Pointer,
        pPart0: cute.Pointer,
        pPart1: cute.Pointer,
        pAlpha0: cute.Pointer,
        pAlpha1: cute.Pointer,
        alpha_mask: Int32,
        pSeed: cute.Pointer,
        pOff: cute.Pointer,
        seed_val: Int64,
        offset_val: Int64,
        captured: Int32,
        n_hidden: Int32,
        m_tokens: Int32,
        stream: cuda.CUstream,
    ):
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            cutlass.BFloat16, cutlass.BFloat16, OperandMajorMode.MN, OperandMajorMode.MN,
            Float32, tcgen05.CtaGroup.ONE, (self.TILE_H, _RHT_DIM),
        )
        a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, cutlass.BFloat16, self.NUM_STAGES
        )
        b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma, (self.TILE_H, _RHT_DIM, _RHT_DIM), cutlass.BFloat16, 1
        )
        a_mks_layout = self._row_view_layout(a_smem_layout_staged)
        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            cpasync.CopyBulkTensorTileG2SOp(), mX, a_smem_layout, self.mma_tiler, tiled_mma, (1, 1, 1, 1),
        )
        num_stages = self.NUM_STAGES
        num_acc = self.NUM_ACC
        num_clc = self.NUM_CLC_STAGES
        clc_words = self.CLC_RESPONSE_BYTES // 4

        tiles_m = n_hidden // self.TILE_H
        tiles_t = m_tokens // self.TILE_T
        tile_sched_params = utils.ClcDynamicPersistentTileSchedulerParams((tiles_m, tiles_t, 1), (1, 1, 1))

        @cute.struct
        class SharedStorage:
            full_mbar: cute.struct.MemRange[cutlass.Int64, num_stages]
            empty_mbar: cute.struct.MemRange[cutlass.Int64, num_stages]
            acc_full_mbar: cute.struct.MemRange[cutlass.Int64, num_acc]
            acc_empty_mbar: cute.struct.MemRange[cutlass.Int64, num_acc]
            tile_full_mbar: cute.struct.MemRange[cutlass.Int64, num_acc]
            clc_mbar: cute.struct.MemRange[cutlass.Int64, 2 * num_clc]
            tmem_holding_buf: cutlass.Int32
            tile_stage: cute.struct.MemRange[cutlass.Int32, num_stages]
            tile_acc: cute.struct.MemRange[cutlass.Int32, num_acc]
            clc_response: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, clc_words * num_clc], 16]
            sSF: cute.struct.Align[cute.struct.MemRange[cutlass.Uint32, 4 * 2 * 128], 128]
            sB: cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, cute.cosize(b_smem_layout_staged.outer)], 1024
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, cute.cosize(a_smem_layout_staged.outer)], 1024
            ]

        self.shared_storage = SharedStorage
        self.kernel(
            tiled_mma, tma_atom_a, tma_tensor_a, mRht,
            a_smem_layout_staged, a_mks_layout, b_smem_layout_staged,
            d_base, sfd_base, qa_base, sfa_base, pAmaxA, pAmaxC, pPart0, pPart1, pAlpha0, pAlpha1, alpha_mask,
            pSeed, pOff, seed_val, offset_val, captured, n_hidden, m_tokens, tile_sched_params,
        ).launch(
            grid=[tiles_m, tiles_t, 1],
            block=[self.NUM_THREADS, 1, 1],
            cluster=[1, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA: cute.Tensor,
        mRht: cute.Tensor,
        a_smem_layout_staged: cute.ComposedLayout,
        a_mks_layout: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        d_base: Int64,
        sfd_base: Int64,
        qa_base: Int64,
        sfa_base: Int64,
        pAmaxA: cute.Pointer,
        pAmaxC: cute.Pointer,
        pPart0: cute.Pointer,
        pPart1: cute.Pointer,
        pAlpha0: cute.Pointer,
        pAlpha1: cute.Pointer,
        alpha_mask: Int32,
        pSeed: cute.Pointer,
        pOff: cute.Pointer,
        seed_val: Int64,
        offset_val: Int64,
        captured: Int32,
        n_hidden: Int32,
        m_tokens: Int32,
        tile_sched_params,
    ):
        S = self.NUM_STAGES
        A = self.NUM_ACC
        D = self.NUM_CLC_STAGES
        clc_words = self.CLC_RESPONSE_BYTES // 4
        END = Int32(self.END)
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()

        smem = cutlass.memory.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        full_ptr = storage.full_mbar.data_ptr()
        empty_ptr = storage.empty_mbar.data_ptr()
        accf_ptr = storage.acc_full_mbar.data_ptr()
        acce_ptr = storage.acc_empty_mbar.data_ptr()
        tilef_ptr = storage.tile_full_mbar.data_ptr()

        if warp_idx == self.TMA_WARP:
            cpasync.prefetch_descriptor(tma_atom_a)
            with cute.arch.elect_one():
                for s in cutlass.range_constexpr(S):
                    cute.arch.mbarrier_init(full_ptr + s, 1)
                    cute.arch.mbarrier_init(empty_ptr + s, 1 + len(self.ROW_WARPS))
                for a in cutlass.range_constexpr(A):
                    cute.arch.mbarrier_init(accf_ptr + a, 1)
                    cute.arch.mbarrier_init(acce_ptr + a, len(self.COL_WARPS))
                    cute.arch.mbarrier_init(tilef_ptr + a, 1)
        clc_pipeline = pipeline.PipelineClcFetchAsync.create(
            barrier_storage=storage.clc_mbar.data_ptr(),
            num_stages=D,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 2 * 32),
            tx_count=self.CLC_RESPONSE_BYTES,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
            defer_sync=True,
        )
        clc_response_ptr = storage.clc_response.data_ptr()

        tmem_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=32 * (1 + len(self.COL_WARPS)))
        done_barrier = pipeline.NamedBarrier(barrier_id=2, num_threads=32 * (1 + len(self.COL_WARPS)))
        tmem = cutlass.memory.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_barrier,
            allocator_warp_id=self.MMA_WARP,
            is_two_cta=False,
            arch=_ARCH,
        )
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        bidx, bidy, _ = cute.arch.block_idx()
        if tidx == 0:
            if bidx == 0:
                if bidy == 0:
                    if (alpha_mask & Int32(1)) != Int32(0):
                        al_a = cute.make_tensor(pAmaxA, cute.make_layout((1,)))[0]
                        al_p0 = cute.make_tensor(pPart0, cute.make_layout((1,)))[0]
                        mAlpha0 = cute.make_tensor(pAlpha0, cute.make_layout((1,)))
                        mAlpha0[0] = (al_a * al_p0) * Float32(_NVFP4_FACTOR_INV)
                    if (alpha_mask & Int32(2)) != Int32(0):
                        al_c = cute.make_tensor(pAmaxC, cute.make_layout((1,)))[0]
                        al_p1 = cute.make_tensor(pPart1, cute.make_layout((1,)))[0]
                        mAlpha1 = cute.make_tensor(pAlpha1, cute.make_layout((1,)))
                        mAlpha1[0] = (al_c * al_p1) * Float32(_NVFP4_FACTOR_INV)

        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        sA_mks = storage.sA.get_tensor(a_mks_layout.outer, swizzle=a_mks_layout.inner)
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
        sSF = storage.sSF.get_tensor(cute.make_layout(4 * 2 * 128))
        sTile = storage.tile_stage.get_tensor(cute.make_layout(S))
        sTileAcc = storage.tile_acc.get_tensor(cute.make_layout(A))

        tiles_m = n_hidden // self.TILE_H
        kmax = (cutlass.min(m_tokens, Int32(1024)) + Int32(127)) // Int32(128)

        acc_shape = tiled_mma.partition_shape_C((self.TILE_H, self.TILE_T))
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, A))

        if warp_idx == self.TMA_WARP:
            tile_sched = utils.ClcDynamicPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim(), clc_response_ptr
            )
            work_tile = tile_sched.initial_work_tile_info()
            clc_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, D)
            gA = cute.local_tile(mA, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None))
            thr_mma = tiled_mma.get_slice(0)
            tCgA = thr_mma.partition_A(gA)
            tAsA, tAgA = cpasync.tma_partition(
                tma_atom_a, 0, cute.make_layout(1), cute.group_modes(sA, 0, 3), cute.group_modes(tCgA, 0, 3),
            )
            i = Int32(0)
            while work_tile.is_valid_tile:
                m_idx = work_tile.tile_idx[0]
                t_idx = work_tile.tile_idx[1]
                s = i % S
                par = ((i // S) & 1) ^ 1
                cute.arch.mbarrier_wait(empty_ptr + s, par)
                with cute.arch.elect_one():
                    sTile[s] = m_idx | (t_idx << Int32(16))
                    cute.arch.mbarrier_arrive_and_expect_tx(full_ptr + s, self.tma_bytes)
                cute.copy(tma_atom_a, tAgA[(None, m_idx, t_idx, 0)], tAsA[(None, s)], tma_bar_ptr=full_ptr + s)
                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.work_tile_info_from_clc_response(
                    (clc_response_ptr + clc_consumer_state.index * clc_words).align(16)
                )
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()
                i += Int32(1)
            s = i % S
            par = ((i // S) & 1) ^ 1
            cute.arch.mbarrier_wait(empty_ptr + s, par)
            with cute.arch.elect_one():
                sTile[s] = END
                cute.arch.mbarrier_arrive(full_ptr + s)

        elif warp_idx == self.SCHED_WARP:
            tile_sched = utils.ClcDynamicPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim(), clc_response_ptr
            )
            work_tile = tile_sched.initial_work_tile_info()
            clc_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.ProducerConsumer, D)
            clc_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, D)
            while work_tile.is_valid_tile:
                clc_pipeline.producer_acquire(clc_producer_state)
                clc_mbar = clc_pipeline.producer_get_barrier(clc_producer_state)
                clc_slot = (clc_response_ptr + clc_producer_state.index * clc_words).align(16)
                with cute.arch.elect_one():
                    cute.arch.issue_clc_query(clc_mbar, clc_slot)
                clc_producer_state.advance()
                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.work_tile_info_from_clc_response(
                    (clc_response_ptr + clc_consumer_state.index * clc_words).align(16)
                )
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()
            clc_pipeline.producer_tail(clc_producer_state)

        elif warp_idx == self.MMA_WARP:
            tmem.allocate(self.TMEM_COLS)
            tmem.wait_for_alloc()
            acc_ptr = tmem.retrieve_ptr(Float32)
            tmem.relinquish_alloc_permit()
            tCtAcc = cute.make_tensor(acc_ptr, tCtAcc_fake.layout)
            thr_mma = tiled_mma.get_slice(0)
            gB = cute.local_tile(mRht, (_RHT_DIM, _RHT_DIM), (None, None, None))
            tCgB = thr_mma.partition_B(gB)
            with cute.arch.elect_one():
                cute.autovec_copy(tCgB[(None, None, None, 0, 0, 0)], sB[(None, None, None, 0)])
                cute.arch.fence_view_async_shared()
            cute.arch.sync_warp()
            tCrA = tiled_mma.make_fragment_A(sA)
            tCrB = tiled_mma.make_fragment_B(sB)
            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            i = Int32(0)
            go = cutlass.Boolean(True)
            while go:
                s = i % S
                par_s = (i // S) & 1
                a = i % A
                par_a = ((i // A) & 1) ^ 1
                cute.arch.mbarrier_wait(full_ptr + s, par_s)
                packed = sTile[s]
                go = packed != END
                cute.arch.mbarrier_wait(acce_ptr + a, par_a)
                with cute.arch.elect_one():
                    sTileAcc[a] = packed
                    cute.arch.mbarrier_arrive(tilef_ptr + a)
                if go:
                    for kb in cutlass.range_constexpr(self.TILE_T // _RHT_DIM):
                        cute.gemm(
                            tiled_mma, tCtAcc[(None, 0, kb, a)],
                            tCrA[(None, 0, kb, s)], tCrB[(None, 0, 0, 0)],
                            tCtAcc[(None, 0, kb, a)],
                        )
                    with cute.arch.elect_one():
                        tcgen05.commit(accf_ptr + a)
                        tcgen05.commit(empty_ptr + s)
                i += Int32(1)
            done_barrier.arrive_and_wait()
            tmem.free(acc_ptr)

        elif warp_idx < len(self.COL_WARPS):
            tmem.wait_for_alloc()
            acc_ptr = tmem.retrieve_ptr(Float32)
            tCtAcc = cute.make_tensor(acc_ptr, tCtAcc_fake.layout)
            tAcc_mn = transform_partitioned_tensor_layout(tCtAcc)
            tAcc_epi = cute.flat_divide(tAcc_mn, (self.TILE_H, 32))
            atom_t2r = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(32)), Float32)
            tiled_t2r = tcgen05.make_tmem_copy(atom_t2r, tAcc_epi[(None, None, 0, 0, 0)])
            tq = tidx % 128
            g = cute.arch.make_warp_uniform(tidx // 128)
            wq = cute.arch.make_warp_uniform(tq // 32)
            thr_t2r = tiled_t2r.get_slice(tq)
            tTR_tAcc = thr_t2r.partition_S(tAcc_epi)
            tTR_cD = thr_t2r.partition_D(cute.make_identity_tensor((self.TILE_H, 32)))
            rAcc0 = cute.make_rmem_tensor(tTR_cD.shape, Float32)
            rAcc1 = cute.make_rmem_tensor(tTR_cD.shape, Float32)

            c_amax = cute.make_tensor(pAmaxC, cute.make_layout((1,)))[0]
            enc_c, dec_c, mult_c = cute.arch.inline_ptx(
                self.ptx_gscale, write_only_types=[Float32, Float32, Float32], read_only_args=[c_amax]
            )
            keys = None
            off_u = Uint64(0)
            if cutlass.const_expr(self.sr):
                seed_u, off_u = _rng_state(pSeed, pOff, seed_val, offset_val, captured)
                keys = _philox_keys(seed_u)
            half_tokens = m_tokens // 2
            i = Int32(0)
            go = cutlass.Boolean(True)
            while go:
                a = i % A
                par = (i // A) & 1
                cute.arch.mbarrier_wait(tilef_ptr + a, par)
                packed = sTileAcc[a]
                go = packed != END
                if go:
                    m_idx = packed & Int32(0xFFFF)
                    t_idx = packed >> Int32(16)
                    cute.arch.mbarrier_wait(accf_ptr + a, par)
                    cute.copy(tiled_t2r, tTR_tAcc[(None, None, None, 0, 2 * g, a)], rAcc0)
                    cute.copy(tiled_t2r, tTR_tAcc[(None, None, None, 0, 2 * g + 1, a)], rAcc1)
                    cute.arch.fence_view_async_tmem_load()
                    cute.arch.sync_warp()
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(acce_ptr + a)
                    hrow = m_idx * self.TILE_H + tq
                    seq = Uint64(0)
                    if cutlass.const_expr(self.sr):
                        lin = m_idx + (t_idx // kmax) * tiles_m
                        seq = (Int32(128) + tq + (t_idx % kmax) * Int32(512) + lin * kmax * Int32(512)).to(Uint64)
                    outs = []
                    sfs = []
                    for v in cutlass.range_constexpr(4):
                        src = rAcc0 if v < 2 else rAcc1
                        base = 16 * (v % 2)
                        words = []
                        for j in cutlass.range_constexpr(8):
                            words.append(
                                cute.arch.inline_ptx(
                                    _PTX_CVT_BF16X2, write_only_types=[Uint32],
                                    read_only_args=[src[base + 2 * j + 1], src[base + 2 * j]],
                                )
                            )
                        amax = cute.arch.inline_ptx(self.ptx_amax, write_only_types=[Float32], read_only_args=words)
                        sf, acc_scale = cute.arch.inline_ptx(
                            self.ptx_blockscale, write_only_types=[Uint32, Float32], read_only_args=[amax, mult_c, dec_c]
                        )
                        if cutlass.const_expr(self.sr):
                            r0, r1, r2, r3 = _philox4(off_u + (Int32(4) * g + Int32(v)).to(Uint64), seq, keys)
                            o0, o1 = cute.arch.inline_ptx(
                                self.ptx_quant, write_only_types=[Uint32, Uint32],
                                read_only_args=words + [acc_scale, r0, r1, r2, r3],
                            )
                        else:
                            o0, o1 = cute.arch.inline_ptx(
                                self.ptx_quant, write_only_types=[Uint32, Uint32], read_only_args=words + [acc_scale]
                            )
                        outs.append(o0)
                        outs.append(o1)
                        sfs.append(sf)
                    d_addr = d_base + Int64(hrow) * Int64(half_tokens) + Int64(t_idx * Int32(64) + g * Int32(32))
                    cute.arch.inline_ptx(_PTX_ST_V8, read_only_args=[d_addr] + outs)
                    sf_word = sfs[0] | (sfs[1] << Uint32(8)) | (sfs[2] << Uint32(16)) | (sfs[3] << Uint32(24))
                    if cutlass.const_expr(self.swizzled_sf):
                        buf = (g * Int32(2) + (i & Int32(1))) * Int32(128)
                        sSF[buf + (tq % 32) * Int32(4) + tq // 32] = sf_word
                        if g == 0:
                            cute.arch.barrier(barrier_id=3, number_of_threads=128)
                        else:
                            cute.arch.barrier(barrier_id=4, number_of_threads=128)
                        if wq == 0:
                            lane = tq % 32
                            if lane < 16:
                                words = [sSF[buf + lane * Int32(8) + j] for j in range(8)]
                                sf_addr = (
                                    sfd_base
                                    + (Int64(m_idx) * Int64(m_tokens // 64) + Int64(t_idx * Int32(2) + g)) * Int64(512)
                                    + Int64(lane * Int32(32))
                                )
                                cute.arch.inline_ptx(_PTX_ST_V8, read_only_args=[sf_addr] + words)
                    else:
                        sf_addr = sfd_base + Int64(hrow) * Int64(m_tokens // 16) + Int64(t_idx * Int32(8) + g * Int32(4))
                        cute.arch.inline_ptx(_PTX_ST_B32, read_only_args=[sf_addr, sf_word])
                i += Int32(1)
            done_barrier.arrive_and_wait()

        else:
            tid_r = tidx - Int32(32 * len(self.COL_WARPS))
            hf = cute.arch.make_warp_uniform(tid_r // 128)
            t_loc = tid_r % 128
            wr = cute.arch.make_warp_uniform(t_loc // 32)
            atom_s2r = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=128)
            tiled_s2r = cute.make_tiled_copy_tv(
                atom_s2r, cute.make_layout((2, 128), stride=(128, 1)), cute.make_layout((64, 1), stride=(1, 64)),
            )
            thr_s2r = tiled_s2r.get_slice(tid_r)
            tXsX = thr_s2r.partition_S(sA_mks)
            n_lead = cute.rank(tXsX) - 1
            rX = cute.make_rmem_tensor(tXsX[(None,) * n_lead + (0,)].shape, cutlass.BFloat16)
            rX32 = cute.recast_tensor(rX, Uint32)

            a_amax = cute.make_tensor(pAmaxA, cute.make_layout((1,)))[0]
            enc_a, dec_a, mult_a = cute.arch.inline_ptx(
                self.ptx_gscale, write_only_types=[Float32, Float32, Float32], read_only_args=[a_amax]
            )
            keys = None
            off_u = Uint64(0)
            if cutlass.const_expr(self.sr):
                seed_u, off_u = _rng_state(pSeed, pOff, seed_val, offset_val, captured)
                keys = _philox_keys(seed_u)
            half_hidden = n_hidden // 2
            i = Int32(0)
            go = cutlass.Boolean(True)
            while go:
                s = i % S
                par = (i // S) & 1
                cute.arch.mbarrier_wait(full_ptr + s, par)
                packed = sTile[s]
                go = packed != END
                if go:
                    m_idx = packed & Int32(0xFFFF)
                    t_idx = packed >> Int32(16)
                    cute.copy(tiled_s2r, tXsX[(None,) * n_lead + (s,)], rX)
                    cute.arch.fence_view_async_shared()
                    cute.arch.sync_warp()
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(empty_ptr + s)
                    token = t_idx * self.TILE_T + t_loc
                    seq = Uint64(0)
                    if cutlass.const_expr(self.sr):
                        lin = m_idx + (t_idx // kmax) * tiles_m
                        seq = (
                            Int32(256) + t_loc * Int32(2) + hf + ((t_idx % kmax) + Int32(1)) * Int32(512)
                            + lin * kmax * Int32(512)
                        ).to(Uint64)
                    outs = []
                    sfs = []
                    for v in cutlass.range_constexpr(4):
                        words = [rX32[8 * v + j] for j in range(8)]
                        amax = cute.arch.inline_ptx(self.ptx_amax, write_only_types=[Float32], read_only_args=words)
                        sf, acc_scale = cute.arch.inline_ptx(
                            self.ptx_blockscale, write_only_types=[Uint32, Float32], read_only_args=[amax, mult_a, dec_a]
                        )
                        if cutlass.const_expr(self.sr):
                            r0, r1, r2, r3 = _philox4(off_u + Uint64(v), seq, keys)
                            o0, o1 = cute.arch.inline_ptx(
                                self.ptx_quant, write_only_types=[Uint32, Uint32],
                                read_only_args=words + [acc_scale, r0, r1, r2, r3],
                            )
                        else:
                            o0, o1 = cute.arch.inline_ptx(
                                self.ptx_quant, write_only_types=[Uint32, Uint32], read_only_args=words + [acc_scale]
                            )
                        outs.append(o0)
                        outs.append(o1)
                        sfs.append(sf)
                    qa_addr = qa_base + Int64(token) * Int64(half_hidden) + Int64(m_idx * Int32(64) + hf * Int32(32))
                    cute.arch.inline_ptx(_PTX_ST_V8, read_only_args=[qa_addr] + outs)
                    sf_word = sfs[0] | (sfs[1] << Uint32(8)) | (sfs[2] << Uint32(16)) | (sfs[3] << Uint32(24))
                    if cutlass.const_expr(self.swizzled_sf):
                        buf = ((hf + Int32(2)) * Int32(2) + (i & Int32(1))) * Int32(128)
                        sSF[buf + (t_loc % 32) * Int32(4) + t_loc // 32] = sf_word
                        if hf == 0:
                            cute.arch.barrier(barrier_id=5, number_of_threads=128)
                        else:
                            cute.arch.barrier(barrier_id=6, number_of_threads=128)
                        if wr == 0:
                            lane = t_loc % 32
                            if lane < 16:
                                words = [sSF[buf + lane * Int32(8) + j] for j in range(8)]
                                sf_addr = (
                                    sfa_base
                                    + (Int64(t_idx) * Int64(n_hidden // 64) + Int64(m_idx * Int32(2) + hf)) * Int64(512)
                                    + Int64(lane * Int32(32))
                                )
                                cute.arch.inline_ptx(_PTX_ST_V8, read_only_args=[sf_addr] + words)
                    else:
                        sf_addr = sfa_base + Int64(token) * Int64(n_hidden // 16) + Int64(m_idx * Int32(8) + hf * Int32(4))
                        cute.arch.inline_ptx(_PTX_ST_B32, read_only_args=[sf_addr, sf_word])
                i += Int32(1)


_DEBUG = int(os.environ.get("OPT_RHT_CAST_DEBUG", "0"))


class _State:
    enabled = False
    installed = False
    kernels = {}
    dummy_i64 = {}
    rht_views = {}
    philox_ext = None
    nvte = None
    dummy_f32 = {}
    alpha_provider = None
    n_casts = 0
    n_debug = 0
    orig_quantize_impl = None
    orig_quantize_with_amax = None
    orig_swiglu_bwd = None


_S = _State()


def _rank() -> int:
    return int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))


def _log(msg: str, all_ranks: bool = False) -> None:
    if all_ranks or _rank() == 0:
        sys.stderr.write(f"[opt rht_cast rank {_rank()}] {msg}\n")
        sys.stderr.flush()


def _check_te_version() -> str:
    import transformer_engine as te

    v = getattr(te, "__version__", "unknown")
    if v not in _TE_VERIFIED_VERSIONS:
        raise RuntimeError(
            f"opt rht_cast_nvfp4: Transformer Engine {v} is not a build this kernel was verified bit-exact "
            f"against ({', '.join(_TE_VERIFIED_VERSIONS)}); its NVTE struct / enum mirror and Philox round count "
            "may not match. Verify the kernel against the new build and add the version, or set "
            "OPT_RHT_CAST_DISABLE=1."
        )
    return v


class _NVTEShape(ctypes.Structure):
    _fields_ = [("data", ctypes.c_size_t * 15), ("ndim", ctypes.c_size_t)]


class _NVTEBasicTensor(ctypes.Structure):
    _fields_ = [("data_ptr", ctypes.c_void_p), ("dtype", ctypes.c_int), ("shape", _NVTEShape)]


def _basic_tensor(ptr: int, dtype: int, shape) -> _NVTEBasicTensor:
    bt = _NVTEBasicTensor()
    bt.data_ptr = ctypes.c_void_p(ptr)
    bt.dtype = dtype
    bt.shape.ndim = len(shape)
    for i, d in enumerate(shape):
        bt.shape.data[i] = int(d)
    return bt


class _Nvte:

    NVTE_DELAYED_TENSOR_SCALING = 0
    NVTE_NVFP4_1D_SCALING = 4
    kNVTERowwiseData = 0
    kNVTEAmax = 3
    kNVTEColumnwiseAmax = 6
    kNVTEFloat32 = 4
    kNVTEBFloat16 = 6

    def __init__(self):
        _check_te_version()
        import transformer_engine
        import transformer_engine_torch  # noqa: F401  (loads libtransformer_engine)

        path = os.path.join(os.path.dirname(os.path.abspath(transformer_engine.__file__)), "libtransformer_engine.so")
        if not os.path.exists(path):
            path = None
            with open("/proc/self/maps") as f:
                for line in f:
                    if "libtransformer_engine" in line and ".so" in line:
                        path = line.split()[-1]
                        break
        if path is None:
            raise RuntimeError("libtransformer_engine.so was not found (package dir / process maps)")
        lib = ctypes.CDLL(path)
        lib.nvte_create_tensor.restype = ctypes.c_void_p
        lib.nvte_create_tensor.argtypes = [ctypes.c_int]
        lib.nvte_destroy_tensor.restype = None
        lib.nvte_destroy_tensor.argtypes = [ctypes.c_void_p]
        lib.nvte_set_tensor_param.restype = None
        lib.nvte_set_tensor_param.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.POINTER(_NVTEBasicTensor)]
        lib.nvte_hadamard_transform_amax.restype = None
        lib.nvte_hadamard_transform_amax.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        self.lib = lib
        self.t_in = ctypes.c_void_p(lib.nvte_create_tensor(self.NVTE_DELAYED_TENSOR_SCALING))
        self.t_out = ctypes.c_void_p(lib.nvte_create_tensor(self.NVTE_NVFP4_1D_SCALING))

    def hadamard_amax(self, x2d: torch.Tensor, amax_row: torch.Tensor, amax_col: torch.Tensor, sign_mask_t: int) -> None:
        bt_in = _basic_tensor(x2d.data_ptr(), self.kNVTEBFloat16, tuple(x2d.shape))
        bt_row = _basic_tensor(amax_row.data_ptr(), self.kNVTEFloat32, (1,))
        bt_col = _basic_tensor(amax_col.data_ptr(), self.kNVTEFloat32, (1,))
        self.lib.nvte_set_tensor_param(ctypes.byref(self.t_in), self.kNVTERowwiseData, ctypes.byref(bt_in))
        self.lib.nvte_set_tensor_param(ctypes.byref(self.t_out), self.kNVTEAmax, ctypes.byref(bt_row))
        self.lib.nvte_set_tensor_param(ctypes.byref(self.t_out), self.kNVTEColumnwiseAmax, ctypes.byref(bt_col))
        stream = torch.cuda.current_stream().cuda_stream
        self.lib.nvte_hadamard_transform_amax(self.t_in, self.t_out, 0, int(sign_mask_t), ctypes.c_void_p(stream))


def _nvte() -> _Nvte:
    if _S.nvte is None:
        _S.nvte = _Nvte()
    return _S.nvte


def _philox_ext():
    if _S.philox_ext is None:
        import tempfile
        from torch.utils.cpp_extension import load

        src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_philox_state.cpp")
        tag = torch.__version__.replace("+", "_").replace(".", "_")
        build_dir = os.path.join(tempfile.gettempdir(), f"opt_philox_state_{os.getuid()}_{tag}")
        os.makedirs(build_dir, exist_ok=True)
        _S.philox_ext = load(
            name="opt_philox_state", sources=[src], extra_cflags=["-O2"], with_cuda=True,
            build_directory=build_dir, verbose=False,
        )
    return _S.philox_ext


def _dummy_f32(dev: torch.device) -> torch.Tensor:
    k = dev.index
    if k not in _S.dummy_f32:
        _S.dummy_f32[k] = torch.zeros(8, dtype=torch.float32, device=dev)
    return _S.dummy_f32[k]


def _dummy_i64(dev: torch.device) -> torch.Tensor:
    k = dev.index
    if k not in _S.dummy_i64:
        _S.dummy_i64[k] = torch.zeros(2, dtype=torch.int64, device=dev)
    return _S.dummy_i64[k]


def _rht_view(rht_matrix: torch.Tensor) -> torch.Tensor:
    k = (rht_matrix.data_ptr(), rht_matrix.device.index)
    v = _S.rht_views.get(k)
    if v is None:
        v = rht_matrix.detach().t().unsqueeze(-1)
        _S.rht_views[k] = v
    return v


class _CompiledCast:
    def __init__(self, sr: bool, swizzled: bool):
        self.kernel = RhtCastNvfp4Kernel(sr, swizzled)
        self.compiled = None

    @staticmethod
    def _args(x2d, rht_view, out, amax_a, amax_c, alpha_spec, seed_ptr, off_ptr, seed_val, off_val, captured):
        M, N = x2d.shape
        gmem = cute.AddressSpace.gmem
        return (
            from_dlpack(torch.as_strided(x2d, (N, M, 1), (1, N, N * M)), assumed_align=16).mark_layout_dynamic(leading_dim=0),
            from_dlpack(rht_view, assumed_align=16),
            cutlass.Int64(out._columnwise_data.data_ptr()),
            cutlass.Int64(out._columnwise_scale_inv.data_ptr()),
            cutlass.Int64(out._rowwise_data.data_ptr()),
            cutlass.Int64(out._rowwise_scale_inv.data_ptr()),
            make_ptr(Float32, amax_a.data_ptr(), gmem, assumed_align=4),
            make_ptr(Float32, amax_c.data_ptr(), gmem, assumed_align=4),
            make_ptr(Float32, alpha_spec[0], gmem, assumed_align=4),
            make_ptr(Float32, alpha_spec[1], gmem, assumed_align=4),
            make_ptr(Float32, alpha_spec[2], gmem, assumed_align=4),
            make_ptr(Float32, alpha_spec[3], gmem, assumed_align=4),
            cutlass.Int32(alpha_spec[4]),
            make_ptr(cutlass.Int64, seed_ptr, gmem, assumed_align=8),
            make_ptr(cutlass.Int64, off_ptr, gmem, assumed_align=8),
            cutlass.Int64(seed_val),
            cutlass.Int64(off_val),
            cutlass.Int32(captured),
            cutlass.Int32(N),
            cutlass.Int32(M),
        )

    def run(self, x2d, rht_view, out, amax_a, amax_c, alpha_spec, seed_ptr, off_ptr, seed_val, off_val, captured):
        args = self._args(x2d, rht_view, out, amax_a, amax_c, alpha_spec, seed_ptr, off_ptr, seed_val, off_val, captured)
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        if self.compiled is None:
            self.compiled = cute.compile(self.kernel, *args, stream)
        self.compiled(*args, stream)


def _get_kernel(sr: bool, swizzled: bool) -> _CompiledCast:
    key = (bool(sr), bool(swizzled))
    k = _S.kernels.get(key)
    if k is None:
        k = _CompiledCast(bool(sr), bool(swizzled))
        _S.kernels[key] = k
    return k


def _eligible(q, tensor: torch.Tensor) -> bool:
    try:
        from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer
    except Exception:  # pragma: no cover
        return False
    if not isinstance(q, NVFP4Quantizer):
        return False
    if not (q.with_rht and q.with_post_rht_amax and q.rowwise_usage and q.columnwise_usage):
        return False
    if q.with_2d_quantization or getattr(q, "row_scaled_nvfp4", False) or getattr(q, "nvfp4_use_4over6", False):
        return False
    if getattr(q, "with_amax_reduction", False) or getattr(q, "nvfp4_e4m3_max", 448) != 448:
        return False
    if os.environ.get("NVTE_USE_FAST_MATH", "0") not in ("0", "", "false", "False"):
        return False
    if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.bfloat16 or tensor.dim() < 2:
        return False
    n = tensor.shape[-1]
    m = tensor.numel() // n if n else 0
    return m % 128 == 0 and n % 128 == 0 and m > 0 and (n // 128) < 65536 and (m // 128) < 65536


def rht_quantize(tensor: torch.Tensor, quantizer, amax_rowwise: Optional[torch.Tensor] = None,
                 amax_columnwise: Optional[torch.Tensor] = None):
    x = tensor.contiguous()
    n = x.shape[-1]
    m = x.numel() // n
    x2d = x.view(m, n)
    out = quantizer.make_empty(tuple(x.shape), dtype=x.dtype, device=x.device)
    if amax_rowwise is not None:
        out._amax_rowwise = amax_rowwise
        out._amax_columnwise = amax_columnwise
    else:
        _nvte().hadamard_amax(x2d, out._amax_rowwise, out._amax_columnwise, quantizer.rht_matrix_random_sign_mask_t)
    dummy = _dummy_i64(x.device)
    seed_ptr = dummy.data_ptr()
    off_ptr = dummy.data_ptr() + 8
    seed_val = 0
    off_val = 0
    captured = 0
    if quantizer.stochastic_rounding:
        st = _philox_ext().philox_cuda_state(_RNG_ELTS_PER_THREAD)
        captured = int(st[0])
        if captured:
            seed_ptr, off_ptr, off_val = int(st[1]), int(st[2]), int(st[3])
        else:
            seed_val, off_val = int(st[1]), int(st[2])
    alpha_spec = _S.alpha_provider(quantizer, out) if _S.alpha_provider is not None else None
    if alpha_spec is None:
        df = _dummy_f32(x.device).data_ptr()
        alpha_spec = (df, df, df + 16, df + 20, 0)
    kern = _get_kernel(quantizer.stochastic_rounding, bool(out._with_gemm_swizzled_scales))
    kern.run(x2d, _rht_view(quantizer.rht_matrix), out, out._amax_rowwise, out._amax_columnwise, alpha_spec,
             seed_ptr, off_ptr, seed_val, off_val, captured)
    _S.n_casts += 1
    return out


def _cmp(name, a, b) -> str:
    if a is None or b is None:
        return f"{name}={'both-none' if a is b else 'ONE-NONE'}"
    if a.shape != b.shape:
        return f"{name}=SHAPE{tuple(a.shape)}vs{tuple(b.shape)}"
    nbad = int((a.view(torch.uint8) != b.view(torch.uint8)).sum().item()) if a.dtype == torch.uint8 else int((a != b).sum().item())
    return f"{name}={'ok' if nbad == 0 else 'BAD:' + str(nbad)}"


def _compare_out(out, ref) -> list:
    return [
        _cmp("rowwise_data", out._rowwise_data, ref._rowwise_data),
        _cmp("rowwise_scale_inv", out._rowwise_scale_inv, ref._rowwise_scale_inv),
        _cmp("columnwise_data", out._columnwise_data, ref._columnwise_data),
        _cmp("columnwise_scale_inv", out._columnwise_scale_inv, ref._columnwise_scale_inv),
        _cmp("amax_rowwise", out._amax_rowwise, ref._amax_rowwise),
        _cmp("amax_columnwise", out._amax_columnwise, ref._amax_columnwise),
    ]


def _debug_run(tag: str, shape, ref_fn, our_fn):
    _S.n_debug += 1
    dev = torch.cuda.current_device()
    state = torch.cuda.get_rng_state(dev)
    ref = ref_fn()
    torch.cuda.set_rng_state(state, dev)
    out = our_fn()
    torch.cuda.synchronize()
    parts = _compare_out(out, ref)
    _log(f"DEBUG {tag} shape={tuple(shape)} swizzled={out._with_gemm_swizzled_scales} " + " ".join(parts), all_ranks=True)
    return out


def _debugging() -> bool:
    return _DEBUG > 0 and _S.n_debug < _DEBUG and not torch.cuda.is_current_stream_capturing()


def _patch() -> None:
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer
    from transformer_engine.pytorch.ops.basic import swiglu as te_swiglu

    orig_impl = NVFP4Quantizer.quantize_impl
    _S.orig_quantize_impl = orig_impl

    def quantize_impl(self, tensor):
        if _S.enabled and _eligible(self, tensor):
            if _debugging():
                return _debug_run(f"quantize_impl sr={self.stochastic_rounding}", tensor.shape,
                                  lambda: orig_impl(self, tensor), lambda: rht_quantize(tensor, self))
            return rht_quantize(tensor, self)
        return orig_impl(self, tensor)

    NVFP4Quantizer.quantize_impl = quantize_impl

    orig_qwa = tex.nvfp4_quantize_with_amax
    _S.orig_quantize_with_amax = orig_qwa

    def nvfp4_quantize_with_amax(tensor, quantizer, rowwise_amax, columnwise_amax):
        if _S.enabled and _eligible(quantizer, tensor):
            if _debugging():
                return _debug_run(f"quantize_with_amax sr={quantizer.stochastic_rounding}", tensor.shape,
                                  lambda: orig_qwa(tensor, quantizer, rowwise_amax, columnwise_amax),
                                  lambda: rht_quantize(tensor, quantizer, rowwise_amax, columnwise_amax))
            return rht_quantize(tensor, quantizer, rowwise_amax, columnwise_amax)
        return orig_qwa(tensor, quantizer, rowwise_amax, columnwise_amax)

    tex.nvfp4_quantize_with_amax = nvfp4_quantize_with_amax

    orig_bwd = te_swiglu.SwiGLU._tex_swiglu_backward
    _S.orig_swiglu_bwd = orig_bwd

    def _tex_swiglu_backward(self, grad_output, input_, quantizer):
        if _S.enabled and quantizer is not None and _eligible(quantizer, input_):
            def ours():
                return rht_quantize(orig_bwd(self, grad_output, input_, None), quantizer)
            if _debugging():
                return _debug_run(f"dswiglu sr={quantizer.stochastic_rounding}", input_.shape,
                                  lambda: orig_bwd(self, grad_output, input_, quantizer), ours)
            return ours()
        return orig_bwd(self, grad_output, input_, quantizer)

    te_swiglu.SwiGLU._tex_swiglu_backward = _tex_swiglu_backward


def install() -> bool:
    global _ARCH
    if _S.installed:
        return _S.enabled
    _S.installed = True
    if os.environ.get("OPT_RHT_CAST_DISABLE", "0") == "1":
        _log("disabled by OPT_RHT_CAST_DISABLE=1")
        return False
    from opt_kernels import arch as _arch
    if not _arch.is_rubin():
        if os.environ.get("OPT_RHT_CAST_ALLOW_BLACKWELL", "0") == "1":
            _ARCH = _arch.sm()
            _log(f"non-Rubin device {_arch.device_name()}: compiling the RHT cast for {_ARCH} "
                 "(OPT_RHT_CAST_ALLOW_BLACKWELL=1; untuned on this device)")
        elif _arch.rubin_only("RHT + NVFP4 cast kernel (CuTe DSL, written for sm_107; "
                              "OPT_RHT_CAST_ALLOW_BLACKWELL=1 tries it anyway)"):
            return False
    v = _check_te_version()
    _patch()
    _S.enabled = True
    _log(f"installed (TE {v}, debug={_DEBUG})")
    return True
