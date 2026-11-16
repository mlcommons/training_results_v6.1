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

"""Fused qkv projection for gpt-oss-20b attention on Rubin (sm_107): MXFP8 block-scaled GEMM + bias + RoPE + amax.

Replaces cuBLAS MXFP8 GEMM (+bias) -> bf16 -> gpt_oss::qkv_rope_fwd_amax (RoPE(q, k) + relocation into one flat q|k|v
buffer + per-tensor amax/scale) with one persistent warp-specialized tcgen05 block-scaled GEMM whose epilogue does the
rest; TE's cast_fp8_1D still follows (the global amax must precede the cast).  Mainloop, pipelines and scheduler are
the stock CUTLASS 4.8 Sm107BlockScaledPersistentDenseGemmKernel (subclassed; its TMA-load and MMA warps are copied
verbatim into kernel_impl); the epilogue warps and the extra host operands are new.
GEMM: C[M, N] = A[M, K] x B[N, K]^T on MXFP8 e4m3 K-major operands with GEMM-swizzled e8m0 scales (TE's
_rowwise_scale_inv as-is), fp32 accumulate; 2-CTA 256x256x64 kind::mxf8f6f4 MMA, 256x256x128 tiler, static persistent
scheduler, 8 A/B stages.  Epilogue: 4 warps (128 threads = the CTA's rows), one 64-wide head per tcgen05.ld: the bias
is added in fp32 (cuBLAS rounds acc + bias to bf16 before RoPE, so outputs are not bitwise equal), TE non-interleaved
RoPE on the q/k heads (cos/sin are [s, 32] since cos[d] == cos[d+32]), v passes through; bf16 head subtiles are staged
in smem (2 stages) and TMA-stored via q / k / v descriptors; amax is reduced warp -> CTA -> one atomic per CTA, and
the last CTA writes scale = 448 / amax (TE compute_scale_from_amax) and resets the workspace.
"""
import os
import sys
from typing import Optional, Tuple, Type, Union

import torch

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass import Int32, Float32, BFloat16
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.cute.nvgpu import cpasync, tcgen05, OperandMajorMode
from cutlass.cute.nvgpu.tcgen05.mma import CollectorOp
from cutlass.cute.runtime import make_ptr
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.rubin_helpers as sm107_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.utils.gemm.sm100 import (
    transform_partitioned_tensor_layout,
    epilogue_tmem_copy_and_partition,
    epilogue_smem_copy_and_partition,
)

# The stock Rubin block-scaled GEMM example this kernel subclasses (CUTLASS 4.8.0, TE's third-party CUTLASS checkout).
EXAMPLES_DIR = os.environ.get(
    "QKV_GEMM_CUTLASS_EXAMPLES",
    "/workspace/transformerengine/3rdparty/cutlass/examples/python/CuTeDSL/cute",
)
if EXAMPLES_DIR not in sys.path:
    sys.path.insert(0, EXAMPLES_DIR)
from rubin.kernel.blockscaled_gemm.dense_blockscaled_gemm_persistent import (  # noqa: E402
    Sm107BlockScaledPersistentDenseGemmKernel,
)

HEAD_D = 64
HALF = 32
SF_VEC = 32
FP8_E4M3_MAX = 448.0
FLT_MAX = 3.4028234663852886e38
F32_INF = float("inf")

CLUSTER_M = int(os.environ.get("QKV_GEMM_CLUSTER_M", "2"))
CLUSTER_N = int(os.environ.get("QKV_GEMM_CLUSTER_N", "1"))
PREFETCH_DIST = int(os.environ.get("QKV_GEMM_PREFETCH", "0"))   # TMA prefetch distance in K tiles; 0 = off (the stock default of num_ab_stage costs 1.6x here)
STORE_MODE = os.environ.get("QKV_GEMM_STORE", "tma")             # tma: smem staging + 3 TMA store descriptors; direct: st.global from registers (198 vs 144 us)
EPI_WARPS = int(os.environ.get("QKV_GEMM_EPI_WARPS", "4"))         # 4: stock warp layout; 8: two epilogue warpgroups splitting the heads (165-181 vs 135-147 us)
SWIZZLE_SIZE = int(os.environ.get("QKV_GEMM_SWIZZLE", "1"))
RASTER_ORDER = os.environ.get("QKV_GEMM_RASTER", "m")
DIAG_SKIP_EPI = os.environ.get("QKV_GEMM_DIAG_SKIP_EPI", "0") == "1"   # DIAGNOSTIC ONLY: mainloop-only timing
# Development-only knobs (slower, or wrong outputs) need an explicit QKV_GEMM_DEV=1: otherwise the import fails, and
# qkv_rope_hook._install_fused_gemm re-raises instead of falling back to cuBLAS, so the run stops at startup.
_DEV_KNOBS = {}
if STORE_MODE != "tma":
    _DEV_KNOBS["QKV_GEMM_STORE"] = STORE_MODE
if EPI_WARPS != 4:
    _DEV_KNOBS["QKV_GEMM_EPI_WARPS"] = EPI_WARPS
if DIAG_SKIP_EPI:
    _DEV_KNOBS["QKV_GEMM_DIAG_SKIP_EPI"] = 1
if _DEV_KNOBS and os.environ.get("QKV_GEMM_DEV", "0") != "1":
    raise RuntimeError(
        f"QKV_GEMM: development-only knob(s) {_DEV_KNOBS} set without QKV_GEMM_DEV=1 "
        "(QKV_GEMM_STORE=direct / QKV_GEMM_EPI_WARPS=8 are measured slower, "
        "QKV_GEMM_DIAG_SKIP_EPI=1 produces wrong outputs); unset them for a training run"
    )
MMA_TILER = (256, 256, 128)
MMA_INST = (256, 256, 64)


def _say(msg: str) -> None:
    rank = os.getenv("RANK", os.getenv("SLURM_PROCID", "0"))
    sys.__stdout__.write(f"QKV_GEMM rank={rank} {msg}\n")
    sys.__stdout__.flush()


class QkvGemmRopeKernel(Sm107BlockScaledPersistentDenseGemmKernel):
    """Stock Rubin block-scaled persistent GEMM + fused bias/RoPE/relocation/amax epilogue."""

    def __init__(
        self,
        ng: int,
        nqpg: int,
        mma_tiler: Tuple[int, int, int] = MMA_TILER,
        mma_inst_shape: Tuple[int, int, int] = MMA_INST,
        cluster_shape_mn: Tuple[int, int] = (CLUSTER_M, CLUSTER_N),
        swizzle_size: int = SWIZZLE_SIZE,
        raster_order: str = RASTER_ORDER,
        prefetch_dist: Optional[int] = PREFETCH_DIST,
        store_mode: str = STORE_MODE,
        epi_warps: int = EPI_WARPS,
    ):
        super().__init__(
            SF_VEC, tuple(mma_inst_shape), tuple(mma_tiler), tuple(cluster_shape_mn),
            prefetch_dist, swizzle_size, raster_order, None,
        )
        if store_mode not in ("tma", "direct"):
            raise ValueError(f"QkvGemmRopeKernel: store_mode must be tma or direct, got {store_mode}")
        self.tma_store = store_mode == "tma"
        if epi_warps not in (4, 8):
            raise ValueError(f"QkvGemmRopeKernel: epi_warps must be 4 or 8, got {epi_warps}")
        self.num_epi_groups = epi_warps // 4
        if epi_warps == 8:
            # Warp roles: 0..7 epilogue (two warpgroups), 8 mma, 9 tma, 10 scheduler (CLC only, unused here).
            # The stock pipelines size themselves from these ids (acc consumer count, TMEM allocation barrier).
            self.epilog_warp_id = tuple(range(8))
            self.epilogue_warp_id = self.epilog_warp_id
            self.mma_warp_id = 8
            self.tma_warp_id = 9
            self.sched_warp_id = 10
            self.threads_per_cta = self.threads_per_warp * len((self.mma_warp_id, self.tma_warp_id, *self.epilog_warp_id))
            self.tmem_alloc_barrier = pipeline.NamedBarrier(
                barrier_id=2, num_threads=self.threads_per_warp * len((self.mma_warp_id, *self.epilog_warp_id))
            )
        # per-warpgroup epilogue barriers (128 threads each) and one across all epilogue warps
        self.epi_group_barriers = [
            pipeline.NamedBarrier(barrier_id=1 + 3 * g, num_threads=self.threads_per_warp * 4) for g in range(self.num_epi_groups)
        ]
        self.epilog_all_barrier = pipeline.NamedBarrier(
            barrier_id=5, num_threads=self.threads_per_warp * len(self.epilog_warp_id)
        )
        if self.enable_breuse:
            raise ValueError("QkvGemmRopeKernel: the fused epilogue assumes one MMA along M (no B-reuse)")
        if mma_tiler[1] % HEAD_D != 0:
            raise ValueError("QkvGemmRopeKernel: the N tile must hold whole 64-wide heads")
        self.ng = int(ng)
        self.nqpg = int(nqpg)
        self.n_cols = self.ng * (self.nqpg + 2) * HEAD_D      # N, static: sizes the bias staging in smem
        self.heads_per_tile = mma_tiler[1] // HEAD_D
        self.diag_skip_epi = DIAG_SKIP_EPI
        self.a_major_mode = OperandMajorMode.K
        self.b_major_mode = OperandMajorMode.K
        self.c_layout = utils.LayoutEnum.ROW_MAJOR
        self.c_dtype = BFloat16

    def _setup_attributes(self):
        super()._setup_attributes()
        # one epilogue subtile per 64-wide head: a single tcgen05.ld (32x32b x64) hands the thread its whole head row
        self.epi_tile = (cute.make_layout(self.cta_tile_shape_mnk[0]), cute.make_layout(HEAD_D))
        self.epi_tile_n = HEAD_D
        # smem staging of one head subtile (128 x 64 bf16 = 16 KB) per stage for the TMA stores
        self.c_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.c_dtype, self.c_layout, self.epi_tile, self.num_c_stage
        )

    @cute.jit
    def _epi_group_barrier(self, epi_group):
        """Named barrier of this thread's epilogue warpgroup (epi_group is warp-uniform)."""
        if cutlass.const_expr(self.num_epi_groups == 1):
            self.epi_group_barriers[0].arrive_and_wait()
        else:
            if epi_group == 0:
                self.epi_group_barriers[0].arrive_and_wait()
            else:
                self.epi_group_barriers[1].arrive_and_wait()

    # ------------------------------------------------------------------ stages: 2 C stages (TMA-store staging) + bias; the rest A/B
    @staticmethod
    def _compute_stages(
        tiled_mma, mma_tiler_mnk, a_dtype, b_dtype, epi_tile, c_dtype, c_layout, sf_dtype, sf_vec_size,
        smem_capacity, occupancy, with_breuse,
    ):
        num_acc_stage = 2
        num_c_stage = 2  # 128x64 bf16 head subtiles in flight for the TMA stores; 2 keeps 8 A/B stages (3: no gain, 4: costs a stage)
        a_one = sm100_utils.make_smem_layout_a(tiled_mma, mma_tiler_mnk, a_dtype, 1)
        b_one = sm100_utils.make_smem_layout_b(tiled_mma, mma_tiler_mnk, b_dtype, 1)
        sfa_one = blockscaled_utils.make_smem_layout_sfa(tiled_mma, mma_tiler_mnk, sf_vec_size, 1)
        sfb_one = blockscaled_utils.make_smem_layout_sfb(tiled_mma, mma_tiler_mnk, sf_vec_size, 1)
        ab_bytes_per_stage = (
            cute.size_in_bytes(a_dtype, a_one)
            + cute.size_in_bytes(b_dtype, b_one)
            + cute.size_in_bytes(sf_dtype, sfa_one)
            + cute.size_in_bytes(sf_dtype, sfb_one)
        )
        mbar_helpers_bytes = 1024
        c_bytes = num_c_stage * 128 * HEAD_D * (c_dtype.width // 8) + 1024  # + the bias staging
        num_ab_stage = (smem_capacity // occupancy - mbar_helpers_bytes - c_bytes) // ab_bytes_per_stage
        return num_acc_stage, num_ab_stage, num_c_stage

    # ------------------------------------------------------------------ host side
    @cute.jit
    def __call__(
        self,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer,
        sfb_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
        bias_ptr: cute.Pointer,
        cos_ptr: cute.Pointer,
        sin_ptr: cute.Pointer,
        scale_ptr: cute.Pointer,
        amax_ptr: cute.Pointer,
        wsf_ptr: cute.Pointer,
        wsi_ptr: cute.Pointer,
        problem_mnkl: Tuple[int, int, int, int],
        batch: Int32,
        seq: Int32,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        self.a_dtype: Type[cutlass.Numeric] = a_ptr.value_type
        self.b_dtype: Type[cutlass.Numeric] = b_ptr.value_type
        self.sf_dtype: Type[cutlass.Numeric] = sfa_ptr.value_type
        self.c_dtype: Type[cutlass.Numeric] = out_ptr.value_type
        m, n, k, l = problem_mnkl
        self._setup_attributes()

        # A [M, K] K-major, B [N, K] K-major (TE MXFP8 rowwise data of the LN output and of the weight)
        a_layout = cute.make_ordered_layout((cute.assume(m, 32), k, l), order=(1, 0, 2))
        b_layout = cute.make_ordered_layout((cute.assume(n, 32), k, l), order=(1, 0, 2))
        # the output viewed as the row-major [M, N] matrix of the GEMM: used for the tile count and the coordinate tensor
        c_layout = cute.make_ordered_layout((m, cute.assume(n, 32), l), order=(1, 0, 2))
        a_tensor = cute.make_tensor(a_ptr, a_layout)
        b_tensor = cute.make_tensor(b_ptr, b_layout)
        c_tensor = cute.make_tensor(out_ptr, c_layout)
        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(a_tensor.shape, self.sf_vec_size)
        sfa_tensor = cute.make_tensor(sfa_ptr, sfa_layout)
        sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(b_tensor.shape, self.sf_vec_size)
        sfb_tensor = cute.make_tensor(sfb_ptr, sfb_layout)

        # the epilogue operands: 32-element vectors (one half head of one row; one cos/sin row; one bias half head)
        n_vec = cute.assume(m * n // HALF, 32)
        out32 = cute.make_tensor(out_ptr, cute.make_layout((HALF, n_vec), stride=(1, HALF)))
        bias32 = cute.make_tensor(bias_ptr, cute.make_layout((HALF, n // HALF), stride=(1, HALF)))
        cos32 = cute.make_tensor(cos_ptr, cute.make_layout((HALF, seq), stride=(1, HALF)))
        sin32 = cute.make_tensor(sin_ptr, cute.make_layout((HALF, seq), stride=(1, HALF)))
        scale_t = cute.make_tensor(scale_ptr, cute.make_layout((1,)))
        amax_t = cute.make_tensor(amax_ptr, cute.make_layout((1,)))
        wsf_t = cute.make_tensor(wsf_ptr, cute.make_layout((1,)))
        wsi_t = cute.make_tensor(wsi_ptr, cute.make_layout((1,)))

        atom_layout_mnk = (1, 1, 1)
        permutation_mnk = self._get_mma_permutation_mnk()
        tiled_mma = sm107_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype, self.b_dtype, self.a_major_mode, self.b_major_mode, self.sf_dtype, self.sf_vec_size,
            self.cta_group, self.mma_inst_shape,
            a_collector_op=CollectorOp.DISCARD, b_collector_op=CollectorOp.DISCARD,
            atom_layout_mnk=atom_layout_mnk, permutation_mnk=permutation_mnk,
        )
        tiled_mma.set(tcgen05.Field.NEGATE_A, False)
        tiled_mma.set(tcgen05.Field.NEGATE_B, False)
        tiled_mma_sfb = sm107_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype, self.b_dtype, self.a_major_mode, self.b_major_mode, self.sf_dtype, self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE, self.mma_inst_shape_sfb,
            a_collector_op=CollectorOp.DISCARD, b_collector_op=CollectorOp.DISCARD,
        )
        tiled_mma_sfb.set(tcgen05.Field.NEGATE_A, False)
        tiled_mma_sfb.set(tcgen05.Field.NEGATE_B, False)
        tiled_mma_bkeep = None
        tiled_mma_breuse = None

        atom_thr_size = cute.size(tiled_mma.thr_id.shape)
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        a_smem_layout = cute.select(self.a_smem_layout_staged, mode=list(range(cute.rank(self.a_smem_layout_staged) - 1)))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op, a_tensor, a_smem_layout, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape,
        )
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id)
        b_smem_layout = cute.select(self.b_smem_layout_staged, mode=list(range(cute.rank(self.b_smem_layout_staged) - 1)))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op, b_tensor, b_smem_layout, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape,
        )
        sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        sfa_smem_layout = cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0))
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            sfa_op, sfa_tensor, sfa_smem_layout, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(self.cluster_shape_mn, tiled_mma.thr_id)
        sfb_smem_layout = cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0))
        tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op, sfb_tensor, sfb_smem_layout, self.mma_tiler_sfb, tiled_mma_sfb, self.cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        sfa_copy_size = cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
        sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        self.num_tma_load_bytes = (a_copy_size + b_copy_size + sfa_copy_size + sfb_copy_size) * atom_thr_size

        # the three destinations of the flat q|k|v buffer as row-major (rows, cols, 1) matrices for the TMA stores
        nqh = self.ng * self.nqpg
        nq_elems = m * (nqh * HEAD_D)
        nk_elems = m * (self.ng * HEAD_D)
        q_tensor = cute.make_tensor(out_ptr, cute.make_ordered_layout((m, nqh * HEAD_D, l), order=(1, 0, 2)))
        k_tensor = cute.make_tensor(out_ptr + nq_elems, cute.make_ordered_layout((m, self.ng * HEAD_D, l), order=(1, 0, 2)))
        v_tensor = cute.make_tensor(out_ptr + (nq_elems + nk_elems), cute.make_ordered_layout((m, self.ng * HEAD_D, l), order=(1, 0, 2)))
        epi_smem_layout = cute.select(
            self.c_smem_layout_staged, mode=list(range(cute.rank(self.c_smem_layout_staged) - 1))
        )
        tma_atom_q, tma_tensor_q = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), q_tensor, epi_smem_layout, self.epi_tile
        )
        tma_atom_k, tma_tensor_k = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), k_tensor, epi_smem_layout, self.epi_tile
        )
        tma_atom_v, tma_tensor_v = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), v_tensor, epi_smem_layout, self.epi_tile
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
            clc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_clc_stage * 2]
            clc_response: cute.struct.MemRange[cutlass.Int32, self.num_clc_response_bytes // 4]
            sAmax: cute.struct.MemRange[cutlass.Float32, 8]
            sBias: cute.struct.Align[cute.struct.MemRange[self.c_dtype, self.n_cols], 128]
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
            c_tensor, out32, bias32, cos32, sin32, scale_t, amax_t, wsf_t, wsi_t, batch,
            tma_atom_q, tma_tensor_q, tma_atom_k, tma_tensor_k, tma_atom_v, tma_tensor_v,
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

    # ------------------------------------------------------------------ device side
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
        gOut32: cute.Tensor,
        gBias32: cute.Tensor,
        gCos32: cute.Tensor,
        gSin32: cute.Tensor,
        gScale: cute.Tensor,
        gAmax: cute.Tensor,
        gWsF: cute.Tensor,
        gWsI: cute.Tensor,
        batch: Int32,
        tma_atom_q: cute.CopyAtom,
        mQ_mnl: cute.Tensor,
        tma_atom_k: cute.CopyAtom,
        mK_mnl: cute.Tensor,
        tma_atom_v: cute.CopyAtom,
        mV_mnl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        tCtSFA_layout: cute.Layout,
        tCtSFB_layout: cute.Layout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout],
        epi_tile: cute.Tile,
        tile_sched_params,
    ):
        self.kernel_impl(
            tiled_mma, tiled_mma_bkeep, tiled_mma_breuse, tiled_mma_sfb,
            tma_atom_a, mA_mkl, tma_atom_b, mB_nkl, tma_atom_sfa, mSFA_mkl, tma_atom_sfb, mSFB_nkl,
            mC_mnl, gOut32, gBias32, gCos32, gSin32, gScale, gAmax, gWsF, gWsI, batch,
            tma_atom_q, mQ_mnl, tma_atom_k, mK_mnl, tma_atom_v, mV_mnl,
            cluster_layout_vmnk, cluster_layout_sfb_vmnk,
            a_smem_layout_staged, b_smem_layout_staged, sfa_smem_layout_staged, sfb_smem_layout_staged,
            tCtSFA_layout, tCtSFB_layout, c_smem_layout_staged, epi_tile, tile_sched_params,
            self.cluster_shape_mn, self.is_a_mcast, self.is_b_mcast,
        )

    @cute.jit
    def kernel_impl(
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
        gOut32: cute.Tensor,
        gBias32: cute.Tensor,
        gCos32: cute.Tensor,
        gSin32: cute.Tensor,
        gScale: cute.Tensor,
        gAmax: cute.Tensor,
        gWsF: cute.Tensor,
        gWsI: cute.Tensor,
        batch: Int32,
        tma_atom_q: cute.CopyAtom,
        mQ_mnl: cute.Tensor,
        tma_atom_k: cute.CopyAtom,
        mK_mnl: cute.Tensor,
        tma_atom_v: cute.CopyAtom,
        mV_mnl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        tCtSFA_layout: cute.Layout,
        tCtSFB_layout: cute.Layout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout],
        epi_tile: cute.Tile,
        tile_sched_params,
        cluster_shape_mn: Tuple[int, int],
        is_a_mcast: bool,
        is_b_mcast: bool,
    ):
        """Stock prologue + TMA-load warp + MMA warp (Sm107BlockScaledPersistentDenseGemmKernel.kernel_impl, CUTLASS 4.8.0),
        with the C tensor replaced by a coordinate (identity) tensor of the same tiling, followed by the fused epilogue."""
        # the coordinate tensor stands in for C: its partitions give every epilogue thread its (row, column) in [M, N]
        cC_mnl = cute.make_identity_tensor(mC_mnl.shape)
        if cute.arch.make_warp_uniform(cute.arch.warp_idx()) == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_q)
            cpasync.prefetch_descriptor(tma_atom_k)
            cpasync.prefetch_descriptor(tma_atom_v)
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        #
        # Prefetch tma desc
        #
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_sfa)
            cpasync.prefetch_descriptor(tma_atom_sfb)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        #
        # Setup cta/thread coordinates
        #
        # Coords inside cluster
        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        # Coord inside cta
        tidx, _, _ = cute.arch.thread_idx()

        #
        # Alloc and init: a+b full/empty, accumulator full/empty, tensor memory dealloc barrier
        #
        smem = cutlass.memory.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # Initialize mainloop ab_pipeline (barrier) and states
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

        # Initialize acc_pipeline (barrier) and states
        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.epilog_warp_id) * (
            2 if use_2cta_instrs else 1
        )
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_acc_consumer_threads
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # Initialize CLC pipeline for dynamic tile scheduling
        clc_pipeline = None
        clc_response_ptr = None
        clc_consumer_state = None
        if cutlass.const_expr(self.use_clc_dynamic_scheduler):
            clc_pipeline_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread
            )
            cluster_size = cute.size(self.cluster_shape_mn)
            num_clc_consumer_threads = 32 * len(
                (
                    self.sched_warp_id,
                    *(
                        cluster_size
                        * (
                            self.tma_warp_id,
                            self.mma_warp_id,
                            *self.epilog_warp_id,
                        )
                    ),
                )
            )
            clc_pipeline_consumer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread, num_clc_consumer_threads
            )
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
            clc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_clc_stage
            )

        # Tensor memory dealloc barrier init
        tmem = cutlass.memory.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.epilog_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
            arch=self.arch,
        )

        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=cluster_shape_mn, is_relaxed=True)

        #
        # Setup smem tensor A/B/SFA/SFB/C
        #
        # (EPI_TILE_M, EPI_TILE_N, STAGE)
        # (MMA, MMA_M, MMA_K, STAGE)
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        # (MMA, MMA_N, MMA_K, STAGE)
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        # (MMA, MMA_M, MMA_K, STAGE)
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        # (MMA, MMA_N, MMA_K, STAGE)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)

        #
        # Compute multicast mask for A/B/SFA/SFB buffer full
        #
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        sfa_full_mcast_mask = None
        sfb_full_mcast_mask = None
        if cutlass.const_expr(is_a_mcast or is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )
            sfa_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            sfb_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_sfb_vmnk, block_in_cluster_coord_sfb_vmnk, mcast_mode=1
            )

        #
        # Local_tile partition global tensors
        #
        # (bM, bK, RestM, RestK, RestL)
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        # (bN, bK, RestN, RestK, RestL)
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        # (bM, bK, RestM, RestK, RestL)
        gSFA_mkl = cute.local_tile(
            mSFA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        # (bN, bK, RestN, RestK, RestL)
        gSFB_nkl = cute.local_tile(
            mSFB_nkl,
            cute.slice_(self.mma_tiler_sfb, (0, None, None)),
            (None, None, None),
        )
        # (bM, bN, RestM, RestN, RestL)
        gC_mnl = cute.local_tile(
            cC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        #
        # Partition global tensor for TiledMMA_A/B/C
        #
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)
        # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
        tCgA = thr_mma.partition_A(gA_mkl)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        tCgB = thr_mma.partition_B(gB_nkl)
        # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
        tCgSFA = thr_mma.partition_A(gSFA_mkl)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        tCgSFB = thr_mma_sfb.partition_B(gSFB_nkl)
        # (MMA, MMA_M, MMA_N, RestM, RestN, RestL)
        tCgC = thr_mma.partition_C(gC_mnl)

        #
        # Partition global/shared tensor for TMA load A/B
        #
        # TMA load A partition_S/D
        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestM, RestK, RestL)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        # TMA load B partition_S/D
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestN, RestK, RestL)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        # TMA load SFA partition_S/D
        sfa_cta_layout = a_cta_layout
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestM, RestK, RestL)
        tAsSFA, tAgSFA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfa,
            block_in_cluster_coord_vmnk[2],
            sfa_cta_layout,
            cute.group_modes(sSFA, 0, 3),
            cute.group_modes(tCgSFA, 0, 3),
        )
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)

        # TMA load SFB partition_S/D
        sfb_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestN, RestK, RestL)
        tBsSFB, tBgSFB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfb,
            block_in_cluster_coord_sfb_vmnk[1],
            sfb_cta_layout,
            cute.group_modes(sSFB, 0, 3),
            cute.group_modes(tCgSFB, 0, 3),
        )
        tBsSFB = cute.filter_zeros(tBsSFB)
        tBgSFB = cute.filter_zeros(tBgSFB)

        #
        # Partition shared/tensor memory tensor for TiledMMA_A/B/C
        #
        # (MMA, MMA_M, MMA_K, STAGE)
        tCrA = tiled_mma.make_fragment_A(sA)
        # (MMA, MMA_N, MMA_K, STAGE)
        tCrB = tiled_mma.make_fragment_B(sB)
        # (MMA, MMA_M, MMA_N)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        # (MMA, MMA_M, MMA_N, STAGE)
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.num_acc_stage)
        )

        #
        # Cluster wait before tensor memory alloc
        #
        pipeline_init_wait(cluster_shape_mn=cluster_shape_mn)

        #
        # Construct the scheduler
        #
        if cutlass.const_expr(self.use_clc_dynamic_scheduler):
            tile_sched = utils.ClcDynamicPersistentTileScheduler.create(
                tile_sched_params,
                cute.arch.block_idx(),
                cute.arch.grid_dim(),
                clc_response_ptr,
            )
        else:
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
        work_tile = tile_sched.initial_work_tile_info()

        #
        # Specialized TMA load warp
        #
        if warp_idx == self.tma_warp_id:
            #
            # Persistent tile scheduling loop
            #
            ab_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )

            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                #
                # Slice to per mma tile index
                #
                # ((atom_v, rest_v), RestK)
                tAgA_slice = tAgA[
                    (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
                ]
                # ((atom_v, rest_v), RestK)
                tBgB_slice = tBgB[
                    (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
                ]

                # ((atom_v, rest_v), RestK)
                tAgSFA_slice = tAgSFA[
                    (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
                ]

                slice_n = mma_tile_coord_mnl[1]
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                    slice_n = mma_tile_coord_mnl[1] // 2

                # ((atom_v, rest_v), RestK)
                tBgSFB_slice = tBgSFB[(None, slice_n, None, mma_tile_coord_mnl[2])]

                #
                # Prefetch: Initial batch of prefetches to prime the pipeline
                #
                if self.prefetch_enabled:
                    for pf_k_tile in cutlass.range(
                        0, min(self.prefetch_dist, k_tile_cnt), unroll=1
                    ):
                        cute.prefetch(tma_atom_a, tAgA_slice[(None, pf_k_tile)])
                        cute.prefetch(tma_atom_b, tBgB_slice[(None, pf_k_tile)])
                        cute.prefetch(tma_atom_sfa, tAgSFA_slice[(None, pf_k_tile)])
                        cute.prefetch(tma_atom_sfb, tBgSFB_slice[(None, pf_k_tile)])

                # Peek (try_wait) AB buffer empty for k_tile = prefetch_k_tile_cnt
                ab_producer_state.reset_count()
                peek_ab_empty_status = cutlass.Boolean(1)
                if ab_producer_state.count < k_tile_cnt:
                    peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                        ab_producer_state
                    )
                #
                # Tma load loop
                #
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    # Conditionally wait for AB buffer empty
                    ab_pipeline.producer_acquire(
                        ab_producer_state, peek_ab_empty_status
                    )

                    # TMA load A/B/SFA/SFB
                    cute.copy(
                        tma_atom_a,
                        tAgA_slice[(None, ab_producer_state.count)],
                        tAsA[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=a_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_slice[(None, ab_producer_state.count)],
                        tBsB[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=b_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_sfa,
                        tAgSFA_slice[(None, ab_producer_state.count)],
                        tAsSFA[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=sfa_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_sfb,
                        tBgSFB_slice[(None, ab_producer_state.count)],
                        tBsSFB[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=sfb_full_mcast_mask,
                    )

                    # Prefetch: Rolling prefetch for next tiles
                    if self.prefetch_enabled:
                        if k_tile < k_tile_cnt - self.prefetch_dist:
                            future_k_tile = ab_producer_state.count + self.prefetch_dist
                            cute.prefetch(
                                tma_atom_a,
                                tAgA_slice[(None, future_k_tile)],
                            )
                            cute.prefetch(
                                tma_atom_b,
                                tBgB_slice[(None, future_k_tile)],
                            )
                            cute.prefetch(
                                tma_atom_sfa,
                                tAgSFA_slice[(None, future_k_tile)],
                            )
                            cute.prefetch(
                                tma_atom_sfb,
                                tBgSFB_slice[(None, future_k_tile)],
                            )

                    # Peek (try_wait) AB buffer empty for k_tile = prefetch_k_tile_cnt + k_tile + 1
                    ab_producer_state.advance()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if ab_producer_state.count < k_tile_cnt:
                        peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                            ab_producer_state
                        )

                #
                # Advance to next tile
                #
                if cutlass.const_expr(self.use_clc_dynamic_scheduler):
                    clc_pipeline.consumer_wait(clc_consumer_state)
                    work_tile = tile_sched.get_current_work()
                    clc_pipeline.consumer_release(clc_consumer_state)
                    clc_consumer_state.advance()
                else:
                    tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()

            #
            # Wait A/B buffer empty
            #
            ab_pipeline.producer_tail(ab_producer_state)

        #
        # Specialized MMA warp
        #
        if warp_idx == self.mma_warp_id:
            # MMA warp needs its own tile_sched for CLC response reads
            if cutlass.const_expr(self.use_clc_dynamic_scheduler):
                tile_sched = utils.ClcDynamicPersistentTileScheduler.create(
                    tile_sched_params,
                    cute.arch.block_idx(),
                    cute.arch.grid_dim(),
                    clc_response_ptr,
                )
            #
            # Bar sync for retrieve tensor memory ptr from shared mem
            #
            tmem.wait_for_alloc()

            #
            # Retrieving tensor memory ptr and make accumulator/SFA/SFB tensor
            #
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # Make accumulator tmem tensor
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            # Make SFA tmem tensor
            sfa_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols,
                dtype=self.sf_dtype,
            )
            # (MMA, MMA_M, MMA_K)
            tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)

            # Make SFB tmem tensor
            sfb_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols,
                dtype=self.sf_dtype,
            )
            # (MMA, MMA_N, MMA_K)
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)

            #
            # Partition for S2T copy of SFA/SFB
            #
            sfa_s2t_bundle = self._mainloop_s2t_copy_and_partition(sSFA, tCtSFA)
            sfb_s2t_bundle = self._mainloop_s2t_copy_and_partition(sSFB, tCtSFB)

            #
            # Persistent tile scheduling loop
            #
            ab_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )

            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                # Get accumulator stage index
                acc_stage_index = acc_producer_state.index

                # Set tensor memory buffer for current tile
                # (MMA, MMA_M, MMA_N)
                tCtAcc = tCtAcc_base[(None, None, None, acc_stage_index)]

                # Peek (try_wait) AB buffer full for k_tile = 0
                ab_consumer_state.reset_count()
                peek_ab_full_status = cutlass.Boolean(1)
                if ab_consumer_state.count < k_tile_cnt and is_leader_cta:
                    peek_ab_full_status = ab_pipeline.consumer_try_wait(
                        ab_consumer_state
                    )

                #
                # Wait for accumulator buffer empty
                #
                if is_leader_cta:
                    acc_pipeline.producer_acquire(acc_producer_state)

                tCtSFB_mma = tCtSFB
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] in {64, 192}):
                    # If this is an ODD tile, shift the TMEM start address for
                    # cta_tile_shape_n=192 or 64 case by two words (ignores first 64 columns of SFB)
                    offset = cutlass.Int32((mma_tile_coord_mnl[1] % 2) * 2)
                    shifted_ptr = cute.recast_ptr(
                        acc_tmem_ptr
                        + self.num_accumulator_tmem_cols
                        + self.num_sfa_tmem_cols
                        + offset,
                        dtype=self.sf_dtype,
                    )
                    tCtSFB_mma = cute.make_tensor(shifted_ptr, tCtSFB_layout)

                #
                # Mma mainloop
                #
                for k_tile in range(k_tile_cnt):
                    if is_leader_cta:
                        # Conditionally wait for AB buffer full
                        ab_pipeline.consumer_wait(
                            ab_consumer_state, peek_ab_full_status
                        )

                        if cutlass.const_expr(not self._is_interleaved_utccp()):
                            # Unless UTCCPs are to be interleaved, all SFA/SFB for all
                            # k_blocks are copied before MMA are executed
                            self._mainloop_s2t_copies(
                                ab_consumer_state.index, sfa_s2t_bundle, sfb_s2t_bundle
                            )

                        # tCtAcc += tCrA * tCrSFA * tCrB * tCrSFB
                        num_kblocks = cute.size(tCrA, mode=[2])
                        for k_block in cutlass.range(num_kblocks, unroll_full=True):
                            if cutlass.const_expr(
                                self.enable_breuse
                                and cute.size(tCtAcc.layout, mode=[1]) == 2
                                and cute.size(tCtAcc.layout, mode=[2]) == 1
                            ):
                                tCtAcc_bkeep = tCtAcc[(None, 0, 0)]
                                tCtAcc_breuse = tCtAcc[(None, 1, 0)]

                                a_kblk_crd_keep = (
                                    None,
                                    0,
                                    k_block,
                                    ab_consumer_state.index,
                                )
                                a_kblk_crd_reuse = (
                                    None,
                                    1,
                                    k_block,
                                    ab_consumer_state.index,
                                )
                                b_kblk_crd = (None, 0, k_block, ab_consumer_state.index)

                                sfa_kblk_crd_keep = (None, 0, k_block)
                                sfa_kblk_crd_reuse = (None, 1, k_block)
                                sfb_kblk_crd = (None, 0, k_block)

                                if cutlass.const_expr(self._is_interleaved_utccp()):
                                    self._mainloop_s2t_interleaved_copies(
                                        k_block,
                                        ab_consumer_state.index,
                                        sfa_s2t_bundle,
                                        sfb_s2t_bundle,
                                    )

                                # Keep
                                tiled_mma_bkeep.set(
                                    tcgen05.Field.ACCUMULATE,
                                    k_tile != 0 or k_block != 0,
                                )
                                cute.gemm(
                                    tiled_mma_bkeep,
                                    tCtAcc_bkeep,
                                    [tCrA[a_kblk_crd_keep], tCtSFA[sfa_kblk_crd_keep]],
                                    [tCrB[b_kblk_crd], tCtSFB_mma[sfb_kblk_crd]],
                                    tCtAcc_bkeep,
                                )
                                # Reuse
                                tiled_mma_breuse.set(
                                    tcgen05.Field.ACCUMULATE,
                                    k_tile != 0 or k_block != 0,
                                )
                                cute.gemm(
                                    tiled_mma_breuse,
                                    tCtAcc_breuse,
                                    [
                                        tCrA[a_kblk_crd_reuse],
                                        tCtSFA[sfa_kblk_crd_reuse],
                                    ],
                                    [tCrB[b_kblk_crd], tCtSFB_mma[sfb_kblk_crd]],
                                    tCtAcc_breuse,
                                )
                            else:
                                kblk_crd = (
                                    None,
                                    None,
                                    k_block,
                                    ab_consumer_state.index,
                                )
                                sf_kblk_crd = (None, None, k_block)

                                tiled_mma.set(
                                    tcgen05.Field.ACCUMULATE,
                                    k_tile != 0 or k_block != 0,
                                )
                                cute.gemm(
                                    tiled_mma,
                                    tCtAcc,
                                    [tCrA[kblk_crd], tCtSFA[sf_kblk_crd]],
                                    [tCrB[kblk_crd], tCtSFB_mma[sf_kblk_crd]],
                                    tCtAcc,
                                )

                        # Async arrive AB buffer empty
                        ab_pipeline.consumer_release(ab_consumer_state)

                    # Peek (try_wait) AB buffer full for k_tile = k_tile + 1
                    ab_consumer_state.advance()
                    peek_ab_full_status = cutlass.Boolean(1)
                    if ab_consumer_state.count < k_tile_cnt:
                        if is_leader_cta:
                            peek_ab_full_status = ab_pipeline.consumer_try_wait(
                                ab_consumer_state
                            )

                #
                # Async arrive accumulator buffer full
                #
                if is_leader_cta:
                    acc_pipeline.producer_commit(acc_producer_state)
                acc_producer_state.advance()

                #
                # Advance to next tile
                #
                if cutlass.const_expr(self.use_clc_dynamic_scheduler):
                    clc_pipeline.consumer_wait(clc_consumer_state)
                    work_tile = tile_sched.get_current_work()
                    clc_pipeline.consumer_release(clc_consumer_state)
                    clc_consumer_state.advance()
                else:
                    tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()

            #
            # Wait for accumulator buffer empty
            #
            acc_pipeline.producer_tail(acc_producer_state)
        #
        # Fused epilogue warps: bias + RoPE + relocation into q|k|v + amax
        #
        if warp_idx < self.mma_warp_id:
            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)
            # ((MMA_ATOM_M, MMA_M), (MMA_ATOM_N, MMA_N), STAGE) -> the CTA's (128, 256) accumulator slab per stage
            tCtAcc_t = transform_partitioned_tensor_layout(tCtAcc_base)
            tCcC_t = transform_partitioned_tensor_layout(tCgC)
            # epilogue warpgroup g (4 warps, 128 threads = the 128 TMEM lanes) handles heads g, g + num_groups, ...
            epi_tidx = tidx % 128
            epi_group = cute.arch.make_warp_uniform(warp_idx // 4)
            tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc = epilogue_tmem_copy_and_partition(
                self, epi_tidx, tCtAcc_t, tCcC_t, epi_tile, use_2cta_instrs
            )
            thr_copy_t2r = tiled_copy_t2r.get_slice(epi_tidx)
            # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, RestM, RestN, RestL): this thread's output coordinates
            tTR_cC = thr_copy_t2r.partition_D(cute.flat_divide(tCcC_t, epi_tile))
            # the thread's 64 accumulators of one head row, as two flat 32-element (lo / hi) register views
            rAcc_lo = cute.make_tensor(tTR_rAcc.iterator, cute.make_layout((HALF,)))
            rAcc_hi = cute.make_tensor(tTR_rAcc.iterator + HALF, cute.make_layout((HALF,)))
            # bf16 result fragment (same partition) and its register -> smem copy for the TMA stores
            tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
            rC_lo = cute.make_tensor(tTR_rC.iterator, cute.make_layout((HALF,)))
            rC_hi = cute.make_tensor(tTR_rC.iterator + HALF, cute.make_layout((HALF,)))
            sC = storage.sC.get_tensor(c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner)
            tiled_copy_r2s, tRS_rC, tRS_sC = epilogue_smem_copy_and_partition(self, tiled_copy_t2r, tTR_rC, epi_tidx, sC)
            # TMA store partitions of the three destinations: ((ATOM_V, REST_V), RowTiles, HeadCols, L)
            bSG_sC, bSG_gQ = cpasync.tma_partition(
                tma_atom_q, 0, cute.make_layout(1),
                cute.group_modes(sC, 0, 2), cute.group_modes(cute.flat_divide(mQ_mnl, epi_tile), 0, 2),
            )
            _, bSG_gK = cpasync.tma_partition(
                tma_atom_k, 0, cute.make_layout(1),
                cute.group_modes(sC, 0, 2), cute.group_modes(cute.flat_divide(mK_mnl, epi_tile), 0, 2),
            )
            _, bSG_gV = cpasync.tma_partition(
                tma_atom_v, 0, cute.make_layout(1),
                cute.group_modes(sC, 0, 2), cute.group_modes(cute.flat_divide(mV_mnl, epi_tile), 0, 2),
            )
            c_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 32 * 4)
            c_stages_per_group = self.num_c_stage // self.num_epi_groups
            c_pipeline = pipeline.PipelineTmaStore.create(num_stages=c_stages_per_group, producer_group=c_producer_group)

            ng = self.ng
            nqpg = self.nqpg
            nqh = ng * nqpg
            m_rows = cute.size(mC_mnl.shape[0])
            nq_elems = m_rows * nqh * HEAD_D
            nk_elems = m_rows * ng * HEAD_D
            heads_per_tile = self.heads_per_tile
            heads_per_group = heads_per_tile // self.num_epi_groups
            all_tidx = tidx % (32 * len(self.epilog_warp_id))
            rows_per_cta = self.cta_tile_shape_mnk[0]

            # stage the bias in smem once per CTA (N bf16; 16-byte vectors spread over the epilogue threads)
            sBias8 = cute.make_tensor(storage.sBias.data_ptr(), cute.make_layout((8, self.n_cols // 8), stride=(1, 8)))
            gBias8 = cute.make_tensor(gBias32.iterator, cute.make_layout((8, self.n_cols // 8), stride=(1, 8)))
            for i in cutlass.range(all_tidx, self.n_cols // 8, 32 * len(self.epilog_warp_id), unroll=1):
                sBias8[(None, i)] = gBias8[(None, i)].load()
            sBias32 = cute.make_tensor(storage.sBias.data_ptr(), cute.make_layout((HALF, self.n_cols // HALF), stride=(1, HALF)))
            self.epilog_all_barrier.arrive_and_wait()

            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )
            thread_amax = Float32(0.0)
            tiles_done = Int32(0)
            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )
                # (T2R, T2R_M, T2R_N, HEAD) accumulator subtiles of this stage; (V, HEAD) coordinates of this tile
                tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_consumer_state.index)]
                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                tTR_cC_tile = tTR_cC[(None, None, None, None, None, *mma_tile_coord_mnl)]
                tTR_cC_tile = cute.group_modes(tTR_cC_tile, 3, cute.rank(tTR_cC_tile))
                tTR_cC_tile = cute.group_modes(tTR_cC_tile, 0, 3)
                crd0 = tTR_cC_tile[(0, 0)]
                row = Int32(crd0[0])
                col0 = Int32(crd0[1])
                s_id = row // batch
                row_tile = (row - epi_tidx % rows_per_cta) // rows_per_cta   # the CTA's 128-row slab index (CTA-uniform)
                # per-row operands, fetched while the accumulator is still being produced
                cos_v = gCos32[(None, s_id)].load()
                sin_v = gSin32[(None, s_id)].load()
                row_q = (row * nqh) * HEAD_D
                row_kv = (row * ng) * HEAD_D
                num_prev_subtiles = tiles_done * heads_per_group

                acc_pipeline.consumer_wait(acc_consumer_state)
                if cutlass.const_expr(not self.diag_skip_epi):
                    for j in cutlass.range_constexpr(heads_per_group):
                        h = epi_group + j * self.num_epi_groups          # this warpgroup's head within the tile
                        cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, h)], tTR_rAcc)
                        hcol = col0 // HEAD_D + h            # head column 0..ng*(nqpg+2)-1 (warpgroup-uniform)
                        g = hcol // (nqpg + 2)
                        r = hcol - g * (nqpg + 2)            # 0..nqpg-1: q head r ; nqpg: k ; nqpg+1: v
                        # branch-free RoPE: v heads rotate with (cos, sin) = (1, 0), which is exact
                        rotf = cutlass.min(nqpg + 1 - r, 1).to(Float32)
                        c_eff = cos_v * rotf + (Float32(1.0) - rotf)
                        s_eff = sin_v * rotf
                        x_lo = rAcc_lo.load() + sBias32[(None, 2 * hcol)].load().to(Float32)
                        x_hi = rAcc_hi.load() + sBias32[(None, 2 * hcol + 1)].load().to(Float32)
                        # TE non-interleaved RoPE: out = x*cos + rotate(x)*sin, rotate = (-x_hi, x_lo); cos[d] == cos[d+32]
                        b_lo = (x_lo * c_eff - x_hi * s_eff).to(BFloat16)
                        b_hi = (x_hi * c_eff + x_lo * s_eff).to(BFloat16)
                        a_lo = cute.math.absf(b_lo.to(Float32)).reduce(cute.ReductionOp.MAX, Float32(0.0), 0)
                        a_hi = cute.math.absf(b_hi.to(Float32)).reduce(cute.ReductionOp.MAX, Float32(0.0), 0)
                        thread_amax = cute.arch.fmax(thread_amax, cute.arch.fmax(a_lo, a_hi))
                        if cutlass.const_expr(self.tma_store):
                            rC_lo.store(b_lo)
                            rC_hi.store(b_hi)
                            c_buffer = epi_group * c_stages_per_group + (num_prev_subtiles + j) % c_stages_per_group
                            cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, c_buffer)])
                            cute.arch.fence_proxy("async.shared", space="cta")
                            self._epi_group_barrier(epi_group)
                            if warp_idx % 4 == 0:
                                if r < nqpg:
                                    cute.copy(tma_atom_q, bSG_sC[(None, c_buffer)], bSG_gQ[(None, row_tile, g * nqpg + r, 0)])
                                elif r == nqpg:
                                    cute.copy(tma_atom_k, bSG_sC[(None, c_buffer)], bSG_gK[(None, row_tile, g, 0)])
                                else:
                                    cute.copy(tma_atom_v, bSG_sC[(None, c_buffer)], bSG_gV[(None, row_tile, g, 0)])
                                c_pipeline.producer_commit()
                                c_pipeline.producer_acquire()
                            self._epi_group_barrier(epi_group)
                        else:
                            dst = row_q + (g * nqpg + r) * HEAD_D
                            if r == nqpg:
                                dst = nq_elems + row_kv + g * HEAD_D
                            if r > nqpg:
                                dst = nq_elems + nk_elems + row_kv + g * HEAD_D
                            vidx = dst // HALF
                            gOut32[(None, vidx)] = b_lo
                            gOut32[(None, vidx + 1)] = b_hi
                # accumulator stage consumed
                with cute.arch.elect_one():
                    acc_pipeline.consumer_release(acc_consumer_state)
                acc_consumer_state.advance()
                tiles_done += 1
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            if cutlass.const_expr(self.tma_store):
                c_pipeline.producer_tail()

            #
            # per-tensor amax: warp -> CTA -> one atomic per CTA; the last CTA derives the scale and resets the workspace
            #
            warp_amax = Float32(cute.arch.warp_redux_sync(value=thread_amax, kind="fmax", mask_and_clamp=0xFFFFFFFF, nan=True))
            sAmax = storage.sAmax.get_tensor(cute.make_layout((8,)))
            if cute.arch.lane_idx() == 0:
                sAmax[warp_idx] = warp_amax
            self.epilog_all_barrier.arrive_and_wait()
            if warp_idx == self.epilog_warp_id[0] and cute.arch.lane_idx() == 0:
                blk = sAmax[0]
                for w in cutlass.range_constexpr(1, len(self.epilog_warp_id)):
                    blk = cute.arch.fmax(blk, sAmax[w])
                cute.arch.atomic_fmax(gWsF.iterator, blk, sign_bit=False)
                cute.arch.fence_acq_rel_gpu()
                gdim = cute.arch.grid_dim()
                nblocks = gdim[0] * gdim[1] * gdim[2]
                old = Int32(cute.arch.atomic_add(gWsI.iterator.llvm_ptr, Int32(1)))
                if old == nblocks - 1:
                    cute.arch.fence_acq_rel_gpu()
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

            tmem.relinquish_alloc_permit()
            tmem.free(acc_tmem_ptr)


# ====================================================================================================== torch side
_COMPILED = {}
_WS = {}
_HW = {}


def _ws(device):
    key = device.index
    ent = _WS.get(key)
    if ent is None:
        ent = (torch.zeros(1, dtype=torch.float32, device=device), torch.zeros(1, dtype=torch.int32, device=device))
        _WS[key] = ent
    return ent


def _ptr(t: torch.Tensor, dtype, align=16):
    return make_ptr(dtype, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=align)


def _max_active_clusters(cluster_size: int) -> int:
    v = _HW.get(cluster_size)
    if v is None:
        v = utils.HardwareInfo().get_max_active_clusters(cluster_size)
        _HW[cluster_size] = v
    return v


def _compiled(ng: int, nqpg: int):
    key = (ng, nqpg, CLUSTER_M, CLUSTER_N, SWIZZLE_SIZE, RASTER_ORDER, DIAG_SKIP_EPI, PREFETCH_DIST, STORE_MODE, EPI_WARPS)
    fn = _COMPILED.get(key)
    if fn is None:
        gemm = QkvGemmRopeKernel(ng, nqpg, cluster_shape_mn=(CLUSTER_M, CLUSTER_N))
        z8 = make_ptr(cutlass.Float8E4M3FN, 0, cute.AddressSpace.gmem, assumed_align=16)
        zsf = make_ptr(cutlass.Float8E8M0FNU, 0, cute.AddressSpace.gmem, assumed_align=32)
        zbf = make_ptr(BFloat16, 0, cute.AddressSpace.gmem, assumed_align=16)
        zf = make_ptr(Float32, 0, cute.AddressSpace.gmem, assumed_align=16)
        zi = make_ptr(Int32, 0, cute.AddressSpace.gmem, assumed_align=16)
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        fn = cute.compile(
            gemm, z8, z8, zsf, zsf, zbf, zbf, zf, zf, zf, zf, zf, zi,
            (Int32(0), Int32(0), Int32(0), Int32(0)), Int32(0), Int32(0),
            _max_active_clusters(CLUSTER_M * CLUSTER_N), stream,
        )
        _COMPILED[key] = fn
        _say(f"compiled fused qkv GEMM+RoPE kernel: tiler={MMA_TILER} inst={MMA_INST} cluster=({CLUSTER_M},{CLUSTER_N}) "
             f"ab_stages={gemm.num_ab_stage} acc_stages={gemm.num_acc_stage} swizzle={SWIZZLE_SIZE} raster={RASTER_ORDER} "
             f"max_active_clusters={_max_active_clusters(CLUSTER_M * CLUSTER_N)} prefetch={PREFETCH_DIST} store={STORE_MODE} "
             f"c_stages={gemm.num_c_stage} epi_warps={EPI_WARPS} threads={gemm.threads_per_cta} diag_skip_epi={DIAG_SKIP_EPI}")
    return fn


@torch.library.custom_op("gpt_oss::qkv_gemm_rope_fwd_amax", mutates_args=())
def qkv_gemm_rope_fwd_amax(
    a_data: torch.Tensor, a_scale_inv: torch.Tensor, w_data: torch.Tensor, w_scale_inv: torch.Tensor,
    bias: torch.Tensor, cos32: torch.Tensor, sin32: torch.Tensor, b: int, ng: int, nqpg: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flat bf16 q|k|v = RoPE(A x W^T + bias) in (s, b, heads, 64) order, plus the Float8CurrentScaling scale and amax.

    a_data [M, K] and w_data [N, K] are MXFP8 e4m3 bytes (K-major), a_scale_inv / w_scale_inv their GEMM-swizzled e8m0
    scales (TE's _rowwise_data / _rowwise_scale_inv); bias bf16 [N]; cos32 / sin32 f32 [s, 32]; N = ng * (nqpg + 2) * 64."""
    M, K = a_data.shape
    N = w_data.shape[0]
    s = M // b
    assert w_data.shape[1] == K and bias.shape == (N,) and N == ng * (nqpg + 2) * HEAD_D
    assert M % 256 == 0 and N % 256 == 0 and K % 32 == 0 and M % b == 0
    assert cos32.shape == (s, HALF) and sin32.shape == (s, HALF)
    assert a_data.dtype == torch.uint8 and w_data.dtype == torch.uint8 and a_scale_inv.dtype == torch.uint8
    assert a_data.is_contiguous() and w_data.is_contiguous() and bias.is_contiguous() and cos32.is_contiguous()
    kpad = -(-(K // SF_VEC) // 4) * 4
    assert a_scale_inv.numel() >= -(-M // 128) * 128 * kpad and w_scale_inv.numel() >= -(-N // 128) * 128 * kpad
    dev = a_data.device
    flat = torch.empty(M * N, dtype=torch.bfloat16, device=dev)
    scale = torch.empty(1, dtype=torch.float32, device=dev)
    amax = torch.empty(1, dtype=torch.float32, device=dev)
    wsf, wsi = _ws(dev)
    fn = _compiled(ng, nqpg)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    fn(
        _ptr(a_data, cutlass.Float8E4M3FN), _ptr(w_data, cutlass.Float8E4M3FN),
        _ptr(a_scale_inv, cutlass.Float8E8M0FNU, 32), _ptr(w_scale_inv, cutlass.Float8E8M0FNU, 32),
        _ptr(flat, BFloat16), _ptr(bias, BFloat16), _ptr(cos32, Float32), _ptr(sin32, Float32),
        _ptr(scale, Float32), _ptr(amax, Float32), _ptr(wsf, Float32), _ptr(wsi, Int32),
        (M, N, K, 1), b, s, stream,
    )
    return flat, scale, amax


@qkv_gemm_rope_fwd_amax.register_fake
def _(a_data, a_scale_inv, w_data, w_scale_inv, bias, cos32, sin32, b, ng, nqpg):
    M = a_data.shape[0]
    N = w_data.shape[0]
    return (torch.empty(M * N, dtype=torch.bfloat16, device=a_data.device),
            torch.empty(1, dtype=torch.float32, device=a_data.device),
            torch.empty(1, dtype=torch.float32, device=a_data.device))


# ---------------------------------------------------------------------------------------------- references / harness
def rope_reference_flat(c_bf16_or_f32: torch.Tensor, cos32: torch.Tensor, sin32: torch.Tensor, b: int, ng: int, nqpg: int):
    """flat q|k|v bf16 from the (bias-added) GEMM output [M, N] (fp32 math; TE non-interleaved RoPE)."""
    M, N = c_bf16_or_f32.shape
    s = M // b
    x = c_bf16_or_f32.float().view(s, b, ng, nqpg + 2, HEAD_D)
    cos = torch.cat([cos32, cos32], dim=-1).view(s, 1, 1, 1, HEAD_D)
    sin = torch.cat([sin32, sin32], dim=-1).view(s, 1, 1, 1, HEAD_D)
    def rot(t):
        return torch.cat([-t[..., HALF:], t[..., :HALF]], dim=-1)
    qk = x[..., : nqpg + 1, :]
    qk = qk * cos + rot(qk) * sin
    q = qk[..., :nqpg, :].reshape(s, b, ng * nqpg, HEAD_D)
    k = qk[..., nqpg, :].reshape(s, b, ng, HEAD_D)
    v = x[..., nqpg + 1, :].reshape(s, b, ng, HEAD_D)
    flat = torch.cat([q.reshape(-1), k.reshape(-1), v.reshape(-1)]).to(torch.bfloat16)
    return flat


def get_inputs(M=24576, K=2880, s=8192, ng=8, nqpg=8, device="cuda", seed=0):
    """Sample inputs at the workload shape: TE MXFP8 quantization (GEMM-swizzled scales) of random bf16 x and w."""
    import transformer_engine.pytorch  # noqa: F401  (loads the TE torch extension)
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
    torch.manual_seed(seed)
    N = ng * (nqpg + 2) * HEAD_D
    b = M // s
    x = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    w = (torch.randn(N, K, device=device, dtype=torch.bfloat16) * 0.02)
    bias = torch.randn(N, device=device, dtype=torch.bfloat16) * 0.1
    freqs = torch.rand(s, HALF, device=device, dtype=torch.float32) * 6.283
    cos32, sin32 = torch.cos(freqs).contiguous(), torch.sin(freqs).contiguous()
    qx = MXFP8Quantizer(tex.DType.kFloat8E4M3, rowwise=True, columnwise=False)
    qw = MXFP8Quantizer(tex.DType.kFloat8E4M3, rowwise=True, columnwise=False)
    qx.optimize_for_gemm = True
    qw.optimize_for_gemm = True
    X = qx(x)
    W = qw(w)
    assert X._with_gemm_swizzled_scales and W._with_gemm_swizzled_scales
    return dict(X=X, W=W, x=x, w=w, bias=bias, cos32=cos32, sin32=sin32, b=b, ng=ng, nqpg=nqpg)


def kernel_fn(X, W, bias, cos32, sin32, b, ng, nqpg):
    return torch.ops.gpt_oss.qkv_gemm_rope_fwd_amax(
        X._rowwise_data, X._rowwise_scale_inv, W._rowwise_data, W._rowwise_scale_inv, bias, cos32, sin32, b, ng, nqpg)


def reference_fn(X, W, bias, cos32, sin32, b, ng, nqpg):
    """fp32 GEMM on the dequantized MXFP8 operands + bias, then RoPE/relocation; amax of the bf16 result."""
    xd = X.dequantize(dtype=torch.float32)
    wd = W.dequantize(dtype=torch.float32)
    c = xd @ wd.t() + bias.float()
    flat = rope_reference_flat(c, cos32, sin32, b, ng, nqpg)
    amax = flat.float().abs().amax()
    return flat, amax
