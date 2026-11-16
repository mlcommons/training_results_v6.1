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

"""ZeRO-2 communication for flat Flux.1 training."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from torch import Tensor

import transformer_engine.pytorch.cpp_extensions as tex

if TYPE_CHECKING:
    from flat.hier_topology import HierTopology

_FP8_AG = int(os.environ.get("FP8_AG", "0")) == 1

# Weight-amax all-gather rides the master -> BF16 shard copy; BF16 AG path only.
_WEIGHT_AMAX_AG = os.environ.get("FLAT_WEIGHT_AMAX_AG", "1") == "1"

# SM count for the green-context comm stream; 0 = plain stream.
_GREEN_SMS = int(os.environ.get("FLAT_NCCL_GREEN_SMS", "0"))

# TE FP8 GEMM TMA descriptors need 16B-aligned operands: 8 BF16 elements.
_TMA_ALIGN_ELEMS = 8

# NCCL symmetric multicast vectorizes only when (in_ptr - out_ptr) % 16 == 0;
# 16 elements keeps shard_offset and grad_offset 16B-aligned for bf16 and fp8.
_SYM_ALIGN_ELEMS = int(os.environ.get("FLAT_SYM_ALIGN_ELEMS", "16"))


def _ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _make_comm_stream(
    device: torch.device, green_sms: int,
) -> "tuple[torch.cuda.Stream, object | None]":
    """Create the overlap comm stream, optionally on a green context.

    Returns (stream, green_ctx); green_ctx (None when unused) must outlive the stream.
    """
    if green_sms <= 0:
        return torch.cuda.Stream(device=device), None
    if not getattr(
        torch.cuda, "GreenContext", None,
    ) or not getattr(
        __import__("torch.cuda.green_contexts", fromlist=["SUPPORTED"]),
        "SUPPORTED",
        False,
    ):
        raise RuntimeError(
            "FLAT_NCCL_GREEN_SMS is set but this PyTorch build has no "
            "CUDA green-context support (needs CUDA >= 12.8 and a build "
            "with PYTORCH_C10_DRIVER_API_SUPPORTED)."
        )
    idx = device.index
    if idx is None:
        idx = torch.cuda.current_device()
    green_ctx = torch.cuda.GreenContext.create(
        num_sms=green_sms, device_id=idx,
    )
    return green_ctx.Stream(), green_ctx


@dataclass
class _Bucket:
    """Metadata for one parameter bucket."""

    bucket_id: int
    params: list  # list of nn.Parameter
    param_views: list  # list of (offset_within_bucket, shape)

    # Sizes (set during layout)
    raw_numel: int = 0  # sum of param numels
    padded_numel: int = 0  # raw_numel padded to a multiple of G
    shard_numel: int = 0  # padded_numel // shard_size (G)

    # Offsets into the flat buffers (set during layout)
    grad_offset: int = 0  # offset into grad_buffer
    shard_offset: int = 0  # offset into reduced/master/opt buffers


class BucketManager:
    """Manages ZeRO-2 communication for flat Flux.1 training."""

    def __init__(
        self,
        model: torch.nn.Module,
        topo: "HierTopology",
        shard_group_name: str,
        replicate_group_name: str,
        device: torch.device,
        max_bucket_numel: int = 0,
        replicate_inflight_max: int = 1,
        replicate_group: "dist.ProcessGroup | None" = None,
        shard_group: "dist.ProcessGroup | None" = None,
    ) -> None:
        self.rank = topo.rank
        self.world_size = topo.world_size
        # Hierarchical ZeRO-2: shard within the NVLink domain (size G),
        # replicate the shard across D domains.
        self.shard_size = topo.shard_size  # G
        self.n_domains = topo.n_domains  # D
        self.shard_rank = topo.intra_pos  # position within domain
        self.shard_group_name = shard_group_name  # NVLS
        self.replicate_group_name = replicate_group_name  # IB SHARP
        self.is_hierarchical = topo.is_hierarchical
        self.device = device
        # Max inter-domain all-reduces kept in flight during RS drain.
        self.replicate_inflight_max = replicate_inflight_max

        # FLAT_SHARP_REGISTER=1: reduced_grad_buf lives in a NCCL-registered MemPool,
        # registered before capture (lazy registration hangs under graph capture).
        self._replicate_group = replicate_group
        self._sharp_register = (
            os.environ.get("FLAT_SHARP_REGISTER", "0") == "1"
            and self.is_hierarchical
            and replicate_group is not None
        )
        # MemPool + its NCCL backend, kept for register/deregister.
        self._reg_pool: "torch.cuda.MemPool | None" = None
        self._reg_backend = None

        self._shard_group = shard_group

        # FLAT_NVLS_AG_REGISTER=1: param_buffer + ag_shard_buf live in a cuMem MemPool
        # registered as NCCL symmetric windows before capture.
        self._nvls_register = (
            os.environ.get("FLAT_NVLS_AG_REGISTER", "0") == "1"
            and self.is_hierarchical
            and shard_group is not None
        )
        self._nvls_pool: "torch.cuda.MemPool | None" = None
        self._nvls_backend = None
        if self._nvls_register:
            self._nvls_backend = shard_group._get_backend(device)
            # Symmetric windows need cuMem memory: a default MemPool (caching allocator)
            # silently downgrades to a non-symmetric registration, so fail hard.
            alloc = getattr(self._nvls_backend, "mem_allocator", None)
            if alloc is None:
                raise RuntimeError(
                    "FLAT_NVLS_AG_REGISTER=1 needs the NCCL backend cuMem "
                    "allocator (backend.mem_allocator) to build symmetric "
                    "windows; it is absent in this PyTorch/NCCL build. Unset "
                    "FLAT_NVLS_AG_REGISTER to fall back to the functional RING "
                    "AllGather."
                )
            self._nvls_pool = torch.cuda.MemPool(alloc)

        # Pending async handles
        self.pending_rs: dict[int, Tensor] = {}
        self.pending_ag: dict[int, Tensor] = {}
        self.pending_ar: dict[int, Tensor] = {}
        # FIFO of bucket ids with an inter-domain all-reduce in flight (drain oldest first).
        self._ar_order: list[int] = []
        # Comm stream bridging the intra-domain RS to the inter-domain AR; with
        # FLAT_NCCL_GREEN_SMS>0 it is a green-context stream capped to that many SMs.
        self._green_sms = _GREEN_SMS
        self._green_active = (
            self._green_sms > 0
            and self.is_hierarchical
            and replicate_group is not None
        )
        self._comm_stream, self._green_ctx = _make_comm_stream(
            device, self._green_sms,
        )
        # AG-forward overlap state (set by prepare_ag_overlap)
        self._next_ag_handle: Tensor | None = None
        self._next_ag_bid: int = -1
        # Weight-amax all-gather state, initialised lazily on the first eager
        # prepare_ag_overlap, after the quantized-weight registry is filled.
        self._t6_enabled = _WEIGHT_AMAX_AG and not _FP8_AG
        self._t6_ready: bool | None = None  # None = not yet attempted
        self._t6_tables: tuple[Tensor, ...] | None = None
        self._t6_partial: Tensor | None = None
        self._t6_dsl_zero = False
        self._t6_gathered: Tensor | None = None
        self._t6_amax: Tensor | None = None
        self._t6_ag_handle: Tensor | None = None
        # Stem (bucket 0) AG join event: created by fire_stem_ag and consumed by
        # finish_stem_ag within the same captured body; never crosses a replay.
        self._stem_join_event: torch.cuda.Event | None = None
        # Bucket layout follows model.parameters() order.
        _param_align = 1
        self.buckets: list[_Bucket] = self._build_buckets(
            model, max_bucket_numel, _param_align,
        )
        self._compute_layout()
        self._build_block_bucket_map(model)

        total_grad = sum(b.padded_numel for b in self.buckets)
        total_shard = sum(b.shard_numel for b in self.buckets)

        self.grad_buffer = torch.zeros(
            total_grad, dtype=torch.bfloat16, device=device,
        )
        # Under FLAT_SHARP_REGISTER the buffer must be pool-allocated: NCCL registers
        # per MemPool, not per tensor.
        if self._sharp_register:
            self._reg_backend = self._replicate_group._get_backend(device)
            try:
                self._reg_pool = torch.cuda.MemPool(
                    self._reg_backend.mem_allocator
                )
            except Exception:
                self._reg_pool = torch.cuda.MemPool()
            with torch.cuda.use_mem_pool(self._reg_pool):
                self.reduced_grad_buf = torch.zeros(
                    total_shard, dtype=torch.bfloat16, device=device,
                )
        else:
            self.reduced_grad_buf = torch.zeros(
                total_shard, dtype=torch.bfloat16, device=device,
            )
        self.master_buf = torch.zeros(
            total_shard, dtype=torch.float32, device=device,
        )
        self.opt_m_buf = torch.zeros(
            total_shard, dtype=torch.float32, device=device,
        )
        self.opt_v_buf = torch.zeros(
            total_shard, dtype=torch.float32, device=device,
        )
        # FP32 copy of the reduced grads for the optimizer.
        self.reduced_grad_f32 = torch.zeros(
            total_shard, dtype=torch.float32, device=device,
        )
        # BF16 shard buffer, the AG input; pool-allocated under FLAT_NVLS_AG_REGISTER.
        if self._nvls_register:
            with torch.cuda.use_mem_pool(self._nvls_pool):
                self.ag_shard_buf = torch.zeros(
                    total_shard, dtype=torch.bfloat16, device=device,
                )
        else:
            self.ag_shard_buf = torch.zeros(
                total_shard, dtype=torch.bfloat16, device=device,
            )
        # Reusable noop flag for TE multi_tensor ops
        self._noop_flag = torch.zeros(
            1, dtype=torch.int32, device=device,
        )
        # Lazily built tables and buffers for the CUTLASS-DSL optimizer kernels.
        self._ctm_state: dict | None = None
        # Capturable optimizer scalar tensors (for CUDA graph)
        self._lr_tensor = torch.zeros(
            1, dtype=torch.float32, device=device,
        )
        self._step_tensor = torch.zeros(
            1, dtype=torch.int32, device=device,
        )
        self._inv_scale = torch.ones(
            1, dtype=torch.float32, device=device,
        )
        # Params are views into param_buffer; the AG writes gathered buckets here.
        # Pool-allocated under FLAT_NVLS_AG_REGISTER.
        if self._nvls_register:
            with torch.cuda.use_mem_pool(self._nvls_pool):
                self.param_buffer = torch.zeros(
                    total_grad, dtype=torch.bfloat16, device=device,
                )
        else:
            self.param_buffer = torch.zeros(
                total_grad, dtype=torch.bfloat16, device=device,
            )

        # FP8 all-gather: weights gathered as E4M3 into double-buffered scratch
        # (avoids a WAR hazard between the FP8 -> BF16 cast and the next AG).
        self._fp8_ag = _FP8_AG
        if self._fp8_ag:
            self.fp8_shard_buf = torch.zeros(
                total_shard, dtype=torch.float8_e4m3fn, device=device,
            )
            max_padded = max(b.padded_numel for b in self.buckets)
            self.fp8_scratch = [
                torch.zeros(
                    max_padded, dtype=torch.float8_e4m3fn,
                    device=device,
                ),
                torch.zeros(
                    max_padded, dtype=torch.float8_e4m3fn,
                    device=device,
                ),
            ]
            self._fp8_scratch_idx = 0

        self._init_master_weights()
        self._setup_param_views()

    # Bucket construction

    @staticmethod
    def _build_buckets(
        model: torch.nn.Module,
        max_bucket_numel: int = 0,
        param_align: int = 1,
    ) -> list[_Bucket]:
        """Build buckets by grouping adjacent model blocks.

        Bucket 0 is the stem and the last bucket is final_layer.
        """
        def _make_bucket(bid: int, params: list) -> _Bucket:
            views, off = [], 0
            for p in params:
                views.append((off, list(p.shape)))
                off += p.numel()
                # Pad this param's end so the next param's offset is aligned.
                off = _ceildiv(off, param_align) * param_align
            return _Bucket(
                bucket_id=bid, params=params,
                param_views=views, raw_numel=off,
            )

        def _chunk_blocks(blocks, max_numel):
            """Group blocks into chunks not exceeding max_numel."""
            chunks = []
            cur_params, cur_numel = [], 0
            for block in blocks:
                block_params = list(block.parameters())
                block_numel = sum(p.numel() for p in block_params)
                if cur_numel + block_numel > max_numel and cur_params:
                    chunks.append(cur_params)
                    cur_params, cur_numel = [], 0
                cur_params.extend(block_params)
                cur_numel += block_numel
            if cur_params:
                chunks.append(cur_params)
            return chunks

        buckets: list[_Bucket] = []
        bid = 0

        # Stem: small, always one bucket
        stem = []
        stem.extend(model.img_in.parameters())
        stem.extend(model.txt_in.parameters())
        stem.extend(model.time_in.parameters())
        stem.extend(model.vector_in.parameters())
        buckets.append(_make_bucket(bid, stem))
        bid += 1

        for chunk in _chunk_blocks(model.double_blocks, max_bucket_numel):
            buckets.append(_make_bucket(bid, chunk))
            bid += 1

        for chunk in _chunk_blocks(model.single_blocks, max_bucket_numel):
            buckets.append(_make_bucket(bid, chunk))
            bid += 1

        # Final layer: small, always one bucket
        fl = list(model.final_layer.parameters())
        buckets.append(_make_bucket(bid, fl))

        return buckets

    def _compute_layout(self) -> None:
        """Compute padded sizes and offsets for all buckets.

        padded_numel = shard_numel * shard_size; grad_offset and shard_offset
        are running sums, each kept 16B-aligned.
        """
        # Both the AG input (ag_shard_buf[shard_offset]) and output
        # (param_buffer[grad_offset]) must be 16B-aligned for NCCL's vectorized path.
        shard_align = _SYM_ALIGN_ELEMS
        if shard_align < 1:  # FLAT_SYM_ALIGN_ELEMS=0: TMA-only align
            shard_align = _TMA_ALIGN_ELEMS // math.gcd(
                self.shard_size, _TMA_ALIGN_ELEMS
            )
        grad_offset = 0
        shard_offset = 0
        for b in self.buckets:
            raw_shard = _ceildiv(b.raw_numel, self.shard_size)
            b.shard_numel = _ceildiv(raw_shard, shard_align) * shard_align
            b.padded_numel = b.shard_numel * self.shard_size
            b.grad_offset = grad_offset
            b.shard_offset = shard_offset
            grad_offset += b.padded_numel
            shard_offset += b.shard_numel

    def _build_block_bucket_map(self, model: torch.nn.Module) -> None:
        """Build mapping from model block IDs to bucket IDs.

        Block IDs: 0 = stem, then double_blocks, single_blocks, final_layer.
        RS fires only when all blocks in a bucket are done.
        """
        n_double = len(model.double_blocks)
        n_single = len(model.single_blocks)

        # Build param_id -> bucket_id lookup
        param_to_bucket: dict[int, int] = {}
        for b in self.buckets:
            for p in b.params:
                param_to_bucket[id(p)] = b.bucket_id

        # Map old block IDs to new bucket IDs
        self._old_to_new_bucket: dict[int, int] = {}

        # Stem (old ID 0)
        stem_params = list(model.img_in.parameters()) + \
            list(model.txt_in.parameters()) + \
            list(model.time_in.parameters()) + \
            list(model.vector_in.parameters())
        if stem_params:
            self._old_to_new_bucket[0] = param_to_bucket[id(stem_params[0])]

        # Double blocks (old IDs 1..n_double)
        for i, block in enumerate(model.double_blocks):
            params = list(block.parameters())
            if params:
                self._old_to_new_bucket[1 + i] = param_to_bucket[id(params[0])]

        # Single blocks (old IDs n_double+1..n_double+n_single)
        for i, block in enumerate(model.single_blocks):
            params = list(block.parameters())
            if params:
                self._old_to_new_bucket[1 + n_double + i] = param_to_bucket[id(params[0])]

        # Final layer (old ID 1+n_double+n_single)
        fl_params = list(model.final_layer.parameters())
        if fl_params:
            self._old_to_new_bucket[1 + n_double + n_single] = param_to_bucket[id(fl_params[0])]

        # Count blocks per bucket (for completion tracking)
        self._blocks_per_bucket: dict[int, int] = {}
        for new_bid in self._old_to_new_bucket.values():
            self._blocks_per_bucket[new_bid] = \
                self._blocks_per_bucket.get(new_bid, 0) + 1

        # Runtime counter: reset before each backward
        self._blocks_done: dict[int, int] = {
            bid: 0 for bid in self._blocks_per_bucket
        }

    def reset_rs_overlap(self) -> None:
        """Reset per-bucket completion counters before a backward pass."""
        for bid in self._blocks_done:
            self._blocks_done[bid] = 0

    def fire_rs_for_block(self, old_block_id: int) -> None:
        """Fire RS (+ chained replicate AR) when a bucket completes."""
        new_bid = self._old_to_new_bucket.get(old_block_id)
        if new_bid is None:
            return
        self._blocks_done[new_bid] += 1
        if self._blocks_done[new_bid] == self._blocks_per_bucket[new_bid]:
            self._fire_rs_and_replicate(new_bid)

    def fire_rs_for_block_graphsafe(self, old_block_id: int) -> None:
        """Fire RS (+ chained replicate AR) when a bucket completes (graph-safe)."""
        new_bid = self._old_to_new_bucket.get(old_block_id)
        if new_bid is None:
            return
        self._blocks_done[new_bid] += 1
        if self._blocks_done[new_bid] == self._blocks_per_bucket[new_bid]:
            self._fire_rs_and_replicate(new_bid)

    def _fire_rs_and_replicate(self, bucket_id: int) -> None:
        """Reduce-scatter a bucket, then chain the inter-domain AR on the comm stream."""
        b = self.buckets[bucket_id]
        grad_slice = self.grad_buffer[
            b.grad_offset : b.grad_offset + b.padded_numel
        ]
        rs = torch.ops._c10d_functional.reduce_scatter_tensor(
            grad_slice, "sum", self.shard_size, self.shard_group_name,
        )
        if not self.is_hierarchical:
            self.pending_rs[bucket_id] = rs
            return
        with torch.cuda.stream(self._comm_stream):
            torch.ops._c10d_functional.wait_tensor(rs)
            # rs was allocated on the compute stream but is consumed on the comm
            # stream; tell the allocator so the block is not reused while the AR reads it.
            rs.record_stream(self._comm_stream)
            if self._green_active:
                # Functional collectives run on NCCL's own stream, so only a blocking
                # all_reduce on the current (green) stream is capped to the green SMs.
                shard = self.reduced_grad_buf[
                    b.shard_offset : b.shard_offset + b.shard_numel
                ]
                shard.copy_(rs)
                dist.all_reduce(
                    shard,
                    op=dist.ReduceOp.SUM,
                    group=self._replicate_group,
                    async_op=False,
                )
                ev = torch.cuda.Event()
                ev.record(self._comm_stream)
                handle = ev
            elif self._sharp_register:
                # In-place all_reduce_ on the registered buffer; the non-underscore
                # all_reduce clones into unregistered memory and defeats registration.
                shard = self.reduced_grad_buf[
                    b.shard_offset : b.shard_offset + b.shard_numel
                ]
                shard.copy_(rs)
                torch.ops._c10d_functional.all_reduce_(
                    shard, "sum", self.replicate_group_name,
                )
                handle = shard
            else:
                handle = torch.ops._c10d_functional.all_reduce(
                    rs, "sum", self.replicate_group_name,
                )
        self.pending_ar[bucket_id] = handle
        self._ar_order.append(bucket_id)
        if len(self._ar_order) > self.replicate_inflight_max:
            self._drain_oldest_replicate()

    def _drain_oldest_replicate(self) -> None:
        """Wait the oldest in-flight replicate AR and copy it into reduced_grad_buf."""
        bucket_id = self._ar_order.pop(0)
        handle = self.pending_ar.pop(bucket_id)
        if self._green_active:
            # handle is an event on the green comm stream; the result is already in
            # reduced_grad_buf, so only order the default stream after it.
            torch.cuda.current_stream().wait_event(handle)
        elif self._sharp_register:
            # In-place all_reduce_ already wrote reduced_grad_buf; only
            # order the default stream after the comm-stream AR completes.
            torch.ops._c10d_functional.wait_tensor(handle)
        else:
            result = torch.ops._c10d_functional.wait_tensor(handle)
            # The AR output was allocated on the comm stream; keep its block alive
            # until this default-stream copy has run.
            result.record_stream(torch.cuda.current_stream())
            b = self.buckets[bucket_id]
            self.reduced_grad_buf[
                b.shard_offset : b.shard_offset + b.shard_numel
            ].copy_(result)

    def register_sharp_pool(self) -> None:
        """Register the reduced-grad MemPool with NCCL (symm=False) before capture.

        No-op unless FLAT_SHARP_REGISTER=1. Call once, after the warm-up
        all-reduce and before graph capture.
        """
        if not self._sharp_register or self._reg_pool is None:
            return
        self._reg_backend.register_mem_pool(self._reg_pool)

    def deregister_sharp_pool(self) -> None:
        """Deregister the reduced-grad MemPool from NCCL; call after the last replay."""
        if not self._sharp_register or self._reg_pool is None:
            return
        self._reg_backend.deregister_mem_pool(self._reg_pool)

    def register_nvls_pool(self) -> None:
        """Register the all-gather MemPool as NCCL symmetric windows before capture.

        No-op unless FLAT_NVLS_AG_REGISTER=1. Call once, after the shard comm
        is initialised and before graph capture.
        """
        if not self._nvls_register or self._nvls_pool is None:
            return
        self._nvls_backend.register_mem_pool(self._nvls_pool, symm=True)

    def deregister_nvls_pool(self) -> None:
        """Deregister the all-gather MemPool from NCCL; call after the last replay."""
        if not self._nvls_register or self._nvls_pool is None:
            return
        self._nvls_backend.deregister_mem_pool(self._nvls_pool)

    def wait_all_reduce_scatters_graphsafe(
        self, fuse_upcast: bool = False, scrub: bool = True,
    ) -> None:
        """Drain reduced gradients into reduced_grad_buf (graph-safe, post-backward).

        Reductions are "sum"; the 1/world scaling is applied by the caller.
        fuse_upcast also fills reduced_grad_f32. Pass scrub=False when the
        consumer scrubs NaN/Inf itself, as the DSL grad-norm kernel does.
        """
        if self.is_hierarchical:
            while self._ar_order:
                self._drain_oldest_replicate()
        else:
            for bucket_id, rs in self.pending_rs.items():
                result = torch.ops._c10d_functional.wait_tensor(rs)
                b = self.buckets[bucket_id]
                soff = b.shard_offset
                sz = b.shard_numel
                self.reduced_grad_buf[soff : soff + sz].copy_(result[:sz])
            self.pending_rs.clear()
        if not scrub:
            return
        # Scrub NaN/Inf once on the reduced bf16 buffer, before the upcast; all
        # downstream state derives from this buffer.
        torch.nan_to_num_(self.reduced_grad_buf, 0.0, 0.0, 0.0)
        if fuse_upcast:
            self.reduced_grad_f32.copy_(self.reduced_grad_buf)

    # Wgrad delay: wgrad GEMMs on a separate CUDA stream

    def init_wgrad_delay(self) -> None:
        """Create the deferred-wgrad CUDA stream; call once before graph capture."""
        self._wgrad_stream = torch.cuda.Stream(device=self.device)
        self._wgrad_fn = None  # set by _init_wgrad_imports
        self._accum_fn = None
        # Count of wgrads deferred since the last RS fire; a bucket with none
        # must not sync the wgrad stream during capture (unforked stream).
        self._wgrad_pending_count = 0

    def _init_wgrad_imports(self) -> None:
        """Lazy-import fp8_ops to avoid circular dependency."""
        if self._wgrad_fn is not None:
            return
        from flat.fp8_ops import fp8_linear_bwd_wgrad
        from flat.model import _accum_grad
        self._wgrad_fn = fp8_linear_bwd_wgrad
        self._accum_fn = _accum_grad

    def defer_wgrad(
        self,
        weight: Tensor,
        bias: Tensor | None,
        wctx: dict,
    ) -> None:
        """Launch a wgrad GEMM on the wgrad stream, ordered after the current stream.

        wctx comes from fp8_linear_bwd_dgrad* and holds fp8_grad, grad_b and fp8_x/x_2d.
        """
        self._init_wgrad_imports()
        self._wgrad_pending_count += 1
        # Record event so wgrad stream sees this layer's dgrad output
        ev = torch.cuda.Event()
        ev.record()
        with torch.cuda.stream(self._wgrad_stream):
            self._wgrad_stream.wait_event(ev)
            grad_w, grad_b = self._wgrad_fn(
                wctx, wgrad_out=weight.grad,
            )
            if bias is not None and grad_b is not None:
                self._accum_fn(bias, grad_b)

    def fire_rs_for_block_wgrad_graphsafe(
        self, old_block_id: int,
    ) -> None:
        """Fire RS after syncing the wgrad stream, if any wgrad was deferred.

        Syncing an un-forked stream during capture raises
        cudaErrorStreamCaptureIsolation, hence the pending-count check.
        """
        new_bid = self._old_to_new_bucket.get(old_block_id)
        if new_bid is None:
            return
        self._blocks_done[new_bid] += 1
        if self._blocks_done[new_bid] == self._blocks_per_bucket[new_bid]:
            if self._wgrad_pending_count > 0:
                wg_ev = torch.cuda.Event()
                wg_ev.record(self._wgrad_stream)
                torch.cuda.current_stream().wait_event(wg_ev)
                self._wgrad_pending_count = 0
            self._fire_rs_and_replicate(new_bid)

    # AG-forward overlap: per-bucket all-gather during forward

    def _fire_ag_into_param_buffer(self, bid: int) -> Tensor:
        """Fire the bf16 intra-domain AG for bucket bid into param_buffer; returns the handle."""
        b = self.buckets[bid]
        soff = b.shard_offset
        shard = self.ag_shard_buf[soff : soff + b.shard_numel]
        out = self.param_buffer[
            b.grad_offset : b.grad_offset + b.padded_numel
        ]
        return torch.ops._c10d_functional.all_gather_into_tensor_out(
            shard, self.shard_size, self.shard_group_name, out=out,
        )

    def _fire_ag_fp8(self, bid: int) -> Tensor:
        """Fire the FP8 AG for bucket bid into the double-buffered scratch.

        Returns the AG handle; the caller casts scratch to BF16 after wait_tensor().
        """
        b = self.buckets[bid]
        shard = self.fp8_shard_buf[
            b.shard_offset : b.shard_offset + b.shard_numel
        ]
        out = self.fp8_scratch[self._fp8_scratch_idx][
            : b.padded_numel
        ]
        self._fp8_scratch_idx ^= 1
        # Use uint8 views for NCCL: float8_e4m3fn hangs during
        # CUDA graph capture. Both are 1 byte so the view is free.
        return torch.ops._c10d_functional.all_gather_into_tensor_out(
            shard.view(torch.uint8),
            self.shard_size,
            self.shard_group_name,
            out=out.view(torch.uint8),
        )

    def _t6_init(self) -> None:
        """Build the weight-amax segment tables and buffers.

        Runs once, on the first eager prepare_ag_overlap, never under capture.
        Slot order is bucket order, so it is identical on every rank.
        """
        self._t6_ready = False
        from flat import nvfp4_ops
        from flat.custom_ops import weight_amax_copy as _wamax
        quantized = nvfp4_ops._class_of_weight
        if not quantized:
            return
        lo_r = self.shard_rank
        spans: list[tuple[int, int, int]] = []
        # nvfp4_ops receives (weight, slot) pairs, so its registry is keyed on
        # weight identity rather than on an address a later model could inherit.
        slot_of: dict[int, int] = {}
        slots: list[tuple[torch.Tensor, int]] = []
        for b in self.buckets:
            win_lo = lo_r * b.shard_numel
            win_hi = win_lo + b.shard_numel
            for p, (local_off, shape) in zip(b.params, b.param_views):
                if quantized.get(p) is None:
                    continue
                slot = slot_of.get(p.data_ptr())
                if slot is None:
                    slot = len(slots)
                    slot_of[p.data_ptr()] = slot
                    slots.append((p, slot))
                g_lo = max(local_off, win_lo)
                g_hi = min(local_off + p.numel(), win_hi)
                if g_lo < g_hi:
                    spans.append((
                        slot,
                        b.shard_offset + (g_lo - win_lo),
                        b.shard_offset + (g_hi - win_lo),
                    ))
        n_slots = len(slots)
        self._t6_tables = _wamax.build_segment_tables(
            spans, self.master_buf.numel(), self.master_buf.device,
        )
        self._t6_partial = torch.zeros(
            n_slots, dtype=torch.float32, device=self.master_buf.device,
        )
        self._t6_gathered = torch.empty(
            self.shard_size * n_slots, dtype=torch.float32,
            device=self.master_buf.device,
        )
        self._t6_amax = torch.zeros_like(self._t6_partial)
        nvfp4_ops.set_weight_amax(self._t6_amax, slots)
        from flat.custom_ops import aten_tail_ctm as _aten_tail
        self._t6_dsl_zero = bool(
            _aten_tail.ENABLED
            and self._t6_partial.numel() % 4 == 0
            and self._t6_partial.data_ptr() % 16 == 0
        )
        if not self._t6_dsl_zero and _aten_tail.ENABLED and self.shard_rank == 0:
            print(
                f"WARNING: t6_partial numel={self._t6_partial.numel()} is not "
                f"DSL-fillable; the step keeps one at::native FillFunctor<float>",
                flush=True,
            )
        self._t6_ready = True

    def prepare_ag_overlap(self) -> None:
        """Stage the BF16 shard buffer and fire bucket 0's AG (call after optimizer step)."""
        if self._fp8_ag:
            self.fp8_shard_buf.copy_(self.master_buf)
            self._fp8_scratch_idx = 0
            self._next_ag_handle = self._fire_ag_fp8(0)
        else:
            if (
                self._t6_enabled
                and self._t6_ready is None
                and not torch.cuda.is_current_stream_capturing()
            ):
                self._t6_init()
            if self._t6_ready:
                if self._t6_dsl_zero:
                    torch.ops.flux.zero_f32_ctm(self._t6_partial)
                else:
                    self._t6_partial.zero_()
                torch.ops.flux.shard_copy_partial_amax(
                    self.master_buf, self.ag_shard_buf,
                    self._t6_partial, *self._t6_tables,
                )
            else:
                self.ag_shard_buf.copy_(self.master_buf)
            self._next_ag_handle = self._fire_ag_into_param_buffer(0)
            if self._t6_ready:
                self._t6_ag_handle = (
                    torch.ops._c10d_functional.all_gather_into_tensor_out(
                        self._t6_partial, self.shard_size,
                        self.shard_group_name, out=self._t6_gathered,
                    )
                )
        self._next_ag_bid = 0

    def wait_ag_for_block(self, old_block_id: int) -> None:
        """Wait the current bucket's AG and fire the next one (graph-safe).

        In FP8 mode the waited scratch is cast into param_buffer first.
        """
        if self._t6_ag_handle is not None and old_block_id >= 1:
            torch.ops._c10d_functional.wait_tensor(self._t6_ag_handle)
            torch.ops.flux.weight_amax_reduce(
                self._t6_gathered, self._t6_amax, self.shard_size,
            )
            self._t6_ag_handle = None
        new_bid = self._old_to_new_bucket.get(old_block_id)
        if new_bid is None or new_bid != self._next_ag_bid:
            return
        if not (new_bid == 0 and self._next_ag_handle is None):
            torch.ops._c10d_functional.wait_tensor(
                self._next_ag_handle,
            )
            if self._fp8_ag:
                # _fire_ag_fp8 already toggled the scratch index, so the data
                # is in the other buffer (idx ^ 1).
                b = self.buckets[new_bid]
                read_idx = self._fp8_scratch_idx ^ 1
                self.param_buffer[
                    b.grad_offset : b.grad_offset + b.padded_numel
                ].copy_(
                    self.fp8_scratch[read_idx][: b.padded_numel]
                )
        # Fire AG for next bucket (prefetch)
        next_bid = new_bid + 1
        if next_bid < len(self.buckets):
            if self._fp8_ag:
                self._next_ag_handle = self._fire_ag_fp8(next_bid)
            else:
                self._next_ag_handle = (
                    self._fire_ag_into_param_buffer(next_bid)
                )
            self._next_ag_bid = next_bid

    def finish_ag_overlap(self) -> None:
        """Wait for the last bucket's AG if not already handled."""
        last_bid = len(self.buckets) - 1
        if self._next_ag_bid == last_bid:
            torch.ops._c10d_functional.wait_tensor(
                self._next_ag_handle,
            )
            if self._fp8_ag:
                b = self.buckets[last_bid]
                read_idx = self._fp8_scratch_idx ^ 1
                self.param_buffer[
                    b.grad_offset : b.grad_offset + b.padded_numel
                ].copy_(
                    self.fp8_scratch[read_idx][: b.padded_numel]
                )

    def fire_stem_ag(self) -> None:
        """Fire the stem (bucket 0) all-gather on the comm stream.

        Call inside the captured body, after O_stem and before O_rest;
        finish_stem_ag must join it before the body ends.
        """
        b0 = self.buckets[0]
        so0 = b0.shard_offset
        sn0 = b0.shard_numel
        fork = torch.cuda.Event()
        fork.record()  # O_stem wrote master_buf[stem]
        self._stem_join_event = torch.cuda.Event()
        with torch.cuda.stream(self._comm_stream):
            self._comm_stream.wait_event(fork)  # fork: comm <- main
            if self._fp8_ag:
                self.fp8_shard_buf[so0 : so0 + sn0].copy_(
                    self.master_buf[so0 : so0 + sn0]
                )
                self._fp8_scratch_idx = 0
                handle = self._fire_ag_fp8(0)
                torch.ops._c10d_functional.wait_tensor(handle)
                read_idx = self._fp8_scratch_idx ^ 1
                self.param_buffer[
                    b0.grad_offset : b0.grad_offset + b0.padded_numel
                ].copy_(self.fp8_scratch[read_idx][: b0.padded_numel])
            else:
                self.ag_shard_buf[so0 : so0 + sn0].copy_(
                    self.master_buf[so0 : so0 + sn0]
                )
                handle = self._fire_ag_into_param_buffer(0)
                torch.ops._c10d_functional.wait_tensor(handle)
            # Join event, recorded on the comm stream after the AG.
            self._stem_join_event.record(self._comm_stream)
        self._next_ag_handle = None
        self._next_ag_bid = 0

    def finish_stem_ag(self) -> None:
        """Join the stem all-gather and stage buckets 1..N for the next forward.

        Call inside the captured body, right after O_rest.
        """
        torch.cuda.current_stream().wait_event(self._stem_join_event)
        # Stage buckets 1..N shard buffer (bucket 0 already staged/gathered).
        if len(self.buckets) > 1:
            rest = self.buckets[1].shard_offset
            self.ag_shard_buf[rest:].copy_(self.master_buf[rest:])
            if self._fp8_ag:
                self.fp8_shard_buf[rest:].copy_(self.master_buf[rest:])

    def prime_ag_buffers(self) -> None:
        """Populate param_buffer and the shard buffers before the first body.

        Runs outside graph capture, once, before the training loop.
        """
        b0 = self.buckets[0]
        self.ag_shard_buf.copy_(self.master_buf)
        if self._fp8_ag:
            self.fp8_shard_buf.copy_(self.master_buf)
            self._fp8_scratch_idx = 0
            handle = self._fire_ag_fp8(0)
            torch.ops._c10d_functional.wait_tensor(handle)
            read_idx = self._fp8_scratch_idx ^ 1
            self.param_buffer[
                b0.grad_offset : b0.grad_offset + b0.padded_numel
            ].copy_(self.fp8_scratch[read_idx][: b0.padded_numel])
        else:
            handle = self._fire_ag_into_param_buffer(0)
            torch.ops._c10d_functional.wait_tensor(handle)
        self._next_ag_handle = None
        self._next_ag_bid = 0

    def _init_master_weights(self) -> None:
        """Initialize FP32 master weight shards from model params.

        Params are placed at their param_views offsets so the shard layout
        matches the gathered buffer exactly.
        """
        for b in self.buckets:
            dtype = b.params[0].data.dtype
            padded = torch.zeros(
                b.padded_numel, dtype=dtype, device=self.device,
            )
            for (local_off, _shape), p in zip(b.param_views, b.params):
                n = p.numel()
                padded[local_off : local_off + n].copy_(
                    p.data.reshape(-1)
                )
            # Extract this rank's shard (position within its domain)
            start = self.shard_rank * b.shard_numel
            end = start + b.shard_numel
            shard = padded[start:end]
            soff = b.shard_offset
            self.master_buf[soff : soff + b.shard_numel].copy_(
                shard.float()
            )

    def _bucket_pbuf(self, bid: int) -> Tensor:
        """Return bucket bid's padded param region in param_buffer (bucket-local offsets)."""
        b = self.buckets[bid]
        return self.param_buffer[
            b.grad_offset : b.grad_offset + b.padded_numel
        ]

    def _setup_param_views(self) -> None:
        """Make each model param's .data a view into its bucket's param buffer."""
        for bid, b in enumerate(self.buckets):
            pbuf = self._bucket_pbuf(bid)
            for param, (local_off, shape) in zip(
                b.params, b.param_views
            ):
                numel = param.numel()
                pbuf[local_off : local_off + numel].copy_(
                    param.data.reshape(-1)
                )
                param.data = pbuf[
                    local_off : local_off + numel
                ].view(shape)

    # Gradient management

    def assign_grad_views(self) -> None:
        """Assign each param's .grad to a view of grad_buffer; call once at init.

        Never set param.grad = None between steps; use zero_grad() instead.
        """
        for b in self.buckets:
            for param, (local_off, shape) in zip(
                b.params, b.param_views
            ):
                start = b.grad_offset + local_off
                numel = param.numel()
                param.grad = self.grad_buffer[
                    start : start + numel
                ].view(shape)

    def zero_grad(self) -> None:
        """Zero the gradient buffer. Call at start of each step."""
        self.grad_buffer.zero_()

    def wait_all_reduce_scatters(self) -> None:
        """Drain reduced gradients into reduced_grad_buf (eager path, post-backward).

        Reductions are "sum"; the 1/world scaling is applied by the caller.
        """
        if self.is_hierarchical:
            while self._ar_order:
                self._drain_oldest_replicate()
        else:
            for bucket_id, rs in self.pending_rs.items():
                result = torch.ops._c10d_functional.wait_tensor(rs)
                b = self.buckets[bucket_id]
                soff = b.shard_offset
                sz = b.shard_numel
                self.reduced_grad_buf[soff : soff + sz].copy_(result[:sz])
            self.pending_rs.clear()

    # Gradient upcast + clipping

    def upcast_reduced_grads(self) -> None:
        """Upcast reduced bf16 grads into reduced_grad_f32.

        Call after wait_all_reduce_scatters() and before get_grad_norm_sq().
        """
        self.reduced_grad_f32.copy_(self.reduced_grad_buf)

    def get_grad_norm_sq(self) -> Tensor:
        """Squared L2 norm of all reduced gradient shards (one multi_tensor_l2norm call).

        Returns a device scalar; the caller must all-reduce it across ranks before sqrt.
        """
        shards = []
        for b in self.buckets:
            shards.append(
                self.reduced_grad_f32[
                    b.shard_offset : b.shard_offset + b.shard_numel
                ]
            )
        norm, _ = tex.multi_tensor_l2norm(
            65536, self._noop_flag, [shards], False,
        )
        return norm.squeeze().pow(2)

    def _ctm_optimizer_state(self) -> dict:
        """Chunk tables and scratch buffers for the DSL grad-norm / AdamW kernels."""
        if self._ctm_state is None:
            from flat.custom_ops import optimizer_ctm as oc

            shards = [(b.shard_offset, b.shard_numel) for b in self.buckets]
            base, length = oc.build_chunk_tables(shards, self.device)
            self._ctm_state = {
                "chunk_base": base,
                "chunk_len": length,
                "partial": torch.zeros(
                    base.numel(), dtype=torch.float32, device=self.device,
                ),
                "norm_sq": torch.zeros(
                    1, dtype=torch.float32, device=self.device,
                ),
                "bc_table": (
                    oc.build_bc_table(0.9, 0.95, oc.BC_MAX_STEP, self.device)
                    if oc.BC_MODE == 1
                    else torch.zeros(2, dtype=torch.float32, device=self.device)
                ),
            }
        return self._ctm_state

    def get_grad_norm_sq_ctm(self) -> Tensor:
        """Squared local gradient norm from the BF16 reduced buffer, DSL kernel.

        Reads reduced_grad_buf directly and scrubs NaN/Inf itself, so the caller
        can skip both the nan_to_num pass and the FP32 upcast.
        """
        st = self._ctm_optimizer_state()
        torch.ops.flux.grad_norm_sq_ctm(
            self.reduced_grad_buf, st["chunk_base"], st["chunk_len"],
            st["partial"], st["norm_sq"], self._noop_flag,
        )
        return st["norm_sq"].squeeze()

    def fused_adamw_step_ctm(
        self, beta1: float, beta2: float, eps: float, weight_decay: float,
    ) -> None:
        """Capturable fused AdamW over the whole shard, DSL kernel."""
        st = self._ctm_optimizer_state()
        torch.ops.flux.adamw_ctm(
            self.reduced_grad_buf, self.master_buf,
            self.opt_m_buf, self.opt_v_buf,
            self._lr_tensor, self._step_tensor, self._inv_scale,
            self._noop_flag, st["bc_table"],
            beta1, beta2, eps, weight_decay,
        )

    def clip_grads(self, max_norm: float, grad_norm: float) -> None:
        """Apply gradient clipping to all reduced shards."""
        if grad_norm > max_norm:
            clip_coef = max_norm / (grad_norm + 1e-6)
            for b in self.buckets:
                soff = b.shard_offset
                self.reduced_grad_buf[
                    soff : soff + b.shard_numel
                ].mul_(clip_coef)

    def clip_grads_gpu(self, max_norm: float, global_norm_sq: Tensor) -> None:
        """GPU-side gradient clipping on reduced_grad_f32 (no CPU sync).

        Call upcast_reduced_grads() first. The 1/world_size average is folded
        into the same in-place scaling (grads are sum-reduced).
        """
        # max_norm is defined against the mean gradient, so convert the summed
        # norm to mean units before computing clip_coef.
        mean_norm = (
            torch.ops.aten.sqrt.default(global_norm_sq) / self.world_size
        )
        clip_coef = torch.ops.aten.clamp.default(
            max_norm / (mean_norm + 1e-6),
            max=1.0,
        )
        self.reduced_grad_f32.mul_(clip_coef / self.world_size)

    # Optimizer

    def adamw_step(
        self,
        lr: float,
        beta1: float,
        beta2: float,
        eps: float,
        weight_decay: float,
        step: int,
    ) -> None:
        """Run AdamW on all bucket shards (FP32 master, FP32 grads); step is 1-indexed."""
        # reduced_grad_f32 already upcast + clipped
        for b in self.buckets:
            soff = b.shard_offset
            sz = b.shard_numel
            shard = self.master_buf[soff : soff + sz]
            grad = self.reduced_grad_f32[soff : soff + sz]
            m = self.opt_m_buf[soff : soff + sz]
            v = self.opt_v_buf[soff : soff + sz]

            shard.mul_(1.0 - lr * weight_decay)
            m.mul_(beta1).add_(grad, alpha=1 - beta1)
            v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            bc1 = 1.0 - beta1 ** step
            bc2 = 1.0 - beta2 ** step
            m_hat = m / bc1
            v_hat = v / bc2
            shard.addcdiv_(
                m_hat,
                torch.ops.aten.sqrt.default(v_hat).add_(eps),
                value=-lr,
            )

    def fused_adamw_step(
        self,
        lr: float,
        beta1: float,
        beta2: float,
        eps: float,
        weight_decay: float,
        step: int,
    ) -> None:
        """Run fused AdamW on all bucket shards via tex.multi_tensor_adam; step is 1-indexed."""
        params = []
        grads = []
        exp_avg = []
        exp_avg_sq = []
        for b in self.buckets:
            soff = b.shard_offset
            sz = b.shard_numel
            params.append(self.master_buf[soff : soff + sz])
            grads.append(self.reduced_grad_f32[soff : soff + sz])
            exp_avg.append(self.opt_m_buf[soff : soff + sz])
            exp_avg_sq.append(self.opt_v_buf[soff : soff + sz])

        # TE expects [g, p, m, v] ordering.
        tex.multi_tensor_adam(
            65536,  # chunk_size
            self._noop_flag,
            [grads, params, exp_avg, exp_avg_sq],
            lr,
            beta1,
            beta2,
            eps,
            step,
            1,  # mode: AdamW
            1,  # bias_correction
            weight_decay,
        )

    def fused_adamw_step_capturable(
        self,
        beta1: float,
        beta2: float,
        eps: float,
        weight_decay: float,
        bucket_ids: "list[int] | None" = None,
    ) -> None:
        """Run fused AdamW via the graph-capturable TE kernel.

        lr and step are read from _lr_tensor / _step_tensor; update them before replay.
        """
        if bucket_ids is None:
            selected = self.buckets
        else:
            selected = [self.buckets[i] for i in bucket_ids]
        if not selected:
            return
        params = []
        grads = []
        exp_avg = []
        exp_avg_sq = []
        for b in selected:
            soff = b.shard_offset
            sz = b.shard_numel
            params.append(self.master_buf[soff : soff + sz])
            grads.append(self.reduced_grad_f32[soff : soff + sz])
            exp_avg.append(self.opt_m_buf[soff : soff + sz])
            exp_avg_sq.append(self.opt_v_buf[soff : soff + sz])

        tex.multi_tensor_adam_capturable(
            65536,  # chunk_size
            self._noop_flag,
            [grads, params, exp_avg, exp_avg_sq],
            self._lr_tensor,
            beta1,
            beta2,
            eps,
            self._step_tensor,
            1,  # mode: AdamW
            1,  # bias_correction
            weight_decay,
            self._inv_scale,
        )

    def reduce_scatter_all_inplace(self) -> None:
        """Graph-safe in-place reduce-scatter into reduced_grad_buf (D == 1 only).

        Reductions are "sum"; the 1/world scaling is applied by the caller.
        """
        assert not self.is_hierarchical, (
            "reduce_scatter_all_inplace lacks the inter-domain all-reduce; "
            "it is only valid at D==1. Use fire_rs_for_block_graphsafe."
        )
        shard_group = dist.distributed_c10d._resolve_process_group(
            self.shard_group_name
        )
        for b in self.buckets:
            output = self.reduced_grad_buf[
                b.shard_offset : b.shard_offset + b.shard_numel
            ]
            input_t = self.grad_buffer[
                b.grad_offset : b.grad_offset + b.padded_numel
            ]
            dist.reduce_scatter_tensor(
                output, input_t, op=dist.ReduceOp.SUM, group=shard_group,
            )

    # Communication: all-gather

    def fire_all_gathers(self) -> None:
        """Fire async all-gathers for all buckets after the optimizer step."""
        self.ag_shard_buf.copy_(self.master_buf)
        for bucket_id, b in enumerate(self.buckets):
            soff = b.shard_offset
            shard_bf16 = self.ag_shard_buf[
                soff : soff + b.shard_numel
            ]
            ag = torch.ops._c10d_functional.all_gather_into_tensor(
                shard_bf16, self.shard_size, self.shard_group_name,
            )
            self.pending_ag[bucket_id] = ag

    def fire_all_gathers_sync(self) -> None:
        """Synchronous all-gather: fire, wait and copy one bucket at a time."""
        self.ag_shard_buf.copy_(self.master_buf)
        for b in self.buckets:
            soff = b.shard_offset
            shard_bf16 = self.ag_shard_buf[
                soff : soff + b.shard_numel
            ]
            ag = torch.ops._c10d_functional.all_gather_into_tensor(
                shard_bf16, self.shard_size, self.shard_group_name,
            )
            gathered = torch.ops._c10d_functional.wait_tensor(ag)
            self.param_buffer[
                b.grad_offset : b.grad_offset + b.raw_numel
            ].copy_(gathered[: b.raw_numel])

    def fire_all_gathers_inplace(self) -> None:
        """All-gather directly into param_buffer slices (no temporaries)."""
        shard_group = dist.distributed_c10d._resolve_process_group(
            self.shard_group_name
        )
        self.ag_shard_buf.copy_(self.master_buf)
        for b in self.buckets:
            soff = b.shard_offset
            shard_bf16 = self.ag_shard_buf[
                soff : soff + b.shard_numel
            ]
            # Output: padded_numel = shard_numel * shard_size (G)
            output = self.param_buffer[
                b.grad_offset : b.grad_offset + b.padded_numel
            ]
            dist.all_gather_into_tensor(
                output, shard_bf16, group=shard_group,
            )
        # param_buffer now has gathered params; model params are views

    def wait_all_gathers_and_copy(self) -> None:
        """Wait all pending all-gathers and copy each into param_buffer."""
        for bucket_id, ag in self.pending_ag.items():
            gathered = torch.ops._c10d_functional.wait_tensor(ag)
            b = self.buckets[bucket_id]
            self.param_buffer[
                b.grad_offset : b.grad_offset + b.raw_numel
            ].copy_(gathered[: b.raw_numel])
        self.pending_ag.clear()

    def wait_ag_bucket(self, bucket_id: int) -> None:
        """Wait one bucket's all-gather and copy its params into param_buffer."""
        ag = self.pending_ag.pop(bucket_id, None)
        if ag is None:
            return
        gathered = torch.ops._c10d_functional.wait_tensor(ag)
        b = self.buckets[bucket_id]
        self.param_buffer[
            b.grad_offset : b.grad_offset + b.raw_numel
        ].copy_(gathered[: b.raw_numel])

    # Single-GPU fallback

    def copy_grads_to_reduced(self) -> None:
        """Single-GPU: copy this rank's grad shards into reduced_grad_buf without comm."""
        for b in self.buckets:
            src_start = b.grad_offset + self.shard_rank * b.shard_numel
            src = self.grad_buffer[
                src_start : src_start + b.shard_numel
            ]
            soff = b.shard_offset
            self.reduced_grad_buf[
                soff : soff + b.shard_numel
            ].copy_(src)

    def copy_master_to_params(self) -> None:
        """Single-GPU: copy master weights into param_buffer without communication."""
        for b in self.buckets:
            soff = b.shard_offset
            # In single-GPU, shard == full params, padded_numel == raw_numel
            self.param_buffer[
                b.grad_offset : b.grad_offset + b.raw_numel
            ].copy_(
                self.master_buf[soff : soff + b.raw_numel]
            )
