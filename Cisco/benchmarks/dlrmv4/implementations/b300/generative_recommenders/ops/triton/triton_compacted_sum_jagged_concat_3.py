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

             

from typing import NamedTuple, Optional, Tuple

import torch
import triton
import triton.language as tl
from generative_recommenders.common import switch_to_contiguous_if_needed


_REDUCTION_CHUNK_SIZE = 512


class _SegmentReductionPlan(NamedTuple):
    ''                                                               

    ordered_grad_rows: torch.Tensor
    segment_offsets: torch.Tensor
    chunk_offsets: torch.Tensor
    chunk_rows: torch.Tensor


@triton.jit
def _count_two_routes(
    Reverse0,
    Reverse1,
    Counts0,
    Counts1,
    M,
    BLOCK_M: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = offsets < M
    rows_0 = tl.load(Reverse0 + offsets, mask=mask, other=0)
    rows_1 = tl.load(Reverse1 + offsets, mask=mask, other=0)
    tl.atomic_add(Counts0 + rows_0, 1, mask=mask)
    tl.atomic_add(Counts1 + rows_1, 1, mask=mask)


@triton.jit
def _scatter_two_routes(
    Reverse0,
    Reverse1,
    OccurrenceGradRows,
    Cursors0,
    Cursors1,
    OrderedGradRows0,
    OrderedGradRows1,
    M,
    BLOCK_M: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = offsets < M
    rows_0 = tl.load(Reverse0 + offsets, mask=mask, other=0)
    rows_1 = tl.load(Reverse1 + offsets, mask=mask, other=0)
    grad_rows = tl.load(OccurrenceGradRows + offsets, mask=mask, other=0)
    destinations_0 = tl.atomic_add(Cursors0 + rows_0, 1, mask=mask)
    destinations_1 = tl.atomic_add(Cursors1 + rows_1, 1, mask=mask)
    tl.store(OrderedGradRows0 + destinations_0, grad_rows, mask=mask)
    tl.store(OrderedGradRows1 + destinations_1, grad_rows, mask=mask)


@triton.jit
def _mark_active_routes(
    Reverse,
    ActiveRows,
    M,
    RowCount,
    BLOCK_M: tl.constexpr,
    MAX_ROWS: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = offsets < M
    routes = tl.load(Reverse + offsets, mask=mask, other=-1)
    for row in tl.static_range(0, MAX_ROWS):
        matches = tl.sum((mask & (routes == row)).to(tl.int32), axis=0)
        tl.atomic_xchg(
            ActiveRows + row,
            1,
            mask=(row < RowCount) & (matches > 0),
        )


def _build_active_routes(
    reverse: torch.Tensor,
    row_count: int,
) -> torch.Tensor:
    if row_count > 16:
        raise RuntimeError("small-domain reduction supports at most 16 rows")
    active_rows = torch.zeros(
        (row_count,), dtype=torch.int32, device=reverse.device
    )
    block_m = 256
    _mark_active_routes[(triton.cdiv(reverse.numel(), block_m),)](
        reverse,
        active_rows,
        reverse.numel(),
        row_count,
        BLOCK_M=block_m,
        MAX_ROWS=16,
        num_warps=8,
    )
    return active_rows


@triton.jit
def _fill_chunk_rows(ChunkOffsets, ChunkRows, RowCount):
    row = tl.program_id(0)
    valid_row = row < RowCount
    chunk_start = tl.load(ChunkOffsets + row, mask=valid_row, other=0)
    chunk_end = tl.load(ChunkOffsets + row + 1, mask=valid_row, other=0)
    for chunk in tl.range(chunk_start, chunk_end):
        tl.store(ChunkRows + chunk, row)


def _segment_metadata_from_counts(
    counts: torch.Tensor,
    occurrence_count: int,
    chunk_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    segment_offsets = torch.cat(
        [
            counts.new_zeros((1,)),
            torch.cumsum(counts, dim=0, dtype=torch.int32),
        ]
    )
    chunks_per_row = torch.div(
        counts + chunk_size - 1,
        chunk_size,
        rounding_mode="floor",
    )
    chunk_offsets = torch.cat(
        [
            chunks_per_row.new_zeros((1,)),
            torch.cumsum(chunks_per_row, dim=0, dtype=torch.int32),
        ]
    )
    chunk_rows = torch.empty(
        (triton.cdiv(occurrence_count, chunk_size) + counts.numel(),),
        dtype=torch.int32,
        device=counts.device,
    )
    _fill_chunk_rows[(counts.numel(),)](
        chunk_offsets,
        chunk_rows,
        counts.numel(),
        num_warps=1,
    )
    return segment_offsets, chunk_offsets, chunk_rows


def _build_counting_segment_reduction_plans(
    reverse_0: torch.Tensor,
    row_count_0: int,
    reverse_1: torch.Tensor,
    row_count_1: int,
    occurrence_grad_rows: torch.Tensor,
    chunk_size: int = _REDUCTION_CHUNK_SIZE,
) -> Tuple[_SegmentReductionPlan, _SegmentReductionPlan]:
    M = reverse_0.numel()
    counts_0 = torch.zeros(
        (row_count_0,), dtype=torch.int32, device=reverse_0.device
    )
    counts_1 = torch.zeros(
        (row_count_1,), dtype=torch.int32, device=reverse_1.device
    )
    block_m = 256
    _count_two_routes[(triton.cdiv(M, block_m),)](
        reverse_0,
        reverse_1,
        counts_0,
        counts_1,
        M,
        BLOCK_M=block_m,
        num_warps=8,
    )
    segment_offsets_0, chunk_offsets_0, chunk_rows_0 = (
        _segment_metadata_from_counts(counts_0, M, chunk_size)
    )
    segment_offsets_1, chunk_offsets_1, chunk_rows_1 = (
        _segment_metadata_from_counts(counts_1, M, chunk_size)
    )
    cursors_0 = segment_offsets_0[:-1].clone()
    cursors_1 = segment_offsets_1[:-1].clone()
    ordered_grad_rows_0 = torch.empty_like(occurrence_grad_rows)
    ordered_grad_rows_1 = torch.empty_like(occurrence_grad_rows)
    _scatter_two_routes[(triton.cdiv(M, block_m),)](
        reverse_0,
        reverse_1,
        occurrence_grad_rows,
        cursors_0,
        cursors_1,
        ordered_grad_rows_0,
        ordered_grad_rows_1,
        M,
        BLOCK_M=block_m,
        num_warps=8,
    )
    return (
        _SegmentReductionPlan(
            ordered_grad_rows_0,
            segment_offsets_0,
            chunk_offsets_0,
            chunk_rows_0,
        ),
        _SegmentReductionPlan(
            ordered_grad_rows_1,
            segment_offsets_1,
            chunk_offsets_1,
            chunk_rows_1,
        ),
    )


@triton.jit
def _compacted_sum_jagged_concat_3_fwd(
    Dense,
    Component0,
    Component1,
    Component2,
    Reverse0,
    Reverse1,
    Reverse2,
    SequenceOffsets,
    Out,
    D: tl.constexpr,
    DENSE_LEN: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    DENSE_STRIDE_B: tl.constexpr,
    DENSE_STRIDE_L: tl.constexpr,
    DENSE_STRIDE_D: tl.constexpr,
    DENSE_FIRST: tl.constexpr = True,
):
    batch = tl.program_id(1)
    rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    sequence_start = tl.load(SequenceOffsets + batch)
    sequence_end = tl.load(SequenceOffsets + batch + 1)
    sequence_len = sequence_end - sequence_start
    total_len = DENSE_LEN + sequence_len
    d = tl.arange(0, BLOCK_D)
    valid_rows = rows < total_len
    d_mask = d[None, :] < D
    dense_positions = rows - (0 if DENSE_FIRST else sequence_len)
    sequence_positions = rows - (DENSE_LEN if DENSE_FIRST else 0)
    dense_mask = (dense_positions >= 0) & (dense_positions < DENSE_LEN)
    sequence_mask = (sequence_positions >= 0) & (sequence_positions < sequence_len)

    if DENSE_LEN > 0:
        dense_values = tl.load(
            Dense
            + batch.to(tl.int64) * DENSE_STRIDE_B
            + dense_positions[:, None].to(tl.int64) * DENSE_STRIDE_L
            + d[None, :] * DENSE_STRIDE_D,
            mask=dense_mask[:, None] & d_mask,
            other=0.0,
        )
    else:
        dense_values = 0.0

    occurrences = sequence_start + sequence_positions
    rows_0 = tl.load(
        Reverse0 + occurrences,
        mask=sequence_mask,
        other=0,
    )
    rows_1 = tl.load(
        Reverse1 + occurrences,
        mask=sequence_mask,
        other=0,
    )
    rows_2 = tl.load(
        Reverse2 + occurrences,
        mask=sequence_mask,
        other=0,
    )
    sequence_values = (
        tl.load(
            Component0 + rows_0[:, None].to(tl.int64) * D + d[None, :],
            mask=sequence_mask[:, None] & d_mask,
            other=0.0,
        )
        + tl.load(
            Component1
            + rows_1[:, None].to(tl.int64) * D
            + d[None, :],
            mask=sequence_mask[:, None] & d_mask,
            other=0.0,
        )
        + tl.load(
            Component2 + rows_2[:, None].to(tl.int64) * D + d[None, :],
            mask=sequence_mask[:, None] & d_mask,
            other=0.0,
        )
    )
    output_rows = batch * DENSE_LEN + sequence_start + rows
    output_ptrs = Out + output_rows[:, None].to(tl.int64) * D + d[None, :]
    tl.store(
        output_ptrs,
        dense_values + sequence_values,
        mask=valid_rows[:, None] & d_mask,
    )


@triton.jit
def _compacted_sum_jagged_concat_3_bwd(
    GradOut,
    Reverse0,
    Reverse1,
    SequenceOffsets,
    GradDense,
    GradComponent0,
    GradComponent1,
    GradComponent2Occurrences,
    D: tl.constexpr,
    DENSE_LEN: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    DENSE_FIRST: tl.constexpr = True,
):
    batch = tl.program_id(1)
    rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    sequence_start = tl.load(SequenceOffsets + batch)
    sequence_end = tl.load(SequenceOffsets + batch + 1)
    sequence_len = sequence_end - sequence_start
    total_len = DENSE_LEN + sequence_len
    d = tl.arange(0, BLOCK_D)
    d_mask = d[None, :] < D
    valid_rows = rows < total_len
    dense_positions = rows - (0 if DENSE_FIRST else sequence_len)
    sequence_positions = rows - (DENSE_LEN if DENSE_FIRST else 0)
    dense_mask = (dense_positions >= 0) & (dense_positions < DENSE_LEN)
    sequence_mask = (sequence_positions >= 0) & (sequence_positions < sequence_len)
    output_rows = batch * DENSE_LEN + sequence_start + rows
    grad = tl.load(
        GradOut + output_rows[:, None].to(tl.int64) * D + d[None, :],
        mask=valid_rows[:, None] & d_mask,
        other=0.0,
    )

    if DENSE_LEN > 0:
        dense_rows = batch * DENSE_LEN + dense_positions
        tl.store(
            GradDense + dense_rows[:, None].to(tl.int64) * D + d[None, :],
            grad,
            mask=dense_mask[:, None] & d_mask,
        )

    occurrences = sequence_start + sequence_positions
    rows_0 = tl.load(
        Reverse0 + occurrences,
        mask=sequence_mask,
        other=0,
    )
    rows_1 = tl.load(
        Reverse1 + occurrences,
        mask=sequence_mask,
        other=0,
    )
    tl.atomic_add(
        GradComponent0 + rows_0[:, None].to(tl.int64) * D + d[None, :],
        grad,
        mask=sequence_mask[:, None] & d_mask,
    )
    tl.atomic_add(
        GradComponent1
        + rows_1[:, None].to(tl.int64) * D
        + d[None, :],
        grad,
        mask=sequence_mask[:, None] & d_mask,
    )
    tl.store(
        GradComponent2Occurrences
        + occurrences[:, None].to(tl.int64) * D
        + d[None, :],
        grad,
        mask=sequence_mask[:, None] & d_mask,
    )


@triton.jit
def _extract_dense_grad(
    GradOut,
    SequenceOffsets,
    GradDense,
    D: tl.constexpr,
    DENSE_LEN: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    DENSE_FIRST: tl.constexpr = True,
):
    batch = tl.program_id(1)
    rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    sequence_start = tl.load(SequenceOffsets + batch + (0 if DENSE_FIRST else 1))
    d = tl.arange(0, BLOCK_D)
    dense_mask = rows < DENSE_LEN
    output_rows = batch * DENSE_LEN + sequence_start + rows
    grad = tl.load(
        GradOut + output_rows[:, None].to(tl.int64) * D + d[None, :],
        mask=dense_mask[:, None] & (d[None, :] < D),
        other=0.0,
    )
    dense_rows = batch * DENSE_LEN + rows
    tl.store(
        GradDense + dense_rows[:, None].to(tl.int64) * D + d[None, :],
        grad,
        mask=dense_mask[:, None] & (d[None, :] < D),
    )


@triton.jit
def _reduce_small_domain_sequence_grad(
    GradOut,
    OccurrenceGradRows,
    Reverse,
    ActiveRows,
    GradComponent,
    M,
    D: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
):
    ''                                                                       

    compacted_row = tl.program_id(0)
    chunk_start = tl.program_id(1) * CHUNK_SIZE
    d = tl.program_id(2) * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d < D
    row_is_active = tl.load(ActiveRows + compacted_row) != 0
    chunk_length = tl.where(
        row_is_active,
        tl.minimum(M - chunk_start, CHUNK_SIZE),
        0,
    )
    lane = tl.arange(0, BLOCK_M)
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for offset in tl.range(
        0, chunk_length, BLOCK_M, num_stages=PIPELINE_STAGES
    ):
        occurrence = chunk_start + offset + lane
        valid = occurrence < M
        route = tl.load(Reverse + occurrence, mask=valid, other=-1)
        output_row = tl.load(
            OccurrenceGradRows + occurrence,
            mask=valid,
            other=0,
        )
        values = tl.load(
            GradOut + output_row[:, None].to(tl.int64) * D + d[None, :],
            mask=(valid & (route == compacted_row))[:, None]
            & d_mask[None, :],
            other=0.0,
        )
        acc += tl.sum(values, axis=0)
    tl.atomic_add(
        GradComponent + compacted_row.to(tl.int64) * D + d,
        acc,
        mask=row_is_active & (chunk_start < M) & d_mask,
    )


def _small_domain_reduce_sequence_grad(
    grad_output: torch.Tensor,
    occurrence_grad_rows: torch.Tensor,
    reverse: torch.Tensor,
    active_rows: torch.Tensor,
    grad_component: torch.Tensor,
    chunk_size: int = 2048,
    block_m: int = 16,
    block_d: int = 512,
    pipeline_stages: int = 3,
    num_warps: int = 4,
) -> None:
    _reduce_small_domain_sequence_grad[
        (
            grad_component.size(0),
            triton.cdiv(reverse.numel(), chunk_size),
            triton.cdiv(grad_component.size(1), block_d),
        )
    ](
        grad_output,
        occurrence_grad_rows,
        reverse,
        active_rows,
        grad_component,
        reverse.numel(),
        D=grad_component.size(1),
        CHUNK_SIZE=chunk_size,
        BLOCK_M=block_m,
        BLOCK_D=block_d,
        PIPELINE_STAGES=pipeline_stages,
        num_warps=num_warps,
    )


@triton.jit
def _chunk_reduce_sequence_grad(
    GradOut,
    OrderedGradRows,
    SegmentOffsets,
    ChunkOffsets,
    ChunkRows,
    GradComponent,
    RowCount,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
):
    ''                                          

    chunk = tl.program_id(0)
    d = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d < D
    total_chunks = tl.load(ChunkOffsets + RowCount)
    valid_chunk = chunk < total_chunks
    target_chunk = tl.where(valid_chunk, chunk, 0)
    compacted_row = tl.load(
        ChunkRows + target_chunk,
        mask=valid_chunk,
        other=0,
    )
    row_chunk_start = tl.load(ChunkOffsets + compacted_row)
    row_chunk_end = tl.load(ChunkOffsets + compacted_row + 1)
    row_start = tl.load(SegmentOffsets + compacted_row)
    row_end = tl.load(SegmentOffsets + compacted_row + 1)
    chunk_start = (
        row_start + (target_chunk - row_chunk_start) * CHUNK_SIZE
    )
    chunk_length = tl.minimum(row_end - chunk_start, CHUNK_SIZE)
    lane = tl.arange(0, BLOCK_M)
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for offset in tl.range(
        0, chunk_length, BLOCK_M, num_stages=PIPELINE_STAGES
    ):
        local = offset + lane
        mask = valid_chunk & (local < chunk_length)
        output_row = tl.load(
            OrderedGradRows + chunk_start + local,
            mask=mask,
            other=0,
        )
        values = tl.load(
            GradOut
            + output_row[:, None].to(tl.int64) * D
            + d[None, :],
            mask=mask[:, None] & d_mask[None, :],
            other=0.0,
        )
        acc += tl.sum(values, axis=0)

    single_chunk = (row_chunk_end - row_chunk_start) == 1
    tl.store(
        GradComponent + compacted_row.to(tl.int64) * D + d,
        acc,
        mask=valid_chunk & single_chunk & d_mask,
    )
    tl.atomic_add(
        GradComponent + compacted_row.to(tl.int64) * D + d,
        acc,
        mask=valid_chunk & (single_chunk == 0) & d_mask,
    )


def _segmented_reduce_sequence_grad(
    grad_output: torch.Tensor,
    plan: _SegmentReductionPlan,
    grad_component: torch.Tensor,
    row_count: int,
    chunk_size: int = _REDUCTION_CHUNK_SIZE,
    block_m: int = 16,
    block_d: int = 512,
    pipeline_stages: int = 3,
    num_warps: int = 4,
) -> None:
    max_chunks = (
        triton.cdiv(plan.ordered_grad_rows.numel(), chunk_size) + row_count
    )
    _chunk_reduce_sequence_grad[
        (
            max_chunks,
            triton.cdiv(grad_component.size(1), block_d),
        )
    ](
        grad_output,
        plan.ordered_grad_rows,
        plan.segment_offsets,
        plan.chunk_offsets,
        plan.chunk_rows,
        grad_component,
        row_count,
        D=grad_component.size(1),
        BLOCK_M=block_m,
        BLOCK_D=block_d,
        CHUNK_SIZE=chunk_size,
        PIPELINE_STAGES=pipeline_stages,
        num_warps=num_warps,
    )


class _CompactedSumJaggedConcat3(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        dense: Optional[torch.Tensor],
        component_0: torch.Tensor,
        component_1: torch.Tensor,
        component_2: torch.Tensor,
        reverse_0: torch.Tensor,
        reverse_1: torch.Tensor,
        reverse_2: torch.Tensor,
        sequence_offsets: torch.Tensor,
        max_sequence_len: int,
        dense_first: bool,
    ) -> torch.Tensor:
        tensors = [
            dense,
            component_0,
            component_1,
            component_2,
        ]
        tensors[1:] = [
            switch_to_contiguous_if_needed(tensor) for tensor in tensors[1:]
        ]
        (
            dense,
            component_0,
            component_1,
            component_2,
        ) = tensors
        D = component_0.shape[1]
        if any(
            component.dim() != 2 or component.size(1) != D
            for component in (component_1, component_2)
        ):
            raise RuntimeError("all components must have shape [rows, D]")
        sequence_count = reverse_0.numel()
        if any(route.numel() != sequence_count for route in (reverse_1, reverse_2)):
            raise RuntimeError("component routes must align with the sequence")
        B = sequence_offsets.numel() - 1
        if dense is not None and (
            dense.dim() != 3 or dense.size(0) != B or dense.size(2) != D
        ):
            raise RuntimeError("dense must have shape [B, L, D]")
        dense_len = dense.size(1) if dense is not None else 0
        dense_rows = B * dense_len
        output = component_0.new_empty(
            (dense_rows + sequence_count, D)
        )
        block_d = triton.next_power_of_2(D)
        block_n = 16
        _compacted_sum_jagged_concat_3_fwd[
            (
                triton.cdiv(
                    dense_len + max_sequence_len,
                    block_n,
                ),
                B,
            )
        ](
            dense,
            component_0,
            component_1,
            component_2,
            reverse_0,
            reverse_1,
            reverse_2,
            sequence_offsets,
            output,
            D=D,
            DENSE_LEN=dense_len,
            DENSE_FIRST=dense_first,
            DENSE_STRIDE_B=dense.stride(0) if dense is not None else 0,
            DENSE_STRIDE_L=dense.stride(1) if dense is not None else 0,
            DENSE_STRIDE_D=dense.stride(2) if dense is not None else 0,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            num_warps=4,
        )
        saved_tensors = [
            reverse_0,
            reverse_1,
            reverse_2,
            sequence_offsets,
        ]
                                                                              
        ctx.use_segment_reduction = (
            ctx.needs_input_grad[1] or ctx.needs_input_grad[2]
        )
        if ctx.use_segment_reduction:
            sequence_lengths = sequence_offsets[1:] - sequence_offsets[:-1]
            occurrence_batches = torch.repeat_interleave(
                torch.arange(B, device=sequence_offsets.device),
                sequence_lengths,
                output_size=sequence_count,
            )
            occurrence_grad_rows = (
                torch.arange(
                    sequence_count,
                    dtype=torch.int32,
                    device=sequence_offsets.device,
                )
                + (occurrence_batches.to(torch.int32) + int(dense_first))
                * dense_len
            )
            component_2_active_rows = (
                _build_active_routes(reverse_2, component_2.size(0))
                if component_2.size(0) <= 16
                else reverse_2.new_empty((0,), dtype=torch.int32)
            )
            plan_0, plan_1 = _build_counting_segment_reduction_plans(
                reverse_0,
                component_0.size(0),
                reverse_1,
                component_1.size(0),
                occurrence_grad_rows,
            )
            saved_tensors.append(occurrence_grad_rows)
            saved_tensors.extend(plan_0)
            saved_tensors.extend(plan_1)
            saved_tensors.append(component_2_active_rows)
        ctx.save_for_backward(*saved_tensors)
        ctx.input_shapes = [
            tensor.shape if tensor is not None else None for tensor in tensors
        ]
        ctx.max_sequence_len = max_sequence_len
        ctx.dense_len = dense_len
        ctx.dense_first = dense_first
        ctx.component_2_rows = component_2.size(0)
        ctx.component_0_rows = component_0.size(0)
        ctx.component_1_rows = component_1.size(0)
        ctx.D = D
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        saved_tensors = ctx.saved_tensors
        (
            reverse_0,
            reverse_1,
            reverse_2,
            sequence_offsets,
        ) = saved_tensors[:4]
        grad_dense = None
        if ctx.input_shapes[0] is not None:
            grad_dense = (
                grad_output.new_empty(ctx.input_shapes[0])
                if ctx.use_segment_reduction
                else grad_output.new_zeros(ctx.input_shapes[0])
            )
        grad_component_0 = grad_output.new_zeros(ctx.input_shapes[1])
        grad_component_1 = grad_output.new_zeros(ctx.input_shapes[2])
        B = sequence_offsets.numel() - 1
        block_d = triton.next_power_of_2(ctx.D)
        block_n = 16 if ctx.use_segment_reduction else 4
        grad_output = switch_to_contiguous_if_needed(grad_output)
        grid = (
            triton.cdiv(
                ctx.dense_len + ctx.max_sequence_len,
                block_n,
            ),
            B,
        )
        if ctx.use_segment_reduction:
            occurrence_grad_rows = saved_tensors[4]
            plan_0 = _SegmentReductionPlan(*saved_tensors[5:9])
            plan_1 = _SegmentReductionPlan(*saved_tensors[9:13])
            component_2_active_rows = saved_tensors[13]
            if ctx.dense_len > 0:
                _extract_dense_grad[
                    (
                        triton.cdiv(ctx.dense_len, block_n),
                        B,
                    )
                ](
                    grad_output,
                    sequence_offsets,
                    grad_dense,
                    D=ctx.D,
                    DENSE_LEN=ctx.dense_len,
                    DENSE_FIRST=ctx.dense_first,
                    BLOCK_N=block_n,
                    BLOCK_D=block_d,
                    num_warps=8 if block_d >= 512 else 4,
                )
            for plan, row_count, grad_component in (
                (plan_0, ctx.component_0_rows, grad_component_0),
                (plan_1, ctx.component_1_rows, grad_component_1),
            ):
                _segmented_reduce_sequence_grad(
                    grad_output=grad_output,
                    plan=plan,
                    grad_component=grad_component,
                    row_count=row_count,
                    chunk_size=_REDUCTION_CHUNK_SIZE,
                    block_m=16,
                    block_d=block_d,
                    pipeline_stages=3,
                    num_warps=4,
                )
            if component_2_active_rows.numel() > 0:
                grad_component_2 = grad_output.new_zeros(ctx.input_shapes[3])
                _small_domain_reduce_sequence_grad(
                    grad_output=grad_output,
                    occurrence_grad_rows=occurrence_grad_rows,
                    reverse=reverse_2,
                    active_rows=component_2_active_rows,
                    grad_component=grad_component_2,
                    block_d=block_d,
                )
            else:
                grad_component_2_occurrences = torch.index_select(
                    grad_output,
                    0,
                    occurrence_grad_rows.to(torch.int64),
                )
                component_2_mask = torch.nn.functional.one_hot(
                    reverse_2.to(torch.int64),
                    num_classes=ctx.component_2_rows,
                ).to(grad_component_2_occurrences.dtype)
                grad_component_2 = torch.mm(
                    component_2_mask.transpose(0, 1),
                    grad_component_2_occurrences,
                )
        else:
            grad_component_2_occurrences = grad_output.new_empty(
                (reverse_0.numel(), ctx.D)
            )
            _compacted_sum_jagged_concat_3_bwd[grid](
                grad_output,
                reverse_0,
                reverse_1,
                sequence_offsets,
                grad_dense,
                grad_component_0,
                grad_component_1,
                grad_component_2_occurrences,
                D=ctx.D,
                DENSE_LEN=ctx.dense_len,
                DENSE_FIRST=ctx.dense_first,
                BLOCK_N=block_n,
                BLOCK_D=block_d,
                num_warps=8 if block_d >= 512 else 4,
            )
            component_2_mask = torch.nn.functional.one_hot(
                reverse_2.to(torch.int64),
                num_classes=ctx.component_2_rows,
            ).to(grad_component_2_occurrences.dtype)
            grad_component_2 = torch.mm(
                component_2_mask.transpose(0, 1),
                grad_component_2_occurrences,
            )
        return (
            grad_dense,
            grad_component_0,
            grad_component_1,
            grad_component_2,
            None,
            None,
            None,
            None,
            None,
            None,
        )


@torch.fx.wrap
def triton_compacted_sum_jagged_concat_3(
    dense: Optional[torch.Tensor],
    component_0: torch.Tensor,
    component_1: torch.Tensor,
    component_2: torch.Tensor,
    reverse_0: torch.Tensor,
    reverse_1: torch.Tensor,
    reverse_2: torch.Tensor,
    sequence_offsets: torch.Tensor,
    max_sequence_len: int,
    dense_first: bool = True,
) -> torch.Tensor:
    return _CompactedSumJaggedConcat3.apply(
        dense,
        component_0,
        component_1,
        component_2,
        reverse_0,
        reverse_1,
        reverse_2,
        sequence_offsets,
        max_sequence_len,
        dense_first,
    )
