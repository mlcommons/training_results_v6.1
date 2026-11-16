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

# pyre-unsafe

from typing import List

import torch
import triton
import triton.language as tl
from generative_recommenders.common import switch_to_contiguous_if_needed


@triton.jit
def _gather_concat_fwd(
    Inputs,
    RowIndices,
    Out,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    value = tl.full((BLOCK,), 0, Out.dtype.element_ty)
    for index in tl.static_range(N):
        input_row = tl.load(RowIndices + row * N + index).to(tl.int64)
        input_col = col - index * D
        value += tl.load(
            Inputs[index] + input_row * D + input_col,
            mask=(input_col >= 0) & (input_col < D),
            other=0.0,
        )
    tl.store(
        Out + row.to(tl.int64) * (N * D) + col,
        value,
        mask=col < N * D,
    )


@triton.jit
def _gather_concat_bwd(
    GradOut,
    RowIndices,
    GradInputs,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    grad = tl.load(
        GradOut + row.to(tl.int64) * (N * D) + col,
        mask=col < N * D,
        other=0.0,
    )
    for index in tl.static_range(N):
        input_row = tl.load(RowIndices + row * N + index).to(tl.int64)
        input_col = col - index * D
        tl.atomic_add(
            GradInputs[index] + input_row * D + input_col,
            grad,
            mask=(input_col >= 0) & (input_col < D),
        )


class _GatherConcat(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        row_indices: torch.Tensor,
        *tensors: torch.Tensor,
    ) -> torch.Tensor:
        tensors = tuple(switch_to_contiguous_if_needed(tensor) for tensor in tensors)
        row_indices = switch_to_contiguous_if_needed(row_indices)
        N = len(tensors)
        D = tensors[0].shape[1]
        output = tensors[0].new_empty((row_indices.shape[0], N * D))
        ctx.block = triton.next_power_of_2(2 * D) if N == 2 else 256
        ctx.num_warps = 8 if N == 2 and ctx.block >= 512 else 4
        ctx.grid = (row_indices.shape[0], triton.cdiv(N * D, ctx.block))
        _gather_concat_fwd[ctx.grid](
            tensors,
            row_indices,
            output,
            N=N,
            D=D,
            BLOCK=ctx.block,
            num_warps=ctx.num_warps,
        )
        ctx.save_for_backward(row_indices)
        ctx.input_shapes = [tensor.shape for tensor in tensors]
        ctx.D = D
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (row_indices,) = ctx.saved_tensors
        grad_inputs = tuple(
            grad_output.new_zeros(shape) for shape in ctx.input_shapes
        )
        _gather_concat_bwd[ctx.grid](
            switch_to_contiguous_if_needed(grad_output),
            row_indices,
            grad_inputs,
            N=len(grad_inputs),
            D=ctx.D,
            BLOCK=ctx.block,
            num_warps=ctx.num_warps,
        )
        return (None, *grad_inputs)


@torch.fx.wrap
def triton_gather_concat(
    tensors: List[torch.Tensor],
    row_indices: torch.Tensor,
) -> torch.Tensor:
    return _GatherConcat.apply(row_indices, *tensors)
