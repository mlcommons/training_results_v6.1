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

# pyre-strict

from typing import List, Optional

import torch
from generative_recommenders.compacted_forward.tensor import CompactedTensor
from generative_recommenders.common import HammerKernel
from generative_recommenders.ops.jagged_tensors import concat_2D_jagged
from generative_recommenders.ops.triton.triton_compacted_sum_jagged_concat_3 import (
    triton_compacted_sum_jagged_concat_3,
)


def _pytorch_compacted_sum_jagged_concat(
    compacted_tensors: List[CompactedTensor],
    dense: Optional[torch.Tensor],
    seq_offsets: torch.Tensor,
    max_seq_len: int,
    dense_first: bool = True,
    kernel: HammerKernel = HammerKernel.PYTORCH,
) -> torch.Tensor:
    values = compacted_tensors[0].decompress_to_tensor()
    for tensor in compacted_tensors[1:]:
        values = values + tensor.decompress_to_tensor()
    if dense is None or dense.size(1) == 0:
        return values
    dense_len = dense.size(1)
    dense_values = dense.flatten(0, 1)
    return concat_2D_jagged(
        max_seq_len=dense_len + max_seq_len,
        values_left=dense_values if dense_first else values,
        values_right=values if dense_first else dense_values,
        max_len_left=dense_len if dense_first else max_seq_len,
        max_len_right=max_seq_len if dense_first else dense_len,
        offsets_left=None if dense_first else seq_offsets,
        offsets_right=seq_offsets if dense_first else None,
        kernel=kernel,
    )


def compacted_sum_jagged_concat(
    compacted_tensors: List[CompactedTensor],
    dense: Optional[torch.Tensor],
    seq_offsets: torch.Tensor,
    max_seq_len: int,
    *,
    dense_first: bool = True,
    kernel: HammerKernel = HammerKernel.PYTORCH,
) -> torch.Tensor:
    """Sum compacted rows in occurrence order and concatenate a dense segment.

    Args:
        compacted_tensors: Inputs with a common logical shape of [M, D].
        dense: Dense segments of shape [B, L, D], or None to skip concatenation.
        seq_offsets: Offsets of shape [B + 1] partitioning the M occurrences.
        max_seq_len: Maximum jagged sequence length, excluding dense segments.
        dense_first: Put each dense segment before its jagged sequence if True,
            or after it otherwise.
        kernel: Execution backend.

    Returns:
        Flattened jagged values of shape [M + B * L, D], or [M, D] if dense is None.
    """
    if not compacted_tensors:
        raise ValueError("compacted_tensors must not be empty")
    if dense is not None and (
        dense.dim() != 3
        or dense.size(0) != seq_offsets.numel() - 1
        or dense.size(2) != compacted_tensors[0].values.size(1)
    ):
        raise ValueError("dense must have shape [B, L, D] matching the jagged inputs")

    if (
        kernel == HammerKernel.TRITON
        and compacted_tensors[0].values.is_cuda
        and len(compacted_tensors) == 3
    ):
        reverse_0, reverse_1, reverse_2 = [
            tensor.reverse_indices for tensor in compacted_tensors
        ]
        if reverse_0 is not None and reverse_1 is not None and reverse_2 is not None:
            return triton_compacted_sum_jagged_concat_3(
                dense=dense,
                component_0=compacted_tensors[0].values,
                component_1=compacted_tensors[1].values,
                component_2=compacted_tensors[2].values,
                reverse_0=reverse_0,
                reverse_1=reverse_1,
                reverse_2=reverse_2,
                sequence_offsets=seq_offsets,
                max_sequence_len=max_seq_len,
                dense_first=dense_first,
            )
    return _pytorch_compacted_sum_jagged_concat(
        compacted_tensors, dense, seq_offsets, max_seq_len, dense_first, kernel
    )


torch.fx.wrap("compacted_sum_jagged_concat")
