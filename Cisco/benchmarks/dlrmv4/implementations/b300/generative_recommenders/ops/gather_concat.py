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

             

from typing import List

import torch
from generative_recommenders.common import HammerKernel
from generative_recommenders.ops.triton.triton_gather_concat import (
    triton_gather_concat,
)


def gather_concat(
    tensors: List[torch.Tensor],
    row_indices: torch.Tensor,
    kernel: HammerKernel,
) -> torch.Tensor:
    ''                                                                         

    if not tensors:
        raise ValueError("gather_concat requires at least one tensor")
    if row_indices.dim() != 2 or row_indices.size(1) != len(tensors):
        raise ValueError("row_indices width must match the number of tensors")
    if any(tensor.dim() != 2 for tensor in tensors):
        raise ValueError("gather_concat tensors must be two-dimensional")
    equal_width = all(
        tensor.size(1) == tensors[0].size(1) for tensor in tensors[1:]
    )
    triton_compatible = (
        row_indices.is_cuda
        and equal_width
        and all(
            tensor.is_cuda
            and tensor.device == tensors[0].device
            and tensor.dtype == tensors[0].dtype
            for tensor in tensors
        )
    )
    if kernel == HammerKernel.TRITON and triton_compatible and len(tensors) in (2, 3):
        return triton_gather_concat(tensors, row_indices)
    return torch.cat(
        [
            torch.index_select(value, 0, row_indices[:, index])
            for index, value in enumerate(tensors)
        ],
        dim=1,
    )
