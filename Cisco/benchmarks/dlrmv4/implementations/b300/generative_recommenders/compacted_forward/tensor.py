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

             

''                                                       

from dataclasses import dataclass
from typing import Any, List, Optional

import torch
from generative_recommenders.common import HammerKernel
from generative_recommenders.ops.gather_concat import gather_concat


@dataclass(frozen=True)
class CompactedTensor:
    ''                                                                        

    values: torch.Tensor
    reverse_indices: Optional[torch.Tensor] = None

    def __getitem__(self, index: Any) -> torch.Tensor:
        if self.reverse_indices is None:
            return self.values[index]
        if isinstance(index, tuple):
            row_index = index[0]
            trailing_index = index[1:]
        else:
            row_index = index
            trailing_index = ()
        compacted_row = self.reverse_indices[row_index]
        return self.values[(compacted_row,) + trailing_index]

    def decompress_to_tensor(self) -> torch.Tensor:
        ''                                            

        if self.reverse_indices is None:
            return self.values
        return torch.index_select(self.values, 0, self.reverse_indices)

    def to(self, dtype: torch.dtype) -> "CompactedTensor":
        ''                                             

        return CompactedTensor(
            values=self.values.to(dtype),
            reverse_indices=self.reverse_indices,
        )


torch.fx.wrap("gather_concat")


@dataclass(frozen=True)
class JointCompactedTensor2D:
    ''                                                                      

    tensors: List[CompactedTensor]
    joint_reverse_indices: Optional[torch.Tensor]

    def to(self, dtype: torch.dtype) -> "JointCompactedTensor2D":
        ''                                                       

        return JointCompactedTensor2D(
            tensors=[tensor.to(dtype) for tensor in self.tensors],
            joint_reverse_indices=self.joint_reverse_indices,
        )

    def to_compacted_tensor(self, kernel: HammerKernel) -> CompactedTensor:
        ''                                                                       

        if not self.tensors:
            raise ValueError("JointCompactedTensor2D requires at least one tensor")

        row_indices: List[torch.Tensor] = []
        for tensor in self.tensors:
            reverse_indices = tensor.reverse_indices
            if reverse_indices is None:
                reverse_indices = torch.arange(
                    tensor.values.size(0),
                    dtype=torch.int64,
                    device=tensor.values.device,
                )
            row_indices.append(reverse_indices)

        values = gather_concat(
            tensors=[tensor.values for tensor in self.tensors],
            row_indices=torch.stack(row_indices, dim=1),
            kernel=kernel,
        )
        return CompactedTensor(
            values=values,
            reverse_indices=self.joint_reverse_indices,
        )
