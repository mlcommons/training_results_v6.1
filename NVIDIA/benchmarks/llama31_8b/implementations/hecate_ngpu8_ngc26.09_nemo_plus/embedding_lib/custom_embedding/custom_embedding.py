# Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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

import torch
from typing import Tuple, Union
import custom_embedding.embedding_ops as embedding_ops

global_embed_wgrad = None
class CustomEmbedding(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        weight,
        indices,
        num_embeddings,
    ):
        ctx.num_embeddings = num_embeddings
        ctx.save_for_backward(indices)
        return embedding_ops.embedding_fwd(weight, indices).clone()

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        (indices,) = ctx.saved_tensors
        global global_embed_wgrad
        if global_embed_wgrad is None:
            global_embed_wgrad = torch.empty((ctx.num_embeddings, grad_output.size(-1)), dtype=grad_output.dtype, device=grad_output.device)
        global_embed_wgrad.detach().zero_()
        global_embed_wgrad = embedding_ops.embedding_bwd(grad_output, indices, ctx.num_embeddings, global_embed_wgrad)
        return (global_embed_wgrad, None, None,)


class CustomEmbeddingSBH(torch.autograd.Function):
    """the run-owned DP engine's embedding path.

    Forward gathers the rows with transposed indices straight into a contiguous [s, b, h]
    tensor (no clone); the caller returns its [b, s, h] transpose view, so the embedding
    module's ``transpose(0, 1).contiguous()`` is free (no transpose copy). Backward reads the
    contiguous [s, b, h] gradient as is (no grad_output copy) and scatter-adds the touched rows
    straight into the weight's main_grad; the engine zeroes only the rows that hold stale
    values (this rank's previous rows and its reduced shard) instead of the whole table, then
    issues the table's reduce-scatter. The weight's autograd gradient is None.
    """

    @staticmethod
    def forward(ctx, weight, indices, num_embeddings):
        ctx.num_embeddings = num_embeddings
        ctx.weight = weight
        ctx.save_for_backward(indices)
        return embedding_ops.embedding_fwd(weight, indices.t())

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[Union[torch.Tensor, None], ...]:
        (indices,) = ctx.saved_tensors
        weight = ctx.weight
        engine = weight._opt_dp_emb
        grad_sbh = grad_output if grad_output.is_contiguous() else grad_output.contiguous()
        idx = indices.t().contiguous()
        main_grad = engine.embedding_grad_buffer(weight, idx)
        embedding_ops.embedding_bwd(grad_sbh, idx, ctx.num_embeddings, main_grad)
        engine.embedding_grad_ready(weight, idx)
        return (None, None, None)
