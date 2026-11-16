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

"""MoE router state helpers. This module does not import modelopt."""

import torch


def _iter_unique_modules(model):
    seen_modules = set()
    for model_chunk in model:
        for module in model_chunk.modules():
            if id(module) in seen_modules:
                continue
            seen_modules.add(id(module))
            yield module


def _snapshot_router_state(model):
    state = []
    for module in _iter_unique_modules(model):
        expert_bias = getattr(module, "expert_bias", None)
        if expert_bias is None:
            continue
        local_tokens = getattr(module, "local_tokens_per_expert", None)
        state.append(
            (
                module,
                expert_bias.detach().clone(),
                local_tokens.detach().clone() if local_tokens is not None else None,
            )
        )
    return state


def _restore_router_state(state):
    with torch.no_grad():
        for router, expert_bias, local_tokens in state:
            router.expert_bias.copy_(expert_bias)
            if local_tokens is not None:
                router.local_tokens_per_expert.copy_(local_tokens)
