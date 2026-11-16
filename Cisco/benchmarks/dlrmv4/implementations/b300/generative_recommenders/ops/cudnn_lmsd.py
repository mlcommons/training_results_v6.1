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

''                                                           

import functools
import os
from typing import Any, Optional, Tuple

import torch
from generative_recommenders.ops.utils import maybe_register_custom_op


def use_cudnn_lmsd() -> bool:
    return os.environ.get("HSTU_USE_CUDNN_LMSD", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@functools.lru_cache(maxsize=1)
def _load_lmsd_apis() -> Tuple[Any, Any]:
    try:
        from cudnn import hstu_lmsd_backward, hstu_lmsd_forward
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "HSTU_USE_CUDNN_LMSD=1 requires cuDNN Frontend develop with "
            "hstu_lmsd_forward/backward and its CuTe DSL/TVM FFI dependencies"
        ) from exc
    return hstu_lmsd_forward, hstu_lmsd_backward


@maybe_register_custom_op("generative_recommenders::cudnn_lmsd_fwd", mutates_args=())
def cudnn_lmsd_fwd(
    x: torch.Tensor,
    u: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    dropout_ratio: float,
    seed: int,
    apply_u_silu: bool,
    concat_u: bool,
    concat_x: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if x.shape[0] == 0:
        return _cudnn_lmsd_fwd_fake(
            x,
            u,
            weight,
            bias,
            eps,
            dropout_ratio,
            seed,
            apply_u_silu,
            concat_u,
            concat_x,
        )
    forward, _ = _load_lmsd_apis()
    y, mean, rstd, mask = forward(
        x_tensor=x,
        u_tensor=u,
        weight_tensor=weight,
        bias_tensor=bias,
        eps=eps,
        dropout_ratio=dropout_ratio,
        seed=seed,
        apply_u_silu=apply_u_silu,
        concat_u=concat_u,
        concat_x=concat_x,
        stream=torch.cuda.current_stream(x.device),
    )
                                                                          
    return (
        y,
        mean,
        rstd,
        mask if mask is not None else x.new_empty((0,), dtype=torch.int8),
    )


@cudnn_lmsd_fwd.register_fake
def _cudnn_lmsd_fwd_fake(
    x,
    u,
    weight,
    bias,
    eps,
    dropout_ratio,
    seed,
    apply_u_silu,
    concat_u,
    concat_x,
):
    n, d = x.shape
    return (
        x.new_empty((n, d * (1 + int(concat_u) + int(concat_x)))),
        x.new_empty((n,), dtype=torch.float32),
        x.new_empty((n,), dtype=torch.float32),
        x.new_empty((n, d) if dropout_ratio > 0 else (0,), dtype=torch.int8),
    )


@maybe_register_custom_op(
    "generative_recommenders::cudnn_lmsd_bwd", mutates_args=("du",)
)
def cudnn_lmsd_bwd(
    dy: torch.Tensor,
    x: torch.Tensor,
    u: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    mask: Optional[torch.Tensor],
    dropout_ratio: float,
    apply_u_silu: bool,
    concat_u: bool,
    concat_x: bool,
    du: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ''                                                                               
    if x.shape[0] == 0:
        du.zero_()
        return torch.empty_like(x), torch.zeros_like(weight), torch.zeros_like(bias)
    _, backward = _load_lmsd_apis()
    dx, _, dweight, dbias = backward(
        dy_tensor=dy,
        x_tensor=x,
        u_tensor=u,
        weight_tensor=weight,
        bias_tensor=bias,
        mean_tensor=mean,
        rstd_tensor=rstd,
        mask_tensor=mask if dropout_ratio > 0 else None,
        dropout_ratio=dropout_ratio,
        apply_u_silu=apply_u_silu,
        concat_u=concat_u,
        concat_x=concat_x,
        du_tensor=du,
        stream=torch.cuda.current_stream(x.device),
    )
    return dx, dweight, dbias


@cudnn_lmsd_bwd.register_fake
def _cudnn_lmsd_bwd_fake(
    dy,
    x,
    u,
    weight,
    bias,
    mean,
    rstd,
    mask,
    dropout_ratio,
    apply_u_silu,
    concat_u,
    concat_x,
    du,
):
    return torch.empty_like(x), torch.empty_like(weight), torch.empty_like(bias)
