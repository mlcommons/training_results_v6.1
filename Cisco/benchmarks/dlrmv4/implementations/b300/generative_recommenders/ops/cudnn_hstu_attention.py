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

#!/usr/bin/env python3

             

''                                       

                                                         
                                                                                 
   

import functools
import os
from typing import Any, Optional, Tuple

import torch


_CUDNN_ATTENTION_SWITCH = "HSTU_USE_CUDNN_ATTENTION"


def use_cudnn_hstu_attention() -> bool:
    ''                                                                

    return os.environ.get(_CUDNN_ATTENTION_SWITCH, "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@functools.lru_cache(maxsize=1)
def _load_cudnn_hstu_apis() -> Tuple[Any, Any]:
    try:
        from cudnn import hstu_attention_backward, hstu_attention_forward
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            f"{_CUDNN_ATTENTION_SWITCH}=1 requires the cuDNN Frontend HSTU build and "
            "its CuTe DSL/TVM FFI runtime dependencies"
        ) from exc
    return hstu_attention_forward, hstu_attention_backward


def _prepare_cudnn_hstu_metadata(
    seq_offsets: torch.Tensor,
    num_targets: Optional[torch.Tensor],
    max_attn_len: int,
    contextual_seq_len: int,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    ''                                                                      

    if num_targets is not None or contextual_seq_len != 0:
        raise NotImplementedError(
            "cuDNN Frontend HSTU attention is configured for plain causal "
            "attention; target-aware and contextual attention must be disabled"
        )
    cu_seqlens = seq_offsets.to(dtype=torch.int32).contiguous()
    window_size = (max_attn_len, 0) if max_attn_len > 0 else (-1, 0)
    return cu_seqlens, window_size


def cudnn_hstu_attention_fwd(
    N: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_offsets: torch.Tensor,
    k_offsets: torch.Tensor,
    num_targets: Optional[torch.Tensor],
    max_attn_len: int,
    contextual_seq_len: int,
    max_seqlen_q: Optional[int] = None,
) -> torch.Tensor:
    ''                                                                             

    if q.shape[0] == 0:
        return torch.empty_like(q)
    hstu_attention_forward, _ = _load_cudnn_hstu_apis()
    cu_seqlens_q, window_size = _prepare_cudnn_hstu_metadata(
        seq_offsets=q_offsets,
        num_targets=num_targets,
        max_attn_len=max_attn_len,
        contextual_seq_len=contextual_seq_len,
    )
    cu_seqlens_k = (
        cu_seqlens_q
        if k_offsets is q_offsets
        else k_offsets.to(dtype=torch.int32).contiguous()
    )
    return hstu_attention_forward(
        q_tensor=q,
        k_tensor=k,
        v_tensor=v,
        cu_seqlens_q_tensor=cu_seqlens_q,
        cu_seqlens_k_tensor=cu_seqlens_k,
        max_seqlen_q=N if max_seqlen_q is None else max_seqlen_q,
        max_seqlen_k=N,
        window_size=window_size,
        alpha=alpha,
        scaling_seqlen=float(N),
        stream=torch.cuda.current_stream(q.device),
    )[0]


def cudnn_hstu_attention_bwd(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    q_offsets: torch.Tensor,
    k_offsets: torch.Tensor,
    num_targets: Optional[torch.Tensor],
    N: int,
    alpha: float,
    max_attn_len: int,
    contextual_seq_len: int,
    max_seqlen_q: Optional[int] = None,
) -> None:
    ''                                                                              

    if q.shape[0] == 0:
        dq.zero_()
        dk.zero_()
        dv.zero_()
        return
    _, hstu_attention_backward = _load_cudnn_hstu_apis()
    cu_seqlens_q, window_size = _prepare_cudnn_hstu_metadata(
        seq_offsets=q_offsets,
        num_targets=num_targets,
        max_attn_len=max_attn_len,
        contextual_seq_len=contextual_seq_len,
    )
    cu_seqlens_k = (
        cu_seqlens_q
        if k_offsets is q_offsets
        else k_offsets.to(dtype=torch.int32).contiguous()
    )
    hstu_attention_backward(
        do_tensor=dout,
        q_tensor=q,
        k_tensor=k,
        v_tensor=v,
        dq_tensor=dq,
        dk_tensor=dk,
        dv_tensor=dv,
        cu_seqlens_q_tensor=cu_seqlens_q,
        cu_seqlens_k_tensor=cu_seqlens_k,
        max_seqlen_q=N if max_seqlen_q is None else max_seqlen_q,
        max_seqlen_k=N,
        window_size=window_size,
        alpha=alpha,
        scaling_seqlen=float(N),
        stream=torch.cuda.current_stream(q.device),
    )
