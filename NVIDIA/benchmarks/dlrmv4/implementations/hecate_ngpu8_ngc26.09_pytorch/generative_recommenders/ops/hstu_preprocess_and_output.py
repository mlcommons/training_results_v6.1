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

"""HSTU layer autograd with a single owner for the packed UVQK gradients."""

import math
from typing import Optional

import torch
import torch.nn.functional as F
from generative_recommenders.ops.cudnn_hstu_attention import (
    cudnn_hstu_attention_bwd,
    cudnn_hstu_attention_fwd,
)
from generative_recommenders.ops.cudnn_lmsd import (
    cudnn_lmsd_bwd,
    cudnn_lmsd_fwd,
)
from generative_recommenders.ops.triton.triton_addmm import (
    maybe_triton_addmm_fwd,
    triton_addmm_bwd,
)
from generative_recommenders.ops.triton.triton_hstu_attention import (
    _should_enable_tma,
    tensor_descriptor_tma,
    triton_hstu_attention_bwd,
    triton_hstu_attention_fwd,
)
from generative_recommenders.ops.triton.triton_hstu_linear import (
    triton_group_norm_mul_dropout_bwd,
    triton_group_norm_mul_dropout_fwd,
    triton_layer_norm_mul_dropout_bwd_out,
    triton_layer_norm_mul_dropout_fwd,
)
from generative_recommenders.ops.triton.triton_layer_norm import (
    compute_BLOCK_D,
    triton_weighted_layer_norm_bwd,
    triton_weighted_layer_norm_fwd,
)


def hstu_preprocess_and_output(
    *,
    x: torch.Tensor,
    input_norm_weight: torch.Tensor,
    input_norm_bias: torch.Tensor,
    uvqk_weight: torch.Tensor,
    uvqk_bias: torch.Tensor,
    output_norm_weight: torch.Tensor,
    output_norm_bias: torch.Tensor,
    output_weight: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: Optional[torch.Tensor],
    max_seq_len: int,
    num_heads: int,
    attn_dim: int,
    hidden_dim: int,
    input_norm_eps: float,
    output_norm_eps: float,
    attn_alpha: float,
    dropout_ratio: float,
    training: bool,
    max_attn_len: int = 0,
    contextual_seq_len: int = 0,
    sort_by_length: bool = False,
    use_cudnn_attention: bool = False,
    use_cudnn_lmsd: bool = False,
    concat_u: bool = False,
    concat_x: bool = False,
    silu_u: bool = True,
    mul_u_activation_type: str = "silu",
    seed: Optional[int] = None,
    enable_tma: Optional[bool] = None,
    group_norm: bool = False,
    defer_u_silu: bool = True,
    causal: bool = True,
    prefill: bool = False,
    recompute_uvqk_in_backward: bool = False,
    recompute_normed_x_in_backward: bool = False,
    recompute_y_in_backward: bool = False,
    num_softmax_heads: int = 0,
    linear_dim: Optional[int] = None,
    validate_metadata: bool = False,
    candidate_only: bool = False,
) -> torch.Tensor:
    """Compose preprocess/attention and output with one autograd owner.

    By default LMSD receives raw U and fuses SiLU. ``defer_u_silu=False``
    applies preprocess SiLU before LMSD. ``silu_u`` controls concatenated U (LayerNorm),
    while ``mul_u_activation_type`` controls its gate independently. Concat
    order is [optional U, optional attention, normalized_attention * gate].
    Output weight shape is [H*hidden_dim*(1+concat_u+concat_x), x.shape[1]].

    Recompute, KV prefill and non-causal/softmax attention are rejected. The
    cuDNN backend supports fewer activation/mask configurations than Triton;
    unsupported combinations raise instead of changing backend. GroupNorm
    requires matching concat flags and a shared U activation.

    Offsets must partition all N rows with lengths <= max_seq_len; targets
    must be in [0, sequence_length]. ``validate_metadata=True`` checks these
    values with a device-to-host synchronization. ``candidate_only`` requires
    cuDNN Frontend attention for the single suffix candidate in each training
    sequence. LMSD can use either cuDNN or Triton. It retains full K/V
    and only candidate U/Q/output, saving activations without recompute.
    Dropout uses compact row numbering; same-seed full-row masks are not preserved.
    Shape/dtype/stride checks
    are always enabled and do not synchronize.
    """
    return HSTUPreprocessAndOutputFunction.apply(
        x,
        input_norm_weight,
        input_norm_bias,
        uvqk_weight,
        uvqk_bias,
        output_norm_weight,
        output_norm_bias,
        output_weight,
        seq_offsets,
        num_targets,
        max_seq_len,
        num_heads,
        attn_dim,
        hidden_dim,
        input_norm_eps,
        attn_alpha,
        max_attn_len,
        contextual_seq_len,
        dropout_ratio,
        training,
        sort_by_length,
        use_cudnn_attention,
        use_cudnn_lmsd,
        concat_u,
        concat_x,
        silu_u,
        mul_u_activation_type,
        output_norm_eps,
        seed,
        enable_tma,
        group_norm,
        defer_u_silu,
        causal,
        prefill,
        recompute_uvqk_in_backward,
        recompute_normed_x_in_backward,
        recompute_y_in_backward,
        num_softmax_heads,
        linear_dim,
        validate_metadata,
        candidate_only,
    )


class HSTUPreprocessAndOutputFunction(torch.autograd.Function):
    """Input LN, UVQK, attention, LMSD and residual output projection, without recompute."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        input_norm_weight: torch.Tensor,
        input_norm_bias: torch.Tensor,
        uvqk_weight: torch.Tensor,
        uvqk_bias: torch.Tensor,
        output_norm_weight: torch.Tensor,
        output_norm_bias: torch.Tensor,
        output_weight: torch.Tensor,
        seq_offsets: torch.Tensor,
        num_targets: Optional[torch.Tensor],
        max_seq_len: int,
        num_heads: int,
        attn_dim: int,
        hidden_dim: int,
        input_norm_eps: float,
        attn_alpha: float,
        max_attn_len: int,
        contextual_seq_len: int,
        dropout_ratio: float,
        training: bool,
        sort_by_length: bool,
        use_cudnn_attention: bool,
        use_cudnn_lmsd: bool,
        concat_u: bool = False,
        concat_x: bool = False,
        silu_u: bool = True,
        mul_u_activation_type: str = "silu",
        output_norm_eps: float = 1e-6,
        seed: Optional[int] = None,
        enable_tma: Optional[bool] = None,
        group_norm: bool = False,
        defer_u_silu: bool = True,
        causal: bool = True,
        prefill: bool = False,
        recompute_uvqk_in_backward: bool = False,
        recompute_normed_x_in_backward: bool = False,
        recompute_y_in_backward: bool = False,
        num_softmax_heads: int = 0,
        linear_dim: Optional[int] = None,
        validate_metadata: bool = False,
        candidate_only: bool = False,
    ) -> torch.Tensor:
        if candidate_only and not use_cudnn_attention:
            raise ValueError("candidate_only requires use_cudnn_attention=True")
        # Scalar/shape checks do not synchronize the GPU. Value checks are opt-in.
        for name, value in (
            ("concat_u", concat_u),
            ("concat_x", concat_x),
            ("silu_u", silu_u),
            ("training", training),
            ("group_norm", group_norm),
            ("defer_u_silu", defer_u_silu),
            ("causal", causal),
            ("prefill", prefill),
            ("sort_by_length", sort_by_length),
            ("use_cudnn_attention", use_cudnn_attention),
            ("use_cudnn_lmsd", use_cudnn_lmsd),
            ("validate_metadata", validate_metadata),
            ("recompute_uvqk_in_backward", recompute_uvqk_in_backward),
            ("recompute_normed_x_in_backward", recompute_normed_x_in_backward),
            ("recompute_y_in_backward", recompute_y_in_backward),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a bool, got {value!r}")
        if enable_tma is not None and not isinstance(enable_tma, bool):
            raise TypeError("enable_tma must be a bool or None")
        if enable_tma and (not tensor_descriptor_tma or torch.version.hip):
            raise NotImplementedError("TMA requires CUDA and Triton tensor descriptors")
        if type(num_softmax_heads) is not int:
            raise TypeError("num_softmax_heads must be an integer")
        if not causal or num_softmax_heads != 0:
            raise NotImplementedError(
                "Only causal HSTU attention without softmax heads is supported"
            )
        if prefill:
            raise NotImplementedError(
                "KV prefill requires fuse_preprocess_and_output=False"
            )
        if any(
            (
                recompute_uvqk_in_backward,
                recompute_normed_x_in_backward,
                recompute_y_in_backward,
            )
        ):
            raise NotImplementedError(
                "Activation recompute is disabled in the unified HSTU Function"
            )
        if mul_u_activation_type not in ("none", "silu", "sigmoid"):
            raise ValueError(
                "mul_u_activation_type must be 'none', 'silu', or 'sigmoid'"
            )
        for name, value in (
            ("num_heads", num_heads),
            ("attn_dim", attn_dim),
            ("hidden_dim", hidden_dim),
            ("max_seq_len", max_seq_len),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in (
            ("max_attn_len", max_attn_len),
            ("contextual_seq_len", contextual_seq_len),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if linear_dim is not None and (
            type(linear_dim) is not int or linear_dim != hidden_dim
        ):
            raise ValueError("linear_dim must match hidden_dim")
        for name, value in (
            ("input_norm_eps", input_norm_eps),
            ("output_norm_eps", output_norm_eps),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(attn_alpha):
            raise ValueError("attn_alpha must be finite")
        if not math.isfinite(dropout_ratio) or not 0 <= dropout_ratio < 1:
            raise ValueError(
                "dropout_ratio must be in [0, 1), including during evaluation"
            )
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**63):
            raise ValueError("seed must be an integer in [0, 2**63) or None")
        if group_norm and (
            concat_u != concat_x
            or mul_u_activation_type != ("silu" if silu_u else "none")
        ):
            raise NotImplementedError(
                "GroupNorm requires matching concat_u/concat_x and the same U activation for the gate and concat"
            )
        if use_cudnn_lmsd and (
            group_norm
            or mul_u_activation_type == "sigmoid"
            or (concat_u and silu_u != (mul_u_activation_type == "silu"))
        ):
            raise NotImplementedError(
                "cuDNN LMSD requires LayerNorm and the same U activation for the gate and concat (none or silu)"
            )
        if use_cudnn_attention and (
            (num_targets is not None and not candidate_only) or contextual_seq_len != 0
        ):
            raise NotImplementedError(
                "cuDNN attention supports plain causal attention only; disable target-aware/contextual attention"
            )
        if use_cudnn_attention and enable_tma is True:
            raise ValueError(
                "enable_tma controls Triton attention, not cuDNN attention"
            )
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a tensor")
        if (
            x.ndim != 2
            or x.shape[1] <= 0
            or x.stride(1) != 1
            or x.stride(0) < x.shape[1]
        ):
            raise ValueError(
                "x must be [N, E] with unit column stride and nonoverlapping rows"
            )
        if not x.is_cuda or x.dtype not in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ):
            raise ValueError("x must be a CUDA FP16/BF16/FP32 tensor")
        if enable_tma and (
            torch.cuda.get_device_capability(x.device)[0] < 9
            or attn_dim * x.element_size() % 16
            or hidden_dim * x.element_size() % 16
        ):
            raise ValueError("TMA requires SM90+ and 16-byte-aligned head dimensions")
        n, e = x.shape
        d = num_heads * hidden_dim
        packed = 2 * num_heads * (hidden_dim + attn_dim)
        norm_size = num_heads if group_norm else d
        for name, tensor, shape in (
            ("input_norm_weight", input_norm_weight, (e,)),
            ("input_norm_bias", input_norm_bias, (e,)),
            ("output_norm_weight", output_norm_weight, (norm_size,)),
            ("output_norm_bias", output_norm_bias, (norm_size,)),
            ("uvqk_weight", uvqk_weight, (e, packed)),
            (
                "output_weight",
                output_weight,
                (d * (1 + int(concat_u) + int(concat_x)), e),
            ),
        ):
            if tensor.shape != shape or tensor.device != x.device:
                raise ValueError(f"{name} must have shape {shape} on {x.device}")
            if tensor.ndim == 1 and tensor.stride(0) != 1:
                raise ValueError(f"{name} must be contiguous")
            if tensor.dtype not in (x.dtype, torch.float32):
                raise ValueError(f"{name} must have dtype {x.dtype} or float32")
        if (
            uvqk_bias.shape not in ((packed,), (n, packed))
            or uvqk_bias.device != x.device
        ):
            raise ValueError(
                f"uvqk_bias must have shape ({packed},) or ({n}, {packed}) on {x.device}"
            )
        if any(t.dtype != x.dtype for t in (uvqk_weight, uvqk_bias, output_weight)):
            raise ValueError("GEMM weights and uvqk_bias must have the same dtype as x")
        if (
            seq_offsets.ndim != 1
            or seq_offsets.numel() < 1
            or (n > 0 and seq_offsets.numel() < 2)
            or seq_offsets.device != x.device
            or seq_offsets.dtype not in (torch.int32, torch.int64)
            or not seq_offsets.is_contiguous()
        ):
            raise ValueError(
                "seq_offsets must be a contiguous int32/int64 [B+1] tensor on x.device"
            )
        if num_targets is not None and (
            num_targets.shape != (seq_offsets.numel() - 1,)
            or num_targets.device != x.device
            or num_targets.dtype not in (torch.int32, torch.int64)
            or not num_targets.is_contiguous()
        ):
            raise ValueError(
                "num_targets must be a contiguous int32/int64 [B] tensor on x.device"
            )
        if use_cudnn_lmsd and (
            x.dtype != torch.bfloat16
            or d % 8
            or not 8 <= d < 1024
            or any(t.dtype != x.dtype for t in (output_norm_weight, output_norm_bias))
        ):
            raise ValueError(
                "cuDNN LMSD requires BF16 x/output norm parameters and H*hidden_dim divisible by 8 in [8, 1024)"
            )
        if use_cudnn_lmsd and torch.cuda.get_device_capability(x.device)[0] != 10:
            raise ValueError("cuDNN LMSD requires an SM10x GPU")
        if validate_metadata:
            offsets = seq_offsets.detach().cpu()
            lengths = offsets[1:] - offsets[:-1]
            if (
                offsets[0].item() != 0
                or offsets[-1].item() != n
                or (lengths < 0).any()
                or (lengths > max_seq_len).any()
            ):
                raise ValueError(
                    "seq_offsets must start at 0, end at N, and define lengths in [0, max_seq_len]"
                )
            if num_targets is not None:
                targets = num_targets.detach().cpu()
                if (targets < 0).any() or (targets > lengths).any():
                    raise ValueError(
                        "num_targets must be between 0 and the sequence length"
                    )
        normed_x, x_mean, x_rstd = triton_weighted_layer_norm_fwd(
            x=x,
            weight=input_norm_weight,
            bias=input_norm_bias,
            eps=input_norm_eps,
        )
        splits = (hidden_dim * num_heads,) * 2 + (attn_dim * num_heads,) * 2
        candidate_rows = candidate_normed_x = w_uq = w_vk = uq = vk = None
        if candidate_only:
            candidate_rows = seq_offsets[1:] - 1
            candidate_normed_x = normed_x.index_select(0, candidate_rows)
            w_u, w_v, w_q, w_k = uvqk_weight.split(splits, dim=1)
            b_u, b_v, b_q, b_k = uvqk_bias.split(splits)
            w_uq, w_vk = torch.cat((w_u, w_q), 1), torch.cat((w_v, w_k), 1)
            uq = maybe_triton_addmm_fwd(
                x=candidate_normed_x, w=w_uq, y=torch.cat((b_u, b_q))
            ).contiguous()
            vk = maybe_triton_addmm_fwd(
                x=normed_x, w=w_vk, y=torch.cat((b_v, b_k))
            ).contiguous()
            u, q = uq.split((splits[0], splits[2]), dim=1)
            v, k = vk.split((splits[1], splits[3]), dim=1)
            uvqk = None
        else:
            uvqk = maybe_triton_addmm_fwd(
                x=normed_x, w=uvqk_weight, y=uvqk_bias
            ).contiguous()
            u, v, q, k = uvqk.split(splits, dim=1)
        lmsd_u = u if defer_u_silu else F.silu(u)
        q, k = q.view(-1, num_heads, attn_dim), k.view(-1, num_heads, attn_dim)
        v = v.view(-1, num_heads, hidden_dim)
        sort_indices = None
        if sort_by_length and not use_cudnn_attention:
            sort_indices = torch.argsort(
                seq_offsets[1:] - seq_offsets[:-1],
                descending=True,
                stable=False,
            )
        if enable_tma is None:
            enable_tma = not use_cudnn_attention and _should_enable_tma()
        q_offsets = seq_offsets
        if candidate_only:
            q_offsets = torch.arange(
                seq_offsets.numel(), device=q.device, dtype=torch.int32
            )
        attn_kwargs = dict(
            N=max_seq_len,
            alpha=attn_alpha,
            q=q,
            k=k,
            v=v,
            num_targets=num_targets,
            max_attn_len=max_attn_len,
            contextual_seq_len=contextual_seq_len,
        )
        if q.shape[0] == 0:
            attn = v.new_empty((0, num_heads, hidden_dim))
        elif use_cudnn_attention:
            attn = cudnn_hstu_attention_fwd(
                **attn_kwargs,
                q_offsets=q_offsets,
                k_offsets=seq_offsets,
                max_seqlen_q=1 if candidate_only else max_seq_len,
            )
        else:
            attn = triton_hstu_attention_fwd(
                **attn_kwargs,
                seq_offsets=seq_offsets,
                sort_by_length_indices=sort_indices,
                enable_tma=enable_tma,
                num_softmax_heads=0,
            )
        attn = attn.view(-1, num_heads * hidden_dim)
        dropout_ratio = dropout_ratio if training else 0.0
        training = training and dropout_ratio > 0.0
        if seed is None and use_cudnn_lmsd:
            seed = (
                torch.randint(0, 2**62, (1,), dtype=torch.int64).item()
                if training
                else 0
            )
        block_d, block_h, num_warps = 0, 0, 0
        if use_cudnn_lmsd:
            y, mean, rstd, mask = cudnn_lmsd_fwd(
                attn,
                lmsd_u,
                output_norm_weight,
                output_norm_bias,
                output_norm_eps,
                dropout_ratio,
                seed,
                mul_u_activation_type == "silu",
                concat_u,
                concat_x,
            )
        elif group_norm:
            y, mean, rstd, block_d, block_h, num_warps, seed = (
                triton_group_norm_mul_dropout_fwd(
                    x=attn,
                    u=lmsd_u,
                    weight=output_norm_weight,
                    bias=output_norm_bias,
                    eps=output_norm_eps,
                    dropout_ratio=dropout_ratio,
                    training=training,
                    silu_u=silu_u,
                    concat_ux=concat_u and concat_x,
                    num_heads=num_heads,
                    linear_dim=hidden_dim,
                    seed=seed,
                )
            )
            mask = None
        else:
            y, mean, rstd, block_d, num_warps, seed, mask = (
                triton_layer_norm_mul_dropout_fwd(
                    x=attn,
                    u=lmsd_u,
                    weight=output_norm_weight,
                    bias=output_norm_bias,
                    eps=output_norm_eps,
                    dropout_ratio=dropout_ratio,
                    training=training,
                    silu_u=silu_u,
                    concat_u=concat_u,
                    concat_x=concat_x,
                    mul_u_activation_type=mul_u_activation_type,
                    seed=seed,
                )
            )
        residual = x.index_select(0, candidate_rows) if candidate_only else x
        out = maybe_triton_addmm_fwd(x=y, w=output_weight, y=residual)
        ctx.save_for_backward(
            x,
            input_norm_weight,
            input_norm_bias,
            uvqk_weight,
            output_norm_weight,
            output_norm_bias,
            output_weight,
            normed_x,
            uvqk,
            attn,
            y,
            x_mean,
            x_rstd,
            mean,
            rstd,
            seq_offsets,
            num_targets,
            sort_indices,
            mask,
            lmsd_u,
            candidate_rows,
            candidate_normed_x,
            w_uq,
            w_vk,
            uq,
            vk,
            q_offsets,
        )
        ctx.candidate_only = candidate_only
        ctx.splits = splits
        ctx.num_heads = num_heads
        ctx.attn_dim = attn_dim
        ctx.hidden_dim = hidden_dim
        ctx.input_norm_eps = input_norm_eps
        ctx.output_norm_eps = output_norm_eps
        ctx.concat_u, ctx.concat_x = concat_u, concat_x
        ctx.silu_u = silu_u
        ctx.mul_u_activation_type = mul_u_activation_type
        ctx.group_norm = group_norm
        ctx.defer_u_silu = defer_u_silu
        ctx.block_h = block_h
        ctx.norm_block_d = compute_BLOCK_D(x)
        ctx.attn_alpha = attn_alpha
        ctx.max_seq_len = max_seq_len
        ctx.max_attn_len = max_attn_len
        ctx.contextual_seq_len = contextual_seq_len
        ctx.use_cudnn_attention = use_cudnn_attention
        ctx.use_cudnn_lmsd = use_cudnn_lmsd
        ctx.enable_tma = enable_tma
        ctx.uvqk_bias_1d = uvqk_bias.dim() == 1
        ctx.dropout_ratio = dropout_ratio
        ctx.training = training
        ctx.seed = seed if seed is not None else 0
        ctx.block_d = block_d
        ctx.num_warps = num_warps
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        (
            x,
            input_norm_weight,
            input_norm_bias,
            uvqk_weight,
            output_norm_weight,
            output_norm_bias,
            output_weight,
            normed_x,
            uvqk,
            attn,
            y,
            x_mean,
            x_rstd,
            mean,
            rstd,
            seq_offsets,
            num_targets,
            sort_indices,
            mask,
            lmsd_u,
            candidate_rows,
            candidate_normed_x,
            w_uq,
            w_vk,
            uq,
            vk,
            q_offsets,
        ) = ctx.saved_tensors
        if ctx.candidate_only:
            u, q = uq.split((ctx.splits[0], ctx.splits[2]), dim=1)
            v, k = vk.split((ctx.splits[1], ctx.splits[3]), dim=1)
            duq = torch.empty_like(uq)
            du, dq = duq.split((ctx.splits[0], ctx.splits[2]), dim=1)
            # cuDNN-FE writes directly into disjoint two-slot views of the VK
            # projection's packed gradient buffer.
            dvk = torch.empty_like(vk)
            dv, dk = dvk.split((ctx.splits[1], ctx.splits[3]), dim=1)
        else:
            duvqk = torch.empty_like(uvqk)
            du, dv, dq, dk = duvqk.split(ctx.splits, dim=1)
            u, v, q, k = uvqk.split(ctx.splits, dim=1)
        dy = torch.mm(dout, output_weight.t())
        if ctx.use_cudnn_lmsd:
            dattn, d_output_norm_weight, d_output_norm_bias = cudnn_lmsd_bwd(
                dy,
                attn,
                lmsd_u,
                output_norm_weight,
                output_norm_bias,
                mean,
                rstd,
                mask,
                ctx.dropout_ratio,
                ctx.mul_u_activation_type == "silu",
                ctx.concat_u,
                ctx.concat_x,
                du,
            )
        elif ctx.group_norm:
            dattn, _, d_output_norm_weight, d_output_norm_bias, _ = (
                triton_group_norm_mul_dropout_bwd(
                    dy=dy,
                    x=attn,
                    u=lmsd_u,
                    weight=output_norm_weight,
                    bias=output_norm_bias,
                    mean=mean,
                    rstd=rstd,
                    BLOCK_D=ctx.block_d,
                    BLOCK_H=ctx.block_h,
                    num_warps=ctx.num_warps,
                    eps=ctx.output_norm_eps,
                    training=ctx.training,
                    dropout_ratio=ctx.dropout_ratio,
                    seed=ctx.seed,
                    silu_u=ctx.silu_u,
                    concat_ux=ctx.concat_u and ctx.concat_x,
                    num_heads=ctx.num_heads,
                    linear_dim=ctx.hidden_dim,
                    compute_y=False,
                    du=du,
                )
            )
        else:
            dattn, d_output_norm_weight, d_output_norm_bias, _ = (
                triton_layer_norm_mul_dropout_bwd_out(
                    dy,
                    attn,
                    lmsd_u,
                    output_norm_weight,
                    output_norm_bias,
                    mean,
                    rstd,
                    ctx.block_d,
                    ctx.num_warps,
                    ctx.output_norm_eps,
                    ctx.training,
                    ctx.dropout_ratio,
                    ctx.seed,
                    ctx.silu_u,
                    ctx.concat_u,
                    ctx.concat_x,
                    ctx.mul_u_activation_type,
                    False,
                    mask if mask is not None else attn.new_empty((0,)),
                    du,
                )
            )
        if not ctx.defer_u_silu:
            torch.ops.aten.silu_backward(du, u, grad_input=du)
        d_output_weight = torch.mm(y.t(), dout)
        attn_kwargs = dict(
            dout=dattn.view(-1, ctx.num_heads, ctx.hidden_dim),
            q=q.view(-1, ctx.num_heads, ctx.attn_dim),
            k=k.view(-1, ctx.num_heads, ctx.attn_dim),
            v=v.view(-1, ctx.num_heads, ctx.hidden_dim),
            dq=dq.view(-1, ctx.num_heads, ctx.attn_dim),
            dk=dk.view(-1, ctx.num_heads, ctx.attn_dim),
            dv=dv.view(-1, ctx.num_heads, ctx.hidden_dim),
            num_targets=num_targets,
            N=ctx.max_seq_len,
            alpha=ctx.attn_alpha,
            max_attn_len=ctx.max_attn_len,
            contextual_seq_len=ctx.contextual_seq_len,
        )
        if ctx.use_cudnn_attention:
            cudnn_hstu_attention_bwd(
                **attn_kwargs,
                q_offsets=q_offsets,
                k_offsets=seq_offsets,
                max_seqlen_q=1 if ctx.candidate_only else ctx.max_seq_len,
            )
        else:
            triton_hstu_attention_bwd(
                **attn_kwargs,
                seq_offsets=seq_offsets,
                sort_by_length_indices=sort_indices,
                enable_tma=ctx.enable_tma,
                num_softmax_heads=0,
            )
        if ctx.candidate_only:
            d_candidate_x, d_w_uq, d_b_uq = triton_addmm_bwd(
                x=candidate_normed_x,
                w=w_uq,
                dz=duq,
                is_y_1d=True,
            )
            d_normed_x, d_w_vk, d_b_vk = triton_addmm_bwd(
                x=normed_x,
                w=w_vk,
                dz=dvk,
                is_y_1d=True,
            )
            d_normed_x.index_add_(0, candidate_rows, d_candidate_x)
            dw_u, dw_q = d_w_uq.split((ctx.splits[0], ctx.splits[2]), dim=1)
            dw_v, dw_k = d_w_vk.split((ctx.splits[1], ctx.splits[3]), dim=1)
            db_u, db_q = d_b_uq.split((ctx.splits[0], ctx.splits[2]))
            db_v, db_k = d_b_vk.split((ctx.splits[1], ctx.splits[3]))
            d_uvqk_weight = torch.cat((dw_u, dw_v, dw_q, dw_k), dim=1)
            d_uvqk_bias = torch.cat((db_u, db_v, db_q, db_k))
        else:
            d_normed_x, d_uvqk_weight, d_uvqk_bias = triton_addmm_bwd(
                x=normed_x,
                w=uvqk_weight,
                dz=duvqk,
                is_y_1d=ctx.uvqk_bias_1d,
            )
        dx, d_input_norm_weight, d_input_norm_bias = triton_weighted_layer_norm_bwd(
            dy=d_normed_x,
            x=x,
            weight=input_norm_weight,
            bias=input_norm_bias,
            mean=x_mean,
            rstd=x_rstd,
            learnable=True,
            eps=ctx.input_norm_eps,
            BLOCK_D=ctx.norm_block_d,
            residual=None if ctx.candidate_only else dout,
        )
        if ctx.candidate_only:
            dx.index_add_(0, candidate_rows, dout)
        return (
            dx,
            d_input_norm_weight,
            d_input_norm_bias,
            d_uvqk_weight,
            d_uvqk_bias,
            d_output_norm_weight,
            d_output_norm_bias,
            d_output_weight,
        ) + (None,) * (len(ctx.needs_input_grad) - 8)
