# Classification: Internal
"""MoE unpermute with the MLP residual add folded in, for experts without an FC2 bias.

NVFP4_RESGRAD_FWD (TE nvfp4_resgrad) folds ``residual + mlp_out`` into the unpermute that
applies the deferred FC2 bias, so it requires NVFP4_DEFER_FC2_BIAS=1. The reference model has
no expert bias; this unpermute sums the routed rows exactly as TE's ``_unpermute_kernel``
(FP32, routing order, one BF16 rounding) and then adds the residual rounded as the PyTorch
BF16 add, so results are bit-identical. The output is handed to nvfp4_resgrad's
bias-dropout-add through its ``_preadded`` map, which also routes the residual gradient to
the RMSNorm backward. Needs NVFP4_RESGRAD=1 installed first, with NVFP4_RESGRAD_FWD=0.
"""

import os

import torch
import triton
import triton.language as tl

_UNROLL = int(os.environ.get("NVFP4_UNPERMUTE_BIAS_UNROLL", "4"))
_WARPS = int(os.environ.get("NVFP4_UNPERMUTE_BIAS_WARPS", "4"))
_stats = {"fused": 0}


@triton.jit
def _unpermute_residual_kernel(
    input_ptr, row_id_map_ptr, residual_ptr, output_ptr,
    stride_row_id_map_token, stride_row_id_map_expert, stride_input_token, stride_output_token,
    num_experts: tl.constexpr, hidden_size: tl.constexpr, UNROLL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Skipped unrolled slots add +0.0, which leaves the accumulator unchanged because it
    # starts at +0.0 and can never become -0.0.
    pid_t = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < hidden_size
    map_row = row_id_map_ptr + pid_t * stride_row_id_map_token
    accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    n_routed = tl.load(map_row + num_experts * 2 * stride_row_id_map_expert)
    for idx in tl.static_range(UNROLL):
        valid = idx < n_routed
        src_row = tl.load(map_row + idx * stride_row_id_map_expert, mask=valid, other=0).to(tl.int64)
        x = tl.load(input_ptr + src_row * stride_input_token + offs, mask=mask & valid, other=0.0)
        accumulator += tl.where(valid, x.to(tl.float32), 0.0)
    for idx in tl.range(UNROLL, n_routed):
        src_row = tl.load(map_row + idx * stride_row_id_map_expert).to(tl.int64)
        x = tl.load(input_ptr + src_row * stride_input_token + offs, mask=mask)
        accumulator += x.to(tl.float32)
    out = accumulator.to(output_ptr.dtype.element_ty)
    row = pid_t.to(tl.int64) * stride_output_token
    r = tl.load(residual_ptr + row + offs, mask=mask, other=0.0)
    out = (r.to(tl.float32) + out.to(tl.float32)).to(output_ptr.dtype.element_ty)
    tl.store(output_ptr + row + offs, out, mask=mask)


def unpermute_residual(inp, row_id_map, residual, num_tokens, num_experts, hidden_size):
    output = torch.empty((num_tokens, hidden_size), dtype=inp.dtype, device=inp.device)
    _unpermute_residual_kernel[(num_tokens,)](
        inp, row_id_map, residual, output,
        row_id_map.stride(0), row_id_map.stride(1), inp.stride(0), output.stride(0),
        num_experts, hidden_size,
        UNROLL=_UNROLL, BLOCK_SIZE=triton.next_power_of_2(hidden_size), num_warps=_WARPS,
    )
    return output


def install():
    import transformer_engine.pytorch.triton.permutation as tp
    from megatron.core.transformer.transformer_layer import TransformerLayer
    from transformer_engine.pytorch.ops.fused import grouped_mlp as _te_grouped_mlp

    rg = _te_grouped_mlp._v7_module("nvfp4_resgrad")
    if os.environ.get("NVFP4_RESGRAD_FWD", "0") == "1":
        raise RuntimeError("FIRMUS_RESGRAD_FWD_NOBIAS replaces NVFP4_RESGRAD_FWD; set it to 0")
    if os.environ.get("NVFP4_DEFER_FC2_BIAS", "0") == "1":
        raise RuntimeError("FIRMUS_RESGRAD_FWD_NOBIAS is for experts without a deferred FC2 bias")
    if _UNROLL <= 0:
        raise RuntimeError("FIRMUS_RESGRAD_FWD_NOBIAS needs NVFP4_UNPERMUTE_BIAS_UNROLL>0")
    orig_unpermute = tp.unpermute_with_mask_map

    def unpermute_with_mask_map(inp, row_id_map, merging_probs, permuted_probs, pad_offsets,
                                num_tokens, num_experts, hidden_size):
        residual = rg._fwd_residual[0]
        if (
            residual is None
            or merging_probs is not None
            or permuted_probs is not None
            or pad_offsets is not None
            or tp._PENDING_EXPERT_BIAS
            or residual.dtype != inp.dtype
            or residual.numel() != num_tokens * hidden_size
            or not residual.is_contiguous()
            or inp.stride(1) != 1
        ):
            return orig_unpermute(inp, row_id_map, merging_probs, permuted_probs, pad_offsets,
                                  num_tokens, num_experts, hidden_size)
        rg._fwd_residual[0] = None
        output = unpermute_residual(inp, row_id_map, residual, num_tokens, num_experts, hidden_size)
        rg._preadded[rg._key(output)] = residual
        _stats["fused"] += 1
        return output, None

    tp.unpermute_with_mask_map = unpermute_with_mask_map
    orig_run_mlp = TransformerLayer._run_mlp

    def run_mlp(self, pre_mlp_layernorm_output, residual, *args, **kwargs):
        if self.training and torch.is_grad_enabled():
            rg._fwd_residual[0] = residual
        try:
            return orig_run_mlp(self, pre_mlp_layernorm_output, residual, *args, **kwargs)
        finally:
            rg._fwd_residual[0] = None

    TransformerLayer._run_mlp = run_mlp
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"FIRMUS resgrad fwd (no bias) installed (unroll {_UNROLL})", flush=True)
