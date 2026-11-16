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

"""Faster MXFP8 attention scale bookkeeping.

Replaces the rowwise BSHD -> BHSD block-scale permute of TE's mxfp8_quantize_fast_path
with a Triton kernel; output is byte-identical.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

from transformer_engine.pytorch.cpp_extensions.fused_attn import tex

MXFP8_BLOCK_SCALING_SIZE = 32

# Innermost byte extent of a rowwise scale the Triton path handles; anything
# else falls back to TE.
_ROWWISE_D_BYTES = 4


@triton.jit
def _transpose_bhsd_4b_kernel(
    i0, i1, i2,
    o0, o1, o2,
    S: tl.constexpr,
    H: tl.constexpr,
    BS: tl.constexpr,
    BH: tl.constexpr,
):
    """Transpose up to 3 x [B, S, H] int32 tensors into [B, H, S].

    One program per (batch, S-tile, tensor); the grid's z extent is the tensor count.
    """
    b = tl.program_id(0)
    ts = tl.program_id(1)
    z = tl.program_id(2)

    ip = tl.where(z == 0, i0, tl.where(z == 1, i1, i2))
    op = tl.where(z == 0, o0, tl.where(z == 1, o1, o2))
    ip = ip.to(tl.pointer_type(tl.int32))
    op = op.to(tl.pointer_type(tl.int32))

    s = ts * BS + tl.arange(0, BS)
    h = tl.arange(0, BH)
    mask = (s < S)[:, None] & (h < H)[None, :]

    val = tl.load(ip + b * S * H + s[:, None] * H + h[None, :],
                  mask=mask, other=0)
    tl.store(op + b * H * S + h[None, :] * S + s[:, None], val, mask=mask)


def _rowwise_transpose_supported(scale_list, outputs) -> bool:
    """Whether the Triton rowwise path applies to these scale tensors."""
    if not 1 <= len(scale_list) <= 3:
        return False
    for s, o in zip(scale_list, outputs):
        if s is None or o is None:
            return False
        if s.dim() != 4 or o.dim() != 4:
            return False
        if s.dtype != torch.uint8 or o.dtype != torch.uint8:
            return False
        # innermost extent must be 4 bytes and unpadded in the output
        if s.shape[-1] != _ROWWISE_D_BYTES or o.shape[-1] != s.shape[-1]:
            return False
        # (B, S, H, d) -> (B, H, S, d)
        if (o.shape[0], o.shape[1], o.shape[2]) != (
            s.shape[0], s.shape[2], s.shape[1]
        ):
            return False
        if not s.is_contiguous() or not o.is_contiguous():
            return False
    return True


def _rowwise_transpose_to_bhsd(scale_list, outputs) -> None:
    """Permute rowwise block scales BSHD -> BHSD, writing into outputs."""
    B, S, H, _ = scale_list[0].shape
    n = len(scale_list)

    src = [s.view(torch.int32).view(B, S, H) for s in scale_list]
    dst = [o.view(torch.int32).view(B, H, S) for o in outputs]
    # Pad to 3 so the kernel signature is fixed; the grid's z extent is n,
    # so the duplicated trailing entries are never visited.
    src = src + [src[-1]] * (3 - n)
    dst = dst + [dst[-1]] * (3 - n)

    BS = 128
    BH = triton.next_power_of_2(H)
    grid = (B, triton.cdiv(S, BS), n)
    _transpose_bhsd_4b_kernel[grid](
        src[0], src[1], src[2], dst[0], dst[1], dst[2],
        S=S, H=H, BS=BS, BH=BH, num_warps=8,
    )


def mxfp8_quantize_fast_path(tensor_quantizer_pairs, src_format):
    """Drop-in replacement for TE's mxfp8_quantize_fast_path; byte-identical output.

    src_format is "bshd" or "sbhd". Returns (fp8_tensors, "bhsd"): data in src_format,
    scale_inv in BHSD.
    """
    if not tensor_quantizer_pairs:
        return [], src_format
    assert src_format in ("bshd", "sbhd"), (
        f"mxfp8_quantize_fast_path only supports bshd/sbhd, got {src_format!r}."
    )
    _s_dim = {"bshd": 1, "sbhd": 0}
    _d_dim = {"bshd": 3, "sbhd": 3}

    fp8_tensors = []
    for tensor, quantizer in tensor_quantizer_pairs:
        original_shape = tensor.shape
        rs_shape = list(original_shape)
        rs_shape[_d_dim[src_format]] //= MXFP8_BLOCK_SCALING_SIZE
        cs_shape = list(original_shape)
        cs_shape[_s_dim[src_format]] //= MXFP8_BLOCK_SCALING_SIZE

        if src_format == "bshd":
            tensor2d = tensor.view(*tensor.shape[:2], -1)
        else:
            tensor2d = tensor.view(tensor.shape[0], -1)

        orig_optimize = quantizer.optimize_for_gemm
        quantizer.optimize_for_gemm = False
        fp8_tensor = quantizer(tensor2d)
        quantizer.optimize_for_gemm = orig_optimize

        fp8_tensor._rowwise_data = (
            fp8_tensor._rowwise_data.view(original_shape)
            if fp8_tensor._rowwise_data is not None else None
        )
        fp8_tensor._columnwise_data = (
            fp8_tensor._columnwise_data.view(original_shape)
            if fp8_tensor._columnwise_data is not None else None
        )
        fp8_tensor._rowwise_scale_inv = (
            fp8_tensor._rowwise_scale_inv.view(rs_shape)
            if fp8_tensor._rowwise_scale_inv is not None else None
        )
        fp8_tensor._columnwise_scale_inv = (
            fp8_tensor._columnwise_scale_inv.view(cs_shape)
            if fp8_tensor._columnwise_scale_inv is not None else None
        )
        fp8_tensors.append(fp8_tensor)

    rs_list = [t._rowwise_scale_inv for t in fp8_tensors]
    cs_list = [t._columnwise_scale_inv for t in fp8_tensors]

    def _align_up(x, a):
        return ((x + a - 1) // a) * a

    def _bhsd_shape(src_4d, d_pad):
        if src_format == "sbhd":
            S_, B_, H_, _ = src_4d.shape
        else:
            B_, S_, H_, _ = src_4d.shape
        return (B_, H_, S_, d_pad)

    def _build_outputs(scale_list, alignment):
        entries = []
        total = 0
        for s in scale_list:
            if s is None:
                entries.append(None)
                continue
            d_pad = _align_up(s.shape[-1], alignment)
            shape = _bhsd_shape(s, d_pad)
            numel = 1
            for dim in shape:
                numel *= dim
            entries.append((total, numel, shape))
            total += numel
        if total == 0:
            return [None] * len(scale_list)
        device = next(s for s in scale_list if s is not None).device
        buf = torch.empty(total, dtype=torch.uint8, device=device)
        return [buf[e[0]:e[0] + e[1]].view(e[2]) if e is not None else None
                for e in entries]

    rs_outs = _build_outputs(rs_list, 4)
    cs_outs = _build_outputs(cs_list, 128)

    # Rowwise: Triton when the layout allows it, TE otherwise.
    if src_format == "bshd" and _rowwise_transpose_supported(rs_list, rs_outs):
        _rowwise_transpose_to_bhsd(rs_list, rs_outs)
        rs_permuted = rs_outs
    else:
        rs_permuted = tex.multi_tensor_transpose_to_bhsd(
            rs_list, original_format=src_format, outputs=rs_outs
        )

    cs_permuted = tex.multi_tensor_transpose_to_bhsd(
        cs_list, original_format=src_format, outputs=cs_outs
    )

    for t, rp, cp in zip(fp8_tensors, rs_permuted, cs_permuted):
        t._rowwise_scale_inv = (
            rp.view(-1, rp.shape[-1]) if rp is not None else None
        )
        t._columnwise_scale_inv = (
            cp.view(-1, cp.shape[-1]) if cp is not None else None
        )

    tex.multi_tensor_swizzle_scales_for_gemm_unchecked_(fp8_tensors, True, False)
    tex.multi_tensor_swizzle_scales_for_gemm_unchecked_(fp8_tensors, False, True)
    for t in fp8_tensors:
        t._with_gemm_swizzled_scales = True

    return fp8_tensors, "bhsd"


def _scale_tail(fp8_tensors, src_format="bshd"):
    """BHSD-permute and GEMM-swizzle the block scales of fp8_tensors.

    The second half of mxfp8_quantize_fast_path, shared with fused producers.
    """
    rs_list = [t._rowwise_scale_inv for t in fp8_tensors]
    cs_list = [t._columnwise_scale_inv for t in fp8_tensors]

    def _align_up(x, a):
        return ((x + a - 1) // a) * a

    def _build_outputs(scale_list, alignment):
        entries = []
        total = 0
        for s in scale_list:
            if s is None:
                entries.append(None)
                continue
            d_pad = _align_up(s.shape[-1], alignment)
            B_, S_, H_, _ = s.shape
            shape = (B_, H_, S_, d_pad)
            numel = 1
            for dim in shape:
                numel *= dim
            entries.append((total, numel, shape))
            total += numel
        if total == 0:
            return [None] * len(scale_list)
        device = next(s for s in scale_list if s is not None).device
        buf = torch.empty(total, dtype=torch.uint8, device=device)
        return [buf[e[0]:e[0] + e[1]].view(e[2]) if e is not None else None
                for e in entries]

    rs_outs = _build_outputs(rs_list, 4)
    cs_outs = _build_outputs(cs_list, 128)

    if src_format == "bshd" and _rowwise_transpose_supported(rs_list, rs_outs):
        _rowwise_transpose_to_bhsd(rs_list, rs_outs)
        rs_permuted = rs_outs
    else:
        rs_permuted = tex.multi_tensor_transpose_to_bhsd(
            rs_list, original_format=src_format, outputs=rs_outs
        )
    cs_permuted = tex.multi_tensor_transpose_to_bhsd(
        cs_list, original_format=src_format, outputs=cs_outs
    )

    for t, rp, cp in zip(fp8_tensors, rs_permuted, cs_permuted):
        t._rowwise_scale_inv = rp.view(-1, rp.shape[-1])
        t._columnwise_scale_inv = cp.view(-1, cp.shape[-1])

    tex.multi_tensor_swizzle_scales_for_gemm_unchecked_(fp8_tensors, True, False)
    tex.multi_tensor_swizzle_scales_for_gemm_unchecked_(fp8_tensors, False, True)
    for t in fp8_tensors:
        t._with_gemm_swizzled_scales = True
    return fp8_tensors


def assemble_prequantized(triples, quantizer, nominal_dtype):
    """Wrap producer-emitted MXFP8 buffers (BSHD data and scales) as MXFP8Tensors.

    triples: one MXFP8Triple per tensor in q/k/v order. Returns (tensors, "bhsd").
    """
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor

    tensors = [
        MXFP8Tensor(
            shape=t.rd.shape,
            dtype=nominal_dtype,
            rowwise_data=t.rd,
            rowwise_scale_inv=t.rs,
            columnwise_data=t.cd,
            columnwise_scale_inv=t.cs,
            quantizer=quantizer,
            requires_grad=False,
            fp8_dtype=getattr(quantizer, "dtype", None)
            or quantizer.fp8_dtype,
            with_gemm_swizzled_scales=False,
        )
        for t in triples
    ]
    return _scale_tail(tensors, "bshd"), "bhsd"


def combine_and_quantize(
    qkv_layout,
    q,
    k,
    v,
    qkv_quantizer,
    used_in_forward: bool = True,
    used_in_backward: bool = False,
    combined_qkv: Optional[torch.Tensor] = None,
    combined_kv: Optional[torch.Tensor] = None,
):
    """MXFP8-only combine_and_quantize over the faster fast path.

    Handles only the separate-QKV bshd/sbhd layouts.
    """
    from transformer_engine.pytorch.attention.dot_product_attention.utils import (
        get_qkv_format,
    )

    qkv_format, q_format, kv_format = get_qkv_format(qkv_layout)
    assert qkv_format in ("bshd", "sbhd"), (
        "combine_and_quantize only supports bshd/sbhd for MXFP8, got "
        f"{qkv_format!r}."
    )
    assert qkv_layout in ("bshd_bshd_bshd", "sbhd_sbhd_sbhd"), (
        f"unsupported qkv_layout {qkv_layout!r} for the fast path"
    )

    _s_dim = {"sbhd": 0, "bshd": 1}
    _d_dim = {"sbhd": 3, "bshd": 3}
    d_qk = q.shape[_d_dim[qkv_format]]
    d_v = v.shape[_d_dim[qkv_format]]
    s_q = q.shape[_s_dim[q_format]]
    s_kv = v.shape[_s_dim[kv_format]]
    assert (
        s_q % 128 == 0 and s_kv % 128 == 0 and d_qk % 32 == 0 and d_v % 32 == 0
    ), (
        "MXFP8 quantization requires s_q % 128 == 0, s_kv % 128 == 0, "
        f"d_qk % 32 == 0, d_v % 32 == 0. Found {s_q=}, {s_kv=}, {d_qk=}, {d_v=}."
    )

    q_quantizer, k_quantizer, v_quantizer = [
        qkv_quantizer.copy() for _ in range(3)
    ]
    if used_in_forward and not used_in_backward:
        q_quantizer.rowwise_usage = True
        q_quantizer.columnwise_usage = False
        k_quantizer.rowwise_usage = True
        k_quantizer.columnwise_usage = False
        v_quantizer.rowwise_usage = False
        v_quantizer.columnwise_usage = True
    elif (not used_in_forward) and used_in_backward:
        q_quantizer.rowwise_usage = True
        q_quantizer.columnwise_usage = True
        k_quantizer.rowwise_usage = True
        k_quantizer.columnwise_usage = True
        v_quantizer.rowwise_usage = True
        v_quantizer.columnwise_usage = False

    (q_fp8, k_fp8, v_fp8), scale_inv_format = mxfp8_quantize_fast_path(
        [(q, q_quantizer), (k, k_quantizer), (v, v_quantizer)], qkv_format
    )
    return q_fp8, k_fp8, v_fp8, qkv_layout, scale_inv_format
