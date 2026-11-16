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

"""Quantize the MoE tokens to MXFP8 inside the permute and hand TE's grouped MLP the pre-quantized grouped operand.

Rebinds deep_ep.HybridEPBuffer.dispatch_with_permute and grouped_mlp._group_quantize_for_grouped_mlp.  At EP=1 the
dispatch copy and permute become one gpt_oss::moe_qperm_gather_quant launch reading each token row once per expert slot
and writing the MXFP8 rowwise/columnwise data and E8M0 scales directly.  The forward returns a never-written bf16 [P, H]
placeholder whose data pointer keys the parked operands, the backward the real bf16 gradient rows; TE's quantize entry
returns a GroupedTensor over the parked buffers and takes its original path otherwise.  MOE_DGLU_STRIP=1: the combine
backward in residual_bias_hook parks its own quantized gradient via park_backward() and the backward branch here
takes HybridEP's path.  MOE_ROUTING_META=1: moe_routing_meta derives the routing metadata from the local [T, E]
map instead of HybridEP's planner; a source-hash-checked copy of _HybridEPManager.dispatch drops the over_budget mask
ops and TE's splits_to_offsets_multi returns the kernel's offsets (MOE_ROUTING_META_DISPATCH / MOE_ROUTING_META_OFFS).
Each fast path is checked bitwise on its first eager call; a failed check, EP > 1, fp8 dispatch, permute fusion or other
dtypes keep the original code.  MOE_QPERM=0 leaves the reference path; MOE_QPERM_FWD / MOE_QPERM_BWD,
MOE_QPERM_SELFCHECK and MOE_QPERM_RW_LAYOUT / MOE_QPERM_CW_LAYOUT / MOE_QPERM_DIV are the finer gates.
"""
import atexit
import functools
import hashlib
import inspect
import os
import sys

import torch

_INSTALLED = False
_ORIG = {}
_STATS = {"fwd_fast": 0, "fwd_plain": 0, "bwd_fast": 0, "bwd_plain": 0, "stash_hit_fwd": 0, "stash_hit_bwd": 0,
          "stash_miss": 0, "fallback": 0, "meta_fast": 0, "meta_slow": 0, "meta_not_ready": 0, "park_bwd": 0,
          "offs_hit": 0, "offs_miss": 0, "offs_evicted": 0, "ob_folded": 0, "dispatch_mask_ops": 0}
_STASH = {}            # data_ptr -> _Parked (the quantized operands waiting for TE's grouped MLP)
_STASH_ORDER = []      # insertion order, for eviction
_R2T_CACHE = {}        # id(handle tuple) -> (handle tuple (strong ref), row_to_token)
_R2T_ORDER = []
_FWD_OK = [True]
_FWD_CHECKED = [False]
_BWD_OK = [True]
_BWD_CHECKED = [False]

ENABLED = os.getenv("MOE_QPERM", "1") == "1"
FWD = os.getenv("MOE_QPERM_FWD", "1") == "1"
BWD = os.getenv("MOE_QPERM_BWD", "1") == "1"
SELFCHECK = os.getenv("MOE_QPERM_SELFCHECK", "1") == "1"
RW_LAYOUT = int(os.getenv("MOE_QPERM_RW_LAYOUT", "0"))
CW_LAYOUT = int(os.getenv("MOE_QPERM_CW_LAYOUT", "3"))
DIV_MODE = int(os.getenv("MOE_QPERM_DIV", "0"))
CW_PARAMS = [int(x) for x in os.getenv("MOE_QPERM_CW_GENERIC", "").split(",") if x.strip()]   # layout 4 only
_PLACEHOLDER_PTRS = set()   # data pointers of forward placeholders whose parked operands have not been consumed
MAX_LOG = 3
MAX_STASH = 64
DGLU = os.getenv("MOE_DGLU_STRIP", "1") == "1"   # the combine backward parks its own operands (see module doc)
# Source of the fc2 bias gradient once TE's grouped dbias/dscales pass is gone.  'gemm': the ones-operand grouped wgrad
# GEMM over the MXFP8 columnwise dz; 'bf16': the combine-backward gather also writes the bf16 dz rows and TE's own Triton
# pass (compute_grouped_dbias, fp32 atomics) reduces them; 'fused': the gather kernel accumulates the fp32 column sum of
# the bf16 (pre-quantization) dz rows per expert, no second pass.
DBIAS2_MODE = os.getenv("MOE_DGLU_STRIP_DBIAS2", "fused").strip().lower()
if DBIAS2_MODE not in ("gemm", "bf16", "fused"):
    raise ValueError(f"MOE_DGLU_STRIP_DBIAS2={DBIAS2_MODE!r}: expected gemm, bf16 or fused")
RECORD_BWD_OPERANDS = False   # set by moe_dglu_strip_hook.install(): its fc2-dbias shim pops every entry it records
_DGLU_ENGAGED = [False]       # set by moe_dglu_strip_hook.install() once the TE epilogue patches are in place
_BWD_OPERANDS = {}      # data_ptr -> _Parked of a consumed backward stash entry, waiting for the fc2 bias gradient
_BWD_ORDER = []
# the EP=1 routing-metadata fast path (see the module doc)
META = os.getenv("MOE_ROUTING_META", "1") == "1"
META_OFFS = os.getenv("MOE_ROUTING_META_OFFS", "1") == "1"
META_DISPATCH = os.getenv("MOE_ROUTING_META_DISPATCH", "1") == "1"
_META_OK = [True]
_META_CFG = {}          # (T, T_slots, H, E, pad, P, fp8) -> (chunk,): HybridEP's template config, read once per shape
_META_READY_SAID = [False]
_NEXT_OB = [None]       # the manager's over_budget accumulator for the dispatch in flight (set by the re-bound dispatch)
_DUMMY_U8 = {}
_OFFS_STASH = {}        # data_ptr(padded counts) -> (padded counts, the six offset tensors) waiting for TE's fuser
_OFFS_ORDER = []
_OFFS_SPEC = {}         # the call shape of TE's splits_to_offsets_multi, learnt and verified on its first call
_OFFS_OK = [None]       # None = not learnt yet, True = kernel offsets handed back, False = TE keeps computing them
_DISPATCH_PATCHED = [False]
# md5 of inspect.getsource(_HybridEPManager.dispatch) in the container's Megatron (3rdparty/Megatron-LM
# megatron/core/transformer/moe/token_dispatcher.py); the re-binding below is a copy of exactly that method and is
# skipped on any other source
_MEGATRON_DISPATCH_MD5 = "2e9c4fe5f933a406ba554d73a1a45805"


def _say(msg: str) -> None:
    rank = os.getenv("RANK", os.getenv("SLURM_PROCID", "0"))
    sys.__stdout__.write(f"MOE_QPERM rank={rank} {msg}\n")
    sys.__stdout__.flush()


def _fallback(what: str) -> None:
    _STATS["fallback"] += 1
    if _STATS["fallback"] <= 12:
        _say(f"fallback: {what}")


class _Parked:
    __slots__ = ("rw", "rws", "cw", "cws", "P", "H", "E", "holder", "direction", "dbias")

    def __init__(self, rw, rws, cw, cws, P, H, E, holder, direction, dbias=None):
        self.rw, self.rws, self.cw, self.cws = rw, rws, cw, cws
        self.P, self.H, self.E = P, H, E
        self.holder = holder            # keeps the keyed tensor alive so its address cannot be reused meanwhile
        self.direction = direction
        self.dbias = dbias              # bwd, DBIAS2_MODE == 'fused': [E, H] fp32 column sum of the bf16 dz rows


def _park(key_tensor, parked):
    key = key_tensor.data_ptr()
    if key in _STASH:
        _fallback(f"stash key reused while still parked ({parked.direction}); the older entry is dropped")
        _STASH.pop(key, None)
        _PLACEHOLDER_PTRS.discard(key)
        if key in _STASH_ORDER:
            _STASH_ORDER.remove(key)
    _STASH[key] = parked
    _STASH_ORDER.append(key)
    while len(_STASH_ORDER) > MAX_STASH:
        old = _STASH_ORDER.pop(0)
        _STASH.pop(old, None)
        _PLACEHOLDER_PTRS.discard(old)
        _STATS["stash_miss"] += 1


def _is_placeholder(t: torch.Tensor) -> bool:
    try:
        return t.data_ptr() in _PLACEHOLDER_PTRS
    except Exception:  # noqa
        return False


def _make_placeholder(P, H, device):
    """The 'dispatched tokens' tensor Megatron hands the TE grouped MLP: a contiguous bf16 [P, H] allocation that is
    never written or read (TE's maybe_dequantize would materialise anything non-contiguous, so a stride-0 view
    cannot stand in).  Its data pointer keys the parked MXFP8 operands; the allocation itself costs no bandwidth."""
    t = torch.empty((P, H), dtype=torch.bfloat16, device=device)
    _PLACEHOLDER_PTRS.add(t.data_ptr())
    return t


def park_backward(P, H, E, rw, rws, cw, cws, device, rows=None, dbias=None):
    """Park the combine backward's pre-quantized fc2 output gradient (rowwise + columnwise MXFP8) under the bf16 [P, H]
    tensor that stands in for the permuted gradient rows: `rows` (the real bf16 dz rows, read by TE's dbias pass in
    DBIAS2_MODE 'bf16') or, when None, a fresh never-written placeholder.  `dbias` (DBIAS2_MODE 'fused') rides along
    for the fc2-dbias shim.  Returns the keyed tensor."""
    if rows is None:
        placeholder = _make_placeholder(P, H, device)
    else:
        assert rows.shape == (P, H) and rows.dtype == torch.bfloat16 and rows.is_contiguous(), (rows.shape, rows.dtype)
        placeholder = rows
        _PLACEHOLDER_PTRS.add(placeholder.data_ptr())
    _park(placeholder, _Parked(rw, rws, cw, cws, P, H, E, placeholder, "bwd", dbias))
    _STATS["park_bwd"] += 1
    return placeholder


def unpark(placeholder):
    """Drop a parked entry (and the placeholder marker) so the tensor can be written and quantized by TE itself."""
    key = placeholder.data_ptr()
    _STASH.pop(key, None)
    _PLACEHOLDER_PTRS.discard(key)
    try:
        _STASH_ORDER.remove(key)
    except ValueError:
        pass


def pop_backward_operand(ptr):
    """The backward stash entry TE's grouped MLP consumed for the tensor at `ptr` (None when it took the plain path)."""
    st = _BWD_OPERANDS.pop(ptr, None)
    if st is not None:
        try:
            _BWD_ORDER.remove(ptr)
        except ValueError:
            pass
    return st


# ------------------------------------------------------------------------------------------------------------
# 2. TE: _group_quantize_for_grouped_mlp
# ------------------------------------------------------------------------------------------------------------
def _grouped_from_parked(st, quantizer, num_groups, split_sizes, tensor_offsets):
    from transformer_engine.pytorch.tensor.grouped_tensor import GroupedTensor
    want_col = bool(getattr(quantizer, "columnwise_usage", True))
    first_dims = split_sizes.to(dtype=torch.int64, device=st.rw.device)
    return GroupedTensor(
        shape=(st.P, st.H),
        dtype=torch.bfloat16,
        quantizer=quantizer,
        num_tensors=num_groups,
        data=st.rw,
        columnwise_data=st.cw if want_col else None,
        scale_inv=st.rws,
        columnwise_scale_inv=st.cws if want_col else None,
        first_dims=first_dims,
        tensor_offsets=tensor_offsets,
        with_gemm_swizzled_scales=True,
    )


def _quantizer_ok(quantizer) -> bool:
    try:
        from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
        from transformer_engine.pytorch.constants import DType
    except Exception:  # noqa
        return False
    return (isinstance(quantizer, MXFP8Quantizer) and quantizer.dtype == DType.kFloat8E4M3
            and bool(getattr(quantizer, "rowwise_usage", True)) and not getattr(quantizer, "with_2d_quantization", False)
            and bool(getattr(quantizer, "optimize_for_gemm", False)))


def _make_group_quantize(orig):
    def _group_quantize_for_grouped_mlp(tensor, quantizer, num_groups, split_sizes, *, tensor_offsets=None,
                                        use_dense_single_group=False):
        st = None
        if torch.is_tensor(tensor) and tensor.dim() == 2 and tensor.dtype == torch.bfloat16 and _STASH:
            st = _STASH.get(tensor.data_ptr())
        if st is not None:
            ok = (st.P == tensor.shape[0] and st.H == tensor.shape[1] and st.E == num_groups
                  and not use_dense_single_group and _quantizer_ok(quantizer) and split_sizes is not None
                  and split_sizes.numel() == num_groups)
            if ok:
                _STASH.pop(tensor.data_ptr(), None)
                _PLACEHOLDER_PTRS.discard(tensor.data_ptr())
                try:
                    _STASH_ORDER.remove(tensor.data_ptr())
                except ValueError:
                    pass
                try:
                    g = _grouped_from_parked(st, quantizer, num_groups, split_sizes, tensor_offsets)
                except Exception as ex:  # noqa
                    if st.direction == "fwd":
                        raise RuntimeError(f"MOE_QPERM: GroupedTensor from the parked operands raised {type(ex).__name__}: {ex}; "
                                           "the forward placeholder cannot be quantized -- refusing to continue") from ex
                    _fallback(f"GroupedTensor from the parked operands raised {type(ex).__name__}: {ex}; TE quantizes the bf16 rows")
                    return orig(tensor, quantizer, num_groups, split_sizes, tensor_offsets=tensor_offsets,
                                use_dense_single_group=use_dense_single_group)
                k = "stash_hit_fwd" if st.direction == "fwd" else "stash_hit_bwd"
                _STATS[k] += 1
                if st.direction == "bwd" and RECORD_BWD_OPERANDS:
                    _BWD_OPERANDS[tensor.data_ptr()] = st        # the fc2 bias gradient reads the columnwise copy
                    _BWD_ORDER.append(tensor.data_ptr())
                    while len(_BWD_ORDER) > MAX_STASH:
                        _BWD_OPERANDS.pop(_BWD_ORDER.pop(0), None)
                if _STATS[k] <= MAX_LOG:
                    _say(f"{st.direction}: TE grouped MLP takes the pre-quantized MXFP8 operand "
                         f"[{st.P}, {st.H}] x {num_groups} groups (columnwise={'yes' if g.columnwise_data is not None else 'no'})")
                return g
            if st.direction == "fwd" or _is_placeholder(tensor):
                raise RuntimeError(f"MOE_QPERM: the {st.direction} placeholder reached TE's quantize with mismatching "
                                   f"metadata (shape {tuple(tensor.shape)} vs parked ({st.P}, {st.H}), groups {num_groups} vs {st.E}, "
                                   f"quantizer {type(quantizer).__name__}); refusing to continue")
            _fallback("parked backward operand does not match TE's request; TE quantizes the bf16 rows")
            _STASH.pop(tensor.data_ptr(), None)
        elif _is_placeholder(tensor):
            raise RuntimeError("MOE_QPERM: a forward placeholder reached TE's quantize without its parked operands "
                               f"(shape {tuple(tensor.shape)}); refusing to continue")
        elif _STASH and torch.is_tensor(tensor) and tensor.dim() == 2 and tensor.dtype == torch.bfloat16 and any(
                v.P == tensor.shape[0] and v.H == tensor.shape[1] for v in _STASH.values()):
            _STATS["stash_miss"] += 1
            if _STATS["stash_miss"] <= 6:
                _say(f"stash miss: TE quantizes a [{tensor.shape[0]}, {tensor.shape[1]}] bf16 tensor (ptr {tensor.data_ptr():#x} stride "
                     f"{tuple(tensor.stride())}) while {len(_STASH)} parked operand(s) wait ({[(hex(k), v.direction) for k, v in list(_STASH.items())[:3]]})")
        return orig(tensor, quantizer, num_groups, split_sizes, tensor_offsets=tensor_offsets,
                    use_dense_single_group=use_dense_single_group)
    return _group_quantize_for_grouped_mlp


# ------------------------------------------------------------------------------------------------------------
# 1. HybridEP: dispatch_with_permute
# ------------------------------------------------------------------------------------------------------------
_HANDLE_FIELDS = ("sparse_to_dense_map", "rdma_to_attn_map", "attn_to_rdma_map", "num_dispatched_tokens_tensor",
                  "local_expert_routing_map", "dense_chunk_layout", "dense_to_expert_map", "tokens_per_expert",
                  "num_of_tokens_per_rank", "config", "num_of_valid_tokens", "overflow_flag")


def _padded_counts(handle_impl, tokens_per_expert, pad_multiple):
    """HybridEP's padded_tokens_per_expert, recomputed from the raw counts (HybridEP fills its own copy inside the
    dispatch kernel call this path no longer makes); int64 on the device, as Megatron consumes it."""
    t = tokens_per_expert.to(torch.int64)
    return ((t + (pad_multiple - 1)) // pad_multiple) * pad_multiple


class _FastMeta:
    """Slot 9 (HybridEP's config) of a handle built by the routing-metadata fast path: the forward's counts and maps,
    kept for the backward.  HybridEP's own Python wrapper cannot unpack such a handle (its config setter rejects the
    object), so a path that still needs HybridEP's runtime fails loudly instead of running on half-filled maps."""
    __slots__ = ("T_slots", "chunk", "tpe32", "padded", "r2t", "pprob", "over_budget_folded")

    def __init__(self, T_slots, chunk, tpe32, padded, r2t, pprob, over_budget_folded):
        self.T_slots, self.chunk = T_slots, chunk
        self.tpe32, self.padded, self.r2t, self.pprob = tpe32, padded, r2t, pprob
        self.over_budget_folded = over_budget_folded


def _fast_meta(handle):
    if isinstance(handle, tuple) and len(handle) == len(_HANDLE_FIELDS) and isinstance(handle[9], _FastMeta):
        return handle[9]
    return None


def _padded_counts_for(handle, tokens_per_expert, pad_multiple):
    """The padded int64 counts of this handle: the fast path's stashed tensor, else the four-launch arithmetic."""
    meta = _fast_meta(handle)
    if meta is not None:
        return meta.padded
    return _padded_counts(None, tokens_per_expert, pad_multiple)


def _tpe32_for(handle, tokens_per_expert):
    meta = _fast_meta(handle)
    if meta is not None:
        return meta.tpe32
    return tokens_per_expert.to(torch.int32).contiguous()


def _tensor_offsets(padded_counts, H):
    z = torch.zeros((1,), dtype=torch.int64, device=padded_counts.device)
    return torch.cat([z, torch.cumsum(padded_counts.to(torch.int64), 0)]) * H


def _te_group_quantize_reference(rows_bf16, padded_counts, H):
    """tex.group_quantize of a bf16 [P, H] tensor the way the fuser calls it (E4M3, rowwise+columnwise, swizzled)."""
    import transformer_engine.pytorch  # noqa: F401  (loads the transformer_engine_torch extension)
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
    q = MXFP8Quantizer(tex.DType.kFloat8E4M3, rowwise=True, columnwise=True)
    q.optimize_for_gemm = True
    q.internal = True
    E = padded_counts.numel()
    split = padded_counts.to(torch.int64)
    offs = _tensor_offsets(split, H)
    try:
        g = tex.group_quantize(rows_bf16, q, E, split, tensor_offsets=offs)
    except TypeError:
        g = tex.group_quantize(rows_bf16, q, E, split)
    return g


def _generic_positions(k, mb, params):
    rk0, rk1, sk0, sk1, sk2, rm0, rm1, sm0, sm1, sm2 = params
    return ((k % rk0) * sk0 + ((k // rk0) % rk1) * sk1 + (k // (rk0 * rk1)) * sk2
            + (mb % rm0) * sm0 + ((mb // rm0) % rm1) * sm1 + (mb // (rm0 * rm1)) * sm2)


def _masked_scale_equal(a, b, P, H, n_used, layout, columnwise, padded_counts=None):
    """Compare two scale buffers only on the entries that belong to rows < n_used (TE may leave the tail)."""
    import custom_kernels.moe_qperm as K
    dev = a.device
    KT = (H + 127) // 128
    if columnwise:
        if layout in (3, 4):
            k = torch.arange(H, device=dev)[:, None].expand(H, n_used // 32)
            mb = torch.arange(n_used // 32, device=dev)[None, :].expand(H, n_used // 32)
            if layout == 3:
                idx = K.grouped_columnwise_positions(k, mb, padded_counts.to(torch.int64), H).reshape(-1)
            else:
                idx = _generic_positions(k.to(torch.int64), mb.to(torch.int64), CW_PARAMS).reshape(-1)
        elif layout == 2:
            K_PAD = ((H + 127) // 128) * 128
            idx = (torch.arange(n_used // 32, device=dev)[:, None] * K_PAD + torch.arange(H, device=dev)[None, :]).reshape(-1)
        else:
            k = torch.arange(H, device=dev)[:, None].expand(H, n_used // 32)
            mb = torch.arange(n_used // 32, device=dev)[None, :].expand(H, n_used // 32)
            if layout == 0:
                idx = K.swizzled_positions(k, mb, P // 128).reshape(-1)
            else:
                idx = (((mb // 4) * KT + k // 128) * 512 + (k % 32) * 16 + ((k // 32) % 4) * 4 + (mb % 4)).reshape(-1)
    else:
        nb = H // 32
        r = torch.arange(n_used, device=dev)[:, None].expand(n_used, nb)
        c = torch.arange(nb, device=dev)[None, :].expand(n_used, nb)
        if layout == 0:
            idx = K.swizzled_positions(r, c, KT).reshape(-1)
        else:
            KB_PAD = ((nb + 3) // 4) * 4
            idx = (r * KB_PAD + c).reshape(-1)
    return bool(torch.equal(a[idx], b[idx]))


def _compare_quantized(tag, rw, rws, cw, cws, ref_rows, padded_counts, P, H):
    """Bitwise comparison of the gather's MXFP8 operands against tex.group_quantize(HybridEP's bf16 rows)."""
    n_used = int(padded_counts.to(torch.int64).sum().item())
    g = _te_group_quantize_reference(ref_rows, padded_counts, H)
    g_rw = g.rowwise_data.reshape(-1)
    g_cw = g.columnwise_data.reshape(-1) if g.columnwise_data is not None else None
    g_rws = g.scale_inv.reshape(-1)
    g_cws = g.columnwise_scale_inv.reshape(-1) if g.columnwise_scale_inv is not None else None
    res = {}
    res["rw_data"] = bool(torch.equal(g_rw[: n_used * H], rw[: n_used * H]))
    res["cw_data"] = bool(g_cw is not None and torch.equal(g_cw[: n_used * H], cw[: n_used * H]))
    res["rws_size"] = (g_rws.numel() == rws.numel())
    res["cws_size"] = (g_cws is not None and g_cws.numel() == cws.numel())
    res["rws_full"] = bool(res["rws_size"] and torch.equal(g_rws, rws))
    res["cws_full"] = bool(res["cws_size"] and torch.equal(g_cws, cws))
    res["rws_used"] = bool(res["rws_size"] and (res["rws_full"] or _masked_scale_equal(g_rws, rws, P, H, n_used, RW_LAYOUT, False)))
    res["cws_used"] = bool(res["cws_size"] and (res["cws_full"] or _masked_scale_equal(g_cws, cws, P, H, n_used, CW_LAYOUT, True, padded_counts)))
    if not res["rw_data"]:
        d = (g_rw[: n_used * H] != rw[: n_used * H]).sum().item()
        res["rw_data_mismatch"] = int(d)
    if not res["cw_data"] and g_cw is not None:
        res["cw_data_mismatch"] = int((g_cw[: n_used * H] != cw[: n_used * H]).sum().item())
    if res["rws_size"] and not res["rws_full"]:
        res["rws_mismatch"] = int((g_rws != rws).sum().item())
    if res["cws_size"] and not res["cws_full"]:
        res["cws_mismatch"] = int((g_cws != cws).sum().item())
    ok = res["rw_data"] and res["cw_data"] and res["rws_used"] and res["cws_used"]
    _say(f"{tag} self-check vs tex.group_quantize: {'OK' if ok else 'FAIL'} {res} n_used={n_used} "
         f"swizzled={getattr(g, '_with_gemm_swizzled_scales', None)} layouts rw={RW_LAYOUT} cw={CW_LAYOUT} div={DIV_MODE}")
    return ok


def _handle_tuple(handle_impl):
    return tuple(getattr(handle_impl, f) for f in _HANDLE_FIELDS)


def _row_to_token(handle, m, probs, layout, P, has_probs):
    """row_to_token (+ permuted probs) for this handle; the forward's map is reused by the backward."""
    meta = _fast_meta(handle)
    if meta is not None and meta.r2t.numel() == P:
        return meta.r2t, (meta.pprob if has_probs else None)
    key = id(handle)
    ent = _R2T_CACHE.get(key)
    if ent is not None and ent[0] is handle and ent[1].numel() == P and not has_probs:
        return ent[1], None
    r2t, pp = torch.ops.gpt_oss.moe_qperm_inverse_map(m, probs, layout, P, has_probs)
    if handle is not None:
        _R2T_CACHE[key] = (handle, r2t)
        _R2T_ORDER.append(key)
        while len(_R2T_ORDER) > MAX_STASH:
            _R2T_CACHE.pop(_R2T_ORDER.pop(0), None)
    return r2t, pp


def _fast_path_common(self, hidden, handle, pad_multiple, num_permuted_tokens, fuse, use_fp8, scaling_factor):
    """Returns None when the call must take HybridEP's own path."""
    if getattr(self, "group_size", None) != 1:
        return "EP group of more than one rank"
    if fuse or use_fp8 or scaling_factor is not None:
        return "permute fusion into HybridEP / fp8 dispatch / scaling factors"
    if not (torch.is_tensor(hidden) and hidden.dim() == 2 and hidden.dtype == torch.bfloat16 and hidden.is_contiguous()):
        return f"hidden dtype/layout {getattr(hidden, 'dtype', None)} {tuple(getattr(hidden, 'shape', ()))}"
    if not (isinstance(num_permuted_tokens, int) and num_permuted_tokens > 0):
        return "num_permuted_tokens not given (blocking HybridEP path)"
    if not (isinstance(pad_multiple, int) and pad_multiple > 0 and pad_multiple % 32 == 0 and num_permuted_tokens % pad_multiple == 0
            and num_permuted_tokens % 128 == 0):
        return f"pad_multiple {pad_multiple} / num_permuted_tokens {num_permuted_tokens}"
    if hidden.shape[1] % 32 != 0:
        return f"hidden size {hidden.shape[1]} not a multiple of 32"
    return None


def _make_dispatch_with_permute(orig):
    @functools.wraps(orig)          # Megatron inspects the signature (fuse_permute_dispatch) before calling
    def dispatch_with_permute(self, **kw):
        hidden = kw.get("hidden")
        handle = kw.get("handle")
        if handle is None:
            return _forward(self, orig, kw)
        return _backward(self, orig, kw)
    return dispatch_with_permute


def _forward(self, orig, kw):
    hidden = kw.get("hidden")
    routing_map = kw.get("routing_map")
    probs = kw.get("probs")
    pad_multiple = kw.get("pad_multiple")
    P = kw.get("num_permuted_tokens")
    E_local = kw.get("num_of_experts_per_rank")
    why = None
    if not (FWD and _FWD_OK[0]):
        why = "forward fast path off"
    else:
        why = _fast_path_common(self, hidden, None, pad_multiple, P, kw.get("fuse_permute_dispatch", False),
                                kw.get("use_fp8"), kw.get("scaling_factor"))
    if why is None:
        if not (torch.is_tensor(routing_map) and routing_map.dim() == 2 and routing_map.shape[0] == hidden.shape[0]
                and kw.get("topk_idx") is None):
            why = "routing map shape / topk_idx routing"
        elif not (torch.is_tensor(probs) and probs.shape == routing_map.shape):
            why = "probs shape"
        elif not (isinstance(E_local, int) and E_local > 0 and E_local & (E_local - 1) == 0 and routing_map.shape[1] == E_local):
            why = f"experts per rank {E_local} vs routing map {tuple(routing_map.shape)} (EP=1 needs them equal, power of two)"
        elif kw.get("num_of_tokens_per_rank") not in (None, hidden.shape[0]):
            why = "num_of_tokens_per_rank differs from the local token count"
    if why is not None:
        _STATS["fwd_plain"] += 1
        if _STATS["fwd_plain"] <= MAX_LOG:
            _fallback(f"forward: {why}; HybridEP dispatch_with_permute")
        return orig(self, **kw)

    T, H = hidden.shape
    first_call = SELFCHECK and not _FWD_CHECKED[0]
    use_meta = META and _META_OK[0] and _meta_ready()
    if META and _META_OK[0] and not use_meta and not first_call:
        _STATS["meta_not_ready"] += 1
    fast = None
    ob_before = None
    if META and _META_OK[0] and (use_meta or first_call):
        try:
            if first_call:
                # the self-check call: HybridEP's handle is the one returned, so the kernel ORs into a scratch copy of
                # over_budget (Megatron's own two mask ops run on the real one) and emits offsets for test strides
                ob_before = _NEXT_OB[0].clone() if torch.is_tensor(_NEXT_OB[0]) else None
                fast = _meta_forward(self, hidden, routing_map, probs, pad_multiple, P, E_local, kw,
                                     ob_override=(ob_before.clone() if ob_before is not None else None),
                                     strides_override=[3, 5, 7, 11])
            else:
                fast = _meta_forward(self, hidden, routing_map, probs, pad_multiple, P, E_local, kw)
        except Exception as ex:  # noqa
            _META_OK[0] = False
            fast = None
            _fallback(f"forward: routing-metadata kernel raised {type(ex).__name__}: {ex}; HybridEP's planner for this run")
    try:
        if fast is not None and not first_call:        # fast is built only when use_meta or first_call
            # ---- fast path: no HybridEP planner, no inverse map, no padded-count arithmetic
            handle, r2t, pprob, tpe32, padded = fast["handle"], fast["r2t"], fast["pprob"], fast["tpe32"], fast["padded"]
            rw, rws, cw, cws, _ = torch.ops.gpt_oss.moe_qperm_gather_quant(hidden, r2t, tpe32, P, pad_multiple, False,
                                                                            RW_LAYOUT, CW_LAYOUT, DIV_MODE, CW_PARAMS)
            _STATS["meta_fast"] += 1
            if _STATS["meta_fast"] <= MAX_LOG:
                _say(f"forward: routing metadata from the local map in 2 launches (histogram + prefix/maps): counts, padded "
                     f"counts, dense map, chunk layout, overflow, row_to_token, probs"
                     f"{', grouped-GEMM offsets' if fast['stashed_offs'] else ''}"
                     f"{', over_budget folded' if fast['has_ob'] else ''}; HybridEP planner skipped (T={T} T_slots={fast['T_s']} "
                     f"E={E_local} P={P} chunk={fast['chunk']})")
        else:
            # HybridEP's own prologue (hybrid_ep_buffer.py dispatch_with_permute): routing data, token slots, template
            # config, metadata preprocessing -> handle_impl with the dense-layout maps.  Only the data movement differs.
            topk, routing_data, probs2, _ = self._prepare_routing_data(
                topk_idx=None, topk_weights=None, num_of_tokens=T, num_of_experts=kw.get("num_of_experts"),
                probs=probs, routing_map=routing_map, num_of_experts_per_rank=E_local)
            num_of_token_slots = self._resolve_token_slots(kw.get("num_of_tokens_per_rank"), T)
            routing_data = self._pad_routing_rows(routing_data, topk, num_of_token_slots)
            config = self.update_template_config(
                hidden_dim=H, num_of_tokens_per_rank=num_of_token_slots, num_local_experts=E_local,
                pad_multiple=pad_multiple, use_fp8=kw.get("use_fp8"), fuse_permute_dispatch=False, topk=topk)
            handle_impl = self.runtime.metadata_preprocessing(
                config=config, routing_map=routing_data, num_of_tokens_per_rank=num_of_token_slots,
                num_of_valid_tokens=T, num_permuted_tokens=P, pad_multiple=pad_multiple, enable_permute=True,
                fuse_permute_dispatch=False, non_blocking=True)
            m = handle_impl.dense_to_expert_map
            layout = handle_impl.dense_chunk_layout
            tpe = handle_impl.tokens_per_expert
            if not (torch.is_tensor(m) and m.dim() == 2 and m.shape[1] == E_local and m.dtype == torch.int32
                    and torch.is_tensor(tpe) and tpe.numel() == E_local and torch.is_tensor(layout)):
                raise RuntimeError(f"unexpected handle maps: map {getattr(m, 'shape', None)} tpe {getattr(tpe, 'shape', None)}")
            tpe32 = tpe.to(torch.int32).contiguous()
            probs_f = probs2.to(torch.float32).contiguous()
            if probs_f.shape[0] < m.shape[0]:
                probs_f = torch.cat([probs_f, probs_f.new_zeros((m.shape[0] - probs_f.shape[0], probs_f.shape[1]))], 0)
            handle = _handle_tuple(handle_impl)
            r2t, pprob = _row_to_token(handle, m.contiguous(), probs_f, layout.contiguous(), P, True)
            rw, rws, cw, cws, _ = torch.ops.gpt_oss.moe_qperm_gather_quant(hidden, r2t, tpe32, P, pad_multiple, False,
                                                                            RW_LAYOUT, CW_LAYOUT, DIV_MODE, CW_PARAMS)
            padded = _padded_counts(handle_impl, tpe, pad_multiple)
            _STATS["meta_slow"] += 1
    except Exception as ex:  # noqa
        _FWD_OK[0] = False
        _fallback(f"forward fast path raised {type(ex).__name__}: {ex}; HybridEP path for this run")
        _STATS["fwd_plain"] += 1
        return orig(self, **kw)

    if first_call:
        _FWD_CHECKED[0] = True
        ok = False
        try:
            ref = orig(self, **kw)           # HybridEP's own dispatch + permute (its handle maps are equivalent)
            ref_rows, ref_probs = ref[0], ref[1]
            n_used = int(padded.to(torch.int64).sum().item())
            with torch.no_grad():
                _, _, _, _, rows_bf16 = torch.ops.gpt_oss.moe_qperm_gather_quant(hidden, r2t, tpe32, P, pad_multiple, True,
                                                                                  RW_LAYOUT, CW_LAYOUT, DIV_MODE, CW_PARAMS)
                same_rows = torch.equal(rows_bf16[:n_used], ref_rows[:n_used])
                same_probs = torch.is_tensor(ref_probs) and ref_probs.numel() == P and torch.equal(pprob[:n_used], ref_probs.reshape(-1)[:n_used])
                same_counts = torch.equal(padded.to(torch.int64).cpu(), ref[3].to(torch.int64).cpu())
                _say(f"forward self-check: gathered rows bitwise HybridEP = {same_rows}; permuted probs bitwise = {same_probs}; "
                     f"padded counts equal = {same_counts} ({ref[3].dtype} {tuple(ref[3].shape)} on {ref[3].device}); "
                     f"T={T} H={H} E={E_local} P={P} n_used={n_used}")
                same_q = _compare_quantized("forward", rw, rws, cw, cws, ref_rows.contiguous(), padded, P, H) if same_rows else False
                ok = bool(same_rows and same_probs and same_counts and same_q)
                if fast is not None:
                    _meta_selfcheck(fast, ref, probs_f, P, pad_multiple, T, E_local, ob_before, hidden)
        except Exception as ex:  # noqa
            _say(f"forward self-check raised {type(ex).__name__}: {ex}")
            ref = None
        if not ok:
            _FWD_OK[0] = False
            _say("forward self-check FAILED: HybridEP dispatch + TE quantize stay for this run")
        _STATS["fwd_plain"] += 1
        return ref if ref is not None else orig(self, **kw)

    placeholder = _make_placeholder(P, H, hidden.device)
    _park(placeholder, _Parked(rw, rws, cw, cws, P, H, E_local, placeholder, "fwd"))
    _STATS["fwd_fast"] += 1
    if _STATS["fwd_fast"] <= MAX_LOG:
        _say(f"forward: gather + MXFP8 quantize in one pass [{T}, {H}] -> [{P}, {H}] rowwise+columnwise; "
             f"no bf16 permuted copy, no dispatch copy")
    return placeholder, pprob, None, padded, handle


# ------------------------------------------------------------------------------------------------------------
# 1b. the EP=1 routing-metadata fast path (moe_routing_meta), the Megatron dispatch re-binding and the TE
#     splits_to_offsets_multi wrapper
# ------------------------------------------------------------------------------------------------------------
def _meta_ready():
    """True once no code path left in this run hands a dispatch handle to HybridEP's own runtime: this hook's forward
    self-check has run, one backward pass has been seen (the other hooks' first-call checks live there) and
    residual_bias_hook's fused combine / dispatch-backward / combine-backward paths are checked and on."""
    if not _FWD_CHECKED[0] and SELFCHECK:
        return False
    if _STATS["bwd_fast"] + _STATS["bwd_plain"] + _STATS["park_bwd"] < 1 and SELFCHECK:
        return False
    RBF = sys.modules.get("custom_kernels.residual_bias_hook")
    why = None
    if RBF is None or not getattr(RBF, "_INSTALLED", False):
        why = "residual_bias_hook is not installed (HybridEP's own combine needs its handle)"
    else:
        checked = not getattr(RBF, "SELFCHECK", True)
        flags = {
            "COMBINE": getattr(RBF, "COMBINE", False), "combine_ok": getattr(RBF, "_COMBINE_OK", [False])[0],
            "combine_checked": checked or getattr(RBF, "_COMBINE_CHECKED", [False])[0],
            "bda_reached": checked or getattr(RBF, "_BDA_REACHED", [False])[0],
            "TAP": getattr(RBF, "TAP", False), "tap_ok": getattr(RBF, "_TAP_OK", [False])[0],
            "tap_checked": checked or getattr(RBF, "_TAP_CHECKED", [False])[0],
        }
        if DGLU and _DGLU_ENGAGED[0]:
            flags["dglu_bwd_ok"] = getattr(RBF, "_DGLU_BWD_OK", [False])[0]
            flags["dglu_bwd_checked"] = checked or getattr(RBF, "_DGLU_BWD_CHECKED", [False])[0]
        else:
            flags["qbd_bwd_on"] = BWD and _BWD_OK[0]
            flags["qbd_bwd_checked"] = (not SELFCHECK) or _BWD_CHECKED[0]
        bad = [k for k, v in flags.items() if not v]
        if bad:
            why = f"paths that would hand a handle to HybridEP's runtime are still live: {bad}"
    if why is not None:
        if not _META_READY_SAID[0] and _STATS["meta_not_ready"] >= 48:
            _META_READY_SAID[0] = True
            _say(f"routing-metadata fast path not engaged after {_STATS['meta_not_ready']} dispatches: {why}")
        return False
    return True


def _dummy_u8(device):
    key = str(device)
    t = _DUMMY_U8.get(key)
    if t is None:
        t = torch.zeros((1,), dtype=torch.uint8, device=device)
        _DUMMY_U8[key] = t
    return t


def _stash_offs(padded, tensors):
    key = padded.data_ptr()
    if key in _OFFS_STASH:
        _OFFS_STASH.pop(key, None)
        try:
            _OFFS_ORDER.remove(key)
        except ValueError:
            pass
    _OFFS_STASH[key] = (padded, tensors)
    _OFFS_ORDER.append(key)
    while len(_OFFS_ORDER) > MAX_STASH:
        _OFFS_STASH.pop(_OFFS_ORDER.pop(0), None)
        _STATS["offs_evicted"] += 1


def _meta_forward(self, hidden, routing_map, probs, pad_multiple, P, E_local, kw, ob_override=None, strides_override=None):
    """The two-launch routing metadata for this dispatch; returns the fast handle and its maps (raises on anything
    outside the kernel's contract -- the caller then keeps HybridEP's planner for the run)."""
    T, H = hidden.shape
    T_s = int(self._resolve_token_slots(kw.get("num_of_tokens_per_rank"), T))
    key = (T, T_s, H, E_local, pad_multiple, P, bool(kw.get("use_fp8")))
    ent = _META_CFG.get(key)
    if ent is None:
        config = self.update_template_config(
            hidden_dim=H, num_of_tokens_per_rank=T_s, num_local_experts=E_local, pad_multiple=pad_multiple,
            use_fp8=kw.get("use_fp8"), fuse_permute_dispatch=False, topk=0)
        chunk = int(getattr(config, "num_of_tokens_per_chunk_preprocessing_api"))
        if chunk <= 0:
            raise RuntimeError(f"HybridEP config num_of_tokens_per_chunk_preprocessing_api={chunk}")
        ent = (chunk,)
        _META_CFG[key] = ent
    chunk = ent[0]
    if routing_map.dtype != torch.bool or routing_map.shape != (T, E_local):
        raise RuntimeError(f"routing map {tuple(routing_map.shape)} {routing_map.dtype}")
    rm = routing_map if routing_map.is_contiguous() else routing_map.contiguous()
    probs_f = probs if (probs.dtype == torch.float32 and probs.is_contiguous()) else probs.to(torch.float32).contiguous()
    if ob_override is not None:
        ob = ob_override
    else:
        ob = _NEXT_OB[0]
        _NEXT_OB[0] = None
    has_ob = torch.is_tensor(ob) and ob.dtype == torch.bool and ob.numel() == 1 and ob.is_cuda
    if strides_override is not None:
        strides, stash_offs = [int(s) for s in strides_override], False
    elif META_OFFS and _OFFS_OK[0] is True:
        strides, stash_offs = list(_OFFS_SPEC["strides4"]), True
    else:
        strides, stash_offs = [0, 0, 0, 0], False
    outs = torch.ops.gpt_oss.moe_routing_meta(rm, probs_f, ob.view(torch.uint8) if has_ob else _dummy_u8(hidden.device),
                                              T_s, P, pad_multiple, chunk, True, has_ob, strides)
    tpe32, padded, d2e, layout, ovf, r2t, pprob, sp, base, o1, o2, o3, o4 = outs
    meta = _FastMeta(T_s, chunk, tpe32, padded, r2t, pprob, has_ob)
    handle = (None, None, None, None, None, layout, d2e, tpe32, T_s, meta, T, ovf)
    if stash_offs:
        _stash_offs(padded, (sp, base, o1, o2, o3, o4))
    if has_ob and ob_override is None:
        _STATS["ob_folded"] += 1
    return dict(handle=handle, r2t=r2t, pprob=pprob, tpe32=tpe32, padded=padded, d2e=d2e, layout=layout, ovf=ovf,
                offs=(sp, base, o1, o2, o3, o4), strides=strides, has_ob=has_ob, ob=ob if has_ob else None, T_s=T_s,
                chunk=chunk, stashed_offs=stash_offs)


def _offsets_reference(padded_i64, strides4):
    incl = torch.cumsum(padded_i64.to(torch.int64), 0)
    base = torch.cat([torch.zeros((1,), dtype=torch.int64, device=incl.device), incl])
    return (incl.to(torch.int32), base) + tuple(base * int(s) for s in strides4)


def _meta_selfcheck(fast, ref, probs_f, P, pad_multiple, T, E, ob_before, hidden):
    """First call: every metadata-kernel output bitwise against HybridEP's own handle (ref = orig's return) and the torch
    offsets reference, plus the rows / probs gathered from `hidden` through the kernel's row_to_token bitwise HybridEP's
    own permuted rows (ref[0]) and probs (ref[1]).  A failure keeps HybridEP's planner for the run."""
    res = {}
    try:
        rh = ref[4]
        n_dense = int(rh[5].reshape(-1)[-1].item())
        res["dense_is_token"] = bool(n_dense == T)
        res["tpe"] = bool(torch.equal(fast["tpe32"], rh[7].reshape(-1).to(torch.int32)))
        res["padded"] = bool(torch.equal(fast["padded"], ref[3].reshape(-1).to(torch.int64)))
        res["layout"] = bool(fast["layout"].shape == rh[5].shape and torch.equal(fast["layout"], rh[5]))
        res["d2e"] = bool(fast["d2e"].shape == rh[6].shape and n_dense <= rh[6].shape[0]
                          and torch.equal(fast["d2e"][:n_dense], rh[6][:n_dense]))
        res["ovf"] = bool(torch.equal(fast["ovf"].reshape(-1), rh[11].reshape(-1).to(torch.int32)))
        res["T_slots"] = bool(fast["T_s"] == int(rh[8]))
        res["T"] = bool(T == int(rh[10]))
        pf = probs_f
        if pf.shape[0] < rh[6].shape[0]:
            pf = torch.cat([pf, pf.new_zeros((rh[6].shape[0] - pf.shape[0], pf.shape[1]))], 0)
        n_used = int(ref[3].reshape(-1).to(torch.int64).sum().item())
        with torch.no_grad():
            _, _, _, _, rows_mine = torch.ops.gpt_oss.moe_qperm_gather_quant(hidden, fast["r2t"], fast["tpe32"], P, pad_multiple, True,
                                                                              RW_LAYOUT, CW_LAYOUT, DIV_MODE, CW_PARAMS)
        res["rows_vs_hybridep"] = bool(torch.is_tensor(ref[0]) and ref[0].shape[0] >= n_used
                                       and torch.equal(rows_mine[:n_used], ref[0][:n_used]))
        res["pprob_vs_hybridep"] = bool(torch.is_tensor(ref[1]) and ref[1].numel() == P
                                        and torch.equal(fast["pprob"][:n_used], ref[1].reshape(-1)[:n_used]))
        if n_dense == T:
            # row_to_token is in token space (row p holds token t, what the gather reads from the token-ordered rows);
            # the inverse-map op on HybridEP's dense map yields dense indices d(t).  They coincide iff every token is
            # routed somewhere (n_dense == T), which top-k routing guarantees and the fast path requires (dense_is_token).
            r2t_ref, pp_ref = torch.ops.gpt_oss.moe_qperm_inverse_map(rh[6].contiguous(), pf.contiguous(), rh[5].contiguous(), P, True)
            res["r2t_vs_inverse_map"] = bool(torch.equal(fast["r2t"], r2t_ref))
            res["pprob_vs_inverse_map"] = bool(torch.equal(fast["pprob"], pp_ref))
        offs_ref = _offsets_reference(ref[3].reshape(-1), fast["strides"][0:4])
        res["offsets"] = all(bool(a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)) for a, b in zip(fast["offs"], offs_ref))
        if fast["has_ob"] and ob_before is not None:
            want = bool(ob_before.reshape(-1)[0].item()) or bool(rh[11].reshape(-1)[0].item() != 0)
            res["over_budget"] = bool(fast["ob"].reshape(-1)[0].item()) == want
        ok = all(res.values())
        _say(f"routing-metadata self-check vs HybridEP's handle: {'OK' if ok else 'FAIL'} {res} n_dense={n_dense} T={T} n_used={n_used} "
             f"T_slots={fast['T_s']} chunk={fast['chunk']} layout_entries={fast['layout'].numel()} "
             f"overflow={int(fast['ovf'].reshape(-1)[0].item())} tpe[:4]={fast['tpe32'][:4].tolist()} padded[:4]={fast['padded'][:4].tolist()}")
    except Exception as ex:  # noqa
        ok = False
        _say(f"routing-metadata self-check raised {type(ex).__name__}: {ex} (partial {res})")
    if not ok:
        _META_OK[0] = False
        _say("routing-metadata self-check FAILED: HybridEP's planner, the inverse map and _padded_counts stay for this run")
    return ok


def _offs_call_shape(args, kwargs):
    names = ("device", "strides", "include_leading_zero", "dtypes")
    vals = {}
    for i, a in enumerate(args):
        if i >= len(names):
            return None
        vals[names[i]] = a
    for k, v in kwargs.items():
        if k not in names or k in vals:
            return None
        vals[k] = v
    if any(k not in vals for k in names):
        return None
    try:
        return dict(device=vals["device"], strides=[int(s) for s in vals["strides"]], lead=[bool(b) for b in vals["include_leading_zero"]],
                    dtypes=list(vals["dtypes"]))
    except Exception:  # noqa
        return None


def _learn_offs_spec(split_sizes, args, kwargs, ret):
    """TE's first splits_to_offsets_multi call: record the call shape and check, bitwise, that the torch offsets
    reference (= what the metadata kernel emits) equals TE's result; only then are kernel offsets handed back."""
    try:
        shape = _offs_call_shape(args, kwargs)
        okshape = (shape is not None and len(shape["strides"]) == 6 and shape["strides"][0] == 1 and shape["strides"][1] == 1
                   and shape["lead"] == [False, True, True, True, True, True]
                   and shape["dtypes"] == [torch.int32, torch.int64, torch.int64, torch.int64, torch.int64, torch.int64]
                   and torch.is_tensor(split_sizes) and split_sizes.dim() == 1 and split_sizes.dtype == torch.int64 and split_sizes.is_cuda
                   and isinstance(ret, (tuple, list)) and len(ret) == 2 and isinstance(ret[1], (tuple, list)) and len(ret[1]) == 6)
        if not okshape:
            _OFFS_OK[0] = False
            _say(f"grouped-GEMM offsets stay with TE: splits_to_offsets_multi call shape {shape} / return {type(ret).__name__} not the expected one")
            return
        mine = _offsets_reference(split_sizes, shape["strides"][2:6])
        same = [bool(torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape and a.device == b.device and torch.equal(a, b))
                for a, b in zip(mine, ret[1])]
        first_is_input = ret[0] is split_sizes
        first_equal = first_is_input or (torch.is_tensor(ret[0]) and ret[0].dtype == split_sizes.dtype and ret[0].shape == split_sizes.shape
                                         and torch.equal(ret[0], split_sizes))
        ok = all(same) and first_equal
        _OFFS_OK[0] = ok
        _OFFS_SPEC.clear()
        _OFFS_SPEC.update(strides=shape["strides"], strides4=shape["strides"][2:6], lead=shape["lead"], dtypes=shape["dtypes"],
                          ret_is_list=isinstance(ret[1], list), outer_is_list=isinstance(ret, list), first_is_input=first_is_input)
        _say(f"splits_to_offsets_multi first call: torch offsets reference bitwise TE = {same}, split_sizes returned "
             f"{'as is' if first_is_input else ('equal copy' if first_equal else 'DIFFERENT')}; strides={shape['strides']} -> "
             f"{'kernel offsets handed back from the next fast dispatch on' if ok else 'TE keeps computing the offsets'}")
    except Exception as ex:  # noqa
        _OFFS_OK[0] = False
        _say(f"grouped-GEMM offsets stay with TE: learning the call shape raised {type(ex).__name__}: {ex}")


def _make_splits_to_offsets(orig):
    def splits_to_offsets_multi(split_sizes, *args, **kwargs):
        if _OFFS_STASH and torch.is_tensor(split_sizes):
            ent = _OFFS_STASH.get(split_sizes.data_ptr())
            if ent is not None:
                padded, tensors = ent
                shape = _offs_call_shape(args, kwargs)
                same_call = (shape is not None and shape["strides"] == _OFFS_SPEC.get("strides")
                             and shape["lead"] == _OFFS_SPEC.get("lead") and shape["dtypes"] == _OFFS_SPEC.get("dtypes"))
                same_tensor = padded is split_sizes or (padded.shape == split_sizes.shape and padded.dtype == split_sizes.dtype)
                if same_call and same_tensor:
                    _OFFS_STASH.pop(split_sizes.data_ptr(), None)
                    try:
                        _OFFS_ORDER.remove(split_sizes.data_ptr())
                    except ValueError:
                        pass
                    _STATS["offs_hit"] += 1
                    if _STATS["offs_hit"] <= MAX_LOG:
                        _say("grouped MLP: the six grouped-GEMM offset tensors come from the routing-metadata kernel (splits_to_offsets skipped)")
                    seq = list(tensors) if _OFFS_SPEC.get("ret_is_list") else tuple(tensors)
                    return [split_sizes, seq] if _OFFS_SPEC.get("outer_is_list") else (split_sizes, seq)
                _STATS["offs_miss"] += 1
                if _STATS["offs_miss"] <= MAX_LOG:
                    _say(f"grouped MLP: splits_to_offsets_multi called with a different shape than recorded ({shape}); TE computes the offsets")
        ret = orig(split_sizes, *args, **kwargs)
        if META_OFFS and _OFFS_OK[0] is None and torch.is_tensor(split_sizes) and not torch.cuda.is_current_stream_capturing():
            _learn_offs_spec(split_sizes, args, kwargs, ret)
        return ret
    return splits_to_offsets_multi


def _make_manager_dispatch(TD):
    """A copy of Megatron's _HybridEPManager.dispatch (source-hash checked at install) whose two mask launches
    (`over_budget = handle[-1] != 0`, `self.over_budget |= over_budget`) are skipped for a fast handle -- the metadata
    kernel OR-ed the overflow flag into self.over_budget itself (handed to it through _NEXT_OB)."""
    def dispatch(self, hidden_states, async_finish=True, allocate_on_comm_stream=True):
        # HybridEP only supports float32 probs
        if self.token_probs.dtype != torch.float32:
            if self.token_probs.dtype in [torch.bfloat16, torch.float16]:
                TD.logger.warning(
                    "HybridEP only supports float32 probs, please set --moe-router-dtype=fp32"
                )
            self.token_probs = self.token_probs.float()  # downcast or upcast
        align_size = TD.get_align_size_for_quantization(self.config)
        if align_size > 0:
            self.pad_multiple = align_size
        if self._padded_num_tokens is not None and hidden_states.shape[0] < self._padded_num_tokens:
            pad_rows = self._padded_num_tokens - hidden_states.shape[0]
            hidden_states = torch.cat(
                [hidden_states, hidden_states.new_zeros((pad_rows, hidden_states.shape[-1]))], dim=0
            )
        cap = self.moe_expert_rank_capacity_factor
        ob = getattr(self, "over_budget", None)
        _NEXT_OB[0] = ob if (cap is not None and torch.is_tensor(ob)) else None
        try:
            dispatched_hidden, self.dispatched_probs, _, tokens_per_expert, self.handle = (
                TD.hybrid_ep_dispatch(
                    x=hidden_states,
                    routing_map=self.routing_map,
                    probs=self.token_probs,
                    group=self.group,
                    num_local_experts=self.num_local_experts,
                    num_sms_dispatch_api=self.config.moe_flex_dispatcher_num_sms,
                    num_sms_combine_api=self.config.moe_flex_dispatcher_num_sms,
                    num_blocks_permute=self.config.moe_hybridep_num_blocks_permute,
                    num_blocks_unpermute=self.config.moe_hybridep_num_blocks_unpermute,
                    num_permuted_tokens=self.num_permuted_tokens,
                    pad_multiple=self.pad_multiple,
                    fused=self.config.moe_permute_fusion_into_hybridep,
                    num_sms_preprocessing_api=self.config.moe_hybridep_num_sms_preprocessing,
                )
            )
        finally:
            _NEXT_OB[0] = None
        if cap is not None:
            meta = _fast_meta(self.handle)
            if meta is None or not meta.over_budget_folded:
                # Static-budget path only: handle[-1] is HybridEP overflow_flag when tokens were
                # dropped because permuted count exceeded num_permuted_tokens from setup_metadata.
                over_budget = self.handle[-1] != 0
                self.over_budget |= over_budget
                _STATS["dispatch_mask_ops"] += 1
        # When capacity factor is None, skip overflow tracking (no token drops). Actual
        # permuted size is resolved below via tokens_per_expert.sum() (CPU sync).

        if self.num_permuted_tokens is None:
            self.tokens_per_expert = tokens_per_expert.to(torch.int64)
            # num_permuted_tokens is necessary to allocate the output tensor for combine.
            self.num_permuted_tokens = self.tokens_per_expert.sum()
        if cap is not None:
            self.tokens_per_expert = tokens_per_expert.to(torch.int64)
        return dispatched_hidden
    return dispatch


def _install_meta_patches(te_grouped_mlp):
    """The routing-metadata path's two optional library re-bindings (a failure only keeps the launches they remove)."""
    parts = []
    if not META:
        _say("routing-metadata fast path disabled by MOE_ROUTING_META=0 (control: HybridEP planner, inverse map, _padded_counts)")
        _META_OK[0] = False
        return parts
    try:
        import custom_kernels.moe_routing_meta  # noqa: F401  (registers gpt_oss::moe_routing_meta)
    except Exception as ex:  # noqa
        _META_OK[0] = False
        _say(f"routing-metadata kernel import failed {type(ex).__name__}: {ex}; HybridEP's planner stays")
        return parts
    if META_OFFS:
        try:
            # grouped_mlp's own `tex` (fuser_forward / fuser_backward call tex.splits_to_offsets_multi through this
            # module-level name): the extension module is reached through the object that uses it, never by a bare
            # `import transformer_engine_torch` (not importable on its own before transformer_engine loads it)
            tex = getattr(te_grouped_mlp, "tex", None)
            if tex is None or getattr(tex, "__name__", "") != "transformer_engine_torch":
                raise RuntimeError(f"grouped_mlp.tex is {getattr(tex, '__name__', tex)!r}, not the transformer_engine_torch module")
            if sys.modules.get("transformer_engine_torch") is not tex:
                raise RuntimeError("grouped_mlp.tex is not sys.modules['transformer_engine_torch']")
            fn = getattr(tex, "splits_to_offsets_multi", None)
            if fn is None:
                raise RuntimeError("transformer_engine_torch has no splits_to_offsets_multi")
            _ORIG["splits_to_offsets_multi"] = fn
            tex.splits_to_offsets_multi = _make_splits_to_offsets(fn)
            parts.append("transformer_engine_torch.splits_to_offsets_multi (kernel offsets handed back)")
        except Exception as ex:  # noqa
            _OFFS_OK[0] = False
            _say(f"grouped-GEMM offsets stay with TE: {type(ex).__name__}: {ex}")
    else:
        _OFFS_OK[0] = False
    if META_DISPATCH:
        try:
            from megatron.core.transformer.moe import token_dispatcher as TD
            RBF = sys.modules.get("custom_kernels.residual_bias_hook")
            cur = TD._HybridEPManager.dispatch
            mega = None
            if RBF is not None and getattr(RBF, "_INSTALLED", False):
                mega = RBF._ORIG.get("_HybridEPManager.dispatch")
            wrapped_by_rbf = mega is not None and cur is not mega
            if mega is None:
                mega = cur
            src = inspect.getsource(mega)
            h = hashlib.md5(src.encode()).hexdigest()
            if h != _MEGATRON_DISPATCH_MD5:
                raise RuntimeError(f"_HybridEPManager.dispatch source md5 {h} is not the expected {_MEGATRON_DISPATCH_MD5}")
            for name in ("get_align_size_for_quantization", "hybrid_ep_dispatch", "logger"):
                if not hasattr(TD, name):
                    raise RuntimeError(f"token_dispatcher has no {name}")
            new = _make_manager_dispatch(TD)
            if wrapped_by_rbf:
                if not hasattr(RBF, "_wrap_manager_dispatch"):
                    raise RuntimeError("residual_bias_hook wrapped _HybridEPManager.dispatch but has no _wrap_manager_dispatch")
                TD._HybridEPManager.dispatch = RBF._wrap_manager_dispatch(new)
            else:
                TD._HybridEPManager.dispatch = new
            _ORIG["_HybridEPManager.dispatch(megatron)"] = mega
            _DISPATCH_PATCHED[0] = True
            parts.append(f"_HybridEPManager.dispatch (over_budget mask ops folded into the kernel{'; residual hook wrapper rebuilt' if wrapped_by_rbf else ''})")
        except Exception as ex:  # noqa
            _say(f"Megatron's over_budget mask ops stay: {type(ex).__name__}: {ex}")
    return parts


def _backward(self, orig, kw):
    hidden = kw.get("hidden")
    handle = kw.get("handle")
    pad_multiple = kw.get("pad_multiple")
    P = kw.get("num_permuted_tokens")
    why = None
    if DGLU and _DGLU_ENGAGED[0]:
        # the fused combine Function parks the quantized gradient itself; a dispatch_with_permute reaching here is the
        # fallback combine's autograd, which needs HybridEP's real bf16 rows
        why = "MOE_DGLU_STRIP: the combine backward quantizes in residual_bias_hook"
    elif not (BWD and _BWD_OK[0]):
        why = "backward fast path off"
    else:
        why = _fast_path_common(self, hidden, handle, pad_multiple, P, kw.get("fuse_permute_dispatch", False),
                                kw.get("use_fp8"), kw.get("scaling_factor"))
    m = layout = tpe = None
    if why is None:
        if not (isinstance(handle, tuple) and len(handle) == len(_HANDLE_FIELDS)):
            why = f"handle with {len(handle) if isinstance(handle, tuple) else '?'} entries"
        else:
            layout, m, tpe = handle[5], handle[6], handle[7]
            if not (torch.is_tensor(m) and m.dim() == 2 and m.dtype == torch.int32 and torch.is_tensor(tpe) and torch.is_tensor(layout)):
                why = "handle maps"
            elif not (tpe.numel() & (tpe.numel() - 1) == 0):
                why = f"{tpe.numel()} local experts (power of two needed)"
            elif kw.get("probs") is not None:
                why = "probs in the backward permute"
            elif m.shape[0] < hidden.shape[0]:
                why = f"map rows {m.shape[0]} < hidden rows {hidden.shape[0]}"
    if why is not None:
        _STATS["bwd_plain"] += 1
        if _STATS["bwd_plain"] <= MAX_LOG:
            _fallback(f"backward: {why}; HybridEP dispatch_with_permute")
        return orig(self, **kw)
    T, H = hidden.shape
    E = tpe.numel()
    try:
        tpe32 = _tpe32_for(handle, tpe)
        r2t, _ = _row_to_token(handle, m.contiguous(), hidden, layout.contiguous(), P, False)
        rw, rws, cw, cws, rows = torch.ops.gpt_oss.moe_qperm_gather_quant(hidden, r2t, tpe32, P, pad_multiple, True,
                                                                           RW_LAYOUT, CW_LAYOUT, DIV_MODE, CW_PARAMS)
        padded = _padded_counts_for(handle, tpe, pad_multiple)
    except Exception as ex:  # noqa
        _BWD_OK[0] = False
        _fallback(f"backward fast path raised {type(ex).__name__}: {ex}; HybridEP path for this run")
        _STATS["bwd_plain"] += 1
        return orig(self, **kw)

    if SELFCHECK and not _BWD_CHECKED[0]:
        _BWD_CHECKED[0] = True
        ok = False
        try:
            ref = orig(self, **kw)
            ref_rows = ref[0]
            n_used = int(padded.sum().item())
            with torch.no_grad():
                same_rows = torch.equal(rows[:n_used], ref_rows[:n_used])
                _say(f"backward self-check: gathered gradient rows bitwise HybridEP = {same_rows}; T={T} H={H} E={E} P={P} n_used={n_used}")
                same_q = _compare_quantized("backward", rw, rws, cw, cws, ref_rows.contiguous(), padded, P, H) if same_rows else False
                ok = bool(same_rows and same_q)
        except Exception as ex:  # noqa
            _say(f"backward self-check raised {type(ex).__name__}: {ex}")
            ref = None
        if not ok:
            _BWD_OK[0] = False
            _say("backward self-check FAILED: HybridEP permute + TE quantize stay for this run")
        _STATS["bwd_plain"] += 1
        return ref if ref is not None else orig(self, **kw)

    _park(rows, _Parked(rw, rws, cw, cws, P, H, E, rows, "bwd"))
    _STATS["bwd_fast"] += 1
    if _STATS["bwd_fast"] <= MAX_LOG:
        _say(f"backward: gradient gather + MXFP8 quantize in one pass [{T}, {H}] -> [{P}, {H}] bf16 + rowwise + columnwise")
    return rows, None, None, padded, handle


# ------------------------------------------------------------------------------------------------------------
# install
# ------------------------------------------------------------------------------------------------------------
def install() -> bool:
    global _INSTALLED
    if _INSTALLED:
        return True
    if not ENABLED:
        _say("disabled by MOE_QPERM=0 (control path: HybridEP dispatch + permute, TE group_quantize)")
        return False
    try:
        import custom_kernels.moe_qperm  # noqa: F401  (registers gpt_oss::moe_qperm_* ops)
        from deep_ep.hybrid_ep_buffer import HybridEPBuffer
        from megatron.core.transformer.moe import fused_a2a
        from transformer_engine.pytorch.ops.fused import grouped_mlp as te_grouped_mlp
    except Exception as ex:  # noqa
        _say(f"not installed: import failed {type(ex).__name__}: {ex}")
        return False
    if getattr(fused_a2a, "HybridEPBuffer", HybridEPBuffer) is not HybridEPBuffer:
        _say("not installed: megatron's fused_a2a.HybridEPBuffer is a different class")
        return False
    missing = []
    sig = inspect.signature(HybridEPBuffer.dispatch_with_permute)
    for p in ("hidden", "routing_map", "probs", "handle", "num_permuted_tokens", "pad_multiple", "non_blocking",
              "fuse_permute_dispatch", "num_of_experts_per_rank", "scaling_factor", "num_of_tokens_per_rank"):
        if p not in sig.parameters:
            missing.append(f"dispatch_with_permute.{p}")
    for name in ("_prepare_routing_data", "_resolve_token_slots", "_pad_routing_rows", "update_template_config"):
        if not hasattr(HybridEPBuffer, name):
            missing.append(f"HybridEPBuffer.{name}")
    gq = getattr(te_grouped_mlp, "_group_quantize_for_grouped_mlp", None)
    if gq is None:
        missing.append("grouped_mlp._group_quantize_for_grouped_mlp")
    else:
        gsig = inspect.signature(gq)
        for p in ("tensor", "quantizer", "num_groups", "split_sizes", "tensor_offsets", "use_dense_single_group"):
            if p not in gsig.parameters:
                missing.append(f"_group_quantize_for_grouped_mlp.{p}")
    if missing:
        _say(f"not installed: missing {missing}")
        return False
    _ORIG["dispatch_with_permute"] = HybridEPBuffer.dispatch_with_permute
    HybridEPBuffer.dispatch_with_permute = _make_dispatch_with_permute(_ORIG["dispatch_with_permute"])
    _ORIG["_group_quantize_for_grouped_mlp"] = gq
    te_grouped_mlp._group_quantize_for_grouped_mlp = _make_group_quantize(gq)
    meta_parts = _install_meta_patches(te_grouped_mlp)
    _INSTALLED = True
    _say(f"installed: HybridEPBuffer.dispatch_with_permute (fwd={'on' if FWD else 'off'}, bwd={'on' if BWD else 'off'}"
         f"{' -> HybridEP path, the combine Function parks (MOE_DGLU_STRIP=1)' if DGLU else ''}), "
         f"grouped_mlp._group_quantize_for_grouped_mlp; layouts rw={RW_LAYOUT} cw={CW_LAYOUT} div={DIV_MODE} cw_params={CW_PARAMS} selfcheck={SELFCHECK}; "
         f"routing-metadata fast path (MOE_ROUTING_META)={'on' if META and _META_OK[0] else 'off'}"
         f"{(': ' + '; '.join(meta_parts)) if meta_parts else ''}")
    atexit.register(lambda: _say(f"stats: {stats()}"))
    return True


def stats():
    return dict(_STATS, meta_ok=_META_OK[0], offs_ok=_OFFS_OK[0], dispatch_patched=_DISPATCH_PATCHED[0],
                offs_pending=len(_OFFS_STASH), fwd_ok=_FWD_OK[0], bwd_ok=_BWD_OK[0])
