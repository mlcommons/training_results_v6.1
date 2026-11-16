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

"""Pad the output buffers of TE's uniform-shape grouped MXFP8 quantize so its scale-factor stores stay in bounds.

With MOE_SINGLE_GROUPED_WEIGHT_BIAS=True + NVTE_GROUPED_LINEAR_SINGLE_PARAM=1 each GroupedLinear weight is one stacked
(num_groups * out_features, in_features) parameter, quantized by a single transformer_engine_torch.group_quantize call
(split_sizes=None: SAME_BOTH_DIMS).  TE's kernel (2.21.0.dev0+9c33764b, group_quantize_mxfp8.cuh) gives every group
DIVUP(rows_per_group, 128) chunks of 128 rows and never clips the last one: for fc2 (2880 = 22 * 128 + 64 rows per
group) the last group writes 64 E8M0 scale rows (5888 bytes) past TE's buffers (the TMA tensor map clips the data
stores, nothing clips the scale stores), corrupting what follows or faulting.  fc1 (5760 = 45 * 128) is unaffected.
Workaround (TE's kernel should clip): for a uniform-shape call (no split sizes, offsets or output: only weight
quantizes) with rows-per-group % 128 != 0, pre-build the output GroupedTensor (as MXFP8Quantizer.create_grouped_tensor
would) with data and scale buffers 128 rows longer (~0.4 MB per call) and pass it through TE's `output=` reuse path;
the logical contents are bitwise those of a clipped kernel.  Aligned and grouped-operand calls take the stock path.
SINGLE_GROUPED_WEIGHT=0 leaves TE untouched (the reference path); SINGLE_GROUPED_WEIGHT_SELFCHECK=1 (default) checks
the first aligned call bitwise through both paths and disables the patch on a mismatch.
"""
import os
import sys

import torch

ENABLED = os.getenv("SINGLE_GROUPED_WEIGHT", "1") == "1"
SELFCHECK = os.getenv("SINGLE_GROUPED_WEIGHT_SELFCHECK", "1") == "1"
PAD_ROWS = 128          # one full chunk of the kernel's direct mapper
TILE_ROWS = 128         # CastTraits<SAME_BOTH_DIMS>::CHUNK_DIM_Y == TILE_DIM_Y
MXFP8_BLOCK = 32
_OUT = sys.__stdout__
_ORIG = {}
_INSTALLED = False
_STATE = {"padded": 0, "plain": 0, "disabled": None, "checked": False, "shapes": {}}
MAX_LOG = 3


def _rank():
    return os.getenv("RANK", os.getenv("SLURM_PROCID", "0"))


def _log(msg, all_ranks=False):
    if all_ranks or _rank() == "0":
        _OUT.write(f"SINGLE_GROUPED_WEIGHT rank={_rank()} {msg}\n")
        _OUT.flush()


def _is_uniform_weight_call(tensor, quantizer, num_tensors, first_dims, last_dims, tensor_offsets, noop_flag, output):
    if first_dims is not None or last_dims is not None or tensor_offsets is not None:
        return False
    if noop_flag is not None or output is not None:
        return False
    if not (torch.is_tensor(tensor) and tensor.dim() == 2 and tensor.is_contiguous()):
        return False
    if num_tensors <= 0 or tensor.shape[0] % num_tensors != 0:
        return False
    try:
        from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
    except Exception:  # noqa: BLE001
        return False
    if not isinstance(quantizer, MXFP8Quantizer) or getattr(quantizer, "with_2d_quantization", False):
        return False
    if not hasattr(quantizer, "get_scale_shape"):
        return False
    return True


def _make_padded_output(tensor, quantizer, num_tensors):
    """The GroupedTensor TE's MXFP8Quantizer.create_grouped_tensor builds for a uniform-shape call, with the
    data and scale buffers extended past their logical extent by one 128-row chunk."""
    from transformer_engine.pytorch.tensor.grouped_tensor import GroupedTensor
    from transformer_engine.pytorch.tensor.storage.grouped_tensor_storage import GroupedTensorStorage

    rows, cols = int(tensor.shape[0]), int(tensor.shape[1])
    dev = tensor.device
    u8 = {"dtype": torch.uint8, "device": dev}
    data = cw = rws = cws = None
    numel = rows * cols
    pad_elems = PAD_ROWS * cols
    if quantizer.rowwise_usage:
        rs = quantizer.get_scale_shape((rows, cols), False)      # (rows padded to 128, k-blocks padded to 4)
        n_rws = int(rs[0]) * int(rs[1])
        data = torch.empty(numel + pad_elems, **u8)[:numel]
        rws = torch.empty(n_rws + PAD_ROWS * int(rs[1]), **u8)[:n_rws]
    if quantizer.columnwise_usage:
        cs = quantizer.get_scale_shape((rows, cols), True)       # (row-blocks padded to 4, cols padded to 128)
        n_cws = int(cs[0]) * int(cs[1])
        cw = torch.empty(numel + pad_elems, **u8)[:numel]
        cws = torch.empty(n_cws + (PAD_ROWS // MXFP8_BLOCK + 4) * int(cs[1]), **u8)[:n_cws]
    cls = GroupedTensorStorage if getattr(quantizer, "internal", False) else GroupedTensor
    return cls(
        shape=(rows, cols),
        dtype=tensor.dtype,
        num_tensors=num_tensors,
        quantizer=quantizer,
        data=data,
        columnwise_data=cw,
        scale_inv=rws,
        columnwise_scale_inv=cws,
        with_gemm_swizzled_scales=bool(getattr(quantizer, "optimize_for_gemm", False)),
    )


def _same(a, b):
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return a.numel() == b.numel() and bool(torch.equal(a.reshape(-1), b.reshape(-1)))


def _selfcheck(orig, tensor, quantizer, num_tensors):
    """fc1-shaped call (rows per group 128-aligned: the stock kernel has no over-run there): stock vs padded output."""
    ref = orig(tensor, quantizer, num_tensors, None)
    out = _make_padded_output(tensor, quantizer, num_tensors)
    got = orig(tensor, quantizer, num_tensors, None, None, None, None, out)
    res = {
        "same_object": got is out,
        "rw": _same(getattr(ref, "rowwise_data", None), getattr(got, "rowwise_data", None)),
        "rws": _same(getattr(ref, "scale_inv", None), getattr(got, "scale_inv", None)),
        "cw": _same(getattr(ref, "columnwise_data", None), getattr(got, "columnwise_data", None)),
        "cws": _same(getattr(ref, "columnwise_scale_inv", None), getattr(got, "columnwise_scale_inv", None)),
        "swz": getattr(ref, "_with_gemm_swizzled_scales", None) == getattr(got, "_with_gemm_swizzled_scales", None),
        "logical": tuple(getattr(ref, "logical_shape", ())) == tuple(getattr(got, "logical_shape", ())),
    }
    ok = all(res.values())
    _log(f"self-check on a {tuple(tensor.shape)} x{num_tensors} uniform quantize (rows/group {tensor.shape[0] // num_tensors}, "
         f"rowwise={quantizer.rowwise_usage} columnwise={quantizer.columnwise_usage}): padded `output=` path bitwise stock = {ok} {res}",
         all_ranks=not ok)
    return ok, ref


def _make_group_quantize(orig):
    def group_quantize(tensor, quantizer, num_tensors, first_dims=None, last_dims=None, tensor_offsets=None,
                       noop_flag=None, output=None):
        st = _STATE
        if st["disabled"] is not None or not _is_uniform_weight_call(tensor, quantizer, num_tensors, first_dims, last_dims,
                                                                     tensor_offsets, noop_flag, output):
            return orig(tensor, quantizer, num_tensors, first_dims, last_dims, tensor_offsets, noop_flag, output)
        rows_per_group = int(tensor.shape[0]) // int(num_tensors)
        if rows_per_group % TILE_ROWS == 0:
            if SELFCHECK and not st["checked"] and not torch.cuda.is_current_stream_capturing():
                st["checked"] = True
                try:
                    ok, ref = _selfcheck(orig, tensor, quantizer, num_tensors)
                except Exception as ex:  # noqa: BLE001
                    st["disabled"] = f"self-check raised {type(ex).__name__}: {ex}"
                    _log(f"DISABLED ({st['disabled']}); TE's stock allocation stays", all_ranks=True)
                    return orig(tensor, quantizer, num_tensors, first_dims, last_dims, tensor_offsets, noop_flag, output)
                if not ok:
                    st["disabled"] = "self-check mismatch"
                    _log("DISABLED (self-check mismatch); TE's stock allocation stays", all_ranks=True)
                return ref
            st["plain"] += 1
            return orig(tensor, quantizer, num_tensors, first_dims, last_dims, tensor_offsets, noop_flag, output)
        # rows per group not 128-aligned: the kernel over-runs the last group's scale rows -> padded output
        try:
            out = _make_padded_output(tensor, quantizer, num_tensors)
        except Exception as ex:  # noqa: BLE001
            st["disabled"] = f"padded output construction raised {type(ex).__name__}: {ex}"
            _log(f"DISABLED ({st['disabled']}); TE's stock allocation stays", all_ranks=True)
            return orig(tensor, quantizer, num_tensors, first_dims, last_dims, tensor_offsets, noop_flag, output)
        st["padded"] += 1
        key = (tuple(tensor.shape), int(num_tensors), bool(quantizer.rowwise_usage), bool(quantizer.columnwise_usage))
        st["shapes"][key] = st["shapes"].get(key, 0) + 1
        if st["padded"] <= MAX_LOG:
            _log(f"uniform grouped quantize of {tuple(tensor.shape)} x{num_tensors} ({rows_per_group} rows/group = "
                 f"{rows_per_group // TILE_ROWS}*128 + {rows_per_group % TILE_ROWS}; rowwise={quantizer.rowwise_usage} "
                 f"columnwise={quantizer.columnwise_usage}): output buffers padded by {PAD_ROWS} rows, TE `output=` path")
        return orig(tensor, quantizer, num_tensors, None, None, None, None, out)

    return group_quantize


def _report():
    st = _STATE
    _log(f"stats: padded={st['padded']} plain={st['plain']} disabled={st['disabled']} shapes={ {str(k): v for k, v in st['shapes'].items()} }",
         all_ranks=True)


def install():
    global _INSTALLED
    if _INSTALLED:
        return True
    if not ENABLED:
        _log("disabled by SINGLE_GROUPED_WEIGHT=0 (control path: TE's stock grouped quantize allocation)")
        return False
    try:
        import transformer_engine.pytorch  # noqa: F401  (loads the extension)
        import transformer_engine_torch as tex
    except Exception as ex:  # noqa: BLE001
        _log(f"not installed: import failed {type(ex).__name__}: {ex}")
        return False
    gq = getattr(tex, "group_quantize", None)
    if gq is None:
        _log("not installed: transformer_engine_torch.group_quantize missing")
        return False
    _ORIG["group_quantize"] = gq
    tex.group_quantize = _make_group_quantize(gq)
    _INSTALLED = True
    import atexit

    atexit.register(_report)
    _log(f"installed: transformer_engine_torch.group_quantize (uniform-shape weight quantizes with rows/group % {TILE_ROWS} != 0 "
         f"get {PAD_ROWS}-row padded output buffers; selfcheck={SELFCHECK})")
    return True
