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

import os
from typing import Optional

import torch

from . import arch as _arch

_ENABLED = os.environ.get("OPT_QUANT_BOOKKEEPING", "1") == "1"
_DIR = os.path.dirname(os.path.abspath(__file__))
_ext = None


def _load():
    global _ext
    if _ext is None:
        import transformer_engine
        from torch.utils.cpp_extension import load

        te_dir = os.path.dirname(os.path.abspath(transformer_engine.__file__))
        prev = os.environ.get("TORCH_CUDA_ARCH_LIST")
        os.environ["TORCH_CUDA_ARCH_LIST"] = _arch.cuda_arch_list()
        try:
            _ext = load(
                name=_arch.ext_name("opt_quant_bookkeeping"),
                sources=[
                    os.path.join(_DIR, "quant_bookkeeping.cu"),
                    os.path.join(_DIR, "quant_bookkeeping_binding.cpp"),
                ],
                extra_include_paths=[_DIR, os.path.join(te_dir, "common", "include")],
                extra_cflags=["-O3"],
                extra_cuda_cflags=["-O3", "-std=c++17", "-lineinfo"],
                extra_ldflags=[f"-L{te_dir}", "-ltransformer_engine", f"-Wl,-rpath,{te_dir}",
                               "-lcuda"],
                build_directory=os.environ.get("OPT_EXT_BUILD_DIR") or None,
                verbose=False,
            )
        finally:
            if prev is None:
                os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = prev
    return _ext


@torch.library.custom_op(
    "optimized::qb_rht_quantize",
    mutates_args=("row_data", "row_scale", "row_amax", "col_data", "col_scale", "col_amax",
                  "rng_state", "ws"),
)
def qb_rht_quantize(
    x: torch.Tensor,
    rht: torch.Tensor,
    sign_mask_t: int,
    row_data: Optional[torch.Tensor],
    row_scale: Optional[torch.Tensor],
    row_amax: Optional[torch.Tensor],
    col_data: Optional[torch.Tensor],
    col_scale: Optional[torch.Tensor],
    col_amax: Optional[torch.Tensor],
    swizzled: bool,
    stochastic: bool,
    rng_state: Optional[torch.Tensor],
    ws: torch.Tensor,
) -> None:
    _load().rht_quantize(x, rht, sign_mask_t, row_data, row_scale, row_amax, col_data, col_scale,
                         col_amax, swizzled, stochastic, rng_state, ws)


@torch.library.custom_op(
    "optimized::qb_quantize_2d",
    mutates_args=("row_data", "row_scale", "row_amax", "col_data", "col_scale", "col_amax", "ws"),
)
def qb_quantize_2d(
    x: torch.Tensor,
    row_data: Optional[torch.Tensor],
    row_scale: Optional[torch.Tensor],
    row_amax: Optional[torch.Tensor],
    col_data: Optional[torch.Tensor],
    col_scale: Optional[torch.Tensor],
    col_amax: Optional[torch.Tensor],
    swizzled: bool,
    noop: Optional[torch.Tensor],
    ws: torch.Tensor,
) -> None:
    _load().quantize_2d(x, row_data, row_scale, row_amax, col_data, col_scale, col_amax, swizzled,
                        noop, ws)


_ws = {}


def _ws_state(device):
    stream = torch.cuda.current_stream(device)
    key = (device.index, stream.cuda_stream)
    st = _ws.get(key)
    if st is None:
        st = torch.zeros(4, dtype=torch.int32, device=device)
        _ws[key] = st
    return st


_orig = {}
_counts = {"rht": 0, "rht_sr": 0, "2d": 0, "fallback": 0}
_announced = False
_FAST_MATH = os.environ.get("NVTE_USE_FAST_MATH", "0") not in ("", "0")


def _announce():
    global _announced
    if not _announced:
        _announced = True
        if int(os.environ.get("RANK", "0")) == 0:
            print("[opt quant_bookkeeping] self-initialising NVFP4 amax / in-kernel Philox "
                  "state / BasicLinear weight pre-swizzle active", flush=True)


def _sm100_family() -> bool:
    major, minor = torch.cuda.get_device_capability()
    return 100 <= major * 10 + minor <= 110


def _common_ok(q) -> bool:
    import transformer_engine_torch as tex

    return (
        not q.with_amax_reduction
        and not q.row_scaled_nvfp4
        and not q.nvfp4_use_4over6
        and q.nvfp4_e4m3_max == 448
        and q.dtype == tex.DType.kFloat4E2M1
        and (q.rowwise_usage or q.columnwise_usage)
        and not _FAST_MATH
    )


def _kind(q, x) -> Optional[str]:
    from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer

    if type(q) is not NVFP4Quantizer or not _common_ok(q):
        return None
    if not (isinstance(x, torch.Tensor) and x.is_cuda and x.dtype == torch.bfloat16
            and x.dim() >= 2 and x.is_contiguous() and x.numel() > 0
            and x.data_ptr() % 16 == 0):
        return None
    if type(x) is not torch.Tensor and not isinstance(x, torch.nn.Parameter):
        return None
    N = x.shape[-1]
    M = x.numel() // N
    if q.with_rht:
        if not q.with_post_rht_amax or q.with_2d_quantization:
            return None
        rht = getattr(q, "rht_matrix", None)
        if rht is None or rht.dtype != torch.bfloat16 or rht.numel() != 256:
            return None
        if M % 64 or N % 128 or not _sm100_family():
            return None
        return "rht"
    if q.with_2d_quantization and not q.stochastic_rounding and M % 16 == 0 and N % 16 == 0:
        return "2d"
    return None


def _usable_output(out, q, x) -> bool:
    from transformer_engine.pytorch.tensor.storage.nvfp4_tensor_storage import NVFP4TensorStorage

    if not isinstance(out, NVFP4TensorStorage):
        return False
    if out._row_scaled_nvfp4 or out._nvfp4_e4m3_max != 448:
        return False
    if (out._rowwise_data is not None) != bool(q.rowwise_usage):
        return False
    if (out._columnwise_data is not None) != bool(q.columnwise_usage):
        return False
    if q.optimize_for_gemm and not out._with_gemm_swizzled_scales:
        return False
    n = x.numel()
    if out._rowwise_data is not None and (out._rowwise_data.numel() * 2 != n
                                          or out._amax_rowwise is None):
        return False
    if out._columnwise_data is not None and (out._columnwise_data.numel() * 2 != n
                                             or out._amax_columnwise is None):
        return False
    return True


def _quantize(kind, x, q, out, noop=None):
    ws = _ws_state(x.device)
    row = out._rowwise_data is not None
    col = out._columnwise_data is not None
    args = (
        out._rowwise_data if row else None,
        out._rowwise_scale_inv if row else None,
        out._amax_rowwise if row else None,
        out._columnwise_data if col else None,
        out._columnwise_scale_inv if col else None,
        out._amax_columnwise if col else None,
        bool(out._with_gemm_swizzled_scales),
    )
    if kind == "rht":
        sr = bool(q.stochastic_rounding)
        rng = torch.empty(2, dtype=torch.int64, device=x.device) if sr else None
        torch.ops.optimized.qb_rht_quantize(x, q.rht_matrix, int(q.rht_matrix_random_sign_mask_t),
                                          *args, sr, rng, ws)
        _counts["rht_sr" if sr else "rht"] += 1
    else:
        torch.ops.optimized.qb_quantize_2d(x, *args, noop, ws)
        _counts["2d"] += 1


def _wrap_quantize(orig):
    def quantize(tensor, quantizer, output=None, noop=None):
        kind = _kind(quantizer, tensor)
        noop_ok = noop is None or (
            kind == "2d" and isinstance(noop, torch.Tensor) and noop.is_cuda
            and noop.dtype == torch.float32 and noop.numel() == 1
        )
        if kind is not None and noop_ok:
            out = output
            if out is None:
                out = quantizer.make_empty(tensor.shape, dtype=tensor.dtype, device=tensor.device)
            if _usable_output(out, quantizer, tensor):
                _announce()
                _quantize(kind, tensor, quantizer, out, noop)
                return out
        _counts["fallback"] += 1
        return orig(tensor, quantizer, output, noop)

    quantize._opt_orig = orig
    return quantize


def _wrap_pre_fuser_forward(orig):
    def pre_fuser_forward(self, *args, **kwargs):
        orig(self, *args, **kwargs)
        _preswizzle_weight(self)

    pre_fuser_forward._opt_orig = orig
    return pre_fuser_forward


def _preswizzle_weight(op):
    from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer
    from transformer_engine.pytorch.quantized_tensor import QuantizedTensorStorage

    try:
        q = op.get_quantizer("forward", 1)
    except Exception:  # noqa: BLE001  (no recipe state yet)
        return
    w = getattr(op, "weight", None)
    if type(q) is not NVFP4Quantizer or w is None or isinstance(w, QuantizedTensorStorage):
        return
    rows, cols = w.numel() // w.shape[-1], w.shape[-1]
    if (
        not q.with_rht
        and q.with_2d_quantization
        and not q.row_scaled_nvfp4
        and not q.nvfp4_use_4over6
        and rows % 128 == 0
        and cols % 128 == 0
        and torch.cuda.get_device_capability() >= (10, 0)
    ):
        q.optimize_for_gemm = True


def set_active(active: bool):
    import transformer_engine.pytorch  # noqa: F401  (loads transformer_engine_torch)
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.ops.basic.basic_linear import BasicLinear

    if not _orig:
        _orig.update(quantize=tex.quantize, pre_fuser_forward=BasicLinear.pre_fuser_forward)
    if active:
        _load()
        tex.quantize = _wrap_quantize(_orig["quantize"])
        BasicLinear.pre_fuser_forward = _wrap_pre_fuser_forward(_orig["pre_fuser_forward"])
    else:
        tex.quantize = _orig["quantize"]
        BasicLinear.pre_fuser_forward = _orig["pre_fuser_forward"]


def install():
    if _ENABLED:
        _arch.check_te_version("quant_bookkeeping (TE's NVFP4 amax / 2D-cast arithmetic and NVTE C API mirrored by hand)",
                               "OPT_QUANT_BOOKKEEPING=0")
        set_active(True)


install()
