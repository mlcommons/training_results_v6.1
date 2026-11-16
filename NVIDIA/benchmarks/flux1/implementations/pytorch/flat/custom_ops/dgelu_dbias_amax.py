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

"""``torch.ops.flux`` boundary for the fused dGELU + dbias + amax producer.

``aten.gelu_backward`` produces 44% of the gradient population and was the only
gradient producer the workload did not own. ``flux.dgelu_dbias_amax`` replaces
the three kernels that used to run on every fc2 dgrad --
``aten.gelu_backward``, ``aten.sum`` (dbias) and TE's ``HadamardAmaxTmaKernel``
pass -- with one kernel that emits the BF16 dgrad, the bias gradient and
both per-tensor amaxes the NVFP4 cast needs. **It never quantizes**: the cast
stays TE's ``nvfp4_quantize_with_amax``, wrapped here as
``flux.te_nvfp4_quantize_with_amax`` in the ``te_quantized_gemm`` idiom, so
``S_enc`` is untouched and the dgrad is bit-identical to aten's.

One kernel stands behind the op: the CUTLASS-DSL kernel of
``dgelu_dbias_amax_ctm`` (production ``04x04x32`` -0.69 ms/step on two node sets). It takes contiguous
BF16 ``[M, N]`` with ``M % 16 == 0`` and ``N % 512 == 0`` -- every production
shape has ``N = 12288``. The CUDA kernel that used to serve the other widths
(``dgelu_dbias_amax_kernel.cu``) never ran at production and was the DSL
kernel's bitwise oracle; it has been deleted, so an
unsupported shape is now an error, never a fallback.

===================================  ==========================================
op                                   what it does
===================================  ==========================================
``flux.dgelu_dbias_amax``            ``(grad, pre, rht) -> (dgelu, dbias, amax[2])``
``flux.dgelu_dbias_amax_pergroup``   the same plus the per-16-row-group post-RHT
                                     amax ``[M/16]`` -- the debug path that
                                     pins the basis orientation
``flux.te_nvfp4_quantize_with_amax`` ``tex.nvfp4_quantize_with_amax`` returning
                                     the six plain tensors
===================================  ==========================================

``amax[0]`` is ``max|dgelu|``; ``amax[1]`` is the post-RHT columnwise amax,
``max |bf16_rn(X_g @ R)|`` over 16-row groups. The Hadamard basis is a
**Tensor** argument (``NVFP4Quantizer.rht_matrix``), never a sign mask, so a
wrong orientation can be driven from outside and caught.

The tanh: ``tanh.approx.f32`` by default (ON since its convergence
run came back null); ``FLAT_TANH_APPROX_DGELU=0``
selects the ``tanhf`` kernel that is bit-identical to aten
(``dgelu_dbias_amax_ctm.production_fast``). The fast dgelu is NOT within 1 ulp
of aten -- the measured envelope (``tests/test_dgelu_dbias_amax.py``) is
``|fast - exact| <= 2**-7 amax``, ``<= 1%`` of elements moved, ``<= 4`` ulp in
the live range; its bits are pinned by the golden fixture of
``tests/test_dgelu_dbias_amax_ctm.py``.

Concurrency and CUDA graphs: as the DSL module states -- one persistent,
self-resetting counter per device and column count, so two launches of this op
must not run concurrently on different streams of one device; every output is
an ordinary allocation made inside the op, no host synchronisation.
"""

from __future__ import annotations

import torch
import torch.library

import transformer_engine.pytorch.cpp_extensions as tex

from flat.custom_ops import dgelu_dbias_amax_ctm as _ctm
from flat.custom_ops import te_quantized_gemm as _te_ops


def dgelu_dbias_amax(
    grad: torch.Tensor, pre: torch.Tensor, rht_matrix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused tanh-GELU backward, bias gradient and NVFP4 amax pair.

    Args:
        grad: Gradient of the GELU output, ``[M, N]`` BF16 contiguous,
            ``M % 16 == 0``, ``N % 512 == 0``.
        pre: Saved pre-GELU activation, same shape/dtype/layout.
        rht_matrix: The quantizer's ``[16, 16]`` BF16 Hadamard basis.

    Returns:
        ``(dgelu, dbias, amax)``: BF16 ``[M, N]`` within the fast-tanh envelope of
        ``aten.gelu_backward(grad, pre, approximate="tanh")`` (bit-identical
        with ``FLAT_TANH_APPROX_DGELU=0``); BF16 ``[N]`` column sum of
        ``dgelu`` (fixed-order fp32, deterministic, within ``aten.sum``
        tolerance); FP32 ``[2]`` = ``(max|dgelu|, post-RHT amax)``, both of the
        emitted ``dgelu`` and bitwise TE's.

    Raises:
        ValueError: On a shape or layout the kernel does not take.
    """
    return _ctm.dgelu_dbias_amax_ctm(grad, pre, rht_matrix)


def dgelu_dbias_amax_pergroup(
    grad: torch.Tensor, pre: torch.Tensor, rht_matrix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """:func:`dgelu_dbias_amax` plus the per-16-row-group post-RHT amax.

    Debug path only: the scalar amax collides between ``R`` and
    ``R.T`` on ordinary draws, the per-group vector does not.

    Args:
        grad: As :func:`dgelu_dbias_amax`.
        pre: As :func:`dgelu_dbias_amax`.
        rht_matrix: As :func:`dgelu_dbias_amax`.

    Returns:
        ``(dgelu, dbias, amax, group_amax)`` with ``group_amax`` FP32 ``[M/16]``.
    """
    return _ctm.dgelu_dbias_amax_ctm_pergroup(grad, pre, rht_matrix)


def te_nvfp4_quantize_with_amax(
    t: torch.Tensor, quantizer_handle_: int,
    amax_rowwise: torch.Tensor, amax_columnwise: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """``tex.nvfp4_quantize_with_amax`` -- the NVFP4 cast with the amax supplied.

    Skips TE's ``ZeroAmaxKernel`` + ``HadamardAmaxTmaKernel`` pass; the cast
    kernel itself is the one ``tex.quantize`` runs (``cast.cpp:90-133`` at TE
    ``27486e03``).

    Args:
        t: Contiguous BF16 ``[M, K]``, both dims divisible by 16.
        quantizer_handle_: Registry handle of an ``NVFP4Quantizer`` with
            ``optimize_for_gemm`` set (see ``te_quantized_gemm``).
        amax_rowwise: FP32 ``[1]`` CUDA, ``max|t|``.
        amax_columnwise: FP32 ``[1]`` CUDA, post-RHT columnwise amax.

    Returns:
        ``(rowwise_data, rowwise_scale_inv, columnwise_data,
        columnwise_scale_inv, amax_rowwise, amax_columnwise)``.
    """
    quantizer = _te_ops._resolve(quantizer_handle_)
    assert quantizer.optimize_for_gemm, (
        "te_nvfp4_quantize_with_amax requires optimize_for_gemm (cast.cpp:83)."
    )
    q = tex.nvfp4_quantize_with_amax(t, quantizer, amax_rowwise, amax_columnwise)
    return (
        q._rowwise_data,
        q._rowwise_scale_inv,
        q._columnwise_data,
        q._columnwise_scale_inv,
        q._amax_rowwise,
        q._amax_columnwise,
    )


# ---------------------------------------------------------------------------
# torch.library registration
# ---------------------------------------------------------------------------

try:
    _lib = torch.library.Library("flux", "FRAGMENT")
except Exception:  # pragma: no cover - only when flux is not yet defined
    _lib = torch.library.Library("flux", "DEF")

_lib.define(
    "dgelu_dbias_amax(Tensor grad, Tensor pre, Tensor rht_matrix)"
    " -> (Tensor, Tensor, Tensor)"
)
_lib.define(
    "dgelu_dbias_amax_pergroup(Tensor grad, Tensor pre, Tensor rht_matrix)"
    " -> (Tensor, Tensor, Tensor, Tensor)"
)
_lib.define(
    "te_nvfp4_quantize_with_amax(Tensor t, int quantizer_handle,"
    " Tensor amax_rowwise, Tensor amax_columnwise)"
    " -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)"
)


@torch.library.impl(_lib, "dgelu_dbias_amax", "CUDA")
def _dgelu_dbias_amax_impl(
    grad: torch.Tensor, pre: torch.Tensor, rht_matrix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return dgelu_dbias_amax(grad, pre, rht_matrix)


@torch.library.impl(_lib, "dgelu_dbias_amax_pergroup", "CUDA")
def _dgelu_dbias_amax_pergroup_impl(
    grad: torch.Tensor, pre: torch.Tensor, rht_matrix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return dgelu_dbias_amax_pergroup(grad, pre, rht_matrix)


@torch.library.impl(_lib, "te_nvfp4_quantize_with_amax", "CUDA")
def _te_nvfp4_quantize_with_amax_impl(
    t: torch.Tensor, quantizer_handle: int,
    amax_rowwise: torch.Tensor, amax_columnwise: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    return te_nvfp4_quantize_with_amax(
        t, quantizer_handle, amax_rowwise, amax_columnwise,
    )
