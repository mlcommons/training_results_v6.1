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

"""NVFP4 GEMM through nvte_cublas_gemm_v2 with an alpha on the GPU.

Before each NVFP4 GEMM, TE starts a one-thread kernel that computes alpha from the two amax
values. TE skips that kernel when the operands have no amax. Then cuBLASLt reads the alpha
pointer that the caller gives (the pointer mode is DEVICE for NVFP4).
The struct layouts and the enum values are those of TE 2.20 (commit 5b28d7aa).
"""

import ctypes
import os

import numpy as np
import torch

# NVTEScalingMode, NVTETensorParam, NVTEDType and NVTEMatmulConfigAttribute of TE 2.20.
_DELAYED_TENSOR_SCALING = 0
_NVFP4_1D_SCALING = 4
_ROWWISE_DATA = 0
_COLUMNWISE_DATA = 1
_ROWWISE_SCALE_INV = 4
_COLUMNWISE_SCALE_INV = 5
_WITH_GEMM_SWIZZLED_SCALES = 7
_NVFP4_E4M3_MAX = 9
_BYTE = 0
_FLOAT32 = 4
_BFLOAT16 = 6
_FLOAT8_E4M3 = 7
_FLOAT4_E2M1 = 10
_CONFIG_USE_SPLIT_ACCUMULATOR = 5
_CONFIG_SM_COUNT = 6

# TE compute_nvfp4_per_tensor_scale_kernel: alpha = ((alpha_in * amax_a) * amax_b) * factor,
# with factor = 1 / (6 * 6 * 448 * 448) in fp32.
ALPHA_FACTOR = float(np.float32(1.0) / np.float32(6.0 * 6.0 * 448.0 * 448.0))


class _Shape(ctypes.Structure):
    _fields_ = [("data", ctypes.c_size_t * 15), ("ndim", ctypes.c_size_t)]


class _BasicTensor(ctypes.Structure):
    _fields_ = [("data_ptr", ctypes.c_void_p), ("dtype", ctypes.c_int), ("shape", _Shape)]


def _basic_tensor(ptr: int, dtype: int, shape: tuple) -> _BasicTensor:
    """Return an NVTEBasicTensor for a raw device pointer."""
    tensor = _BasicTensor()
    tensor.data_ptr = ctypes.c_void_p(ptr)
    tensor.dtype = dtype
    tensor.shape.ndim = len(shape)
    for index, size in enumerate(shape):
        tensor.shape.data[index] = int(size)
    return tensor


def _library_path() -> str:
    """Return the path of libtransformer_engine.so."""
    import transformer_engine
    import transformer_engine_torch  # noqa: F401  (loads libtransformer_engine)

    folder = os.path.dirname(os.path.abspath(transformer_engine.__file__))
    path = os.path.join(folder, "libtransformer_engine.so")
    if os.path.exists(path):
        return path
    with open("/proc/self/maps") as maps:
        for line in maps:
            if "libtransformer_engine" in line and ".so" in line:
                return line.split()[-1]
    raise RuntimeError("libtransformer_engine.so was not found")


def te_workspace_bytes(workspace: torch.Tensor) -> int:
    """Return the workspace size that TE gives cuBLASLt when it computes the alpha itself.

    TE keeps the last 4 aligned bytes for the alpha. The same size keeps the same algorithm.
    """
    return (workspace.numel() // 4) * 4 - 4


def math_sm_count(device: torch.device) -> int:
    """Return the SM count that TE sets in the GEMM config (0 when NVTE_EXT_MARGIN_SM is unset)."""
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    return sm_count - int(os.environ.get("NVTE_EXT_MARGIN_SM", sm_count))


class NvteGemm:
    """nvte_cublas_gemm_v2 on NVFP4 operands with no amax and an alpha on the GPU."""

    def __init__(self) -> None:
        lib = ctypes.CDLL(_library_path())
        lib.nvte_create_tensor.restype = ctypes.c_void_p
        lib.nvte_create_tensor.argtypes = [ctypes.c_int]
        lib.nvte_destroy_tensor.restype = None
        lib.nvte_destroy_tensor.argtypes = [ctypes.c_void_p]
        lib.nvte_set_tensor_param_v2.restype = None
        lib.nvte_set_tensor_param_v2.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
        lib.nvte_create_matmul_config.restype = ctypes.c_void_p
        lib.nvte_create_matmul_config.argtypes = []
        lib.nvte_set_matmul_config_attribute.restype = None
        lib.nvte_set_matmul_config_attribute.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
        lib.nvte_cublas_gemm_v2.restype = None
        lib.nvte_cublas_gemm_v2.argtypes = [ctypes.c_int] * 2 + [ctypes.c_void_p] * 9
        self.lib = lib
        self.config = ctypes.c_void_p(lib.nvte_create_matmul_config())
        self._e4m3_max = ctypes.c_int(448)
        self._swizzled = ctypes.c_uint8(1)

    def _set(self, tensor: ctypes.c_void_p, param: int, basic: _BasicTensor) -> None:
        """Set one buffer parameter of an NVTE tensor."""
        self.lib.nvte_set_tensor_param_v2(tensor, param, ctypes.byref(basic), ctypes.sizeof(basic))

    def _nvfp4(self, data: torch.Tensor, scales: torch.Tensor, columnwise: bool) -> ctypes.c_void_p:
        """Return an NVFP4 NVTE tensor with one usage, swizzled scales and no amax."""
        tensor = ctypes.c_void_p(self.lib.nvte_create_tensor(_NVFP4_1D_SCALING))
        shape = list(data.shape)
        shape[-1] *= 2  # Two e2m1 values in each byte.
        data_param = _COLUMNWISE_DATA if columnwise else _ROWWISE_DATA
        scale_param = _COLUMNWISE_SCALE_INV if columnwise else _ROWWISE_SCALE_INV
        self._set(tensor, data_param, _basic_tensor(data.data_ptr(), _FLOAT4_E2M1, tuple(shape)))
        self._set(tensor, scale_param, _basic_tensor(scales.data_ptr(), _FLOAT8_E4M3, tuple(scales.shape)))
        self.lib.nvte_set_tensor_param_v2(
            tensor, _WITH_GEMM_SWIZZLED_SCALES, ctypes.byref(self._swizzled), 1)
        self.lib.nvte_set_tensor_param_v2(tensor, _NVFP4_E4M3_MAX, ctypes.byref(self._e4m3_max), 4)
        return tensor

    def _plain(self, ptr: int, dtype: int, shape: tuple) -> ctypes.c_void_p:
        """Return a plain NVTE tensor (output or workspace)."""
        tensor = ctypes.c_void_p(self.lib.nvte_create_tensor(_DELAYED_TENSOR_SCALING))
        self._set(tensor, _ROWWISE_DATA, _basic_tensor(ptr, dtype, shape))
        return tensor

    def gemm(self, layout: str, operands: tuple, out: torch.Tensor, scalars: tuple,
             options: tuple) -> None:
        """Run D = alpha * op(A) op(B) + beta * D.

        operands: (a_data, a_scales, b_data, b_scales). scalars: (alpha, beta) device fp32 tensors.
        options: (use_split_accumulator, sm_count, workspace, workspace_bytes).
        """
        a_data, a_scales, b_data, b_scales = operands
        alpha, beta = scalars
        use_split_accumulator, sm_count, workspace, workspace_bytes = options
        transa = layout[0] == "T"
        transb = layout[1] == "T"
        tensor_a = self._nvfp4(a_data, a_scales, columnwise=not transa)
        tensor_b = self._nvfp4(b_data, b_scales, columnwise=transb)
        out_dtype = _FLOAT32 if out.dtype == torch.float32 else _BFLOAT16
        tensor_d = self._plain(out.data_ptr(), out_dtype, tuple(out.shape))
        tensor_w = self._plain(workspace.data_ptr(), _BYTE, (workspace_bytes,))
        try:
            split = ctypes.c_uint8(1 if use_split_accumulator else 0)
            self.lib.nvte_set_matmul_config_attribute(
                self.config, _CONFIG_USE_SPLIT_ACCUMULATOR, ctypes.byref(split), 1)
            sms = ctypes.c_int32(int(sm_count))
            self.lib.nvte_set_matmul_config_attribute(
                self.config, _CONFIG_SM_COUNT, ctypes.byref(sms), 4)
            stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
            self.lib.nvte_cublas_gemm_v2(
                1 if transa else 0, 1 if transb else 0, ctypes.c_void_p(alpha.data_ptr()),
                tensor_a, tensor_b, ctypes.c_void_p(beta.data_ptr()), tensor_d, tensor_d,
                tensor_w, self.config, stream)
        finally:
            for tensor in (tensor_a, tensor_b, tensor_d, tensor_w):
                self.lib.nvte_destroy_tensor(tensor)
