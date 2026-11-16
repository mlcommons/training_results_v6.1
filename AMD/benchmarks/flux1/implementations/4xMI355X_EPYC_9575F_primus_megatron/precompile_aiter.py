# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Pre-compile the aiter JIT modules the Flux MXFP6 training path loads.

aiter compiles its C++/HIP extension modules on first use. Building them into the image
keeps every compilation out of the timed region and out of the 30-minute init budget.

aiter has two loaders and they build differently, so each module is built the way its own
loader would build it (aiter/jit/core.py): pybind modules with their configured arguments,
ctypes modules with torch_exclude forced on. A module built the other way would sit at the
path the loader checks and be used as-is. The README's verification step checks that a run
builds nothing at runtime.
"""
import importlib.util
import os
import sys
import types

os.environ.setdefault("GPU_ARCHS", "gfx950")

# Import only aiter's build machinery (aiter/jit). The package's own __init__ imports the
# operator modules, which initialise Triton's GPU driver and fail where there is no GPU
# (as in a docker build). A bare package object for `aiter` skips that __init__.
_spec = importlib.util.find_spec("aiter")
_pkg = types.ModuleType("aiter")
_pkg.__path__ = list(_spec.submodule_search_locations)
_pkg.__spec__ = _spec
_pkg.__file__ = _spec.origin
sys.modules["aiter"] = _pkg

from aiter.jit.core import build_module, get_args_of_build  # noqa: E402

PYBIND_MODULES = [
    "module_aiter_core",
    "module_quant",
    "module_fmha_v3_fwd",
    "module_fmha_v3_bwd",
]
CTYPES_MODULES = [
    "module_gemm_a6w6_asm",
    # the MX GEMMs on the tilescale layout (every config) and the MXFP4 GEMMs' per-shape fallback
    "module_gemm_tilescale_asm",
    "module_gemm_a4w4_blob_asm",
    # the AdaLN modulation GEMMs (mxfp6_adaln_gemm_backend: turbo)
    "module_adaln_gemm_asm",
]


def build(name, ctypes):
    a = get_args_of_build(name)
    if ctypes:
        a["torch_exclude"] = True
    build_module(
        name,
        a["srcs"],
        a["flags_extra_cc"],
        a["flags_extra_hip"],
        a["blob_gen_cmd"],
        a["extra_include"],
        a["extra_ldflags"],
        a["verbose"],
        a["is_python_module"],
        a["is_standalone"],
        a["torch_exclude"],
        a.get("third_party", []),
        False if ctypes else a.get("hipify", False),
        flags_extra_hip_per_source=a.get("flags_extra_hip_per_source", {}),
    )
    print(f"  compiled {name} ({'ctypes' if ctypes else 'pybind'})")


for name in PYBIND_MODULES:
    build(name, ctypes=False)
for name in CTYPES_MODULES:
    build(name, ctypes=True)

print("AITER pre-compile OK")
