# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Build the CUDA extensions of lora_kernels for sm_100a, sm_103a and sm_107a.

The Dockerfile runs this script one time. It needs nvcc but no GPU. The loaders then import
the prebuilt modules from build/<folder>/ and do not compile at run time.
Usage: python build.py [build root]
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor

from torch.utils.cpp_extension import load

_FOLDER = os.path.dirname(os.path.abspath(__file__))
_ARCHS = ("100a", "103a", "107a")
# Same optimization flags as TE (no fast math). The FP4 PTX needs the "a" targets.
# Explicit -gencode flags also stop torch from adding arch flags of its own.
FLAGS = ["-O3", "--expt-relaxed-constexpr"] + [
    f"-gencode=arch=compute_{arch},code=sm_{arch}" for arch in _ARCHS
]
# (module name that the loader imports, source file, sub-folder of the build root)
EXTENSIONS = [
    ("lora_fast_rope", "fast_rope.cu", "fast_rope"),
    ("nvfp4q_cuda", "nvfp4q.cu", "nvfp4q"),
    ("lora_swiglu_fused", "swiglu_fused.cu", "swiglu"),
    ("swiglu_sr_cuda", "swiglu_sr.cu", "swiglu_sr"),
    ("lora_attn", "lora_attn.cu", "lora_attn"),
    ("lora_fused_cuda", "lora_fused.cu", "lora_fused"),
    ("resnorm_cuda", "resnorm.cu", "resnorm"),
    ("fp8_o_cuda", "fp8_o.cu", "fp8_o"),
]


def build_extension(entry: tuple, build_root: str) -> str:
    """Compile one extension and return the path of its module file."""
    name, source, folder = entry
    directory = os.path.join(build_root, folder)
    os.makedirs(directory, exist_ok=True)
    load(name=name, sources=[os.path.join(_FOLDER, source)], extra_cuda_cflags=FLAGS,
         build_directory=directory, verbose=False)
    return os.path.join(directory, f"{name}.so")


def main() -> int:
    """Build all extensions in parallel and check that each module file exists."""
    build_root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_FOLDER, "build")
    with ProcessPoolExecutor(max_workers=len(EXTENSIONS)) as pool:
        paths = list(pool.map(build_extension, EXTENSIONS, [build_root] * len(EXTENSIONS)))
    missing = [path for path in paths if not os.path.exists(path)]
    for path in paths:
        print(f"built {path}", flush=True)
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
