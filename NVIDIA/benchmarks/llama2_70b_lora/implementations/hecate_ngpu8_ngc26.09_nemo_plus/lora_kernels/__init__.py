# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Kernel patches for the LoRA NVFP4 recipe.

Each patch except LORA_PATCH_FUSED_LORA and LORA_PATCH_RESNORM gives the same bits as the TE
path that it replaces, so the losses and the eval losses do not change. Each switch turns on
one patch, and all switches are off by default.
callback_nvfp4.py calls apply_patches() at import, before train.py builds the model.
"""

import importlib
import os
import sys

_FOLDER = os.path.dirname(os.path.abspath(__file__))

# Loader variable -> sub-folder of build/ with the prebuilt extension (see build.py).
BUILD_DIRS = {
    "LORA_FAST_ROPE_BUILD_DIR": "fast_rope",
    "NVFP4Q_BUILD_DIR": "nvfp4q",
    "LORA_SWIGLU_BUILD_DIR": "swiglu",
    "SWIGLU_SR_BUILD_DIR": "swiglu_sr",
    "LORA_ATTN_BUILD_DIR": "lora_attn",
    "LORA_FUSED_BUILD_DIR": "lora_fused",
    "RESNORM_BUILD_DIR": "resnorm",
    "FP8_O_BUILD_DIR": "fp8_o",
}

# (switch, module) in the order of the tested runs.
PATCHES = [
    ("LORA_PATCH_FAST_ROPE", "fast_rope"),
    ("LORA_PATCH_FAST_KERNELS", "nvfp4q_swiglu"),
    ("LORA_SWIGLU_SR_TWO_PASS", "swiglu_sr_patch"),
    ("LORA_PATCH_FAST_DROPOUT", "fast_dropout"),
    ("LORA_PATCH_FP8_ROPE", "fp8_rope"),
    # Must come last: it chains in front of the quantize and RoPE patches.
    ("LORA_PATCH_AMAX_PRODUCERS", "amax_producers"),
    # Not bit-identical: the fp32 sum order of the LoRA dots changes.
    ("LORA_PATCH_FUSED_LORA", "lora_fused_patch"),
    # Not bit-identical: the rstd sum order and the grad sum change. Must come after the fused
    # LoRA patch: it wraps the function that it installs.
    ("LORA_PATCH_RESNORM", "resnorm_patch"),
    # Same bits as TE: the attention returns the FP8 O; the proj quantize and the LoRA branch read
    # the FP8 bytes. It chains in front of the quantize patches, so it comes after them.
    ("LORA_PATCH_FP8_O", "fp8_o_patch"),
    # Same bits as TE: the quantize kernels write the GEMM alpha, so TE starts no alpha kernel.
    ("LORA_PATCH_GEMM_ALPHA", "gemm_alpha"),
]
_applied: list = []


def enabled_patches() -> list:
    """Return the module names of the patches that their switches turn on."""
    return [module for switch, module in PATCHES if os.environ.get(switch, "0") == "1"]


def apply_patches() -> list:
    """Apply the patches that the switches turn on. Return their module names."""
    names = enabled_patches()
    if not names:
        return []
    build_root = os.environ.get("LORA_KERNELS_BUILD_DIR", os.path.join(_FOLDER, "build"))
    for variable, folder in BUILD_DIRS.items():
        os.environ.setdefault(variable, os.path.join(build_root, folder))
    # The modules import each other by plain name.
    if _FOLDER not in sys.path:
        sys.path.insert(0, _FOLDER)
    for name in names:
        module = importlib.import_module(name)
        module.apply()
        _applied.append(module)
    return names


def on_model_ready(model: object) -> None:
    """Give the built model to the applied patches that need it (the resnorm layer links)."""
    for module in _applied:
        mark_model = getattr(module, "mark_model", None)
        if mark_model is not None:
            mark_model(model)


def report() -> dict:
    """Return the counters of the applied patches that have a stats() function."""
    result = {}
    for module in _applied:
        stats = getattr(module, "stats", None)
        if stats is not None:
            result[module.__name__] = stats()
    return result


def on_train_step_start() -> None:
    """Clear the per-step state of the applied patches (the amax side channel)."""
    for module in _applied:
        clear = getattr(module, "clear", None)
        if clear is not None:
            clear()
