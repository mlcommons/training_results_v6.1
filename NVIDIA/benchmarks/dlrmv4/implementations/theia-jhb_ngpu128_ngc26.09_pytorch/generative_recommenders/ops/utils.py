# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python3

# pyre-ignore-all-errors

import functools
import os

import torch


class _PlainFuncWrapper:
    """Thin wrapper around a plain function that provides no-op register_fake
    and register_kernel methods, mirroring the CustomOpDef API so that
    downstream @func.register_fake / func.register_kernel("cpu") calls
    don't break when the function is not wrapped as a custom op."""

    def __init__(self, func):
        self._func = func
        functools.update_wrapper(self, func)

    def __call__(self, *args, **kwargs):
        return self._func(*args, **kwargs)

    def register_fake(self, fake_func):
        return fake_func

    def register_kernel(self, device):
        def inner(func):
            return func

        return inner


def maybe_register_custom_op(op_name, mutates_args):
    """
    Conditionally registers a function as a torch custom op.

    When AOTI_LOWER is set in the environment, the function is returned
    unwrapped so that torch.export / Dynamo can trace through the plain
    Python implementation instead of treating the custom op as opaque.
    """

    def decorator(func):
        if os.environ.get("AOTI_LOWER"):
            return _PlainFuncWrapper(func)
        return torch.library.custom_op(op_name, func, mutates_args=mutates_args)

    return decorator


_FUSED_LN_BWD_CHAIN_SWITCH = "HSTU_FUSED_LN_BWD_CHAIN"
_FUSED_PRE_CHAIN_SWITCH = "HSTU_FUSED_PRE_CHAIN"


def _switch_on(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in {"1", "true", "yes", "on"}


def use_fused_ln_bwd_chain() -> bool:
    """Whether every HSTU layer's input-LayerNorm backward runs as the fused
    chain (LN-bwd + residual add + the first layer's input-dropout backward +
    dgamma/dbeta partials in one launch), with the input dropout moved into
    the first layer's Function and the two backward kernels before it (the
    position / time table scatter, the <= 16-row component reduction) in
    their tiled forms. One switch for all of it: the pieces were tuned and
    measured together, and the dropout move only pays with the fused chain.
    Off unless the switch is set."""
    return _switch_on(_FUSED_LN_BWD_CHAIN_SWITCH)


def use_fused_pre_chain() -> bool:
    """Whether the compacted path runs the pre-HSTU forward chain (compacted
    component sum + contextual concat, timestamp concat, position / time
    embedding add, input dropout) as one Triton launch that writes x0 and the
    dropout mask once. It builds on the fused LayerNorm-backward chain: the
    first layer's Function owns the dropout backward and that switch's tiled
    kernels form the fused forward's backward, so it requires
    HSTU_FUSED_LN_BWD_CHAIN. Off unless the switch is set."""
    if not _switch_on(_FUSED_PRE_CHAIN_SWITCH):
        return False
    if not use_fused_ln_bwd_chain():
        raise RuntimeError(
            f"{_FUSED_PRE_CHAIN_SWITCH} requires {_FUSED_LN_BWD_CHAIN_SWITCH}: the "
            "fused pre-HSTU chain hands its dropout mask to the first layer's "
            "fused backward"
        )
    return True


def is_sm100_plus() -> bool:
    """
    Check if this is a Blackwell (sm_100-103, B200-GB300) or Rubin (sm_107)
    Datacenter GPU.

    Rubin takes the Blackwell-tuned dispatch paths by default; set
    HSTU_RUBIN_USE_SM100_PATHS=0 to fall back to the generic paths without a
    rebuild if a Blackwell-only kernel (e.g. TLX) breaks during bringup.
    """
    if not torch.cuda.is_available():
        return False
    props = torch.cuda.get_device_properties(0)
    if props.major != 10:
        return False
    if props.minor <= 3:
        return True
    return props.minor == 7 and os.environ.get("HSTU_RUBIN_USE_SM100_PATHS", "1") == "1"


def is_sm90() -> bool:
    if not torch.cuda.is_available():
        return False
    props = torch.cuda.get_device_properties(0)
    return props.major == 9 and props.minor == 0


def is_sm90_plus() -> bool:
    return is_sm100_plus() or is_sm90()


@functools.lru_cache(maxsize=None)
def is_amd_mi350() -> bool:
    """Detect an AMD Instinct MI350-class GPU (gfx950) running under ROCm.

    MI350 benefits from the same multi-row, separated-RNG layer-norm-mul-dropout
    path as Blackwell datacenter parts (sm_100), so it is gated together with
    is_sm100_plus() at the kernel dispatch sites.
    """
    if not torch.cuda.is_available():
        return False
    if getattr(torch.version, "hip", None) is None:
        return False
    try:
        arch = torch.cuda.get_device_properties(0).gcnArchName or ""
    except (AssertionError, RuntimeError, AttributeError):
        return False
    return "gfx950" in arch


def use_separated_rng_ln_mul_dropout() -> bool:
    """Hardware that should use the autotuned, multi-row ``_ln_mul_dropout_fwd_rng``
    kernel with a precomputed dropout mask instead of the legacy single-row,
    fused-RNG ``_ln_mul_dropout_fwd`` kernel.

    Blackwell datacenter GPUs (sm_100-103) and AMD MI350 (gfx950) both prefer the
    separated-RNG path: it batches rows per program and lets the backward reuse the
    same mask, which is a large win over launching one program per row.
    """
    return is_sm100_plus() or is_amd_mi350()


def copy_if_different_ptr(dst: torch.Tensor, src: torch.Tensor) -> None:
    if torch.compiler.is_compiling():
        # .data_ptr() will break PT2
        dst.copy_(src)
    else:
        if dst.data_ptr() != src.data_ptr():
            dst.copy_(src)
