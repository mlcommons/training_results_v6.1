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

import functools
import os

import torch

RUBIN = (10, 7)
BLACKWELL = (10, 3)

_announced: set = set()


@functools.lru_cache(maxsize=None)
def capability() -> tuple:
    major, minor = torch.cuda.get_device_capability(0)
    return (int(major), int(minor))


def sm() -> str:
    major, minor = capability()
    return f"sm_{major}{minor}"


def device_name() -> str:
    return torch.cuda.get_device_name(0)


def is_rubin() -> bool:
    return capability() == RUBIN


def is_blackwell() -> bool:
    return capability() == BLACKWELL


def cuda_arch_list() -> str:
    major, minor = capability()
    return f"{major}.{minor}a"


def ext_name(base: str) -> str:
    return f"{base}_{sm()}"


def _rank0() -> bool:
    return int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", "0"))) == 0


def rubin_only(feature: str) -> bool:
    if is_rubin():
        return False
    msg = (f"[opt] {feature}: skipped, needs Rubin (sm_107); this device is {device_name()} "
           f"({sm()}); the stock path runs instead")
    if os.environ.get("OPT_REQUIRE_RUBIN", "0") == "1":
        raise RuntimeError(msg)
    if feature not in _announced:
        _announced.add(feature)
        if _rank0():
            print(msg, flush=True)
    return True


TE_VERIFIED_VERSIONS = ("2.20.0+5b28d7aa",)


def check_te_version(feature: str, disable_hint: str) -> str:
    import transformer_engine as te

    v = getattr(te, "__version__", "unknown")
    if v in TE_VERIFIED_VERSIONS:
        return v
    msg = (f"[opt] {feature}: Transformer Engine {v} is not a build this module was verified bitwise "
           f"against ({', '.join(TE_VERIFIED_VERSIONS)}); re-run its --selftest on the new build and add the "
           f"version to opt_kernels/arch.py TE_VERIFIED_VERSIONS, or set {disable_hint}")
    if os.environ.get("OPT_TE_VERSION_CHECK", "1") != "0":
        raise RuntimeError(msg)
    if _rank0():
        print(msg + " (OPT_TE_VERSION_CHECK=0: continuing unverified)", flush=True)
    return v


_STACKS = ("rubin", "blackwell", "off")


@functools.lru_cache(maxsize=None)
def stack() -> str:
    requested = os.environ.get("OPT_STACK", "").strip().lower()
    if not torch.cuda.is_available():
        choice, why = "off", "no CUDA device"
    elif requested == "":
        choice, why = "off", "OPT_STACK not exported by the config: stock code path"
    elif requested in _STACKS:
        choice, why = requested, "OPT_STACK exported by the config"
    elif requested == "auto":
        choice = "rubin" if is_rubin() else ("blackwell" if is_blackwell() else "off")
        why = "OPT_STACK=auto: by device architecture"
    else:
        raise ValueError(f"OPT_STACK={requested!r}: expected one of {_STACKS}, 'auto' or unset")
    if choice == "blackwell" and not is_blackwell():
        msg = (f"[opt] OPT_STACK=blackwell: the Blackwell stack's kernels are built for sm_103a "
               f"(GB300 / B300) only; this device is {device_name()} ({sm()}). Leave OPT_STACK unset for "
               "the stock path, or set OPT_ALLOW_UNVERIFIED_ARCH=1 (development only)")
        if os.environ.get("OPT_ALLOW_UNVERIFIED_ARCH", "0") != "1":
            raise RuntimeError(msg)
        why += "; OPT_ALLOW_UNVERIFIED_ARCH=1 overrides the sm_103 check"
    if _rank0():
        dev = f"; device {device_name()}, {sm()}" if torch.cuda.is_available() else ""
        print(f"[opt] stack={choice} ({why}{dev})", flush=True)
    return choice
