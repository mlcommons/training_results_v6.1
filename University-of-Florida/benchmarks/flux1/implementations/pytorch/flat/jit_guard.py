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

"""Count JIT compiles and report any that happen after the run clock starts."""

import functools
import threading
import time

_lock = threading.Lock()
_frozen = False
_counts: dict[str, int] = {"cpp_extension.load": 0, "cute.compile": 0}
_late: list[str] = []


def _wrap(fn, kind, describe):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        with _lock:
            _counts[kind] += 1
            frozen = _frozen
        if frozen:
            what = describe(args, kwargs)
            with _lock:
                _late.append(what)
            print(f"[jit_guard] JIT compile after run_start: {what} "
                  f"({time.perf_counter() - t0:.1f} s)", flush=True)
        return result

    wrapper._jit_guard = True
    return wrapper


def _describe_load(args, kwargs):
    name = kwargs.get("name", args[0] if args else "?")
    return f"cpp_extension.load(name={name!r})"


def _describe_cute(args, kwargs):
    target = args[0] if args else None
    return f"cute.compile({getattr(target, '__name__', type(target).__name__)})"


def install() -> None:
    """Patch the two JIT entry points in place. Idempotent."""
    import torch.utils.cpp_extension as cpp_extension

    if not getattr(cpp_extension.load, "_jit_guard", False):
        cpp_extension.load = _wrap(
            cpp_extension.load, "cpp_extension.load", _describe_load,
        )
    try:
        import cutlass.cute as cute
    except ImportError as exc:
        print(f"[jit_guard] cutlass.cute not importable ({exc}); "
              "cute.compile is not monitored", flush=True)
        return
    if not getattr(cute.compile, "_jit_guard", False):
        cute.compile = _wrap(cute.compile, "cute.compile", _describe_cute)


def freeze() -> None:
    """From here on every compile is reported as late."""
    global _frozen
    with _lock:
        _frozen = True


def compile_count() -> int:
    """Number of compiles seen so far, late or not."""
    return sum(_counts.values())


def summary() -> str:
    """Per-entry-point compile counts, e.g. ``7 cpp_extension.load, 5 cute.compile``."""
    return ", ".join(f"{n} {kind}" for kind, n in _counts.items())


def late_compiles() -> list[str]:
    """Descriptions of the compiles that happened after :func:`freeze`."""
    return list(_late)
