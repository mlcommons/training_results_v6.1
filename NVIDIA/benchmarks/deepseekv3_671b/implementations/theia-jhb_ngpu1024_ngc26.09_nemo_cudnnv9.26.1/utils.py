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
import logging
import sys
import time
from contextlib import contextmanager

from functools import wraps


def get_rank():
    return int(os.getenv("SLURM_PROCID", 0))


def env_flag(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def startup_timing_enabled():
    return env_flag("STARTUP_TIMING")


def should_log_startup_timing():
    return startup_timing_enabled() and (
        env_flag("STARTUP_TIMING_ALL_RANKS") or get_rank() == 0
    )


def startup_mark(phase, event="mark", start_ns=None, *, at_ns=None, **metadata):
    if not should_log_startup_timing():
        return
    now_ns = at_ns if at_ns is not None else time.time_ns()
    fields = {
        "phase": phase,
        "event": event,
        "rank": get_rank(),
        "epoch_ms": now_ns // 1_000_000,
        "iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(now_ns // 1_000_000_000))
        + f".{(now_ns // 1_000_000) % 1000:03d}Z",
    }
    if start_ns is not None:
        fields["elapsed_ms"] = (now_ns - start_ns) // 1_000_000
    fields.update({key: value for key, value in metadata.items() if value is not None})
    print(
        "STARTUP_TIMING " + " ".join(f"{key}={value}" for key, value in fields.items()),
        flush=True,
    )


def startup_span(phase, start_ns, **metadata):
    startup_mark(phase, "start", at_ns=start_ns, **metadata)
    startup_mark(phase, "end", start_ns, status="ok", **metadata)


@contextmanager
def startup_timer(phase, **metadata):
    if not startup_timing_enabled():
        yield
        return

    start_ns = time.time_ns()
    startup_mark(phase, "start", start_ns, **metadata)
    try:
        yield
    except Exception as exc:
        startup_mark(
            phase, "end", start_ns, status="error",
            error=type(exc).__name__, **metadata
        )
        raise
    else:
        startup_mark(phase, "end", start_ns, status="ok", **metadata)


def rank_last(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        from megatron.bridge.utils.common_utils import is_last_rank

        if is_last_rank():
            return func(*args, **kwargs)
    return wrapper


class RankZeroFilter(logging.Filter):
    def filter(self, record):
        return get_rank() == 0 or record.levelno >= logging.WARNING


_rank_zero_filter = RankZeroFilter()


def init_logging():
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    for handler in root.handlers:
        handler.addFilter(_rank_zero_filter)

    # Silence experimental-API and discrete_col_sfd warnings from cuDNN grouped-gemm
    # kernels (logger name == kernel class name; see cudnn api_base.py:388).
    for kernel_api in (
        "GroupedGemmGluSm100",
        "GroupedGemmQuantSm100",
        "GroupedGemmDgluSm100",
        "GroupedGemmWgradSm100",
        "GroupedGemmGluBlockScaledAPI",
        "GroupedGemmDgluBlockScaledAPI",
        "GroupedGemmWgradBlockScaledAPI",
        "GroupedGemmWgradBf16API",
    ):
        logging.getLogger(kernel_api).setLevel(logging.ERROR)

    l = logging.getLogger("megatron.core.dist_checkpointing.validation")
    l.setLevel(logging.WARNING)
    if not l.handlers:
        h = logging.StreamHandler()
        h.setLevel(logging.WARNING)
        h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        l.addHandler(h)
    l.propagate = False  # prevent duplicate logs if root has handlers
