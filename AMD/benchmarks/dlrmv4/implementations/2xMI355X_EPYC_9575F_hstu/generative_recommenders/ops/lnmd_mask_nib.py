# Copyright (c) Meta Platforms, Inc. and affiliates.
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
import threading
from typing import Dict, Optional

import torch

KNOB: str = "DLRM_HSTU_FP8_LNMD_MASK_NIB"
FAMILY: str = "LN*U*dropout keep mask NIB"
D: int = 512
ROW_BYTES: int = 256
_INT8_ATTR: str = "_lnmd_mask_nib_int8"

_lock = threading.Lock()
_COUNTS: Dict[str, int] = {"unpack": 0}
_printed: set = set()


def requested() -> bool:
    return os.environ.get(KNOB, "").strip() == "1"


def is_nib(t: Optional[torch.Tensor]) -> bool:
    return (
        isinstance(t, torch.Tensor)
        and t.dtype == torch.uint8
        and t.dim() == 2
        and int(t.shape[1]) == ROW_BYTES
    )


def layout_ok(t: Optional[torch.Tensor], m: int) -> bool:
    return (
        is_nib(t)
        and int(t.shape[0]) == int(m)
        and tuple(t.stride()) == (ROW_BYTES, 1)
        and t.data_ptr() % 16 == 0
    )


def pack_ref(mask: torch.Tensor) -> torch.Tensor:
    n = int(mask.shape[0])
    v = mask.reshape(n, D // 8, 8).view(torch.uint8)
    return ((v[:, :, 0:4] & 7) | ((v[:, :, 4:8] & 7) << 4)).reshape(n, ROW_BYTES).contiguous()


def unpack(packed: torch.Tensor) -> torch.Tensor:
    n = int(packed.shape[0])
    p = packed.reshape(n, D // 8, 4)
    out = torch.empty((n, D), dtype=torch.uint8, device=packed.device)
    o = out.view(n, D // 8, 8)
    torch.bitwise_and(p, 7, out=o[:, :, 0:4])
    hi = o[:, :, 4:8]
    torch.bitwise_right_shift(p, 4, out=hi)
    hi.bitwise_and_(7)
    return out.view(torch.int8)


def counters() -> Dict[str, int]:
    with _lock:
        return dict(_COUNTS)


def reset_for_testing() -> None:
    with _lock:
        for k in _COUNTS:
            _COUNTS[k] = 0
        _printed.clear()


def as_int8(mask: Optional[torch.Tensor], where: str) -> Optional[torch.Tensor]:
    if not is_nib(mask):
        return mask
    cached = getattr(mask, _INT8_ATTR, None)
    if cached is not None:
        return cached
    out = unpack(mask)
    try:
        setattr(mask, _INT8_ATTR, out)
    except Exception:
        pass
    with _lock:
        _COUNTS["unpack"] += 1
        count = _COUNTS["unpack"]
        first = where not in _printed
        _printed.add(where)
    if first:
        line = (
            f"[lnmd-mask-nib] {FAMILY}: exact unpack to int8 for {where} (a non-DO8-NIB consumer read a NIB mask; "
            f"counted, unpack={count})"
        )


        try:
            from generative_recommenders.ops.utils import arm_print_all_ranks
        except Exception:
            print(f"{line} [rank {os.environ.get('RANK', '?')}]", flush=True)
        else:
            arm_print_all_ranks(line)
    return out
