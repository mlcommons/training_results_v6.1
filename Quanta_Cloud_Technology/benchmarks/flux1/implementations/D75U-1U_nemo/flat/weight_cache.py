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

"""Per-weight tables keyed on tensor identity rather than on an address."""

from __future__ import annotations

import weakref
from typing import Iterable

from torch import Tensor


def _evict(
    entries: dict[tuple[int, str | None], tuple[weakref.ref, object]],
    key: tuple[int, str | None],
    dead: weakref.ref,
) -> None:
    """Drop ``key`` when the tensor that owned it is collected."""
    entry = entries.get(key)
    if entry is not None and entry[0] is dead:
        del entries[key]


class WeightCache:
    """Quantized weights memoised against the identity of the source tensor.

    Keys are ``(data_ptr, fmt)``; ``fmt`` is ``None`` for a single-format cache.
    """

    __slots__ = ("_entries",)

    def __init__(self) -> None:
        self._entries: dict[
            tuple[int, str | None], tuple[weakref.ref, object]
        ] = {}

    @classmethod
    def of(cls, pairs: Iterable[tuple[Tensor, object]]) -> "WeightCache":
        """Build a cache from ``(weight, value)`` pairs."""
        cache = cls()
        for weight, value in pairs:
            cache.put(weight, value)
        return cache

    def get(self, weight: Tensor, fmt: str | None = None) -> object | None:
        """Return the memoised cast of ``weight`` in ``fmt``, or ``None`` on a miss."""
        entry = self._entries.get((weight.data_ptr(), fmt))
        if entry is None:
            return None
        owner, value = entry
        return value if owner() is weight else None

    def put(self, weight: Tensor, value: object, fmt: str | None = None) -> None:
        """Memoise ``value`` as the cast of ``weight`` in ``fmt``."""
        key = (weight.data_ptr(), fmt)
        entries = self._entries
        entries[key] = (
            weakref.ref(weight, lambda dead: _evict(entries, key, dead)),
            value,
        )

    def clear(self) -> None:
        """Drop every entry. Call after the optimizer mutates the weights."""
        self._entries.clear()

    def __len__(self) -> int:
        """Return the number of live entries."""
        return len(self._entries)
