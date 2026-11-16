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

"""One zeroed FP32 amax arena per region, instead of one ``zeros`` per call.

Both dual-format cast kernels (``flux.mxfp8_rowwise_cast_amax`` and its CUTLASS
DSL twin) reduce into their ``amax`` accumulator with ``atomicMax``, so the
accumulator must be **zero on entry**. Allocating it per call as
``torch.zeros(2)`` costs one ``FillFunctor`` node per call: 1.0-1.4 us of device
time x 266 calls per iteration ~= 0.3 ms/iter, and measurement showed that an in-graph fill costs exactly its device time (deleting the node
without deleting the work does **not** help -- in-graph nodes pipeline). This
module is a bump allocator over **blocks**: one ``[SLOTS, 2]`` buffer zeroed by
``flux.zero_f32_ctm`` (the DSL fill, not ``at::native``'s) -- one
fill -- serves the next ``SLOTS`` casts, so 266 fills become one.

A slot is **never reused**. That is not frugality about cursors, it is the
contract of what an amax is here: TE stores the amax tensor itself on the
quantized tensor as ``_amax_columnwise``, and ``te_nvfp4_gemm`` reads it to
derive the dequant scale at **GEMM** time -- which for a cached quantized weight
can be an iteration or more after the cast. A rewound slot would zero an amax a
later GEMM still reads, silently. So a block is retired rather than rewound, and
each block lives exactly as long as some slice of it does: ordinary Python
refcounting on the base tensor, no lifetime rule to get wrong.

The hazard that is left is CUDA-graph capture, and it is the reason for the
rest. A block used inside a graph must be re-zeroed *inside that same graph* on
every replay; if its fill happened before capture instead, the second replay
reduces into the first replay's result and reports an amax that is too large --
silently, and only for inputs whose amax fell. Three rules give that:

* a block is opened **lazily**, at the first :func:`alloc` that needs one, so
  its fill is a node of whatever graph the first consumer is captured into, on
  that consumer's stream (a block opened during capture is a graph-pool
  allocation at a fixed address, which is exactly what a replayed graph wants);
* a region is keyed by the stream's **capture id**
  (``cudaStreamGetCaptureInfo``), which is unique per capture and shared by
  every stream forked inside that capture, so a capture can never inherit a
  block filled outside it, and a second graph captured back-to-back with the
  first can never inherit the first one's;
* :func:`reset` opens a region at a chosen point -- called from
  ``FluxModel.forward``, so one iteration is one region and one block.

A block's fill covers the whole block, so **every consumer of a block must be
ordered behind the** :func:`alloc` **that opened it**. Production satisfies that
by construction -- the first cast of the step runs in the stem on the capture
stream, and every later stream forks from it. Nothing here can *prove* that of
an arbitrary caller, so it is not assumed: an :func:`alloc` from a stream other
than the one that opened the block is served the old way, with a private
``torch.zeros(2)``.

Failure is never silent. That private fallback also serves the case where the
capture id cannot be read, so a second capture might inherit the first one's
block: then any captured region no :func:`reset` opened is refused. Both are
today's shipped behaviour -- correct, merely not faster -- and counted in
:func:`stats`.

Environment:
    ``FLAT_AMAX_ARENA=0`` restores the per-call ``torch.zeros(2)`` (the arm the
    measurement pairs against; not a fallback -- it
    changes nothing but the allocation).
    ``FLAT_AMAX_ARENA_SLOTS`` sets the block size in 2-float slots (default
    1024; production issues 266 casts per iteration).
"""

from __future__ import annotations

import ctypes
import os

import torch

#: 2-float slots. Production issues 266 casts per iteration; 1024 slots is 8 KB.
SLOTS = int(os.environ.get("FLAT_AMAX_ARENA_SLOTS", "1024"))
#: False restores the per-call ``torch.zeros(2)``.
ENABLED = os.environ.get("FLAT_AMAX_ARENA", "1") == "1"


def _load_capture_info():
    """Bind ``cudaStreamGetCaptureInfo``, or return ``None`` if it cannot be found."""
    for lib in ("libcudart.so", "libcudart.so.13", "libcudart.so.12"):
        try:
            handle = ctypes.CDLL(lib)
        except OSError:
            continue
        for sym in ("cudaStreamGetCaptureInfo_v2", "cudaStreamGetCaptureInfo"):
            try:
                return getattr(handle, sym)
            except AttributeError:
                continue
    return None


_capture_info = _load_capture_info()


def _capture_id(device: torch.device) -> int | None:
    """The current stream's capture id: 0 when not capturing, ``None`` if unreadable.

    Args:
        device: The device whose current stream is queried.

    Returns:
        A unique id per capture, 0 outside capture, or ``None`` when
        ``cudaStreamGetCaptureInfo`` is unavailable or fails.
    """
    if _capture_info is None:
        return None
    stream = torch.cuda.current_stream(device).cuda_stream
    status = ctypes.c_int(0)
    ident = ctypes.c_ulonglong(0)
    rc = _capture_info(
        ctypes.c_void_p(stream), ctypes.byref(status), ctypes.byref(ident),
        None, None, None,
    )
    if rc != 0:
        return None
    return int(ident.value) if status.value != 0 else 0


class _Arena:
    """One device's block: the buffer it bumps through and the region it serves."""

    __slots__ = ("buf", "cursor", "region", "trusted", "stream", "high_water",
                 "blocks", "offstream", "private")

    def __init__(self) -> None:
        self.buf: torch.Tensor | None = None
        self.cursor = 0
        self.region: tuple | None = None
        self.trusted = True
        self.stream = 0
        self.high_water = 0
        self.blocks = 0
        self.offstream = 0
        self.private = 0


_arenas: dict[int, _Arena] = {}
#: Bumped by :func:`reset`; part of the region key, so a reset always opens one.
_generation = 0
#: Whether the last :func:`reset` ran on a capturing stream.
_reset_capturing = False


def reset() -> None:
    """Open a new region: the next :func:`alloc` opens a fresh, zeroed block.

    Host-only -- it issues no CUDA work itself, so it is safe to call anywhere,
    including immediately inside a capture. The block, and so the fill, lands at
    the first :func:`alloc` that follows, which is what puts the fill inside the
    graph.

    ``FluxModel.forward`` calls this, so one production step, one eval step and
    one benchmark iteration are each exactly one region and one block.
    """
    global _generation, _reset_capturing
    _generation += 1
    _reset_capturing = torch.cuda.is_current_stream_capturing()


def alloc(device: torch.device) -> torch.Tensor:
    """A zeroed contiguous FP32 ``[2]`` amax accumulator on ``device``.

    The slot is never handed out again: it stays valid for as long as the
    caller (or a quantized tensor built on it) holds it.

    Args:
        device: The device the cast runs on.

    Returns:
        A slice of the current block, or a private ``torch.zeros(2)`` where the
        block cannot be proved correct (see the module docstring): a stream
        that did not open the block, or -- on a build that cannot read capture
        ids -- a captured region no :func:`reset` opened.
    """
    if not ENABLED:
        return torch.zeros(2, dtype=torch.float32, device=device)
    capturing = torch.cuda.is_current_stream_capturing()
    arena = _arenas.get(device.index)
    if arena is None:
        arena = _Arena()
        _arenas[device.index] = arena
    capture = _capture_id(device)
    stream = torch.cuda.current_stream(device).cuda_stream
    region = (capture, capturing, _generation)
    if region != arena.region:
        arena.region = region
        arena.cursor = SLOTS               # force a new block below
        arena.trusted = (
            capture is not None or not capturing or _reset_capturing
        )
    if not arena.trusted:
        arena.private += 1
        return torch.zeros(2, dtype=torch.float32, device=device)
    if arena.cursor >= SLOTS:
        # One fill for the next SLOTS casts. Inside a capture this is a
        # graph-pool allocation, so the replayed graph re-zeroes the same
        # address; outside one it is an ordinary tensor, freed when the last
        # slice of it dies. The fill is the DSL's ``flux.zero_f32_ctm``:
        # the ``at::native`` FillFunctor was the model-side
        # graph's last third-party launch.
        arena.buf = torch.empty(SLOTS, 2, dtype=torch.float32, device=device)
        torch.ops.flux.zero_f32_ctm(arena.buf)
        arena.cursor = 0
        arena.stream = stream
        arena.blocks += 1
    if stream != arena.stream:
        # The block's fill was issued on another stream, and nothing here can
        # prove this one is ordered behind it. Serve the old way.
        arena.offstream += 1
        arena.private += 1
        return torch.zeros(2, dtype=torch.float32, device=device)
    slot = arena.buf[arena.cursor]
    arena.cursor += 1
    arena.high_water = max(arena.high_water, arena.cursor)
    return slot


def stats(device: torch.device | None = None) -> dict[str, int | bool]:
    """Counters for tests and for the bench: what the arena actually did.

    Args:
        device: The device to report; defaults to the current one.

    Returns:
        ``{"slots", "cursor", "high_water", "blocks", "private", "offstream",
        "trusted", "capture_id_readable"}`` -- ``blocks`` is the number of fills
        issued, ``private`` the calls served the old way.
    """
    index = device.index if device is not None and device.index is not None \
        else torch.cuda.current_device()
    arena = _arenas.get(index)
    return {
        "slots": SLOTS,
        "cursor": arena.cursor if arena else 0,
        "high_water": arena.high_water if arena else 0,
        "blocks": arena.blocks if arena else 0,
        "private": arena.private if arena else 0,
        "offstream": arena.offstream if arena else 0,
        "trusted": arena.trusted if arena else True,
        "capture_id_readable": _capture_info is not None,
    }
