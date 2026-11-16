
"""Pinned batch storage for one asynchronous H2D transfer.

Packing runs in DataLoader's pin-memory thread. KJT metadata is computed on
the CPU and passed through its public constructor when rebuilding device
views, so rebuilding a batch does not introduce device-to-host readbacks.
"""

import os
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Tuple

import torch
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.streamable import Pipelineable


class _TensorSlice(NamedTuple):
    field: str
    component: Optional[str]
    dtype: torch.dtype
    shape: torch.Size
    offset: int
    nbytes: int


class PackedBatch(Pipelineable):
    """Batch with tensor/KJT attributes and optional packed pinned storage.

    ``to`` mutates and returns the batch, matching the existing batch API.
    Unpinned batches and PACKED_H2D=0 use the per-field copy path. Both paths
    default to non_blocking=True and honor an explicit caller override.
    """

    _packed_buffer: Optional[torch.Tensor] = None

    def _tensor_fields(self) -> Iterator[Tuple[str, Any]]:
        for name, value in vars(self).items():
            if not name.startswith("_packed_") and isinstance(
                value, (torch.Tensor, KeyedJaggedTensor)
            ):
                yield name, value

    def _restore_views(self, buffer: torch.Tensor) -> None:
        kjt_tensors: Dict[str, Dict[str, torch.Tensor]] = {}
        for field, component, dtype, shape, offset, nbytes in self._packed_layout:
            view = buffer[offset : offset + nbytes].view(dtype).view(shape)
            if component is None:
                setattr(self, field, view)
            else:
                kjt_tensors.setdefault(field, {})[component] = view
        for field, tensors in kjt_tensors.items():
            metadata = dict(self._packed_kjt_metadata[field])
            inverse_keys = metadata.pop("inverse_keys")
            inverse = tensors.pop("inverse_indices", None)
            setattr(
                self,
                field,
                KeyedJaggedTensor(
                    **metadata,
                    **tensors,
                    inverse_indices=(
                        (inverse_keys, inverse) if inverse is not None else None
                    ),
                ),
            )
        self._packed_buffer = buffer

    def pin_memory(self) -> "PackedBatch":
        if self._packed_buffer is not None and self._packed_buffer.is_pinned():
            return self
        if os.environ.get("PACKED_H2D", "1") == "0":
            for name, value in self._tensor_fields():
                setattr(self, name, value.pin_memory())
            return self

        leaves: List[Tuple[str, Optional[str], torch.Tensor]] = []
        self._packed_kjt_metadata: Dict[str, Dict[str, Any]] = {}
        for name, value in self._tensor_fields():
            if isinstance(value, torch.Tensor):
                leaves.append((name, None, value))
                continue
            inverse = value.inverse_indices_or_none()
            self._packed_kjt_metadata[name] = {
                "keys": value.keys(),
                "stride": None if value.variable_stride_per_key() else value.stride(),
                "stride_per_key_per_rank": (
                    value.stride_per_key_per_rank()
                    if value.variable_stride_per_key()
                    else None
                ),
                "stride_per_key": value.stride_per_key(),
                "length_per_key": value.length_per_key(),
                "offset_per_key": value.offset_per_key(),
                "lengths_offset_per_key": value.lengths_offset_per_key(),
                "index_per_key": value.index_per_key(),
                "inverse_keys": inverse[0] if inverse is not None else None,
            }
            for component, tensor in (
                ("values", value.values()),
                ("lengths", value.lengths_or_none()),
                ("offsets", value.offsets_or_none()),
                ("weights", value.weights_or_none()),
                ("inverse_indices", inverse[1] if inverse is not None else None),
            ):
                if tensor is not None:
                    leaves.append((name, component, tensor))

        layout: List[_TensorSlice] = []
        size = 0
        for name, component, tensor in leaves:
            if tensor.device.type != "cpu":
                raise ValueError("PackedBatch.pin_memory requires CPU inputs")
            nbytes = tensor.numel() * tensor.element_size()
            layout.append(
                _TensorSlice(name, component, tensor.dtype, tensor.shape, size, nbytes)
            )
            size += (nbytes + 63) // 64 * 64
        buffer = torch.empty(size, dtype=torch.uint8, pin_memory=True)
        for (_, _, tensor), entry in zip(leaves, layout):
            buffer[entry.offset : entry.offset + entry.nbytes].view(entry.dtype).view(
                entry.shape
            ).copy_(tensor)
        self._packed_layout = layout
        self._restore_views(buffer)
        return self

    def to(self, device: torch.device, non_blocking: bool = True) -> "PackedBatch":
        if self._packed_buffer is not None:
            self._restore_views(
                self._packed_buffer.to(device=device, non_blocking=non_blocking)
            )
        else:
            for name, value in self._tensor_fields():
                setattr(self, name, value.to(device=device, non_blocking=non_blocking))
        return self

    def record_stream(self, stream: torch.Stream) -> None:
        for _, value in self._tensor_fields():
            value.record_stream(stream)
        if self._packed_buffer is not None:
            self._packed_buffer.record_stream(stream)
