# Copyright (c) Meta Platforms, Inc. and affiliates.
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

# pyre-unsafe

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from generative_recommenders.compacted_forward.config import (
    content_feature_pairs,
)
from generative_recommenders.dlrm_v4.datasets.dataset import (
    kjt_batch_func,
)
from generative_recommenders.modules.dlrm_hstu import DlrmHSTUConfig
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.streamable import Pipelineable


_COMPACTED_INDEX_DTYPE = torch.int32


@dataclass
class CompactedSamples(Pipelineable):
    """Batch representation used by the compacted preprocessor."""

    payloads_hist_candidate_kjt: KeyedJaggedTensor
    payload_feature_to_merged_key: Dict[str, str]
    max_uih_len: int
    uih_lengths: torch.Tensor
    max_num_candidates: int
    num_candidates: torch.Tensor
    uih_offsets: torch.Tensor
    candidate_offsets: torch.Tensor
    candidate_positions: torch.Tensor
    compacted_keys: KeyedJaggedTensor
    reverse_indices: torch.Tensor
    additional_row_indices: Optional[torch.Tensor]
    additional_reverse_indices: Optional[torch.Tensor]
    candidate_indices: torch.Tensor

    def to(
        self, device: torch.device, non_blocking: bool = False
    ) -> "CompactedSamples":
        for name, value in vars(self).items():
            if value is None or isinstance(value, (dict, int)):
                continue
            setattr(
                self,
                name,
                value.to(device=device, non_blocking=non_blocking),
            )
        return self

    def record_stream(self, stream: torch.Stream) -> None:
        for name, value in vars(self).items():
            if value is None or isinstance(value, (dict, int)):
                continue
            value.record_stream(stream)

    def pin_memory(self) -> "CompactedSamples":
        for name, value in vars(self).items():
            if value is None or isinstance(value, (dict, int)):
                continue
            setattr(self, name, value.pin_memory())
        return self

    def batch_size(self) -> int:
        return self.payloads_hist_candidate_kjt.stride()


def _balanced_lengths(
    rows: int,
    batch_size: int,
    prototype: torch.Tensor,
) -> torch.Tensor:
    """Spread compacted rows over the batch stride for TorchRec sharding."""

    per_batch, remainder = divmod(rows, batch_size)
    lengths = torch.full(
        (batch_size,),
        per_batch,
        dtype=prototype.dtype,
        device=prototype.device,
    )
    if remainder:
        lengths[:remainder] += 1
    return lengths


def _sequence_positions(
    history_lengths: torch.Tensor,
    candidate_lengths: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return destinations for per-sample ``history + candidate`` packing."""

    if history_lengths.shape != candidate_lengths.shape:
        raise ValueError("history and candidate lengths must have the same shape")
    history_lengths_i64 = history_lengths.to(torch.int64)
    candidate_lengths_i64 = candidate_lengths.to(torch.int64)
    sequence_lengths = history_lengths + candidate_lengths
    sequence_offsets = torch.cat(
        [
            sequence_lengths.new_zeros((1,), dtype=torch.int64),
            sequence_lengths.to(torch.int64).cumsum(0),
        ]
    )

    def positions(lengths: torch.Tensor, starts: torch.Tensor) -> torch.Tensor:
        offsets = torch.cat([lengths.new_zeros((1,)), lengths.cumsum(0)])
        batches = torch.repeat_interleave(
            torch.arange(lengths.numel(), device=lengths.device), lengths
        )
        local = torch.arange(
            int(offsets[-1].item()), device=lengths.device
        ) - torch.repeat_interleave(offsets[:-1], lengths)
        return starts[batches] + local

    history_positions = positions(history_lengths_i64, sequence_offsets[:-1])
    candidate_positions = positions(
        candidate_lengths_i64,
        sequence_offsets[:-1] + history_lengths_i64,
    )
    return history_positions, candidate_positions


def _compact_rows(
    rows: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Deduplicate aligned compacted-row tuples."""

    if rows.dim() != 2 or rows.size(1) == 0:
        raise ValueError("joint routing requires a non-empty row width")
    if rows.size(0) == 0:
        return (
            rows.to(_COMPACTED_INDEX_DTYPE),
            rows.new_empty((0,), dtype=_COMPACTED_INDEX_DTYPE),
        )
    if rows.size(1) == 2:
        values = rows.to(torch.int64)
        packed = torch.bitwise_or(
            torch.bitwise_left_shift(values[:, 0], 32), values[:, 1]
        )
        unique, reverse = torch.unique(packed, sorted=True, return_inverse=True)
        indices = torch.stack(
            [
                torch.bitwise_and(
                    torch.bitwise_right_shift(unique, 32), 0xFFFFFFFF
                ),
                torch.bitwise_and(unique, 0xFFFFFFFF),
            ],
            dim=1,
        )
    else:
        indices, reverse = torch.unique(
            rows.to(torch.int64), sorted=True, return_inverse=True, dim=0
        )
    reverse = reverse.to(_COMPACTED_INDEX_DTYPE)
    return (
        indices.to(_COMPACTED_INDEX_DTYPE),
        reverse,
    )


def _compact_sparse_inputs(
    uih_features: KeyedJaggedTensor,
    candidate_features: KeyedJaggedTensor,
    hstu_config: "DlrmHSTUConfig",
    history_positions: torch.Tensor,
    candidate_positions: torch.Tensor,
) -> Tuple[
    KeyedJaggedTensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    torch.Tensor,
]:
    """Compact complete per-sample sequences and preserve candidate routing."""

    if uih_features.stride() != candidate_features.stride():
        raise ValueError("history and candidate KJTs must have the same stride")
    if (
        uih_features.weights_or_none() is not None
        or candidate_features.weights_or_none() is not None
    ):
        raise ValueError("compacted sparse inputs do not support weighted KJTs")

    pairs = content_feature_pairs(hstu_config)
    sequence_feature = hstu_config.uih_post_id_feature_name
    sequence_candidate_feature = [
        candidate
        for history, candidate in pairs
        if history == sequence_feature
    ][0]
    history_lengths = uih_features[sequence_feature].lengths()
    candidate_lengths = candidate_features[sequence_candidate_feature].lengths()
    for history_feature, candidate_feature in pairs:
        if not torch.equal(
            uih_features[history_feature].lengths(), history_lengths
        ) or not torch.equal(
            candidate_features[candidate_feature].lengths(),
            candidate_lengths,
        ):
            raise ValueError("content features must share one sequence layout")

    batch_size = uih_features.stride()
    lookup_keys: List[str] = []
    lookup_values: List[torch.Tensor] = []
    lookup_lengths: List[torch.Tensor] = []

    for key in hstu_config.contextual_feature_to_max_length.keys():
        values = uih_features[key].values()
        lengths = uih_features[key].lengths()
        lookup_keys.append(key)
        lookup_values.append(values)
        lookup_lengths.append(lengths)

    sequence_reverse: Dict[str, torch.Tensor] = {}
    candidate_reverse: List[torch.Tensor] = []
    for history_feature, candidate_feature in pairs:
        history = uih_features[history_feature].values()
        candidate = candidate_features[candidate_feature].values()
        sequence = history.new_empty(history.numel() + candidate.numel())
        sequence[history_positions] = history
        sequence[candidate_positions] = candidate
        unique, reverse = torch.unique(sequence, sorted=True, return_inverse=True)
        reverse = reverse.to(_COMPACTED_INDEX_DTYPE)
        sequence_reverse[history_feature] = reverse
        candidate_reverse.append(reverse[candidate_positions])
        lookup_keys.append(history_feature)
        lookup_values.append(unique)
        lookup_lengths.append(
            _balanced_lengths(unique.numel(), batch_size, history_lengths)
        )

    additional_features = [
        history
        for history, _ in pairs
        if history != sequence_feature
    ]
    additional_compaction = (
        _compact_rows(
            torch.stack(
                [sequence_reverse[feature] for feature in additional_features],
                dim=1,
            ),
        )
        if additional_features
        else None
    )
    additional_row_indices = (
        additional_compaction[0] if additional_compaction is not None else None
    )
    additional_reverse_indices = (
        additional_compaction[1] if additional_compaction is not None else None
    )
    candidate_indices = torch.stack(candidate_reverse, dim=1)

    return (
        KeyedJaggedTensor.from_lengths_sync(
            keys=lookup_keys,
            values=torch.cat(lookup_values),
            lengths=torch.cat(lookup_lengths),
        ),
        sequence_reverse[sequence_feature],
        additional_row_indices,
        additional_reverse_indices,
        candidate_indices,
    )

def collate_compacted(
    samples: List[Tuple[KeyedJaggedTensor, KeyedJaggedTensor]],
    hstu_config: DlrmHSTUConfig,
    is_inference: bool = False,
) -> CompactedSamples:
    """Collate embedding keys once and retain occurrence routing."""

    uih_list, candidate_list = list(zip(*samples))
    uih_features = kjt_batch_func(uih_list)
    candidate_features = kjt_batch_func(candidate_list)
    pairs = content_feature_pairs(hstu_config)
    sequence_feature = hstu_config.uih_post_id_feature_name
    sequence_candidate_feature = [
        candidate for history, candidate in pairs if history == sequence_feature
    ][0]
    uih_lengths = uih_features[sequence_feature].lengths()
    num_candidates = candidate_features[sequence_candidate_feature].lengths()
    history_positions, candidate_positions = _sequence_positions(
        uih_lengths, num_candidates
    )
    sequence_lengths = uih_lengths + num_candidates
    uih_offsets = torch.cat(
        [
            uih_lengths.new_zeros((1,)),
            torch.cumsum(uih_lengths, dim=0, dtype=uih_lengths.dtype),
        ]
    )
    candidate_offsets = torch.cat(
        [
            num_candidates.new_zeros((1,)),
            torch.cumsum(num_candidates, dim=0, dtype=num_candidates.dtype),
        ]
    )
    (
        compacted_keys,
        reverse_indices,
        additional_row_indices,
        additional_reverse_indices,
        candidate_indices,
    ) = _compact_sparse_inputs(
        uih_features=uih_features,
        candidate_features=candidate_features,
        hstu_config=hstu_config,
        history_positions=history_positions,
        candidate_positions=candidate_positions,
    )
    payload_pairs = [
        (history, candidate)
        for history, candidate in hstu_config.merge_uih_candidate_feature_mapping
        if candidate not in hstu_config.item_embedding_feature_names
        and history not in hstu_config.user_embedding_feature_names
    ]
    zero_candidate_keys = (
        {
            hstu_config.candidates_weight_feature_name,
            hstu_config.candidates_watchtime_feature_name,
        }
        if is_inference
        else set()
    )
    payload_keys: List[str] = []
    payload_values: List[torch.Tensor] = []
    payload_feature_to_merged_key: Dict[str, str] = {}
    for history_key, candidate_key in payload_pairs:
        history = uih_features[history_key].values()
        candidate = candidate_features[candidate_key].values()
        if candidate_key in zero_candidate_keys:
            candidate = torch.zeros_like(candidate)
        sequence = history.new_empty(history.numel() + candidate.numel())
        sequence[history_positions] = history
        sequence[candidate_positions] = candidate
        payload_keys.append(history_key)
        payload_values.append(sequence)
        payload_feature_to_merged_key[history_key] = history_key
        payload_feature_to_merged_key[candidate_key] = history_key
    payloads_hist_candidate_kjt = KeyedJaggedTensor.from_lengths_sync(
        keys=payload_keys,
        values=torch.cat(payload_values),
        lengths=torch.cat([sequence_lengths for _ in payload_keys]),
    )
    return CompactedSamples(
        payloads_hist_candidate_kjt=payloads_hist_candidate_kjt,
        payload_feature_to_merged_key=payload_feature_to_merged_key,
        max_uih_len=int(uih_lengths.max()),
        uih_lengths=uih_lengths,
        max_num_candidates=int(num_candidates.max()),
        num_candidates=num_candidates,
        uih_offsets=uih_offsets,
        candidate_offsets=candidate_offsets,
        candidate_positions=candidate_positions,
        compacted_keys=compacted_keys,
        reverse_indices=reverse_indices,
        additional_row_indices=additional_row_indices,
        additional_reverse_indices=additional_reverse_indices,
        candidate_indices=candidate_indices,
    )
