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

# pyre-strict

from typing import Dict, List, TYPE_CHECKING, Tuple

from torchrec.modules.embedding_configs import EmbeddingConfig

if TYPE_CHECKING:
    from generative_recommenders.modules.dlrm_hstu import DlrmHSTUConfig


def content_feature_pairs(
    config: "DlrmHSTUConfig",
) -> List[Tuple[str, str]]:
    """Return aligned (history, candidate) features in candidate order."""

    user_features = set(config.user_embedding_feature_names)
    history_by_candidate = {
        candidate: history
        for history, candidate in config.merge_uih_candidate_feature_mapping
        if history in user_features
        and candidate in config.item_embedding_feature_names
    }
    missing = [
        candidate
        for candidate in config.item_embedding_feature_names
        if candidate not in history_by_candidate
    ]
    if missing:
        raise ValueError(
            "candidate embedding features need history partners in "
            f"merge_uih_candidate_feature_mapping: {missing}"
        )

    pairs = [
        (history_by_candidate[candidate], candidate)
        for candidate in config.item_embedding_feature_names
    ]
    if sum(history == config.uih_post_id_feature_name for history, _ in pairs) != 1:
        raise ValueError(
            "exactly one content feature must match "
            f"uih_post_id_feature_name={config.uih_post_id_feature_name!r}"
        )

    covered = set(config.contextual_feature_to_max_length.keys()).union(
        history for history, _ in pairs
    )
    unclassified = user_features.difference(covered)
    if unclassified:
        raise ValueError(
            "user embedding features must be contextual or paired content "
            f"features: {sorted(unclassified)}"
        )
    return pairs


def configure_embedding_tables(
    tables: Dict[str, EmbeddingConfig],
    config: "DlrmHSTUConfig",
) -> Dict[str, EmbeddingConfig]:
    """Make shared tables expose one lookup feature per compacted feature pair."""

    pairs = content_feature_pairs(config)
    configured = set()
    for table in tables.values():
        table_features = set(table.feature_names)
        table_pairs = [
            pair
            for pair in pairs
            if pair[0] in table_features and pair[1] in table_features
        ]
        if table_pairs:
            table.feature_names = [history for history, _ in table_pairs]
            configured.update(history for history, _ in table_pairs)

    missing = {history for history, _ in pairs}.difference(configured)
    if missing:
        raise ValueError(
            "compacted content features must map to shared embedding tables: "
            f"{sorted(missing)}"
        )
    return tables
