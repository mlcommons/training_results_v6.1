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

"""Hierarchical ZeRO-2 topology: map ranks to NVLink-domain groups."""
from dataclasses import dataclass

# Minimum inter-domain (replicate) group size, in ranks, for an IB SHARP tree.
SHARP_MIN_REPLICATE_SIZE = 4


@dataclass
class HierTopology:
    """Rank's place in the 2-level (shard x replicate) data-parallel mesh."""

    world_size: int
    rank: int
    shard_size: int
    n_domains: int
    domain_id: int
    intra_pos: int
    shard_group_ranks: list[int]
    replicate_group_ranks: list[int]
    is_hierarchical: bool
    replicate_group_size: int
    n_replicate_groups: int

    @property
    def sharp_worthwhile(self) -> bool:
        """Whether the inter-domain group is large enough for IB SHARP."""
        return self.is_hierarchical and (
            self.replicate_group_size >= SHARP_MIN_REPLICATE_SIZE
        )


def build_topology(
    world_size: int,
    rank: int,
    segment_nodes: int,
    gpus_per_node: int,
) -> HierTopology:
    """Compute the shard/replicate group membership for a rank."""
    shard_size = segment_nodes * gpus_per_node
    if shard_size <= 0 or world_size % shard_size != 0:
        raise ValueError(
            f"world_size={world_size} not divisible by shard_size="
            f"{shard_size} (segment_nodes={segment_nodes}, "
            f"gpus_per_node={gpus_per_node})"
        )
    n_domains = world_size // shard_size
    domain_id = rank // shard_size
    intra_pos = rank % shard_size
    shard_group_ranks = list(
        range(domain_id * shard_size, (domain_id + 1) * shard_size)
    )
    replicate_group_ranks = list(range(intra_pos, world_size, shard_size))
    return HierTopology(
        world_size=world_size,
        rank=rank,
        shard_size=shard_size,
        n_domains=n_domains,
        domain_id=domain_id,
        intra_pos=intra_pos,
        shard_group_ranks=shard_group_ranks,
        replicate_group_ranks=replicate_group_ranks,
        is_hierarchical=n_domains > 1,
        replicate_group_size=n_domains,
        n_replicate_groups=shard_size,
    )
