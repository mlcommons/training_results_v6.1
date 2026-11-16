# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Same-allocation relocation of rank-local checkpoints, without aggregation."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path


WORKER_FQN = "utility.qwen35_local_checkpoint_worker.LocalCheckpointPolicyWorker"
CHUNK_BYTES = 32 * 1024 * 1024


def local_endpoint_root(weights_path: str) -> Path:
    """Use separate native local-manager roots for first/final GRPO steps."""
    # Evaluation reaches the endpoint through its one-entry checkpoint-view
    # symlink. Resolve that alias before deriving the experiment directory.
    weights = Path(weights_path).resolve()
    endpoint = weights.parent.parent
    match = re.fullmatch(r"(?:tmp_)?step_(\d+)", endpoint.name)
    if weights.parts[-2:] != ("policy", "weights") or match is None:
        raise ValueError(f"not a GRPO endpoint weights path: {weights}")
    root = Path(os.environ["GRPO_LOCAL_CHECKPOINT_DIR"])
    if not root.is_absolute() or not root.is_dir():
        raise ValueError(f"local checkpoint mount is unavailable: {root}")
    return root / endpoint.parent.name / f"step_{match[1]}"


def publish_rank_location(weights_path, rank, local_path, iteration):
    """Publish only location/size metadata, after the native writer has drained."""
    import ray
    import torch.distributed as dist

    directory = Path(local_path)
    files = [
        {"name": path.name, "size": path.stat().st_size}
        for path in sorted(directory.glob("*.pt"))
        if path.is_file() and not path.is_symlink()
    ]
    if not files or any(file["size"] == 0 for file in files):
        raise RuntimeError(f"local checkpoint has no complete rank files: {directory}")
    location = {
        "rank": rank,
        "world_size": dist.get_world_size(),
        "node_id": ray.get_runtime_context().get_node_id(),
        "directory": str(directory),
        "iteration": iteration,
        "files": files,
    }
    output = Path(weights_path)
    output.mkdir(parents=True, exist_ok=True)
    with (output / f"rank_{rank}.json").open("x") as stream:
        json.dump(location, stream)


class CheckpointFileReader:
    """CPU-only actor placed on a former training node, including generation nodes."""

    def __init__(self, directory):
        self.directory = Path(directory)

    def read(self, name, offset, length):
        if Path(name).name != name or not 0 < length <= CHUNK_BYTES or offset < 0:
            raise ValueError("invalid local checkpoint read")
        with (self.directory / name).open("rb") as stream:
            stream.seek(offset)
            return stream.read(length)


def copy_rank_files(files, destination, read):
    """Bound memory per rank; never materialize a whole model in Ray's object store."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for file in files:
        name, size = file["name"], file["size"]
        if Path(name).name != name or size <= 0:
            raise ValueError("invalid local checkpoint file inventory")
        target = destination / name
        if target.exists():
            raise FileExistsError(target)
        partial = target.with_name(target.name + ".partial")
        with partial.open("xb") as stream:
            for offset in range(0, size, CHUNK_BYTES):
                length = min(CHUNK_BYTES, size - offset)
                data = read(name, offset, length)
                if len(data) != length:
                    raise IOError(f"short checkpoint read: {name} at {offset}")
                stream.write(data)
        partial.rename(target)


def fetch_rank_checkpoint(weights_path, rank, destination):
    """Move this logical rank's files directly from their original physical node."""
    import ray
    import torch.distributed as dist
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    with (Path(weights_path) / f"rank_{rank}.json").open() as stream:
        location = json.load(stream)
    if location["rank"] != rank or location["world_size"] != dist.get_world_size():
        raise ValueError(
            "local checkpoint restore requires the original policy world size"
        )
    if location["node_id"] == ray.get_runtime_context().get_node_id():
        if Path(location["directory"]) != Path(destination):
            raise ValueError("same-node checkpoint directory changed")
        for file in location["files"]:
            if (Path(destination) / file["name"]).stat().st_size != file["size"]:
                raise IOError("incomplete local checkpoint")
    else:
        if not any(
            node["Alive"] and node["NodeID"] == location["node_id"]
            for node in ray.nodes()
        ):
            raise RuntimeError(
                "checkpoint owner node is unavailable; this requires the original allocation"
            )
        reader = (
            ray.remote(CheckpointFileReader)
            .options(
                num_cpus=0,
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    location["node_id"], soft=False
                ),
            )
            .remote(location["directory"])
        )
        try:
            copy_rank_files(
                location["files"],
                destination,
                lambda name, offset, length: ray.get(
                    reader.read.remote(name, offset, length), timeout=300
                ),
            )
        finally:
            ray.kill(reader, no_restart=True)
    print(
        f"Local checkpoint rank {rank}: restored from node {location['node_id']} to {ray.get_runtime_context().get_node_id()}",
        flush=True,
    )
    return location["iteration"]


def register_local_checkpoint_worker(config):
    """Register the saved policy worker in NeMo's standard Megatron environment."""
    if config.get("worker_extension_cls_fqn") != WORKER_FQN:
        return
    from nemo_rl.distributed.ray_actor_environment_registry import (
        ACTOR_ENVIRONMENT_REGISTRY,
    )

    standard = (
        "nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker"
    )
    ACTOR_ENVIRONMENT_REGISTRY[WORKER_FQN] = ACTOR_ENVIRONMENT_REGISTRY[standard]
