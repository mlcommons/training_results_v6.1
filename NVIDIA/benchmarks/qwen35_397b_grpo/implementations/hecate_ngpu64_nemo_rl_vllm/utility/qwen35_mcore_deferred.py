#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Save native GRPO checkpoints, refit vLLM, and report deferred pass@4."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any

import yaml


def configure_deferred_evaluation(config: dict) -> dict | None:
    """Save native first/final checkpoints while leaving ordinary training alone."""
    from utility.qwen35_local_checkpoint import WORKER_FQN

    grpo = config["grpo"]
    deferred = dict(grpo.get("deferred_evaluation") or {})
    enabled = deferred.get("enabled", False)
    if not isinstance(enabled, (bool, int)) or enabled not in (0, 1):
        raise ValueError("deferred_evaluation.enabled must be a boolean or 0/1")
    launch_mode = os.environ.get("OFFLINE_EVAL")
    if launch_mode is not None and bool(enabled) != (launch_mode == "1"):
        raise ValueError(
            "grpo.deferred_evaluation.enabled must match the OFFLINE_EVAL launcher flag"
        )
    local_directory = os.environ.get("GRPO_LOCAL_CHECKPOINT_DIR")
    worker = config.get("policy", {}).get("worker_extension_cls_fqn")
    if worker == WORKER_FQN and not local_directory:
        raise ValueError("local checkpoint worker requires GRPO_LOCAL_CHECKPOINT_DIR")
    if not enabled:
        if local_directory:
            raise ValueError("local checkpoints require deferred evaluation")
        return None
    first = grpo.get("val_start_at")
    if type(first) is not int or first < 1:
        raise ValueError("grpo.val_start_at must be a positive integer")
    if not grpo.get("async_grpo", {}).get("enabled"):
        raise ValueError("deferred evaluation requires async GRPO")
    if config["policy"]["generation"]["backend"] != "vllm":
        raise ValueError("deferred evaluation requires vLLM generation")
    if not config["policy"].get("megatron_cfg", {}).get("enabled"):
        raise ValueError("deferred evaluation requires an MCore policy")
    if config.get("env", {}).get("nemo_gym", {}).get("is_trajectory_collection"):
        raise ValueError("deferred evaluation cannot collect trajectories only")
    if (
        grpo.get("val_num_generations_per_prompt") != 4
        or grpo.get("stop_at_validation_metric") != "pass_k"
    ):
        raise ValueError("deferred evaluation requires pass@4 validation")
    threshold = grpo.get("stop_at_validation_threshold")
    if threshold is None:
        threshold = config.get("logger", {}).get("mlperf", {}).get("target_accuracy")
    if type(threshold) not in (int, float) or not 0 <= threshold <= 1:
        raise ValueError("deferred evaluation requires a threshold in [0, 1]")
    checkpointing = config["checkpointing"]
    if (
        checkpointing.get("checkpoint_must_save_by") is not None
        or checkpointing.get("ft_save_period") is not None
    ):
        raise ValueError(
            "deferred evaluation requires checkpoint_must_save_by and ft_save_period to be unset"
        )
    root = Path(checkpointing["checkpoint_dir"])
    if root.exists() and any(root.iterdir()):
        raise ValueError(
            f"deferred evaluation requires a fresh checkpoint directory: {root}"
        )
    if local_directory:
        local_root = Path(local_directory)
        if not local_root.is_absolute() or not local_root.is_dir():
            raise ValueError(f"local checkpoint mount is unavailable: {local_root}")
        if worker not in (None, WORKER_FQN):
            raise ValueError(f"local checkpoint worker conflicts with {worker}")

    deferred.update(
        enabled=True,
        threshold=float(threshold),
    )
    grpo.update(
        deferred_evaluation=deferred,
        max_num_steps=first + 1,
        val_period=0,
        val_at_start=False,
        val_at_end=False,
        stop_at_validation_metric=None,
        stop_at_validation_threshold=None,
    )
    checkpointing.update(
        enabled=True,
        metric_name=None,
        save_optimizer=False,
        save_period=first,
        keep_top_k=max(checkpointing.get("keep_top_k") or 0, 2),
    )
    config["policy"]["megatron_cfg"].setdefault("checkpoint", {})["async_save"] = True
    if local_directory:
        config["policy"]["worker_extension_cls_fqn"] = WORKER_FQN
        deferred["local_checkpoint_directory"] = local_directory
        config["policy"]["megatron_cfg"].setdefault("env_vars", {})[
            "GRPO_LOCAL_CHECKPOINT_DIR"
        ] = local_directory
    return deferred


@contextmanager
def timestamp_policy_training(policy: Any, grpo_state: Any):
    """Persist each successful update's end time in native training_info.json."""
    missing = object()
    previous = vars(policy).get("train", missing)
    train = policy.train

    @wraps(train)
    def timed_train(*args, **kwargs):
        result = train(*args, **kwargs)
        grpo_state.training_step_end_time_ms = time.time_ns() // 1_000_000
        return result

    policy.train = timed_train
    try:
        yield
    finally:
        if previous is missing:
            delattr(policy, "train")
        else:
            policy.train = previous


def wait_for_ray_resources_released(
    ray_module, placement_groups=(), *, timeout_seconds=300
):
    """Wait for asynchronous actor termination and placement-group removal."""
    expected = ray_module.cluster_resources().get("GPU", 0)
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        states = [
            ray_module.util.placement_group_table(group).get("state", "REMOVED")
            for group in placement_groups
        ]
        available = ray_module.available_resources().get("GPU", 0)
        if available >= expected and all(
            state in {"REMOVED", "DEAD"} for state in states
        ):
            return
        time.sleep(1)
    raise RuntimeError(
        f"Ray resources still occupied: GPUs {available}/{expected}, placement groups {states}"
    )


def shutdown_resources(nemo_gym, generation, policy, clusters, checkpointer):
    """Flush saves and release actors before another phase reuses this allocation."""
    import ray

    active_error = sys.exc_info()[1]
    if all(
        resource is None
        for resource in (nemo_gym, generation, policy, clusters, checkpointer)
    ):
        return
    clusters = list({id(cluster): cluster for cluster in (clusters or ())}.values())
    placement_groups = [
        group
        for cluster in clusters
        for group in (getattr(cluster, "_node_placement_groups", None) or ())
    ]
    operations = []
    if checkpointer is not None:
        operations.append(checkpointer.shutdown)
    if nemo_gym is not None:
        # The shard-set owner shuts down its actors and releases their placement group.
        operations.append(nemo_gym.shutdown)
    for resource in {
        id(resource): resource
        for resource in (generation, policy)
        if resource is not None
    }.values():
        operations.append(resource.shutdown)
    operations.extend(cluster.shutdown for cluster in clusters)
    operations.append(lambda: wait_for_ray_resources_released(ray, placement_groups))
    failures = []
    for operation in operations:
        try:
            if operation() is False:
                raise RuntimeError("shutdown returned False")
        except Exception as error:
            failures.append(str(error))
    if failures:
        message = "deferred resource teardown failed: " + "; ".join(failures)
        if active_error is not None:
            print(f"Warning: {message}; preserving original error: {active_error}", flush=True)
        else:
            raise RuntimeError(message)


def read_checkpoint(checkpoint: Path, step: int) -> tuple[dict, dict]:
    with (checkpoint / "config.yaml").open() as stream:
        config = yaml.safe_load(stream)
    with (checkpoint / "training_info.json").open() as stream:
        info = json.load(stream)
    if info["total_steps"] != step:
        raise ValueError(f"{checkpoint}: training_info.total_steps differs from {step}")
    if info["training_step_end_time_ms"] <= 0 or info["consumed_samples"] <= 0:
        raise ValueError(f"{checkpoint}: missing training timestamp or sample count")
    if not (checkpoint / "policy" / "weights").is_dir():
        raise FileNotFoundError(f"{checkpoint}: MCore weights directory is missing")
    if config["grpo"]["val_num_generations_per_prompt"] != 4:
        raise ValueError("deferred evaluation requires four generations per prompt")
    return config, info


def evaluation_config(config: dict, checkpoint: Path, output: Path) -> dict:
    """Keep the saved policy/generation topology and enable validation only."""
    raw = copy.deepcopy(config)
    # Evaluation restores weights but never saves; do not spawn an unused writer.
    raw["policy"]["megatron_cfg"].setdefault("checkpoint", {})["async_save"] = False
    raw["logger"].update(
        mlperf_enabled=False,
        wandb_enabled=False,
        tensorboard_enabled=False,
        log_dir=str(output / "runtime"),
    )
    raw["grpo"].update(
        val_period=1,
        val_at_start=False,
        val_at_end=False,
        stop_at_validation_metric=None,
        stop_at_validation_threshold=None,
    )
    # setup() selects the latest checkpoint. A private, single-entry root makes
    # its ordinary restore path select the requested endpoint without copying.
    view = output / "checkpoint-view"
    view.mkdir(parents=True)
    (view / checkpoint.name).symlink_to(checkpoint.resolve())
    raw["checkpointing"].update(
        checkpoint_dir=str(view),
        enabled=False,
        save_optimizer=False,
        metric_name=None,
    )
    return raw


def policy_without_optimizer(**kwargs: Any) -> Any:
    from nemo_rl.models.policy.lm_policy import Policy
    from utility.qwen35_local_checkpoint import register_local_checkpoint_worker

    kwargs.update(init_optimizer=False, optimizer_path=None, init_reference_model=False)
    register_local_checkpoint_worker(kwargs.get("config", {}))
    return Policy(**kwargs)


def log_phase_timing(config, step, phase, started):
    """Operational phase timings, separate from backdated benchmark MLLOG time."""
    if config["grpo"]["deferred_evaluation"].get("local_checkpoint_directory"):
        print("QWEN35_DEFERRED_PHASE_TIMING " + json.dumps({
            "step": step, "phase": phase, "seconds": time.perf_counter() - started,
        }), flush=True)


def evaluate_checkpoint(
    checkpoint: Path,
    config: dict,
    step: int,
    output: Path,
    checkpoint_time_ms: int,
    mlperf_logger: Any,
) -> float:
    import numpy as np
    import torch

    from nemo_rl.algorithms.grpo import (
        MasterConfig,
        refit_policy_generation,
        setup,
        validate,
    )
    from nemo_rl.algorithms.utils import get_tokenizer
    from nemo_rl.data.utils import setup_response_data

    seed = int(config["grpo"]["seed"])
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    master_config = MasterConfig(**evaluation_config(config, checkpoint, output))
    tokenizer = get_tokenizer(master_config.policy["tokenizer"])
    train_dataset, val_dataset = setup_response_data(
        tokenizer, master_config.data, env_configs=None
    )
    policy = generation = nemo_gym = checkpointer = clusters = None
    try:
        started = time.perf_counter()
        (
            policy,
            generation,
            nemo_gym,
            clusters,
            _dataloader,
            val_dataloader,
            _loss_fn,
            _logger,
            checkpointer,
            _state,
            materialized_config,
            _teachers,
            _aliases,
        ) = setup(
            master_config,
            tokenizer,
            train_dataset,
            val_dataset,
            policy_factory=policy_without_optimizer,
        )
        log_phase_timing(config, step, "setup_including_restore", started)
        if val_dataloader is None:
            raise RuntimeError("setup did not create a validation dataloader")
        started = time.perf_counter()
        refit_policy_generation(
            policy,
            generation,
            bool(materialized_config.policy["generation"]["colocated"]["enabled"]),
        )
        if not generation.invalidate_kv_cache():
            raise RuntimeError("vLLM cache invalidation failed after MCore refit")
        log_phase_timing(config, step, "refit_and_cache_invalidation", started)

        mlperf_logger.start_eval(step)
        started = time.perf_counter()
        metrics, timings = validate(
            generation,
            val_dataloader,
            tokenizer,
            {"nemo_gym": nemo_gym},
            step,
            materialized_config,
        )
        log_phase_timing(config, step, "validation", started)
        accuracy = float(metrics["pass_k"])
        if not 0 <= accuracy <= 1:
            raise ValueError(f"invalid validation pass@4: {accuracy}")
        # Record the result and any terminal event before fallible teardown.
        mlperf_logger.end_eval(
            step, metrics, timings, checkpoint_time_ms=checkpoint_time_ms
        )
        print(f"Deferred evaluation step {step}: pass@4={accuracy}", flush=True)
        return accuracy
    finally:
        started = time.perf_counter()
        shutdown_resources(nemo_gym, generation, policy, clusters, checkpointer)
        log_phase_timing(config, step, "teardown", started)


def evaluate_endpoints(
    checkpoint_root: Path, log_dir: Path, config: dict, mlperf_logger: Any
) -> None:
    grpo = config["grpo"]
    first, last = grpo["val_start_at"], grpo["max_num_steps"]
    mlperf_logger.target_accuracy = float(grpo["deferred_evaluation"]["threshold"])
    for step in (first, last) if first < last else (last,):
        checkpoint = checkpoint_root / f"step_{step}"
        saved_config, info = read_checkpoint(checkpoint, step)
        evaluate_checkpoint(
            checkpoint, saved_config, step, log_dir / f"step_{step}",
            info["training_step_end_time_ms"], mlperf_logger,
        )
        if mlperf_logger.run_stopped:
            break


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    args = parser.parse_args()
    # NeMo-RL publishes completed checkpoints by renaming tmp_step_N to step_N.
    checkpoints = sorted(
        (
            path
            for path in args.checkpoint_root.glob("step_*")
            if path.is_dir() and path.name.removeprefix("step_").isdigit()
        ),
        key=lambda path: int(path.name.removeprefix("step_")),
    )
    if not checkpoints:
        raise FileNotFoundError(f"no completed checkpoints in {args.checkpoint_root}")
    config, _info = read_checkpoint(
        checkpoints[0], int(checkpoints[0].name.removeprefix("step_"))
    )

    from nemo_rl.algorithms.mlperf_grpo_logging import MLPerfGRPOLogger
    from nemo_rl.data_plane.factory import maybe_configure_data_plane_env
    from nemo_rl.distributed.virtual_cluster import init_ray
    from nemo_rl.utils.config import register_omegaconf_resolvers

    register_omegaconf_resolvers()
    maybe_configure_data_plane_env(config.get("data_plane"))
    init_ray(log_dir=None)
    evaluate_endpoints(args.checkpoint_root, args.log_dir, config, MLPerfGRPOLogger(config))


if __name__ == "__main__":
    main()
