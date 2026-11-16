# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import argparse
import math
import os
import pprint
import time
from contextlib import nullcontext

from utility.qwen35_mcore_deferred import (
    configure_deferred_evaluation,
    shutdown_resources,
    timestamp_policy_training,
)

                                                                                                         
import wandb.util

wandb.util.VALUE_BYTES_LIMIT = 10_000_000

from omegaconf import OmegaConf
from wandb import Table

from nemo_rl.algorithms.grpo import (
    ColocatablePolicyInterface,
    EnvironmentInterface,
    GenerationInterface,
    GRPOSaveState,
    Logger,
    MasterConfig,
    StatefulDataLoader,
    TokenizerType,
    grpo_train,
    refit_policy_generation,
    setup,
)
from nemo_rl.algorithms.mlperf_grpo_logging import (
    MLPerfGRPOLogger,
    MLPerfMetricsLoggerShim,
    create_mlperf_logger,
)
from mlperf_warmup import maybe_run_mlperf_warmup
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.utils import setup_response_data
from nemo_rl.data_plane.factory import maybe_configure_data_plane_env
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.environments.nemo_gym import (
    setup_nemo_gym_config,
    should_use_nemo_gym,
)
from nemo_rl.environments.utils import shutdown_environments
from nemo_rl.experience.rollouts import run_nemo_gym_rollout_sync
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.generation.vllm.config import materialize_vllm_video_config

try:
                                                                            
                                                
    from nemo_rl.models.policy.draft_config import draft_refit_enabled
except ImportError:

    def draft_refit_enabled(draft: object) -> bool:
        return bool(draft) and (
            draft["enabled"] if isinstance(draft, dict) else draft.enabled
        )


from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.utils.logger import get_next_experiment_dir, log_container_init_timing
from nemo_rl.utils.timer import Timer


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    ''                                 
    parser = argparse.ArgumentParser(description="Run GRPO training with configuration")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )
                                     
    args, overrides = parser.parse_known_args()
    if (
        os.environ.get("RANDOM_INIT_POLICY", "0") == "1"
        or os.environ.get("PROXY_NUM_LAYERS")
        or any(
            arg.split("=", 1)[0] in ("--random-init-policy", "--proxy-num-layers")
            for arg in overrides
        )
    ):
        parser.error("Random-init proxy models are not supported by this TRT-LLM benchmark")

    return args, overrides


                                                                                                                                  
def collect_trajectories(
    policy: ColocatablePolicyInterface,
    policy_generation: GenerationInterface,
    val_dataloader: StatefulDataLoader,
    tokenizer: TokenizerType,
    val_task_to_env: dict[str, EnvironmentInterface],
    logger: Logger,
    master_config: MasterConfig,
) -> None:
    ''                              
                               
    colocated_inference = master_config.policy["generation"]["colocated"]["enabled"]
    refit_policy_generation(policy, policy_generation, colocated_inference)

    log_filename = "trajectory_collection.jsonl"

    print("\n🔍 Running trajectory collection...", flush=True)
    generation_config = master_config.policy["generation"]
    for val_batch in val_dataloader:
        nemo_gym_rollout_result = run_nemo_gym_rollout_sync(
            policy_generation=policy_generation,
            input_batch=val_batch,
            tokenizer=tokenizer,
            task_to_env=val_task_to_env,
            max_seq_len=master_config.policy["max_total_sequence_length"],
            generation_config=generation_config,
                                                                                   
            log_full_result_tables=True,
            num_generations_per_prompt=1,
            max_rollout_turns=None,
            greedy=False,
        )

        rows_to_log: list[str] = []
        for key, value in nemo_gym_rollout_result.rollout_metrics.items():
            if "full_result" not in key:
                continue

            value: Table
            data: list[list[str]] = value.data          
            rows_to_log.extend(v[0] for v in data)

        logger.log_string_list_as_jsonl(rows_to_log, log_filename)

                                                                                                                                         
                                                                    

    policy_generation.finish_generation()


def log_mlperf_run_start(
    mlperf_logger: MLPerfGRPOLogger | None,
    config: MasterConfig,
    grpo_state: GRPOSaveState,
) -> None:
    ''                                                                                       
    if mlperf_logger is None:
        return
    mlperf_logger.log_init_stop_run_start()
    start_step = int(grpo_state.total_steps)
    if not (config.grpo.val_at_start and start_step == 0):
        mlperf_logger.start_train_block(start_step)


def main() -> None:
    ''                     
    main_start = time.perf_counter()
    log_container_init_timing()
    rl_init_timer = Timer(context={"worker": "driver"})

    register_omegaconf_resolvers()
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__),
            "conf",
            "grpo_qwen35_397b_swe_openhands_async_trtllm.yaml",
        )

    with rl_init_timer.time("config"):
        config = load_config(args.config)
        print(f"Loaded configuration from: {args.config}")

        if overrides:
            print(f"Overrides: {overrides}")
            config = parse_hydra_overrides(config, overrides)

        config = OmegaConf.to_container(config, resolve=True)
        deferred_evaluation = configure_deferred_evaluation(config)
        from utility.qwen35_local_checkpoint import register_local_checkpoint_worker

        register_local_checkpoint_worker(config["policy"])

                                                                             
                                                                               
                                              
    mlperf_logger = create_mlperf_logger(config)
    mlperf_init_started = time.monotonic()
    if mlperf_logger is not None:
        mlperf_logger.log_init_start()
    mlperf_warmup_cfg = config.pop("mlperf_warmup", None) or {}
    config = MasterConfig(**config)
    materialize_vllm_video_config(config.policy, config.data)
    print("Applied CLI overrides")

                                                           
    config.logger["log_dir"] = get_next_experiment_dir(config.logger["log_dir"])
    print(f"📊 Using log directory: {config.logger['log_dir']}")
    if config.checkpointing["enabled"]:
        print(
            f"📊 Using checkpoint directory: {config.checkpointing['checkpoint_dir']}"
        )

    with rl_init_timer.time("tokenizer"):
        is_vlm = bool(config.policy.get("is_vlm"))
        if is_vlm:
            processor = get_tokenizer(config.policy["tokenizer"], get_processor=True)
            tokenizer = processor.tokenizer
        else:
            processor = None
            tokenizer = get_tokenizer(config.policy["tokenizer"])

        assert config.policy["generation"] is not None, (
            "A generation config is required for GRPO"
        )
        has_refit_draft_weights = draft_refit_enabled(config.policy.get("draft"))
        trains_mtp = (
            "megatron_cfg" in config.policy
            and config.policy["megatron_cfg"]["enabled"]
            and bool(config.policy["megatron_cfg"].get("mtp_num_layers"))
        )
        config.policy["generation"] = configure_generation_config(
            config.policy["generation"],
            tokenizer,
            has_refit_draft_weights=has_refit_draft_weights,
            trains_mtp=trains_mtp,
        )
        if is_vlm and "vllm_cfg" in config.policy["generation"]:
            assert not config.policy["generation"]["vllm_cfg"]["skip_tokenizer_init"], (
                "VLMs require tokenizer to be initialized before generation, "
                "so skip_tokenizer_init must be set to False."
            )

                                         
        setup_nemo_gym_config(config, tokenizer)

                                                                                      
    assert should_use_nemo_gym(config)

                                                                                                                      
    with rl_init_timer.time("data"):
        print("\n▶ Setting up data...")
        data_tokenizer = processor if processor is not None else tokenizer
        train_dataset, val_dataset = setup_response_data(
            data_tokenizer, config.data, env_configs=None
        )

                                      
    if config.grpo.max_val_samples is not None:
        raise ValueError(
            """A non-null `grpo.max_val_samples` parameter is not supported.

Gym principle is that there is no hidden data pre or post processing from you. What you see is what you get.

The validation set you pass in will directly be used for validation with no additional preprocessing. If you want to have some number of repetitions, please include that in your dataset, via ``num_repeats``, in your dataset config and `ng_prepare_data` will prepare it accordingly."""
        )

    if val_dataset is not None:
                                                                                
                                                                                    
        config.grpo.max_val_samples = len(val_dataset)
        if config.grpo.val_batch_size is None:
            print(
                f"Setting `grpo.max_val_samples` and `grpo.val_batch_size` to the length of the validation dataset, which is {len(val_dataset)}"
            )
            config.grpo.val_batch_size = config.grpo.max_val_samples
        else:
            waves = math.ceil(len(val_dataset) / config.grpo.val_batch_size)
            print(
                f"Setting `grpo.max_val_samples` to the length of the validation dataset ({len(val_dataset)}); "
                f"`grpo.val_batch_size`={config.grpo.val_batch_size} -> validation runs in {waves} wave(s)"
            )

    if mlperf_logger is not None:
        mlperf_logger.log_hyperparams(train_dataset, val_dataset)

                  
    print("Final config:")
    pprint.pprint(config)

    with rl_init_timer.time("ray_connect"):
                                                                                   
        maybe_configure_data_plane_env(config.data_plane)
        init_ray()

                                                                            
                                                                              
                                               
    is_trajectory_collection = (
        config.env["nemo_gym"].pop("is_trajectory_collection", False) or False
    )

    with rl_init_timer.time("setup"):
        (
            policy,
            policy_generation,
            nemo_gym,
            clusters,
            dataloader,
            val_dataloader,
            loss_fn,
            logger,
            checkpointer,
            grpo_state,
            master_config,
            teacher_worker_groups,
            alias_to_group_alias,
        ) = setup(
            config,
            tokenizer,
            train_dataset,
            val_dataset,
            processor=processor,
        )

    rl_init_timer.record("total", time.perf_counter() - main_start)
    rl_init_metrics = rl_init_timer.get_timing_metrics(reduction_op="sum")
    print("\n" + "=" * 60)
    print(" " * 14 + "RL INIT TIMING BREAKDOWN")
    for label, value in sorted(rl_init_metrics.items()):
        if isinstance(value, (int, float)):
            print(f"  {label}: {value:.1f}s")
    print("=" * 60 + "\n", flush=True)

    if mlperf_logger is not None:
        mlperf_logger.log_hyperparams_after_setup(
            data_parallel_size=policy.data_parallel_size
        )
                                                                        
                                                                       
                                                                     
        logger = MLPerfMetricsLoggerShim(logger, mlperf_logger)

                                                                           
                                                                
                                                           
    task_to_env = {"nemo_gym": nemo_gym}
    val_task_to_env = task_to_env

    if not is_trajectory_collection:
        if mlperf_logger is not None:
                                                                            
            if time.monotonic() - mlperf_init_started >= 1790:
                raise RuntimeError("Model setup exceeded the safe Closed initialization budget")
        maybe_run_mlperf_warmup(
            policy=policy,
            policy_generation=policy_generation,
            tokenizer=tokenizer,
            loss_fn=loss_fn,
            master_config=master_config,
            warmup_cfg=mlperf_warmup_cfg,
        )
        if mlperf_logger is not None:
                                                                             
                                                                            
            init_seconds = time.monotonic() - mlperf_init_started
            if init_seconds >= 1790:
                raise RuntimeError(f"Initialization including warmup exceeded the safe Closed budget: {init_seconds:.1f}s")
            mlperf_logger.log_init_stop_run_start()
            print(f"QWEN_MLPERF_TIMING: initialization including warmup={init_seconds:.1f}s; limit=1800s", flush=True)

    trainer_owns_environment_shutdown = False
    training_succeeded = False
    try:
        if is_trajectory_collection:
            if mlperf_logger is not None:
                mlperf_logger.log_init_stop_run_start()
            collect_trajectories(
                policy=policy,
                policy_generation=policy_generation,
                val_dataloader=val_dataloader,
                tokenizer=tokenizer,
                val_task_to_env=val_task_to_env,
                logger=logger,
                master_config=master_config,
            )
                                        
        elif config.grpo.async_grpo.enabled:
                                                                                                             
            if config.grpo.use_dynamic_sampling:
                raise NotImplementedError(
                    "use_dynamic_sampling is not supported with async GRPO"
                )
            if config.grpo.reward_scaling.enabled:
                raise NotImplementedError(
                    "reward_scaling is not supported with async GRPO"
                )
            if config.grpo.reward_shaping.enabled:
                raise NotImplementedError(
                    "reward_shaping is not supported with async GRPO"
                )

                                                              
            if config.data["use_multiple_dataloader"]:
                raise NotImplementedError(
                    "use_multiple_dataloader is not supported with async GRPO"
                )

            from nemo_rl.algorithms.grpo import async_grpo_train

            print("🚀 Running async GRPO training")

            log_mlperf_run_start(mlperf_logger, config, grpo_state)
                                     
            with (
                timestamp_policy_training(policy, grpo_state)
                if deferred_evaluation is not None
                else nullcontext()
            ):
                trainer_owns_environment_shutdown = True
                async_grpo_train(
                    policy=policy,
                    policy_generation=policy_generation,
                    dataloader=dataloader,
                    val_dataloader=val_dataloader,
                    tokenizer=tokenizer,
                    loss_fn=loss_fn,
                    task_to_env=task_to_env,
                    val_task_to_env=val_task_to_env,
                    logger=logger,
                    checkpointer=checkpointer,
                    grpo_save_state=grpo_state,
                    master_config=master_config,
                    max_trajectory_age_steps=config.grpo.async_grpo.max_trajectory_age_steps,
                    teacher_worker_groups=teacher_worker_groups,
                    alias_to_group_alias=alias_to_group_alias,
                    processor=processor,
                )
            if (
                deferred_evaluation is not None
                and grpo_state.total_steps != config.grpo.max_num_steps
            ):
                raise RuntimeError(
                    "deferred training stopped before the final checkpoint step"
                )
        else:
            print("🚀 Running synchronous GRPO training")

            log_mlperf_run_start(mlperf_logger, config, grpo_state)
                                        
            trainer_owns_environment_shutdown = True
            grpo_train(
                policy,
                policy_generation,
                dataloader,
                val_dataloader,
                tokenizer,
                loss_fn,
                task_to_env,
                val_task_to_env,
                logger,
                checkpointer,
                grpo_state,
                master_config,
                processor=processor,
            )
        training_succeeded = True
    finally:
        if deferred_evaluation is not None:
            try:
                shutdown_resources(
                    nemo_gym, policy_generation, policy, clusters, checkpointer
                )
            except Exception:
                if mlperf_logger is not None:
                    mlperf_logger.finalize()
                raise
            if mlperf_logger is not None:
                if training_succeeded:
                    mlperf_logger.stop_train_block(
                        grpo_state.total_steps,
                        time_ms=grpo_state.training_step_end_time_ms,
                    )
                else:
                    mlperf_logger.finalize()
        else:
            if mlperf_logger is not None:
                mlperf_logger.finalize()
                                                                                    
            if not trainer_owns_environment_shutdown:
                shutdown_environments(task_to_env, val_task_to_env)
            try:
                policy_generation.shutdown()
            except Exception as error:
                print(f"Error shutting down generation: {error}", flush=True)


if __name__ == "__main__":
    main()
