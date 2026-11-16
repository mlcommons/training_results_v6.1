# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

             
import argparse
import gc
import logging

logging.basicConfig(level=logging.INFO)
import os
import sys
import traceback

import gin
import torch

                                                                          
                                                             
                                                                      
                                                                           

logger: logging.Logger = logging.getLogger(__name__)


SUPPORTED_CONFIGS = {
    "debug": "debug.gin",
    "kuairand-1k": "kuairand_1k.gin",
    "movielens-1m": "movielens_1m.gin",
    "movielens-20m": "movielens_20m.gin",
    "movielens-13b": "movielens_13b.gin",
    "movielens-18b": "movielens_18b.gin",
    "streaming-400m": "streaming_400m.gin",
    "streaming-200b": "streaming_200b.gin",
    "streaming-100b": "streaming_100b.gin",
    "yambda-5b": "yambda_5b.gin",
}


def _main_func(
    rank: int,
    local_rank: int,
    world_size: int,
    local_world_size: int,
    master_addr: str,
    master_port: int,
    gin_file: str,
    mode: str,
) -> None:
    device = torch.device(f"cuda:{local_rank}")
    logger.info(
        f"rank: {rank} (local_rank={local_rank}), "
        f"world_size: {world_size}, device: {device}"
    )
                                                                      
                                                                    
                                                                        
                                                                         
                                                                    
    from generative_recommenders.dlrm_v4.train._env_bootstrap import apply_env_bootstrap
    from generative_recommenders.dlrm_v4.train.mlperf_logging_utils import (
        get_mlperf_logger,
        mlperf_checkpoint_present,
    )

    gin.parse_config_file(gin_file, skip_unknown=True)
    apply_env_bootstrap()

                                                                               
                                                                                 
                                                                                  
                                                                            
    mlperf_resume = mlperf_checkpoint_present(os.environ.get("CKPT_PATH", ""))
                                                                                 
                                                                               
                                                                 
    mlperf_logger = (
        get_mlperf_logger(rank=rank, fresh=not mlperf_resume)
        if mode == "streaming-train-eval"
        else None
    )
                                                                              
                                                                            
    mlperf_cold_start = mlperf_logger is not None and not mlperf_resume
    if mlperf_cold_start:
        mlperf_logger.event(key=mlperf_logger.constants.CACHE_CLEAR, value=True)
        mlperf_logger.start(key=mlperf_logger.constants.INIT_START)

                                                                           
                                                    
    from generative_recommenders.dlrm_v4.checkpoint import load_dmp_checkpoint
    from generative_recommenders.dlrm_v4.train.utils import (
        cleanup,
        decorrelate_runtime_rng,
        eval_loop,
        make_model,
        make_optimizer_and_shard,
        make_train_test_dataloaders,
        seed_everything,
        setup,
        streaming_train_eval_loop,
        train_eval_loop,
        train_loop,
        warmup_model,
    )
    from generative_recommenders.dlrm_v4.utils import (
        MetricsLogger,
        get_gpu_peak_flops,
    )

    setup(
        rank=rank,
        world_size=world_size,
        master_addr=master_addr,
        master_port=master_port,
        device=device,
    )
                                                                        
                                                                    
                                                                          
                                                                       
                                       
    gin.parse_config_file(gin_file)

    warmup_batches = int(os.environ.get("NUM_WARMUP_BATCHES", "20"))
    if warmup_batches < 0:
        raise ValueError(
            "NUM_WARMUP_BATCHES must be non-negative, got "
            f"{warmup_batches}"
        )
    if mlperf_cold_start and warmup_batches > 0:
                                                                                
                                                                             
                                                                             
                                                                                
        seed_everything(rank=rank)
        warmup_batch_size = int(os.environ.get("BATCH_SIZE", "1024"))
        warmup_embedding_rows = 32 * warmup_batch_size
        temporary_model, warmup_configs, warmup_table_configs = make_model(
            max_embedding_table_rows=warmup_embedding_rows
        )
        temporary_model, temporary_optimizer = make_optimizer_and_shard(
            model=temporary_model,
            device=device,
            world_size=world_size,
            local_world_size=local_world_size,
            embedding_table_configs=warmup_table_configs,
        )
        decorrelate_runtime_rng(rank=rank)
        warmup_model(
            rank=rank,
            model=temporary_model,
            config=warmup_configs,
            batch_size=warmup_batch_size,
            device=device,
            num_batches=warmup_batches,
            seed=int(os.environ.get("SEED", "1")),
            embedding_value_bound=warmup_embedding_rows,
        )
        del temporary_optimizer, temporary_model
        gc.collect()
        torch.cuda.empty_cache()
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

                                                                               
                                                                              
                                                                           
                                                                    
    if mlperf_cold_start and warmup_batches > 0:
        seed_everything(seed=int(os.environ["SEED"]), rank=rank)
    else:
        seed_everything(rank=rank)

    model, model_configs, embedding_table_configs = make_model()
    model, optimizer = make_optimizer_and_shard(
        model=model,
        device=device,
        world_size=world_size,
        local_world_size=local_world_size,
        embedding_table_configs=embedding_table_configs,
    )
                                                                             
                                                                           
                                                                                
                                                                              
    decorrelate_runtime_rng(rank=rank)
    train_dataloader, test_dataloader = make_train_test_dataloaders(
        hstu_config=model_configs,
        embedding_table_configs=embedding_table_configs,
    )
                                                                              
                                                                              
                        
    inner_model = model.module if hasattr(model, "module") else model
    enable_tflops_logging = bool(getattr(inner_model, "enable_tflops_logging", False))
    num_flops_per_sample = (
        float(inner_model.get_num_flops_per_sample())
        if enable_tflops_logging
        and hasattr(inner_model, "get_num_flops_per_sample")
        else 0.0
    )
    gpu_peak_flops = (
        get_gpu_peak_flops(
            "bf16" if getattr(model_configs, "bf16_training", True) else "fp32"
        )
        if enable_tflops_logging
        else 0.0
    )
                                                                              
                                                                            
                                                                                 
    metrics = MetricsLogger(
        multitask_configs=model_configs.multitask_configs,
        batch_size=train_dataloader.batch_size,
        window_size=2500,
        device=device,
        rank=rank,
                                                                            
                                                                               
                                                                        
        world_size=world_size,
        num_flops_per_sample=num_flops_per_sample,
        gpu_peak_flops=gpu_peak_flops,
        model=model,
        eval_cumulative=(mode == "streaming-train-eval"),
                                                                              
                                                                              
                                                                               
                                                                  
    )
                                                                        
                                                                           
                                                                         
                                                                             
    resume_train_ts, resume_batch_idx_in_window, resume_split_contract, resume_cold_start = (
        load_dmp_checkpoint(
            model=model,
            optimizer=optimizer,
            metric_logger=metrics,
            device=device,
            rank=rank,
        )
    )

                                                                         
                                                                                 
                                                                                
                                                                                  
                                                                                 
                                                                                 
    if mlperf_cold_start and not metrics.mlperf_run_started:
                                                                                 
                                                                                
                                                       
        mlperf_logger.log_run_start(
            global_batch_size=world_size * int(train_dataloader.batch_size),
            seed=int(os.environ.get("SEED", "1")),
            micro_batch_size=int(train_dataloader.batch_size),
            target_accuracy=metrics.auc_threshold,
        )
        metrics.mlperf_run_started = True
                                                                                
                                                                                 
    mlperf_run_active = mlperf_logger is not None and metrics.mlperf_run_started

                
    try:
        if mode == "train":
            train_loop(
                rank=rank,
                model=model,
                dataloader=train_dataloader,
                optimizer=optimizer,
                metric_logger=metrics,
                device=device,
            )
        elif mode == "eval":
                                            
            metrics = MetricsLogger(
                multitask_configs=model_configs.multitask_configs,
                batch_size=train_dataloader.batch_size,
                window_size=1000,
                device=device,
                rank=rank,
                world_size=world_size,
            )
            eval_loop(
                rank=rank,
                model=model,
                dataloader=test_dataloader,
                metric_logger=metrics,
                device=device,
            )
        elif mode == "train-eval":
            train_eval_loop(
                rank=rank,
                model=model,
                train_dataloader=train_dataloader,
                eval_dataloader=test_dataloader,
                optimizer=optimizer,
                metric_logger=metrics,
                device=device,
            )
        elif mode == "streaming-train-eval":
            streaming_train_eval_loop(
                rank=rank,
                model=model,
                optimizer=optimizer,
                metric_logger=metrics,
                device=device,
                hstu_config=model_configs,
                embedding_table_configs=embedding_table_configs,
                resume_train_ts=resume_train_ts,
                resume_batch_idx_in_window=resume_batch_idx_in_window,
                resume_split_contract=resume_split_contract,
                resume_cold_start=resume_cold_start,
                                                                               
                                                               
                mlperf_logger=mlperf_logger if mlperf_run_active else None,
            )
    except Exception as e:
        logger.info(traceback.format_exc())
        raise Exception(e)
    finally:
                                                                               
                                                                             
                                                                         
        if torch.distributed.is_initialized():
            try:
                torch.distributed.barrier()
            except Exception:
                logger.info("teardown barrier failed (non-fatal)")
            try:
                cleanup()
            except Exception:
                logger.info("teardown destroy_process_group failed (non-fatal)")


def get_args():                   
    ''                      
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default="debug", choices=SUPPORTED_CONFIGS.keys(), help="dataset"
    )
    parser.add_argument(
        "--mode",
        default="train",
        choices=["train", "eval", "train-eval", "streaming-train-eval"],
        help="mode",
    )
    args, unknown_args = parser.parse_known_args()
    logger.warning(f"unknown_args: {unknown_args}")
    return args


def main() -> None:
    args = get_args()
    logger.info(args)
    assert args.dataset in SUPPORTED_CONFIGS, f"Unsupported dataset: {args.dataset}"
    assert args.mode in [
        "train",
        "eval",
        "train-eval",
        "streaming-train-eval",
    ], f"Unsupported mode: {args.mode}"
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    if (
        world_size <= 0
        or local_world_size <= 0
        or world_size % local_world_size != 0
        or not 0 <= rank < world_size
        or not 0 <= local_rank < local_world_size
        or rank % local_world_size != local_rank
    ):
        raise ValueError(
            f"invalid distributed topology: rank={rank}, "
            f"local_rank={local_rank}, world_size={world_size}, "
            f"local_world_size={local_world_size}"
        )
    gin_path = f"{os.path.dirname(__file__)}/gin/{SUPPORTED_CONFIGS[args.dataset]}"
    logger.info(
        f"external launch: rank={rank} local_rank={local_rank} "
        f"world_size={world_size} local_world_size={local_world_size} "
        f"master={os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"
    )
    _main_func(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        local_world_size=local_world_size,
        master_addr=os.environ["MASTER_ADDR"],
        master_port=int(os.environ["MASTER_PORT"]),
        gin_file=gin_path,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
