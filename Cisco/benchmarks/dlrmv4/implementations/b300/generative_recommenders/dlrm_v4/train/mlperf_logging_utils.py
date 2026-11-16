# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

             
''                                                                             

                                                                                
                                                                            
   

import logging
import os
from typing import Any, Dict, Optional

import gin
import torch

logger: logging.Logger = logging.getLogger(__name__)

try:
    from mlperf_logging import mllog
    from mlperf_logging.mllog import constants as mllog_constants

    _MLLOG_AVAILABLE = True
except Exception as e:                                        
    mllog = None  # type: ignore[assignment]
    mllog_constants = None  # type: ignore[assignment]
    _MLLOG_AVAILABLE = False
    logger.warning(
        "mlperf_logging not importable (%s); MLPerf logging disabled. "
        "Install via `pip install git+https://github.com/mlcommons/logging.git`.",
        e,
    )


def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def _barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


class MLPerfLogger:
    ''                                    

                                                                            
                                                                                 
                                                
       

    def __init__(
        self,
        rank: Optional[int] = None,
        log_path: Optional[str] = None,
        default_stack_offset: int = 2,
                                                                         
                                                                               
                                                                        
                                                                          
        benchmark_name: str = "dlrmv4",
        submitter_name: str = "SUBMISSION_ORG_PLACEHOLDER",
        submission_platform: str = "SUBMISSION_PLATFORM_PLACEHOLDER",
        fresh: bool = True,
    ):
        self.enabled: bool = _MLLOG_AVAILABLE
                                                                                
                                                                            
        self.rank: int = rank if rank is not None else _rank()
        self.benchmark_name: str = benchmark_name
        self.submitter_name: str = submitter_name
        self.submission_platform: str = submission_platform
        self._logger = None
        if not self.enabled:
            return
                                                                   
        if log_path and self.rank == 0:
            log_dir = os.path.dirname(log_path)
            if log_dir:                                                     
                os.makedirs(log_dir, exist_ok=True)
                                                                                  
                                                                                 
                                                                                
                                                                                 
                                                                           
                                                                                
            if fresh:
                open(log_path, "w").close()
            mllog.config(filename=log_path, default_stack_offset=default_stack_offset)
        else:
            mllog.config(default_stack_offset=default_stack_offset)
        self._logger = mllog.get_mllogger()

    @property
    def constants(self):                  
        return mllog_constants

    def event(
        self,
        key: str,
        value: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
        sync: bool = False,
    ) -> None:
        if sync:
            _barrier()
        if self.enabled and self.rank == 0:
            self._logger.event(key=key, value=value, metadata=metadata or {})

    def start(
        self,
        key: str,
        value: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
        sync: bool = False,
    ) -> None:
        if sync:
            _barrier()
        if self.enabled and self.rank == 0:
            self._logger.start(key=key, value=value, metadata=metadata or {})

    def end(
        self,
        key: str,
        value: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
        sync: bool = False,
    ) -> None:
        if sync:
            _barrier()
        if self.enabled and self.rank == 0:
            self._logger.end(key=key, value=value, metadata=metadata or {})

    def submission_info(self) -> None:
        ''                                                                      
        if not (self.enabled and self.rank == 0):
            return
        c = mllog_constants
        self.event(key=c.SUBMISSION_BENCHMARK, value=self.benchmark_name)
        self.event(key=c.SUBMISSION_ORG, value=self.submitter_name)
        self.event(key=c.SUBMISSION_DIVISION, value=c.CLOSED)
        self.event(key=c.SUBMISSION_STATUS, value=c.ONPREM)
        self.event(key=c.SUBMISSION_PLATFORM, value=self.submission_platform)

    def log_run_start(
        self,
        global_batch_size: int,
        seed: int,
        micro_batch_size: int,
        gradient_accumulation_steps: int = 1,
        target_accuracy: Optional[float] = None,
    ) -> None:
        ''                                                                         

                                                                                 
                                                                              
                                                                               
                                
                                                                           
                                                                                 
                                                                                 
                                                                           
                                                                                  
                                                                               
                     
           
        c = self.constants
        self.submission_info()
        self.event(key=c.GLOBAL_BATCH_SIZE, value=int(global_batch_size))
        self.event(
            key=c.GRADIENT_ACCUMULATION_STEPS, value=int(gradient_accumulation_steps)
        )
        self.event(key=c.MICRO_BATCH_SIZE, value=int(micro_batch_size))
        self.event(key=c.TENSOR_PARALLELISM, value=1)
        self.event(key=c.PIPELINE_PARALLELISM, value=1)
        self.event(key=c.CONTEXT_PARALLELISM, value=1)
        self.event(key=c.EXPERT_PARALLELISM, value=1)
        self.event(
            key=c.CONFIG_FILENAME,
            value=f"config_{os.environ.get('DGXSYSTEM', 'UNKNOWN')}.sh",
        )
        self.event(key=c.SEED, value=int(seed))
        self.event(
            key=c.OPT_NAME,
            value=_gin_param("dense_optimizer_factory_and_class.optimizer_name", "Adam"),
        )
        self.event(
            key=c.OPT_BASE_LR,
            value=_gin_param("dense_optimizer_factory_and_class.learning_rate", None),
        )
        betas = _gin_param("dense_optimizer_factory_and_class.betas", (0.95, 0.999))
        self.event(key=c.OPT_ADAM_BETA_1, value=float(betas[0]))
        self.event(key=c.OPT_ADAM_BETA_2, value=float(betas[1]))
        self.event(
            key=c.OPT_ADAM_EPSILON,
            value=float(_gin_param("dense_optimizer_factory_and_class.eps", 1e-8)),
        )
        self.event(
            key=c.OPT_WEIGHT_DECAY,
            value=float(
                _gin_param("dense_optimizer_factory_and_class.weight_decay", 0.0)
            ),
        )
        self.event(
            key=c.OPT_GRADIENT_CLIP_NORM,
            value=float(_gin_param("streaming_train_eval_loop.grad_clip_norm", 0.0)),
        )
        self.event(
            key=c.OPT_LR_WARMUP_STEPS,
            value=int(_gin_param("streaming_train_eval_loop.lr_warmup_steps", 0)),
        )
                                                                               
                                                                           
        self.event(
            key="opt_learning_rate_warmup_start_lr",
            value=float(
                _gin_param("streaming_train_eval_loop.lr_warmup_start_lr", 0.0)
            ),
        )
        self.event(
            key="opt_sparse_name",
            value=_gin_param(
                "sparse_optimizer_factory_and_class.optimizer_name", "RowWiseAdagrad"
            ),
        )
        self.event(
            key="opt_sparse_base_learning_rate",
            value=_gin_param(
                "sparse_optimizer_factory_and_class.learning_rate", None
            ),
        )
        self.event(
            key=c.MAX_SEQUENCE_LENGTH,
            value=int(_gin_param("get_hstu_configs.max_seq_len", 4096)),
        )
                                                                   
                                                                               
                                                                               
                                                                            
                                                                           
        self.event(
            key="eval_every_data_pct",
            value=float(
                _gin_param("streaming_train_eval_loop.eval_every_data_pct", 0.0)
            ),
        )
        self.event(
            key="skip_eval_epoch_pct",
            value=float(
                _gin_param("streaming_train_eval_loop.skip_eval_epoch_pct", 0.0)
            ),
        )
        if target_accuracy is not None:
            self.event(key="target_accuracy", value=float(target_accuracy))
        else:
            logger.warning(
                "target_accuracy is None; skipping the event. MetricsLogger."
                "auc_threshold is unbound, so this run has no convergence target"
            )
        self._log_numerical_precision()
        self.end(key=c.INIT_STOP, sync=True)
        self.start(key=c.RUN_START, sync=True)

    def _log_numerical_precision(self) -> None:
        ''                                                                

                                                                           
                                                             
                                                                               
                                                                        

                                  
                                                                              
                                                                             
                                                                            
                                                                               
                                                                          
                                                                            
                                                                              
                                                                           
                                                                            

                                                                               
                                                                             
                                                                         
                                                                             
           
        for key, env in (
            (self.constants.LOWEST_NUMERICAL_PRECISION_IN_LINEAR, "MLPERF_LINEAR_PRECISION"),
            (self.constants.LOWEST_NUMERICAL_PRECISION_IN_ATTN, "MLPERF_ATTN_PRECISION"),
            (self.constants.LOWEST_NUMERICAL_PRECISION_IN_COMM, "MLPERF_COMM_PRECISION"),
        ):
            value = os.environ.get(env, "").strip()
            if not value:
                logger.warning(
                    "$%s is unset; %s is mandatory in v6.1 and will fail "
                    "compliance", env, key
                )
            if value not in _MLPERF_PRECISIONS:
                                                                             
                                                                          
                logger.warning(
                    "$%s=%r is not one of %s; the compliance checker will "
                    "reject it", env, value, sorted(_MLPERF_PRECISIONS)
                )
            self.event(key=key, value=value)


                                                                            
                                                               
_MLPERF_PRECISIONS = frozenset({
    "fp64", "fp32", "tf32", "fp16", "fp8", "nvfp4", "mxfp4", "bfloat16",
    "Graphcore FLOAT 16.16", "int8", "uint8", "int4", "uint4",
})


def _gin_param(name: str, default: Any) -> Any:
    ''                                                                         

                                                                            
                                                                             
       
    try:
        value = gin.query_parameter(name)
    except (ValueError, KeyError):
        return default
    if hasattr(value, "scoped_configurable_fn"):
        try:
            return value.scoped_configurable_fn()
        except Exception:
            return default
    return value


class MLPerfRunTracker:
    ''                                                                      

                                                                             
                                                              
                                                                           
                                                                           
                                               
       

                                                              
    _EVAL_METRIC_SHORT = "window_auc"

    def __init__(
        self,
        logger: Optional[MLPerfLogger],
        metric_logger: Any,
        total_train_samples: int,
        rank: int,
        device: Any,
    ):
        self.logger = logger
        self.metric_logger = metric_logger
        self.total_train_samples = int(total_train_samples)
        self.rank = int(rank)
        self.device = device
        self.run_stopped: bool = False
                                                                              
                                                                    
        self._block_open: bool = False

    @property
    def enabled(self) -> bool:
        return self.logger is not None

    def _progress(self) -> Dict[str, Any]:
        c = self.logger.constants
        return {c.SAMPLES_COUNT: self.metric_logger.cumulative_train_samples}

    def log_dataset_sizes(self, eval_samples: Optional[int] = None) -> None:
        if not self.enabled:
            return
        c = self.logger.constants
        self.logger.event(key=c.TRAIN_SAMPLES, value=self.total_train_samples)
        if eval_samples is not None:
            self.logger.event(key=c.EVAL_SAMPLES, value=int(eval_samples))

    def block_start(self) -> None:
        if self.enabled and not self._block_open:
            self.logger.start(
                key=self.logger.constants.BLOCK_START, metadata=self._progress()
            )
            self._block_open = True

    def block_stop(self) -> None:
        if self.enabled and self._block_open:
            self.logger.end(
                key=self.logger.constants.BLOCK_STOP, metadata=self._progress()
            )
            self._block_open = False

    def eval_start(self) -> None:
        if self.enabled:
            self.logger.start(
                key=self.logger.constants.EVAL_START, metadata=self._progress()
            )

    def _target_metric(self, metrics: Dict[str, float]) -> Optional[float]:
                                                                                 
                                              
        for key, val in metrics.items():
            short = key.split("/")[-2] if "/" in key else key
            if short == self._EVAL_METRIC_SHORT:
                return float(val)
        return None

    def _meets_target(self, value: Optional[float]) -> bool:
        thr = self.metric_logger.auc_threshold
        if value is None or thr is None:
            return False
        return value >= thr

    def run_stop(self, status: object) -> None:
                                                                               
                                                         
        if not self.enabled or self.run_stopped:
            return
        c = self.logger.constants
        self.logger.end(
            key=c.RUN_STOP,
            metadata={c.STATUS: status, **self._progress()},
            sync=True,
        )
        self.run_stopped = True

    def eval_stop(self, eval_metrics: Dict[str, float]) -> bool:
                                                                           
                                                                                  
                                                                       
        if not self.enabled:
            return False
        c = self.logger.constants
        eval_value = self._target_metric(eval_metrics)
        if eval_value is not None:
            self.logger.event(
                key=c.EVAL_ACCURACY, value=eval_value, metadata=self._progress()
            )
        self.logger.end(key=c.EVAL_STOP, metadata=self._progress())
        decision = torch.zeros(1, device=self.device)
        if self.rank == 0 and not self.run_stopped and self._meets_target(eval_value):
            decision[0] = 1.0
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.broadcast(decision, src=0)
        should_stop = bool(decision.item() > 0.5)
        if should_stop:
                                                                         
            self.run_stop(c.SUCCESS)
        return should_stop

    def finalize(self, final_metrics: Dict[str, float]) -> None:
                                                                                
                                                           
        if not self.enabled or self.run_stopped:
            return
        c = self.logger.constants
        success = self._meets_target(self._target_metric(final_metrics))
        self.run_stop(c.SUCCESS if success else c.ABORTED)


def mlperf_checkpoint_present(ckpt_path: str) -> bool:
    ''                                                                           

                                                                         
                                                                                
                                                                               
                                                                         
                                                                                
                                                                   
       
    if not ckpt_path:
        return False
    base = ckpt_path.rstrip("/")
                                                                              
    if os.path.basename(base).isdigit():
        return os.path.isdir(base)
    if not os.path.isdir(base):
        return False
    for name in os.listdir(base):
        if name.isdigit() and os.path.isdir(os.path.join(base, name)):
            return True
    return False


@gin.configurable
def get_mlperf_logger(
    rank: int = 0,
    log_path: str = "",
    benchmark_name: str = "dlrmv4",
    submitter_name: str = "SUBMISSION_ORG_PLACEHOLDER",
    submission_platform: str = "SUBMISSION_PLATFORM_PLACEHOLDER",
    fresh: bool = True,
) -> Optional[MLPerfLogger]:
    ''                                                                      

                                                                           
                                                                    

                                                                               
                                                                                 
                                                                                 
                                                                            
       
    if not _MLLOG_AVAILABLE:
        return None
    if os.environ.get("MLPERF_LOGGING", "1").strip().lower() in (
        "0", "false", "no", "off",
    ):
        logger.info("MLPerf logging disabled via $MLPERF_LOGGING=0")
        return None
    resolved_path = os.environ.get("MLPERF_LOG_PATH", log_path)
                                                             
                                                                               
                                                                             
                                                                          
                                                                               
                                                             
    num_nodes = os.environ.get("SLURM_JOB_NUM_NODES", 1)
    resolved_platform = os.environ.get("MLPERF_SUBMISSION_PLATFORM")
    if resolved_platform is None:
        resolved_platform = "{}x{}".format(num_nodes, submission_platform)
    resolved_submitter = os.environ.get("MLPERF_SUBMISSION_ORG", submitter_name)
    return MLPerfLogger(
        rank=rank,
        log_path=resolved_path,
        benchmark_name=benchmark_name,
        submitter_name=resolved_submitter,
        submission_platform=resolved_platform,
        fresh=fresh,
    )
