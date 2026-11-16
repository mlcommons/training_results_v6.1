# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

import json
import os
import time
from itertools import repeat

import numpy as np
import torch
from megatron.core.num_microbatches_calculator import get_num_microbatches
from mlperf_logging.mllog import constants
from nemo.collections.nlp.data.language_modeling.megatron.blendable_dataset import BlendableDataset
from nemo.collections.nlp.data.language_modeling.megatron.megatron_batch_samplers import MegatronPretrainingBatchSampler
from nemo.collections.nlp.models.language_modeling.megatron_gpt_sft_model import MegatronGPTSFTModel
from nemo.utils import logging
from lightning.pytorch import Callback
from lightning.pytorch.loggers import Logger
from lightning.pytorch.utilities import rank_zero_only
from torch.utils.data import default_collate
from prof_handler import get_profiler, TOTAL_WARMUP_STEPS, TOTAL_ACTIVE_STEPS


def run_training_warmup(trainer, warmup_train_steps, warmup_validation_steps):
    torch.distributed.barrier()
    start = time.time()
    # Run forward and backward (no optimizer step)
    for _ in range(warmup_train_steps):
        trainer.model.training_step(trainer.model.get_synthetic_input())
    torch.distributed.barrier()
    logging.info(f"Finished training warmup: {time.time() - start}s. Starting validation warmup")

    if warmup_validation_steps > 0:
        trainer.training = False
        for _ in range(warmup_validation_steps):
            trainer.model.validation_step(trainer.model.get_synthetic_input(validation=True))
        trainer.training = True
        trainer.model.validation_step_outputs.clear()

    # For GPT `zero_grad` is a noop, but included here for completeness
    trainer.model.zero_grad()
    trainer._logger_connector.reset_results()
    trainer._logger_connector.reset_metrics()
    torch.distributed.barrier()
    logging.info(f"Time spent in run_training_warmup: {time.time() - start}s")


def reset_fp8_state(model):
    """Sets `fp8_initialized` flag to False in every TE layer which will force reinitialization."""
    logging.info("Forcing FP8 stats reinitialization")

    def reset_fp8(m):
        if hasattr(m, "fp8_initialized"):
            m.fp8_initialized = False

    models = model.model
    for model in models if isinstance(models, list) else [models]:
        model.apply(reset_fp8)


class Timer:
    def __init__(self, gbs):
        self.start_time = None
        self.stop_time = None
        self.elapsed_time = 0
        self.samples = 0
        self.gbs = gbs

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.stop_time = time.time()
        self.samples += self.gbs
        self.elapsed_time += self.stop_time - self.start_time

    def get_throughput(self):
        throughput = self.samples / self.elapsed_time
        self.samples = 0
        self.elapsed_time = 0
        return throughput


GIB = 1024**3


def log_memory(where, step):
    """trainloom: this rank's allocator peaks and the device's free memory, one JSON line per call."""
    free, total = torch.cuda.mem_get_info()
    record = {
        "rank": int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))),
        "where": where,
        "step": step,
        "max_alloc_gib": round(torch.cuda.max_memory_allocated() / GIB, 3),
        "max_reserved_gib": round(torch.cuda.max_memory_reserved() / GIB, 3),
        "reserved_gib": round(torch.cuda.memory_reserved() / GIB, 3),
        "free_gib": round(free / GIB, 3),
        "total_gib": round(total / GIB, 3),
    }
    print("[trainloom] mem " + json.dumps(record), flush=True)


class EvalTimer:
    """trainloom: TRAINLOOM_EVAL_TIMING=1 prints, on rank 0, where each eval's wall time goes: the cache release before
    and after it, the eval batches, the gaps between, and the first training step after it. Synchronises the GPU at
    each mark, so it is for diagnosis, not for scored runs."""

    def __init__(self):
        self.on = os.getenv("TRAINLOOM_EVAL_TIMING", "0") == "1" and os.environ.get(
            "RANK", os.environ.get("LOCAL_RANK", "0")
        ) == "0"
        self.marks = {}
        self.batches = 0

    def mark(self, name):
        if self.on:
            torch.cuda.synchronize()
            self.marks[name] = time.perf_counter()
            if name == "first_batch":
                self.batches = 0

    def report(self, step):
        m = self.marks
        if not self.on or "end_after_cache" not in m:
            return
        span = lambda a, b: round(m[b] - m[a], 3) if a in m and b in m else None
        record = {
            "step": step,
            "cache_before_s": span("start", "after_cache"),
            "to_first_batch_s": span("after_cache", "first_batch"),
            "batches_s": span("first_batch", "last_batch"),
            "n_batches": self.batches,
            "after_batches_s": span("last_batch", "end"),
            "cache_after_s": span("end", "end_after_cache"),
            "to_next_step_s": span("end_after_cache", "next_step"),
            "next_step_s": span("next_step", "next_step_end"),
            "eval_total_s": span("start", "end_after_cache"),
        }
        print("[trainloom] eval_timing " + json.dumps(record), flush=True)
        self.marks = {}


class EvalProfiler:
    """trainloom: TRAINLOOM_EVAL_PROFILE=1 records a torch-profiler trace of eval batches 2-4 of the first eval after
    training starts, on rank 0, to /results/torchprof_eval/trace_eval_0.json; one ProfilerStep per eval batch, so the
    trace reads like a training-step profile. For diagnosis only."""

    def __init__(self):
        self.on = os.getenv("TRAINLOOM_EVAL_PROFILE", "0") == "1" and os.environ.get(
            "RANK", os.environ.get("LOCAL_RANK", "0")
        ) == "0"
        self.prof = None

    def batch_start(self, trainer, batch_idx):
        if self.on and self.prof is None and trainer.global_step > 0 and batch_idx == 0:
            os.makedirs("/results/torchprof_eval", exist_ok=True)
            self.prof = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
                on_trace_ready=lambda p: p.export_chrome_trace("/results/torchprof_eval/trace_eval_0.json"),
            )
            self.prof.start()

    def batch_end(self, batch_idx):
        if self.on and self.prof is not None:
            self.prof.step()
            if batch_idx == 4:
                self.prof.stop()
                self.on = False


class CustomCallback(Callback):
    def __init__(self, cfg, mllogger):
        super().__init__()
        self.cfg = cfg
        self.gbs = cfg.model.global_batch_size
        self.mllogger = mllogger
        self.timer = Timer(self.gbs)
        self.force_success = os.getenv("FORCE_SUCCESS_STATUS", "0") == "1"
        self.logging_interval = 'step'
        self.log_freq = int(os.environ.get("TRAINING_LOSS_LOG_FREQ", 1000))

        self.torchprof = None
        self.rpd = None
        self.rpd_status = None
        if os.getenv("PROFILER", '') == 'pytorch':
            self.torchprof = get_profiler('torchprof')
            self.torchprof.start()
        if os.getenv("PROFILER", '') == 'rpd':
            from rpdTracerControl import rpdTracerControl
            rpdTracerControl.setFilename(name=f"trace.rpd", append=True)
            self.rpd = rpdTracerControl()

        # trainloom: TRAINLOOM_MEM_SNAPSHOT=1 records rank 0's allocations and dumps them at the first eval
        self.mem_snapshot = os.getenv("TRAINLOOM_MEM_SNAPSHOT", "0") == "1" and os.environ.get(
            "RANK", os.environ.get("LOCAL_RANK", "0")
        ) == "0"
        if self.mem_snapshot:
            torch.cuda.memory._record_memory_history(max_entries=2_000_000, stacks="python")

        self.eval_timer = EvalTimer()
        self.eval_profiler = EvalProfiler()

        # training loss logging
        self.losses = []
        self.print_every_n_batches = int(os.environ.get("LOG_FREQ", 10))

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if "end_after_cache" in self.eval_timer.marks and "next_step" not in self.eval_timer.marks:
            self.eval_timer.mark("next_step")
        self.timer.start()

        if self.rpd and trainer.global_step >= TOTAL_WARMUP_STEPS and not self.rpd_status:
            self.rpd.start()
            self.rpd_status = "running"
        
        if self.torchprof: 
            self.torchprof.step()            

        if trainer.global_step > TOTAL_WARMUP_STEPS + TOTAL_ACTIVE_STEPS:
            if self.torchprof:
                self.torchprof.stop()
            if self.rpd and self.rpd_status == "running":
                self.rpd.stop()
                self.rpd_status = "finished"

        return super().on_train_batch_start(trainer, pl_module, batch, batch_idx)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.timer.stop()
        if "next_step" in self.eval_timer.marks:
            self.eval_timer.mark("next_step_end")
            self.eval_timer.report(trainer.global_step)

        if trainer.global_step % self.log_freq == 0:
            self.mllogger.event(
                "train_loss",
                value=outputs["loss"].item(),
                metadata={
                    self.mllogger.constants.SAMPLES_COUNT: trainer.global_step
                    * self.gbs,
                    "lr": trainer.optimizers[0].param_groups[0]["lr"],
                },
        )

        return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    def on_validation_start(self, trainer, pl_module):
        self.eval_timer.marks = {}
        self.eval_timer.mark("start")
        log_memory("eval_start", trainer.global_step)
        if self.mem_snapshot:
            torch.cuda.memory._dump_snapshot("/results/mem_snapshot_rank0.pickle")
            torch.cuda.memory._record_memory_history(enabled=None)
            self.mem_snapshot = False
        # trainloom: RCCL allocates eval collective buffers with raw hipMalloc, which fails at low ACL
        # while the caching allocator holds the rest of the GPU. HS_EMPTY_CACHE_AT_EVAL=1.
        if os.environ.get("HS_EMPTY_CACHE_AT_EVAL", "0") == "1":
            torch.cuda.empty_cache()
        self.eval_timer.mark("after_cache")
        self.log_throughput(trainer.global_step)
        self.mllogger.end(
            constants.BLOCK_STOP,
            metadata={
                "samples_count": trainer.global_step * self.gbs,
                "curr_global_step": trainer.global_step, 
                "self.gbs": self.gbs
            },
            sync=False,
        )
        self.mllogger.start(
            key=constants.EVAL_START,
            metadata={"samples_count": trainer.global_step * self.gbs},
            sync=False,
        )
        # subsequent evaluations are every 384 sequences
        # if TP=4; CP=1; PP=1; MINIBS=4 then GBS=8. Therefore validation check runs every 384 // 8 = 48 training batches. 
        # lighthing sets the def value of val_check_interval=1. 
        trainer.val_check_interval = int(os.environ.get("VAL_CHECK_INTERVAL", 384)) // self.gbs
        trainer.val_check_batch = int(os.environ.get("VAL_CHECK_INTERVAL", 384)) // self.gbs
        return super().on_validation_start(trainer, pl_module)

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        if "first_batch" not in self.eval_timer.marks:
            self.eval_timer.mark("first_batch")
        self.eval_profiler.batch_start(trainer, batch_idx)
        return super().on_validation_batch_start(trainer, pl_module, batch, batch_idx, dataloader_idx)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.eval_timer.mark("last_batch")
        self.eval_timer.batches += 1
        self.eval_profiler.batch_end(batch_idx)
        return super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_validation_end(self, trainer, pl_module):
        self.eval_timer.mark("end")
        if os.environ.get("HS_EMPTY_CACHE_AT_EVAL", "0") == "1":
            torch.cuda.empty_cache()
        self.eval_timer.mark("end_after_cache")
        if trainer.should_stop:
            self.eval_timer.report(trainer.global_step)
        if not trainer.should_stop:
            self.mllogger.start(
                constants.BLOCK_START,
                metadata={"samples_count": trainer.global_step * self.gbs},
                sync=False,
            )
        return super().on_validation_end(trainer, pl_module)

    def on_train_start(self, trainer, pl_module):
        # Warmup
        if pl_module.cfg.custom.warmup:
            run_training_warmup(
                trainer, pl_module.cfg.custom.warmup_train_steps, pl_module.cfg.custom.warmup_validation_steps
            )
            if pl_module.cfg.fp8 and pl_module.cfg.custom.reset_fp8_stats_after_warmup:
                reset_fp8_state(pl_module)

        self.mllogger.log_init_stop_run_start()
        self.mllogger.start(constants.BLOCK_START, metadata={"samples_count": 0}, sync=False)
        self.train_start = time.time()

    def on_train_end(self, trainer, pl_module):
        log_memory("train_end", trainer.global_step)
        if self.rpd and self.rpd_status == "running":
            self.rpd.stop()
            self.rpd_status = "finished"
        run_duration_ms = time.time() - self.train_start
        self.mllogger.end(
            constants.RUN_STOP,
            metadata={
                "samples_count": trainer.global_step * self.gbs,
                "status": "success" if getattr(trainer, "target_reached", False) else "aborted",
                "duration": f"{run_duration_ms} sec -> {run_duration_ms / 60.0} minutes"
            },
        )
        return super().on_train_end(trainer, pl_module)

    def log_throughput(self, global_step):
        throughput = self.timer.get_throughput()
        self.mllogger.event(
            key="tracked_stats",
            metadata={"step": global_step * self.gbs},
            value={"throughput": throughput},
        )


class MetricsLogger(Logger):
    def __init__(self, cfg, mllogger, trainer):
        super().__init__()
        self.cfg = cfg
        self.gbs = cfg.model.global_batch_size
        self.mllogger = mllogger
        self.experiment = None
        self.trainer = trainer
        trainer.run_stop_logged = False

    def log_metrics(self, metrics, step):
        if "val_loss" in metrics:
            val_loss = metrics["val_loss"]
            self.mllogger.event(
                constants.EVAL_ACCURACY,
                value=val_loss,
                metadata={"samples_count": self.trainer.global_step * self.gbs},
            )
            self.mllogger.end(
                key=constants.EVAL_STOP,
                metadata={"samples_count": self.trainer.global_step * self.gbs},
                sync=False,
            )

            if val_loss < float(os.environ.get("TARGET_EVAL_LOSS", "0.925")):
                self.trainer.target_reached = True
                self.trainer.should_stop = True

    @rank_zero_only
    def log_hyperparams(self, params, *args, **kwargs):
        self.mllogger.event(key=constants.CACHE_CLEAR, value=True)
        self.mllogger.start(key=constants.INIT_START)
        self.mllogger.mlperf_submission_log(benchmark="llama2_70b_lora", num_nodes=self.cfg.trainer.num_nodes)
        self.mllogger.event(
            key=constants.SEED,
            value=self.cfg.model.seed,
            sync=False,
            unique=True,
        )
        self.mllogger.event(
            key=constants.GLOBAL_BATCH_SIZE,
            value=self.cfg.model.global_batch_size,
            sync=False,
        )
        # trainloom: the keys MLPerf Training v6.1 requires (as Krai's v4.2 logs them); MLLOG_* overrides
        m = self.cfg.model
        for key, value in (
            ("lowest_numerical_precision_in_linear", os.getenv("MLLOG_LOWEST_NUMERICAL_PRECISION_LINEAR", "fp8" if m.get("fp8", False) else "bfloat16")),
            ("lowest_numerical_precision_in_attn", os.getenv("MLLOG_LOWEST_NUMERICAL_PRECISION_ATTN", "fp8" if int(m.get("fp8_dot_product_attention", 0) or 0) else "bfloat16")),
            ("lowest_numerical_precision_in_comm", os.getenv("MLLOG_LOWEST_NUMERICAL_PRECISION_COMM", "bfloat16")),
            ("tensor_parallelism", int(os.getenv("MLLOG_TENSOR_PARALLELISM", m.get("tensor_model_parallel_size", 1)))),
            ("pipeline_parallelism", int(os.getenv("MLLOG_PIPELINE_PARALLELISM", m.get("pipeline_model_parallel_size", 1)))),
            ("context_parallelism", int(os.getenv("MLLOG_CONTEXT_PARALLELISM", m.get("context_parallel_size", 1)))),
            ("expert_parallelism", int(os.getenv("MLLOG_EXPERT_PARALLELISM", 1))),
            ("micro_batch_size", int(os.getenv("MLLOG_MICRO_BATCH_SIZE", m.get("micro_batch_size", 1)))),
            ("config_filename", os.getenv("MLLOG_CONFIG_FILENAME", f"config_{os.getenv('DGXSYSTEM', 'MI300X_1x8x1')}.sh")),
        ):
            self.mllogger.event(key=key, value=value, sync=False)
        self.mllogger.event(
            key=constants.TRAIN_SAMPLES,
            value=np.load("/data/train.npy", allow_pickle=True).shape[0],
        )
        self.mllogger.event(
            key=constants.EVAL_SAMPLES,
            value=np.load("/data/validation.npy", allow_pickle=True).shape[0],
        )
        self.mllogger.event(
            key=constants.OPT_LR_WARMUP_FACTOR,
            value=self.cfg.model.optim.sched.warmup_ratio,
        )
        self.mllogger.event(
            key=constants.OPT_ADAMW_WEIGHT_DECAY,
            value=self.cfg.model.optim.weight_decay,
        )
        self.mllogger.event(
            key=constants.OPT_GRADIENT_CLIP_NORM,
            value=self.cfg.trainer.gradient_clip_val,
        )
        ga = int(os.getenv("MINIBS", "1")) // self.cfg.model.micro_batch_size
        self.mllogger.event(key=constants.GRADIENT_ACCUMULATION_STEPS, value=ga)
        self.mllogger.event(key=constants.OPT_LR_TRAINING_STEPS, value=self.cfg.trainer.max_steps)
        self.mllogger.event(key=constants.OPT_BASE_LR, value=self.cfg.model.optim.lr)
        self.mllogger.event(key="lora_rank", value=self.cfg.model.peft.lora_tuning.adapter_dim)
        self.mllogger.event(key="lora_alpha", value=self.cfg.model.peft.lora_tuning.alpha)

    @property
    def name(self):
        return "mlperf-metrics"

    @property
    def version(self):
        return 1


class CustomMegatronGPTSFTModel(MegatronGPTSFTModel):
    def get_synthetic_input(self, validation=False):
        # Needed because init_global_step is not initialized at warmup
        self.init_global_step = self.trainer.global_step

        # Synthetic data generation below may need to change
        # if either dataset or sampler changes
        if not isinstance(self._train_ds, BlendableDataset) or not isinstance(
            self._train_dl.batch_sampler, MegatronPretrainingBatchSampler
        ):
            raise NotImplementedError(
                f"No synthetic data implementation for "
                f'dataset "{self._train_ds}" and '
                f'data sampler "{self._train_dl.batch_sampler}"'
            )

        # Create arbitrary text of sequence length
        seq_length = self.cfg.encoder_seq_length
        text = torch.ones(seq_length + 1, dtype=torch.int64) * 3545

        tokens = text[:-1].contiguous()
        tokens[-1] = 2

        labels = text[1:].contiguous()
        labels[-1] = 2

        attention_mask_shape = [self.cfg.micro_batch_size, seq_length, seq_length]
        attention_mask = torch.ones(attention_mask_shape, dtype=torch.bool)

        loss_mask = torch.ones(seq_length, dtype=torch.int64)
        loss_mask[-1] = 0

        position_ids = torch.tensor([i for i in range(seq_length)], dtype=torch.int64)
        position_ids[-1] = 0

        token_count = [seq_length - 1]

        single_data = {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
            "token_count": token_count,
            "attention_mask": attention_mask,
        }

        batch = default_collate([single_data] * self.cfg.micro_batch_size * get_num_microbatches())
        if validation:
            return repeat((batch, 0, 0))
        return repeat(batch)
