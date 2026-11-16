# Copyright (c) 2024-2026, NVIDIA CORPORATION. All rights reserved.
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


import time

from megatron.bridge.training.callbacks import Callback, CallbackContext
from megatron.core.full_cuda_graph import FullCudaGraphWrapper
from mlperf_common.frameworks.pyt import PyTCommunicationHandler
from mlperf_common.logging import MLLoggerWrapper


class DeltaTimer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.start_time = time.perf_counter()
        return self.start_time

    def get_delta(self):
        prev_time = self.start_time
        return self.reset() - prev_time


mllogger = MLLoggerWrapper(PyTCommunicationHandler())


class MLPerfLoggingCallback(Callback):
    """MLPerf logging callback."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.global_batch_size = self.cfg.model.global_batch_size
        self.train_block_started = False
        self.train_block_samples_count = None
        self.train_current_block = 0
        self.force_success = cfg.custom.force_success_status

    def on_data_init_start(self, context: CallbackContext):
        mllogger.log_init_stop_run_start()

    def on_train_start(self, context: CallbackContext):
        context.state.should_stop = False
        self._start_train_block(
            context.state,
            block_steps=context.state.cfg.validation.start_eval_at_iter,
        )
        self.timer = DeltaTimer()

    def on_train_end(self, context: CallbackContext):
        if self.train_block_started:
            self._end_train_block(context.state)

        # Clear the entries in place instead of rebinding the attribute: mcore and
        # mbridge both expect cuda_graph to stay a dict, and mbridge >= 4628f3e64
        # re-reads it in _finish_train() after this hook has run.
        for stage in FullCudaGraphWrapper.cuda_graph:
            FullCudaGraphWrapper.cuda_graph[stage] = None

    def on_eval_start(self, context: CallbackContext):
        """Log validation start."""
        if hasattr(context.state, "warmup") and context.state.warmup:
            return
        self.timer.reset()
        if self.train_block_started:
            self._end_train_block(context.state)

        mllogger.start(
            mllogger.constants.EVAL_START,
            metadata={
                mllogger.constants.SAMPLES_COUNT: self._get_samples_count(context.state),
                "step": self._get_step(context.state),
            },
        )

    def on_eval_end(self, context: CallbackContext):
        if hasattr(context.state, "warmup") and context.state.warmup:
            return
        self._log_custom_timedelta("validation_time", self._get_step(context.state))

        samples_count = self._get_samples_count(context.state)
        loss = context.total_loss_dict["lm loss"].item()

        mllogger.event(
            key=mllogger.constants.EVAL_ACCURACY,
            metadata={mllogger.constants.SAMPLES_COUNT: samples_count},
            value=loss,
        )
        mllogger.end(
            mllogger.constants.EVAL_STOP,
            metadata={
                mllogger.constants.SAMPLES_COUNT: samples_count,
                "step": self._get_step(context.state),
            },
        )

        if loss < self.cfg.custom.target_log_ppl:
            context.state.should_stop = True
            mllogger.end(
                mllogger.constants.RUN_STOP,
                metadata={mllogger.constants.SAMPLES_COUNT: samples_count, "status": "success"},
            )
        elif context.state.train_state.step >= self.cfg.trainer.max_steps:
            context.state.should_stop = True
            status = "success" if self.force_success else "aborted"
            mllogger.end(
                mllogger.constants.RUN_STOP,
                metadata={mllogger.constants.SAMPLES_COUNT: samples_count, "status": status},
            )

        if not context.state.should_stop:
            self._start_train_block(context.state)
        else:
            context.state.train_state.step = self.cfg.trainer.max_steps + 1
            context.state.train_state.do_valid = False
            context.state.train_state.do_test = False

    def on_train_step_end(self, context: CallbackContext):
        # train_state.step has not advanced to the completed step yet in this callback.
        step = context.state.train_state.step + 1
        last_step = step >= self.cfg.trainer.max_steps
        validation_cfg = context.state.cfg.validation
        eval_after_this_step = (
            step % validation_cfg.eval_interval == 0
            and (
                validation_cfg.start_eval_at_iter is None
                or step >= validation_cfg.start_eval_at_iter
            )
        )
        if last_step and not eval_after_this_step:
            samples_count = step * self.global_batch_size
            status = "success" if self.force_success else "aborted"
            self._end_train_block(context.state, step=step)
            mllogger.end(
                mllogger.constants.RUN_STOP,
                metadata={mllogger.constants.SAMPLES_COUNT: samples_count, "status": status},
            )
            self.train_block_started = False
            context.state.should_stop = True
            context.state.train_state.do_valid = False
            context.state.train_state.do_test = False

    def _start_train_block(self, global_state, block_steps=None) -> None:
        start_step = self._get_step(global_state)
        remaining_steps = max(self.cfg.trainer.max_steps - start_step, 0)
        if block_steps is None:
            block_steps = global_state.cfg.validation.eval_interval
        block_steps = min(block_steps, remaining_steps)

        self.train_block_started = True
        self.train_block_samples_count = block_steps * self.global_batch_size
        mllogger.start(
            mllogger.constants.BLOCK_START,
            metadata={
                mllogger.constants.SAMPLES_COUNT: self.train_block_samples_count,
                "step": start_step,
            },
        )

    def _end_train_block(self, global_state, step=None) -> None:
        assert self.train_block_samples_count is not None
        if step is None:
            step = self._get_step(global_state)
        mllogger.end(
            mllogger.constants.BLOCK_STOP,
            metadata={
                mllogger.constants.SAMPLES_COUNT: self.train_block_samples_count,
                "step": step,
            },
        )
        self.train_block_started = False
        self.train_block_samples_count = None

    def _log_custom_timedelta(self, value_key, step: int = 0):
        mllogger.event(
            key="tracked_stats",
            metadata={"step": step},
            value={value_key: self.timer.get_delta()},
        )

    def _get_step(self, global_state):
        return global_state.train_state.step

    def _get_samples_count(self, global_state):
        return self._get_step(global_state) * self.global_batch_size


class DeltaTimingCallback(Callback):
    def __init__(self, cfg):
        self.t0 = 0
        self.total_train_step_time = [0, 0]
        self.global_batch_size = cfg.model.global_batch_size
        self.log_every_n_steps = cfg.trainer.log_every_n_steps

    def on_train_start(self, context: CallbackContext):
        self.t0 = time.time()

    def on_train_step_end(self, context: CallbackContext):
        t1 = time.time()
        d = t1 - self.t0
        self.total_train_step_time[0] += d
        self.total_train_step_time[1] += 1
        self.t0 = t1

        step = context.state.train_state.step + 1
        if step % self.log_every_n_steps == 0:
            mllogger.event(
                key="tracked_stats",
                metadata={mllogger.constants.SAMPLES_COUNT: self.global_batch_size * step},
                value={
                    "train_step_time": d,
                    "reduced_train_loss": context.loss_dict["lm loss"].item(),
                },
            )

    def on_eval_end(self, context: CallbackContext):
        """Reset timer after validation to avoid including validation time in first train step."""
        self.t0 = time.time()
