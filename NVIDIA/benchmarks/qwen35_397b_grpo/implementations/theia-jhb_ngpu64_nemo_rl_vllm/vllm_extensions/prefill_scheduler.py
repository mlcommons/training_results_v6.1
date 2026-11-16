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

"""Custom vLLM schedulers used by the optimized workload."""

from vllm.v1.core.sched.async_scheduler import AsyncScheduler


class _CadencedPrefillMixin:
    """Apply prefill cadence locally when vLLM data parallelism is disabled."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prefill_schedule_step_counter = 0

    def schedule(self, throttle_prefills: bool = False):
        is_non_dp = self.parallel_config.data_parallel_size == 1

        if is_non_dp:
            interval = self.scheduler_config.prefill_schedule_interval
            local_throttle = (
                interval > 1
                and self.prefill_schedule_step_counter % interval != 0
            )
            # Preserve a synchronized throttle decision from DPEngineCoreProc.
            throttle_prefills = throttle_prefills or local_throttle

        scheduler_output = super().schedule(throttle_prefills=throttle_prefills)

        # This matches EngineCore's model_executed condition. The counter is
        # deliberately continuous across idle periods.
        if is_non_dp and scheduler_output.total_num_scheduled_tokens > 0:
            self.prefill_schedule_step_counter += 1

        return scheduler_output


class CadencedPrefillAsyncScheduler(_CadencedPrefillMixin, AsyncScheduler):
    """Asynchronous vLLM scheduler with non-DP prefill cadence."""
