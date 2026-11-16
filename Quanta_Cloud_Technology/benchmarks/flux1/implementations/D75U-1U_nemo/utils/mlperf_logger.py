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

"""MLPerf Training 6.1 compliance logging for Flux.1.

Emits all required keys for closed-division flux1 compliance: common.yaml: submission_*, init_start/stop, run_start/stop, block_start/stop, eval_start/stop, eval_accuracy, train_samples, eval_samples closed_common.yaml: submission_benchmark, gradient_accumulation_steps closed_flux1.yaml: global_batch_size, evaluation_frequency, opt_*
"""

import os
from mlperf_logging import mllog
from mlperf_logging.mllog import constants


def get_mllogger():
    """Get configured mllogger instance."""
    mllogger = mllog.get_mllogger()
    mllog.config(filename=os.path.join(
        os.environ.get("RESULTS_DIR", "/results"),
        "result.txt",
    ))
    return mllogger


def log_init_start(mllogger, rank):
    if rank == 0:
        mllogger.start(key=constants.INIT_START)


def log_submission_info(mllogger, rank, num_nodes=None):
    """Mirror of mlperf_common's mlperf_submission_log, which flux1 cannot call directly: it takes a comm_handler, this module logs with explicit ranks.

    Two keys that helper emits are deliberately absent -- submission_status (dead since ruleset 4.1.0) and cache_clear (run.sub logs it where the caches are actually dropped).
    """
    if rank != 0:
        return
    if num_nodes is None:
        num_nodes = os.environ.get("SLURM_JOB_NUM_NODES", 1)

    org = os.environ.get("MLPERF_SUBMISSION_ORG", "SUBMISSION_ORG_PLACEHOLDER")
    platform = os.environ.get("MLPERF_SUBMISSION_PLATFORM",
                              "SUBMISSION_PLATFORM_PLACEHOLDER")

    mllogger.event(key=constants.SUBMISSION_BENCHMARK, value="flux1")
    mllogger.event(key=constants.SUBMISSION_ORG, value=org)
    mllogger.event(key=constants.SUBMISSION_DIVISION, value=constants.CLOSED)
    mllogger.event(key=constants.SUBMISSION_PLATFORM,
                   value=f"{num_nodes}x{platform}")


def log_target_accuracy(mllogger, rank, target):
    if rank == 0:
        mllogger.event(key="target_accuracy", value=target)


def log_hyperparams(mllogger, rank, hparams, num_nodes, gpus_per_node, gbs,
                     lr, warmup_steps, val_check_interval,
                     train_samples=1099776, eval_samples=29696):
    if rank != 0:
        return
    world_size = num_nodes * gpus_per_node
    mbs = gbs // world_size
    grad_accum = max(gbs // (mbs * world_size), 1)

    mllogger.event(key=constants.GLOBAL_BATCH_SIZE, value=gbs)
    mllogger.event(key=constants.TRAIN_SAMPLES, value=train_samples)
    mllogger.event(key=constants.EVAL_SAMPLES, value=eval_samples)
    mllogger.event(key="evaluation_frequency", value=val_check_interval)
    mllogger.event(key=constants.GRADIENT_ACCUMULATION_STEPS, value=grad_accum)
    mllogger.event(key=constants.MICRO_BATCH_SIZE, value=mbs)
    mllogger.event(key=constants.TENSOR_PARALLELISM, value=1)
    mllogger.event(key=constants.PIPELINE_PARALLELISM, value=1)
    mllogger.event(key=constants.CONTEXT_PARALLELISM, value=1)
    mllogger.event(key=constants.EXPERT_PARALLELISM, value=1)
    mllogger.event(key=constants.CONFIG_FILENAME,
                   value=f"config_{os.environ.get('DGXSYSTEM', 'UNKNOWN')}.sh")
    mllogger.event(key=constants.OPT_NAME, value=constants.ADAMW)
    mllogger.event(key=constants.OPT_BASE_LR, value=lr)
    mllogger.event(key=constants.OPT_LR_WARMUP_STEPS, value=warmup_steps)
    mllogger.event(key=constants.OPT_ADAMW_BETA_1, value=hparams["beta1"])
    mllogger.event(key=constants.OPT_ADAMW_BETA_2, value=hparams["beta2"])
    mllogger.event(key=constants.OPT_ADAMW_EPSILON, value=hparams["eps"])
    mllogger.event(key=constants.OPT_ADAMW_WEIGHT_DECAY,
                   value=hparams["weight_decay"])
    mllogger.event(key=constants.OPT_GRADIENT_CLIP_NORM,
                   value=hparams["max_grad_norm"])


# Allowed values for the lowest_numerical_precision_in_* keys, mirrored from
# mlperf_logging/compliance_checker/training_6.1.0/common.yaml.
_MLPERF_PRECISIONS = frozenset({
    "fp64", "fp32", "tf32", "fp16", "fp8", "nvfp4", "mxfp4", "bfloat16",
    "Graphcore FLOAT 16.16", "int8", "uint8", "int4", "uint4",
})


# Ascending precision order, lowest first. Used to pick the value the
# lowest_numerical_precision_in_* keys must disclose when a layer runs at more
# than one precision. Entries absent here sort last (treated as high
# precision); the allow-list check below is what rejects unknown values.
_PRECISION_RANK = {
    p: i for i, p in enumerate((
        "int4", "uint4", "nvfp4", "mxfp4", "int8", "uint8", "fp8", "fp16",
        "bfloat16", "Graphcore FLOAT 16.16", "tf32", "fp32", "fp64",
    ))
}


def _lowest_linear_precision():
    """The lowest precision any linear-layer GEMM runs at.

    The forward GEMM uses $MLPERF_LINEAR_PRECISION; dgrad and wgrad use $MLPERF_GRAD_PRECISION when set, else the same. A hybrid recipe (MXFP8 forward, NVFP4 backward) therefore runs *some* linear GEMMs below the forward precision, and the key is `lowest_numerical_precision_in_linear` -- so it must disclose the backward format, not the forward one.
    """
    fwd = os.environ.get("MLPERF_LINEAR_PRECISION", "").strip()
    grad = os.environ.get("MLPERF_GRAD_PRECISION", "").strip() or fwd
    if not fwd:
        return grad
    if not grad:
        return fwd
    return min(fwd, grad, key=lambda p: _PRECISION_RANK.get(p, len(_PRECISION_RANK)))


def log_numerical_precision(mllogger, rank):
    """Emit the three lowest_numerical_precision_in_* disclosure keys.

    ATTN and COMM come straight from $MLPERF_{ATTN,COMM}_PRECISION -- the same env-var contract the other NVIDIA benchmarks use. LINEAR is the *lowest* of the forward and backward GEMM formats (see `_lowest_linear_precision`), because a hybrid recipe runs them at different precisions.

    training_6.1.0 made all three REQ: EXACTLY_ONE (they were OPTIONAL in 6.0.0), so an unset or off-allow-list value fails compliance either way. Emit regardless and warn, so the run log names the bad env var rather than leaving it to be inferred from the checker.
    """
    if rank != 0:
        return
    for key, env in (
        (constants.LOWEST_NUMERICAL_PRECISION_IN_LINEAR,
         "MLPERF_LINEAR_PRECISION/MLPERF_GRAD_PRECISION"),
        (constants.LOWEST_NUMERICAL_PRECISION_IN_ATTN, "MLPERF_ATTN_PRECISION"),
        (constants.LOWEST_NUMERICAL_PRECISION_IN_COMM, "MLPERF_COMM_PRECISION"),
    ):
        if key == constants.LOWEST_NUMERICAL_PRECISION_IN_LINEAR:
            value = _lowest_linear_precision()
        else:
            value = os.environ.get(env, "").strip()
        if not value:
            print(f"[mlperf_logger] ${env} is unset; {key} is mandatory in "
                  f"v6.1 and will fail compliance", flush=True)
        if value not in _MLPERF_PRECISIONS:
# Emit anyway: the compliance checker is the authority, and a loud
# failure there beats silently dropping a disclosure.
            print(f"[mlperf_logger] ${env}={value!r} is not in the MLPerf "
                  f"allow-list; the compliance checker will reject it",
                  flush=True)
        mllogger.event(key=key, value=value)


def log_opt_lr_training_steps(mllogger, rank, max_steps):
    if rank == 0:
        mllogger.event(key=constants.OPT_LR_TRAINING_STEPS, value=max_steps)


def log_seed(mllogger, rank, seed):
    if rank == 0:
        mllogger.event(key=constants.SEED, value=seed)


def log_init_stop(mllogger, rank):
    if rank == 0:
        mllogger.end(key=constants.INIT_STOP)


def log_run_start(mllogger, rank):
    if rank == 0:
        mllogger.start(key=constants.RUN_START)


def log_run_stop(mllogger, rank, status, step, samples_count):
    if rank == 0:
        mllogger.end(key=constants.RUN_STOP,
                     metadata={
                         "step": step,
                         constants.SAMPLES_COUNT: samples_count,
                         "status": status,
                     })


def log_block_start(mllogger, rank, block_samples_count, step):
    """Log BLOCK_START. block_samples_count = block size (val_check_interval)."""
    if rank == 0:
        mllogger.start(key=constants.BLOCK_START,
                       metadata={
                           constants.SAMPLES_COUNT: block_samples_count,
                           "step": step,
                       })


def log_block_stop(mllogger, rank, block_samples_count, step):
    """Log BLOCK_STOP. block_samples_count = block size (val_check_interval)."""
    if rank == 0:
        mllogger.end(key=constants.BLOCK_STOP,
                     metadata={
                         constants.SAMPLES_COUNT: block_samples_count,
                         "step": step,
                     })


def log_eval_start(mllogger, rank, samples_count, step):
    if rank == 0:
        mllogger.start(key=constants.EVAL_START,
                       metadata={
                           constants.SAMPLES_COUNT: samples_count,
                           "step": step,
                       })


def log_eval_stop(mllogger, rank, samples_count, step):
    if rank == 0:
        mllogger.end(key=constants.EVAL_STOP,
                     metadata={
                         constants.SAMPLES_COUNT: samples_count,
                         "step": step,
                     })


def log_eval_accuracy(mllogger, rank, val_loss, samples_count):
    if rank == 0:
        mllogger.event(key=constants.EVAL_ACCURACY,
                       value=val_loss,
                       metadata={constants.SAMPLES_COUNT: samples_count})


def log_tracked_stats(mllogger, rank, train_loss, lr, step, samples_count,
                       step_time):
    if rank == 0:
        mllogger.event(
            key="tracked_stats",
            value={"train_loss": train_loss, "train_step_time": step_time},
            metadata={
                constants.SAMPLES_COUNT: samples_count,
                "lr": lr,
                "step": step,
            },
        )
