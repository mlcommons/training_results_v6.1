# MLPerf Training v6.1 Google Submission

This directory holds Google's closed division submission to the MLPerf Training
v6.1 round, in the `Available cloud` category. The reference implementations are
at https://github.com/mlcommons/training.

## Benchmarks

<!-- mdformat off(GitHub renders this file: keep single-line table rows) -->

Benchmark          | System                  | Implementation                    | Accelerators                          | README
------------------ | ----------------------- | --------------------------------- | ------------------------------------- | ------
`deepseekv3_671b`  | `tpu7x-8x16x16`         | `maxtext` (MaxText, JAX)          | 2,048 TPU7x (one slice)               | [`benchmarks/deepseekv3_671b/implementations/tpu7x-8x16x16/README.md`](benchmarks/deepseekv3_671b/implementations/tpu7x-8x16x16/README.md)
`deepseekv3_671b`  | `tpu7x-8x16x32`         | `maxtext` (MaxText, JAX)          | 4,096 TPU7x (one slice)               | [`benchmarks/deepseekv3_671b/implementations/tpu7x-8x16x32/README.md`](benchmarks/deepseekv3_671b/implementations/tpu7x-8x16x32/README.md)
`deepseekv3_671b`  | `tpu7x-8x32x32`         | `maxtext` (MaxText, JAX)          | 8,192 TPU7x (one slice)               | [`benchmarks/deepseekv3_671b/implementations/tpu7x-8x32x32/README.md`](benchmarks/deepseekv3_671b/implementations/tpu7x-8x32x32/README.md)
`qwen35_397b_grpo` | `tpu7x-4x4x8-128x2x2x1` | `tunix` (Tunix + MaxText + vLLM)  | 640 TPU7x (128 trainer, 512 rollout)  | [`benchmarks/qwen35_397b_grpo/implementations/tpu7x-4x4x8-128x2x2x1/README.md`](benchmarks/qwen35_397b_grpo/implementations/tpu7x-4x4x8-128x2x2x1/README.md)

<!-- mdformat on -->

## Contents

*   `systems/<system>.json`: the description of each system.
*   `benchmarks/<benchmark>/implementations/<framework>/`: the code that ran,
    with `SOURCE_VERSION.txt` and `requirements_frozen.txt`, which name the
    source version (release tag and commit) and list the Python packages of the
    container.
*   `benchmarks/<benchmark>/implementations/<system>/`: the launch scripts and
    configuration of the benchmark on the system, and a `README.md` that lists
    the steps to build the container and reproduce the results.
*   `results/<system>/<benchmark>/`: one `result_<i>.txt` file per run, with the
    run's MLLOG lines and its framework logs.

## Terms

*   **TPU7x (Ironwood)**: Google's seventh-generation TPU. Each chip has 192 GiB
    of HBM3e and 2 TensorCores, which JAX sees as 2 devices; each host has 4
    chips.
*   **System names**: the TPU7x topology of the slices a benchmark runs on.
    `tpu7x-8x16x16`, `tpu7x-8x16x32` and `tpu7x-8x32x32` are one slice of
    8x16x16, 8x16x32 and 8x32x32 chips (2,048, 4,096 and 8,192 chips).
    `tpu7x-4x4x8-128x2x2x1` is one trainer slice of 4x4x8 chips (128 chips) and
    128 rollout slices of 2x2x1 chips (4 chips each, 512 chips).
*   **MaxText**: the JAX LLM library
    ([AI-Hypercomputer/maxtext](https://github.com/AI-Hypercomputer/maxtext)).
*   **Tunix**: the JAX post-training library
    ([google/tunix](https://github.com/google/tunix)) that runs the GRPO loop.
*   **Pathways**: Google Cloud's runtime that lets a single JAX client run
    programs on multi-host TPU slices.
*   **Raiden** (`tpu_sync_jax`,
    [google/tpu-sync](https://github.com/google/tpu-sync)): transfers the
    updated weights from the trainer to the rollout slices over the data center
    network (DCN) after each step.
*   **XPK, JobSet, Kueue and agent-sandbox**: tools and Kubernetes controllers
    for running TPU workloads on GKE
    ([XPK](https://github.com/AI-Hypercomputer/xpk)), groups of jobs
    ([JobSet](https://github.com/kubernetes-sigs/jobset)), job queueing
    ([Kueue](https://github.com/kubernetes-sigs/kueue)), and isolated task
    sandboxes
    ([agent-sandbox](https://github.com/kubernetes-sigs/agent-sandbox)).

## Running benchmarks

These benchmarks ran on Google Cloud TPU7x on Google Kubernetes Engine (GKE).
Generally, a benchmark runs with these steps; each system README has the
details:

1.  Build the container image and push it to a registry.
2.  Set up the cluster and prepare the dataset and the checkpoint.
3.  Run `run_and_time.sh` from the system directory, once per result.
4.  Build each result file from the run's logs with `postprocess_logs.sh`.
