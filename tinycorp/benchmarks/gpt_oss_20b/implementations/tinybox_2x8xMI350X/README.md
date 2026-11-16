# 1. Problem

small moe llm pretraining: gpt-oss 20b on c4.

## Requirements

Install tinygrad from branch `mlperf_gptoss_2x_v61 @ 7f16d455b2a66d7484ea020fa469883fe6c4fabf`.
```
git clone https://github.com/tinygrad/tinygrad.git -b mlperf_gptoss_2x_v61
pip install -e ".[testing]"
```

Install latest `mlperf_logging`.

# 2. Directions

## Steps to download and verify data

### 1. Download raw data

follow mlperf steps to download the preprocessed c4 dataset.

## Running

### tinybox_8xMI350X

#### Steps to run benchmark
```
# ensure amdgpu is unloaded on all nodes

# run setup on both nodes
examples/mlperf/training_submission_v6.1/tinycorp/benchmarks/gpt_oss/implementations/tinybox_2x8xMI350X/setup.sh down

# start server on one nodes
PYTHONPATH=. DEV=PCI+AMD python extra/remote/serve.py 6667

# start training on the other node
examples/mlperf/training_submission_v6.1/tinycorp/benchmarks/gpt_oss/implementations/tinybox_2x8xMI350X/run_and_time.sh
```
