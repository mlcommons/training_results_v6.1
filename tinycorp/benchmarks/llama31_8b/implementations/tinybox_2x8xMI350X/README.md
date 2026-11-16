# 1. Problem

small moe llm pretraining: llama 3.1 8b on c4.

## Requirements

Install tinygrad from branch `mlperf_llama_2x_v61 @ a63658f6ec65e092a2947c2206ccbbb1700819d2`.
```
git clone https://github.com/tinygrad/tinygrad.git -b mlperf_llama_2x_v61
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
examples/mlperf/training_submission_v6.1/tinycorp/benchmarks/llama31_8b/implementations/tinybox_2x8xMI350X/setup.sh

# start server on one nodes
PYTHONPATH=. DEV=PCI+AMD python extra/remote/serve.py 6667

# start training on the other node
examples/mlperf/training_submission_v6.1/tinycorp/benchmarks/llama31_8b/implementations/tinybox_2x8xMI350X/run_and_time.sh
```
