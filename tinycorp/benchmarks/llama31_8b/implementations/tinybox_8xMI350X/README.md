# 1. Problem

small llm pretraining: llama 3.1 8b on c4.

## Requirements

Install tinygrad from branch `mlperf_llama_v61 @ a36430df1e4895265a93a1e39ba98d4966155603`.
```
git clone https://github.com/tinygrad/tinygrad.git -b mlperf_llama_v61
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

examples/mlperf/training_submission_v6.1/tinycorp/benchmarks/llama31_8b/implementations/tinybox_8xMI350X/run_and_time.sh
```
