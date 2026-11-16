# 1. Problem

small moe llm pretraining: gpt-oss 20b on c4.

## Requirements

Install tinygrad from branch `mlperf_gptoss_1x_v61 @ f8d2469c11aaad64039498d01f8ba0b5130eb9ab`.
```
git clone https://github.com/tinygrad/tinygrad.git -b mlperf_gptoss_1x_v61
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

examples/mlperf/training_submission_v6.1/tinycorp/benchmarks/gpt_oss_20b/implementations/tinybox_8xMI350X/run_and_time.sh
```
