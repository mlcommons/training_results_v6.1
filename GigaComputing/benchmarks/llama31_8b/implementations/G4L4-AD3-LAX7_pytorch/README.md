# MLPerf Training Submission

This directory contains platform-specific scripts for the MLPerf Training benchmark.

The complete benchmark implementation, including source code, environment configuration, dependencies, and data preparation utilities, is located in the shared directory:

`../pytorch/`

## Files

- `setup.sh`: Prepares the platform-specific container environment.
- `run_and_time.sh`: Loads the system configuration and launches the benchmark using the shared implementation.

## Setup and Execution

```bash
bash setup.sh
bash run_and_time.sh
```

## Additional Information

For detailed instructions on environment setup, dataset and model preparation, and benchmark execution, please refer to:

`../pytorch/README.md`

All system-specific scripts use the shared implementation to avoid source code duplication across different hardware configurations.

