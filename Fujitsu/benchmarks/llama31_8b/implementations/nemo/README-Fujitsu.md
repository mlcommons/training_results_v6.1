
This document describes how to reproduce the llama3.1-8b benchmark results submitted by Fujitsu for MLPerf Training v6.1.
The benchmark execution environment and dataset are fundmentally same as those used in the submission by NVIDIA for
Training v6.0, with the following modifications:

- The container environment has been adjusted to fit Fujitsu's internal netowrk. These changes are defined in the comments
of `Dockerfile.fujitsu`.

- In v6.1, it is mandatory to include information regarding precision and parallelism in the results;
therefore, code has been added to `run.sub` to output this information.

The procedure for running the benchmark in this environment is as follows:

First, create the necessary directories and buildthe container image:
```
$ mkdir ../sqsh logs outputs
$ do_build.sh
$ do_export.sh
```

Next, edit `run_benchmark.sh` as follows:

- Edit the `SLURM_HOME` variable to point to your Slurm installation directory,
- Edit the `PMIX3_HOME` variable to point to your PMIx3 installation directory,
- Edit the `DATADIR` variable to point to the dataset directory, and
- Edit the `NEXP` variable to specify the number of benchmark runs.

```
$ vim run_benchmark.sh
```

Change the operating modes for the CPU and GPU by running two scripts.
These scripts does the following:

- Set the CPU scaling governor to performance, and
- Set the GPU boost slider to `vboost=1`.

```
$ bash set-cpugov.sh
$ bash set-vboost.sh
```

One the preparations are complete, run the benchmark:

```
$ bash run_benchmark.sh
```
