# Running Google DeepSeek-V3 671B MaxText (JAX) MLPerf Training Benchmark on TPU7x

This file contains the instructions for running the Google DeepSeek-V3 671B MaxText MLPerf
Training benchmark (closed division) on one slice of Google Cloud TPU7x (Ironwood) `tpu7x-8x32x32`.

## 1. Hardware Requirements

- One slice of TPU7x topology `8x32x32` (8192 chips per slice, ICI-connected): 8192 TPU7x chips on 2048 hosts (`tpu7x-standard-4t`,
  4 chips per host; each chip exposes 2 JAX devices, 16384 in total), 192 GiB HBM3e per chip,
  provisioned as GKE node pools.
- A Google Cloud Storage bucket in the same region as the slice for the dataset, the initial
  checkpoint and the run outputs.
- For the one-time checkpoint preparation (section 3.3): a CPU VM with at least 2.2 TB of RAM
  (e.g. `m1-ultramem-160`) and about 1.5 TB of local disk for the downloaded HuggingFace checkpoint.
  No TPU is needed for this step.
- For the one-time dataset preparation (section 3.2): any machine with about 310 GB of free disk.

## 2. Software Requirements

- [Google Kubernetes Engine](https://cloud.google.com/kubernetes-engine) cluster with the TPU7x node
  pool, created and driven with [XPK](https://github.com/AI-Hypercomputer/xpk) (JobSet + Kueue).
- `gcloud` CLI with access to the project, cluster, bucket and container registry.
- [Docker](https://www.docker.com/) to build the container image.
- Inside the container (versions used by the submitted runs; full list in
  `../maxtext/requirements_frozen.txt`): Python 3.12, JAX / jaxlib 0.11.1 / 0.11.1,
  libtpu 0.0.48, mlperf-logging 4.1.67.

## 3. Set up

Throughout, replace the angle-bracket placeholders (`<project>`, `<cluster>`, `<zone>`, `<bucket>`,
`<registry>`) with your own values. The package versions are the exact ones the submitted runs
used.

### 3.1 Build the container

The container image is built from public sources only, with `Dockerfile`:

- base: the public MaxText 0.2.5 pre-training image
  `us-docker.pkg.dev/cloud-tpu-images/maxtext-images/tpu_pre_training`, pinned by digest
  (Python 3.12 and the MaxText dependencies);
- the package versions of section 2, released on PyPI and pinned with `==`;
- the Orbax checkpoint-restore barrier timeout raised from 300 s to 1800 s (one `sed` on the
  installed `orbax-checkpoint`; restoring the 671B checkpoint on 2048 hosts can take longer);
- MaxText [`mlperf-dsv3-v6.1`](https://github.com/AI-Hypercomputer/maxtext/releases/tag/mlperf-dsv3-v6.1) (commit `24e75daf8e16`): the Dockerfile downloads that commit from GitHub and copies its `src/` to
  `/app/src` (the run command sets `PYTHONPATH=/app/src`). `../maxtext/SOURCE_VERSION.txt` names
  the release and `../maxtext/requirements_frozen.txt` lists the Python packages of the container.

`setup.sh` builds the image, pushes it as `WORKLOAD_IMAGE` and lists the GKE cluster-creation
commands:

```bash
export WORKLOAD_IMAGE=<your-registry>/mlperf-google:deepseekv3_671b-maxtext
PROJECT_ID=<project> CLUSTER_NAME=<cluster> ZONE=<zone> bash setup.sh
```

### 3.2 Prepare dataset

DeepSeek-V3 671B uses the same C4 dataset and Llama 3.1 8B tokenizer as the Llama 3.1 8B
benchmark. The reference data is the MLCommons `llama-3-1-8b-preprocessed-c4-dataset` bundle
(Megatron `.bin`/`.idx`, already tokenized, about 84 GB), which is converted once into TFDS-style
TFRecord shards for MaxText.

```bash
export DATADIR=<path/to/dataset>
export DATASET_PATH=gs://<bucket>/<dataset>/
# 0. Reference bundle (MLCommons R2 downloader)
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
  -d "$DATADIR/mlcommons" https://training.mlcommons-storage.org/metadata/llama-3-1-8b-preprocessed-c4-dataset.uri
# 1-3. Convert the train split, build the eval set, split the train shards (see data_scripts/README.md)
pip install -r data_scripts/requirements.txt
python3 data_scripts/build_dataset.py --src_dir="$DATADIR/mlcommons" --dst_dir="$DATADIR/tfds_2048" --stages=train,eval --num_workers=32
python3 data_scripts/split_shards.py --src_dir="$DATADIR/tfds_2048" --dst_dir="$DATADIR/tfds" --num_workers=32
gcloud storage cp -r "$DATADIR/tfds/c4" "${DATASET_PATH}"
```

(`init_datasets.sh` runs sections 3.2 and 3.3 end-to-end when `DATADIR`, `DATASET_PATH`, `HF_CKPT`, and `CKPT_BUCKET` are exported.)

What the scripts produce:

| Step | Output |
| --- | --- |
| 1. Convert | Train dataset: 2048 TFRecord shards, one record (`ids`, int32) per Megatron document |
| 2. Eval set | Eval dataset: 2048 TFRecord shards holding the first 1024 packed 4096-sequences of the reference validation documents (shard count is a multiple of 128: 128 shards x 8 sequences cover the 1024 eval sequences once; 2048 supports up to 2048 eval data-loading hosts) |
| 3. Split | 16384 train shards |

At the end, the directory structure should look like:

```bash
$tree -L 3 $DATADIR/tfds
$DATADIR/tfds
`-- c4
    `-- en
        |-- <train version>/           # train dataset: 16384 shards (config.yml dataset_name)
        |   |-- c4-train.tfrecord-00000-of-16384
        |   |-- ...
        |   |-- dataset_info.json
        |   `-- features.json
        `-- <eval version>/            # eval dataset: 2048 shards (config.yml eval_dataset_name)
            |-- c4-validation.tfrecord-00000-of-02048
            |-- ...
            |-- dataset_info.json
            `-- features.json
```

Training reads the train dataset (`dataset_name` in `config.yml`) and evaluation reads the eval dataset
(`eval_dataset_name`).

The tokenizer is included in the code tree: `src/maxtext/assets/tokenizers/tokenizer_llama3.tiktoken` of the MaxText commit
(`tokenizer_type: tiktoken`); the data is pre-tokenized, so it is only used for vocabulary size.

### 3.3 Model and checkpoint preparation

#### 3.3.1 Publication/Attribution

[MaxText](https://github.com/AI-Hypercomputer/maxtext) is Google's open-source, high-performance
LLM training framework written in pure Python/[JAX](https://github.com/jax-ml/jax) and targeting
Google Cloud TPUs. The MoE and attention kernels are written in [Pallas](https://docs.jax.dev/en/latest/pallas/index.html);
training runs on the [libtpu](https://cloud.google.com/tpu/docs/runtimes) runtime via XLA.

#### 3.3.2 Model Architecture

The model follows the [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) and the
MLCommons reference implementation.

| Parameter | Value |
|-----------|-------|
| Model Size | 671B parameters |
| Architecture | Mixture of Experts (256 routed + 1 shared expert, 8 active), Multi-head Latent Attention, 1 Multi-Token-Prediction layer |
| Sequence Length | 4096 |
| MaxText model config | `deepseek3-671b-lineage` (`src/maxtext/configs/models/` of the MaxText commit) |

#### 3.3.3 Checkpoint download

MLCommons hosts the checkpoint for download **exclusively by MLCommons Members** at
[MLCommons storage](https://training.mlcommons-storage.org/index.html#deepseekv3-benchmark).

To download, use the [MLCommons R2 Downloader](https://github.com/mlcommons/r2-downloader):

```bash
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
  -d <path/to/hf_checkpoint> https://training.mlcommons-storage.org/metadata/deepseekv3_checkpoint_bf16.uri
```

The checkpoint is ~1.4TB.

#### 3.3.4 Starting checkpoint

The checkpoint distributed for this benchmark is provided in HuggingFace format (bf16). It was
produced by MLCommons from the original HuggingFace DeepSeek-V3 671B checkpoint by training for
50 iterations with a sequence auxiliary load-balancing loss weight of 1e-2 and converting back to
HuggingFace format; the warm-up lets the MoE routers adapt to the Llama 3.1 8B tokenizer used by
this benchmark. We start from these weights unchanged.

#### 3.3.5 Checkpoint preprocessing

MaxText loads [Orbax](https://github.com/google/orbax) checkpoints, so the HuggingFace checkpoint is
converted once, in two steps, on the CPU VM of section 1, inside the container of section 3.1
(no TPU involved). Only the storage format and the sharding layout change; parameter values are
preserved (the bf16 weights are stored as float16 and the float32 MoE router bias stays float32,
as in the checkpoints the submitted runs loaded).

```bash
# 1. HuggingFace bf16 -> MaxText/Orbax (unsharded "base" checkpoint, ~1.05 TB, several hours)
python3 -m maxtext.checkpoint_conversion.standalone_scripts.convert_deepseek_family_ckpt \
  --model_size deepseek3-671b --enable_mtp true --weight_dtype keep \
  --base_model_path <path/to/hf_checkpoint> \
  --maxtext_model_path gs://<bucket>/<ckpt>/base \
  --simulated_cpu_devices_count 16
# 2. Reshard "base" into the layout the training mesh loads: the mesh and logical axis rules of the
#    submitted runs (ici_parallelism [1,8,32,32,2]; see init_datasets.sh for the complete flags;
#    on the CPU host VM, ensure vm.max_map_count >= 262144 for 16384 simulated CPU devices)
sudo sysctl -w vm.max_map_count=262144
python3 -m maxtext.checkpoint_conversion.reshard_checkpoint \
  model_name=deepseek3-671b use_lineage=False use_2d_fsdp_sharding=False ici_parallelism='[1,8,32,32,2]' ... \
  load_parameters_path=gs://<bucket>/<ckpt>/base/0/items \
  base_output_directory=gs://<bucket>/<ckpt>/resharded
export CHECKPOINT_PATH=gs://<bucket>/<ckpt>/resharded/0/items
```

`init_datasets.sh` holds the complete commands. The converter stores the two MoE router `gate/bias`
tensors under the `params` collection while the model requests them under `MoEBiasVar`, so the
checkpoint loader backfills them from `params` (`Checkpoint predates the 'MoEBiasVar' collection;
restoring its 2 weight(s) from the 'params' collection`) and then logs a benign name-mismatch
warning (`WARNING: 2 checkpoint weight(s) were not requested by the model and will be ignored ...
gate/bias`) for the same two paths; all weights are restored.

## 4. Launch training

Training runs as an XPK workload (one JobSet, one container per host) on the GKE cluster.

```bash
export PROJECT_ID=<project> CLUSTER_NAME=<cluster> ZONE=<zone>
export WORKLOAD_IMAGE=<as/set/above>          # section 3.1
export DATASET_PATH=<as/set/above>            # section 3.2
export CHECKPOINT_PATH=<as/set/above>         # section 3.3
export BASE_OUTPUT_DIR=gs://<bucket>/<output>  # run outputs: ${BASE_OUTPUT_DIR}/<workload>/
WORKLOAD_NAME=<workload-name> bash run_and_time.sh
```

`run_and_time.sh` submits `xpk workload create --docker-image="${WORKLOAD_IMAGE}"` with every XLA
flag, MaxText override and xpk flag of the submitted runs, then runs
`python3 -m maxtext.trainers.pre_train.train maxtext/configs/base.yml ${MAXTEXT_ARGS}`. Run it once
per result (the submission has three). MaxText writes the MLPerf log to
`${BASE_OUTPUT_DIR}/<workload-name>/mllog_<seed>.log`. Each
`results/tpu7x-8x32x32/deepseekv3_671b/result_*.txt` file of this submission is that MLLOG (every line
unchanged) merged in time order with the MaxText log lines of the run, built with
`postprocess_logs.sh`.

### Config

`config.yml` is the human-readable record of the MaxText settings that `run_and_time.sh` passes as
`MAXTEXT_ARGS` on top of the model config `deepseek3-671b-lineage.yml` (logged as `config_filename`
in the MLPerf logs). Global batch size: 16384 sequences (`per_device_batch_size: 1` x 16384 JAX
devices x `gradient_accumulation_steps: 1`); the run is capped at 100 training steps.

## 5. Quality

### Quality metric
Log Perplexity (eval loss)

### Quality target
3.6

### Evaluation frequency
Every training step (`eval_interval: 1`), starting after the 43rd training step
(`eval_start_step: 42`, 0-indexed); the run stops at the first evaluation at or below the target.

### Evaluation thoroughness
Evaluation on the validation subset that consists of 1024 sequences (= 4.2M tokens),
section 3.2 step 2.

## 6. Additional notes

### Parallelism

- **Mesh:** With `use_lineage: true`, MaxText shards the model and batch over a 5-axis device mesh
  `['dcn', 'x', 'y', 'z', 'core']` of shape ICI `[1, 8, 32, 32, 2]` (the `8x32x32` chip torus
  and the 2 TensorCores per chip) x DCN `[1, 1, 1, 1, 1]` (across the single slice, 16384 JAX
  devices total; `ici_parallelism` / `dcn_parallelism` in `config.yml`).
- **Activations & attention:** Input batch activations are sharded 8192-way over `('x', 'y', 'z')`
  (and `dcn` when multi-slice); `core` provides 2-way sequence parallelism on activations and
  2-way head parallelism in attention (`tensor_parallelism: 2` in the MLPerf logs).
- **MoE & FSDP weights:** The 256 routed MoE experts use 16-way expert parallelism over
  `('x', 'core')` (`expert_parallelism: 16`); weights are FSDP-sharded over `('y', 'z')` (1024-way
  for routed and shared experts, dense MLPs, and attention `q_up`/`k_up`/`v_up`/`out` projections;
  32-way over `'z'` for `q_down`/`kv_down`; and 2048-way over `('y', 'z', 'core')` for the
  embedding and LM head).
- **Microbatch interleaving:** Each training step accumulates gradients over
  `gradient_accumulation_steps: 1` micro-steps of 16384 sequences (`per_device_batch_size: 1` x
  16384 JAX devices = 16384 global batch size); in the sparse MoE layers, Lineage further splits
  each micro-step into 2 microbatches (8192 sequences each) and interleaves one microbatch's
  attention compute with the other's MoE dispatch/combine collectives and Pallas grouped-matmul
  expert compute to overlap communication and computation.

To run on a different system, change `--device-type` / `--num-slices` in `run_and_time.sh`,
`ici_parallelism` / `dcn_parallelism` so that their product equals the number of JAX devices, and reshard the checkpoint (section 3.3.5) to the
new mesh.

### Seeds

Each run draws one seed from `/dev/urandom` (`run_and_time.sh`, when `SEED` is unset) and passes it
as `data_shuffle_seed` (and `init_weights_seed`); it is logged as the MLPerf `seed` event and
appears in the MLPerf log file name.
