## Running NVIDIA Large MoE DeepSeek v3 671B PyTorch MLPerf Benchmark on Google Cloud A4X Max (GB300 NVL72)

This implementation is NVIDIA's MLPerf Training v6.1 DeepSeek v3 671B implementation, run unmodified in model, data pipeline and hyperparameters on Google Cloud A4X Max.

## 1. Hardware Requirements

- 64 x Google Cloud A4X Max nodes (`a4x-maxgpu-4g-metal`): NVIDIA GB300 NVL72, 4 GPUs and 2 Grace CPUs per node.
- The 64 nodes span 4 NVLink domains, 16 nodes per domain. The config places each pipeline stage inside one NVLink domain, so the job must get exactly 16 nodes from each of 4 domains.
- Per-node local NVMe SSD of at least 1.5TB for the checkpoint (~1.4TB), dataset (~80GB) and container image. The run itself uses only node-local storage for data; a small shared filesystem (e.g. NFS) is needed for the code, logs and the dataset sample index.
- GPUs are not required for dataset preparation.

## 2. Software Requirements

- Slurm with [Pyxis](https://github.com/NVIDIA/pyxis) and [Enroot](https://github.com/NVIDIA/enroot), deployed with [Google Cloud Cluster Toolkit](https://github.com/GoogleCloudPlatform/cluster-toolkit)
- [Docker](https://www.docker.com/) on an arm64 machine (e.g. one of the compute nodes) to build the container
- `gcloud` CLI on the compute nodes for staging data from Cloud Storage

## 3. Set up

### 3.1 Cluster

Deploy a Slurm cluster with Cluster Toolkit, one static nodeset per NVLink domain, all in a single partition. Each node carries a Slurm feature naming its NVLink domain (`nvl0` ... `nvl3`), which can be used to request 16 nodes per domain.

### 3.2 Build the container

Build on an arm64 host (the image is `linux/arm64`) and convert it to an Enroot squashfs:

```bash
docker build -t <docker/registry>/mlperf-nvidia:deepseekv3_671b-pyt .
# optionally: docker push <docker/registry>/mlperf-nvidia:deepseekv3_671b-pyt
enroot import -o deepseekv3_671b-arm.sqsh dockerd://<docker/registry>/mlperf-nvidia:deepseekv3_671b-pyt
```

The Transformer Engine build in the Dockerfile compiles many CUDA kernels in parallel; the build host needs enough free disk under the Docker data root (several hundred GB) and memory.

### 3.3 Prepare dataset

**DeepSeek v3 671B uses the same dataset as LLama3.1 8b!**
* **If you already have the LLama3.1 8b dataset you can skip this part**
* **If you don't, please mind that there are pointers/names related to 8b - this has been done on purpose**

```bash
export DATADIR=<path/to/dataset>
export HF_DIR="$DATADIR"/config   # the script downloads the HuggingFace model config here
bash data_scripts/download.sh
```

At the end, `$DATADIR/8b/` contains the preprocessed C4 files and the tokenizer, and `$DATADIR/config/` contains the HuggingFace DeepSeek V3 model config:

```bash
$DATADIR/
|-- 8b
|   |-- LICENSE.txt
|   |-- NOTICE.txt
|   |-- c4-train.en_6_text_document.bin
|   |-- c4-train.en_6_text_document.idx
|   |-- c4-validation-91205-samples.en_text_document.bin
|   |-- c4-validation-91205-samples.en_text_document.idx
|   |-- llama-3-1-8b-preprocessed-c4-dataset.md5
|   `-- tokenizer
`-- config
    |-- hub/models--deepseek-ai--DeepSeek-V3/...
    `-- modules/transformers_modules/...
```

### 3.4 Model checkpoint

MLCommons hosts the checkpoint for download **exclusively by MLCommons Members** at [MLCommons storage](https://training.mlcommons-storage.org/index.html#deepseekv3-benchmark). Download it with the [MLCommons R2 Downloader](https://github.com/mlcommons/r2-downloader):

```bash
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
  -d deepseekv3_checkpoint_bf16_megatron_bridge \
  https://training.mlcommons-storage.org/metadata/deepseekv3_checkpoint_bf16_megatron_bridge.uri
```

The checkpoint (~1.4TB) is a Megatron-Bridge distributed checkpoint (`latest_train_state.pt` and `iter_0000000/`). It was produced by loading the original HuggingFace DeepSeek V3 671B checkpoint into Megatron-Bridge and training for **50 iterations** with a sequence auxiliary load balancing loss weight of **1e-2**, so that the MoE router adapts to the Llama 3.1 8B tokenizer used by this benchmark.

### 3.5 Stage data to node-local SSD

Upload the dataset, checkpoint and squashfs image to a Cloud Storage bucket once, then copy them to the local SSD of every node with the same path on all nodes:

```bash
# once
gcloud storage rsync -r $DATADIR                    gs://<bucket>/mlperf-v6.1-deepseekv3/dataset
gcloud storage rsync -r <dir with checkpoint>       gs://<bucket>/mlperf-v6.1-deepseekv3/checkpoints
gcloud storage cp deepseekv3_671b-arm.sqsh          gs://<bucket>/mlperf-v6.1-deepseekv3/images/

# on every node (e.g. one Slurm job per NVLink domain, one task per node)
STAGE=/mnt/localssd/$USER/mlperf-v6.1-deepseekv3
mkdir -p $STAGE
# when many nodes share one NFS home, keep gcloud rsync state per node
export CLOUDSDK_STORAGE_RSYNC_FILES_DIRECTORY=$STAGE/gcloud-state/rsync
export CLOUDSDK_STORAGE_TRACKER_FILES_DIRECTORY=$STAGE/gcloud-state/trackers
for d in dataset checkpoints images; do
  gcloud storage rsync -r gs://<bucket>/mlperf-v6.1-deepseekv3/$d $STAGE/$d
done
```

## 4. Launch training

`launch_gcp_64xGB300.sh` sources `config_GB300_64x4x240xtp1pp4ep32cp1_mxfp8_full_cg.sh`, sets the Google Cloud specific environment, and submits `run.sub`:

```bash
export LOGDIR=<shared/output/dir>      # must be on a shared filesystem
export NODELIST=<16 nodes from each of the 4 NVLink domains>
./launch_gcp_64xGB300.sh
```

What the launch script sets on top of the NVIDIA config:

| Setting | Reason |
|---|---|
| `CONT`, `DATADIR`, `LOAD_CHECKPOINTS_DIR` under `/mnt/localssd/$USER/mlperf-v6.1-deepseekv3` | data, checkpoint and image are read from node-local SSD |
| `LOAD_CHECKPOINT=/checkpoints/deepseekv3_checkpoint_bf16_megatron_bridge` | container path of the starting checkpoint |
| `NPY_INDEX_DIR=$LOGDIR/npy_index` | rank 0 builds the dataset sample index and all ranks read it, so it must be on shared storage; a fresh directory is used per run |
| `HF_HOME=/hf_home`, `EXTRA_MOUNTS=$DATADIR/config:/hf_home:ro` | HuggingFace model config, offline |
| `NCCL_SOCKET_IFNAME==eth0`, `NCCL_SOCKET_FAMILY=AF_INET` | NCCL bootstrap over the host network (`eth0`, IPv4); the RDMA NICs carry only RDMA traffic |
| `NVTE_ALLOW_NONDETERMINISTIC_ALGO=1` | the fused grouped-MLP backward in this Transformer Engine version has no bit-exact (deterministic) path for the SwiGLU router scale gradient and raises when determinism is requested |
| `BF16_EVAL=0`, `WARMUP_MOCK_PIPELINE_COMPILE_MODE=off` | evaluation precision and warmup settings used for the submitted runs |
| `MLPERF_SUBMISSION_ORG`, `MLPERF_SUBMISSION_PLATFORM` | submission metadata in the MLPerf log |
| `SEED` / `SEED_BASE` | a fresh random seed per run from `secrets.randbelow` (system randomness); recorded in `$LOGDIR/launch-seed.txt` and in the MLPerf log |

Inter-node traffic uses NVLink (MNNVL) inside each NVLink domain and RoCE through Google's gIB NCCL network plugin (loaded from `/opt/gcp/nccl-plugins`) across domains.

Each run writes the MLPerf result log to `$LOGDIR/<DATESTAMP>_1.log`.

## 5. Quality

### Quality metric
Log Perplexity

### Quality target
3.6

### Evaluation frequency
First evaluation at `GBS * floor(42 + 24576 / GBS)` samples, then every step. At GBS=15360 this is step 43 (660,480 samples).

### Evaluation thoroughness
Evaluation on the validation subset that consists of 1024 sequences (=4.2M tokens).

## 6. Additional notes

### Config naming convention

Configuration files follow the format `config_<SYSTEM_NAME>_<NODES>x<GPUS/NODE>x<BATCH/DP-RANK>xtp<TP>pp<PP>ep<EP>cp<CP>[_<precision>][_full_cg].sh`, where TP is tensor parallel, PP is pipeline parallel, EP is expert parallel, CP is context parallel, `<precision>` is `mxfp8` or `fp4` (bf16 when absent) and `_full_cg` marks full-iteration CUDA graphs. The submitted runs use `config_GB300_64x4x240xtp1pp4ep32cp1_mxfp8_full_cg.sh`: 64 GB300 nodes x 4 GPUs, 240 sequences per data-parallel rank, TP1 PP4 EP32 CP1, MXFP8 and full-iteration CUDA graphs.

```
MP = TP * PP * CP
DP = WS // MP = (NNODES * GPUS_PER_NODE) / (TP * PP * CP)
GBS = miniBS * DP
```

Changing `MICRO_BATCH_SIZE` doesn't affect GBS; it controls gradient accumulation (`GA = miniBS // microBS`). DeepSeek V3 671B must run at GBS=15360 or above.

### Seeds

Megatron-Core builds the dataset sample index on one process only and derives the index file name from the `SEED` value, so all processes must use the same `SEED`. `launch_gcp_64xGB300.sh` generates a random seed for every run and passes the same value to all ranks.
