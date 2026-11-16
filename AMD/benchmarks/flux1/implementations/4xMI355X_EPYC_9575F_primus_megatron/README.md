# MLPerf Training v6.1 -- flux1 (Flux.1-schnell) -- AMD Instinct MI355X, Primus

This directory trains the MLPerf Training flux1 benchmark with
[Primus](https://github.com/AMD-AGI/Primus) (Megatron-LM backend) on 4 nodes x 8 AMD Instinct
MI355X.

Numerics:
- The forward GEMMs of the transformer linear layers run in **MXFP6**: OCP E2M3 elements with one
  E8M0 scale per 32 elements, and a 32-point Hadamard transform applied to both operands of every
  GEMM along the contraction dimension.
- The joint blocks' text stream (MLP fc1 / fc2 and attention out-projection) runs its forward in
  **MXFP4** (OCP E2M1, E8M0 scale per 32 elements) until step 7325. From step 7325 on (forward healing),
  these forwards run **MXFP6** like the rest; the MLPerf log records it once (`mxfp6_fwd_fp4_heal`:
  iteration 7325, 19 of 19 layers, target mxfp6).
- Both backward GEMMs (dgrad, wgrad) of every linear layer run in **MXFP4**, rounded to nearest.
- Attention, normalisation, the embedders, the AdaLN modulation and the communication run in
  bf16.

The software is built from public repositories at pinned commits (see the `Dockerfile`):

| Component | Repository | Role |
|---|---|---|
| Primus | AMD-AGI/Primus | training framework, Flux model, recipe, MLPerf logging |
| Primus-Turbo | AMD-AGI/Primus-Turbo | MXFP6 packers (Hadamard + E2M3 + scales) and fused prologues |
| aiter | ROCm/aiter | MXFP6 / MXFP4 GEMM kernels and tuned configurations, attention |

## 1. Build the image

```bash
docker build -t <image> .
```

No GPU is needed to build. The aiter JIT modules used by training are compiled into the image
(`precompile_aiter.py`), so nothing is compiled during a timed run. The image records the exact
commits it was built from in `/opt/mlperf/stack.txt`.

## 2. Dataset

Training uses the MLCommons-preprocessed CC12M subset (1,099,776 samples). Evaluation uses the
COCO-2014 subset (29,696 samples, each with its evaluation timestep). Guidance dropout uses the
MLCommons empty encodings.

`scripts/prepare_data.sh` produces all three in one Megatron-Energon dataset root. It runs
Primus's streaming ingest (`primus data diffusion-ingest`), which:
- downloads each MLCommons Arrow file;
- converts it into WebDataset shards;
- deletes it before moving on, with a small prefetch window.

So the peak disk use is the final dataset size plus a buffer of a few files (measured below),
instead of the raw download plus a converted copy.

```bash
mkdir -p ${DATADIR}
docker run --rm -v ${DATADIR}:/data <image> bash /workspace/code/scripts/prepare_data.sh
docker run --rm -v ${DATADIR}:/data <image> bash /workspace/code/scripts/verify_data.sh /data/mlperf_flux1
```

`verify_data.sh` checks:
- the shard counts (4762 train, 129 eval) and sample counts;
- that every evaluation sample carries its timestep;
- the empty encodings and the Energon metadata;
- the sha256 of every shard against `scripts/expected_data_manifest.sha256`.

Every node needs the dataset on durable storage: a shared filesystem, or a copy on each
node's local disk. MLPerf requires the data to start on durable storage (not RAM, so not a
ramdisk), and the page cache is flushed before every run, so the first pass over the data is
read cold. A sample is 2.17 MB: at the step time of this configuration each node reads about
1.5 GB/s. If the shared filesystem is slower, copy the dataset to each node's local NVMe
(about 2.4 TB) and point `DATADIR` there. If the dataset directory inside `DATADIR` has another
name than `mlperf_flux1`, set `DATASET` in `job.sh` to it (and pass it to `verify_data.sh`).

## 3. Run

Edit the variables at the top of `job.sh` (image, dataset directory, result directory,
node-local cache directory), then:

```bash
sbatch job.sh
```

On every node, `run_with_docker_slurm.sh`:
1. pulls the image;
2. checks that the GPUs are idle;
3. starts one container;
4. logs the host settings and warns where they differ from the tuned ones (`runtime_tunables.sh`;
   it changes nothing);
5. runs `NEXP` timed runs.

Before **every** run it flushes the page cache on the node (`vm.drop_caches=3`: host `sudo` if it
needs no password, else through the privileged container) and refuses to start if that fails. Each run is `run_and_time.sh` → Primus
`examples/mlperf/flux1/megatron/run_and_time.sh`, and writes its MLPerf log to
`${LOGDIR}/result_<n>.txt`.

Without Slurm, set `NNODES`, `NODE_RANK` and `MASTER_ADDR` on each node and run
`run_with_docker_slurm.sh` there.

Site-specific settings are in `config_common.sh`:

- `NCCL_IB_HCA`: the AI NIC (ionic) device list of the nodes.
- `NCCL_IB_GID_INDEX`: the RoCE v2 GID index of the AI NICs (1).
- `RCCL_TOPO_FILE`: optional. The path **on the host** of an RCCL topology XML for the node
  type, if the site provides one (set it in `job.sh`). The launcher mounts it read-only into
  the container and sets `NCCL_TOPO_FILE` to it. Without it, RCCL discovers the topology itself. Use the same setting for all scored runs.

The launcher also mounts the host's AI NIC verbs provider (`libionic.so.1`) read-only over the
image's, so that it matches the host's kernel driver (`HOST_LIBIONIC=0` keeps the image's), and
takes the bootstrap interface from the route to the master node (`IP_INTERFACE` overrides it).

## 4. Configuration

| | |
|---|---|
| System config | `config_MI355X_04x08x32_mxfp6heal_hb128.sh`: 4 nodes x 8 GPUs, micro batch 32, global batch 1024 (the `config_filename` of the results) |
| Recipe | `conf/flux1_mlperf_4n_gbs1024.yaml`, identical to Primus `examples/megatron/configs/MI355X/diffusion/flux_12b_ddp_energon_schnell_resample_local_spec_mxfp6_fp4_fast_mlperf_4n_gbs1024.yaml` except for the run settings (`scripts/check_conf_sync.sh`): head bucket 128 MB (`ddp_head_bucket_size`, communication grouping only), forward healing from step 7325 (`mxfp6_fwd_fp4_heal_iter`), `exp_name` |
| Precision | MXFP6 forward, MXFP4 text-stream forward until step 7325 (MXFP6 after), MXFP4 backward (section above); `lowest_numerical_precision_in_linear` = `mxfp4` |
| Optimizer | AdamW, beta1 0.9, beta2 0.95, eps 1e-8, weight decay 0.1, gradient clip 1.0 |
| Schedule | lr 2.5e-4 constant after 800 linear warmup steps, lr = peak x (step + 1) / 800 as in the reference |
| Evaluation | validation loss on the COCO subset every 262,144 training samples; target 0.586 |
| Seeds | per run, `ROOT_SEED` + run index; logged through MLPerf logging |
| Step boundary | no host synchronisation: event-based log timing, bf16 loss scale without a sync, gradient clipping on the device, cached RoPE ids, next inputs prepared ahead |
| RCCL | 2 queue pairs per connection, no data split, 1 MB buffers (`NCCL_BUFFSIZE=1048576`): with the default 4 MB, the back-to-back collectives of the sync-free step boundary drop packets on the fabric |

All transformer GEMMs (MXFP6 and MXFP4) run on aiter's tilescale assembly kernels (`mxfp6_gemm_layout:
tilescale`): the operands are packed in the tilescale layout, and each GEMM runs an exact-shape or shape-generic
code object shipped with aiter. Nothing is compiled at runtime.

## 5. Quality metric

Validation loss (mean over the 29,696 COCO samples, each at its fixed timestep) ≤ 0.586,
evaluated every 262,144 samples. A submission needs 10 runs.
