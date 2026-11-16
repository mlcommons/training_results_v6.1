# dlrmv4 on 2xMI355X-288GB_2xEPYC-9575F (Crusoe Cloud, Kubernetes)

2 node(s) x 8 AMD Instinct MI355X, 8x AMD Pollara 400G RoCE per node. Implementation `pytorch_v61drop_k8s`:
AMD's MLPerf Training v6.1 dlrmv4 PyTorch implementation (generative-recommenders, `amd_v6.1_MI355X_EPYC_9575F_hstu/`,
unmodified) run as a Kubernetes Indexed Job by `k8s/`.

| | |
|---|---|
| Global batch size | 16384 (local 1024 per GPU, gradient accumulation 1) |
| Learning rate (dense / sparse) | 2e-06 / 2e-06 (Adam / RowWiseAdagrad) |
| Warmup | 24000 steps from 0.0 |
| First eval / eval interval | skip 0.0303 of the epoch, then every 0.001 |
| Precision (linear / attention / comm) | mxfp6 / fp8 / fp16 |
| Software | rocm/primus:v26.3 base image, PyTorch 2.12.0+rocm7.2, TorchRec bf55480, FBGEMM-GPU 10b7757 (built from source for gfx950), AMD mlperf-common 7771aea, libionic 1.117.5-a-77 (AMD Pollara AINIC RoCE), Kubernetes (Crusoe Managed Kubernetes) |

Steps: `setup.sh` (image), `init_datasets.sh` (RAID0 scratch + data + index), then `run_and_time.sh` once per seed.
Changes vs AMD's v6.1 code, all applied in `k8s/` at run time:
- **Image:** AMD's v6.1 Dockerfile plus the libionic RoCE userspace provider (AMD Pollara AINIC),
  appended as `k8s/Dockerfile.v61drop-libionic`.
- **Launch:** `k8s/launch-cdrop.sh` (Kubernetes Indexed Job, one pod per node, AMD's `run_and_time.sh`) sets
  AMD's environment for any node count: local batch = GBS / (8 x nodes); dense and sparse learning rate
  1e-6 x GBS / 8192 (AMD's 1e-6 at GBS 8192, linear in GBS as in the RCP table); 24,000 warmup steps from 0;
  `SKIP_EVAL_EPOCH_PCT` 0.025 / 0.0303 / 0.041 at GBS 8192 / 16384 / 32768 (AMD's configs at the same GBS);
  eval every 0.1% of the data, AUC target 0.75.
- **Logging:** AMD's `mlperf_logging_utils.py` defaults `submission_org` to "AMD"; the launcher rewrites that
  default to the submitter (one `sed`, checked) before the run. `MLPERF_VERBOSE_LOGS=1` so the rank-0 stdout
  is the full trainer log. `cache_clear` is logged by AMD's trainer; the page cache is dropped on every node
  by a privileged init container before each run (`k8s/train-cdrop.template.yaml`), and the launcher sets
  `MLPERF_CACHE_CLEARED=1` (which AMD's trainer requires to log `cache_clear: true`) only when that node's
  flush evidence (and, after staging, the post-staging flush evidence) is present.
- **Environment fix:** AMD's image force-installs torchaudio 2.11 over torch 2.12 and its native library is
  missing, so `import torchaudio` (pulled in by torchmetrics) raises at import time. The launcher uninstalls
  torchaudio when it cannot be imported; torchmetrics then treats it as absent. DLRMv4 uses no audio code;
  no training code changes.
- **Dataset:** Yambda-5b HSTU cache (~229 GB, MLCommons source, md5-verified) on the shared RWX volume, with
  AMD's time-window index prebuilt there once. AMD's trainer will not read it from NFS, so the launcher copies
  it to node-local NVMe RAID0 (`/local/dlrmv4`, mounted at `/data`; `k8s/local-raid0-ds.yaml`) the first time
  a node runs DLRMv4, before AMD's `run_and_time.sh` starts (outside the timed region), and reuses it after.
- **Multi-node:** RCCL uses 1 QP per connection / 8 channels per network peer (ionic QP budget).
  No effect on the math.
- **Environment on top of AMD's config:** `MLPERF_VERBOSE_LOGS=1` (`-VAR` unsets VAR).
