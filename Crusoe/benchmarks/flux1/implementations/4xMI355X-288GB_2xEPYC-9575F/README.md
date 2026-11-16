# flux1 on 4xMI355X-288GB_2xEPYC-9575F (Crusoe Cloud, Kubernetes)

4 node(s) x 8 AMD Instinct MI355X, 8x AMD Pollara 400G RoCE per node. Implementation `primus_v61drop_k8s_253c561`:
AMD's MLPerf Training v6.1 flux1 Primus implementation (`amd_v6.1_MI355X_EPYC_9575F_primus/`, unmodified) run as a
Kubernetes Indexed Job by `k8s/`.

| | |
|---|---|
| Global batch size | 1024 (micro batch 32, gradient accumulation 1) |
| Learning rate / warmup | 0.00025 / 1600 steps |
| Precision (linear / attention / comm) | mxfp4 / bfloat16 / mxfp4 |
| Software | rocm/primus:v26.7 base image, Primus 88c495a, Primus-Turbo ec940e3, AMD mlperf-common 7771aea, RCCL built from ROCm/rocm-systems d755079, libionic 1.117.5-a-77 (AMD Pollara AINIC RoCE), Kubernetes (Crusoe Managed Kubernetes) |

Steps: `setup.sh` (image), `init_datasets.sh` (data), then `run_and_time.sh` once per seed. Changes vs AMD's
v6.1 code, all applied in `k8s/` at run time:
- **Image:** AMD's v6.1 Dockerfile plus the libionic RoCE userspace provider (AMD Pollara AINIC),
  appended as `k8s/Dockerfile.v61drop-libionic`.
- **Launch:** `k8s/launch-cdrop.sh` (Kubernetes Indexed Job, one pod per node, AMD's `run_and_time.sh`) writes
  `config_MI355X_4x8x32.sh`, which sources AMD's `config_common.sh` and sets this scale's
  node count, GBS, MBS, LR and warmup (AMD's 4-node values at GBS 1024; the RCP-table row at GBS 2048).
- **Logging:** `submission_org` set to the submitter; `cache_clear` is logged after a privileged init container
  has dropped the page cache on every node (`k8s/train-cdrop.template.yaml`).
- **Multi-node communication (no effect on the math):** `NCCL_RMA_DISABLE=1` (AMD's SDMA all-gather hook sets
  `NCCL_CUMEM_ENABLE=1`, whose RMA queue pairs fail on the ionic NICs); `FSDP_ALL_GATHER_BACKEND=rccl` (the
  symmetric-memory all-gather does not complete its rendezvous at 4+ nodes); RCCL 1 QP per connection and
  8 channels per network peer (ionic QP budget).
- **Compile cache:** `INDUCTOR_LOCAL=1` keeps the TorchInductor cache on node-local disk instead of the shared
  volume (concurrent writers from many nodes left unreadable kernels). Compilation happens before `run_start`.
