# gpt_oss_20b on 1xMI355X-288GB_2xEPYC-9575F (Crusoe Cloud, Kubernetes)

1 node(s) x 8 AMD Instinct MI355X, 8x AMD Pollara 400G RoCE per node. Implementation `primus_v61drop_k8s_253c561`:
AMD's MLPerf Training v6.1 gpt_oss_20b Primus implementation (`amd_v6.1_MI355X_EPYC_9575F_primus/`, unmodified) run as a
Kubernetes Indexed Job by `k8s/`.

| | |
|---|---|
| Global batch size | 32 |
| Micro batch size | 4 (gradient accumulation 1) |
| Learning rate / warmup | 0.00045 / 234 steps |
| Parallelism | data parallel 8 (TP 1, CP 1, EP 1) |
| Precision (linear / attention / comm) | mxfp4 / bfloat16 / bfloat16 |
| Software | rocm/primus:v26.6 base image (ROCm/PyTorch/TransformerEngine/AITER as shipped in it), Primus 472e722, Primus-Turbo 3a1cd8f, AMD mlperf-common 7771aea, RCCL built from ROCm/rocm-systems dde68f3, libionic 1.117.5-a-77 (AMD Pollara AINIC RoCE), Kubernetes (Crusoe Managed Kubernetes) |

Steps: `setup.sh` (image), `init_datasets.sh` (data), then `run_and_time.sh` once per seed. Changes vs AMD's
v6.1 code, all applied in `k8s/` at run time:
- **Image:** AMD's v6.1 Dockerfile plus the libionic RoCE userspace provider (AMD Pollara AINIC),
  appended as `k8s/Dockerfile.v61drop-libionic`.
- **Launch:** `k8s/launch-v61drop.sh` sources AMD's single-node config and adds the multi-node rendezvous
  (Kubernetes Indexed Job, one pod per node, AMD's `run_and_time.sh`). Learning rate and warmup follow AMD's
  recipe at GBS 32 and the RCP-table row at every other batch size; decay = 1,200,000 - warmup steps.
- **Logging:** `submission_org` set to the submitter; `cache_clear` is logged after a privileged init container
  has dropped the page cache on every node (`k8s/train-v61drop.template.yaml`).
