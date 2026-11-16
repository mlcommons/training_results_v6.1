# flux1 on 16xMI355X-288GB_2xEPYC-9575F (Crusoe Cloud, Kubernetes)

16 node(s) x 8 AMD Instinct MI355X, 8x AMD Pollara 400G RoCE per node. Implementation `pytorch_k8s`:
AMD's MLPerf v6.0 flux1 implementation (`amd_v6.0_MI355X_EPYC_9575F_pytorch/`, unmodified) run as a Kubernetes
Indexed Job (one pod per node, `torchrun`) by `k8s/`.

| | |
|---|---|
| Global batch size | 2048 (RCP row 2048) |
| Micro batch size | 16 (= GBS / 128 GPUs; no gradient accumulation) |
| Learning rate / warmup | 2.5e-4 / 0 steps |
| Parallelism | data parallel |

Steps: `setup.sh` (image), `init_datasets.sh` (data), then `run_and_time.sh` once per seed, on
nodes carrying `SELKEY=SELVAL`. Changes vs AMD's v6.0 code, all applied in `k8s/` at run time:
- **Image:** adds the libionic RoCE userspace provider (AMD Pollara AINIC).
- **Logging:** emits the v6.1 precision-disclosure keys and `cache_clear`. `cache_clear` is emitted only
  after a privileged init container has dropped the page cache on every node.
- **Config:** AMD's `config_MI355X_08x08x32.sh` with the batch size, micro batch, learning rate,
  warmup and node count above (`k8s/train-32node-compliance.template.yaml`).
