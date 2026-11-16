# llama2_70b_lora on 1xMI355X-288GB_2xEPYC-9575F (Crusoe Cloud, Kubernetes)

1 node(s) x 8 AMD Instinct MI355X, 8x AMD Pollara 400G RoCE per node. Implementation `primus_k8s`:
AMD's MLPerf v6.0 Primus implementation (`amd_v6.0_MI355X_EPYC_9575F_primus/`, unmodified) run as a Kubernetes
Indexed Job (one pod per node, `torchrun`, static rendezvous) by `k8s/`.

| | |
|---|---|
| Global batch size | 8 |
| Micro batch size | 1 |
| Parallelism | data parallel |
| Hyperparameters | LR / steps per RCP row (k8s/launch-llama.sh) |
| Manual Python GC | off (AMD default) |

Steps: `setup.sh` (images), `init_datasets.sh` (data + Megatron checkpoint), then
`run_and_time.sh` once per seed. Changes vs AMD's v6.0 code, all applied in `k8s/`:
- **Image:** adds the libionic RoCE userspace provider, and rebuilds AITER's `module_gemm_a4w4_asm`
  with the macro that AITER's build defines (the shipped binary rejects float4 inputs).
- **Logging:** emits the v6.1 precision-disclosure keys and `cache_clear`. `cache_clear` is emitted only
  after a privileged init container has dropped the page cache on every node.
- **LoRA with CP > 1:** sets `calculate_per_token_loss`, which Megatron-Bridge requires for
  context-parallel fine-tuning.
