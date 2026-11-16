# llama2_70b_lora on 2xMI355X-288GB_2xEPYC-9575F (Crusoe Cloud, Kubernetes)

2 node(s) x 8 AMD Instinct MI355X, 8x AMD Pollara 400G RoCE per node. Implementation `primus_k8s`:
AMD's MLPerf v6.0 Primus implementation (`amd_v6.0_MI355X_EPYC_9575F_primus/`, unmodified) run as a Kubernetes
Indexed Job (one pod per node, `torchrun`, static rendezvous) by `k8s/`.

| | |
|---|---|
| Global batch size | 8 |
| Micro batch size | 1 |
| Parallelism | context parallel 2 x data parallel 8 |
| Learning rate | 0.0006 |
| Manual Python GC | on (`MANUAL_GC=1`) |

Steps: `setup.sh` (images), `init_datasets.sh` (data + Megatron checkpoint), then `run_and_time.sh` once per seed.
Changes vs AMD's v6.0 code, all applied in `k8s/`:
- **Image:** adds the libionic RoCE userspace provider, and rebuilds AITER's `module_gemm_a4w4_asm`
  with the macro that AITER's build defines (the shipped binary rejects float4 inputs).
- **Logging:** emits the v6.1 precision-disclosure keys and `cache_clear` (only after a privileged init container
  has dropped the page cache on every node).
- **LoRA with CP > 1:** sets `calculate_per_token_loss`, which Megatron-Bridge requires for context-parallel fine-tuning.
- **Run-time pip pin:** AMD's post-train hook pip-installs Megatron-Bridge requirements on every run; `datasets` is
  pinned to 5.0.1 (the version installed for the v6.0-implementation runs) because newer releases break
  `import datasets`. No effect on training code or math.
- **Manual GC:** Megatron's `manual_gc` / `manual_gc_eval` (interval 0). No effect on the math.
