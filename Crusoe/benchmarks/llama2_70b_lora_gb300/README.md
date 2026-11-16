# llama2_70b_lora — MLPerf Training v6.1

NVIDIA MLPerf Training v6.1 reference implementation, run via the NVIDIA-shipped NGC container `nvcr.io/nvdlfwea/mlperftv61/llama2_70b_lora-arm:20260828`. The submitter-authored artifacts are the Slurm launch wrappers, which source NVIDIA's own shipped config files and submit via `sbatch` + Pyxis/Enroot.

## Scales submitted

- 8 GPU (2 nodes x 4 GPU): `config_GB300_2x4x1xtp1pp1cp1_fp4.sh` (NVIDIA-shipped)
- 72 GPU (18 nodes x 4 GPU): `config_GB300_18x4x1xtp1pp1cp8.sh` (NVIDIA-shipped)

## Launch wrappers

- `launch_llama2_lora_8gpu.sh` — 8 GPU scale (10 parallel seeded jobs across 5 node-pairs)
- `launch_llama2_lora_72gpu.sh` — 72 GPU scale (single sequential NEXP=10 job)

## Container

No custom build. NVIDIA ships a ready-to-run NGC container for this benchmark; it is pulled and converted to an Enroot `.sqsh`.

```bash
docker login nvcr.io -u '$oauthtoken' --password-stdin < ~/.config/ngc/api_key
docker pull nvcr.io/nvdlfwea/mlperftv61/llama2_70b_lora-arm:20260828
enroot import -o /data/mlperf/images/llama2_70b_lora-arm.sqsh dockerd://nvcr.io/nvdlfwea/mlperftv61/llama2_70b_lora-arm:20260828
```

Known-good sha256 of the image actually used for the runs in this package:

```
ea816f194b002538cfe1f1865966c07848c14252b8a0f3b4e3259c0f379a41fc
```
