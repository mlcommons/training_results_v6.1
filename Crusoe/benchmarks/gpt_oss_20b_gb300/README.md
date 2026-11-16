# gpt_oss_20b — MLPerf Training v6.1

NVIDIA MLPerf Training v6.1 reference implementation, run via the NVIDIA-shipped NGC container `nvcr.io/nvdlfwea/mlperftv61/gpt_oss_20b-arm`. The submitter-authored artifacts are the Slurm launch wrappers, which source NVIDIA's own shipped config files and submit via `sbatch` + Pyxis/Enroot.

## Scales submitted

- 8 GPU (2 nodes x 4 GPU): `config_GB300_2x4x3xtp1pp1cp1ep1_mxfp8.sh` (NVIDIA-shipped)
- 72 GPU (18 nodes x 4 GPU): `config_GB300_18x4x1xtp1pp1cp2ep4_mxfp8.sh` (NVIDIA-shipped)

## Launch wrappers

- `launch_gpt_oss_8gpu.sh` — 8 GPU scale (10 parallel seeded jobs across 5 node-pairs)
- `launch_gpt_oss_72gpu.sh` — 72 GPU scale (single sequential NEXP=10 job)

## Container

No custom build. NVIDIA ships a ready-to-run NGC container for this benchmark; it is pulled and converted to an Enroot `.sqsh`.

```bash
docker login nvcr.io -u '$oauthtoken' --password-stdin < ~/.config/ngc/api_key
docker pull nvcr.io/nvdlfwea/mlperftv61/gpt_oss_20b-arm
enroot import -o /data/mlperf/images/gpt_oss_20b-arm.sqsh dockerd://nvcr.io/nvdlfwea/mlperftv61/gpt_oss_20b-arm
```

Known-good sha256 of the image actually used for the runs in this package:

```
353807c5894245dc9956a275576212599df3abdfff716f05eaa48cd0caf8dc66
```
