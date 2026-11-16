# flux1 — MLPerf Training v6.1

NVIDIA MLPerf Training v6.1 reference implementation, run via the NVIDIA-shipped NGC container `nvcr.io/nvdlfwea/mlperftv61/flux1-arm`. The submitter-authored artifacts are the Slurm launch wrappers, which source NVIDIA's own shipped config files and submit via `sbatch` + Pyxis/Enroot.

## Scales submitted

- 8 GPU (2 nodes x 4 GPU): `config_GB300_02x04x64.sh` (NVIDIA-shipped)
- 16 GPU (4 nodes x 4 GPU): `config_GB300_04x04x32.sh` (NVIDIA-shipped)
- 72 GPU (18 nodes x 4 GPU): `config_GB300_18x04x16.sh` (NVIDIA-shipped)

## Launch wrappers

- `launch_flux1_8gpu.sh` — 8 GPU scale (single 2-node job, NEXP seeded runs)
- `launch_flux1_16gpu.sh` — 16 GPU scale (10 parallel seeded jobs across 4-node quads)
- `launch_flux1_72gpu_20260929.sh` — 72 GPU scale (single sequential NEXP=10 job)

## Container

No custom build. NVIDIA ships a ready-to-run NGC container for this benchmark; it is pulled and converted to an Enroot `.sqsh`.

The 8 GPU and 72 GPU runs use `nvcr.io/nvdlfwea/mlperftv61/flux1-arm:20260929`. The 16 GPU runs use the earlier, untagged `nvcr.io/nvdlfwea/mlperftv61/flux1-arm` image.

```bash
docker login nvcr.io -u '$oauthtoken' --password-stdin < ~/.config/ngc/api_key
docker pull nvcr.io/nvdlfwea/mlperftv61/flux1-arm:20260929
enroot import -o /data/mlperf/images/flux1-arm-20260929.sqsh dockerd://nvcr.io/nvdlfwea/mlperftv61/flux1-arm:20260929
```

Known-good sha256 of the images actually used for the runs in this package:

- `:20260929` (8 GPU and 72 GPU runs):

```
beca5e6a64156f78ec7cc7a587d8073f716417c4a188c0a9e517cc5f67959e1b
```

- earlier untagged image (16 GPU runs):

```
f2d8f2afeef173762c5fd9ac0cb575f2d16a52d75c07efc7d759379df07fb779
```
