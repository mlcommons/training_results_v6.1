# dlrmv4 — MLPerf Training v6.1

NVIDIA MLPerf Training v6.1 reference implementation (HSTU / generative-recommenders), run via the NVIDIA-shipped NGC container `nvcr.io/nvdlfwea/mlperftv61/dlrmv4-arm:20260929`. The submitter-authored artifact is the Slurm launch wrapper (`launch_dlrmv4_64gpu_20260929_nvme.sh`), which sources NVIDIA's own shipped config file and submits via `sbatch` + Pyxis/Enroot.

## Scale submitted

- 64 GPU (16 nodes x 4 GPU): `config_GB300_16x4x256.sh` (NVIDIA-shipped; global batch size 16384)

NVIDIA ships only two GB300 configs for this benchmark: 8 GPU (2 nodes) and 64 GPU (16 nodes).

## Dataset staging

The dataset (including the benchmark's preprocessed HSTU event cache) is stored on NFS at `SLOW_DATADIR` and is staged to node-local NVMe (`DATADIR`) by NVIDIA's `run.sub` on every node before the experiments start. The submission run reads the dataset from node-local NVMe.

## Launch wrapper

See `launch_dlrmv4_64gpu_20260929_nvme.sh` in this directory.

## Container

No custom build. NVIDIA ships a ready-to-run NGC container for this benchmark; it is pulled and converted to an Enroot `.sqsh`. Verified to support both Vera Rubin and GB300 NVL72 systems (per the NGC catalog listing) — ships `config_GB300_*.sh` and `config_VR200_*.sh` configs side by side.

```bash
docker login nvcr.io -u '$oauthtoken' --password-stdin < ~/.config/ngc/api_key
docker pull nvcr.io/nvdlfwea/mlperftv61/dlrmv4-arm:20260929
enroot import -o /data/mlperf/images/dlrmv4-arm-20260929.sqsh dockerd://nvcr.io/nvdlfwea/mlperftv61/dlrmv4-arm:20260929
```
