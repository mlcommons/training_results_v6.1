# llama31_8b — MLPerf Training v6.1

NVIDIA MLPerf Training v6.1 reference implementation, run via the NVIDIA-shipped NGC container `nvcr.io/nvdlfwea/mlperftv61/llama31_8b-arm`. The submitter-authored artifact is the Slurm launch wrapper (`launch_18node.sh`), which sources NVIDIA's own shipped config file and submits the job via `sbatch` + Pyxis/Enroot.

## Scale submitted

- 72 GPU (18 nodes x 4 GPU): `config_GB300_18x4x1xtp1pp1cp1_8b_fp4.sh` (NVIDIA-shipped)

## Launch wrapper

See `launch_18node.sh` in this directory.

## Container

No custom build. NVIDIA ships a ready-to-run NGC container for this benchmark; it is pulled and converted to an Enroot `.sqsh`.

```bash
docker login nvcr.io -u '$oauthtoken' --password-stdin < ~/.config/ngc/api_key
docker pull nvcr.io/nvdlfwea/mlperftv61/llama31_8b-arm
enroot import -o /data/mlperf/images/llama31_8b-arm.sqsh dockerd://nvcr.io/nvdlfwea/mlperftv61/llama31_8b-arm
```

Known-good sha256 of the image actually used for the runs in this package:

```
c446f24def3d81c0d319536aea294d230f76e0a2037996f0e0da7b754464adb4
```
