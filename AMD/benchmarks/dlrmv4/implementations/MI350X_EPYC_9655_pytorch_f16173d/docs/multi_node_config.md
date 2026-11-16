# Multi-node configuration (2 and 4 nodes, MI350X, RoCEv2)

This file describes how the 2-node (`2xMI350X_EPYC_9655_pytorch`) and 4-node
(`4xMI350X_EPYC_9655_pytorch`) submitted runs were set up, and which settings depend on the
network. Launch steps are in [`../README.md`](../README.md). Training settings are in
[`training_recipe.md`](training_recipe.md).

## 1. Topology and rank layout

- Each node runs one container with 8 GPUs. Slurm starts one task per node. In each node's
  container the trainer starts 8 local ranks with `torch.multiprocessing` (no `torchrun`).
- Global rank = `NODE_RANK x 8 + local rank`. The CUDA/HIP device stays node-local.
  `NNODES` and `NODE_RANK` come from `SLURM_NNODES` and `SLURM_NODEID`.
- Rendezvous uses the `env://` method of `torch.distributed` with the NCCL (RCCL) backend.
  `MASTER_ADDR` is the first node of the allocation and `MASTER_PORT` is
  `20000 + SLURM_JOB_ID % 20000`. Both are resolved on the host, because the container has no
  Slurm client. The process-group timeout is 1800 s.
- Containers run with `--network=host --ipc=host` and `--ulimit memlock=-1:-1` (RDMA memory
  registration needs unlimited locked memory). The generated `run_with_docker_slurm.sh` uses the
  same flags as the submitted runs.

## 2. Network of the submitted runs

- Data plane: RoCEv2 RDMA over 8 Broadcom bnxt_re HCAs per node (one per GPU), with GPUDirect
  RDMA enabled (`NCCL_NET_GDR_LEVEL=5`). The host kernel provides the peer-memory client, so RCCL
  moves data directly between GPU memory and the NIC.
- Bootstrap: TCP over the node's routable front-end interface (`NCCL_SOCKET_IFNAME`). The per-GPU
  RoCE interfaces are point-to-point links that do not route plain TCP between nodes, so RCCL must
  not choose one of them for the bootstrap.

## 3. Settings

Common to both multi-node systems (set in `config_MI350X_02x08x1024.sh` and
`config_MI350X_04x08x1024.sh`):

| Variable | Value | Purpose |
|---|---|---|
| `NCCL_IB_TIMEOUT` | `14` | IB/RoCE transport timeout |
| `NCCL_IGNORE_CPU_AFFINITY` | `1` | RCCL uses the GPU's CPU affinity and ignores the process's CPU affinity |
| `RCCL_MSCCL_ENABLE` | `0` | MSCCL off |
| `NCCL_NET_GDR_LEVEL` | `5` | use GPUDirect RDMA for every GPU/NIC pair, including pairs across the CPU root complex |
| `NCCL_DMABUF_ENABLE` | `1` | DMA-BUF registration where the kernel supports it (peer memory carries GPUDirect otherwise) |
| `HSA_ENABLE_IPC_MODE_LEGACY` | `1`, only when `/sys/module/amdgpu/version` exists | ROCr IPC through the KFD ioctls of the DKMS amdgpu driver. It works around an intermittent RCCL IPC-import crash at multi-node start (ROCm/rocm-systems#8178). Single-node runs leave it unset |

Site-specific (values of the submitted runs; change them for another fabric):

| Variable | Value in the submitted runs | How to find yours |
|---|---|---|
| `NCCL_IB_HCA` | `bnxt_re0,...,bnxt_re7` | `ibv_devices` |
| `NCCL_IB_GID_INDEX` | `3` (RoCEv2, IPv4) | `show_gids` |
| `NCCL_IB_TC` | `104` (lossless traffic class) | fabric configuration |
| `NCCL_SOCKET_IFNAME` | the routable front-end interface | `ip -br addr`. `run_with_docker_slurm.sh` defaults to the interface of the default route |

Other NCCL/RCCL variables were left at their defaults. `NCCL_DEBUG=WARN` applies to all systems.

## 4. RDMA userspace libraries

The container image ships rdma-core 50 (bnxt_re provider ABI `rdmav34`). The hosts of the
submitted runs ran a newer bnxt_re kernel driver with rdma-core 61 (provider ABI `rdmav59`).
With the container's provider, RCCL's queue-pair creation fails with
`ibv_create_qp ... Bad address`: the old provider handles shallow send queues but faults on the
deep ones RCCL creates (256 work requests).

The 2- and 4-node runs therefore loaded the host's rdma-core userspace inside the container
without modifying the image. Before training, the launcher copied the host's
`libibverbs.so.1`, `libnl-3.so.200`, `libnl-route-3.so.200` and the provider libraries
(`libibverbs/*-rdmav59.so`) to a shared folder, and added a `libibverbs.so -> libibverbs.so.1`
symlink. The trainer process then ran with:

```bash
export LD_LIBRARY_PATH="$RDMA_OVERLAY/lib:$RDMA_OVERLAY/lib/libibverbs:$LD_LIBRARY_PATH"
export LD_PRELOAD="$RDMA_OVERLAY/lib/libibverbs.so.1"
```

The unversioned `libibverbs.so` symlink is required: importing torch loads the unversioned name,
and without the symlink the container's library takes the `libibverbs.so.1` slot first.
Single-node runs do not use RDMA and did not load these libraries.

On another system:

- If the container's rdma-core provider matches the host driver (common with Mellanox `mlx5`),
  no host libraries are needed.
- Otherwise, either install a matching rdma-core in the image, or copy the host's libraries as
  above and set `RDMA_OVERLAY` to that folder for `scripts/launch_slurm.sh` (or export the two
  variables above in the container).

## 5. Launchers

- `job.sh` + `run_with_docker_slurm.sh` (generated for each multi-node system) reproduce the
  container flags, environment, trainer command and rank layout of the submitted runs. They leave
  image distribution, dataset staging and the RDMA libraries of section 4 to the site.
- `scripts/launch_slurm.sh` is the launcher the submitted runs used. It also provisions the
  container, stages the dataset to node-local disk and applies the RDMA libraries. Its
  site-specific settings are marked `[CLUSTER-SPECIFIC]`
  (`grep '\[CLUSTER-SPECIFIC\]' scripts/launch_slurm.sh`).
- For debugging only, `NCCL_NET_TRANSPORT=socket` makes `scripts/launch_slurm.sh` disable
  IB/RoCE and run the collectives over TCP. This is much slower and was not used by any
  submitted run.
