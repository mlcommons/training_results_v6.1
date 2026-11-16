"""Pin each rank to the NUMA node of the GPU it drives.

The launchers do no CPU or memory affinity at all. On this class of host the
GPU/NUMA map is not the identity -- on a dual-socket EPYC in NPS4 the PCIe roots
interleave, e.g. BDF-ordered GPUs map to NUMA nodes [1,2,3,0,5,6,7,4]. A naive
rank==node binding would therefore be wrong for most ranks, so the mapping is
detected from sysfs at runtime.

Pinning the process also makes the kernel's first-touch policy place the
dataloader and pinned host buffers on the local node.

Enable with MLPERF_NUMA_BIND=1. Import before torch/framework initialisation.
"""
import os
import sys


def _parse_cpulist(text):
    cpus = set()
    for part in text.strip().split(","):
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            cpus.update(range(int(lo), int(hi) + 1))
        else:
            cpus.add(int(part))
    return cpus


def _gpu_numa_nodes():
    """Return NUMA node per GPU, ordered the way ROCm enumerates devices (BDF)."""
    entries = []
    drm = "/sys/class/drm"
    if not os.path.isdir(drm):
        return []
    for name in os.listdir(drm):
        dev = os.path.join(drm, name, "device")
        vendor = os.path.join(dev, "vendor")
        if not os.path.isfile(vendor):
            continue
        try:
            with open(vendor) as fh:
                if fh.read().strip() != "0x1002":
                    continue
            bdf = os.path.basename(os.path.realpath(dev))
            # a real GPU function, not an audio/other function on the same device
            if not os.path.isfile(os.path.join(dev, "numa_node")):
                continue
            with open(os.path.join(dev, "numa_node")) as fh:
                node = int(fh.read().strip())
        except (OSError, ValueError):
            continue
        entries.append((bdf, node))
    # de-duplicate by BDF, then sort -- ROCm enumerates in ascending BDF order
    uniq = sorted({bdf: node for bdf, node in entries}.items())
    return [node for _, node in uniq]


def apply():
    if os.environ.get("MLPERF_NUMA_BIND", "0") != "1":
        return
    # Only bind actual training ranks. The torchrun/primus launcher process has no
    # LOCAL_RANK and must stay unpinned, otherwise it would be confined to one node.
    if "LOCAL_RANK" not in os.environ:
        return
    local_rank = int(os.environ["LOCAL_RANK"])
    try:
        nodes = _gpu_numa_nodes()
        if local_rank >= len(nodes):
            print(
                f"[numa] rank {local_rank}: only {len(nodes)} GPUs detected, skipping",
                file=sys.stderr,
            )
            return
        node = nodes[local_rank]
        if node < 0:
            print(f"[numa] rank {local_rank}: GPU has no NUMA node, skipping", file=sys.stderr)
            return
        cpulist = f"/sys/devices/system/node/node{node}/cpulist"
        with open(cpulist) as fh:
            cpus = _parse_cpulist(fh.read())
        if not cpus:
            return
        os.sched_setaffinity(0, cpus)
        # Keep OpenMP inside the same node unless the caller already chose.
        os.environ.setdefault("OMP_NUM_THREADS", str(max(1, len(cpus) // 4)))
        print(
            f"[numa] rank {local_rank} -> GPU{local_rank} numa={node} "
            f"cpus={min(cpus)}-{max(cpus)} ({len(cpus)})",
            file=sys.stderr,
        )
    except Exception as exc:  # never let affinity tuning break a run
        print(f"[numa] rank {local_rank}: binding failed: {exc}", file=sys.stderr)


apply()
