# HPE: pin a training process to the NUMA node of its GPU (opt-in, HPE_NUMA_BIND=1).
# Same sysfs logic as llama31_8b's _numa_affinity.py. On lodin the BDF-ordered
# GPUs map to NUMA nodes [1,2,3,0,5,6,7,4], so rank==node would be wrong.
# Called at the top of each spawned rank, before CUDA init and before the data
# loader forks its workers, so workers inherit the binding and first-touch
# allocations land on the local node.
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
    """NUMA node per AMD GPU, in ROCm enumeration (ascending BDF) order."""
    drm = "/sys/class/drm"
    found = {}
    if not os.path.isdir(drm):
        return []
    for name in os.listdir(drm):
        dev = os.path.join(drm, name, "device")
        try:
            with open(os.path.join(dev, "vendor")) as fh:
                if fh.read().strip() != "0x1002":
                    continue
            with open(os.path.join(dev, "numa_node")) as fh:
                found[os.path.basename(os.path.realpath(dev))] = int(fh.read().strip())
        except (OSError, ValueError):
            continue
    return [found[bdf] for bdf in sorted(found)]


def bind(local_rank):
    if os.environ.get("HPE_NUMA_BIND", "0") != "1":
        return
    try:
        visible = os.environ.get("HIP_VISIBLE_DEVICES") or os.environ.get("CUDA_VISIBLE_DEVICES")
        gpu = int(visible.split(",")[local_rank]) if visible else local_rank
        nodes = _gpu_numa_nodes()
        if gpu >= len(nodes) or nodes[gpu] < 0:
            print(f"[numa] rank {local_rank}: no NUMA node for GPU{gpu}, skipping", file=sys.stderr)
            return
        node = nodes[gpu]
        with open(f"/sys/devices/system/node/node{node}/cpulist") as fh:
            cpus = _parse_cpulist(fh.read())
        os.sched_setaffinity(0, cpus)
        print(f"[numa] rank {local_rank} -> GPU{gpu} numa={node} cpus={min(cpus)}-{max(cpus)} ({len(cpus)})",
              file=sys.stderr, flush=True)
    except Exception as exc:  # never let affinity tuning break a run
        print(f"[numa] rank {local_rank}: binding failed: {exc}", file=sys.stderr)
