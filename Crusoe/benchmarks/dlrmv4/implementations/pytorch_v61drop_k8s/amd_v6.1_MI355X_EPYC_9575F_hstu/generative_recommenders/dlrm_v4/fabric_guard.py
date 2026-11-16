# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

import logging
import os
from typing import Dict, List, MutableMapping

logger: logging.Logger = logging.getLogger(__name__)

IB_ROOT = "/sys/class/infiniband"
ALLOW_ENV = "DLRM_ALLOW_GDR_FLUSH_DISABLE"

MISNAMED = (
    ("NET_OPTIONAL_RECV_COMPLETION", "NCCL_NET_OPTIONAL_RECV_COMPLETION"),
    ("NCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING", "RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING"),
)


def ionic_devices(ib_root: str = IB_ROOT) -> List[str]:
    try:
        return sorted(d for d in os.listdir(ib_root) if d.startswith("ionic"))
    except OSError:
        return []


def uses_anp_plugin(environ: Dict[str, str]) -> bool:
    plugin = os.path.basename(environ.get("NCCL_NET_PLUGIN", "") or "").lower()
    return "anp" in plugin


def check_rccl_env(
    environ: MutableMapping[str, str] = os.environ,  # pyre-ignore [9]
    ib_root: str = IB_ROOT,
) -> List[str]:
    notes: List[str] = []
    ionic = ionic_devices(ib_root)
    if ionic and not uses_anp_plugin(environ) and environ.get("NCCL_GDR_FLUSH_DISABLE") == "1":
        if environ.get(ALLOW_ENV) == "1":
            notes.append(
                f"NCCL_GDR_FLUSH_DISABLE=1 kept ({ALLOW_ENV}=1) on {len(ionic)} ionic NICs with "
                "RCCL's built-in IB transport: inter-node data can arrive stale"
            )
        else:
            environ["NCCL_GDR_FLUSH_DISABLE"] = "0"
            notes.append(
                f"NCCL_GDR_FLUSH_DISABLE=1 -> 0: {len(ionic)} ionic NICs with RCCL's built-in IB "
                "transport (no ANP plugin); without the GDR flush inter-node data can arrive stale. "
                f"{ALLOW_ENV}=1 keeps 1."
            )
    for wrong, right in MISNAMED:
        if wrong in environ and right not in environ:
            notes.append(f"{wrong}={environ[wrong]} has no effect: RCCL reads {right}")
    return notes


def apply(rank: int = 0) -> List[str]:
    notes = check_rccl_env()
    if notes and rank % 8 == 0:
        for n in notes:
            logger.warning("fabric_guard: %s", n)
    return notes
