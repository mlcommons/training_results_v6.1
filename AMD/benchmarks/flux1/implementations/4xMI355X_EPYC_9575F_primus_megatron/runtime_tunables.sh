#!/bin/bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.
#
# Logs the host settings the runs were tuned under and warns where a node differs. Changes nothing: every result of
# this recipe was measured with the hosts as they are, so the scored runs keep them that way.
#   kernel.numa_balancing   0: no automatic page migration. Each rank is bound to its GPU's NUMA node
#                           (PRIMUS_NUMA_BIND, config_common.sh), so its memory is local already; balancing would only
#                           add page-scan faults.
#   transparent hugepages   never: synchronous compaction (defrag=always) stalls the host threads that feed the GPUs.
#   kernel.nmi_watchdog     0: no periodic NMI on every core.
_read() { cat "$1" 2>/dev/null || echo "n/a"; }
_sel() { grep -o '\[[a-z+]*\]' "$1" 2>/dev/null | tr -d '[]' || echo "n/a"; }
numa_balancing=$(_read /proc/sys/kernel/numa_balancing)
thp=$(_sel /sys/kernel/mm/transparent_hugepage/enabled)
thp_defrag=$(_sel /sys/kernel/mm/transparent_hugepage/defrag)
nmi_watchdog=$(_read /proc/sys/kernel/nmi_watchdog)
echo "[$(hostname)] host: numa_balancing=${numa_balancing} thp=${thp} thp_defrag=${thp_defrag}" \
    "nmi_watchdog=${nmi_watchdog} numa_nodes=$(ls -d /sys/devices/system/node/node[0-9]* 2>/dev/null | wc -l)" \
    "governor=$(_read /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor) kernel=$(uname -r)"
[[ "${numa_balancing}" == 0 ]] || echo "[$(hostname)] WARNING: kernel.numa_balancing=${numa_balancing} (tuned with 0)" >&2
[[ "${thp}" == never ]] || echo "[$(hostname)] WARNING: transparent hugepages ${thp}/${thp_defrag} (tuned with never)" >&2
[[ "${nmi_watchdog}" == 0 ]] || echo "[$(hostname)] WARNING: kernel.nmi_watchdog=${nmi_watchdog} (tuned with 0)" >&2
exit 0
