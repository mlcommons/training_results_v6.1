#!/bin/bash
#SBATCH --partition=def
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --exclusive
#SBATCH --comment=docker
#SBATCH --time=0:30:00
#SBATCH --job-name=bruno-rccl-vmm
#SBATCH --output=/home/hpebench/slurm-logs/%x-%j.out
# Plain-HIP VMM peer access (GPU0 buffer -> GPU1) with HIP and libhsakmt debug
# logs, to see which layer rejects hipMemSetAccess on lodin.
set -uo pipefail
HERE=${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "$0")")}
for _ in $(seq 1 60); do [[ $(squeue -h -w "$(hostname -s)" -t CG | wc -l) -eq 0 ]] && break; sleep 10; done
for img in rocm/amd-mlperf:gpt_oss_20b_training_6.1 rocm/primus:v26.7; do
  echo "===== $img"
  docker run --rm --init --net=host --ipc=host --privileged --device /dev/dri --device /dev/kfd --group-add video \
    --security-opt seccomp=unconfined -v "${HERE}:/probe" -w /probe \
    -e PEER_ONLY=1 -e AMD_LOG_LEVEL=4 -e HSAKMT_DEBUG_LEVEL=7 -e HIP_VISIBLE_DEVICES=0,1 \
    --entrypoint python3 "$img" vmm_probe.py > /tmp/vmm_$$.log 2>&1
  grep -nE "^\s+hip|^\[|ok$|rc=" /tmp/vmm_$$.log | head -20
  echo "--- debug lines around the failing hipMemSetAccess (peer)"
  n=$(grep -n "hipMemSetAccess" /tmp/vmm_$$.log | sed -n 2p | cut -d: -f1)
  [[ -n "$n" ]] && sed -n "$((n>120 ? n-120 : 1)),$((n+40))p" /tmp/vmm_$$.log | grep -iE "vmem|setaccess|SetAccess|access|ioctl|fail|error|invalid|dmabuf|import|export|map|kfd|peer" | cut -c1-260 | tail -40
  grep -iE "KFD ioctl|ioctl version|kfd version|hsaKmtGetVersion|Kernel interface" /tmp/vmm_$$.log | head -5
done
