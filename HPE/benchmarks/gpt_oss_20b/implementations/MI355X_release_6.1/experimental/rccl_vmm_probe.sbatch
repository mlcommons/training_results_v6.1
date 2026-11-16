#!/bin/bash
#SBATCH --partition=def
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --exclusive
#SBATCH --comment=docker
#SBATCH --time=1:00:00
#SBATCH --job-name=bruno-rccl-vmm
#SBATCH --output=/home/hpebench/slurm-logs/%x-%j.out
#
# Why does RCCL's cuMem/VMM path (NCCL_CUMEM_ENABLE=1, used by the drop's
# rccl_sdma all-gather) fail with HIP 'invalid argument' in hipMemSetAccess on
# lodin? Read-only host facts, then in each 6.1 image: direct HIP VMM probe
# (vmm_probe.py) and an 8-GPU RCCL all-reduce with cuMem on and off.
set -uo pipefail
HERE=${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "$0")")}
for _ in $(seq 1 60); do [[ $(squeue -h -w "$(hostname -s)" -t CG | wc -l) -eq 0 ]] && break; sleep 10; done
docker info >/dev/null 2>&1 || { echo "FATAL: no docker"; exit 1; }

echo "===== host $(hostname) $(date -u)"
uname -r
cat /sys/module/amdgpu/version 2>/dev/null || echo "no /sys/module/amdgpu/version (in-tree kernel driver, not AMD DKMS)"
modinfo amdgpu 2>/dev/null | grep -E "^(filename|version|srcversion|vermagic)" 
dkms status 2>/dev/null | head -5
dpkg -l 2>/dev/null | grep -E "amdgpu-dkms|amdgpu-core|rocm-core|hsa-rocr|linux-modules-extra" | awk '{print $2, $3}' | head
cat /opt/rocm/.info/version 2>/dev/null
for p in /sys/module/amdgpu/parameters/{noretry,vm_fragment_size,hws_max_conc_proc,mtype_local,svm_default_granularity,dcfeaturemask}; do [[ -r $p ]] && echo "$(basename $p)=$(cat $p)"; done
cat /sys/class/kfd/kfd/topology/nodes/2/properties 2>/dev/null | grep -E "capability|fw_version|sdma_fw|gfx_target|num_sdma" 
sudo -n dmesg 2>/dev/null | grep -iE "kfd|amdgpu.*(ioctl|version|xgmi)" | head -8
ls /sys/class/kfd/kfd/topology/nodes/2/io_links | head -3; cat /sys/class/kfd/kfd/topology/nodes/2/io_links/*/properties 2>/dev/null | grep -E "^type|weight" | sort | uniq -c | head

# run IMAGE [docker env flags...] -- COMMAND...
run () {
  local img=$1; shift
  local flags=()
  while [[ $# -gt 0 && $1 != -- ]]; do flags+=("$1"); shift; done
  shift
  docker run --rm --init --net=host --uts=host --ipc=host --privileged \
    --device /dev/dri --device /dev/kfd --group-add video \
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    --ulimit memlock=-1:-1 -v "${HERE}:/probe" -w /probe "${flags[@]}" \
    --entrypoint "$1" "$img" "${@:2}"
}
for img in rocm/amd-mlperf:gpt_oss_20b_training_6.1 rocm/amd-mlperf:llama31_8b_training_6.1 rocm/primus:v26.7; do
  docker image inspect "$img" >/dev/null 2>&1 || { echo "skip $img (not on node)"; continue; }
  echo; echo "===== image $img"
  run "$img" -- bash -c 'python3 -c "import torch;print(\"torch\",torch.__version__,\"hip\",torch.version.hip)"; ls -l /opt/venv/lib/python3.12/site-packages/_rocm_sdk_libraries/lib/librccl.so* 2>/dev/null | head -2'
  echo "--- vmm_probe"
  run "$img" -- python3 vmm_probe.py 2>&1 | tail -45
  for cm in 0 1; do
    echo "--- all-reduce NCCL_CUMEM_ENABLE=$cm"
    run "$img" -e NCCL_CUMEM_ENABLE=$cm -e NCCL_DEBUG=WARN -- timeout 300 \
      torchrun --nproc_per_node=8 allreduce_test.py 2>&1 | grep -E "ALLREDUCE|WARN|rror" | sed 's#/tmp/rocm-systems-rccl-ce/projects/rccl/build/release/hipify/src/##' | sort | uniq -c | head -8
  done
done
echo "===== done $(date -u)"
