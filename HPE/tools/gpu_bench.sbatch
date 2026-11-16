#!/bin/bash
#SBATCH --partition=def
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --comment=docker
#SBATCH --time=0:20:00
#SBATCH --job-name=gpu-bench
#SBATCH --output=/nas/SHARED/MLPERF/probes/gpu-bench-%N-%j.out
# Per-GPU bf16 GEMM throughput, all 8 GPUs loaded at once (so power/thermal
# conditions match training). Finds a single slow GPU that drags the job.
set -uo pipefail
for _ in $(seq 1 60); do [[ $(squeue -h -w "$(hostname -s)" -t CG | wc -l) -eq 0 ]] && break; sleep 10; done
# multiprocessing 'spawn' re-imports __main__, so the script must be a real file.
docker run --rm -i --device /dev/kfd --device /dev/dri --group-add video --ipc=host \
  rocm/amd-mlperf:llama31_8b_training_6.0 bash -c 'cat > /tmp/b.py && python /tmp/b.py' <<'PY'
import torch, torch.multiprocessing as mp, time
def run(i, q):
    torch.cuda.set_device(i); n=8192
    a=torch.randn(n,n,device='cuda',dtype=torch.bfloat16); b=torch.randn(n,n,device='cuda',dtype=torch.bfloat16)
    for _ in range(50): a@b
    torch.cuda.synchronize(); t=time.time(); k=0
    while time.time()-t<60:
        for _ in range(20): a@b
        torch.cuda.synchronize(); k+=20
    q.put((i, 2*n**3*k/(time.time()-t)/1e12))
if __name__=='__main__':
    mp.set_start_method('spawn'); q=mp.Queue()
    ps=[mp.Process(target=run,args=(i,q)) for i in range(8)]
    [p.start() for p in ps]; r=sorted(q.get() for _ in ps); [p.join() for p in ps]
    for i,t in r: print(f"GPU{i} {t:7.1f} TFLOPS")
    v=[t for _,t in r]; print(f"min {min(v):.1f}  max {max(v):.1f}  spread {100*(max(v)/min(v)-1):.1f}%")
PY
