#!/bin/bash
#SBATCH --partition=def
#SBATCH --nodelist=lodin2
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --comment=docker
#SBATCH --time=0:20:00
#SBATCH --job-name=flux-idx
#SBATCH --output=/nas/SHARED/MLPERF/results/flux1/idx-fix-%j.out
# energon prepare runs as root and writes 0600 *.tar.idx files, which the
# hpebench rsync in backup_to_nas.sbatch cannot read. Copy them as root.
set -uo pipefail
for _ in $(seq 1 60); do [[ $(squeue -h -w "$(hostname -s)" -t CG | wc -l) -eq 0 ]] && break; sleep 10; done
docker run --rm -v /lvol0/mlperf/flux1/energon:/src:ro -v /nas/SHARED/MLPERF/flux1_energon:/dst \
  rocm/amd-mlperf:flux1_training_6.0 bash -c '
    cd /src && for f in $(find . -name "*.tar.idx"); do cp --preserve=timestamps "$f" "/dst/$f"; chmod 644 "/dst/$f"; done
    echo "idx src=$(find /src -name "*.tar.idx" | wc -l) dst=$(find /dst -name "*.tar.idx" | wc -l)"
    echo "files src=$(find /src -type f | wc -l) dst=$(find /dst -type f | wc -l)"'
