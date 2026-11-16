#!/bin/bash
#SBATCH --partition=def
#SBATCH --nodelist=lodin1
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --exclusive
#SBATCH --comment=docker
#SBATCH --time=1:00:00
#SBATCH --job-name=aiter-check
#SBATCH --output=/nas/SHARED/MLPERF/results/aiter-check-%j.out
#
# 1) Finish the lodin1 flux1 tree: `energon prepare` (answers y/y/11, verified
#    on lodin2 on 2026-09-30).
# 2) Report whether AITER is installed AND wired into each training image.
set -uo pipefail

echo "=== energon prepare on $(hostname) ==="
if [[ -d /lvol0/mlperf/flux1/energon/.nv-meta ]]; then
    echo "already prepared"
else
    docker run --rm -i --volume /lvol0/mlperf/flux1:/dataset rocm/amd-mlperf:flux1_training_6.0 \
        bash -c "cd /dataset/energon && printf 'y\ny\n11\n' | energon prepare --split-parts 'train:train/.*' --split-parts 'val:val/.*' ./ >/dev/null 2>&1; cp -rn /dataset/../flux1/energon/empty_encodings . 2>/dev/null; cat .nv-meta/split.yaml"
fi
ls /lvol0/mlperf/flux1/energon

for img in rocm/amd-mlperf:llama2_70b_training_6.0 rocm/amd-mlperf:llama31_8b_training_6.0; do
    echo
    echo "=== ${img}"
    docker run --rm --device /dev/kfd --device /dev/dri --group-add video "${img}" bash -c '
        cd /workspace/aiter && echo "aiter commit: $(git log -1 --format="%h %cs")"
        echo "prebuilt JIT modules:"; ls aiter/jit/*.so 2>/dev/null | xargs -n1 basename | sed "s/^/  /"
        python - <<PY
import importlib, os
try:
    import aiter; print("import aiter: OK")
except Exception as e:
    print("import aiter: FAIL", e)
import transformer_engine, glob
root = os.path.dirname(transformer_engine.__file__)
hits = sorted({os.path.relpath(f, root) for f in glob.glob(root + "/**/*.py", recursive=True)
               if "aiter" in open(f, errors="ignore").read().lower()})
print("TransformerEngine files that call AITER:", len(hits))
for h in hits[:8]: print("  ", h)
PY'
done
