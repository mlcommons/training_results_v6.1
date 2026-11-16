#!/bin/bash
#SBATCH --partition=def
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --exclusive
#SBATCH --comment=docker
#SBATCH --time=8:00:00
#SBATCH --job-name=bruno-gptoss
#SBATCH --output=/nas/SHARED/MLPERF/results/gpt_oss_20b/run-%j.out
#
# GPT-OSS 20B, 1 node x 8 MI355X, from the MLCommons reference
# (small_llm_moe_pretraining/primus, AMD-authored). No prebuilt training image
# exists, so build it on first use from rocm/primus:v25.11.
#
# Data = the same C4 files and Llama 3.1 tokenizer staged for llama31_8b.
# Config = GBS 16, lr 4e-4, warmup 128 iters (2,048 samples) = the BS 16 RCP.
#
# Usage: sbatch run_gptoss.sbatch [NEXP]     (default 1)
set -uo pipefail

IMPL=${IMPL:-${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "$0")")}}   # submit from this directory
export NEXP=${1:-${NEXP:-1}}
export CONT=${CONT:-hpe/mlperf:gpt_oss_20b_6.0}
export DATADIR=${DATADIR:-/lvol0/mlperf/llama31_8b/data}
export MODELDIR=${MODELDIR:-/lvol0/mlperf/llama31_8b/model}
export LOGDIR=${LOGDIR:-/nas/SHARED/MLPERF/results/gpt_oss_20b/${TAG:-first}}

echo "=== gpt_oss_20b $(date -u) on $(hostname), NEXP=${NEXP}"
for _ in $(seq 1 60); do
    [[ $(squeue -h -w "$(hostname -s)" -t CG | wc -l) -eq 0 ]] && break; sleep 10; done
docker info >/dev/null 2>&1 || { echo "FATAL: docker daemon not available"; exit 1; }
for f in c4-train.en_6_text_document.bin c4-validation-91205-samples.en_text_document.bin; do
    [[ -e ${DATADIR}/$f ]] || { echo "FATAL: missing ${DATADIR}/$f"; exit 1; }; done
[[ -e ${MODELDIR}/tokenizer.json ]] || { echo "FATAL: missing tokenizer in ${MODELDIR}"; exit 1; }

cd "${IMPL}"
# Always build: Docker's layer cache makes this seconds when nothing changed,
# and it guarantees the image matches this Dockerfile. Skipping the build when
# an image existed once reused a broken image after the Dockerfile was fixed.
echo "--- building ${CONT} ($(date -u))"
docker build -t "${CONT}" . || { echo "FATAL: image build failed"; exit 1; }
echo "--- build done ($(date -u))"

mkdir -p "${LOGDIR}"
export PYTHONPATH=""
source config_MI355X_1x8x1.sh
export NEXP                     # the config may reset it
bash run_with_docker.sh
echo "=== done $(date -u)"
