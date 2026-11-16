# Defaults shared by the launchers (run_hpg.sh, run_hpg.sub, run_b300.sh). Sourced on the host.
# Input: BENCH, HOST (hpg or b300), KIT (kit root). Override any value in the environment.
# Sets: SUB (submission tree), NVIDIA_IMPL, CTX (NVIDIA code dir), WORK_DIR (path in the image),
#       CONT (PINNED=1: pinned SIF variant), DATADIR, MODEL (LoRA)

SUB=${KIT}/University-of-Florida
case "${BENCH}" in
    llama31_8b|gpt_oss_20b|deepseekv3_671b) NVIDIA_IMPL=nemo; : "${WORK_DIR:=/workspace/llm}" ;;
    llama2_70b_lora) NVIDIA_IMPL=nemo; : "${WORK_DIR:=/workspace/ft-llm}" ;;
    flux1) NVIDIA_IMPL=pytorch; : "${WORK_DIR:=/workspace/flux}" ;;
    *) echo "ERROR: unknown BENCH=${BENCH}" >&2; return 1 ;;
esac
# NVIDIA's code, as in the image: configs and config_mounts.sh are sourced from here on the host
CTX=${SUB}/benchmarks/${BENCH}/implementations/${NVIDIA_IMPL}

_sif=${BENCH}-amd_20261006${PINNED:+_pinned}.sif
if [[ "${HOST}" == b300 ]]; then
    : "${CONT:=/raid/home/rse-ce/minh/v61-build/${BENCH}/${_sif}}"
    case "${BENCH}" in
        # ${DATADIR}/8b/{tokenizer,c4-*}; the same files for both (same Llama 3.1 tokenizer)
        llama31_8b|gpt_oss_20b) : "${DATADIR:=/raid/home/rse-ce/yang/data}" ;;
        llama2_70b_lora)
            # v6.1 dataset (scripts/download_dataset.py); Qian's v6.1 checkpoint
            : "${DATADIR:=${KIT}/data/llama2_70b_lora/gov_report}"
            : "${MODEL:=/raid/home/rse-ce/qian/NVIDIA/benchmarks/llama2_70b_lora/implementations/nemo/data/model}" ;;
    esac
else
    declare -A _team_dir=([flux1]=flux [llama31_8b]=llama3-1-8b [gpt_oss_20b]=gpt-oss-20b
                          [llama2_70b_lora]=llama2-70b [deepseekv3_671b]=deepseekv3)
    _t=/blue/rc-rse/mlperf/training/training
    : "${CONT:=${_t}/${_team_dir[${BENCH}]}/${_sif}}"
    case "${BENCH}" in
        llama31_8b) : "${DATADIR:=${_t}/llama3-1-8b}" ;;
        gpt_oss_20b) : "${DATADIR:=${_t}/gpt-oss-20b}" ;;
        deepseekv3_671b) : "${DATADIR:=${_t}/llama3-1-8b}" ;;   # the same C4 data (DATASET=8b)
        llama2_70b_lora)
            : "${DATADIR:=${_t}/llama2-70b/gov_report_v61}"
            : "${MODEL:=${_t}/llama2-70b/model_v61}" ;;
    esac
fi
export SUB NVIDIA_IMPL CTX WORK_DIR CONT DATADIR MODEL
