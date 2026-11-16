#!/bin/bash

set -e

mkdir -p /results

# Profiler setup
if [[ $PROFILER == "rpd" ]]; then
    export NVTE_NVTX_ENABLED=1
    bash loadTracer.sh
fi

cd /workspace/Primus

# Start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)

# Build CLI extra args for pretrained checkpoint
POSTTRAIN_CLI_EXTRA=""
if [ -n "${PRETRAINED_CHECKPOINT:-}" ]; then
    POSTTRAIN_CLI_EXTRA="modules.post_trainer.overrides.pretrained_checkpoint=${PRETRAINED_CHECKPOINT}"
fi

# Launch training via primus-cli.
# In quiet mode (default), filter stdout to only pass :::MLLOG lines and
# genuine errors; suppress hook/pip/framework noise that bypasses
# PRIMUS_LOG_LEVEL.  Set MLLOG_VERBOSE_LOGS=1 to restore full output.
if [[ "${MLLOG_VERBOSE_LOGS:-0}" == "1" ]]; then
    ./runner/primus-cli direct train posttrain \
        --config ${EXP} \
        ${POSTTRAIN_CLI_EXTRA}
    ret_code=$?
else
    ./runner/primus-cli direct train posttrain \
        --config ${EXP} \
        ${POSTTRAIN_CLI_EXTRA} 2>/dev/null \
        | grep --line-buffered -E ':::MLLOG |^RESULT,|Training failed'
    ret_code=${PIPESTATUS[0]}
fi

# End timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)

# Report result
result=$(( end - start ))
result_name="LLAMA2_70B_LORA_SFT"
echo "RESULT,$result_name,,$result,AMD,$start_fmt"

if [[ $PROFILER == "rpd" ]]; then
    sqlite3 -header -csv trace.rpd "select * from top;" > trace.csv
    source /workspace/utilities/metrics/profiling/add_trace_csv_header.sh
    add_trace_csv_header "LLAMA2_70B_LORA_SFT"
    python3 /workspace/utilities/metrics/profiling/pie_chart_plot.py trace.csv "LLama2-70B LoRA SFT E2E" "${IMAGE_NAME}" 2.0
    mkdir -p /results/artifacts
    cp *.png *.csv *.rpd /results/artifacts
fi

if [[ $ret_code != 0 ]]; then
    echo "Training failed with exit code: $ret_code"
    exit $ret_code
fi

exit 0
