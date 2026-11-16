#!/bin/bash

set -e

mkdir -p /results

# Re-copy the recipe modules and preset so a mounted checkout overrides the image.
RECIPE_DIRS=(
    /workspace/Primus/primus/recipes
    /workspace/Primus/primus/backends/megatron_bridge/recipes/mlperf_llama2_70b
)
for _recipe_dir in "${RECIPE_DIRS[@]}"; do
    mkdir -p "${_recipe_dir}"
    for _mod in llama2_custom _log_suppression; do
        cp /workspace/code/src/${_mod}.py "${_recipe_dir}/${_mod}.py"
    done
done
mkdir -p /workspace/Primus/primus/configs/models/megatron_bridge
cp /workspace/code/src/llama2_70b_lora.yaml \
   /workspace/Primus/primus/configs/models/megatron_bridge/llama2_70b_lora.yaml

cd /workspace/Primus

export PYTHONUNBUFFERED=1

start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

POSTTRAIN_CLI_EXTRA=""
if [ -n "${PRETRAINED_CHECKPOINT:-}" ]; then
    POSTTRAIN_CLI_EXTRA="modules.post_trainer.overrides.pretrained_checkpoint=${PRETRAINED_CHECKPOINT}"
fi

set +e
./runner/primus-cli direct train posttrain \
    --config "${EXP}" \
    ${POSTTRAIN_CLI_EXTRA} 2>&1
ret_code=$?
set -e

end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

result=$(( end - start ))
result_name="LLAMA2_70B_LORA_SFT"
echo "RESULT,$result_name,,$result,AMD,$start_fmt"

if [[ $ret_code != 0 ]]; then
    echo "Training failed with exit code: $ret_code"
    exit $ret_code
fi

exit 0
