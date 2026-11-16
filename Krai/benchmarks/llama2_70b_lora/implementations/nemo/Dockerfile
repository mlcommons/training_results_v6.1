# Copyright (c) 2026 Krai Ltd. Licensed under the Apache License, Version 2.0 (see LICENSE).
# Krai v6: AMD's MLPerf Training 5.1 Llama 2 70B LoRA image with the changed recipe files, Transformer Engine's two
# one-line changes and the patch modules built in (see README.md).
FROM rocm/amd-mlperf:llama2_70b_training_5.1

# TE 2.2 linear.py / layernorm_linear.py: grad-output cast without the transposed copy when no wgrad is needed
COPY te_patches/linear.py te_patches/layernorm_linear.py /opt/venv/lib/python3.10/site-packages/transformer_engine/pytorch/module/
# patch modules imported by src/train.py (TRAINLOOM_PY_PATCHES); the runs read them from /workspace/trainloom_patches
COPY patches /workspace/trainloom_patches
COPY config_MI300X_1x8x1.sh /workspace/code/config_MI300X_1x8x1.sh
COPY src/train.py /workspace/code/src/train.py
COPY src/callbacks/custom_callbacks.py /workspace/code/src/callbacks/custom_callbacks.py
COPY src/conf/megatron_gpt_peft_tuning_config.yaml /workspace/code/src/conf/megatron_gpt_peft_tuning_config.yaml
