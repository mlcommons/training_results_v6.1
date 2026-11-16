# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# GBS512 batch-shape override: 32 prompts × 16 generations.
# Source this AFTER a hardware profile config to set the GBS-dependent
# knobs (LR, grad-norm, max-steps, val-start-at).
#
# Compliance rules (mlcommons/logging training_6.1.0/closed_qwen35_397b_grpo.yaml):
#   LR = 1e-6 × sqrt(GBS / 256) = 1.4142135624e-6
#   max_grad_norm = 0.125 × sqrt(256 / GBS) = 0.08838834765
#   val_start_at (steps) = ceil(2.5 + 3840 / GBS) = ceil(2.5 + 7.5) = 10
#   num_generations_per_prompt = 16; num_prompts_per_step = GBS / 16 = 32
#   max_steps: no constraint (just must be logged)

export NUM_PROMPTS_PER_STEP=32
export NUM_GENERATIONS_PER_PROMPT=16
export MAX_STEPS=20
export VAL_START_AT=10
export LEARNING_RATE=1.4142135624e-06
export MAX_GRAD_NORM=0.08838834765
