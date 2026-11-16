# 60-step smoke on top of the exact BS32 4-GPU submission config.
# Sources the real config so every training-relevant knob is identical; only the
# step budget and wall time are cut down. DGXSYSTEM is then re-pointed at the real
# config name so the mllog `config_filename` key reports what we actually submit.
source $(dirname ${BASH_SOURCE[0]})/config_NVLink_H100_BF16_RCP_BS32_4GPU.sh

export MAX_STEPS=60
export OPT_LR_DECAY_STEPS=1200000          # keep the LR schedule identical to the real run
export VAL_CHECK_INTERVAL=60               # one eval at the end
export LOAD_MINIMAL_NUM_SAMPLES=0
export WALLTIME_RUNANDTIME=40
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))

# Report the submission config filename, not this smoke wrapper.
export DGXSYSTEM=NVLink_H100_BF16_RCP_BS32_4GPU
