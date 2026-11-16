#!/bin/bash
# MLPerf Training v6.1 / DeepSeek-V3 671B / tpu7x-8x32x32 / MaxText on TPU7x
# One-time dataset and initial-checkpoint preparation (README.md sections 3.2 and 3.3).
#
# Converts the MLCommons C4 dataset and HuggingFace initial checkpoint into
# MaxText TFDS and Orbax formats.
#
# Replace the <placeholders>. Runs on a CPU machine (no TPU); step 2 needs ~2.2 TB RAM.
set -euo pipefail
cd "$(dirname "$0")"

export DATADIR="${DATADIR:?set DATADIR (<local scratch dir, ~310 GB>)}"
export HF_CKPT="${HF_CKPT:?set HF_CKPT (<local dir for the 1.4 TB HuggingFace checkpoint>)}"
export DATASET_PATH="${DATASET_PATH:?set DATASET_PATH (gs://<bucket>/<dataset>/)}"
export CKPT_BUCKET="${CKPT_BUCKET:?set CKPT_BUCKET (gs://<bucket>/<ckpt>)}"
R2="https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh"

# ---------------------------------------------------------------------------
# 1. Dataset: MLCommons llama-3-1-8b-preprocessed-c4-dataset (Megatron .bin/.idx, ~84 GB)
#    -> TFDS train dataset (2048 -> 16384 shards; dataset_name in config.yml)
#    -> TFDS eval dataset (first 1024 packed 4096-sequences, 2048 shards;
#       eval_dataset_name in config.yml).
# ---------------------------------------------------------------------------
bash <(curl -s "$R2") -d "${DATADIR}/mlcommons" \
  https://training.mlcommons-storage.org/metadata/llama-3-1-8b-preprocessed-c4-dataset.uri
pip install -r data_scripts/requirements.txt
python3 data_scripts/build_dataset.py --src_dir="${DATADIR}/mlcommons" --dst_dir="${DATADIR}/tfds_2048" --stages=train,eval --num_workers=32
python3 data_scripts/split_shards.py --src_dir="${DATADIR}/tfds_2048" --dst_dir="${DATADIR}/tfds" --num_workers=32
gcloud storage cp -r "${DATADIR}/tfds/c4" "${DATASET_PATH}"

# ---------------------------------------------------------------------------
# 2. Initial checkpoint: MLCommons HuggingFace bf16 checkpoint (members only, ~1.4 TB;
#    the original DeepSeek-V3 warmed up 50 iterations with the Llama 3.1 8B tokenizer by
#    MLCommons, README 3.3.4) -> MaxText/Orbax "base" -> resharded to the training mesh.
#    Both commands run inside the submission container (section 3.1), which contains the
#    conversion tools under maxtext/checkpoint_conversion/; `pip install torch` (CPU) is
#    needed by the HuggingFace reader.
# ---------------------------------------------------------------------------
bash <(curl -s "$R2") -d "${HF_CKPT}" \
  https://training.mlcommons-storage.org/metadata/deepseekv3_checkpoint_bf16.uri

# 2a. HuggingFace -> MaxText/Orbax (unsharded base, ~1.05 TB; several hours, disk-bound).
#     --weight_dtype keep stores the bf16 weights as float16 and keeps the float32 router bias
#     (e_score_correction_bias) as float32, as in the checkpoints the submitted runs loaded.
pip install -q torch --index-url https://download.pytorch.org/whl/cpu
python3 -m maxtext.checkpoint_conversion.standalone_scripts.convert_deepseek_family_ckpt \
  --model_size deepseek3-671b --enable_mtp true --weight_dtype keep \
  --base_model_path "${HF_CKPT}" \
  --maxtext_model_path "${CKPT_BUCKET}/base" \
  --simulated_cpu_devices_count 16

# 2b. Reshard base -> the layout the tpu7x-8x32x32 runs load. Mesh and logical axis rules are those of
#     the submitted runs (mesh [dcn,x,y,z,core] = [1,8,32,32,2], see config.yml / the MLPerf logs);
#     dcn_parallelism is all 1 because DCN only replicates weights. --simulated_cpu_devices_count is
#     the product of ici_parallelism (8*32*32*2 = 16384). Values unchanged; ~1.5 h, peak ~2.2 TB host RAM.
#     16384 simulated CPU devices create ~49K threads / ~102K VMAs; raise vm.max_map_count on the host VM.
sudo sysctl -w vm.max_map_count=262144 2>/dev/null || sysctl -w vm.max_map_count=262144 2>/dev/null || true
python3 -m maxtext.checkpoint_conversion.reshard_checkpoint \
  model_name=deepseek3-671b mtp_num_layers=1 scan_layers=True weight_dtype=bfloat16 \
  use_lineage=False use_2d_fsdp_sharding=False capacity_factor=3.0 use_batch_split_schedule=False batch_split_factor=1 \
  mesh_axes='[dcn,x,y,z,core]' ici_parallelism='[1,8,32,32,2]' dcn_parallelism='[1,1,1,1,1]' \
  data_sharding='[[dcn,x,y,z],[core]]' \
  logical_axis_rules='[[embed,[z]],[mlp,[z]],[embed_vocab,[z,core]],[embed_attn,[z]],[q_lora,[z]],[kv_lora,[z]],[q_heads,[core]],[kv_heads,[core]],[heads,[core]],[exp,[x,core]],[embed_moe,[y]],[mlp_moe,[z]],[activation_batch,[dcn,x,y,z]],[activation_length,[core]]]' \
  override_logical_axis_rules=True allow_split_physical_axes=True shard_mode=explicit \
  remove_size_one_mesh_axis_from_type=False \
  load_parameters_path="${CKPT_BUCKET}/base/0/items" \
  base_output_directory="${CKPT_BUCKET}/resharded" \
  checkpoint_storage_concurrent_gb=1024 checkpoint_storage_use_ocdbt=True checkpoint_storage_use_zarr3=True \
  skip_jax_distributed_system=True --simulated_cpu_devices_count=16384

echo "DATASET_PATH=${DATASET_PATH}"
echo "CHECKPOINT_PATH=${CKPT_BUCKET}/resharded/0/items"
