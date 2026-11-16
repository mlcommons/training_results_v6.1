#!/bin/bash
# MLPerf Training v6.1 / Qwen3.5-397B-A17B GRPO (qwen35_397b_grpo) / tpu7x-4x4x8-128x2x2x1
# Dataset and initial base checkpoint preparation. Needs Python with datasets,
# MaxText and JAX (as in the container image), uv/uvx, git and gcloud.
#
# Uses the download scripts of the reference implementation:
#   git clone https://github.com/mlcommons/training
#   git -C training submodule update --init --recursive llm_post_training/RL
#   export MLCOMMONS_REF_DIR=$PWD/training/llm_post_training
#   - download_dataset.sh (R2E-Gym/R2E-Gym-Subset @ e8b9fcbce43eaca0dc2c0d4798ee6f3e965f590a)
#   - verify_dataset.sh
#   - download_checkpoint.sh (Qwen/Qwen3.5-397B-A17B @ 8472618112abcbd45acbcdc58436aff4233c23f7)
#
# Usage (each step runs only when its variable is 1):
#   DATASET_GCS_DIR=gs://<bucket>/benchmark-r2e-gym-easy \
#   MAXTEXT_SCANNED_CKPT=gs://<bucket>/qwen35_397b/scanned_reshard_fsdp32_tp2/0/items \
#   RUN_DATASET_DOWNLOAD=1 RUN_CKPT_CONVERSION=1 bash init_datasets.sh
set -euo pipefail

export DATASET_GCS_DIR="${DATASET_GCS_DIR:?set DATASET_GCS_DIR to the gs:// directory for the prepared dataset (run_and_time.sh DATASET_PATH)}"
export HF_MODEL_ID="${HF_MODEL_ID:-Qwen/Qwen3.5-397B-A17B}"
export HF_MODEL_REVISION="${HF_MODEL_REVISION:-8472618112abcbd45acbcdc58436aff4233c23f7}"
export MAXTEXT_SCANNED_CKPT="${MAXTEXT_SCANNED_CKPT:?set MAXTEXT_SCANNED_CKPT to gs://<path>/scanned_reshard_fsdp32_tp2/0/items for the converted checkpoint (run_and_time.sh MAXTEXT_CKPT)}"

export DATASET_DIR="${DATASET_DIR:-${HOME}/datasets/R2E-Gym__R2E-Gym-Subset}"
export OUTPUT_DIR="${OUTPUT_DIR:-${HOME}/datasets/R2E-Gym-Easy-Processed}"
export CACHE_DIR="${CACHE_DIR:-${HOME}/datasets/r2e-repository-cache}"
export OUTPUT_HF_DIR="${OUTPUT_HF_DIR:-${HOME}/datasets/R2E-Gym-Easy-Processed-HF}"
export MLCOMMONS_REF_DIR="${MLCOMMONS_REF_DIR:-${HOME}/training/llm_post_training}"

if [[ "${RUN_DATASET_DOWNLOAD:-0}" != "1" && "${RUN_CKPT_CONVERSION:-0}" != "1" ]]; then
  echo "WARNING: set RUN_DATASET_DOWNLOAD=1 and/or RUN_CKPT_CONVERSION=1 to prepare the dataset and the checkpoint; only printing their paths." >&2
fi

# -----------------------------------------------------------------------------
# 1. Training & Validation Dataset (R2E-Gym-Subset easy split)
#    Source: R2E-Gym/R2E-Gym-Subset @ e8b9fcbce43eaca0dc2c0d4798ee6f3e965f590a
#    Qualified JSONL outputs (verified via verify_dataset.sh):
#      - benchmark_r2e_gym_easy_train.jsonl: 700 curriculum-ordered train rows
#        (446,364,412 bytes, SHA-256 e772af09599270ab16a04ccbbcef395bcb2929607f050478033a84b087a606fc)
#      - benchmark_r2e_gym_easy_val.jsonl: 251 validation rows
#        (172,078,629 bytes, SHA-256 2d5bddf717afb4389510a7d197f26ce5eb7f610aab4b20d6af6e0150de6f2c79)
#    Target: ${DATASET_GCS_DIR} (HuggingFace DatasetDict save_to_disk Arrow directory)
#      - dataset_dict.json: {"splits": ["train", "validation"]}
#      - train/data-00000-of-00001.arrow (700 curriculum-ordered prompts; with
#        SHUFFLE=false, MAX_STEPS=10 steps x 72 prompts/step x 16 rollouts/prompt
#        = 11,520 training trajectories -> MLLOG train_samples=11520)
#      - validation/data-00000-of-00001.arrow (251 validation problems -> MLLOG eval_samples=251)
#      - sandbox_images.txt: the 700 task images of the train split (docker_image,
#        sorted, one per line).
# -----------------------------------------------------------------------------
if [[ "${RUN_DATASET_DOWNLOAD:-0}" == "1" ]]; then
  pushd "${MLCOMMONS_REF_DIR}"
  ./download_dataset.sh \
    --dataset-dir "${DATASET_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --cache-dir "${CACHE_DIR}"
  # download_dataset.sh writes the training file with its curriculum suffix;
  # verify_dataset.sh expects benchmark_r2e_gym_easy_train.jsonl.
  if [[ ! -f "${OUTPUT_DIR}/benchmark_r2e_gym_easy_train.jsonl" ]]; then
    cp "${OUTPUT_DIR}"/benchmark_r2e_gym_easy_train.*.jsonl \
      "${OUTPUT_DIR}/benchmark_r2e_gym_easy_train.jsonl"
  fi
  ./verify_dataset.sh "${OUTPUT_DIR}"
  popd

  # Convert ~/datasets/R2E-Gym-Easy-Processed to HuggingFace DatasetDict structure
  # matching the ~/datasets/R2E-Gym-Subset features:
  # The order and content of the data remain the same.
  python3 - <<'PY'
import os
import sys
import json
import shutil

# Avoid local directory name collision if running from home
sys.path = [p for p in sys.path if p != '']
from datasets import load_dataset, Dataset, DatasetDict, load_from_disk

subset_path = os.path.expanduser(
    os.environ.get('DATASET_DIR', '~/datasets/R2E-Gym__R2E-Gym-Subset')
)
if not os.path.exists(subset_path):
  subset_path = os.path.expanduser('~/datasets/R2E-Gym-Subset')
easy_processed_path = os.path.expanduser(
    os.environ.get('OUTPUT_DIR', '~/datasets/R2E-Gym-Easy-Processed')
)
output_path = os.path.expanduser(
    os.environ.get('OUTPUT_HF_DIR', '~/datasets/R2E-Gym-Easy-Processed-HF')
)

# 1. Load Reference Dataset (Subset) to get the target schema
print("Loading reference dataset (R2E-Gym-Subset)...")
try:
    data_dir = os.path.join(subset_path, 'data')
    parquet_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.parquet')]
    subset_ds = load_dataset('parquet', data_files=parquet_files)['train']
    print(f"Loaded reference dataset with {len(subset_ds)} rows.")
except Exception as e:
    print(f"Failed to load reference dataset: {e}")
    sys.exit(1)
ref_features = subset_ds.features
ref_keys = list(ref_features.keys())

# 2. Load Easy Processed Dataset (the two verified JSONL files)
print("\nLoading easy processed dataset...")
data_files = {
    'train': os.path.join(easy_processed_path, 'benchmark_r2e_gym_easy_train.jsonl'),
    'validation': os.path.join(easy_processed_path, 'benchmark_r2e_gym_easy_val.jsonl'),
}
# Rows of the reference dataset (verify_dataset.sh): MLLOG train prompts and
# eval_samples.
expected_rows = {'train': 700, 'validation': 251}
jsonl_rows = {}
for split, f in data_files.items():
    if not os.path.isfile(f):
        print(f"Missing verified dataset file: {f}")
        sys.exit(1)
    with open(f, encoding='utf-8') as fh:
        jsonl_rows[split] = sum(1 for line in fh if line.strip())
easy_ds = load_dataset('json', data_files=data_files)

# 3. Convert by flattening 'instance_dict'; any missing or malformed row is an
#    error (no row may be dropped).
print("\nConverting dataset...")
converted_splits = {}
for split in easy_ds.keys():
    print(f"Processing split: {split}")
    converted_rows = []
    for i, row in enumerate(easy_ds[split]):
        metadata = (row.get('responses_create_params') or {}).get('metadata') or {}
        instance_dict_str = metadata.get('instance_dict')
        if not instance_dict_str:
            print(f"Error: row {i} of split '{split}' has no instance_dict.")
            sys.exit(1)
        try:
            instance_dict = json.loads(instance_dict_str)
        except json.JSONDecodeError as e:
            print(f"Error: row {i} of split '{split}' has a malformed instance_dict: {e}")
            sys.exit(1)
        # Extract only the keys that match the reference dataset
        converted_rows.append({k: instance_dict.get(k) for k in ref_keys})
    n = len(converted_rows)
    if not n == len(easy_ds[split]) == jsonl_rows[split] == expected_rows[split]:
        print(f"Error: split '{split}' has {n} converted rows, {len(easy_ds[split])} loaded rows and "
              f"{jsonl_rows[split]} JSONL rows; expected {expected_rows[split]}.")
        sys.exit(1)
    try:
        # Enforce reference features to ensure matching schema and types
        converted_splits[split] = Dataset.from_list(converted_rows, features=ref_features)
        print(f"Created converted dataset for split '{split}' with {len(converted_splits[split])} rows.")
    except Exception as e:
        print(f"Failed to create Dataset for split '{split}' with reference features: {e}")
        sys.exit(1)
converted_ds = DatasetDict(converted_splits)

# 4. Save the converted dataset to disk
if os.path.exists(output_path):
    print(f"\nRemoving existing output directory: {output_path}")
    shutil.rmtree(output_path)
print(f"\nSaving converted dataset to {output_path}...")
try:
    converted_ds.save_to_disk(output_path)
    print("Dataset saved successfully.")
except Exception as e:
    print(f"Failed to save dataset: {e}")
    sys.exit(1)

# 5. Verify the saved dataset
print("\nVerifying saved dataset...")
try:
    loaded_ds = load_from_disk(output_path)
    print("Successfully loaded dataset from disk:")
    print(loaded_ds)
    for split in loaded_ds.keys():
        if loaded_ds[split].features != ref_features:
            print(f"Error: Features for split '{split}' do not match reference after saving/loading.")
            sys.exit(1)
        if len(loaded_ds[split]) != expected_rows[split]:
            print(f"Error: split '{split}' has {len(loaded_ds[split])} rows after saving/loading; expected {expected_rows[split]}.")
            sys.exit(1)
        print(f"Features and {expected_rows[split]} rows of split '{split}' match the reference.")
    print("\nSUCCESS: Dataset converted, saved, and verified.")
except Exception as e:
    print(f"Verification failed: {e}")
    sys.exit(1)

# 6. Sandbox image manifest: the task image of every training prompt, sorted
#    and deduplicated (700 images).
images = sorted(set(loaded_ds['train']['docker_image']))
with open(os.path.join(output_path, 'sandbox_images.txt'), 'w') as f:
    f.write('\n'.join(images) + '\n')
print(f"Wrote sandbox_images.txt with {len(images)} images.")
PY

  gcloud storage cp --recursive "${OUTPUT_HF_DIR}/"* "${DATASET_GCS_DIR}/"
fi

# -----------------------------------------------------------------------------
# 2. Initial Base Checkpoint Conversion (HF bfloat16 -> scanned MaxText checkpoint)
#    Source: ${HF_MODEL_ID} @ ${HF_MODEL_REVISION} (bfloat16, non-FP8)
#    Target: ${MAXTEXT_SCANNED_CKPT} (FSDP=32, TP=2, non-OCDBT Zarr layout)
#    The scored runs used a checkpoint in the Flax NNX layout (<name>.value)
#    with checkpoint_storage_use_ocdbt=false (non-OCDBT Zarr layout); see the README.
# -----------------------------------------------------------------------------
if [[ "${RUN_CKPT_CONVERSION:-0}" == "1" ]]; then
  # download_checkpoint.sh takes no arguments and downloads into the Hugging
  # Face cache under HF_HOME.
  export HF_HOME="${HF_HOME:-${HOME}/hf_home}"
  export HF_CKPT_DIR="${HF_CKPT_DIR:-${HF_HOME}/hub/models--${HF_MODEL_ID//\//--}/snapshots/${HF_MODEL_REVISION}}"
  export SCANNED_UNSHARDED_DIR="${SCANNED_UNSHARDED_DIR:-${MAXTEXT_SCANNED_CKPT%/scanned_reshard_fsdp32_tp2/0/items}/scanned}"
  export RESHARDED_BASE_DIR="${MAXTEXT_SCANNED_CKPT%/0/items}"

  # 2a. Download the official HuggingFace checkpoint (bfloat16)
  pushd "${MLCOMMONS_REF_DIR}"
  ./download_checkpoint.sh
  popd

  # 2b. Convert HuggingFace bfloat16 weights to scanned MaxText checkpoint
  #     (run from the MaxText source tree: the config path is relative).
  pushd "${MAXTEXT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../tunix/maxtext" && pwd)}"
  JAX_PLATFORMS=cpu python3 -m maxtext.checkpoint_conversion.to_maxtext \
    src/maxtext/configs/base.yml \
    model_name=qwen3.5-397b-a17b \
    base_output_directory="${SCANNED_UNSHARDED_DIR}" \
    hf_access_token="${HF_TOKEN:-}" \
    scan_layers=true \
    use_multimodal=false \
    checkpoint_storage_use_ocdbt=false \
    checkpoint_storage_use_zarr3=false \
    --hf_model_path="${HF_CKPT_DIR}" \
    --lazy_load_tensors=true \
    --save_dtype=bfloat16

  # 2c. Reshard scanned MaxText checkpoint to (FSDP=32, TP=2) non-OCDBT Zarr layout
  #     on 64 simulated CPU devices. The stored sharding does not have to match
  #     the training mesh: the trainer restores the arrays onto its own
  #     256-device mesh (FSDP=32, CP=4, EP=2, TP=1).
  python3 -m maxtext.checkpoint_conversion.reshard_checkpoint \
    src/maxtext/configs/base.yml \
    model_name=qwen3.5-397b-a17b \
    scan_layers=true \
    use_multimodal=false \
    load_parameters_path="${SCANNED_UNSHARDED_DIR}/0/items" \
    base_output_directory="${RESHARDED_BASE_DIR}" \
    weight_dtype=bfloat16 \
    checkpoint_storage_concurrent_gb=1024 \
    checkpoint_storage_use_ocdbt=false \
    checkpoint_storage_use_zarr3=false \
    skip_jax_distributed_system=true \
    ici_fsdp_parallelism=32 \
    ici_tensor_parallelism=2 \
    --simulated_cpu_devices_count=64
  popd
fi

echo "Verify dataset and checkpoint paths:"
echo "  Dataset (HF DatasetDict): ${DATASET_GCS_DIR}/{dataset_dict.json,train,validation,sandbox_images.txt}"
echo "  Checkpoint (MaxText):     ${MAXTEXT_SCANNED_CKPT}"

