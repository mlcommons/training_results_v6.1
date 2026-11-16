#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

# Default the cache to the in-repo hf_hub/v1/ directory so config_mounts.sh can
# mount it as /hf_home in synthetic runs without requiring the user to point
# HF_DIR at an external location. The "v1" subdir matches the layout used in
# config_data_*.sh. Override by exporting HF_DIR=<path>.
: "${HF_DIR:=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/hf_hub/v1}"

cache_root="$HF_DIR"
commit="e815299b0bcbac849fa540c768ef21845365c9eb"
repo_cache_dir="$cache_root/hub/models--deepseek-ai--DeepSeek-V3"
snapshot_dir="$repo_cache_dir/snapshots/$commit"
modules_dir="$cache_root/modules/transformers_modules/deepseek_hyphen_ai/DeepSeek_hyphen_V3/$commit"
files=("config.json" "configuration_deepseek.py")

mkdir -p "$repo_cache_dir/refs" "$repo_cache_dir/blobs" "$snapshot_dir" "$modules_dir"
printf '%s' "$commit" > "$repo_cache_dir/refs/main"

for file in "${files[@]}"; do
  curl -fL "https://huggingface.co/deepseek-ai/DeepSeek-V3/resolve/${commit}/${file}" \
    -o "$snapshot_dir/$file"
  cp "$snapshot_dir/$file" "$modules_dir/$file"
done

# make packages importable
touch \
  "$cache_root/modules/__init__.py" \
  "$cache_root/modules/transformers_modules/__init__.py" \
  "$cache_root/modules/transformers_modules/deepseek_hyphen_ai/__init__.py" \
  "$cache_root/modules/transformers_modules/deepseek_hyphen_ai/DeepSeek_hyphen_V3/__init__.py" \
  "$modules_dir/__init__.py"