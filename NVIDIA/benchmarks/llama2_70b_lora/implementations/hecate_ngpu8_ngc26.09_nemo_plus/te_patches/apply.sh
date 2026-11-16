#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

# Apply the TE patches of the LoRA benchmark to the installed transformer_engine.
# Each patch stays off until you set its switch (see the patch). If a patch does not fit the
# installed TE (for example after a TE update in the base image), skip it with a warning.

set -euo pipefail

patch_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# find_spec does not import TE, so this step needs no GPU.
site_dir="$(python -c 'import importlib.util, os; spec = importlib.util.find_spec("transformer_engine"); print(os.path.dirname(spec.submodule_search_locations[0]))')"

for patch_file in "${patch_dir}"/*.patch; do
    if patch --dry-run --forward --silent --directory="${site_dir}" -p1 < "${patch_file}"; then
        patch --forward --silent --directory="${site_dir}" -p1 < "${patch_file}"
        echo "te_patches: applied $(basename "${patch_file}") to ${site_dir}"
    else
        echo "te_patches: WARNING: $(basename "${patch_file}") does not fit the installed TE, skipped" >&2
    fi
done
