#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export VLM_API_BACKEND=gemini
export VLM_LAUNCHER_NAME=$(basename "$0")
export VLM_DEFAULT_REASONING_EFFORT=high
exec "$script_dir/run_prepared_80.sh" "$@"
