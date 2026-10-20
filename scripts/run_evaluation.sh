#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${VLM_BASE_URL:?Export the model API base URL}"
: "${VLM_API_KEY:?Export your own model API key}"
: "${VLM_MODEL:?Export the exact model ID provided by your API}"
exec bash render_env.sh -m spatialscout.benchmark.batch \
  --batch prepared_80 --model "$VLM_MODEL" \
  --api-backend "${VLM_API_BACKEND:-openai}" \
  --api-mode "${VLM_API_MODE:-responses}" \
  --reasoning-effort "${REASONING_EFFORT:-xhigh}" \
  --max-turns 100 --parallelism "${PARALLELISM:-1}" \
  --output-root "${EVAL_OUTPUT_DIR:-outputs/room_benchmark/reproduction}" "$@"
