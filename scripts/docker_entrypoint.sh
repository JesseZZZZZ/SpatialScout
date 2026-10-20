#!/usr/bin/env bash
set -euo pipefail
cd /workspace
if [[ ! -f /data/benchmark/prepared_80_index.json ]]; then
  printf 'Mount the SpatialScout dataset at /data (read-only is supported).\n' >&2
  exit 1
fi
python scripts/attach_data.py /data
exec bash render_env.sh "$@"
