#!/usr/bin/env bash
set -euo pipefail
render_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$render_root"
export MS_ASSET_DIR="$render_root/assets"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/slam-bench-runtime-$(id -u)}"
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"
if [[ -z ${VK_ICD_FILENAMES:-} ]]; then
  shopt -s nullglob
  render_icds=(/usr/share/vulkan/icd.d/lvp_icd*.json /etc/vulkan/icd.d/lvp_icd*.json)
  if [[ ${#render_icds[@]} -eq 0 ]]; then
    printf 'Mesa lavapipe Vulkan ICD missing; run setup_render_environment.sh.\n' >&2
    exit 1
  fi
  export VK_ICD_FILENAMES="${render_icds[0]}"
fi
if [[ -f /usr/share/glvnd/egl_vendor.d/50_mesa.json ]]; then
  export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
fi
export LP_NUM_THREADS="${LP_NUM_THREADS:-8}"
if [[ ! -x .venv/bin/python ]]; then
  printf 'Environment missing; run bash setup_render_environment.sh first.\n' >&2
  exit 1
fi
if [[ $# -eq 0 ]]; then
  set -- -m spatialscout.tools.capture --size 256
fi
exec "$render_root/.venv/bin/python" "$@"
