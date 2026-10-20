#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [[ "${SKIP_SYSTEM_PACKAGES:-0}" != 1 ]]; then
  if [[ $(id -u) -eq 0 ]]; then
    apt-get update
    apt-get install -y libvulkan1 mesa-vulkan-drivers libegl1 libegl-mesa0 libgl1 libglx-mesa0 libglib2.0-0 libgomp1 libsm6 libxext6 libxrender1 python3-venv build-essential cmake git ca-certificates
  else
    sudo apt-get update
    sudo apt-get install -y libvulkan1 mesa-vulkan-drivers libegl1 libegl-mesa0 libgl1 libglx-mesa0 libglib2.0-0 libgomp1 libsm6 libxext6 libxrender1 python3-venv build-essential cmake git ca-certificates
  fi
fi

render_python=${RENDER_PYTHON:-python3.10}
if ! command -v "$render_python" >/dev/null 2>&1; then
  printf 'Python 3.10 is required. Set RENDER_PYTHON to its executable or use Dockerfile.render.\n' >&2
  exit 1
fi
"$render_python" -c 'import platform, sys; assert sys.version_info[:2] == (3, 10), "Python 3.10 required"; assert platform.system() == "Linux" and platform.machine() in ("x86_64", "AMD64"), "Linux x86_64 required"'
if [[ ! -x .venv/bin/python ]]; then
  "$render_python" -m venv .venv
fi
.venv/bin/python -m pip install 'pip>=24' 'setuptools==80.10.2' wheel
.venv/bin/python -m pip install 'torch==2.5.1+cpu' --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements/render-cpu.txt
.venv/bin/python -m pip install --no-deps -e ./ManiSkill3
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python -m spatialscout.tools.patch_sapien_cpu
printf '\nInstalled. Render with: bash render_env.sh -m spatialscout.tools.capture --size 256\n'
