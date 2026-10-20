#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'Usage: %s BASE_URL API_KEY N_CONCURRENT MODEL_NAME [--reasoning-effort LEVEL] [--resume-path PATH]\n' "${VLM_LAUNCHER_NAME:-$(basename "$0")}" >&2
  if [[ ${VLM_API_BACKEND:-openai} == gemini ]]; then
    printf 'LEVEL: low, medium, high (default: high). At most 100 agent turns per scene.\n' >&2
  else
    printf 'LEVEL: none, low, medium, high, xhigh, max (default: xhigh). At most 100 agent turns per scene.\n' >&2
  fi
}

if [[ $# -lt 4 ]]; then
  usage
  exit 2
fi

base_url=$1
api_key=$2
n_concurrent=$3
model_name=$4
shift 4

resume_path=
reasoning_effort=${VLM_DEFAULT_REASONING_EFFORT:-xhigh}
while [[ $# -gt 0 ]]; do
  case $1 in
    --reasoning-effort)
      if [[ $# -lt 2 ]]; then
        usage
        exit 2
      fi
      reasoning_effort=$2
      shift 2
      ;;
    --resume-path)
      if [[ $# -lt 2 ]]; then
        usage
        exit 2
      fi
      if [[ -z $2 ]]; then
        usage
        exit 2
      fi
      resume_path=$2
      shift 2
      ;;
    *)
      usage
      exit 2
      ;;
  esac
done

if [[ -z $base_url || -z $api_key || -z $model_name || ! $n_concurrent =~ ^[1-9][0-9]*$ || ! $reasoning_effort =~ ^(none|low|medium|high|xhigh|max)$ ]]; then
  usage
  exit 2
fi
if [[ ${VLM_API_BACKEND:-openai} == gemini && ! $reasoning_effort =~ ^(low|medium|high)$ ]]; then
  usage
  exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python_bin="$script_dir/.venv/bin/python"
if [[ ! -x $python_bin ]]; then
  printf 'Python environment missing: %s\n' "$python_bin" >&2
  exit 1
fi

runner_args=()
if [[ -n $resume_path ]]; then
  if [[ ! -d $resume_path ]]; then
    printf 'Resume directory not found: %s\n' "$resume_path" >&2
    exit 1
  fi
  output_root=$(cd -- "$resume_path" && pwd)
  runner_args+=(--resume)
  run_mode=resume
else
  output_root="$script_dir/outputs/room_benchmark/run_$(date -u +%Y%m%dT%H%M%SZ)_$$"
  run_mode=new
fi
export VLM_BASE_URL="$base_url"
export VLM_API_KEY="$api_key"

printf 'Running prepared_80: model=%s effort=%s concurrent=%s output=%s mode=%s\n' "$model_name" "$reasoning_effort" "$n_concurrent" "$output_root" "$run_mode"
exec bash "$script_dir/render_env.sh" -m spatialscout.benchmark.batch \
  --batch prepared_80 \
  --parallelism "$n_concurrent" \
  --model "$model_name" \
  --reasoning-effort "$reasoning_effort" \
  --max-turns 100 \
  --output-root "$output_root" \
  "${runner_args[@]}"
