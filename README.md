# SpatialScout

![SpatialScout overview](docs/images/spatialscout_camera_overview_v3_white.png)

Interactive 3D object localization with vision-language models: **80 scenes and 3,140 target objects**.

## Installation

Requires **Linux x86_64 and Python 3.10 with venv support**. The installer sets up
CPU rendering and installs system packages through apt/sudo.

```bash
# Set TORCH_INDEX_URL to your CPU wheel package index before installation.
RENDER_PYTHON=python3.10 bash setup_render_environment.sh

# Attach the separately supplied scene resources.
.venv/bin/python scripts/attach_data.py ../SpatialScout-data
```

## Evaluation

Run all 80 tasks with your own model API (100 turns per task):

```bash
# Set MODEL_API_BASE to your model service endpoint.
export VLM_BASE_URL="$MODEL_API_BASE"
export VLM_API_KEY='YOUR_MODEL_API_KEY'
export VLM_MODEL='EXACT_MODEL_ID'
export REASONING_EFFORT=xhigh
export EVAL_OUTPUT_DIR=outputs/room_benchmark/my_model
bash scripts/run_evaluation.sh

# Compute all eight IoU metrics.
.venv/bin/python -m spatialscout.evaluation.score "$EVAL_OUTPUT_DIR"
```

The default API mode is Responses. For Chat Completions, set `VLM_API_MODE=chat`;
for native Anthropic/Gemini endpoints, set `VLM_API_BACKEND` accordingly.
Original model settings: [configs/model_settings.json](configs/model_settings.json).

Scores cover mean IoU and fractions above 0%, 50%, and 90%, for all objects and
submitted objects. Task-average scores use trajectories ending in successful
`finish_mapping`.

[Detailed setup, Docker and protocol](docs/reproduction.md) · [Code structure](docs/architecture.md) · [中文](README_zh.md)

## Results

### Qualitative reconstruction

![Accurate, incomplete, and mislocalized reconstructions](docs/images/qualitative_reconstruction_examples.png)

### Reconstruction IoU vs. cost

![Reconstruction IoU versus cost per task](docs/images/performance_cost_pareto.png)
