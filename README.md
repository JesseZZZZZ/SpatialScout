# SpatialScout
[![arXiv](https://img.shields.io/badge/Arxiv-SpatialScout-red?style=flat)](https://arxiv.org/pdf/2610.26659)
[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-SpatialScout-FFD21E)](https://huggingface.co/datasets/jessezhaoxizhang/SpatialScout)

![SpatialScout overview](docs/images/spatialscout_camera_overview_v3_white.png)

Interactive 3D object localization with vision-language models: **80 scenes and 3,140 target objects**.

## Installation

Requires **Linux x86_64 and Python 3.10 with venv support**. The installer sets up
CPU rendering and installs system packages through apt/sudo.

```bash
git clone https://github.com/JesseZZZZZ/SpatialScout.git
cd SpatialScout
RENDER_PYTHON=python3.10 bash setup_render_environment.sh

# Download and attach the scene resources.
.venv/bin/python - <<'PYTHON'
from huggingface_hub import snapshot_download
snapshot_download("jessezhaoxizhang/SpatialScout", repo_type="dataset",
                  local_dir="../SpatialScout-data")
PYTHON
.venv/bin/python scripts/attach_data.py ../SpatialScout-data
```

## Evaluation

Run all 80 tasks with your own model API (100 turns per task):

```bash
export VLM_BASE_URL='https://YOUR_API_HOST/v1'
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

## Citation

If you use SpatialScout, please cite our paper:

```bibtex
@misc{zhang2026spatialscoutvlmagentsactively,
      title={SpatialScout: Can VLM Agents Actively Explore and Map 3D Scenes?},
      author={Zhaoxi Zhang and Hao Liang and Linxuan Wu and Yiming Wu and Yifu Yuan and Kun Liang and Chenming Tang and Meiyi Qiang and Wentao Zhang and Weikang Li},
      year={2026},
      eprint={2610.26659},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2610.26659},
}
```
