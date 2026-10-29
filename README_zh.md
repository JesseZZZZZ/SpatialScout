# SpatialScout


![SpatialScout overview](docs/images/spatialscout_camera_overview_v3_white.png)

基于视觉语言模型的交互式 3D 物体定位评测：**80 个场景、3,140 个目标物体**。

## Installation

需要 **Linux x86_64 和 Python 3.10（含 venv）**。脚本通过 apt/sudo 安装系统依赖，
使用 CPU 无界面渲染。

```bash
# Set TORCH_INDEX_URL to your CPU wheel package index before installation.
RENDER_PYTHON=python3.10 bash setup_render_environment.sh

# Attach the separately supplied scene resources.
.venv/bin/python scripts/attach_data.py ../SpatialScout-data
```

## Evaluation

配置自己的模型 API，运行全部 80 个任务（每个任务最多 100 轮）：

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

默认为 Responses 接口；Chat Completions 设置 `VLM_API_MODE=chat`。
Anthropic/Gemini 原生接口设置 `VLM_API_BACKEND=anthropic` 或 `gemini`。
原实验配置见 [configs/model_settings.json](configs/model_settings.json)。

八项指标为全部物体和已提交物体上的平均 IoU、IoU > 0%、> 50%、> 90% 的比例。
主要使用逐任务平均，仅纳入成功调用 `finish_mapping` 结束的轨迹。

[完整安装、Docker 和评测说明](docs/reproduction.md) · [代码结构](docs/architecture.md)

## Results

### 重建效果

![重建效果对比](docs/images/qualitative_reconstruction_examples.png)

### IoU 与费用

![IoU 与每任务平均费用](docs/images/performance_cost_pareto.png)
