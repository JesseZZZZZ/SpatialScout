# 复现 SpatialScout 评测

[![Hugging Face 数据集](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-SpatialScout-FFD21E)](https://huggingface.co/datasets/jessezhaoxizhang/SpatialScout)

本发布目录的目的是让其他人**用自己的模型 API 重新运行评测**，不包含已有模型
原始输出或轨迹；README 展示三张论文说明图。代码和数据地址：

- GitHub：https://github.com/JesseZZZZZ/SpatialScout
- Hugging Face：https://huggingface.co/datasets/jessezhaoxizhang/SpatialScout

完整参数、指标定义和 Docker 命令见英文 README。

## 代码结构

- `spatialscout/agent/`：相机工具、模型 API 适配和 token 记录。
- `spatialscout/benchmark/`：单场景任务和 80 场景批量评测。
- `spatialscout/evaluation/`：IoU 匹配、有效轨迹筛选和八项指标。
- `spatialscout/data/`：资源下载、场景准备、索引和初始相机选择。
- `spatialscout/tools/`：截图工具及 SAPIEN CPU 补丁。
- `configs/`：场景目录及原实验模型设置；`requirements/`：环境依赖。
- `scripts/`：一键评测、数据接入和上传脚本；`docs/`：说明与 README 图片。

Python 入口统一使用 `python -m spatialscout.…`。仿真入口通过
`bash render_env.sh -m spatialscout.…` 启动。以下命令均在仓库根目录执行。

## 一、环境

- Linux x86_64；Python **3.10**（含 venv 支持）。
- 无需 GPU、桌面或 X server；CPU 物理仿真 + Mesa lavapipe 无界面 Vulkan 渲染。
- SAPIEN **3.0.0b1**、NumPy **1.26.4**、PyTorch **2.5.1+cpu** 和固定 ManiSkill3 源码。
- 首次安装需要网络；系统依赖由 apt 安装，普通用户需要 sudo。
- 新机器使用 `setup_render_environment.sh`，不要直接安装带原机器 CUDA 依赖的
  `requirements/frozen.txt`。CPU SAPIEN 补丁由安装脚本自动应用。

## 二、接入数据并安装

下载 HF 数据目录，确认其中有 `assets/`、`runtime/`、`benchmark/`、`MANIFEST.json`。
Ubuntu 22.04 可先执行 `sudo apt-get install python3.10 python3.10-venv`。
其他发行版推荐使用 Docker，或自行提供 Python 3.10。进入 GitHub 代码根目录：

```bash
export SPATIALSCOUT_DATA=/你的绝对路径/SpatialScout
python3 scripts/attach_data.py "$SPATIALSCOUT_DATA"
RENDER_PYTHON=python3.10 bash setup_render_environment.sh
```

接入脚本创建符号链接；不要移动或删除数据目录。可给接入命令添加 `--verify`
核对 SHA-256。数据不是模型输出，其中的 GT 仅供评分，不能作为模型输入。

不请求 API，仅生成一个场景：

```bash
bash render_env.sh -m spatialscout.benchmark.room --mode prepare \
  --catalog-scene ithor_kitchen_13 --output-dir outputs/prepare_demo
```

所有仿真命令通过 `render_env.sh` 启动，以正确配置 Vulkan/EGL 和资源路径。

## 三、运行正式 80 个任务

```bash
export VLM_BASE_URL='https://你的接口地址/v1'
export VLM_API_KEY='你自己的密钥'
export VLM_MODEL='接口实际支持的完整模型ID'
export VLM_API_BACKEND=openai
export VLM_API_MODE=responses
export REASONING_EFFORT=xhigh
export PARALLELISM=1
export EVAL_OUTPUT_DIR=outputs/room_benchmark/my_model_xhigh
bash scripts/run_evaluation.sh
```

- 正式评测使用 `prepared_80`：80 场景，3,140 个目标物体；每场景最多 **100 次模型响应**。
- `configs/scene_catalog.json` 还保留 10 个开发场景；不要用 `--batch all` 代替正式 80 场景。
- Responses 接口设置 `VLM_API_MODE=responses`，Chat Completions 设置为 `chat`。
- Anthropic/Gemini 原生接口分别设置 `VLM_API_BACKEND=anthropic` 或 `gemini`。
  Gemini 仅支持 low/medium/high；接口模型必须支持图像和工具调用。
- medium 模式改为 `REASONING_EFFORT=medium`，并使用新的输出目录。
- 批次中断后保持原模型和参数，用 `bash scripts/run_evaluation.sh --resume` 续跑。
  runner 的“已完成”是文件/进程状态，评分还会检查 finish_mapping。达到轮数上限
  但被 runner 标记完成的任务需另行重跑。

## 四、汇报八个分数

```bash
.venv/bin/python -m spatialscout.evaluation.score "$EVAL_OUTPUT_DIR"
```

分别统计“全部 GT 物体”和“已提交物体”上的：平均 IoU、IoU > 0%、> 50%、> 90%
的物体比例，共八项。严格大于阈值；缺失物体在全部物体指标中记 0。
只纳入最后执行工具是 `finish_mapping` 且返回 `finished=true` 的任务；中间可恢复
工具错误不影响资格。JSON 中列出每项未纳入任务的原因。

主要汇报 **macro：先逐任务计算，再对有效任务求平均**，分母是该模型有效任务数；
同时输出 micro 供核查。务必同时汇报有效任务数，不要混用 macro 和 micro。
文件中的分数为 0–1，终端打印为百分数。输出为运行目录中的
`eight_metrics.json` / `.csv`。

## 五、Docker

无需宿主机 Python 3.10，可按英文 README 的 Docker 步骤构建并挂载数据目录。
镜像首次构建仍需网络。此发布整理过程没有在新机器重新安装环境、构建镜像或
运行模型评测；不能声称已完成跨机器验证。模型服务随机性及版本更新可能影响精确分数。


## 六、一键上传

```bash
python3 scripts/upload_github.py
python3 scripts/upload_huggingface.py --folder "$SPATIALSCOUT_DATA"
```

分别输入 GitHub 个人访问令牌（仓库 Contents 读写权限）和 Hugging Face 写入令牌，
不是模型 API key。输入不显示，不写入代码或远端 URL。HF 脚本需要时会在独立缓存
环境安装上传依赖。完整说明见 [docs/publishing.md](docs/publishing.md)。
