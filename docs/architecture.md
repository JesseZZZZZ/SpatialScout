# Code organization

## Runtime and evaluation

| Module | Responsibility |
| --- | --- |
| `spatialscout.agent.harness` | Camera control, image input, model API adapters |
| `spatialscout.agent.token_usage` | Token usage collection |
| `spatialscout.benchmark.room` | Task prompt, tool schemas, scene loading, object submission |
| `spatialscout.benchmark.batch` | Concurrency, 80-task scheduling, resume handling |
| `spatialscout.evaluation.iou` | Bounding-box IoU and equivalence-group matching |
| `spatialscout.evaluation.score` | finish_mapping eligibility, eight metrics, macro/micro summaries |
| `spatialscout.paths` | Repository root and catalog path |

## Dataset and environment utilities

| Module | Responsibility |
| --- | --- |
| `spatialscout.data.download` | Scene-resource subset download |
| `spatialscout.data.prepare` | Prepare task annotations and reference views |
| `spatialscout.data.index` | Index the 80 prepared tasks |
| `spatialscout.data.select_camera` | Dataset-development camera selection |
| `spatialscout.tools.capture` | Standalone simulator capture |
| `spatialscout.tools.patch_sapien_cpu` | Patch the pinned SAPIEN CPU-rendering binary |

`configs/scene_catalog.json` defines scene membership, target selection and initial
camera poses. `configs/model_settings.json` records the original model/adapter
settings. The simulator source stays in `ManiSkill3/`; scene resources stay in the
separate Hugging Face dataset. `scripts/` contains small operational entry points.

## Entry points

From the repository root after installation and dataset attachment:

```bash
# Prepare a task without model API calls.
bash render_env.sh -m spatialscout.benchmark.room --mode prepare \
  --catalog-scene ithor_kitchen_13 --output-dir outputs/prepare_demo

# Evaluate 80 tasks using the API/environment variables described in README.
bash scripts/run_evaluation.sh

# Compute all eight IoU metrics for newly generated results.
.venv/bin/python -m spatialscout.evaluation.score outputs/room_benchmark/my_model_xhigh
```

The CPU installer installs both ManiSkill3 and SpatialScout in editable mode.
Keep the checkout and the attached dataset available after installation.
