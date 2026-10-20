# Reproduction release

The task prompt, tool schemas, fixed scene/camera catalog, geometric matching and
IoU formulas are preserved. The public code contains evaluation, task preparation,
environment setup, resource attachment, and IoU scoring only, plus publication
helpers and the three README illustrations. Project plotting scripts, historical
trajectory analysis, old export scripts and paper-building code are excluded.

Portability additions:
- `scripts/attach_data.py`: attach the separate SpatialScout dataset.
- `scripts/run_evaluation.sh`: launch the 80-task evaluation using your own API.
- `spatialscout/evaluation/score.py`: successful finish_mapping filtering and all eight metrics,
  with both task-average macro and instance-average micro results.
- `spatialscout/benchmark/batch.py`: selectable Responses/Chat Completions API mode.
- `spatialscout/data/index.py`: relative source-file metadata support.
- `setup_headless.sh`: delegates to the portable CPU installation script.
- `scripts/upload_github.py` and `scripts/upload_huggingface.py`: publication helpers.

The vendored `ManiSkill3/mani_skill` package is retained for simulation and includes
internal rendering/visualization dependencies. Upstream training examples (including the package examples), tests,
documentation trees and tutorials are excluded. Runtime binary resources remain
in the separate dataset folder. No new model evaluation or fresh environment
installation was run during release preparation.


## Package organization

The former top-level Python scripts are now grouped under `spatialscout/agent`,
`benchmark`, `evaluation`, `data`, and `tools`. Imports use absolute package paths;
`spatialscout.paths` owns the repository root and catalog path. Subprocesses launch
modules with the current Python interpreter. Commands in the launchers, Docker
image and READMEs have been updated. Configuration files and dependency lists
now live under `configs/` and `requirements/`. Scoring formulas and task settings
are unchanged. The CPU installer installs SpatialScout in editable mode.
