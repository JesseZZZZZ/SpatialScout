For the separated GitHub/Hugging Face reproduction release, follow README.md
or README_zh.md. Attach the dataset first using scripts/attach_data.py.

Native Linux CPU environment:
  RENDER_PYTHON=python3.10 bash setup_render_environment.sh
  bash render_env.sh -m spatialscout.benchmark.room --mode prepare --catalog-scene ithor_kitchen_13

Docker installation and model evaluation commands are documented in README.md.
No historical model outputs are included or needed for running a fresh evaluation.
