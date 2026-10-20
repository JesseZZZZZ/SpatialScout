"""Shared repository paths, independent of the current working directory."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "configs" / "scene_catalog.json"
