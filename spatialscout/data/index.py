"""Index prepared additional_50 and procthor_30 scene artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


from spatialscout.paths import ROOT, CATALOG_PATH
BATCHES = ("additional_50", "procthor_30")


def main() -> None:
    catalog = json.loads(CATALOG_PATH.read_text())["scenes"]
    scenes = []
    for entry in catalog:
        batch = entry.get("batch")
        if batch not in BATCHES:
            continue
        directory = ROOT / "outputs" / "room_benchmark" / batch / entry["name"]
        gt_path = directory / "ground_truth.json"
        library_path = directory / "entity_library.json"
        image_path = directory / "gt_view_0.png"
        for path in (gt_path, library_path, image_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        gt = json.loads(gt_path.read_text())
        library = json.loads(library_path.read_text())
        if len(gt["objects"]) != entry["target_count"] or len(library) != entry["target_count"]:
            raise ValueError(f"Wrong target count in {entry['name']}")
        if gt["scene_builder"] != entry["builder"] or {item["id"] for item in library} != {item["id"] for item in gt["objects"]}:
            raise ValueError(f"GT/library mismatch in {entry['name']}")
        if entry.get("source_object_count") is not None and gt["loaded_object_count"] != entry["source_object_count"]:
            raise ValueError(f"Loaded object count differs from original scene: {entry['name']}")
        if not all(np.isfinite(item["xyz_m"]).all() for item in gt["objects"]):
            raise ValueError(f"Non-finite GT coordinate in {entry['name']}")
        if not np.allclose(gt["origin_world_m"], entry["camera_start"][:3]) or not np.allclose(gt["initial_camera_yaw_pitch_deg"], entry["camera_start"][3:]):
            raise ValueError(f"GT camera does not match catalog: {entry['name']}")
        source_path = Path(gt["source_file"])
        if not source_path.is_absolute():
            source_path = ROOT / source_path
        if not source_path.is_file():
            raise FileNotFoundError(gt["source_file"])
        scenes.append({
            "name": entry["name"],
            "batch": batch,
            "builder": entry["builder"],
            "source_scene": entry["source_scene"],
            "loaded_object_count": gt["loaded_object_count"],
            "target_count": len(gt["objects"]),
            "camera_start": entry["camera_start"],
            "entity_library": str(library_path.relative_to(ROOT)),
            "ground_truth": str(gt_path.relative_to(ROOT)),
            "preview_png": str(image_path.relative_to(ROOT)),
        })
    summary = {
        "description": "Prepared scene index; GT is for evaluation and is not exposed to the VLM.",
        "scene_count": len(scenes),
        "target_count": sum(scene["target_count"] for scene in scenes),
        "coordinate_frame": "+x initial camera forward, +y initial camera left, +z initial camera up; meters",
        "scenes": scenes,
    }
    output = ROOT / "outputs" / "room_benchmark" / "prepared_80_index.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(f"Indexed {len(scenes)} prepared scenes and {summary['target_count']} targets: {output}")


if __name__ == "__main__":
    main()
