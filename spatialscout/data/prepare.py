"""Load and render every scene in a catalog batch, then record target counts."""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np


from spatialscout.paths import ROOT, CATALOG_PATH


def prepare(entry, output_root):
    directory = output_root / entry["name"]
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "ground_truth.json"
    image_path = directory / "gt_view_0.png"
    if manifest_path.is_file() and image_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        start = entry["camera_start"]
        same_camera = np.allclose(manifest["origin_world_m"], start[:3]) and np.allclose(manifest["initial_camera_yaw_pitch_deg"], start[3:])
        if same_camera and len(manifest["objects"]) == entry["target_count"] and manifest["loaded_object_count"] >= entry["target_count"]:
            return {"name": entry["name"], "ok": True, "loaded": manifest["loaded_object_count"], "targets": len(manifest["objects"]), "skipped": True}
    command = [
        sys.executable, "-m", "spatialscout.benchmark.room",
        "--mode", "prepare", "--catalog-scene", entry["name"], "--output-dir", str(directory),
    ]
    with (directory / "prepare.log").open("w") as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        return {"name": entry["name"], "ok": False, "exit_code": result.returncode, "log": str(directory / "prepare.log")}
    manifest = json.loads(manifest_path.read_text())
    return {
        "name": entry["name"], "ok": image_path.is_file() and len(manifest["objects"]) == entry["target_count"],
        "loaded": manifest["loaded_object_count"], "targets": len(manifest["objects"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", default="additional_50")
    parser.add_argument("--parallelism", type=int, default=4)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "room_benchmark" / "additional_50")
    args = parser.parse_args()
    if args.parallelism < 1:
        parser.error("parallelism must be positive")
    entries = [x for x in json.loads(CATALOG_PATH.read_text())["scenes"] if x.get("batch") == args.batch]
    if not entries:
        parser.error(f"No scenes in batch {args.batch}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    results = []
    with ThreadPoolExecutor(max_workers=args.parallelism) as pool:
        futures = [pool.submit(prepare, entry, args.output_root) for entry in entries]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"{result['name']}: {result}", flush=True)
    summary = {"batch": args.batch, "results": sorted(results, key=lambda x: x["name"])}
    (args.output_root / "prepare_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    if not all(result["ok"] for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
