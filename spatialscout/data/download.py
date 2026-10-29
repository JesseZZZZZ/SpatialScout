"""Download only the AI2THOR assets required by selected ManiSkill scenes."""
from __future__ import annotations

import argparse
import json
import os
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from remotezip import RemoteZip


from spatialscout.paths import ROOT, CATALOG_PATH
URL = os.environ.get("SPATIALSCOUT_ASSET_URL", "")
DEFAULT_SCENES = (
    "FloorPlan1", "FloorPlan7", "FloorPlan201", "FloorPlan227",
    "FloorPlan301", "FloorPlan320", "FloorPlan401", "FloorPlan421",
    "ArchitecTHOR-Test-00", "ArchitecTHOR-Val-00",
)
_local = threading.local()


def archive() -> RemoteZip:
    if not hasattr(_local, "archive"):
        _local.archive = RemoteZip(URL)
    return _local.archive


def fetch(path: str, destination: Path) -> str:
    output = destination / path.removeprefix("ai2thor/")
    if output.is_file():
        return path
    for attempt in range(5):
        try:
            data = archive().read(path)
            break
        except Exception:
            if attempt == 4:
                raise
            if hasattr(_local, "archive"):
                _local.archive.close()
                del _local.archive
            time.sleep(2 ** attempt)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_bytes(data)
    temporary.replace(output)
    return path


def repair_texcoords(path: Path) -> bool:
    """SAPIEN rejects some source GLBs with TEXCOORD_1 but no TEXCOORD_0."""
    blob = path.read_bytes()
    magic, version, length = struct.unpack_from("<III", blob)
    if magic != 0x46546C67 or version != 2 or length != len(blob):
        raise ValueError(f"Invalid GLB header: {path}")
    chunks = []
    offset = 12
    while offset < len(blob):
        chunk_length, chunk_type = struct.unpack_from("<II", blob, offset)
        offset += 8
        chunks.append((chunk_type, blob[offset:offset + chunk_length]))
        offset += chunk_length
    if not chunks or chunks[0][0] != 0x4E4F534A:
        raise ValueError(f"GLB has no JSON chunk: {path}")
    document = json.loads(chunks[0][1])
    changed = False
    for mesh in document.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            attributes = primitive.get("attributes", {})
            present = [int(name.split("_")[1]) for name in attributes if name.startswith("TEXCOORD_")]
            if present:
                for index in range(max(present)):
                    name = f"TEXCOORD_{index}"
                    if name not in attributes:
                        attributes[name] = attributes[f"TEXCOORD_{min(present)}"]
                        changed = True
                ordered = dict(sorted(attributes.items(), key=lambda entry: (0, int(entry[0].split("_")[1])) if entry[0].startswith("TEXCOORD_") else (1, entry[0])))
                if list(ordered) != list(attributes):
                    primitive["attributes"] = ordered
                    changed = True
    if not changed:
        return False
    data = json.dumps(document, separators=(",", ":")).encode()
    data += b" " * (-len(data) % 4)
    chunks[0] = (0x4E4F534A, data)
    body = b"".join(struct.pack("<II", len(data), kind) + data for kind, data in chunks)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_bytes(struct.pack("<III", 0x46546C67, 2, 12 + len(body)) + body)
    temporary.replace(path)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenes", nargs="*")
    parser.add_argument("--catalog-batch", choices=("original_10", "additional_50", "procthor_30", "all"), help="Download the AI2THOR scenes in a catalog batch")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--destination", type=Path, default=ROOT / "assets" / "scene_datasets" / "ai2thor")
    args = parser.parse_args()
    if not URL:
        parser.error("Set SPATIALSCOUT_ASSET_URL or attach the supplied local dataset with scripts/attach_data.py")
    if args.catalog_batch and args.scenes:
        parser.error("Pass scene names or --catalog-batch, not both")
    if args.catalog_batch:
        catalog = json.loads(CATALOG_PATH.read_text())["scenes"]
        args.scenes = list(dict.fromkeys(
            entry.get("config_file", entry["source_scene"]) for entry in catalog
            if entry["builder"] != "ReplicaCAD" and (args.catalog_batch == "all" or entry.get("batch", "original_10") == args.catalog_batch)
        ))
    elif not args.scenes:
        args.scenes = list(DEFAULT_SCENES)
    paths = {"ai2thor/ai2thor-hab/configs/object_semantic_id_mapping.json"}
    with RemoteZip(URL) as source:
        available = set(source.namelist())
        for name in args.scenes:
            if name.startswith("FloorPlan"):
                if name.startswith("FloorPlan_"):
                    prefix = "RoboTHOR"
                    filename = f"{name}.scene_instance.json"
                else:
                    prefix = "iTHOR"
                    filename = f"{name.removesuffix('_physics')}_physics.scene_instance.json"
            elif name.startswith("ArchitecTHOR-"):
                prefix = "ArchitecTHOR"
                filename = f"{name}.scene_instance.json"
            elif "/ProcTHOR-" in name or name.startswith("ProcTHOR-"):
                prefix = "ProcTHOR"
                filename = name if name.endswith(".scene_instance.json") else f"{name}.scene_instance.json"
            else:
                parser.error(f"Unsupported scene name: {name}")
            scene_path = f"ai2thor/ai2thor-hab/configs/scenes/{prefix}/{filename}"
            config = json.loads(source.read(scene_path))
            paths.add(scene_path)
            paths.add(f"ai2thor/ai2thor-hab/assets/{config['stage_instance']['template_name']}.glb")
            for obj in config["object_instances"]:
                template = obj["template_name"]
                paths.add(f"ai2thor/ai2thorhab-uncompressed/assets/{template}.glb")
                paths.add(f"ai2thor/ai2thorhab-uncompressed/configs/{template}.object_config.json")
        missing = paths - available
        if missing:
            raise FileNotFoundError(f"Missing in AI2THOR archive: {sorted(missing)[:5]}")
        print(f"Selected {len(args.scenes)} scenes and {len(paths)} required files", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch, path, args.destination) for path in sorted(paths)]
        for index, future in enumerate(as_completed(futures), 1):
            future.result()
            if index % 50 == 0 or index == len(futures):
                print(f"Downloaded {index}/{len(futures)} files", flush=True)
    repaired = sum(repair_texcoords(path) for path in args.destination.rglob("*.glb"))
    print(f"Repaired {repaired} GLB texture-coordinate records", flush=True)
    manifest_path = args.destination / "subset_manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {"scenes": [], "files": []}
    manifest = {
        "source": URL,
        "scenes": sorted(set(previous["scenes"]) | set(args.scenes)),
        "files": sorted(set(previous["files"]) | paths),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
