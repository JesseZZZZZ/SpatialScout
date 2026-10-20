"""Find a visually informative first camera view for one catalog scene."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from spatialscout.paths import CATALOG_PATH
from spatialscout.agent.harness import ROOT, CameraWorld
from spatialscout.benchmark.room import SCENE_BUILDERS, render_bodies


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene")
    parser.add_argument("--grid", type=int, default=5)
    parser.add_argument("--size", type=int, default=128)
    args = parser.parse_args()
    entry = next(x for x in json.loads(CATALOG_PATH.read_text())["scenes"] if x["name"] == args.scene)
    if entry["builder"] == "ReplicaCAD":
        config = ROOT / "assets" / "scene_datasets" / "replica_cad_dataset" / "configs" / "scenes" / f"{entry['source_scene']}.scene_instance.json"
        env_id = "ReplicaCAD_SceneManipulation-v1"
        env_kwargs = {"build_config_idxs": entry["scene_index"]}
    else:
        config_file = entry.get("config_file", f"{entry['source_scene']}.scene_instance.json")
        config = ROOT / "assets" / "scene_datasets" / "ai2thor" / "ai2thor-hab" / "configs" / "scenes" / entry["builder"] / config_file
        env_id = "SceneManipulation-v1"
        env_kwargs = {"build_config_idxs": entry["scene_index"], "scene_builder_cls": SCENE_BUILDERS[entry["builder"]]}
    objects = json.loads(config.read_text())["object_instances"]
    centers = np.asarray([[item["translation"][0], -item["translation"][2]] for item in objects])
    fractions = np.linspace(0.15, 0.85, args.grid)
    xs, ys = [np.quantile(centers[:, i], fractions) for i in range(2)]
    output = ROOT / "outputs" / "camera_search" / args.scene
    output.mkdir(parents=True, exist_ok=True)
    camera = CameraWorld(
        env_id, args.size, 0, output,
        env_kwargs=env_kwargs,
        position_bound=15,
    )
    for body in render_bodies(camera.env.unwrapped.agent.robot):
        body.visibility = 0
    candidates = []
    try:
        for x in xs:
            for y in ys:
                camera.position = np.array([x, y, 1.45])
                for yaw in range(0, 360, 45):
                    camera.yaw, camera.pitch = yaw, -12
                    camera._apply_pose()
                    image = camera.env.unwrapped.render_rgb_array(camera_name=camera.camera_name)[0].detach().cpu().numpy().astype(np.uint8)
                    gray = image.astype(float).mean(axis=2)
                    black = float((image.max(axis=2) < 12).mean())
                    edge = float(np.abs(gray[:, 1:] - gray[:, :-1]).mean() + np.abs(gray[1:, :] - gray[:-1, :]).mean())
                    color_bins = image.astype(np.int32) // 32
                    palette = color_bins[:, :, 0] * 64 + color_bins[:, :, 1] * 8 + color_bins[:, :, 2]
                    counts = np.bincount(palette.ravel(), minlength=512)
                    probabilities = counts[counts > 0] / palette.size
                    entropy = float(-np.sum(probabilities * np.log2(probabilities)))
                    dominance = float(probabilities.max())
                    score = float(18 * entropy + 0.5 * edge - 100 * black - 30 * max(0, dominance - 0.5))
                    candidates.append((score, [round(float(x), 3), round(float(y), 3), 1.45, yaw, -12], black, image))
    finally:
        camera.close()
    candidates.sort(key=lambda item: item[0], reverse=True)
    for index, (score, pose, black, image) in enumerate(candidates[:5]):
        Image.fromarray(image).save(output / f"candidate_{index}.png")
        print(index, "score", round(score, 2), "black", round(black, 3), "pose", pose, flush=True)
    (output / "candidates.json").write_text(json.dumps([
        {"score": score, "camera_start": pose, "black_fraction": black}
        for score, pose, black, _ in candidates[:20]
    ], indent=2) + "\n")


if __name__ == "__main__":
    main()
