"""SpatialScout object-localization benchmark for a camera-only VLM agent."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import sys
import time
from pathlib import Path

from spatialscout.paths import CATALOG_PATH
from spatialscout.agent.harness import (
    ROOT,
    TOOLS as CAMERA_TOOLS,
    CameraWorld,
    anthropic_completion,
    anthropic_image_message,
    anthropic_messages_url,
    chat_url,
    completion,
    finite_number,
    gemini_completion,
    gemini_generate_content_url,
    gemini_image_message,
    image_message,
    responses_image_message,
    responses_completion,
    responses_url,
    resolve_model,
)
import numpy as np
import sapien
import sapien.render
import transforms3d
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment
from mani_skill.utils.scene_builder.ai2thor.constants import SCENE_SOURCE_TO_DATASET
from mani_skill.utils.scene_builder.ai2thor.variants import (
    ArchitecTHORSceneBuilder,
    iTHORSceneBuilder,
    RoboTHORSceneBuilder,
    ProcTHORSceneBuilder,
)
from spatialscout.agent.token_usage import TokenUsageRecorder


SCENE_BUILDERS = {
    "iTHOR": iTHORSceneBuilder,
    "ArchitecTHOR": ArchitecTHORSceneBuilder,
    "RoboTHOR": RoboTHORSceneBuilder,
    "ProcTHOR": ProcTHORSceneBuilder,
}


ROOM_TOOLS = copy.deepcopy(CAMERA_TOOLS) + [
    {
        "type": "function",
        "function": {
            "name": "list_entities",
            "description": "List every target entity ID and category in this room. No locations are revealed.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "triangulate_observations",
            "description": "Estimate one entity's absolute 3D center from the same image point marked in two or more captured frames taken from different camera positions. Returns the least-squares ray intersection and ray mismatch; no ground truth is used.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "observations": {
                        "type": "array", "minItems": 2,
                        "items": {
                            "type": "object",
                            "properties": {
                                "frame": {"type": "string", "description": "Captured filename, such as frame_0001.png"},
                                "pixel_xy": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                            },
                            "required": ["frame", "pixel_xy"], "additionalProperties": False,
                        },
                    },
                },
                "required": ["id", "observations"], "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_positions",
            "description": "Submit or update a batch of estimated geometric-center coordinates in the original-camera frame. Use exact entity IDs from the library. You may call this repeatedly.",
            "parameters": {
                "type": "object",
                "properties": {
                    "predictions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "xyz_m": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                            },
                            "required": ["id", "xyz_m"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["predictions"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish_mapping",
            "description": "Finish exploration and score all submitted positions. Missing IDs count as missing predictions.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
]
for tool in ROOM_TOOLS:
    fn = tool["function"]
    if fn["name"] == "set_camera_position":
        fn["description"] = "Teleport the camera to absolute x,y,z coordinates in the ORIGINAL camera frame, in meters."
    elif fn["name"] == "get_camera_pose":
        fn["description"] = "Read camera position and full rotation matrix in the ORIGINAL camera frame."
    elif fn["name"] == "move_camera":
        fn["description"] = "Move relative to the camera's current heading: forward, right, up in meters, up to 1 meter per call."


def gemini_tool_args(name: str, args):
    """Discard explanatory `reason` fields that are not declared tool arguments."""
    declaration = next((tool["function"] for tool in ROOM_TOOLS if tool["function"]["name"] == name), None)
    if declaration is None:
        return args

    def clean(value, schema):
        if isinstance(value, dict):
            properties = schema.get("properties", {})
            return {
                key: clean(item, properties.get(key, {}))
                for key, item in value.items()
                if key != "reason" or key in properties
            }
        if isinstance(value, list):
            return [clean(item, schema.get("items", {})) for item in value]
        return value

    return clean(args, declaration["parameters"])


def render_bodies(item):
    if hasattr(item, "links"):
        entities = [link._objs[0].entity for link in item.links]
    else:
        entities = [item._objs[0]]
    for entity in entities:
        body = entity.find_component_by_type(sapien.render.RenderBodyComponent)
        if body is not None:
            yield body


def task_camera_basis(pose):
    """SAPIEN camera pose axes are forward (+x), left (+y), up (+z)."""
    return transforms3d.quaternions.quat2mat(pose.q)


def equivalence_group(category: str) -> str:
    """Ignore asset variant numbers when instances look like the same type."""
    return re.sub(r"_\d+$", "", category)


class RoomBenchmark:
    def __init__(self, scene_index: int, size: int, seed: int, output_dir: Path, target_radius_m: float | None = None,
                 target_count: int | None = None, scene_builder: str = "ReplicaCAD",
                 camera_start: list[float] | None = None):
        self.output_dir = output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scene_index = scene_index
        self.target_radius_m = target_radius_m
        self.target_count = target_count
        self.scene_builder_name = scene_builder
        env_id = "ReplicaCAD_SceneManipulation-v1" if scene_builder == "ReplicaCAD" else "SceneManipulation-v1"
        env_kwargs = {"build_config_idxs": scene_index}
        if scene_builder != "ReplicaCAD":
            env_kwargs["scene_builder_cls"] = SCENE_BUILDERS[scene_builder]
        self.base = CameraWorld(
            env_id, size, seed, self.output_dir,
            env_kwargs=env_kwargs, position_bound=15,
        )
        # The camera is free floating; the Fetch robot is not part of the task.
        for body in render_bodies(self.base.env.unwrapped.agent.robot):
            body.visibility = 0
        if camera_start is not None:
            self.base.position = np.asarray(camera_start[:3], dtype=float)
            self.base.yaw, self.base.pitch = camera_start[3:]
            self.base._apply_pose()
        self.origin = self.base.position.copy()
        initial_pose = self.base.camera._render_cameras[0].get_local_pose()
        self.camera_to_world = task_camera_basis(initial_pose)
        extrinsic_cv = self.base.camera.get_extrinsic_matrix()[0].cpu().numpy()
        projected_basis = np.column_stack((extrinsic_cv[2, :3], -extrinsic_cv[0, :3], -extrinsic_cv[1, :3]))
        if not np.allclose(self.camera_to_world, projected_basis, atol=1e-5):
            raise RuntimeError("Camera pose axes disagree with camera projection matrix")
        self.initial_yaw = self.base.yaw
        self.initial_pitch = self.base.pitch
        self.builder = self.base.env.unwrapped.scene_builder
        build_config = self.builder.build_configs[scene_index]
        if scene_builder == "ReplicaCAD":
            self.source_config = build_config
            self.source_file = ROOT / "assets" / "scene_datasets" / "replica_cad_dataset" / "configs" / "scenes" / build_config
        else:
            self.source_config = f"{build_config.source}/{Path(build_config.config_file).name}"
            self.source_file = Path(SCENE_SOURCE_TO_DATASET[build_config.source].dataset_path) / build_config.config_file
        self.items = {}
        self.ground_truth = {}
        if scene_builder != "ReplicaCAD":
            mapping_path = ROOT / "assets" / "scene_datasets" / "ai2thor" / "ai2thor-hab" / "configs" / "object_semantic_id_mapping.json"
            semantic_mapping = json.loads(mapping_path.read_text())
            self.semantic_labels = {value: key for key, value in semantic_mapping.items()}
        self.all_loaded_count = len(self.builder.scene_objects)
        candidates = []
        for key, item in self.builder.scene_objects.items():
            entity_id = key.removeprefix("env-0_")
            bounds = [np.asarray(body.compute_global_aabb_tight(), dtype=float) for body in render_bodies(item)]
            if not bounds:
                raise RuntimeError(f"No render geometry for {entity_id}")
            low = np.min([b[0] for b in bounds], axis=0)
            high = np.max([b[1] for b in bounds], axis=0)
            center_world = (low + high) / 2
            distance = float(np.linalg.norm(center_world - self.origin))
            label = entity_id.rsplit("-", 1)[0].split("/")[-1].removeprefix("frl_apartment_")
            semantic_known = True
            if self.scene_builder_name != "ReplicaCAD":
                template_name = entity_id.rsplit("_", 1)[0]
                config_path = ROOT / "assets" / "scene_datasets" / "ai2thor" / "ai2thorhab-uncompressed" / "configs" / f"{template_name}.object_config.json"
                semantic_id = json.loads(config_path.read_text())["semantic_id"]
                label = self.semantic_labels.get(semantic_id, "Undefined")
                semantic_known = label != "Undefined"
                if not semantic_known:
                    label = re.sub(r"_[0-9a-f]+$", "", template_name.split("/")[-1])
            candidates.append((distance, entity_id, item, low, high, center_world, label, semantic_known))
        eligible = [entry for entry in candidates if self.target_radius_m is None or entry[0] <= self.target_radius_m]
        eligible.sort(key=lambda entry: (not entry[7], entry[0], entry[1]))
        if self.target_count is not None:
            if len(eligible) < self.target_count:
                raise ValueError(f"{self.source_config} has only {len(eligible)} eligible objects; requested {self.target_count}")
            eligible = eligible[:self.target_count]
            if self.target_radius_m is None:
                self.target_radius_m = max(entry[0] for entry in eligible)
        chosen = {entry[1] for entry in eligible}
        for distance, entity_id, item, low, high, center_world, label, semantic_known in candidates:
            if entity_id not in chosen:
                for body in render_bodies(item):
                    body.visibility = 0
                continue
            center_task = self.to_task(center_world)
            self.items[entity_id] = item
            self.ground_truth[entity_id] = {
                "id": entity_id,
                "category": label,
                "group": equivalence_group(label),
                "xyz_m": center_task.tolist(),
                "center_world_m": center_world.tolist(),
                "bbox_world_min_m": low.tolist(),
                "bbox_world_max_m": high.tolist(),
            }
        self.predictions = {}
        self.finished = False
        self.observation_frames = {}

    def close(self):
        self.base.close()

    def to_task(self, xyz_world):
        return self.camera_to_world.T @ (np.asarray(xyz_world, dtype=float) - self.origin)

    def to_world(self, xyz_task):
        return self.origin + self.camera_to_world @ np.asarray(xyz_task, dtype=float)

    def camera_pose(self):
        raw = self.base.camera._render_cameras[0].get_local_pose()
        rotation = task_camera_basis(raw)
        return {
            "xyz_m": self.to_task(raw.p).round(6).tolist(),
            "camera_to_task_rotation": (self.camera_to_world.T @ rotation).round(6).tolist(),
            "axes": "+x initial camera forward, +y initial camera left, +z initial camera up",
            "units": "meters",
        }

    def library(self):
        return [{"id": x["id"], "category": x["category"], "group": x["group"]} for x in self.ground_truth.values()]

    def save_metadata(self):
        source = self.source_file
        if not source.exists():
            raise FileNotFoundError(source)
        source_data = json.loads(source.read_text())
        expected = len(source_data["object_instances"])
        if self.scene_builder_name == "ReplicaCAD":
            expected += sum("door" not in x["template_name"] for x in source_data["articulated_object_instances"])
        if expected != self.all_loaded_count:
            raise RuntimeError(f"Scene config has {expected} loaded objects but scene builder has {self.all_loaded_count}")
        (self.output_dir / "entity_library.json").write_text(json.dumps(self.library(), ensure_ascii=False, indent=2))
        manifest = {
            "scene": self.source_config,
            "scene_index": self.scene_index,
            "scene_builder": self.scene_builder_name,
            "target_radius_m": self.target_radius_m,
            "target_count": self.target_count,
            "loaded_object_count": self.all_loaded_count,
            "source_file": str(source),
            "origin_world_m": self.origin.tolist(),
            "initial_camera_yaw_pitch_deg": [self.initial_yaw, self.initial_pitch],
            "camera_to_world_rotation": self.camera_to_world.tolist(),
            "coordinate_frame": "+x initial camera forward, +y initial camera left, +z initial camera up",
            "center_definition": "center of the tight world-axis-aligned render-geometry bounding box, transformed to the initial camera frame",
            "excluded": "architectural stage, Fetch robot, lights, and objects outside the selected target subset",
            "objects": list(self.ground_truth.values()),
        }
        (self.output_dir / "ground_truth.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        return manifest

    def submit(self, entries):
        if not isinstance(entries, list):
            raise ValueError("predictions must be a list")
        batch = {}
        rejected_ids = []
        rejected_outside_radius = []
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {"id", "xyz_m"}:
                raise ValueError("Each prediction needs exactly id and xyz_m")
            entity_id = entry["id"]
            if entity_id not in self.items:
                rejected_ids.append(entity_id)
                continue
            xyz = entry["xyz_m"]
            if not isinstance(xyz, list) or len(xyz) != 3:
                raise ValueError(f"xyz_m for {entity_id} must have three numbers")
            coordinates = [finite_number(value, f"{entity_id} coordinate") for value in xyz]
            if self.target_radius_m is not None and np.linalg.norm(coordinates) > self.target_radius_m + 0.5:
                rejected_outside_radius.append(entity_id)
                continue
            batch[entity_id] = coordinates
        self.predictions.update(batch)
        self.save_predictions()
        return {"accepted": len(batch), "rejected_unknown_ids": rejected_ids, "rejected_outside_radius": rejected_outside_radius, "target_radius_m": self.target_radius_m, "submitted_total": len(self.predictions), "entity_total": len(self.items)}

    def triangulate(self, entity_id, observations):
        if entity_id not in self.items:
            raise ValueError(f"Unknown entity ID: {entity_id}")
        if not isinstance(observations, list) or len(observations) < 2:
            raise ValueError("At least two observations are required")
        rays = []
        origins = []
        for observation in observations:
            if not isinstance(observation, dict) or set(observation) != {"frame", "pixel_xy"}:
                raise ValueError("Each observation needs frame and pixel_xy")
            frame = self.observation_frames.get(observation["frame"])
            if frame is None:
                raise ValueError(f"Unknown frame: {observation['frame']}")
            pixel = observation["pixel_xy"]
            if not isinstance(pixel, list) or len(pixel) != 2:
                raise ValueError("pixel_xy needs two numbers")
            u, v = [finite_number(value, "pixel coordinate") for value in pixel]
            k = frame["K"]
            if not (0 <= u < frame["width"] and 0 <= v < frame["height"]):
                raise ValueError("Pixel is outside its captured frame")
            ray_camera = np.array([1.0, -(u - k[0][2]) / k[0][0], -(v - k[1][2]) / k[1][1]])
            direction = np.asarray(frame["rotation"]) @ ray_camera
            direction /= np.linalg.norm(direction)
            origins.append(np.asarray(frame["origin"]))
            rays.append(direction)
        if max(np.linalg.norm(a - b) for a in origins for b in origins) < 0.05:
            raise ValueError("Camera positions need at least 5 cm of separation for triangulation")
        identity = np.eye(3)
        projections = [identity - np.outer(ray, ray) for ray in rays]
        matrix = sum(projections)
        if np.linalg.cond(matrix) > 1e6:
            raise ValueError("Observation rays are nearly parallel")
        point = np.linalg.solve(matrix, sum(p @ origin for p, origin in zip(projections, origins)))
        forward_distances = [float(ray @ (point - origin)) for ray, origin in zip(rays, origins)]
        if any(distance <= 0 for distance in forward_distances):
            raise ValueError("Triangulated point is behind at least one camera")
        distances = [float(np.linalg.norm(p @ (point - origin))) for p, origin in zip(projections, origins)]
        pair_angles = [math.degrees(math.acos(float(np.clip(ray_a @ ray_b, -1, 1)))) for i, ray_a in enumerate(rays) for ray_b in rays[i + 1:]]
        max_parallax = max(pair_angles)
        ray_rms = float(np.sqrt(np.mean(np.square(distances))))
        return {
            "id": entity_id, "xyz_m": point.round(4).tolist(),
            "ray_rms_m": round(ray_rms, 4),
            "ray_distances_m": [round(x, 4) for x in distances],
            "max_parallax_deg": round(max_parallax, 3),
            "caution": "Estimated point is outside the target region" if self.target_radius_m is not None and np.linalg.norm(point) > self.target_radius_m + 0.5 else ("Weak parallax or inconsistent pixel markers" if max_parallax < 3 or ray_rms > 0.1 else None),
            "observation_count": len(observations),
        }

    def call(self, name: str, args: dict):
        if name == "list_entities":
            if args:
                raise ValueError("list_entities takes no arguments")
            return {"entities": self.library(), "count": len(self.items), "target_radius_m": self.target_radius_m}, None
        if name == "submit_positions":
            if set(args) != {"predictions"}:
                raise ValueError("submit_positions requires predictions")
            return self.submit(args["predictions"]), None
        if name == "triangulate_observations":
            if set(args) != {"id", "observations"}:
                raise ValueError("triangulate_observations needs id and observations")
            return self.triangulate(args["id"], args["observations"]), None
        if name == "finish_mapping":
            if args:
                raise ValueError("finish_mapping takes no arguments")
            self.finished = True
            return {"finished": True, "submitted": len(self.predictions), "total": len(self.items)}, None
        if name == "get_camera_pose":
            if args:
                raise ValueError("get_camera_pose takes no arguments")
            return self.camera_pose(), None
        if name == "set_camera_position":
            if set(args) != {"x", "y", "z"}:
                raise ValueError("set_camera_position needs x, y, z")
            values = [finite_number(args[k], k) for k in ("x", "y", "z")]
            target_world = self.to_world(values)
            self.base.set_position(*target_world)
            return self.camera_pose(), None
        result, path = self.base.call(name, args)
        if name in ("move_camera", "rotate_camera"):
            return self.camera_pose(), None
        if name == "capture_image":
            result["pose"] = self.camera_pose()
            pose = result["pose"]
            intrinsics = self.base.intrinsics()
            self.observation_frames[path.name] = {
                "origin": pose["xyz_m"], "rotation": pose["camera_to_task_rotation"],
                "K": intrinsics["K"], "width": result["width"], "height": result["height"],
            }
        return result, path

    def save_predictions(self):
        data = {
            "scene": self.source_config,
            "coordinate_frame": "+x initial camera forward, +y initial camera left, +z initial camera up",
            "predictions": [
                {"id": entity_id, "category": self.ground_truth[entity_id]["category"], "xyz_m": xyz}
                for entity_id, xyz in self.predictions.items()
            ],
        }
        path = self.output_dir / "predictions.json"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        return path

    def load_predictions(self, path: Path):
        data = json.loads(path.read_text())
        if data.get("scene") not in (None, self.source_config):
            raise ValueError(f"Predictions use scene {data['scene']}; expected {self.source_config}")
        self.submit([{"id": item["id"], "xyz_m": item["xyz_m"]} for item in data["predictions"]])

    def score(self, missing_penalty_m: float):
        matches = self.match_predictions()
        errors = {gt_id: error for gt_id, pred_id, error in matches}
        values = list(errors.values())
        n = len(self.items)
        scored = {
            "scene": self.source_config,
            "entity_count": n,
            "predicted_count": len(values),
            "coverage": len(values) / n,
            "mean_euclidean_error_m": float(np.mean(values)) if values else None,
            "median_euclidean_error_m": float(np.median(values)) if values else None,
            "rmse_m": float(np.sqrt(np.mean(np.square(values)))) if values else None,
            "missing_count": n - len(values),
            "missing_penalty_m": missing_penalty_m,
            "full_scene_loss_m": (sum(values) + missing_penalty_m * (n - len(values))) / n,
            "per_entity_error_m": errors,
            "matched_pairs": [{"gt_id": gt_id, "predicted_id": pred_id, "error_m": error} for gt_id, pred_id, error in matches],
        }
        usage_path = self.output_dir / "token_usage.json"
        if usage_path.is_file():
            scored["token_usage"] = {key: value for key, value in json.loads(usage_path.read_text()).items() if key != "turns"}
        (self.output_dir / "metrics.json").write_text(json.dumps(scored, ensure_ascii=False, indent=2))
        return scored

    def match_predictions(self):
        """Match duplicate and visually equivalent instances by minimum total distance."""
        groups = sorted({gt["group"] for gt in self.ground_truth.values()})
        matches = []
        for group in groups:
            gt_ids = [key for key, value in self.ground_truth.items() if value["group"] == group]
            pred_ids = [key for key in self.predictions if self.ground_truth[key]["group"] == group]
            if not pred_ids:
                continue
            cost = np.array([[np.linalg.norm(np.asarray(self.ground_truth[gt_id]["xyz_m"]) - np.asarray(self.predictions[pred_id])) for gt_id in gt_ids] for pred_id in pred_ids])
            rows, columns = linear_sum_assignment(cost)
            matches.extend((gt_ids[j], pred_ids[i], float(cost[i, j])) for i, j in zip(rows, columns))
        return matches

    def _set_reference_view(self, offset_yaw: float):
        self.base.position = self.origin.copy()
        self.base.yaw = self.initial_yaw + offset_yaw
        self.base.pitch = self.initial_pitch
        self.base._apply_pose()

    def _render(self, path: Path):
        image = self.base.env.unwrapped.render_rgb_array(camera_name=self.base.camera_name)
        rgb = image[0].detach().cpu().numpy().astype(np.uint8)
        Image.fromarray(rgb).save(path)

    def render_comparison(self):
        offsets = (0, -55, 55)
        gt_paths = []
        pred_paths = []
        for i, offset in enumerate(offsets):
            self._set_reference_view(offset)
            path = self.output_dir / f"gt_view_{i}.png"
            self._render(path)
            gt_paths.append(path)

        matched_positions = {gt_id: self.predictions[pred_id] for gt_id, pred_id, _ in self.match_predictions()}
        for entity_id, item in self.items.items():
            if entity_id not in matched_positions:
                for body in render_bodies(item):
                    body.visibility = 0
                continue
            target_world = self.to_world(matched_positions[entity_id])
            center_world = np.asarray(self.ground_truth[entity_id]["center_world_m"])
            delta = target_world - center_world
            pose = item.pose.sp
            item.set_pose(sapien.Pose(p=np.asarray(pose.p) + delta, q=pose.q))
        for i, offset in enumerate(offsets):
            self._set_reference_view(offset)
            path = self.output_dir / f"reconstruction_view_{i}.png"
            self._render(path)
            pred_paths.append(path)

        width = self.base.camera.get_width()
        height = self.base.camera.get_height()
        sheet = Image.new("RGB", (2 * width, 3 * (height + 30)), "white")
        draw = ImageDraw.Draw(sheet)
        for i, (gt_path, pred_path) in enumerate(zip(gt_paths, pred_paths)):
            y = i * (height + 30)
            draw.text((8, y + 7), f"GT view {i}", fill="black")
            draw.text((width + 8, y + 7), f"Agent reconstruction view {i}", fill="black")
            sheet.paste(Image.open(gt_path), (0, y + 30))
            sheet.paste(Image.open(pred_path), (width, y + 30))
        comparison = self.output_dir / "comparison.png"
        sheet.save(comparison)
        return comparison


def mapping_prompt():
    return (
        "Map every target object instance in this simulated scene. First call list_entities to receive the complete entity library. Estimate the geometric center "
        "of each object's rendered geometry in meters, in the ORIGINAL camera frame: origin at the "
        "camera's initial position, +x its initial optical forward, +y its initial left, +z its initial up. "
        "Camera pose tools return transforms in this same frame. Use multiple images and camera poses. "
        "For objects seen from two different camera positions, mark the same visual center pixel in each frame and use triangulate_observations before submitting its returned xyz_m. "
        "Only submit a position when you have visual evidence; do not invent locations for occluded items. "
        "The library includes equivalence groups. IDs within the same group are interchangeable for scoring, so submit one distinct valid ID per observed instance without worrying about their order. "
        "Start submitting visible objects after the first two views, "
        "then update and add estimates as you explore. Submit estimates in batches using submit_positions, then "
        "call finish_mapping. The architectural stage mesh and robot are outside the target library; some furniture and fixtures are separate target instances. "
        "Do not assume every visible object belongs to the library."
        " This benchmark keeps only a selected subset of targets visible; map only IDs returned by list_entities."
    )


def unfinished_mapping_prompt(room: RoomBenchmark) -> str:
    return (
        f"The mapping task is still active. You have submitted {len(room.predictions)} of {len(room.items)} target positions. "
        "Your last response did not call a tool. Continue with a tool call; if you are done exploring, "
        "submit any well-supported positions and call finish_mapping."
    )


def record_idle_stop(room: RoomBenchmark, turn: int) -> None:
    error = {
        "type": "agent_stopped_without_finish",
        "turn": turn,
        "error": "The model returned three consecutive responses without a tool call or finish_mapping",
        "submitted_count": len(room.predictions),
    }
    (room.output_dir / "run_error.json").write_text(json.dumps(error, ensure_ascii=False, indent=2))
    print(f"Agent stopped without finish_mapping after three tool-free responses (turn {turn}).", file=sys.stderr)


def run_agent(room: RoomBenchmark, url: str, key: str, model: str, max_turns: int, timeout: int):
    prompt = mapping_prompt()
    messages = [
        {"role": "system", "content": "You are a visual mapping agent. Use the camera tools and the supplied entity library. Never assume that camera intrinsics alone reveal depth; use views from different positions and camera pose transforms. Return absolute 3D positions in the initial-camera frame."},
        {"role": "user", "content": prompt},
    ]
    _, initial_picture = room.call("capture_image", {})
    messages.append(image_message(initial_picture))
    (room.output_dir / "run_info.json").write_text(json.dumps({
        "model": model,
        "scene": room.source_config,
        "scene_builder": room.scene_builder_name,
        "target_count": room.target_count,
        "target_radius_m": room.target_radius_m,
        "max_turns": max_turns,
        "initial_image": str(initial_picture),
    }, ensure_ascii=False, indent=2))
    usage_recorder = TokenUsageRecorder(room.output_dir, "openai_chat")
    idle_turns = 0
    trace = room.output_dir / "agent_trace.jsonl"
    with trace.open("w", encoding="utf-8") as log:
        for turn in range(max_turns):
            request_started = time.perf_counter()
            try:
                message, raw_usage = completion(url, key, model, messages, ROOM_TOOLS, timeout, include_usage=True)
            except RuntimeError as exc:
                error = {"turn": turn, "error": str(exc)}
                log.write(json.dumps(error, ensure_ascii=False) + "\n")
                log.flush()
                (room.output_dir / "run_error.json").write_text(json.dumps(error, ensure_ascii=False, indent=2))
                print(f"Agent stopped after API error: {exc}", file=sys.stderr)
                break
            request_seconds = time.perf_counter() - request_started
            usage = usage_recorder.record(turn, raw_usage)
            calls = message.get("tool_calls") or []
            log.write(json.dumps({"turn": turn, "api_seconds": round(request_seconds, 3), "usage": usage, "assistant": {"content": message.get("content"), "tool_calls": calls}}, ensure_ascii=False) + "\n")
            log.flush()
            if not calls:
                idle_turns += 1
                if idle_turns >= 3:
                    record_idle_stop(room, turn)
                    break
                messages.append({"role": "assistant", "content": message.get("content") or ""})
                messages.append({"role": "user", "content": unfinished_mapping_prompt(room)})
                continue
            idle_turns = 0
            messages.append({"role": "assistant", "content": message.get("content"), "tool_calls": [
                {"id": call["id"], "type": "function", "function": call["function"]} for call in calls
            ]})
            pictures = []
            for call in calls:
                name = call["function"]["name"]
                tool_started = time.perf_counter()
                try:
                    args = json.loads(call["function"].get("arguments") or "{}")
                    result, picture = room.call(name, args)
                    if picture:
                        pictures.append(picture)
                    data = result
                except (ValueError, KeyError, TypeError, RuntimeError) as exc:
                    data = {"error": str(exc)}
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(data, ensure_ascii=False)})
                tool_seconds = time.perf_counter() - tool_started
                log.write(json.dumps({"tool": name, "tool_seconds": round(tool_seconds, 3), "result": data}, ensure_ascii=False) + "\n")
                print(f"{name} ({tool_seconds:.2f}s, API {request_seconds:.1f}s): {str(data)[:300]}", file=sys.stderr)
            for picture in pictures:
                messages.append(image_message(picture))
            log.flush()
            if room.finished:
                break
    return room.save_predictions()


def run_agent_responses(room: RoomBenchmark, url: str, key: str, model: str, max_turns: int, timeout: int, reasoning_effort: str):
    _, initial_picture = room.call("capture_image", {})
    input_items = [
        {"role": "developer", "content": "You are a visual mapping agent. Use the camera tools and supplied entity library. Return absolute 3D positions in the initial-camera frame. Triangulate from multiple views when possible; camera intrinsics alone do not provide depth."},
        {"role": "user", "content": mapping_prompt()},
        responses_image_message(initial_picture),
    ]
    (room.output_dir / "run_info.json").write_text(json.dumps({
        "model": model, "api_mode": "responses", "reasoning_effort": reasoning_effort,
        "scene": room.source_config, "scene_builder": room.scene_builder_name,
        "target_count": room.target_count, "target_radius_m": room.target_radius_m,
        "max_turns": max_turns,
        "initial_image": str(initial_picture),
    }, ensure_ascii=False, indent=2))
    usage_recorder = TokenUsageRecorder(room.output_dir, "openai")
    idle_turns = 0
    trace = room.output_dir / "agent_trace.jsonl"
    with trace.open("w", encoding="utf-8") as log:
        for turn in range(max_turns):
            request_started = time.perf_counter()
            try:
                response = responses_completion(
                    url, key, model, input_items, ROOM_TOOLS, timeout,
                    reasoning_effort=reasoning_effort,
                )
            except RuntimeError as exc:
                error = {"turn": turn, "error": str(exc)}
                log.write(json.dumps(error, ensure_ascii=False) + "\n")
                log.flush()
                (room.output_dir / "run_error.json").write_text(json.dumps(error, ensure_ascii=False, indent=2))
                print(f"Agent stopped after API error: {exc}", file=sys.stderr)
                break
            request_seconds = time.perf_counter() - request_started
            usage = usage_recorder.record(turn, response.get("usage"))
            response_id = response["id"]
            calls = [item for item in response["output"] if item.get("type") == "function_call"]
            content = "\n".join(
                part.get("text", "")
                for item in response["output"] if item.get("type") == "message"
                for part in item.get("content", []) if part.get("type") == "output_text"
            )
            log.write(json.dumps({"turn": turn, "api_seconds": round(request_seconds, 3), "usage": usage, "response_id": response_id, "content": content, "tool_calls": calls}, ensure_ascii=False) + "\n")
            log.flush()
            if not calls:
                idle_turns += 1
                if idle_turns >= 3:
                    record_idle_stop(room, turn)
                    break
                input_items.extend(response["output"])
                input_items.append({"role": "user", "content": unfinished_mapping_prompt(room)})
                continue
            idle_turns = 0
            input_items.extend(response["output"])
            pictures = []
            for call in calls:
                name = call["name"]
                tool_started = time.perf_counter()
                try:
                    args = json.loads(call.get("arguments") or "{}")
                    result, picture = room.call(name, args)
                    if picture:
                        pictures.append(picture)
                    data = result
                except (ValueError, KeyError, TypeError, RuntimeError) as exc:
                    data = {"error": str(exc)}
                input_items.append({"type": "function_call_output", "call_id": call["call_id"], "output": json.dumps(data, ensure_ascii=False)})
                tool_seconds = time.perf_counter() - tool_started
                log.write(json.dumps({"tool": name, "tool_seconds": round(tool_seconds, 3), "result": data}, ensure_ascii=False) + "\n")
                print(f"{name} ({tool_seconds:.2f}s, API {request_seconds:.1f}s): {str(data)[:300]}", file=sys.stderr)
            for picture in pictures:
                input_items.append(responses_image_message(picture))
            if turn == max_turns - 11:
                input_items.append({"role": "user", "content": "Ten model calls remain. Submit all well-supported positions now, then use the remaining calls for corrections. Missing targets count as missing predictions."})
            if turn == max_turns - 4:
                input_items.append({"role": "user", "content": "Three model calls remain. Submit any final verified positions and call finish_mapping."})
            log.flush()
            if room.finished:
                break
    return room.save_predictions()


def run_agent_anthropic(room: RoomBenchmark, url: str, key: str, model: str, max_turns: int, timeout: int, reasoning_effort: str):
    _, initial_picture = room.call("capture_image", {})
    system = "You are a visual mapping agent. Use the camera tools and supplied entity library. Return absolute 3D positions in the initial-camera frame. Triangulate from multiple views when possible; camera intrinsics alone do not provide depth."
    messages = [
        {"role": "user", "content": mapping_prompt()},
        anthropic_image_message(initial_picture),
    ]
    (room.output_dir / "run_info.json").write_text(json.dumps({
        "model": model, "api_mode": "anthropic", "reasoning_effort": reasoning_effort,
        "scene": room.source_config, "scene_builder": room.scene_builder_name,
        "target_count": room.target_count, "target_radius_m": room.target_radius_m,
        "max_turns": max_turns,
        "initial_image": str(initial_picture),
    }, ensure_ascii=False, indent=2))
    usage_recorder = TokenUsageRecorder(room.output_dir, "anthropic")
    idle_turns = 0
    trace = room.output_dir / "agent_trace.jsonl"
    with trace.open("w", encoding="utf-8") as log:
        for turn in range(max_turns):
            request_started = time.perf_counter()
            try:
                response = anthropic_completion(
                    url, key, model, messages, ROOM_TOOLS, timeout,
                    system=system, reasoning_effort=reasoning_effort,
                )
            except RuntimeError as exc:
                error = {"turn": turn, "error": str(exc)}
                log.write(json.dumps(error, ensure_ascii=False) + "\n")
                log.flush()
                (room.output_dir / "run_error.json").write_text(json.dumps(error, ensure_ascii=False, indent=2))
                print(f"Agent stopped after API error: {exc}", file=sys.stderr)
                break
            request_seconds = time.perf_counter() - request_started
            usage = usage_recorder.record(turn, response.get("usage"))
            calls = [block for block in response["content"] if block.get("type") == "tool_use"]
            content = "\n".join(block.get("text", "") for block in response["content"] if block.get("type") == "text")
            log.write(json.dumps({"turn": turn, "api_seconds": round(request_seconds, 3), "usage": usage, "response_id": response["id"], "content": content, "tool_calls": calls}, ensure_ascii=False) + "\n")
            log.flush()
            if not calls:
                idle_turns += 1
                if idle_turns >= 3:
                    record_idle_stop(room, turn)
                    break
                messages.append({"role": "assistant", "content": response["content"]})
                messages.append({"role": "user", "content": unfinished_mapping_prompt(room)})
                continue
            idle_turns = 0
            messages.append({"role": "assistant", "content": response["content"]})
            results = []
            pictures = []
            for call in calls:
                name = call["name"]
                tool_started = time.perf_counter()
                try:
                    result, picture = room.call(name, call.get("input") or {})
                    if picture:
                        pictures.append(picture)
                    data = result
                except (ValueError, KeyError, TypeError, RuntimeError) as exc:
                    data = {"error": str(exc)}
                results.append({"type": "tool_result", "tool_use_id": call["id"], "content": json.dumps(data, ensure_ascii=False), "is_error": "error" in data})
                tool_seconds = time.perf_counter() - tool_started
                log.write(json.dumps({"tool": name, "tool_seconds": round(tool_seconds, 3), "result": data}, ensure_ascii=False) + "\n")
                print(f"{name} ({tool_seconds:.2f}s, API {request_seconds:.1f}s): {str(data)[:300]}", file=sys.stderr)
            messages.append({"role": "user", "content": results})
            for picture in pictures:
                messages.append(anthropic_image_message(picture))
            if turn == max_turns - 11:
                messages.append({"role": "user", "content": "Ten model calls remain. Submit all well-supported positions now, then use the remaining calls for corrections. Missing targets count as missing predictions."})
            if turn == max_turns - 4:
                messages.append({"role": "user", "content": "Three model calls remain. Submit any final verified positions and call finish_mapping."})
            log.flush()
            if room.finished:
                break
    return room.save_predictions()


def run_agent_gemini(room: RoomBenchmark, url: str, key: str, model: str, max_turns: int, timeout: int, reasoning_effort: str):
    _, initial_picture = room.call("capture_image", {})
    system = "You are a visual mapping agent. Use the camera tools and supplied entity library. Return absolute 3D positions in the initial-camera frame. Triangulate from multiple views when possible; camera intrinsics alone do not provide depth."
    contents = [
        {"role": "user", "parts": [{"text": mapping_prompt()}]},
        gemini_image_message(initial_picture),
    ]
    (room.output_dir / "run_info.json").write_text(json.dumps({
        "model": model, "api_mode": "gemini", "reasoning_effort": reasoning_effort,
        "scene": room.source_config, "scene_builder": room.scene_builder_name,
        "target_count": room.target_count, "target_radius_m": room.target_radius_m,
        "max_turns": max_turns,
        "initial_image": str(initial_picture),
    }, ensure_ascii=False, indent=2))
    usage_recorder = TokenUsageRecorder(room.output_dir, "gemini")
    successful_tool_calls = 0
    failed_tool_calls = 0
    idle_turns = 0
    trace = room.output_dir / "agent_trace.jsonl"
    with trace.open("w", encoding="utf-8") as log:
        for turn in range(max_turns):
            request_started = time.perf_counter()
            try:
                response = gemini_completion(
                    url, key, contents, ROOM_TOOLS, timeout,
                    system=system, reasoning_effort=reasoning_effort,
                )
            except RuntimeError as exc:
                error = {"turn": turn, "error": str(exc)}
                log.write(json.dumps(error, ensure_ascii=False) + "\n")
                log.flush()
                (room.output_dir / "run_error.json").write_text(json.dumps(error, ensure_ascii=False, indent=2))
                print(f"Agent stopped after API error: {exc}", file=sys.stderr)
                break
            request_seconds = time.perf_counter() - request_started
            usage = usage_recorder.record(turn, response.get("usageMetadata"))
            model_content = response["candidates"][0]["content"]
            parts = model_content["parts"]
            calls = [part["functionCall"] for part in parts if isinstance(part, dict) and "functionCall" in part]
            content = "\n".join(part.get("text", "") for part in parts if isinstance(part, dict) and "text" in part and not part.get("thought"))
            log.write(json.dumps({
                "turn": turn, "api_seconds": round(request_seconds, 3), "usage": usage,
                "content": content, "tool_calls": calls,
            }, ensure_ascii=False) + "\n")
            log.flush()
            if not calls:
                idle_turns += 1
                if idle_turns >= 3:
                    record_idle_stop(room, turn)
                    break
                contents.append(model_content)
                contents.append({"role": "user", "parts": [{"text": unfinished_mapping_prompt(room)}]})
                continue
            idle_turns = 0
            # Return Google's original model content, including thoughtSignature and part order.
            contents.append(model_content)
            results = []
            pictures = []
            for call in calls:
                name = call["name"]
                raw_args = call.get("args") or {}
                args = gemini_tool_args(name, raw_args)
                tool_started = time.perf_counter()
                try:
                    result, picture = room.call(name, args)
                    if picture:
                        pictures.append(picture)
                    data = result
                    successful_tool_calls += 1
                except (ValueError, KeyError, TypeError, RuntimeError) as exc:
                    data = {"error": str(exc)}
                    failed_tool_calls += 1
                function_response = {"name": name, "response": data}
                if call.get("id"):
                    function_response["id"] = call["id"]
                results.append({"functionResponse": function_response})
                tool_seconds = time.perf_counter() - tool_started
                log.write(json.dumps({"tool": name, "tool_seconds": round(tool_seconds, 3), "ignored_undeclared_reason": args != raw_args, "result": data}, ensure_ascii=False) + "\n")
                print(f"{name} ({tool_seconds:.2f}s, API {request_seconds:.1f}s): {str(data)[:300]}", file=sys.stderr)
            contents.append({"role": "user", "parts": results})
            for picture in pictures:
                contents.append(gemini_image_message(picture))
            if turn == max_turns - 11:
                contents.append({"role": "user", "parts": [{"text": "Ten model calls remain. Submit all well-supported positions now, then use the remaining calls for corrections. Missing targets count as missing predictions."}]})
            if turn == max_turns - 4:
                contents.append({"role": "user", "parts": [{"text": "Three model calls remain. Submit any final verified positions and call finish_mapping."}]})
            log.flush()
            if room.finished:
                break
    if failed_tool_calls and not successful_tool_calls and not (room.output_dir / "run_error.json").exists():
        error = {"type": "tool_protocol_error", "error": "Every Gemini tool call failed", "failed_tool_calls": failed_tool_calls}
        (room.output_dir / "run_error.json").write_text(json.dumps(error, ensure_ascii=False, indent=2))
    return room.save_predictions()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "run", "evaluate"), default="prepare")
    parser.add_argument("--scene-builder", choices=("ReplicaCAD", *SCENE_BUILDERS), default="ReplicaCAD")
    parser.add_argument("--scene-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--catalog-scene", help="Name from scene_catalog.json; supplies builder, scene index, camera start and target count")
    parser.add_argument("--list-catalog", action="store_true", help="Print available scene names and exit")
    parser.add_argument("--target-radius-m", type=float, default=4.0, help="Keep only target objects within this distance of the initial camera; 0 keeps all")
    parser.add_argument("--target-count", type=int, help="Keep exactly this many nearest eligible target objects; use --target-radius-m 0 for unrestricted selection")
    parser.add_argument("--camera-start", type=float, nargs=5, metavar=("X", "Y", "Z", "YAW", "PITCH"), help="Initial camera world xyz and yaw/pitch in degrees")
    parser.add_argument("--predictions", type=Path, help="Predictions JSON for --mode evaluate")
    parser.add_argument("--base-url", default=os.getenv("VLM_BASE_URL"))
    parser.add_argument("--api-key", default=os.getenv("VLM_API_KEY"))
    parser.add_argument("--credentials-dir", type=Path, default=ROOT / "ManiSkill3" / "api", help="Directory with base-url and api-key files")
    parser.add_argument("--model", default=os.getenv("VLM_MODEL"))
    parser.add_argument("--api-backend", choices=("openai", "anthropic", "gemini"), default=os.getenv("VLM_API_BACKEND", "openai"))
    parser.add_argument("--api-mode", choices=("responses", "chat"), default="responses")
    parser.add_argument("--reasoning-effort", choices=("none", "low", "medium", "high", "xhigh", "max"), default="high")
    parser.add_argument("--max-turns", type=int, default=80)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--missing-penalty-m", type=float, default=5.0)
    args = parser.parse_args()
    catalog = json.loads(CATALOG_PATH.read_text())
    catalog_by_name = {entry["name"]: entry for entry in catalog["scenes"]}
    if args.list_catalog:
        for entry in catalog["scenes"]:
            print(f"{entry['name']}: {entry['type']} ({entry['builder']}, {entry['source_scene']})")
        return
    if args.catalog_scene:
        if args.catalog_scene not in catalog_by_name:
            parser.error(f"Unknown catalog scene: {args.catalog_scene}")
        selected = catalog_by_name[args.catalog_scene]
        args.scene_builder = selected["builder"]
        args.scene_index = selected["scene_index"]
        args.camera_start = selected["camera_start"]
        args.target_radius_m = 0.0
        args.target_count = selected.get("target_count", 30)
    if args.output_dir is None:
        output_name = args.catalog_scene or "apt_0_local"
        args.output_dir = ROOT / "outputs" / "room_benchmark" / output_name
    if args.credentials_dir.is_dir():
        if not args.base_url and (args.credentials_dir / "base-url").is_file():
            args.base_url = (args.credentials_dir / "base-url").read_text().strip()
        if not args.api_key and (args.credentials_dir / "api-key").is_file():
            args.api_key = (args.credentials_dir / "api-key").read_text().strip()
    if args.size < 64 or args.size > 1024 or args.max_turns < 1 or args.timeout < 1 or args.scene_index < 0 or (args.target_count is not None and args.target_count < 1) or not math.isfinite(args.missing_penalty_m) or args.missing_penalty_m < 0 or not math.isfinite(args.target_radius_m) or args.target_radius_m < 0:
        parser.error("Invalid size, max-turns, timeout, missing penalty, or target radius")
    if args.mode == "run" and not all((args.base_url, args.api_key)):
        parser.error("run mode needs --base-url and --api-key, or VLM_BASE_URL and VLM_API_KEY")
    if args.mode == "evaluate" and args.predictions is None:
        parser.error("evaluate mode needs --predictions")
    try:
        url = ""
        model = ""
        if args.mode == "run":
            model = args.model if args.api_backend in ("anthropic", "gemini") else resolve_model(args.base_url, args.api_key, args.model, args.timeout)
            if not model:
                parser.error("Anthropic and Gemini backends require --model or VLM_MODEL")
            if args.api_backend == "gemini":
                if args.reasoning_effort not in ("low", "medium", "high"):
                    parser.error("Gemini reasoning effort must be low, medium, or high")
                url = gemini_generate_content_url(args.base_url, model)
            elif args.api_backend == "anthropic":
                url = anthropic_messages_url(args.base_url)
            else:
                url = responses_url(args.base_url) if args.api_mode == "responses" else chat_url(args.base_url)
        room = RoomBenchmark(args.scene_index, args.size, args.seed, args.output_dir, target_radius_m=args.target_radius_m or None,
                             target_count=args.target_count, scene_builder=args.scene_builder,
                             camera_start=args.camera_start)
        try:
            manifest = room.save_metadata()
            print(f"Prepared {len(manifest['objects'])} entities from {manifest['scene']} in {room.output_dir}")
            if args.mode == "prepare":
                room._set_reference_view(0)
                room._render(room.output_dir / "gt_view_0.png")
            else:
                if args.mode == "run":
                    if args.api_backend == "anthropic":
                        run_agent_anthropic(room, url, args.api_key, model, args.max_turns, args.timeout, args.reasoning_effort)
                    elif args.api_backend == "gemini":
                        run_agent_gemini(room, url, args.api_key, model, args.max_turns, args.timeout, args.reasoning_effort)
                    elif args.api_mode == "responses":
                        run_agent_responses(room, url, args.api_key, model, args.max_turns, args.timeout, args.reasoning_effort)
                    else:
                        run_agent(room, url, args.api_key, model, args.max_turns, args.timeout)
                else:
                    room.load_predictions(args.predictions)
                    room.save_predictions()
                scored = room.score(args.missing_penalty_m)
                comparison = room.render_comparison()
                print(json.dumps({k: v for k, v in scored.items() if k not in ("per_entity_error_m", "matched_pairs")}, indent=2))
                print(f"Comparison image: {comparison}")
        finally:
            room.close()
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
