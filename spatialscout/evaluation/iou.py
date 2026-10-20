"""Score translated render-geometry bounding boxes in a room benchmark batch.

Each prediction supplies a center, not a shape. The reconstructed box therefore
uses the submitted entity's original world-axis-aligned GT box, translated to
the submitted center. Equivalent objects are matched one-to-one per threshold
to maximize the number whose 3D box IoU passes that threshold.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


THRESHOLDS = (0.0, 0.5, 0.9)


def box_iou(first, second):
    first_low, first_high = first
    second_low, second_high = second
    intersection = np.prod(
        np.maximum(0.0, np.minimum(first_high, second_high) - np.maximum(first_low, second_low))
    )
    first_volume = np.prod(first_high - first_low)
    second_volume = np.prod(second_high - second_low)
    union = first_volume + second_volume - intersection
    # A zero-thickness render box has no 3D volume, so its volume IoU is scored as zero.
    return float(intersection / union) if union > 0 else 0.0


def score_scene(directory):
    gt_data = json.loads((directory / "ground_truth.json").read_text())
    pred_data = json.loads((directory / "predictions.json").read_text())
    gt = {item["id"]: item for item in gt_data["objects"]}
    predictions = {item["id"]: item for item in pred_data["predictions"]}
    origin = np.asarray(gt_data["origin_world_m"], dtype=float)
    rotation = np.asarray(gt_data["camera_to_world_rotation"], dtype=float)

    reconstructed_boxes = {}
    for entity_id, prediction in predictions.items():
        item = gt[entity_id]
        predicted_world_center = origin + rotation @ np.asarray(prediction["xyz_m"], dtype=float)
        translation = predicted_world_center - np.asarray(item["center_world_m"], dtype=float)
        reconstructed_boxes[entity_id] = (
            np.asarray(item["bbox_world_min_m"], dtype=float) + translation,
            np.asarray(item["bbox_world_max_m"], dtype=float) + translation,
        )

    hits = {threshold: 0 for threshold in THRESHOLDS}
    iou_sum = 0.0
    for group in {item["group"] for item in gt.values()}:
        pred_ids = [entity_id for entity_id in predictions if gt[entity_id]["group"] == group]
        gt_ids = [entity_id for entity_id, item in gt.items() if item["group"] == group]
        if not pred_ids:
            continue
        ious = np.asarray([
            [box_iou(reconstructed_boxes[pred_id], (
                np.asarray(gt[gt_id]["bbox_world_min_m"], dtype=float),
                np.asarray(gt[gt_id]["bbox_world_max_m"], dtype=float),
            )) for gt_id in gt_ids]
            for pred_id in pred_ids
        ])
        rows, columns = linear_sum_assignment(-ious)
        iou_sum += float(ious[rows, columns].sum())
        for threshold in THRESHOLDS:
            rewards = (ious > threshold).astype(float)
            rows, columns = linear_sum_assignment(-rewards - ious * 1e-8)
            hits[threshold] += int(rewards[rows, columns].sum())

    submitted = len(predictions)
    return {
        "scene": directory.name,
        "submitted": submitted,
        "gt_count": len(gt),
        "iou_sum": iou_sum,
        "mean_iou": iou_sum / submitted if submitted else 0.0,
        "hits": {str(threshold): hits[threshold] for threshold in THRESHOLDS},
        "rates": {str(threshold): hits[threshold] / submitted if submitted else 0.0 for threshold in THRESHOLDS},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_dirs", type=Path, nargs="+", help="One or more batch directories")
    parser.add_argument("--output", type=Path, help="Summary JSON path; required when combining batches")
    args = parser.parse_args()
    if len(args.batch_dirs) > 1 and args.output is None:
        parser.error("--output is required when combining batches")
    scenes = [score_scene(path) for batch_dir in args.batch_dirs for path in sorted(batch_dir.iterdir()) if path.is_dir()]
    scene_names = [scene["scene"] for scene in scenes]
    if len(scene_names) != len(set(scene_names)):
        parser.error("duplicate scene names across batch directories")
    total_submitted = sum(scene["submitted"] for scene in scenes)
    if total_submitted == 0:
        parser.error("no submitted objects found")
    summary = {
        "definition": "IoU of translated world-axis-aligned render-geometry bounding boxes; same-group one-to-one matching maximizes total IoU for mean IoU and threshold hits for each threshold; missing GT objects excluded from denominator",
        "threshold_comparison": "strictly greater than",
        "batch_dirs": [str(path) for path in args.batch_dirs],
        "scenes": scenes,
        "total_submitted": total_submitted,
        "total_gt": sum(scene["gt_count"] for scene in scenes),
        "micro": {
            "mean_iou": sum(scene["iou_sum"] for scene in scenes) / total_submitted,
            **{
                str(threshold): sum(scene["hits"][str(threshold)] for scene in scenes) / total_submitted
                for threshold in THRESHOLDS
            },
        },
        "macro": {
            "mean_iou": float(np.mean([scene["mean_iou"] for scene in scenes])),
            **{
                str(threshold): float(np.mean([scene["rates"][str(threshold)] for scene in scenes]))
                for threshold in THRESHOLDS
            },
        },
    }
    output = args.output or args.batch_dirs[0] / "iou_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
