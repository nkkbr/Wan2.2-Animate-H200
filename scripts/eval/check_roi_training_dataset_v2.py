#!/usr/bin/env python
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan.utils.roi_dataset_schema import ROI_DATASET_SPLITS, ROI_SEMANTIC_TAGS, ROI_TASK_TYPES, validate_records


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())


def main():
    parser = argparse.ArgumentParser(description="Check ROI training dataset v2 correctness and governance.")
    parser.add_argument("--dataset_npz", required=True)
    parser.add_argument("--dataset_json", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    dataset = np.load(args.dataset_npz)
    info = _read_json(Path(args.dataset_json))
    records = info["records"]
    validate_records(records)

    errors = []
    sample_count = int(info["sample_count"])
    if sample_count <= 0:
        errors.append("Dataset contains no samples.")

    required_arrays = [
        "foreground_patch",
        "background_patch",
        "input_soft_alpha",
        "input_trimap_unknown",
        "input_boundary_roi_mask",
        "input_person_mask",
        "input_boundary_band",
        "input_uncertainty",
        "gt_soft_alpha",
        "gt_trimap_unknown",
        "gt_boundary_mask",
        "gt_occlusion_band",
        "gt_face_boundary",
        "gt_hair_edge",
        "gt_hand_boundary",
        "gt_cloth_boundary",
        "gt_semi_transparent",
        "semantic_target_mask",
        "difficulty_score",
    ]
    for key in required_arrays:
        if key not in dataset:
            errors.append(f"Missing array: {key}")
        elif len(dataset[key]) != sample_count:
            errors.append(f"Array {key} has length {len(dataset[key])}, expected {sample_count}.")

    patch_shape = tuple(dataset["foreground_patch"].shape[1:3]) if "foreground_patch" in dataset else None
    if patch_shape is not None and patch_shape[0] != patch_shape[1]:
        errors.append(f"Foreground patch is not square: {patch_shape}")

    task_counts = Counter(record["task_type"] for record in records)
    semantic_counts = Counter(record["semantic_boundary_tag"] for record in records)
    split_counts = Counter(record["dataset_split"] for record in records)
    split_task_counts = defaultdict(Counter)
    split_semantic_counts = defaultdict(Counter)
    for record in records:
        split_task_counts[record["dataset_split"]][record["task_type"]] += 1
        split_semantic_counts[record["dataset_split"]][record["semantic_boundary_tag"]] += 1

    for task_type in ROI_TASK_TYPES:
        if task_counts[task_type] == 0:
            errors.append(f"Task {task_type} has zero samples.")
    for split in ("train", "val"):
        for task_type in ROI_TASK_TYPES:
            if split_task_counts[split][task_type] == 0:
                errors.append(f"Split {split} is missing task {task_type}.")

    for split in ("train", "val"):
        for semantic_tag in ("face", "hair", "hand", "cloth", "occluded", "semi_transparent"):
            if split_semantic_counts[split][semantic_tag] == 0:
                errors.append(f"Split {split} is missing semantic tag {semantic_tag}.")

    hard_negative_ratio = {
        split: (
            sum(1 for record in records if record["dataset_split"] == split and record["is_hard_negative"])
            / max(split_counts[split], 1)
        )
        for split in ROI_DATASET_SPLITS
        if split_counts[split] > 0
    }
    if hard_negative_ratio.get("train", 0.0) < 0.12:
        errors.append(f"Train hard_negative_ratio too low: {hard_negative_ratio.get('train', 0.0):.4f}")

    # Leakage check: no near-neighbor frame overlap (gap <= 2) across splits for same case.
    frames_by_case_split = defaultdict(lambda: defaultdict(set))
    for record in records:
        frames_by_case_split[record["case_id"]][record["dataset_split"]].add(int(record["preprocess_frame_index"]))
    leakage_pairs = []
    split_order = ("train", "val", "test")
    for case_id, by_split in frames_by_case_split.items():
        for i in range(len(split_order)):
            for j in range(i + 1, len(split_order)):
                left = split_order[i]
                right = split_order[j]
                for frame_l in by_split[left]:
                    for frame_r in by_split[right]:
                        if abs(frame_l - frame_r) <= 2:
                            leakage_pairs.append({
                                "case_id": case_id,
                                "left_split": left,
                                "right_split": right,
                                "frame_left": frame_l,
                                "frame_right": frame_r,
                            })
    if leakage_pairs:
        errors.append(f"Temporal leakage pairs found: {len(leakage_pairs)}")

    # Sanity check that foreground/background are not identical everywhere.
    fg_bg_mean_abs = _mean_abs_diff(dataset["foreground_patch"], dataset["background_patch"])
    if fg_bg_mean_abs < 1.0:
        errors.append(f"Foreground/background patches are suspiciously similar: mean_abs_diff={fg_bg_mean_abs:.4f}")

    output = {
        "dataset_npz": str(Path(args.dataset_npz).resolve()),
        "dataset_json": str(Path(args.dataset_json).resolve()),
        "sample_count": sample_count,
        "patch_shape": list(patch_shape) if patch_shape is not None else None,
        "task_counts": dict(task_counts),
        "semantic_counts": dict(semantic_counts),
        "split_counts": dict(split_counts),
        "split_task_counts": {k: dict(v) for k, v in split_task_counts.items()},
        "split_semantic_counts": {k: dict(v) for k, v in split_semantic_counts.items()},
        "hard_negative_ratio": hard_negative_ratio,
        "fg_bg_mean_abs_diff": fg_bg_mean_abs,
        "leakage_pair_count": len(leakage_pairs),
        "leakage_pairs_preview": leakage_pairs[:20],
        "error_count": len(errors),
        "errors": errors,
        "schema_valid": len(errors) == 0,
    }

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "sample_count": sample_count,
        "split_counts": dict(split_counts),
        "hard_negative_ratio": hard_negative_ratio,
        "leakage_pair_count": len(leakage_pairs),
        "error_count": len(errors),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
