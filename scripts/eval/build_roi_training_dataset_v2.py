#!/usr/bin/env python
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan.utils.media_io import load_mask_artifact, load_person_mask_artifact, load_rgb_artifact
from wan.utils.roi_dataset_schema import ROI_SPLIT_POLICIES, validate_records


SEMANTIC_TARGET_KEYS = {
    "face": "gt_face_boundary",
    "hair": "gt_hair_edge",
    "hand": "gt_hand_boundary",
    "cloth": "gt_cloth_boundary",
    "occluded": "gt_occlusion_band",
    "semi_transparent": "gt_semi_transparent",
    "mixed_boundary": "gt_boundary_mask",
    "hard_negative": "gt_boundary_mask",
}


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _load_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _center_from_mask(mask: np.ndarray) -> tuple[int, int] | None:
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return None
    center_yx = coords.mean(axis=0)
    return int(round(center_yx[1])), int(round(center_yx[0]))


def _crop_with_padding(arr: np.ndarray, center_x: int, center_y: int, patch_size: int):
    half = patch_size // 2
    h, w = arr.shape[:2]
    x0 = center_x - half
    y0 = center_y - half
    x1 = x0 + patch_size
    y1 = y0 + patch_size
    pad_left = max(0, -x0)
    pad_top = max(0, -y0)
    pad_right = max(0, x1 - w)
    pad_bottom = max(0, y1 - h)
    x0_clip = max(0, x0)
    y0_clip = max(0, y0)
    x1_clip = min(w, x1)
    y1_clip = min(h, y1)
    cropped = arr[y0_clip:y1_clip, x0_clip:x1_clip]
    if arr.ndim == 2:
        padded = np.pad(cropped, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="edge")
    else:
        padded = np.pad(cropped, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode="edge")
    return padded[:patch_size, :patch_size], [int(x0_clip), int(y0_clip), int(x1_clip), int(y1_clip)]


def _temporal_quarantine_split(frame_idx: int) -> str | None:
    if frame_idx <= 19:
        return "train"
    if 23 <= frame_idx <= 32:
        return "val"
    if frame_idx >= 36:
        return "test"
    return None


def _derive_dataset_split(frame_idx: int, source_review_split: str, split_policy: str) -> str | None:
    if split_policy == "reviewed_split_v1":
        return {"seed_eval": "train", "expansion_eval": "val", "holdout_eval": "test"}[source_review_split]
    if split_policy == "temporal_quarantine_v1":
        return _temporal_quarantine_split(frame_idx)
    raise ValueError(f"Unsupported split_policy: {split_policy}")


def _input_trimap_from_alpha(pred_alpha: np.ndarray, pred_boundary: np.ndarray, pred_uncertainty: np.ndarray) -> np.ndarray:
    return np.logical_or(
        np.logical_and(pred_alpha > 0.08, pred_alpha < 0.92),
        np.logical_or(pred_boundary > 0.10, pred_uncertainty > 0.10),
    ).astype(np.uint8)


def _largest_component_center(mask: np.ndarray) -> tuple[int, int] | None:
    if int(mask.sum()) == 0:
        return None
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return _center_from_mask(mask)
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return _center_from_mask((labels == label).astype(np.uint8))


def _pick_hard_negative_center(hard_foreground: np.ndarray, boundary_mask: np.ndarray) -> tuple[int, int] | None:
    background_region = np.logical_and(hard_foreground == 0, boundary_mask == 0).astype(np.uint8)
    if int(background_region.sum()) == 0:
        return None
    distance = cv2.distanceTransform(background_region, cv2.DIST_L2, 5)
    y, x = np.unravel_index(int(np.argmax(distance)), distance.shape)
    return int(x), int(y)


def _semantic_masks(label_json: dict, label_root: Path) -> dict[str, np.ndarray]:
    return {
        "face": (_load_gray(label_root / label_json["annotations"]["face_boundary_mask"]["path"]) > 127).astype(np.uint8),
        "hair": (_load_gray(label_root / label_json["annotations"]["hair_edge_mask"]["path"]) > 127).astype(np.uint8),
        "hand": (_load_gray(label_root / label_json["annotations"]["hand_boundary_mask"]["path"]) > 127).astype(np.uint8),
        "cloth": (_load_gray(label_root / label_json["annotations"]["cloth_boundary_mask"]["path"]) > 127).astype(np.uint8),
        "occluded": (_load_gray(label_root / label_json["annotations"]["occlusion_band"]["path"]) > 127).astype(np.uint8),
        "semi_transparent": (_load_gray(label_root / label_json["annotations"]["semi_transparent_boundary_mask"]["path"]) > 127).astype(np.uint8),
    }


def main():
    parser = argparse.ArgumentParser(description="Build ROI training dataset v2 from reviewed edge benchmark v3.")
    parser.add_argument("--reviewed_dataset_dir", required=True)
    parser.add_argument("--prediction_preprocess_dir", required=True)
    parser.add_argument("--task_taxonomy", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--patch_size", type=int, default=192)
    parser.add_argument("--split_policy", choices=ROI_SPLIT_POLICIES, default="reviewed_split_v1")
    parser.add_argument("--hard_negative_per_frame", type=int, default=1)
    args = parser.parse_args()

    reviewed_dir = Path(args.reviewed_dataset_dir)
    preprocess_dir = Path(args.prediction_preprocess_dir)
    taxonomy = _read_json(Path(args.task_taxonomy))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = _read_json(reviewed_dir / "summary.json")
    metadata = _read_json(preprocess_dir / "metadata.json")

    def load_mask(name: str):
        artifact = metadata["src_files"][name]
        loader = load_person_mask_artifact if name == "person_mask" else load_mask_artifact
        return loader(preprocess_dir / artifact["path"], artifact.get("format")).astype(np.float32)

    rgb_frames = np.stack([cv2.cvtColor(cv2.imread(record["image_path"], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                           for record in summary["cases"][0]["records"]], axis=0)
    # Map by preprocess_frame_index for direct lookup.
    image_by_frame = {
        int(record["preprocess_frame_index"]): cv2.cvtColor(cv2.imread(record["image_path"], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        for record in summary["cases"][0]["records"]
    }
    del rgb_frames

    background_frames = load_rgb_artifact(
        preprocess_dir / metadata["src_files"]["background"]["path"],
        metadata["src_files"]["background"].get("format"),
    ).astype(np.uint8)
    input_person_mask = load_mask("person_mask")
    input_soft_alpha = load_mask("soft_alpha")
    input_boundary_band = load_mask("boundary_band")
    input_uncertainty = load_mask("uncertainty_map")

    arrays = defaultdict(list)
    records = []
    split_frames = defaultdict(list)

    sample_counter = 0
    for case in summary["cases"]:
        for record in case["records"]:
            label_json_path = Path(record["label_json_path"])
            label_json = _read_json(label_json_path)
            label_root = label_json_path.parent
            frame_idx = int(label_json["preprocess_frame_index"])
            source_split = label_json["review_metadata"]["split"]
            dataset_split = _derive_dataset_split(frame_idx, source_split, args.split_policy)
            if dataset_split is None:
                continue

            image_rgb = image_by_frame[frame_idx]
            gt_hard = (_load_gray(label_root / label_json["annotations"]["hard_foreground"]["path"]) > 127).astype(np.uint8)
            gt_alpha = _load_gray(label_root / label_json["annotations"]["soft_alpha"]["path"]).astype(np.float32) / 255.0
            gt_trimap_unknown = (_load_gray(label_root / label_json["annotations"]["trimap_unknown"]["path"]) > 127).astype(np.uint8)
            gt_boundary = (_load_gray(label_root / label_json["annotations"]["boundary_mask"]["path"]) > 127).astype(np.uint8)
            gt_occluded = (_load_gray(label_root / label_json["annotations"]["occlusion_band"]["path"]) > 127).astype(np.uint8)
            gt_semi = (_load_gray(label_root / label_json["annotations"]["semi_transparent_boundary_mask"]["path"]) > 127).astype(np.uint8)
            semantic_masks = _semantic_masks(label_json, label_root)

            pred_alpha = np.clip(input_soft_alpha[frame_idx], 0.0, 1.0).astype(np.float32)
            pred_boundary = np.clip(input_boundary_band[frame_idx], 0.0, 1.0).astype(np.float32)
            pred_unc = np.clip(input_uncertainty[frame_idx], 0.0, 1.0).astype(np.float32)
            pred_person = (input_person_mask[frame_idx] > 0.5).astype(np.uint8)
            pred_trimap = _input_trimap_from_alpha(pred_alpha, pred_boundary, pred_unc)
            background_rgb = background_frames[frame_idx]

            roi_specs = []
            mixed_center = _largest_component_center(np.logical_or(gt_boundary > 0, gt_trimap_unknown > 0).astype(np.uint8))
            if mixed_center is not None:
                roi_specs.extend([
                    ("alpha_refinement", "mixed_boundary", mixed_center, False),
                    ("matte_completion", "mixed_boundary", mixed_center, False),
                    ("boundary_uncertainty_refinement", "mixed_boundary", mixed_center, False),
                    ("compositing_aware_edge_correction", "mixed_boundary", mixed_center, False),
                ])

            for tag, mask in semantic_masks.items():
                center = _largest_component_center(mask)
                if center is not None:
                    roi_specs.append(("semantic_boundary_expert", tag, center, False))

            neg_center = _pick_hard_negative_center(gt_hard, gt_boundary)
            for _ in range(int(args.hard_negative_per_frame)):
                if neg_center is not None:
                    roi_specs.append(("alpha_refinement", "hard_negative", neg_center, True))

            for task_type, semantic_tag, center, is_hard_negative in roi_specs:
                center_x, center_y = center
                fg_patch, roi_box = _crop_with_padding((image_rgb * pred_alpha[..., None]).astype(np.uint8), center_x, center_y, args.patch_size)
                bg_patch, _ = _crop_with_padding(background_rgb, center_x, center_y, args.patch_size)
                arrays["foreground_patch"].append(fg_patch.astype(np.uint8))
                arrays["background_patch"].append(bg_patch.astype(np.uint8))
                for key, volume in [
                    ("input_soft_alpha", pred_alpha),
                    ("input_trimap_unknown", pred_trimap),
                    ("input_boundary_roi_mask", gt_boundary),
                    ("input_person_mask", pred_person),
                    ("input_boundary_band", pred_boundary),
                    ("input_uncertainty", pred_unc),
                    ("gt_soft_alpha", gt_alpha),
                    ("gt_trimap_unknown", gt_trimap_unknown),
                    ("gt_boundary_mask", gt_boundary),
                    ("gt_occlusion_band", gt_occluded),
                    ("gt_face_boundary", semantic_masks["face"]),
                    ("gt_hair_edge", semantic_masks["hair"]),
                    ("gt_hand_boundary", semantic_masks["hand"]),
                    ("gt_cloth_boundary", semantic_masks["cloth"]),
                    ("gt_semi_transparent", gt_semi),
                ]:
                    patch, _ = _crop_with_padding(volume, center_x, center_y, args.patch_size)
                    arrays[key].append(patch.astype(np.float32) if key == "gt_soft_alpha" or key.startswith("input_") and volume.dtype != np.uint8 else patch.astype(np.uint8))

                semantic_target_key = SEMANTIC_TARGET_KEYS[semantic_tag]
                arrays["semantic_target_mask"].append(arrays[semantic_target_key][-1])
                arrays["difficulty_score"].append(np.float32(record["difficulty_score"]))
                sample_id = f"{case['case_id']}_{frame_idx:06d}_{sample_counter:05d}"
                sample_counter += 1
                split_frames[dataset_split].append(frame_idx)
                records.append({
                    "sample_id": sample_id,
                    "dataset_split": dataset_split,
                    "source_review_split": source_split,
                    "case_id": case["case_id"],
                    "preprocess_frame_index": frame_idx,
                    "source_frame_index": int(label_json["source_frame_index"]),
                    "task_type": task_type,
                    "semantic_boundary_tag": semantic_tag,
                    "difficulty_score": float(record["difficulty_score"]),
                    "is_hard_negative": bool(is_hard_negative),
                    "roi_box_xyxy": roi_box,
                    "label_json_path": str(label_json_path.resolve()),
                })

    validate_records(records)

    npz_payload = {}
    for key, values in arrays.items():
        if key == "difficulty_score":
            npz_payload[key] = np.asarray(values, dtype=np.float32)
        elif key in ("foreground_patch", "background_patch"):
            npz_payload[key] = np.stack(values).astype(np.uint8)
        elif key == "gt_soft_alpha" or key.startswith("input_") and key not in ("input_trimap_unknown", "input_person_mask", "input_boundary_roi_mask"):
            npz_payload[key] = np.stack(values).astype(np.float16)
        else:
            npz_payload[key] = np.stack(values).astype(np.uint8)

    dataset_npz = output_dir / "roi_dataset_v2.npz"
    np.savez_compressed(dataset_npz, **npz_payload)

    task_counts = Counter(record["task_type"] for record in records)
    semantic_counts = Counter(record["semantic_boundary_tag"] for record in records)
    split_counts = Counter(record["dataset_split"] for record in records)
    hard_negative_ratio = {
        split: (
            sum(1 for record in records if record["dataset_split"] == split and record["is_hard_negative"])
            / max(sum(1 for record in records if record["dataset_split"] == split), 1)
        )
        for split in split_counts
    }

    dataset_json = output_dir / "roi_dataset_v2.json"
    dataset_json.write_text(
        json.dumps(
            {
                "reviewed_dataset_dir": str(reviewed_dir.resolve()),
                "prediction_preprocess_dir": str(preprocess_dir.resolve()),
                "task_taxonomy": str(Path(args.task_taxonomy).resolve()),
                "patch_size": int(args.patch_size),
                "split_policy": args.split_policy,
                "sample_count": int(len(records)),
                "task_counts": dict(task_counts),
                "semantic_counts": dict(semantic_counts),
                "split_counts": dict(split_counts),
                "hard_negative_ratio": hard_negative_ratio,
                "records": records,
            },
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    print(json.dumps({
        "dataset_npz": str(dataset_npz.resolve()),
        "sample_count": len(records),
        "split_policy": args.split_policy,
        "split_counts": dict(split_counts),
        "hard_negative_ratio": hard_negative_ratio,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
