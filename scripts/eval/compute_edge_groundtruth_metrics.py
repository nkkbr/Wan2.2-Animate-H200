#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _load_npz_volume(path: Path):
    data = np.load(path)
    if len(data.files) == 1:
        return data[data.files[0]]
    for key in ("arr_0", "volume", "data", "mask", "alpha", "band", "prior", "uncertainty"):
        if key in data:
            return data[key]
    raise KeyError(f"Unsupported npz keys in {path}: {data.files}")


def _load_volume(path: Path):
    if path.suffix == ".npz":
        return _load_npz_volume(path)
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _boundary_f1(pred_mask: np.ndarray, label_mask: np.ndarray):
    kernel = np.ones((3, 3), np.uint8)
    pred_edge = cv2.morphologyEx(pred_mask.astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0
    label_edge = cv2.morphologyEx(label_mask.astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0
    tolerance_kernel = np.ones((5, 5), np.uint8)
    pred_match_region = cv2.dilate(pred_edge.astype(np.uint8), tolerance_kernel, iterations=1) > 0
    label_match_region = cv2.dilate(label_edge.astype(np.uint8), tolerance_kernel, iterations=1) > 0

    matched_pred = np.logical_and(pred_edge, label_match_region).sum()
    matched_label = np.logical_and(label_edge, pred_match_region).sum()
    precision = float(matched_pred / max(pred_edge.sum(), 1))
    recall = float(matched_label / max(label_edge.sum(), 1))
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def _sad(pred_alpha: np.ndarray, label_alpha: np.ndarray):
    return float(np.abs(pred_alpha - label_alpha).sum())


def main():
    parser = argparse.ArgumentParser(description="Compute optimization4 edge groundtruth metrics against the mini benchmark dataset.")
    parser.add_argument("--dataset_dir", required=True, type=str)
    parser.add_argument("--prediction_preprocess_dir", required=True, type=str)
    parser.add_argument("--output_json", required=True, type=str)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    summary = _read_json(dataset_dir / "summary.json")

    preprocess_dir = Path(args.prediction_preprocess_dir)
    metadata = _read_json(preprocess_dir / "metadata.json")
    pred_person_mask = _load_npz_volume(preprocess_dir / metadata["src_files"]["person_mask"]["path"]).astype(np.float32)
    pred_soft_alpha = _load_npz_volume(preprocess_dir / metadata["src_files"]["soft_alpha"]["path"]).astype(np.float32)
    pred_boundary_band = _load_npz_volume(preprocess_dir / metadata["src_files"]["boundary_band"]["path"]).astype(np.float32)
    pred_occlusion_band = _load_npz_volume(preprocess_dir / metadata["src_files"]["occlusion_band"]["path"]).astype(np.float32)

    records = []
    boundary_f1_scores = []
    trimap_errors = []
    alpha_mae_values = []
    alpha_sad_values = []
    hard_iou_values = []
    boundary_iou_values = []
    occlusion_iou_values = []

    for case in summary["cases"]:
        for record in case["records"]:
            label_json = _read_json(Path(record["label_json_path"]))
            preprocess_idx = int(label_json["preprocess_frame_index"])
            label_root = Path(record["label_json_path"]).parent

            hard_label = (_load_volume(label_root / label_json["annotations"]["hard_foreground"]["path"]) > 127).astype(np.uint8)
            alpha_label = _load_volume(label_root / label_json["annotations"]["soft_alpha"]["path"]).astype(np.float32) / 255.0
            trimap = _load_volume(label_root / label_json["annotations"]["trimap"]["path"]).astype(np.uint8)
            boundary_label = (_load_volume(label_root / label_json["annotations"]["boundary_mask"]["path"]) > 127).astype(np.uint8)
            occlusion_label = (_load_volume(label_root / label_json["annotations"]["occlusion_band"]["path"]) > 127).astype(np.uint8)

            pred_hard = (pred_person_mask[preprocess_idx] > 0.5).astype(np.uint8)
            pred_alpha = np.clip(pred_soft_alpha[preprocess_idx].astype(np.float32), 0.0, 1.0)
            pred_boundary = (pred_boundary_band[preprocess_idx] > 0.10).astype(np.uint8)
            pred_occlusion = (pred_occlusion_band[preprocess_idx] > 0.10).astype(np.uint8)

            inter = np.logical_and(pred_hard, hard_label).sum()
            union = np.logical_or(pred_hard, hard_label).sum()
            hard_iou = float(inter / max(union, 1))

            b_inter = np.logical_and(pred_boundary, boundary_label).sum()
            b_union = np.logical_or(pred_boundary, boundary_label).sum()
            boundary_iou = float(b_inter / max(b_union, 1))

            o_inter = np.logical_and(pred_occlusion, occlusion_label).sum()
            o_union = np.logical_or(pred_occlusion, occlusion_label).sum()
            occlusion_iou = float(o_inter / max(o_union, 1))

            boundary_f1 = _boundary_f1(pred_hard, hard_label)
            alpha_mae = float(np.abs(pred_alpha - alpha_label).mean())
            alpha_sad = _sad(pred_alpha, alpha_label)
            trimap_unknown = (trimap == 128).astype(np.float32)
            trimap_error = float((np.abs(pred_alpha - alpha_label) * trimap_unknown).sum() / max(trimap_unknown.sum(), 1.0))

            boundary_f1_scores.append(boundary_f1)
            trimap_errors.append(trimap_error)
            alpha_mae_values.append(alpha_mae)
            alpha_sad_values.append(alpha_sad)
            hard_iou_values.append(hard_iou)
            boundary_iou_values.append(boundary_iou)
            occlusion_iou_values.append(occlusion_iou)

            records.append(
                {
                    "case_id": case["case_id"],
                    "preprocess_frame_index": preprocess_idx,
                    "boundary_f1": boundary_f1,
                    "trimap_error": trimap_error,
                    "alpha_mae": alpha_mae,
                    "alpha_sad": alpha_sad,
                    "hard_mask_iou": hard_iou,
                    "boundary_mask_iou": boundary_iou,
                    "occlusion_band_iou": occlusion_iou,
                }
            )

    result = {
        "dataset_dir": str(dataset_dir.resolve()),
        "prediction_preprocess_dir": str(preprocess_dir.resolve()),
        "label_count": len(records),
        "boundary_f1_mean": float(np.mean(boundary_f1_scores)) if boundary_f1_scores else None,
        "trimap_error_mean": float(np.mean(trimap_errors)) if trimap_errors else None,
        "alpha_mae_mean": float(np.mean(alpha_mae_values)) if alpha_mae_values else None,
        "alpha_sad_mean": float(np.mean(alpha_sad_values)) if alpha_sad_values else None,
        "hard_mask_iou_mean": float(np.mean(hard_iou_values)) if hard_iou_values else None,
        "boundary_mask_iou_mean": float(np.mean(boundary_iou_values)) if boundary_iou_values else None,
        "occlusion_band_iou_mean": float(np.mean(occlusion_iou_values)) if occlusion_iou_values else None,
        "records": records,
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in [
        "label_count",
        "boundary_f1_mean",
        "trimap_error_mean",
        "alpha_mae_mean",
        "alpha_sad_mean",
    ]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
