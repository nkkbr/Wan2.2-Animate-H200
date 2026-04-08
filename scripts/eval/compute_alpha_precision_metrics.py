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


def _iou(pred: np.ndarray, label: np.ndarray) -> float:
    inter = np.logical_and(pred > 0, label > 0).sum()
    union = np.logical_or(pred > 0, label > 0).sum()
    return float(inter / max(union, 1))


def _load_optional_mask(preprocess_dir: Path, metadata: dict, key: str):
    if key not in metadata["src_files"]:
        return None
    artifact = metadata["src_files"][key]
    return _load_npz_volume(preprocess_dir / artifact["path"]).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Compute optimization4 Step 02 alpha precision metrics.")
    parser.add_argument("--dataset_dir", required=True, type=str)
    parser.add_argument("--prediction_preprocess_dir", required=True, type=str)
    parser.add_argument("--output_json", required=True, type=str)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    summary = _read_json(dataset_dir / "summary.json")
    preprocess_dir = Path(args.prediction_preprocess_dir)
    metadata = _read_json(preprocess_dir / "metadata.json")

    pred_hard = _load_npz_volume(preprocess_dir / metadata["src_files"]["person_mask"]["path"]).astype(np.float32)
    pred_alpha = _load_npz_volume(preprocess_dir / metadata["src_files"]["soft_alpha"]["path"]).astype(np.float32)
    pred_trimap = _load_optional_mask(preprocess_dir, metadata, "trimap_v2")
    pred_fine_boundary = _load_optional_mask(preprocess_dir, metadata, "fine_boundary_mask")
    pred_hair_edge = _load_optional_mask(preprocess_dir, metadata, "hair_edge_mask")
    pred_alpha_uncertainty = _load_optional_mask(preprocess_dir, metadata, "alpha_uncertainty_v2")

    records = []
    boundary_f1_scores = []
    trimap_errors = []
    alpha_mae_values = []
    alpha_sad_values = []
    fine_boundary_iou_values = []
    trimap_unknown_iou_values = []
    hair_boundary_overlap_values = []
    uncertainty_error_focus_values = []

    for case in summary["cases"]:
        for record in case["records"]:
            label_json = _read_json(Path(record["label_json_path"]))
            preprocess_idx = int(label_json["preprocess_frame_index"])
            label_root = Path(record["label_json_path"]).parent

            hard_label = (_load_volume(label_root / label_json["annotations"]["hard_foreground"]["path"]) > 127).astype(np.uint8)
            alpha_label = _load_volume(label_root / label_json["annotations"]["soft_alpha"]["path"]).astype(np.float32) / 255.0
            trimap_label = _load_volume(label_root / label_json["annotations"]["trimap"]["path"]).astype(np.uint8)
            boundary_label = (_load_volume(label_root / label_json["annotations"]["boundary_mask"]["path"]) > 127).astype(np.uint8)

            pred_hard_frame = (pred_hard[preprocess_idx] > 0.5).astype(np.uint8)
            pred_alpha_frame = np.clip(pred_alpha[preprocess_idx].astype(np.float32), 0.0, 1.0)

            boundary_f1 = _boundary_f1(pred_hard_frame, hard_label)
            alpha_mae = float(np.abs(pred_alpha_frame - alpha_label).mean())
            alpha_sad = float(np.abs(pred_alpha_frame - alpha_label).sum())
            trimap_unknown = (trimap_label == 128).astype(np.float32)
            trimap_error = float((np.abs(pred_alpha_frame - alpha_label) * trimap_unknown).sum() / max(trimap_unknown.sum(), 1.0))

            fine_boundary_iou = None
            if pred_fine_boundary is not None:
                fine_boundary_iou = _iou(pred_fine_boundary[preprocess_idx] > 0.15, boundary_label > 0)
                fine_boundary_iou_values.append(fine_boundary_iou)

            trimap_unknown_iou = None
            if pred_trimap is not None:
                pred_unknown = np.logical_and(pred_trimap[preprocess_idx] > 0.25, pred_trimap[preprocess_idx] < 0.75)
                trimap_unknown_iou = _iou(pred_unknown, trimap_label == 128)
                trimap_unknown_iou_values.append(trimap_unknown_iou)

            hair_boundary_overlap = None
            if pred_hair_edge is not None:
                coords = np.argwhere(boundary_label > 0)
                if coords.size > 0:
                    y0 = int(coords[:, 0].min())
                    y1 = int(coords[:, 0].max())
                    cutoff = y0 + int(round((y1 - y0 + 1) * 0.55))
                    top_boundary = np.zeros_like(boundary_label, dtype=np.uint8)
                    top_boundary[y0: max(cutoff, y0 + 1), :] = boundary_label[y0: max(cutoff, y0 + 1), :]
                    hair_boundary_overlap = _iou(pred_hair_edge[preprocess_idx] > 0.10, top_boundary > 0)
                    hair_boundary_overlap_values.append(hair_boundary_overlap)

            uncertainty_focus = None
            if pred_alpha_uncertainty is not None:
                abs_error = np.abs(pred_alpha_frame - alpha_label).astype(np.float32)
                threshold = float(np.quantile(abs_error, 0.90))
                high_error = (abs_error >= threshold).astype(np.float32)
                unc = np.clip(pred_alpha_uncertainty[preprocess_idx].astype(np.float32), 0.0, 1.0)
                focus_mean = float((unc * high_error).sum() / max(high_error.sum(), 1.0))
                global_mean = float(unc.mean())
                uncertainty_focus = float(focus_mean / max(global_mean, 1e-6))
                uncertainty_error_focus_values.append(uncertainty_focus)

            boundary_f1_scores.append(boundary_f1)
            trimap_errors.append(trimap_error)
            alpha_mae_values.append(alpha_mae)
            alpha_sad_values.append(alpha_sad)
            records.append(
                {
                    "case_id": case["case_id"],
                    "preprocess_frame_index": preprocess_idx,
                    "boundary_f1": boundary_f1,
                    "trimap_error": trimap_error,
                    "alpha_mae": alpha_mae,
                    "alpha_sad": alpha_sad,
                    "fine_boundary_iou": fine_boundary_iou,
                    "trimap_unknown_iou": trimap_unknown_iou,
                    "hair_boundary_overlap": hair_boundary_overlap,
                    "uncertainty_error_focus_ratio": uncertainty_focus,
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
        "fine_boundary_iou_mean": float(np.mean(fine_boundary_iou_values)) if fine_boundary_iou_values else None,
        "trimap_unknown_iou_mean": float(np.mean(trimap_unknown_iou_values)) if trimap_unknown_iou_values else None,
        "hair_boundary_overlap_mean": float(np.mean(hair_boundary_overlap_values)) if hair_boundary_overlap_values else None,
        "uncertainty_error_focus_ratio_mean": float(np.mean(uncertainty_error_focus_values)) if uncertainty_error_focus_values else None,
        "records": records,
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "label_count": result["label_count"],
        "boundary_f1_mean": result["boundary_f1_mean"],
        "trimap_error_mean": result["trimap_error_mean"],
        "alpha_mae_mean": result["alpha_mae_mean"],
        "fine_boundary_iou_mean": result["fine_boundary_iou_mean"],
        "trimap_unknown_iou_mean": result["trimap_unknown_iou_mean"],
        "uncertainty_error_focus_ratio_mean": result["uncertainty_error_focus_ratio_mean"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
