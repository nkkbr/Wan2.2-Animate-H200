#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan.utils.animate_contract import load_preprocess_metadata
from wan.utils.media_io import load_mask_artifact
from wan.utils.trainable_edge_model import EdgeRefinementNet, split_outputs


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _load_gray(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _load_rgb(path: Path, size: tuple[int, int]) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
    return img.astype(np.float32) / 255.0


def _boundary_f1(pred_mask: np.ndarray, label_mask: np.ndarray):
    kernel = np.ones((3, 3), np.uint8)
    pred_edge = cv2.morphologyEx(pred_mask.astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0
    label_edge = cv2.morphologyEx(label_mask.astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0
    tol = np.ones((5, 5), np.uint8)
    pred_match = cv2.dilate(pred_edge.astype(np.uint8), tol, iterations=1) > 0
    label_match = cv2.dilate(label_edge.astype(np.uint8), tol, iterations=1) > 0
    matched_pred = np.logical_and(pred_edge, label_match).sum()
    matched_label = np.logical_and(label_edge, pred_match).sum()
    precision = float(matched_pred / max(pred_edge.sum(), 1))
    recall = float(matched_label / max(label_edge.sum(), 1))
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def _masked_boundary_f1(pred_boundary: np.ndarray, label_boundary: np.ndarray, category_mask: np.ndarray):
    if int(category_mask.sum()) == 0:
        return None
    pred = np.logical_and(pred_boundary > 0, category_mask > 0).astype(np.uint8)
    label = np.logical_and(label_boundary > 0, category_mask > 0).astype(np.uint8)
    return _boundary_f1(pred, label)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trainable edge checkpoint directly on reviewed GT frames.")
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--dataset_dir", required=True, type=str)
    parser.add_argument("--baseline_preprocess_dir", required=True, type=str)
    parser.add_argument("--output_json", required=True, type=str)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    summary = _read_json(dataset_dir / "summary.json")
    baseline_dir = Path(args.baseline_preprocess_dir).resolve()
    metadata = load_preprocess_metadata(baseline_dir)
    artifacts = metadata["src_files"]

    soft_alpha = load_mask_artifact(baseline_dir / artifacts["soft_alpha"]["path"], artifacts["soft_alpha"].get("format"))
    boundary_band = load_mask_artifact(baseline_dir / artifacts["boundary_band"]["path"], artifacts["boundary_band"].get("format"))
    occlusion_band = load_mask_artifact(baseline_dir / artifacts["occlusion_band"]["path"], artifacts["occlusion_band"].get("format"))

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    size = tuple(checkpoint["size"])
    model = EdgeRefinementNet()
    model.load_state_dict(checkpoint["model"])
    model.eval()

    metrics = {
        "boundary_f1": [],
        "alpha_mae": [],
        "alpha_sad": [],
        "trimap_error": [],
        "face_boundary_f1": [],
        "hair_boundary_f1": [],
        "hand_boundary_f1": [],
        "cloth_boundary_f1": [],
        "occluded_boundary_f1": [],
    }
    records = []

    for case in summary["cases"]:
        for record in case["records"]:
            preprocess_idx = int(record["preprocess_frame_index"])
            label_json = _read_json(Path(record["label_json_path"]))
            label_root = Path(record["label_json_path"]).parent

            rgb = _load_rgb(Path(record["image_path"]), size)
            alpha_base = cv2.resize(soft_alpha[preprocess_idx], size, interpolation=cv2.INTER_LINEAR)
            boundary_base = cv2.resize(boundary_band[preprocess_idx], size, interpolation=cv2.INTER_LINEAR)
            inputs = np.concatenate([rgb, alpha_base[..., None], boundary_base[..., None]], axis=-1)
            inputs_t = torch.from_numpy(inputs).permute(2, 0, 1).unsqueeze(0).float()
            with torch.no_grad():
                outputs = split_outputs(model(inputs_t))
            pred_alpha = outputs.alpha.squeeze().cpu().numpy().astype(np.float32)
            pred_boundary = outputs.boundary.squeeze().cpu().numpy().astype(np.float32)

            target_size = (int(metadata["width"]), int(metadata["height"]))
            pred_alpha = cv2.resize(pred_alpha, target_size, interpolation=cv2.INTER_LINEAR)
            pred_boundary = cv2.resize(pred_boundary, target_size, interpolation=cv2.INTER_LINEAR)

            hard_label = (_load_gray(label_root / label_json["annotations"]["hard_foreground"]["path"]) > 127).astype(np.uint8)
            alpha_label = _load_gray(label_root / label_json["annotations"]["soft_alpha"]["path"]).astype(np.float32) / 255.0
            trimap_unknown = (_load_gray(label_root / label_json["annotations"]["trimap_unknown"]["path"]) > 127).astype(np.float32)
            boundary_label = (_load_gray(label_root / label_json["annotations"]["boundary_mask"]["path"]) > 127).astype(np.uint8)
            occlusion_label = (_load_gray(label_root / label_json["annotations"]["occlusion_band"]["path"]) > 127).astype(np.uint8)
            face_mask = (_load_gray(label_root / label_json["annotations"]["face_boundary_mask"]["path"]) > 127).astype(np.uint8)
            hair_mask = (_load_gray(label_root / label_json["annotations"]["hair_edge_mask"]["path"]) > 127).astype(np.uint8)
            hand_mask = (_load_gray(label_root / label_json["annotations"]["hand_boundary_mask"]["path"]) > 127).astype(np.uint8)
            cloth_mask = (_load_gray(label_root / label_json["annotations"]["cloth_boundary_mask"]["path"]) > 127).astype(np.uint8)

            pred_hard = (pred_alpha > 0.5).astype(np.uint8)
            pred_boundary_mask = (pred_boundary > 0.10).astype(np.uint8)
            pred_occ = (occlusion_band[preprocess_idx] > 0.10).astype(np.uint8)

            boundary_f1 = _boundary_f1(pred_boundary_mask, boundary_label)
            alpha_mae = float(np.abs(pred_alpha - alpha_label).mean())
            alpha_sad = float(np.abs(pred_alpha - alpha_label).sum())
            trimap_error = float((np.abs(pred_alpha - alpha_label) * trimap_unknown).sum() / max(trimap_unknown.sum(), 1.0))

            face_boundary_f1 = _masked_boundary_f1(pred_boundary_mask, boundary_label, face_mask)
            hair_boundary_f1 = _masked_boundary_f1(pred_boundary_mask, boundary_label, hair_mask)
            hand_boundary_f1 = _masked_boundary_f1(pred_boundary_mask, boundary_label, hand_mask)
            cloth_boundary_f1 = _masked_boundary_f1(pred_boundary_mask, boundary_label, cloth_mask)
            occluded_boundary_f1 = _masked_boundary_f1(pred_occ, occlusion_label, occlusion_label)

            metrics["boundary_f1"].append(boundary_f1)
            metrics["alpha_mae"].append(alpha_mae)
            metrics["alpha_sad"].append(alpha_sad)
            metrics["trimap_error"].append(trimap_error)
            for key, value in [
                ("face_boundary_f1", face_boundary_f1),
                ("hair_boundary_f1", hair_boundary_f1),
                ("hand_boundary_f1", hand_boundary_f1),
                ("cloth_boundary_f1", cloth_boundary_f1),
                ("occluded_boundary_f1", occluded_boundary_f1),
            ]:
                if value is not None:
                    metrics[key].append(float(value))

            records.append({
                "case_id": case["case_id"],
                "preprocess_frame_index": preprocess_idx,
                "boundary_f1": boundary_f1,
                "alpha_mae": alpha_mae,
                "alpha_sad": alpha_sad,
                "trimap_error": trimap_error,
                "face_boundary_f1": face_boundary_f1,
                "hair_boundary_f1": hair_boundary_f1,
                "hand_boundary_f1": hand_boundary_f1,
                "cloth_boundary_f1": cloth_boundary_f1,
                "occluded_boundary_f1": occluded_boundary_f1,
            })

    result = {
        "dataset_dir": str(dataset_dir),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "baseline_preprocess_dir": str(baseline_dir),
        "label_count": len(records),
        "boundary_f1_mean": float(np.mean(metrics["boundary_f1"])) if metrics["boundary_f1"] else None,
        "alpha_mae_mean": float(np.mean(metrics["alpha_mae"])) if metrics["alpha_mae"] else None,
        "alpha_sad_mean": float(np.mean(metrics["alpha_sad"])) if metrics["alpha_sad"] else None,
        "trimap_error_mean": float(np.mean(metrics["trimap_error"])) if metrics["trimap_error"] else None,
        "face_boundary_f1_mean": float(np.mean(metrics["face_boundary_f1"])) if metrics["face_boundary_f1"] else None,
        "hair_boundary_f1_mean": float(np.mean(metrics["hair_boundary_f1"])) if metrics["hair_boundary_f1"] else None,
        "hand_boundary_f1_mean": float(np.mean(metrics["hand_boundary_f1"])) if metrics["hand_boundary_f1"] else None,
        "cloth_boundary_f1_mean": float(np.mean(metrics["cloth_boundary_f1"])) if metrics["cloth_boundary_f1"] else None,
        "occluded_boundary_f1_mean": float(np.mean(metrics["occluded_boundary_f1"])) if metrics["occluded_boundary_f1"] else None,
        "records": records,
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in [
        "label_count",
        "boundary_f1_mean",
        "alpha_mae_mean",
        "trimap_error_mean",
        "face_boundary_f1_mean",
        "hair_boundary_f1_mean",
        "hand_boundary_f1_mean",
        "cloth_boundary_f1_mean",
        "occluded_boundary_f1_mean",
    ]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
