#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan.modules.animate.preprocess.external_alpha_backgroundmattingv2 import (
    BackgroundMattingV2Adapter,
    BackgroundMattingV2Config,
)
from wan.utils.media_io import load_person_mask_artifact


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _read_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _boundary_f1(pred_mask: np.ndarray, label_mask: np.ndarray) -> float:
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--background_preprocess_dir", required=True)
    parser.add_argument("--baseline_preprocess_dir", default=None)
    parser.add_argument("--baseline_blend", type=float, default=0.0)
    parser.add_argument("--external_blend", type=float, default=1.0)
    parser.add_argument("--hair_boost", type=float, default=0.0)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    summary = _read_json(dataset_dir / "summary.json")
    bg_dir = Path(args.background_preprocess_dir) / "src_bg"
    adapter = BackgroundMattingV2Adapter(BackgroundMattingV2Config(model_id=args.model_id))
    baseline_alpha = None
    if args.baseline_preprocess_dir:
        meta = _read_json(Path(args.baseline_preprocess_dir) / "metadata.json")
        artifact = meta["src_files"]["soft_alpha"]
        baseline_alpha = load_person_mask_artifact(Path(args.baseline_preprocess_dir) / artifact["path"], artifact.get("format")).astype(np.float32)

    metrics = {
        "alpha_mae": [],
        "trimap_error": [],
        "hair_boundary_f1": [],
        "runtime_sec": [],
    }
    records = []

    for case in summary["cases"]:
        for record in case["records"]:
            label_json = _read_json(Path(record["label_json_path"]))
            label_root = Path(record["label_json_path"]).parent
            preprocess_idx = int(label_json["preprocess_frame_index"])
            image_rgb = _read_rgb(Path(record["image_path"]))
            background_rgb = _read_rgb(bg_dir / f"{preprocess_idx:06d}.png")
            out = adapter.infer(image_rgb, background_rgb)
            alpha = out["alpha"].astype(np.float32)
            if baseline_alpha is not None:
                alpha = np.clip(
                    args.external_blend * alpha + args.baseline_blend * baseline_alpha[preprocess_idx],
                    0.0,
                    1.0,
                ).astype(np.float32)
            trimap_unknown = (_read_gray(label_root / label_json["annotations"]["trimap_unknown"]["path"]) > 127).astype(np.float32)
            alpha_label = _read_gray(label_root / label_json["annotations"]["soft_alpha"]["path"]).astype(np.float32) / 255.0
            hair_boundary = (_read_gray(label_root / label_json["annotations"]["hair_edge_mask"]["path"]) > 127).astype(np.uint8)
            boundary_label = (_read_gray(label_root / label_json["annotations"]["boundary_mask"]["path"]) > 127).astype(np.uint8)

            if args.hair_boost > 0:
                alpha = np.clip(alpha + args.hair_boost * out["hair_alpha"].astype(np.float32), 0.0, 1.0)
            pred_boundary = (cv2.Canny((alpha * 255).astype(np.uint8), 32, 96) > 0).astype(np.uint8)
            alpha_mae = float(np.abs(alpha - alpha_label).mean())
            trimap_error = float((np.abs(alpha - alpha_label) * trimap_unknown).sum() / max(trimap_unknown.sum(), 1.0))
            hair_boundary_f1 = _boundary_f1(np.logical_and(pred_boundary > 0, hair_boundary > 0).astype(np.uint8),
                                            np.logical_and(boundary_label > 0, hair_boundary > 0).astype(np.uint8))

            metrics["alpha_mae"].append(alpha_mae)
            metrics["trimap_error"].append(trimap_error)
            metrics["hair_boundary_f1"].append(hair_boundary_f1)
            metrics["runtime_sec"].append(float(out["runtime_sec"]))
            records.append({
                "case_id": case["case_id"],
                "preprocess_frame_index": preprocess_idx,
                "alpha_mae": alpha_mae,
                "trimap_error": trimap_error,
                "hair_boundary_f1": hair_boundary_f1,
                "runtime_sec": float(out["runtime_sec"]),
            })

    result = {
        "dataset_dir": str(dataset_dir.resolve()),
        "background_preprocess_dir": str(Path(args.background_preprocess_dir).resolve()),
        "model_id": args.model_id,
        "label_count": len(records),
        "alpha_mae_mean": float(np.mean(metrics["alpha_mae"])) if metrics["alpha_mae"] else None,
        "trimap_error_mean": float(np.mean(metrics["trimap_error"])) if metrics["trimap_error"] else None,
        "hair_boundary_f1_mean": float(np.mean(metrics["hair_boundary_f1"])) if metrics["hair_boundary_f1"] else None,
        "runtime_sec_mean": float(np.mean(metrics["runtime_sec"])) if metrics["runtime_sec"] else None,
        "baseline_preprocess_dir": str(Path(args.baseline_preprocess_dir).resolve()) if args.baseline_preprocess_dir else None,
        "baseline_blend": float(args.baseline_blend),
        "external_blend": float(args.external_blend),
        "hair_boost": float(args.hair_boost),
        "records": records,
    }
    _write_json(Path(args.output_json), result)
    print(json.dumps({
        "model_id": args.model_id,
        "alpha_mae_mean": result["alpha_mae_mean"],
        "trimap_error_mean": result["trimap_error_mean"],
        "hair_boundary_f1_mean": result["hair_boundary_f1_mean"],
        "runtime_sec_mean": result["runtime_sec_mean"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
