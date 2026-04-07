#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _load_volume(path: Path):
    if path.suffix == ".npz":
        data = np.load(path)
        if len(data.files) == 1:
            return data[data.files[0]]
        for key in ("arr_0", "volume", "data", "mask", "alpha", "band", "prior", "uncertainty"):
            if key in data:
                return data[key]
        raise KeyError(f"Unsupported npz keys in {path}: {data.files}")
    if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f"Failed to read image: {path}")
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    raise ValueError(f"Unsupported volume path: {path}")


def _boundary_f1(pred_mask: np.ndarray, label_mask: np.ndarray):
    kernel = np.ones((3, 3), np.uint8)
    pred_edge = cv2.morphologyEx(pred_mask.astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0
    label_edge = cv2.morphologyEx(label_mask.astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0
    tp = np.logical_and(pred_edge, label_edge).sum()
    precision = float(tp / max(pred_edge.sum(), 1))
    recall = float(tp / max(label_edge.sum(), 1))
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def main():
    parser = argparse.ArgumentParser(description="Compute boundary precision metrics for optimization3.")
    parser.add_argument("--src_root_path", required=True, type=str)
    parser.add_argument("--label_json", type=str, default=None)
    parser.add_argument("--output_json", required=True, type=str)
    args = parser.parse_args()

    src_root = Path(args.src_root_path)
    metadata = _read_json(src_root / "metadata.json")

    person_mask = _load_volume(src_root / metadata["src_files"]["person_mask"]["path"]).astype(np.float32)
    soft_alpha_path = metadata["src_files"].get("soft_alpha", {}).get("path")
    boundary_band_path = metadata["src_files"].get("boundary_band", {}).get("path")
    uncertainty_path = metadata["src_files"].get("uncertainty_map", {}).get("path")

    soft_alpha = _load_volume(src_root / soft_alpha_path).astype(np.float32) if soft_alpha_path else None
    boundary_band = _load_volume(src_root / boundary_band_path).astype(np.float32) if boundary_band_path else None
    uncertainty = _load_volume(src_root / uncertainty_path).astype(np.float32) if uncertainty_path else None

    person_mask_f = person_mask.astype(np.float32)
    result = {
        "mode": "proxy",
        "frame_count": int(person_mask_f.shape[0]) if person_mask_f.ndim == 3 else 1,
        "hard_foreground_mean": float(person_mask_f.mean()),
        "hard_foreground_std": float(person_mask_f.std()),
        "soft_alpha_available": soft_alpha is not None,
        "boundary_band_available": boundary_band is not None,
        "uncertainty_available": uncertainty is not None,
        "soft_alpha_mean": None if soft_alpha is None else float(soft_alpha.mean()),
        "boundary_band_mean": None if boundary_band is None else float(boundary_band.mean()),
        "uncertainty_mean": None if uncertainty is None else float(uncertainty.mean()),
        "artifacts": {
            "person_mask": metadata["src_files"]["person_mask"]["path"],
            "soft_alpha": soft_alpha_path,
            "boundary_band": boundary_band_path,
            "uncertainty_map": uncertainty_path,
        },
    }

    if args.label_json:
        label = _read_json(Path(args.label_json))
        frame_index = int(label["frame_index"])
        pred_mask = (person_mask_f[frame_index] > 0.5).astype(np.uint8)
        hard_path = label["annotations"]["hard_foreground"]["path"]
        alpha_path = label["annotations"].get("soft_alpha", {}).get("path")
        band_path = label["annotations"].get("boundary_band", {}).get("path")
        label_mask = (_load_volume(Path(args.label_json).parent / hard_path) > 127).astype(np.uint8)

        intersection = np.logical_and(pred_mask, label_mask).sum()
        union = np.logical_or(pred_mask, label_mask).sum()
        result["mode"] = "labelled"
        result["hard_mask_iou"] = float(intersection / max(union, 1))
        result["boundary_f1"] = _boundary_f1(pred_mask, label_mask)

        if alpha_path and soft_alpha is not None:
            label_alpha = _load_volume(Path(args.label_json).parent / alpha_path).astype(np.float32)
            if label_alpha.max() > 1.0:
                label_alpha = label_alpha / 255.0
            pred_alpha = soft_alpha[frame_index].astype(np.float32)
            if pred_alpha.max() > 1.0:
                pred_alpha = pred_alpha / max(pred_alpha.max(), 1.0)
            result["soft_alpha_mae"] = float(np.abs(pred_alpha - label_alpha).mean())

        if band_path and boundary_band is not None:
            label_band = (_load_volume(Path(args.label_json).parent / band_path) > 127).astype(np.uint8)
            pred_band = (boundary_band[frame_index] > 0.5).astype(np.uint8)
            inter = np.logical_and(pred_band, label_band).sum()
            uni = np.logical_or(pred_band, label_band).sum()
            result["boundary_band_iou"] = float(inter / max(uni, 1))

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
