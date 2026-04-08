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


def _trimap_from_mask(mask: np.ndarray, radius: int = 6) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    kernel = np.ones((radius * 2 + 1, radius * 2 + 1), dtype=np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=1)
    eroded = cv2.erode(mask, kernel, iterations=1)
    trimap = np.clip(dilated - eroded, 0, 1).astype(np.uint8)
    return trimap


def _dilate_binary(mask: np.ndarray, radius: int = 2) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    kernel = np.ones((radius * 2 + 1, radius * 2 + 1), dtype=np.uint8)
    return cv2.dilate(mask, kernel, iterations=1).astype(np.uint8)


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
    occlusion_path = metadata["src_files"].get("occlusion_band", {}).get("path")
    uncertainty_path = metadata["src_files"].get("uncertainty_map", {}).get("path")

    soft_alpha = _load_volume(src_root / soft_alpha_path).astype(np.float32) if soft_alpha_path else None
    boundary_band = _load_volume(src_root / boundary_band_path).astype(np.float32) if boundary_band_path else None
    occlusion_band = _load_volume(src_root / occlusion_path).astype(np.float32) if occlusion_path else None
    uncertainty = _load_volume(src_root / uncertainty_path).astype(np.float32) if uncertainty_path else None

    person_mask_f = person_mask.astype(np.float32)
    result = {
        "mode": "proxy",
        "frame_count": int(person_mask_f.shape[0]) if person_mask_f.ndim == 3 else 1,
        "hard_foreground_mean": float(person_mask_f.mean()),
        "hard_foreground_std": float(person_mask_f.std()),
        "soft_alpha_available": soft_alpha is not None,
        "boundary_band_available": boundary_band is not None,
        "occlusion_band_available": occlusion_band is not None,
        "uncertainty_available": uncertainty is not None,
        "soft_alpha_mean": None if soft_alpha is None else float(soft_alpha.mean()),
        "boundary_band_mean": None if boundary_band is None else float(boundary_band.mean()),
        "occlusion_band_mean": None if occlusion_band is None else float(occlusion_band.mean()),
        "uncertainty_mean": None if uncertainty is None else float(uncertainty.mean()),
        "artifacts": {
            "person_mask": metadata["src_files"]["person_mask"]["path"],
            "soft_alpha": soft_alpha_path,
            "boundary_band": boundary_band_path,
            "occlusion_band": occlusion_path,
            "uncertainty_map": uncertainty_path,
        },
    }
    if boundary_band is not None and uncertainty is not None:
        high_uncertainty = (uncertainty >= np.quantile(uncertainty, 0.8)).astype(np.float32)
        band_focus = (boundary_band > 0.25).astype(np.float32)
        result["uncertainty_boundary_focus_ratio"] = float((high_uncertainty * band_focus).sum() / max(high_uncertainty.sum(), 1.0))
        if soft_alpha is not None:
            transition_region = (
                (boundary_band > 0.10)
                | ((soft_alpha - person_mask_f) > 0.03)
                | ((occlusion_band if occlusion_band is not None else 0.0) > 0.10)
            ).astype(np.uint8)
        else:
            transition_region = ((boundary_band > 0.10) | ((occlusion_band if occlusion_band is not None else 0.0) > 0.10)).astype(np.uint8)
        transition_region_dilated = np.stack([_dilate_binary(frame, radius=2) for frame in transition_region]).astype(np.uint8)
        interior_region = ((person_mask_f > 0.5) & (transition_region == 0)).astype(np.uint8)
        background_region = ((person_mask_f <= 0.05) & (transition_region == 0)).astype(np.uint8)
        result["transition_region_mean"] = float(transition_region.mean())
        result["transition_region_dilated_mean"] = float(transition_region_dilated.mean())
        result["uncertainty_transition_focus_ratio"] = float(
            (high_uncertainty * transition_region).sum() / max(high_uncertainty.sum(), 1.0)
        )
        result["uncertainty_transition_focus_ratio_dilated"] = float(
            (high_uncertainty * transition_region_dilated).sum() / max(high_uncertainty.sum(), 1.0)
        )
        result["uncertainty_transition_coverage"] = float(
            (high_uncertainty * transition_region).sum() / max(transition_region.sum(), 1.0)
        )
        result["uncertainty_transition_coverage_dilated"] = float(
            (high_uncertainty * transition_region_dilated).sum() / max(transition_region_dilated.sum(), 1.0)
        )
        transition_mean = float(uncertainty[transition_region > 0].mean()) if transition_region.any() else 0.0
        interior_mean = float(uncertainty[interior_region > 0].mean()) if interior_region.any() else 0.0
        background_mean = float(uncertainty[background_region > 0].mean()) if background_region.any() else 0.0
        result["uncertainty_transition_mean"] = transition_mean
        result["uncertainty_interior_mean"] = interior_mean
        result["uncertainty_background_mean"] = background_mean
        result["uncertainty_transition_to_interior_ratio"] = float(transition_mean / max(interior_mean, 1e-6))
        result["uncertainty_transition_to_background_ratio"] = float(transition_mean / max(background_mean, 1e-6))

    if args.label_json:
        label = _read_json(Path(args.label_json))
        frame_index = int(label["frame_index"])
        pred_mask = (person_mask_f[frame_index] > 0.5).astype(np.uint8)
        hard_path = label["annotations"]["hard_foreground"]["path"]
        alpha_path = label["annotations"].get("soft_alpha", {}).get("path")
        band_path = label["annotations"].get("boundary_band", {}).get("path")
        occlusion_label_path = label["annotations"].get("occlusion_band", {}).get("path")
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
            trimap = _trimap_from_mask(label_mask)
            trimap_error = np.abs(pred_alpha - label_alpha) * trimap.astype(np.float32)
            result["trimap_error"] = float(trimap_error.sum() / max(trimap.sum(), 1))

        if band_path and boundary_band is not None:
            label_band = (_load_volume(Path(args.label_json).parent / band_path) > 127).astype(np.uint8)
            pred_band = (boundary_band[frame_index] > 0.5).astype(np.uint8)
            inter = np.logical_and(pred_band, label_band).sum()
            uni = np.logical_or(pred_band, label_band).sum()
            result["boundary_band_iou"] = float(inter / max(uni, 1))
        if occlusion_label_path and occlusion_band is not None:
            label_occ = (_load_volume(Path(args.label_json).parent / occlusion_label_path) > 127).astype(np.uint8)
            pred_occ = (occlusion_band[frame_index] > 0.5).astype(np.uint8)
            inter = np.logical_and(pred_occ, label_occ).sum()
            uni = np.logical_or(pred_occ, label_occ).sum()
            result["occlusion_band_iou"] = float(inter / max(uni, 1))
        if uncertainty is not None:
            error_mask = np.zeros_like(label_mask, dtype=np.uint8)
            error_mask[pred_mask != label_mask] = 1
            if alpha_path and soft_alpha is not None:
                pred_alpha = soft_alpha[frame_index].astype(np.float32)
                if pred_alpha.max() > 1.0:
                    pred_alpha = pred_alpha / max(pred_alpha.max(), 1.0)
                alpha_error = (np.abs(pred_alpha - label_alpha) > 0.15).astype(np.uint8)
                error_mask = np.maximum(error_mask, alpha_error)
            uncertainty_frame = uncertainty[frame_index].astype(np.float32)
            threshold = float(np.quantile(uncertainty_frame, 0.8))
            top_uncertainty = (uncertainty_frame >= threshold).astype(np.uint8)
            result["uncertainty_top20_error_coverage"] = float(
                np.logical_and(top_uncertainty > 0, error_mask > 0).sum() / max(error_mask.sum(), 1)
            )

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
