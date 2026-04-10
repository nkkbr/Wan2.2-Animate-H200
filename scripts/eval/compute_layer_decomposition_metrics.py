#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan.utils.animate_contract import load_preprocess_metadata, read_video_rgb
from wan.utils.media_io import load_mask_artifact


def _load_output_frames(path: str | Path) -> np.ndarray:
    path = Path(path)
    return read_video_rgb(path)


def _masked_temporal_fluctuation(frames: np.ndarray, mask: np.ndarray) -> float:
    values = []
    for idx in range(1, len(frames)):
        m = mask[idx].astype(np.float32)[..., None]
        denom = float(m.sum() * frames.shape[-1])
        if denom <= 0:
            continue
        diff = np.abs(frames[idx].astype(np.float32) - frames[idx - 1].astype(np.float32))
        values.append(float((diff * m).sum() / denom))
    return float(np.mean(values)) if values else 0.0


def _masked_gradient_mean(frames: np.ndarray, mask: np.ndarray) -> float:
    values = []
    for idx in range(len(frames)):
        frame = frames[idx].astype(np.float32)
        gray = frame.mean(axis=2)
        gy, gx = np.gradient(gray)
        grad = np.sqrt(gx * gx + gy * gy)
        m = mask[idx].astype(np.float32)
        denom = float(m.sum())
        if denom <= 0:
            continue
        values.append(float((grad * m).sum() / denom))
    return float(np.mean(values)) if values else 0.0


def _masked_contrast_mean(frames: np.ndarray, mask: np.ndarray) -> float:
    values = []
    for idx in range(len(frames)):
        gray = frames[idx].astype(np.float32).mean(axis=2)
        m = mask[idx] > 0
        if int(m.sum()) == 0:
            continue
        values.append(float(gray[m].std()))
    return float(np.mean(values)) if values else 0.0


def _pct(before: float, after: float, lower_is_better: bool) -> float:
    if before == 0:
        return 0.0
    if lower_is_better:
        return float((before - after) / abs(before) * 100.0)
    return float((after - before) / abs(before) * 100.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Optimization9 Step03 layer decomposition metrics.")
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--src_root_path", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    metadata = load_preprocess_metadata(args.src_root_path)
    before = _load_output_frames(args.before).astype(np.float32) / 255.0
    after = _load_output_frames(args.after).astype(np.float32) / 255.0
    artifacts = metadata["src_files"]

    def _load_mask(name: str):
        if name not in artifacts:
            return None
        return load_mask_artifact(Path(args.src_root_path) / artifacts[name]["path"], artifacts[name].get("format")).astype(np.float32)

    occluded_boundary = _load_mask("occluded_boundary")
    background_unresolved = _load_mask("background_unresolved")
    layer_roi_mask = _load_mask("layer_roi_mask")

    if occluded_boundary is None:
        raise SystemExit("occluded_boundary artifact is required.")
    if background_unresolved is None:
        raise SystemExit("background_unresolved artifact is required.")

    frame_count = min(len(before), len(after), len(occluded_boundary), len(background_unresolved))
    if layer_roi_mask is not None:
        frame_count = min(frame_count, len(layer_roi_mask))
    occlusion_roi = np.maximum(occluded_boundary[:frame_count], background_unresolved[:frame_count])
    if layer_roi_mask is not None:
        occlusion_roi = np.maximum(occlusion_roi, layer_roi_mask[:frame_count])
    occlusion_roi = np.clip(occlusion_roi, 0.0, 1.0).astype(np.float32)

    before_temporal = _masked_temporal_fluctuation(before[:frame_count], occlusion_roi)
    after_temporal = _masked_temporal_fluctuation(after[:frame_count], occlusion_roi)
    before_grad = _masked_gradient_mean(before[:frame_count], occlusion_roi)
    after_grad = _masked_gradient_mean(after[:frame_count], occlusion_roi)
    before_contrast = _masked_contrast_mean(before[:frame_count], occlusion_roi)
    after_contrast = _masked_contrast_mean(after[:frame_count], occlusion_roi)

    payload = {
        "before_path": str(Path(args.before).resolve()),
        "after_path": str(Path(args.after).resolve()),
        "src_root_path": str(Path(args.src_root_path).resolve()),
        "frame_count": int(frame_count),
        "occlusion_roi_mean": float(occlusion_roi.mean()),
        "occlusion_temporal_before_mean": before_temporal,
        "occlusion_temporal_after_mean": after_temporal,
        "occlusion_gradient_before_mean": before_grad,
        "occlusion_gradient_after_mean": after_grad,
        "occlusion_contrast_before_mean": before_contrast,
        "occlusion_contrast_after_mean": after_contrast,
        "occlusion_temporal_improvement_pct": _pct(before_temporal, after_temporal, True),
        "occlusion_gradient_gain_pct": _pct(before_grad, after_grad, False),
        "occlusion_contrast_gain_pct": _pct(before_contrast, after_contrast, False),
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
