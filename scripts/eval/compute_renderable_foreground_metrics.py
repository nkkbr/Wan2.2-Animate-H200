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


def _masked_silhouette_jitter(masks: np.ndarray, roi: np.ndarray) -> float:
    values = []
    for idx in range(1, len(masks)):
        m = roi[idx].astype(np.float32)
        denom = float(m.sum())
        if denom <= 0:
            continue
        diff = np.abs(masks[idx].astype(np.float32) - masks[idx - 1].astype(np.float32))
        values.append(float((diff * m).sum() / denom))
    return float(np.mean(values)) if values else 0.0


def _pct(before: float, after: float) -> float:
    if before == 0:
        return 0.0
    return float((before - after) / abs(before) * 100.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Optimization9 Step05 renderable foreground metrics.")
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--baseline_root_path", required=True)
    parser.add_argument("--src_root_path", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    baseline_meta = load_preprocess_metadata(args.baseline_root_path)
    candidate_meta = load_preprocess_metadata(args.src_root_path)
    before = read_video_rgb(Path(args.before)).astype(np.float32) / 255.0
    after = read_video_rgb(Path(args.after)).astype(np.float32) / 255.0

    def _load_mask(meta: dict, root: Path, name: str):
        artifacts = meta["src_files"]
        if name not in artifacts:
            return None
        return load_mask_artifact(root / artifacts[name]["path"], artifacts[name].get("format")).astype(np.float32)

    baseline_root = Path(args.baseline_root_path)
    candidate_root = Path(args.src_root_path)
    baseline_person = _load_mask(baseline_meta, baseline_root, "person_mask")
    candidate_person = _load_mask(candidate_meta, candidate_root, "person_mask")
    baseline_boundary = _load_mask(baseline_meta, baseline_root, "boundary_band")
    candidate_silhouette = _load_mask(candidate_meta, candidate_root, "boundary_band")
    baseline_occlusion = _load_mask(baseline_meta, baseline_root, "occlusion_band")
    candidate_occlusion = _load_mask(candidate_meta, candidate_root, "occlusion_band")
    renderable_roi = _load_mask(candidate_meta, candidate_root, "renderable_roi_mask")

    if baseline_person is None or candidate_person is None or candidate_silhouette is None:
        raise SystemExit("Required person/boundary artifacts are missing.")

    frame_count = min(len(before), len(after), len(baseline_person), len(candidate_person), len(candidate_silhouette))
    if baseline_boundary is not None:
        frame_count = min(frame_count, len(baseline_boundary))
    if baseline_occlusion is not None:
        frame_count = min(frame_count, len(baseline_occlusion))
    if candidate_occlusion is not None:
        frame_count = min(frame_count, len(candidate_occlusion))
    if renderable_roi is not None:
        frame_count = min(frame_count, len(renderable_roi))

    silhouette_roi = np.maximum(candidate_silhouette[:frame_count], renderable_roi[:frame_count] if renderable_roi is not None else 0.0)
    occlusion_roi = np.maximum(
        baseline_occlusion[:frame_count] if baseline_occlusion is not None else 0.0,
        candidate_occlusion[:frame_count] if candidate_occlusion is not None else 0.0,
    )
    occlusion_roi = np.maximum(occlusion_roi, silhouette_roi * 0.2)

    before_silhouette = _masked_silhouette_jitter((baseline_person[:frame_count] > 0.5).astype(np.float32), silhouette_roi)
    after_silhouette = _masked_silhouette_jitter((candidate_person[:frame_count] > 0.5).astype(np.float32), silhouette_roi)
    before_motion = _masked_temporal_fluctuation(before[:frame_count], baseline_person[:frame_count])
    after_motion = _masked_temporal_fluctuation(after[:frame_count], candidate_person[:frame_count])
    before_occlusion = _masked_temporal_fluctuation(before[:frame_count], occlusion_roi)
    after_occlusion = _masked_temporal_fluctuation(after[:frame_count], occlusion_roi)

    payload = {
        "before_path": str(Path(args.before).resolve()),
        "after_path": str(Path(args.after).resolve()),
        "baseline_root_path": str(Path(args.baseline_root_path).resolve()),
        "src_root_path": str(Path(args.src_root_path).resolve()),
        "frame_count": int(frame_count),
        "silhouette_temporal_before_mean": before_silhouette,
        "silhouette_temporal_after_mean": after_silhouette,
        "motion_temporal_before_mean": before_motion,
        "motion_temporal_after_mean": after_motion,
        "occlusion_temporal_before_mean": before_occlusion,
        "occlusion_temporal_after_mean": after_occlusion,
        "silhouette_temporal_improvement_pct": _pct(before_silhouette, after_silhouette),
        "motion_temporal_improvement_pct": _pct(before_motion, after_motion),
        "occlusion_temporal_improvement_pct": _pct(before_occlusion, after_occlusion),
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
