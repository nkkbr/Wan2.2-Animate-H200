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

from wan.utils.animate_contract import load_preprocess_metadata, read_video_rgb
from wan.utils.media_io import load_mask_artifact


def _load_output_frames(path: str | Path) -> np.ndarray:
    return read_video_rgb(Path(path))


def _load_video_frames(video_path: Path, frame_count: int, shape_hw: tuple[int, int]) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or frame_count)
    if total_frames <= 0:
        total_frames = frame_count
    frame_indices = np.linspace(0, max(total_frames - 1, 0), num=frame_count).round().astype(int)
    frames = []
    target_h, target_w = shape_hw
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read frame {idx} from {video_path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release()
    return np.stack(frames, axis=0).astype(np.uint8)


def _masked_mae(frames: np.ndarray, source: np.ndarray, mask: np.ndarray) -> float:
    values = []
    for idx in range(len(frames)):
        m = mask[idx].astype(np.float32)[..., None]
        denom = float(m.sum() * frames.shape[-1])
        if denom <= 0:
            continue
        diff = np.abs(frames[idx].astype(np.float32) - source[idx].astype(np.float32))
        values.append(float((diff * m).sum() / denom))
    return float(np.mean(values)) if values else 0.0


def _pct(before: float, after: float) -> float:
    if before == 0:
        return 0.0
    return float((before - after) / abs(before) * 100.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Optimization9 Step04 RGBA foreground metrics.")
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--src_root_path", required=True)
    parser.add_argument("--source_video_path", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    metadata = load_preprocess_metadata(args.src_root_path)
    before = _load_output_frames(args.before).astype(np.float32)
    after = _load_output_frames(args.after).astype(np.float32)
    artifacts = metadata["src_files"]

    def _load_mask(name: str):
        if name not in artifacts:
            return None
        return load_mask_artifact(Path(args.src_root_path) / artifacts[name]["path"], artifacts[name].get("format")).astype(np.float32)

    roi_mask = _load_mask("rgba_roi_mask")
    boundary_band = _load_mask("boundary_band")
    hair_boundary = _load_mask("hair_boundary")
    hand_boundary = _load_mask("hand_boundary")
    cloth_boundary = _load_mask("cloth_boundary")
    if roi_mask is None:
        raise SystemExit("rgba_roi_mask artifact is required.")
    if boundary_band is None:
        raise SystemExit("boundary_band artifact is required.")

    frame_count = min(len(before), len(after), len(roi_mask), len(boundary_band))
    shape_hw = before.shape[1:3]
    source = _load_video_frames(Path(args.source_video_path).resolve(), frame_count, shape_hw)

    boundary_roi = np.maximum(roi_mask[:frame_count], boundary_band[:frame_count])
    hair_roi = np.clip(hair_boundary[:frame_count] * 0.8 + boundary_roi * 0.2, 0.0, 1.0) if hair_boundary is not None else boundary_roi
    hand_roi = np.clip(hand_boundary[:frame_count] * 0.8 + boundary_roi * 0.2, 0.0, 1.0) if hand_boundary is not None else boundary_roi
    cloth_roi = np.clip(cloth_boundary[:frame_count] * 0.8 + boundary_roi * 0.2, 0.0, 1.0) if cloth_boundary is not None else boundary_roi

    before_boundary = _masked_mae(before[:frame_count], source[:frame_count], boundary_roi)
    after_boundary = _masked_mae(after[:frame_count], source[:frame_count], boundary_roi)
    before_hair = _masked_mae(before[:frame_count], source[:frame_count], hair_roi)
    after_hair = _masked_mae(after[:frame_count], source[:frame_count], hair_roi)
    before_hand = _masked_mae(before[:frame_count], source[:frame_count], hand_roi)
    after_hand = _masked_mae(after[:frame_count], source[:frame_count], hand_roi)
    before_cloth = _masked_mae(before[:frame_count], source[:frame_count], cloth_roi)
    after_cloth = _masked_mae(after[:frame_count], source[:frame_count], cloth_roi)

    payload = {
        "before_path": str(Path(args.before).resolve()),
        "after_path": str(Path(args.after).resolve()),
        "src_root_path": str(Path(args.src_root_path).resolve()),
        "source_video_path": str(Path(args.source_video_path).resolve()),
        "frame_count": int(frame_count),
        "rgba_roi_mean": float(boundary_roi.mean()),
        "rgba_boundary_reconstruction_before_mean": before_boundary,
        "rgba_boundary_reconstruction_after_mean": after_boundary,
        "rgba_hair_reconstruction_before_mean": before_hair,
        "rgba_hair_reconstruction_after_mean": after_hair,
        "rgba_hand_reconstruction_before_mean": before_hand,
        "rgba_hand_reconstruction_after_mean": after_hand,
        "rgba_cloth_reconstruction_before_mean": before_cloth,
        "rgba_cloth_reconstruction_after_mean": after_cloth,
        "rgba_boundary_reconstruction_improvement_pct": _pct(before_boundary, after_boundary),
        "rgba_hair_reconstruction_improvement_pct": _pct(before_hair, after_hair),
        "rgba_hand_reconstruction_improvement_pct": _pct(before_hand, after_hand),
        "rgba_cloth_reconstruction_improvement_pct": _pct(before_cloth, after_cloth),
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
