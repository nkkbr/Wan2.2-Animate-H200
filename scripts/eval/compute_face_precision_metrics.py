#!/usr/bin/env python
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description="Compute face precision proxy metrics for optimization3.")
    parser.add_argument("--src_root_path", required=True, type=str)
    parser.add_argument("--label_json", type=str, default=None)
    parser.add_argument("--output_json", required=True, type=str)
    args = parser.parse_args()

    src_root = Path(args.src_root_path)
    curve = _read_json(src_root / "face_bbox_curve.json")
    frames = curve.get("frames", [])
    centers = np.array([[f.get("center_x", 0.0), f.get("center_y", 0.0)] for f in frames], dtype=np.float32)
    widths = np.array([f.get("width", 0.0) for f in frames], dtype=np.float32)
    heights = np.array([f.get("height", 0.0) for f in frames], dtype=np.float32)
    valid_points = np.array([f.get("valid_face_points", 0.0) for f in frames], dtype=np.float32)

    center_step = np.linalg.norm(np.diff(centers, axis=0), axis=1) if len(centers) > 1 else np.zeros((0,), dtype=np.float32)
    width_step = np.abs(np.diff(widths)) if len(widths) > 1 else np.zeros((0,), dtype=np.float32)
    height_step = np.abs(np.diff(heights)) if len(heights) > 1 else np.zeros((0,), dtype=np.float32)

    result = {
        "mode": "proxy",
        "frame_count": len(frames),
        "center_jitter_mean": float(center_step.mean()) if len(center_step) else 0.0,
        "center_jitter_max": float(center_step.max()) if len(center_step) else 0.0,
        "width_jitter_mean": float(width_step.mean()) if len(width_step) else 0.0,
        "height_jitter_mean": float(height_step.mean()) if len(height_step) else 0.0,
        "valid_face_points_mean": float(valid_points.mean()) if len(valid_points) else 0.0,
        "valid_face_points_min": float(valid_points.min()) if len(valid_points) else 0.0,
        "source_counts": dict(Counter(f.get("source", "unknown") for f in frames)),
    }

    if args.label_json:
        label = _read_json(Path(args.label_json))
        result["mode"] = "labelled"
        result["label_available"] = True
        result["landmark_nme"] = None
    else:
        result["label_available"] = False

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

