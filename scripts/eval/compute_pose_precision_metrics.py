#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import numpy as np


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description="Compute pose precision proxy metrics for optimization3.")
    parser.add_argument("--src_root_path", required=True, type=str)
    parser.add_argument("--label_json", type=str, default=None)
    parser.add_argument("--output_json", required=True, type=str)
    args = parser.parse_args()

    src_root = Path(args.src_root_path)
    curve = _read_json(src_root / "pose_conf_curve.json")
    frames = curve.get("frames", [])
    aggregate = curve.get("aggregate_stats", {})

    raw_body = np.array([f.get("raw_body_mean_conf", 0.0) for f in frames], dtype=np.float32)
    raw_face = np.array([f.get("raw_face_mean_conf", 0.0) for f in frames], dtype=np.float32)
    raw_hand = np.array([f.get("raw_hand_mean_conf", 0.0) for f in frames], dtype=np.float32)
    smooth_body = np.array([f.get("smoothed_body_mean_conf", 0.0) for f in frames], dtype=np.float32)

    body_delta = np.abs(np.diff(smooth_body)) if len(smooth_body) > 1 else np.zeros((0,), dtype=np.float32)
    result = {
        "mode": "proxy",
        "frame_count": len(frames),
        "raw_body_mean_conf": float(raw_body.mean()) if len(raw_body) else 0.0,
        "raw_face_mean_conf": float(raw_face.mean()) if len(raw_face) else 0.0,
        "raw_hand_mean_conf": float(raw_hand.mean()) if len(raw_hand) else 0.0,
        "smoothed_body_mean_conf": float(smooth_body.mean()) if len(smooth_body) else 0.0,
        "body_conf_delta_mean": float(body_delta.mean()) if len(body_delta) else 0.0,
        "aggregate_stats": aggregate,
    }

    if args.label_json:
        result["mode"] = "labelled"
        result["pck"] = None
        result["label_available"] = True
    else:
        result["label_available"] = False

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

