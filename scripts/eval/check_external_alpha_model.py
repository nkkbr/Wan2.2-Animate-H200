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


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--background_preprocess_dir", required=True)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    summary = _read_json(dataset_dir / "summary.json")
    record = summary["cases"][0]["records"][0]
    label_json = _read_json(Path(record["label_json_path"]))
    preprocess_idx = int(label_json["preprocess_frame_index"])
    image_rgb = _read_rgb(Path(record["image_path"]))
    bg_path = Path(args.background_preprocess_dir) / "src_bg" / f"{preprocess_idx:06d}.png"
    background_rgb = _read_rgb(bg_path)

    adapter = BackgroundMattingV2Adapter(BackgroundMattingV2Config(model_id=args.model_id))
    result = adapter.infer(image_rgb, background_rgb)
    alpha = result["alpha"]

    ok = (
        alpha.shape == image_rgb.shape[:2]
        and np.isfinite(alpha).all()
        and 0.0 <= float(alpha.min()) <= float(alpha.max()) <= 1.0
    )
    print(json.dumps({
        "model_id": args.model_id,
        "alpha_shape": list(alpha.shape),
        "alpha_min": float(alpha.min()),
        "alpha_max": float(alpha.max()),
        "runtime_sec": result["runtime_sec"],
        "synthetic_passed": bool(ok),
    }, ensure_ascii=False))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
