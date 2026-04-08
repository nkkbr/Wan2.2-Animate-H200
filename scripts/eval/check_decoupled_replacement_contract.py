#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan.utils.animate_contract import resolve_preprocess_artifacts
from wan.utils.media_io import load_mask_artifact, load_rgb_artifact


def main():
    parser = argparse.ArgumentParser(description="Validate optimization5 Step03 decoupled replacement contract artifacts.")
    parser.add_argument("--src_root_path", type=str, required=True)
    args = parser.parse_args()

    src_root = Path(args.src_root_path).resolve()
    artifacts, metadata = resolve_preprocess_artifacts(src_root, replace_flag=True)
    if metadata is None:
        raise SystemExit("metadata.json is required for decoupled contract validation.")

    required = [
        "foreground_rgb",
        "foreground_alpha",
        "foreground_confidence",
        "background_rgb",
        "background_visible_support",
        "background_unresolved",
        "composite_roi_mask",
    ]
    for key in required:
        if key not in metadata["src_files"]:
            raise SystemExit(f"Missing decoupled contract artifact in metadata: {key}")

    fg_rgb = load_rgb_artifact(artifacts["foreground_rgb"]["path"], artifacts["foreground_rgb"].get("format"))
    fg_alpha = load_mask_artifact(artifacts["foreground_alpha"]["path"], artifacts["foreground_alpha"].get("format"))
    fg_conf = load_mask_artifact(artifacts["foreground_confidence"]["path"], artifacts["foreground_confidence"].get("format"))
    bg_rgb = load_rgb_artifact(artifacts["background_rgb"]["path"], artifacts["background_rgb"].get("format"))
    vis = load_mask_artifact(artifacts["background_visible_support"]["path"], artifacts["background_visible_support"].get("format"))
    unres = load_mask_artifact(artifacts["background_unresolved"]["path"], artifacts["background_unresolved"].get("format"))
    composite = load_mask_artifact(artifacts["composite_roi_mask"]["path"], artifacts["composite_roi_mask"].get("format"))

    frame_count = fg_rgb.shape[0]
    expected_hw = fg_rgb.shape[1:3]
    for name, arr in {
        "foreground_alpha": fg_alpha,
        "foreground_confidence": fg_conf,
        "background_visible_support": vis,
        "background_unresolved": unres,
        "composite_roi_mask": composite,
    }.items():
        if arr.shape[0] != frame_count:
            raise SystemExit(f"{name} frame count mismatch: {arr.shape[0]} vs {frame_count}")
        if arr.shape[1:3] != expected_hw:
            raise SystemExit(f"{name} size mismatch: {arr.shape[1:3]} vs {expected_hw}")
    if bg_rgb.shape[0] != frame_count or bg_rgb.shape[1:3] != expected_hw:
        raise SystemExit("background_rgb shape mismatch with foreground_rgb")

    payload = {
        "status": "PASS",
        "frame_count": int(frame_count),
        "height": int(expected_hw[0]),
        "width": int(expected_hw[1]),
        "foreground_alpha_mean": float(np.mean(fg_alpha)),
        "foreground_confidence_mean": float(np.mean(fg_conf)),
        "background_visible_support_mean": float(np.mean(vis)),
        "background_unresolved_mean": float(np.mean(unres)),
        "composite_roi_mask_mean": float(np.mean(composite)),
    }
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
