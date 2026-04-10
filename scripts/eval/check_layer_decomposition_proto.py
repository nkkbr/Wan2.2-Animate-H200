#!/usr/bin/env python
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan.utils.layer_decomposition_proto import build_layer_roi_mask, decompose_layers


def main() -> None:
    rng = np.random.default_rng(1234)
    h, w = 96, 128
    source_rgb = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
    background_rgb = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
    foreground_alpha = np.clip(rng.normal(0.2, 0.25, size=(h, w)), 0.0, 1.0).astype(np.float32)
    foreground_confidence = np.clip(rng.normal(0.8, 0.15, size=(h, w)), 0.0, 1.0).astype(np.float32)
    foreground_rgb = np.clip(source_rgb.astype(np.float32) * foreground_alpha[..., None], 0.0, 255.0).astype(np.uint8)
    visible_support = np.clip(rng.normal(0.7, 0.2, size=(h, w)), 0.0, 1.0).astype(np.float32)
    unresolved = np.clip(rng.normal(0.15, 0.18, size=(h, w)), 0.0, 1.0).astype(np.float32)
    composite_roi = np.clip(rng.normal(0.10, 0.20, size=(h, w)), 0.0, 1.0).astype(np.float32)
    occlusion_band = np.clip(rng.normal(0.08, 0.16, size=(h, w)), 0.0, 1.0).astype(np.float32)
    occluded_boundary = np.clip(rng.normal(0.06, 0.12, size=(h, w)), 0.0, 1.0).astype(np.float32)
    uncertainty = np.clip(rng.normal(0.10, 0.15, size=(h, w)), 0.0, 1.0).astype(np.float32)

    roi = build_layer_roi_mask(
        composite_roi_mask=composite_roi,
        background_unresolved=unresolved,
        occlusion_band=occlusion_band,
        occluded_boundary=occluded_boundary,
        uncertainty_map=uncertainty,
    )
    result = decompose_layers(
        source_rgb=source_rgb,
        foreground_rgb=foreground_rgb,
        foreground_alpha=foreground_alpha,
        foreground_confidence=foreground_confidence,
        background_rgb=background_rgb,
        background_visible_support=visible_support,
        background_unresolved=unresolved,
        composite_roi_mask=composite_roi,
        occlusion_band=occlusion_band,
        occluded_boundary=occluded_boundary,
        uncertainty_map=uncertainty,
        mode="round1",
        occlusion_strength=0.65,
        alpha_mix=0.40,
        residual_mix=0.35,
    )
    payload = {
        "roi_mean": float(roi.mean()),
        "occlusion_alpha_mean": float(result["occlusion_alpha"].mean()),
        "total_alpha_mean": float(result["total_alpha"].mean()),
        "person_mask_mean": float(result["person_mask"].mean()),
        "trimap_unknown_mean": float(result["trimap_unknown"].mean()),
        "composite_range": [int(result["composite_rgb"].min()), int(result["composite_rgb"].max())],
        "occlusion_rgb_mean": float(result["occlusion_rgb"].mean()),
    }
    if result["composite_rgb"].shape != source_rgb.shape:
        raise SystemExit("Composite shape mismatch.")
    if result["total_alpha"].shape != foreground_alpha.shape:
        raise SystemExit("Alpha shape mismatch.")
    if not (0.0 <= payload["total_alpha_mean"] <= 1.0):
        raise SystemExit("Alpha mean out of range.")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
