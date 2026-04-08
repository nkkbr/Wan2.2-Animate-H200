#!/usr/bin/env python
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
PREPROCESS_DIR = REPO_ROOT / "wan" / "modules" / "animate" / "preprocess"
if str(PREPROCESS_DIR) not in sys.path:
    sys.path.insert(0, str(PREPROCESS_DIR))

from matting_adapter import run_matting_adapter  # noqa: E402


def _make_synthetic_case():
    height, width = 96, 96
    frame = np.full((height, width, 3), 32, dtype=np.uint8)
    frame[:, width // 2 :, :] = np.array([200, 180, 170], dtype=np.uint8)
    for x in range(26, 34):
        frame[10:55, x, :] = np.array([220, 210, 200], dtype=np.uint8)
    frame[28:68, 36:62, :] = np.array([210, 185, 168], dtype=np.uint8)

    hard_mask = np.zeros((1, height, width), dtype=np.float32)
    hard_mask[0, 28:68, 36:62] = 1.0
    soft_band = np.zeros_like(hard_mask)
    soft_band[0, 24:72, 30:66] = 0.5

    parsing_boundary = np.zeros_like(hard_mask)
    parsing_boundary[0, 24:72, 30:66] = 0.25
    head_prior = np.zeros_like(hard_mask)
    head_prior[0, 8:56, 22:42] = 1.0
    hand_prior = np.zeros_like(hard_mask)
    hand_prior[0, 48:60, 58:70] = 1.0
    occlusion_prior = np.zeros_like(hard_mask)
    part_foreground = np.maximum(head_prior, hand_prior)

    return (
        frame[None, ...],
        hard_mask,
        soft_band,
        parsing_boundary,
        head_prior,
        hand_prior,
        occlusion_prior,
        part_foreground,
    )


def main():
    frames, hard_mask, soft_band, parsing_boundary, head_prior, hand_prior, occlusion_prior, part_foreground = _make_synthetic_case()
    result = run_matting_adapter(
        frames=frames,
        hard_mask=hard_mask,
        soft_band=soft_band,
        parsing_boundary_prior=parsing_boundary,
        parsing_head_prior=head_prior,
        parsing_hand_prior=hand_prior,
        parsing_occlusion_prior=occlusion_prior,
        parsing_part_foreground_prior=part_foreground,
        mode="high_precision_v2",
    )

    assert result["soft_alpha"].shape == hard_mask.shape
    assert result["alpha_v2"].shape == hard_mask.shape
    assert result["trimap_v2"].shape == hard_mask.shape
    assert result["alpha_uncertainty_v2"].shape == hard_mask.shape
    assert result["fine_boundary_mask"].shape == hard_mask.shape
    assert result["hair_edge_mask"].shape == hard_mask.shape
    assert result["alpha_confidence"].shape == hard_mask.shape
    assert result["refined_hard_foreground"].shape == hard_mask.shape
    assert float(result["fine_boundary_mask"].mean()) > 0.0
    assert float(result["hair_edge_mask"].mean()) > 0.0
    assert float(result["alpha_confidence"].mean()) > 0.0
    assert np.isfinite(result["soft_alpha"]).all()
    assert np.isfinite(result["alpha_uncertainty_v2"]).all()
    print(json.dumps({
        "soft_alpha_mean": float(result["soft_alpha"].mean()),
        "alpha_uncertainty_v2_mean": float(result["alpha_uncertainty_v2"].mean()),
        "fine_boundary_mask_mean": float(result["fine_boundary_mask"].mean()),
        "hair_edge_mask_mean": float(result["hair_edge_mask"].mean()),
        "status": "PASS",
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
