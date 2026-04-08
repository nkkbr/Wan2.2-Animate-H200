import json
import tempfile
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan.utils.boundary_refinement import refine_boundary_frames


def _make_synthetic_case(frame_count=8, height=128, width=160):
    bg = np.zeros((frame_count, height, width, 3), dtype=np.uint8)
    bg[..., 0] = np.linspace(10, 60, width, dtype=np.uint8)[None, None, :]
    bg[..., 1] = 40
    bg[..., 2] = 90

    gen = bg.copy()
    person_mask = np.zeros((frame_count, height, width), dtype=np.float32)
    soft_band = np.zeros_like(person_mask)
    soft_alpha = np.zeros_like(person_mask)
    uncertainty = np.zeros_like(person_mask)
    occlusion = np.zeros_like(person_mask)
    face_preserve = np.zeros_like(person_mask)
    face_conf = np.zeros_like(person_mask)
    detail_release = np.zeros_like(person_mask)
    trimap_unknown = np.zeros_like(person_mask)
    edge_detail = np.zeros_like(person_mask)

    yy, xx = np.mgrid[:height, :width]
    for idx in range(frame_count):
        cx = 48 + idx * 4
        cy = 64
        ellipse = (((xx - cx) / 22.0) ** 2 + ((yy - cy) / 34.0) ** 2) <= 1.0
        band = ((((xx - cx) / 24.0) ** 2 + ((yy - cy) / 36.0) ** 2) <= 1.0) & (~ellipse)
        person_mask[idx, ellipse] = 1.0
        soft_band[idx, band] = 1.0
        soft_alpha[idx] = np.clip(person_mask[idx] + 0.55 * soft_band[idx], 0.0, 1.0)
        detail_release[idx, band] = 1.0
        trimap_unknown[idx, band] = 1.0
        edge_detail[idx, band] = 1.0
        uncertainty[idx, band] = 0.05
        gen[idx, ellipse] = np.array([210, 170, 130], dtype=np.uint8)
        gen[idx, band] = np.array([165, 140, 118], dtype=np.uint8)
    return gen, bg, person_mask, soft_band, soft_alpha, uncertainty, occlusion, face_preserve, face_conf, detail_release, trimap_unknown, edge_detail


def main():
    generated, background, person_mask, soft_band, soft_alpha, uncertainty, occlusion, face_preserve, face_confidence, detail_release, trimap_unknown, edge_detail = _make_synthetic_case()
    refined, debug = refine_boundary_frames(
        generated_frames=generated,
        background_frames=background,
        person_mask=person_mask,
        soft_band=soft_band,
        soft_alpha=soft_alpha,
        background_confidence=np.ones_like(person_mask, dtype=np.float32),
        uncertainty_map=uncertainty,
        occlusion_band=occlusion,
        face_preserve_map=face_preserve,
        face_confidence_map=face_confidence,
        detail_release_map=detail_release,
        trimap_unknown_map=trimap_unknown,
        edge_detail_map=edge_detail,
        structure_guard_strength=1.0,
        mode="roi_v1",
        strength=0.45,
        sharpen=0.20,
    )
    assert refined.shape == generated.shape
    assert "roi_mask" in debug and debug["roi_mask"].shape == person_mask.shape
    assert debug["metrics"]["roi_area_ratio"] > 0.0
    assert debug["metrics"]["roi_band_gradient_after_mean"] >= debug["metrics"]["roi_band_gradient_before_mean"]
    outside_roi = np.clip(1.0 - debug["roi_mask"], 0.0, 1.0)[..., None]
    outside_delta = np.abs(refined.astype(np.float32) - generated.astype(np.float32)) * outside_roi
    assert float(outside_delta.mean()) < 8.0, "ROI refine should not heavily disturb outside-ROI pixels."
    print("Synthetic boundary ROI refine: PASS")


if __name__ == "__main__":
    main()
