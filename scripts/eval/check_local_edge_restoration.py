import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan.utils.boundary_refinement import refine_boundary_frames


def _make_case(frame_count=8, height=144, width=176):
    bg = np.zeros((frame_count, height, width, 3), dtype=np.uint8)
    bg[..., 0] = np.linspace(12, 70, width, dtype=np.uint8)[None, None, :]
    bg[..., 1] = 42
    bg[..., 2] = 90

    gen = bg.copy()
    person_mask = np.zeros((frame_count, height, width), dtype=np.float32)
    soft_band = np.zeros_like(person_mask)
    soft_alpha = np.zeros_like(person_mask)
    uncertainty = np.zeros_like(person_mask)
    detail_release = np.zeros_like(person_mask)
    trimap_unknown = np.zeros_like(person_mask)
    edge_detail = np.zeros_like(person_mask)
    hair_boundary = np.zeros_like(person_mask)
    face_boundary = np.zeros_like(person_mask)
    cloth_boundary = np.zeros_like(person_mask)
    hand_boundary = np.zeros_like(person_mask)
    occluded_boundary = np.zeros_like(person_mask)

    yy, xx = np.mgrid[:height, :width]
    for idx in range(frame_count):
        cx = 52 + idx * 4
        cy = 74
        body = (((xx - cx) / 24.0) ** 2 + ((yy - cy) / 34.0) ** 2) <= 1.0
        band = ((((xx - cx) / 26.0) ** 2 + ((yy - cy) / 36.0) ** 2) <= 1.0) & (~body)
        hair = (((xx - cx) / 18.0) ** 2 + ((yy - (cy - 22)) / 10.0) ** 2) <= 1.0
        face = (((xx - cx) / 14.0) ** 2 + ((yy - (cy - 12)) / 12.0) ** 2) <= 1.0
        cloth = body & (yy >= cy)
        hand = (((xx - (cx + 19)) / 5.5) ** 2 + ((yy - cy) / 5.5) ** 2) <= 1.0

        person_mask[idx, body] = 1.0
        soft_band[idx, band] = 1.0
        soft_alpha[idx] = np.clip(person_mask[idx] + 0.60 * soft_band[idx], 0.0, 1.0)
        uncertainty[idx, band] = 0.05
        detail_release[idx, band] = 1.0
        trimap_unknown[idx, band] = 1.0
        edge_detail[idx, band] = 1.0
        hair_boundary[idx, hair & band] = 1.0
        face_boundary[idx, face & band] = 1.0
        cloth_boundary[idx, cloth & band] = 1.0
        hand_boundary[idx, hand & band] = 1.0

        gen[idx, body] = np.array([210, 170, 130], dtype=np.uint8)
        gen[idx, band] = np.array([150, 128, 110], dtype=np.uint8)
        gen[idx, hair & band] = np.array([118, 98, 82], dtype=np.uint8)
        gen[idx, face] = np.array([214, 178, 146], dtype=np.uint8)
    return {
        "generated": gen,
        "background": bg,
        "person_mask": person_mask,
        "soft_band": soft_band,
        "soft_alpha": soft_alpha,
        "uncertainty": uncertainty,
        "detail_release": detail_release,
        "trimap_unknown": trimap_unknown,
        "edge_detail": edge_detail,
        "face_boundary": face_boundary,
        "hair_boundary": hair_boundary,
        "cloth_boundary": cloth_boundary,
        "hand_boundary": hand_boundary,
        "occluded_boundary": occluded_boundary,
    }


def main():
    data = _make_case()
    roi_frames, roi_debug = refine_boundary_frames(
        generated_frames=data["generated"],
        background_frames=data["background"],
        person_mask=data["person_mask"],
        soft_band=data["soft_band"],
        soft_alpha=data["soft_alpha"],
        background_confidence=np.ones_like(data["person_mask"], dtype=np.float32),
        uncertainty_map=data["uncertainty"],
        detail_release_map=data["detail_release"],
        trimap_unknown_map=data["trimap_unknown"],
        edge_detail_map=data["edge_detail"],
        mode="roi_v1",
        strength=0.42,
        sharpen=0.24,
    )
    local_frames, local_debug = refine_boundary_frames(
        generated_frames=data["generated"],
        background_frames=data["background"],
        person_mask=data["person_mask"],
        soft_band=data["soft_band"],
        soft_alpha=data["soft_alpha"],
        background_confidence=np.ones_like(data["person_mask"], dtype=np.float32),
        uncertainty_map=data["uncertainty"],
        detail_release_map=data["detail_release"],
        trimap_unknown_map=data["trimap_unknown"],
        edge_detail_map=data["edge_detail"],
        face_boundary_map=data["face_boundary"],
        hair_boundary_map=data["hair_boundary"],
        hand_boundary_map=data["hand_boundary"],
        cloth_boundary_map=data["cloth_boundary"],
        occluded_boundary_map=data["occluded_boundary"],
        mode="local_edge_v1",
        strength=0.42,
        sharpen=0.24,
    )
    assert local_frames.shape == data["generated"].shape
    assert "local_edge_focus" in local_debug
    assert local_debug["metrics"]["roi_band_gradient_after_mean"] >= roi_debug["metrics"]["roi_band_gradient_after_mean"] * 0.99
    assert local_debug["metrics"]["roi_band_edge_contrast_after_mean"] >= roi_debug["metrics"]["roi_band_edge_contrast_after_mean"] * 1.0
    outside_roi = np.clip(1.0 - local_debug["roi_mask"], 0.0, 1.0)[..., None]
    outside_delta = np.abs(local_frames.astype(np.float32) - roi_frames.astype(np.float32)) * outside_roi
    assert float(outside_delta.mean()) < 6.0, "local_edge_v1 should stay local to ROI."
    print("Synthetic local edge restoration: PASS")


if __name__ == "__main__":
    main()
