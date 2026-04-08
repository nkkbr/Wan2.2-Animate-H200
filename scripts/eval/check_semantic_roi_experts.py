#!/usr/bin/env python
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan.utils.boundary_refinement import refine_boundary_frames


def main():
    t, h, w = 2, 96, 96
    generated = np.full((t, h, w, 3), 128, dtype=np.uint8)
    background = np.full((t, h, w, 3), 96, dtype=np.uint8)
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    person_mask = (((xx - 48) ** 2 + (yy - 48) ** 2) < 26 ** 2).astype(np.float32)
    person_mask = np.stack([person_mask, np.roll(person_mask, 2, axis=1)], axis=0)
    outer = np.clip(person_mask - np.stack([np.pad(person_mask[0, 2:-2, 2:-2], 2), np.pad(person_mask[1, 2:-2, 2:-2], 2)], axis=0), 0.0, 1.0)
    soft_alpha = np.clip(person_mask * 0.85, 0.0, 1.0)
    zeros = np.zeros_like(person_mask, dtype=np.float32)

    face = zeros.copy()
    face[:, 22:42, 34:62] = outer[:, 22:42, 34:62]
    hair = zeros.copy()
    hair[:, 16:34, 28:68] = outer[:, 16:34, 28:68]
    hand = zeros.copy()
    hand[:, 46:68, 18:34] = outer[:, 46:68, 18:34]
    cloth = zeros.copy()
    cloth[:, 52:82, 28:72] = outer[:, 52:82, 28:72]

    refined, debug = refine_boundary_frames(
        generated_frames=generated,
        background_frames=background,
        person_mask=person_mask,
        soft_band=outer,
        soft_alpha=soft_alpha,
        background_confidence=np.ones_like(person_mask, dtype=np.float32),
        uncertainty_map=zeros,
        occlusion_band=zeros,
        face_preserve_map=face * 0.8,
        face_confidence_map=np.ones_like(person_mask, dtype=np.float32) * 0.9,
        detail_release_map=cloth * 0.6,
        trimap_unknown_map=hair * 0.7,
        edge_detail_map=cloth * 0.5 + hand * 0.4,
        face_boundary_map=face,
        hair_boundary_map=hair,
        hand_boundary_map=hand,
        cloth_boundary_map=cloth,
        occluded_boundary_map=zeros,
        mode="semantic_experts_v1",
        strength=0.35,
        sharpen=0.15,
    )
    payload = {
        "status": "PASS",
        "refined_shape": list(refined.shape),
        "roi_area_ratio": float(debug.get("roi_area_ratio", 0.0)),
        "face_coverage_mean": float(debug["metrics"].get("face_coverage_mean", 0.0)),
        "hair_coverage_mean": float(debug["metrics"].get("hair_coverage_mean", 0.0)),
        "hand_coverage_mean": float(debug["metrics"].get("hand_coverage_mean", 0.0)),
        "cloth_coverage_mean": float(debug["metrics"].get("cloth_coverage_mean", 0.0)),
    }
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
