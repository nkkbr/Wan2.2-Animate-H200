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
    h, w = 112, 112
    generated = np.zeros((2, h, w, 3), dtype=np.uint8)
    background = np.zeros_like(generated)
    background[:, :, :, 1] = 18
    person_mask = np.zeros((2, h, w), dtype=np.float32)
    soft_band = np.zeros_like(person_mask)
    uncertainty = np.zeros_like(person_mask)
    detail = np.zeros_like(person_mask)
    hair = np.zeros_like(person_mask)
    face = np.zeros_like(person_mask)
    cloth = np.zeros_like(person_mask)

    person_mask[:, 28:84, 34:78] = 1.0
    soft_band[:, 22:90, 28:84] = 0.40
    uncertainty[:, 22:90, 28:84] = 0.18
    detail[:, 26:88, 30:82] = 0.55
    hair[:, 18:40, 34:78] = 0.75
    face[:, 32:58, 40:72] = 0.65
    cloth[:, 58:88, 34:78] = 0.55

    refined, debug = refine_boundary_frames(
        generated_frames=generated,
        background_frames=background,
        person_mask=person_mask,
        soft_band=soft_band,
        uncertainty_map=uncertainty,
        detail_release_map=detail,
        trimap_unknown_map=detail,
        edge_detail_map=detail,
        face_boundary_map=face,
        hair_boundary_map=hair,
        cloth_boundary_map=cloth,
        mode="roi_gen_v2",
        strength=0.42,
        sharpen=0.22,
    )
    assert refined.shape == generated.shape
    assert "roi_mask" in debug
    assert "roi_boxes" in debug
    assert float(np.asarray(debug["roi_mask"]).mean()) > 0.0
    print(json.dumps({
        "status": "PASS",
        "roi_area_ratio": float(np.asarray(debug["roi_mask"]).mean()),
        "box_count": len(debug["roi_boxes"]),
    }))


if __name__ == "__main__":
    main()
