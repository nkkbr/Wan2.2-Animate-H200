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
    h, w = 96, 96
    generated = np.zeros((2, h, w, 3), dtype=np.uint8)
    background = np.zeros_like(generated)
    person_mask = np.zeros((2, h, w), dtype=np.float32)
    soft_band = np.zeros_like(person_mask)
    person_mask[:, 24:72, 28:68] = 1.0
    soft_band[:, 20:76, 24:72] = 0.35
    uncertainty = np.zeros_like(person_mask)
    uncertainty[:, 20:76, 24:72] = 0.2
    detail = np.zeros_like(person_mask)
    detail[:, 22:74, 26:70] = 0.5
    refined, debug = refine_boundary_frames(
        generated_frames=generated,
        background_frames=background,
        person_mask=person_mask,
        soft_band=soft_band,
        uncertainty_map=uncertainty,
        detail_release_map=detail,
        trimap_unknown_map=detail,
        edge_detail_map=detail,
        mode="roi_gen_v1",
        strength=0.35,
        sharpen=0.15,
    )
    assert refined.shape == generated.shape
    assert "roi_mask" in debug
    assert float(np.asarray(debug["roi_mask"]).mean()) > 0.0
    print(json.dumps({"status": "PASS", "roi_area_ratio": float(np.asarray(debug["roi_mask"]).mean())}))


if __name__ == "__main__":
    main()
