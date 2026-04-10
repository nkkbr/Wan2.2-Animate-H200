#!/usr/bin/env python
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan.utils.renderable_foreground_proto import build_renderable_foreground_frame


def main() -> None:
    rng = np.random.default_rng(13)
    h, w = 48, 64
    source = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    background = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    alpha = np.clip(rng.normal(0.2, 0.25, size=(h, w)), 0.0, 1.0).astype(np.float32)
    fg = np.clip(source.astype(np.float32) - background.astype(np.float32) * (1.0 - alpha[..., None]), 0.0, 255.0).astype(np.uint8)
    soft_alpha = np.clip(alpha + rng.normal(0.0, 0.05, size=(h, w)), 0.0, 1.0).astype(np.float32)
    hard = (alpha > 0.7).astype(np.float32)
    boundary = np.clip(np.abs(np.gradient(alpha)[0]) + np.abs(np.gradient(alpha)[1]), 0.0, 1.0).astype(np.float32)
    roi = np.clip(boundary + (alpha > 0.1).astype(np.float32) * 0.3, 0.0, 1.0).astype(np.float32)
    conf = np.clip(rng.normal(0.8, 0.15, size=(h, w)), 0.0, 1.0).astype(np.float32)
    occ = (rng.random((h, w)) > 0.97).astype(np.float32)
    unresolved = (rng.random((h, w)) > 0.95).astype(np.float32)
    uncertainty = np.clip(rng.normal(0.15, 0.1, size=(h, w)), 0.0, 1.0).astype(np.float32)

    out = build_renderable_foreground_frame(
        source_rgb=source,
        background_rgb=background,
        foreground_rgb=fg,
        foreground_alpha=alpha,
        soft_alpha=soft_alpha,
        hard_foreground=hard,
        boundary_band=boundary,
        composite_roi_mask=roi,
        foreground_confidence=conf,
        occlusion_band=occ,
        background_unresolved=unresolved,
        uncertainty_map=uncertainty,
        prev_state=None,
        mode="round1",
    )

    required = {
        "render_rgb",
        "render_alpha",
        "render_depth",
        "render_person_mask",
        "render_silhouette_band",
        "render_composite_rgb",
        "render_roi_mask",
    }
    missing = required.difference(out)
    if missing:
        raise SystemExit(f"Missing outputs: {sorted(missing)}")
    if out["render_rgb"].shape != source.shape:
        raise SystemExit("render_rgb shape mismatch")
    if out["render_alpha"].shape != alpha.shape:
        raise SystemExit("render_alpha shape mismatch")
    print("Renderable foreground proto synthetic checks: PASS")


if __name__ == "__main__":
    main()
