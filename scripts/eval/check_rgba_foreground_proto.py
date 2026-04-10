#!/usr/bin/env python
import numpy as np
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan.utils.rgba_foreground_proto import build_rgba_foreground


def main() -> None:
    rng = np.random.default_rng(7)
    h, w = 48, 64
    source = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    bg = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    fg_alpha = np.clip(rng.normal(0.2, 0.25, size=(h, w)), 0.0, 1.0).astype(np.float32)
    soft_alpha = np.clip(fg_alpha + rng.normal(0.0, 0.04, size=(h, w)), 0.0, 1.0).astype(np.float32)
    fg = np.clip(source.astype(np.float32) - bg.astype(np.float32) * (1.0 - fg_alpha[..., None]), 0.0, 255.0).astype(np.uint8)
    hard = (fg_alpha > 0.7).astype(np.float32)
    boundary = np.clip(np.abs(np.gradient(fg_alpha)[0]) + np.abs(np.gradient(fg_alpha)[1]), 0.0, 1.0).astype(np.float32)
    trimap = ((fg_alpha > 0.05) & (fg_alpha < 0.95)).astype(np.float32)
    roi = np.clip(boundary + trimap, 0.0, 1.0).astype(np.float32)
    conf = np.clip(rng.normal(0.7, 0.2, size=(h, w)), 0.0, 1.0).astype(np.float32)
    visible = np.clip(rng.normal(0.8, 0.15, size=(h, w)), 0.0, 1.0).astype(np.float32)
    unresolved = np.clip(rng.normal(0.1, 0.15, size=(h, w)), 0.0, 1.0).astype(np.float32)
    hair_alpha = np.clip(fg_alpha + rng.normal(0.0, 0.03, size=(h, w)), 0.0, 1.0).astype(np.float32)
    hair = (trimap > 0).astype(np.float32) * 0.6
    hand = (rng.random((h, w)) > 0.96).astype(np.float32)
    cloth = (rng.random((h, w)) > 0.92).astype(np.float32)
    uncertainty = np.clip(rng.normal(0.15, 0.12, size=(h, w)), 0.0, 1.0).astype(np.float32)

    out = build_rgba_foreground(
        source_rgb=source,
        foreground_rgb=fg,
        foreground_alpha=fg_alpha,
        soft_alpha=soft_alpha,
        hard_foreground=hard,
        boundary_band=boundary,
        trimap_unknown=trimap,
        composite_roi_mask=roi,
        foreground_confidence=conf,
        background_rgb=bg,
        background_visible_support=visible,
        background_unresolved=unresolved,
        hair_alpha=hair_alpha,
        hair_boundary=hair,
        hand_boundary=hand,
        cloth_boundary=cloth,
        uncertainty_map=uncertainty,
        mode="round1",
    )

    required = {
        "rgba_roi_mask",
        "rgba_foreground_alpha",
        "rgba_foreground_rgb",
        "rgba_composite_rgb",
        "rgba_boundary_band",
        "rgba_trimap_unknown",
        "rgba_person_mask",
    }
    missing = required.difference(out)
    if missing:
        raise SystemExit(f"Missing outputs: {sorted(missing)}")
    if out["rgba_foreground_rgb"].shape != source.shape:
        raise SystemExit("Foreground RGB shape mismatch.")
    if out["rgba_foreground_alpha"].shape != fg_alpha.shape:
        raise SystemExit("Foreground alpha shape mismatch.")
    if out["rgba_composite_rgb"].dtype != np.uint8:
        raise SystemExit("Composite output must be uint8.")
    print("RGBA foreground proto synthetic checks: PASS")


if __name__ == "__main__":
    main()
