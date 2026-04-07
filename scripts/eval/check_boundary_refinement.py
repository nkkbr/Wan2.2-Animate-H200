from pathlib import Path
import sys
import tempfile

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan.utils.boundary_refinement import (
    refine_boundary_frames,
    write_boundary_refinement_debug_artifacts,
)
from wan.utils.replacement_masks import build_soft_boundary_band


def _make_synthetic_case(frame_count: int = 6, height: int = 96, width: int = 128):
    background = np.full((frame_count, height, width, 3), 56, dtype=np.uint8)
    generated = background.copy()
    generated[:, 24:72, 40:88] = np.array([214, 188, 172], dtype=np.uint8)
    person_mask = np.zeros((frame_count, height, width), dtype=np.float32)
    person_mask[:, 24:72, 40:88] = 1.0
    soft_band = build_soft_boundary_band(person_mask, band_width=8, blur_kernel_size=5)

    for index in range(frame_count):
        generated[index] = cv2.GaussianBlur(generated[index], (7, 7), sigmaX=1.4)
        halo = np.repeat((soft_band[index] * 52.0).astype(np.uint8)[:, :, None], repeats=3, axis=2)
        generated[index] = np.clip(generated[index].astype(np.int16) + halo.astype(np.int16), 0, 255).astype(np.uint8)

    return generated, background, person_mask, soft_band


def main():
    generated, background, person_mask, soft_band = _make_synthetic_case()

    refined_zero, debug_zero = refine_boundary_frames(
        generated_frames=generated,
        background_frames=background,
        person_mask=person_mask,
        soft_band=soft_band,
        strength=0.0,
        sharpen=0.0,
    )
    assert np.array_equal(refined_zero, generated)
    assert debug_zero["metrics"]["band_mad_mean"] == 0.0

    refined, debug = refine_boundary_frames(
        generated_frames=generated,
        background_frames=background,
        person_mask=person_mask,
        soft_band=soft_band,
        strength=0.75,
        sharpen=0.35,
    )
    metrics = debug["metrics"]
    assert metrics["band_gradient_after_mean"] > metrics["band_gradient_before_mean"]
    assert metrics["halo_ratio_after"] < metrics["halo_ratio_before"]
    assert metrics["band_edge_contrast_after_mean"] < metrics["band_edge_contrast_before_mean"]
    assert metrics["band_mad_mean"] > 0.0

    with tempfile.TemporaryDirectory() as tmpdir:
        artifacts = write_boundary_refinement_debug_artifacts(
            save_debug_dir=tmpdir,
            fps=12.0,
            generated_frames=generated,
            refined_frames=refined,
            background_frames=background,
            person_mask=person_mask,
            debug_data=debug,
        )
        assert Path(artifacts["comparison"]).exists()
        assert Path(artifacts["outer_band"]).exists()
        assert Path(artifacts["inner_band"]).exists()
        assert Path(artifacts["metrics"]).exists()
        assert Path(artifacts["crops_dir"]).is_dir()

    print("Synthetic boundary refinement zero-strength passthrough: PASS")
    print("Synthetic boundary refinement edge metrics: PASS")
    print("Synthetic boundary refinement debug export: PASS")


if __name__ == "__main__":
    main()
