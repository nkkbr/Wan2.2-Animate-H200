from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan.utils.clip_blending import (
    blend_clip_overlap,
    mean_abs_difference,
    write_seam_debug_artifacts,
    write_seam_summary,
)


def main():
    overlap_len = 5
    height = width = 48

    prev_overlap = torch.full((1, 3, overlap_len, height, width), -0.55, dtype=torch.float32)
    curr_overlap = torch.full((1, 3, overlap_len, height, width), -0.05, dtype=torch.float32)
    prev_overlap[:, :, :, 14:34, 14:34] = 0.55
    curr_overlap[:, :, :, 14:34, 14:34] = 0.75

    hard_background_keep = torch.ones((overlap_len, height, width), dtype=torch.float32)
    transition_band = torch.zeros((overlap_len, height, width), dtype=torch.float32)
    free_replacement = torch.zeros((overlap_len, height, width), dtype=torch.float32)
    free_replacement[:, 16:32, 16:32] = 1.0
    hard_background_keep[:, 16:32, 16:32] = 0.0
    transition_band[:, 14:34, 14:34] = 1.0
    transition_band[:, 16:32, 16:32] = 0.0
    hard_background_keep[:, 14:34, 14:34] = 0.0

    linear = blend_clip_overlap(prev_overlap, curr_overlap, mode="linear")
    mask_aware = blend_clip_overlap(
        prev_overlap,
        curr_overlap,
        mode="mask_aware",
        pixel_regions={
            "hard_background_keep": hard_background_keep,
            "transition_band": transition_band,
            "free_replacement": free_replacement,
        },
        background_current_strength=0.2,
    )

    background_distance_linear = mean_abs_difference(linear["blended"], prev_overlap, mask=hard_background_keep)
    background_distance_mask = mean_abs_difference(mask_aware["blended"], prev_overlap, mask=hard_background_keep)
    person_distance_linear = mean_abs_difference(linear["blended"], curr_overlap, mask=free_replacement)
    person_distance_mask = mean_abs_difference(mask_aware["blended"], curr_overlap, mask=free_replacement)

    assert background_distance_mask < background_distance_linear
    assert abs(person_distance_mask - person_distance_linear) < 1e-6
    assert mask_aware["stats"]["alpha_background_mean"] < mask_aware["stats"]["alpha_person_mean"]
    assert mask_aware["stats"]["alpha_background_mean"] < mask_aware["stats"]["alpha_transition_mean"]
    assert mask_aware["stats"]["alpha_transition_mean"] <= mask_aware["stats"]["alpha_person_mean"] + 1e-6

    with tempfile.TemporaryDirectory() as tmpdir:
        seam_stat = {
            "seam_index": 1,
            "boundary_score_before": 0.2,
            "boundary_score_after": 0.1,
            **mask_aware["stats"],
        }
        debug_paths = write_seam_debug_artifacts(
            save_debug_dir=tmpdir,
            seam_index=1,
            fps=30.0,
            prev_overlap=prev_overlap,
            curr_overlap=curr_overlap,
            blended_overlap=mask_aware["blended"],
            alpha_map=mask_aware["alpha_map"],
        )
        summary_path = write_seam_summary(tmpdir, [seam_stat])
        assert Path(debug_paths["comparison"]).exists()
        assert Path(debug_paths["alpha"]).exists()
        assert Path(summary_path).exists()

    print("Synthetic clip overlap blending: PASS")
    print("Synthetic seam debug export: PASS")


if __name__ == "__main__":
    main()
