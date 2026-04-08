#!/usr/bin/env python
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan.utils.preprocess_candidate_selection import DEFAULT_SCORE_POLICY, score_candidates


def main():
    baseline = {
        "name": "default_precision",
        "is_default": True,
        "passed": True,
        "contract_passed": True,
        "metrics": {
            "boundary": {
                "uncertainty_transition_focus_ratio_dilated": 0.30,
                "uncertainty_transition_to_interior_ratio": 8.0,
                "uncertainty_mean": 0.05,
            },
            "face": {
                "center_jitter_mean": 2.5,
                "width_jitter_mean": 0.5,
                "valid_face_points_mean": 66.0,
                "landmark_confidence_mean": 0.90,
            },
            "pose": {
                "body_conf_delta_mean": 0.012,
                "body_jitter_mean": 0.00024,
                "hand_jitter_mean": 0.050,
                "velocity_spike_rate": 0.060,
                "limb_continuity_score": 0.80,
            },
            "background": {
                "background_stats": {
                    "temporal_fluctuation_mean": 1.10,
                    "band_adjacent_background_stability": 0.45,
                    "unresolved_ratio_mean": 0.80,
                },
                "support_confidence_corr": 0.65,
            },
            "runtime": {
                "stage_seconds": {"total": 80.0},
                "peak_memory_gb": 3.5,
            },
        },
    }
    improved = {
        "name": "face_motion_highres",
        "is_default": False,
        "passed": True,
        "contract_passed": True,
        "metrics": {
            "boundary": {
                "uncertainty_transition_focus_ratio_dilated": 0.44,
                "uncertainty_transition_to_interior_ratio": 12.0,
                "uncertainty_mean": 0.035,
            },
            "face": {
                "center_jitter_mean": 1.5,
                "width_jitter_mean": 0.22,
                "valid_face_points_mean": 69.0,
                "landmark_confidence_mean": 0.97,
            },
            "pose": {
                "body_conf_delta_mean": 0.009,
                "body_jitter_mean": 0.00017,
                "hand_jitter_mean": 0.041,
                "velocity_spike_rate": 0.045,
                "limb_continuity_score": 0.86,
            },
            "background": {
                "background_stats": {
                    "temporal_fluctuation_mean": 0.88,
                    "band_adjacent_background_stability": 0.29,
                    "unresolved_ratio_mean": 0.70,
                },
                "support_confidence_corr": 0.79,
            },
            "runtime": {
                "stage_seconds": {"total": 102.0},
                "peak_memory_gb": 4.0,
            },
        },
    }
    failed = {
        "name": "failed_candidate",
        "is_default": False,
        "passed": False,
        "contract_passed": False,
        "metrics": {},
    }

    result = score_candidates([baseline, improved, failed], json.loads(json.dumps(DEFAULT_SCORE_POLICY)))
    assert result["selected_candidate"] == "face_motion_highres"
    assert result["selected_better_than_default"] is True
    assert "failed_candidate" in result["invalid_candidates"]
    assert result["ranking"][0]["name"] == "face_motion_highres"
    print("Synthetic candidate selection: PASS")


if __name__ == "__main__":
    main()
