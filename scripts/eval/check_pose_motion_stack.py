#!/usr/bin/env python
import tempfile
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
PREPROCESS_ROOT = REPO_ROOT / "wan" / "modules" / "animate" / "preprocess"
if str(PREPROCESS_ROOT) not in sys.path:
    sys.path.insert(0, str(PREPROCESS_ROOT))

from pose_motion_analysis import (
    run_pose_motion_stack,
    write_pose_motion_artifacts,
)


def _make_meta(body_points, left_hand, right_hand, face):
    return {
        "keypoints_body": np.asarray(body_points, dtype=np.float32),
        "keypoints_left_hand": np.asarray(left_hand, dtype=np.float32),
        "keypoints_right_hand": np.asarray(right_hand, dtype=np.float32),
        "keypoints_face": np.asarray(face, dtype=np.float32),
        "width": 128,
        "height": 96,
    }


def _build_sequence():
    metas = []
    raw_metas = []
    frames = []
    rng = np.random.default_rng(0)
    for frame_index in range(8):
        canvas = np.zeros((96, 128, 3), dtype=np.uint8)
        frames.append(canvas)
        t = frame_index / 7.0
        body = np.zeros((20, 3), dtype=np.float32)
        body[1] = [0.50, 0.28 + 0.01 * np.sin(3 * t), 0.98]
        body[2] = [0.42 + 0.01 * t, 0.32, 0.95]
        body[3] = [0.36 + 0.01 * t, 0.42, 0.93]
        body[4] = [0.30 + 0.01 * t, 0.54, 0.90]
        body[5] = [0.58 + 0.01 * t, 0.32, 0.95]
        body[6] = [0.65 + 0.01 * t, 0.42, 0.93]
        body[7] = [0.72 + 0.01 * t, 0.55 + 0.02 * np.sin(5 * t), 0.88]
        body[8] = [0.45, 0.55, 0.95]
        body[9] = [0.44, 0.74, 0.92]
        body[10] = [0.43, 0.90, 0.90]
        body[11] = [0.55, 0.55, 0.95]
        body[12] = [0.57, 0.74, 0.92]
        body[13] = [0.59, 0.90, 0.90]
        body[:, :2] += rng.normal(scale=0.01, size=body[:, :2].shape).astype(np.float32)
        body[:, 2] = np.maximum(body[:, 2], 0.0)

        left_hand = np.zeros((21, 3), dtype=np.float32)
        right_hand = np.zeros((21, 3), dtype=np.float32)
        for idx in range(21):
            left_hand[idx] = [0.29 + 0.015 * idx / 20.0, 0.54 + 0.01 * np.sin(t * 6 + idx), 0.82]
            right_hand[idx] = [0.73 + 0.015 * idx / 20.0, 0.54 + 0.02 * np.cos(t * 7 + idx), 0.82]
        left_hand[:, :2] += rng.normal(scale=0.015, size=left_hand[:, :2].shape).astype(np.float32)
        right_hand[:, :2] += rng.normal(scale=0.018, size=right_hand[:, :2].shape).astype(np.float32)
        if frame_index == 4:
            right_hand[:, 2] = 0.15

        face = np.zeros((69, 3), dtype=np.float32)
        for idx in range(69):
            angle = 2 * np.pi * idx / 69.0
            face[idx] = [0.50 + 0.06 * np.cos(angle), 0.18 + 0.08 * np.sin(angle), 0.92]
        face[:, :2] += rng.normal(scale=0.003, size=face[:, :2].shape).astype(np.float32)

        raw_metas.append(_make_meta(body, left_hand, right_hand, face))
        metas.append(_make_meta(body, left_hand, right_hand, face))

    occlusion = np.zeros((8, 96, 128), dtype=np.float32)
    occlusion[4, 46:66, 84:116] = 0.8
    uncertainty = np.zeros_like(occlusion)
    return np.stack(frames), metas, raw_metas, occlusion, uncertainty


def main():
    frames, metas, raw_metas, occlusion, uncertainty = _build_sequence()
    baseline = run_pose_motion_stack(
        export_frames=frames,
        pose_metas=metas,
        raw_pose_metas=raw_metas,
        image_shape=(96, 128),
        occlusion_band=occlusion,
        uncertainty_map=uncertainty,
        mode="none",
    )
    refined = run_pose_motion_stack(
        export_frames=frames,
        pose_metas=metas,
        raw_pose_metas=raw_metas,
        image_shape=(96, 128),
        occlusion_band=occlusion,
        uncertainty_map=uncertainty,
        mode="v1",
    )

    assert refined["stats"]["body_jitter_mean"] <= baseline["stats"]["body_jitter_mean"], "body jitter should improve."
    assert refined["stats"]["hand_jitter_mean"] <= baseline["stats"]["hand_jitter_mean"], "hand jitter should improve."
    assert refined["stats"]["limb_roi_coverage_ratio"] >= 0.95, "limb ROI coverage should stay high."
    assert refined["stats"]["hand_roi_coverage_ratio"] >= 0.95, "hand ROI coverage should stay high."
    assert refined["pose_uncertainty"][4].mean() > refined["pose_uncertainty"][0].mean(), "occluded frame should have higher uncertainty."

    with tempfile.TemporaryDirectory() as tmpdir:
        artifacts = write_pose_motion_artifacts(tmpdir, refined, fps=5, write_json_fn=lambda path, payload: Path(path).write_text(__import__("json").dumps(payload, indent=2), encoding="utf-8"))
        for key in ("pose_tracks", "limb_tracks", "hand_tracks", "pose_visibility", "pose_uncertainty"):
            assert key in artifacts
            assert (Path(tmpdir) / artifacts[key]["path"]).exists()

    print("Synthetic pose motion stack: PASS")


if __name__ == "__main__":
    main()
