import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
PREPROCESS_ROOT = REPO_ROOT / "wan" / "modules" / "animate" / "preprocess"
if str(PREPROCESS_ROOT) not in sys.path:
    sys.path.insert(0, str(PREPROCESS_ROOT))

from wan.modules.animate.preprocess.signal_stabilization import (
    stabilize_face_bboxes,
    stabilize_pose_metas,
)


def build_synthetic_metas(num_frames=24):
    rng = np.random.default_rng(7)
    metas = []
    base_body = np.stack([np.linspace(0.2, 0.8, 20), np.linspace(0.2, 0.85, 20)], axis=1)
    base_face = np.stack([
        np.full(69, 0.5) + np.linspace(-0.08, 0.08, 69),
        np.full(69, 0.32) + np.sin(np.linspace(0, np.pi, 69)) * 0.04,
    ], axis=1)
    base_hand_left = np.stack([np.linspace(0.28, 0.38, 21), np.linspace(0.45, 0.58, 21)], axis=1)
    base_hand_right = np.stack([np.linspace(0.62, 0.72, 21), np.linspace(0.45, 0.58, 21)], axis=1)

    for frame_index in range(num_frames):
        jitter_scale = 0.018 if frame_index % 5 == 0 else 0.008
        body = np.concatenate(
            [
                base_body + rng.normal(0.0, jitter_scale, size=base_body.shape),
                np.full((20, 1), 0.92, dtype=np.float32),
            ],
            axis=1,
        ).astype(np.float32)
        left_hand = np.concatenate(
            [
                base_hand_left + rng.normal(0.0, 0.02, size=base_hand_left.shape),
                np.full((21, 1), 0.75, dtype=np.float32),
            ],
            axis=1,
        ).astype(np.float32)
        right_hand = np.concatenate(
            [
                base_hand_right + rng.normal(0.0, 0.02, size=base_hand_right.shape),
                np.full((21, 1), 0.75, dtype=np.float32),
            ],
            axis=1,
        ).astype(np.float32)
        face = np.concatenate(
            [
                base_face + rng.normal(0.0, jitter_scale * 1.5, size=base_face.shape),
                np.full((69, 1), 0.9, dtype=np.float32),
            ],
            axis=1,
        ).astype(np.float32)

        if frame_index in {6, 7, 14}:
            face[10:25, 2] = 0.1
        if frame_index in {9, 10}:
            body[3:6, 2] = 0.15
            body[3:6, :2] += 0.09

        metas.append({
            "keypoints_body": np.clip(body, 0.0, 1.0),
            "keypoints_left_hand": np.clip(left_hand, 0.0, 1.0),
            "keypoints_right_hand": np.clip(right_hand, 0.0, 1.0),
            "keypoints_face": np.clip(face, 0.0, 1.0),
            "width": 1280,
            "height": 720,
        })
    return metas


def bbox_center_curve(bbox_curve):
    centers = np.array([[frame["center_x"], frame["center_y"]] for frame in bbox_curve["frames"]], dtype=np.float32)
    return centers


def main():
    metas = build_synthetic_metas()
    raw_face_curve = []
    for meta in metas:
        face_points = meta["keypoints_face"][1:, :2] * np.array([1280, 720], dtype=np.float32)
        min_xy = np.min(face_points, axis=0)
        max_xy = np.max(face_points, axis=0)
        raw_face_curve.append((min_xy + max_xy) / 2.0)
    raw_face_curve = np.stack(raw_face_curve)

    stabilized_metas, _ = stabilize_pose_metas(metas)
    _, stabilized_bbox_curve = stabilize_face_bboxes(
        stabilized_metas,
        image_shape=(720, 1280),
        conf_thresh=0.45,
        min_valid_points=15,
    )

    body_track_before = np.stack([meta["keypoints_body"][:, :2] for meta in metas], axis=0)
    body_track_after = np.stack([meta["keypoints_body"][:, :2] for meta in stabilized_metas], axis=0)

    raw_body_jitter = float(np.std(np.diff(body_track_before[:, 0, 0])))
    stabilized_body_jitter = float(np.std(np.diff(body_track_after[:, 0, 0])))
    raw_face_jitter = float(np.std(np.diff(raw_face_curve[:, 0])))
    stabilized_face_jitter = float(np.std(np.diff(bbox_center_curve(stabilized_bbox_curve)[:, 0])))

    if stabilized_body_jitter > raw_body_jitter:
        raise AssertionError(
            f"Body keypoint smoothing did not reduce jitter: before={raw_body_jitter} after={stabilized_body_jitter}"
        )
    if stabilized_face_jitter > raw_face_jitter:
        raise AssertionError(
            f"Face bbox smoothing did not reduce jitter: before={raw_face_jitter} after={stabilized_face_jitter}"
        )

    print("Synthetic pose jitter reduction: PASS")
    print("Synthetic face bbox jitter reduction: PASS")


if __name__ == "__main__":
    main()
