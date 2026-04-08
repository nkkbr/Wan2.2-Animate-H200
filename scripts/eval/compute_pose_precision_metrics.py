#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _load_npz_array(path: Path):
    data = np.load(path)
    for key in ("mask", "labels", "arr_0"):
        if key in data:
            return np.asarray(data[key])
    return np.asarray(data[data.files[0]])


def _as_points(frames, key="points"):
    return np.asarray([frame.get(key, []) for frame in frames], dtype=np.float32)


def _as_states(frames):
    return np.asarray([frame.get("states", []) for frame in frames], dtype=np.int32)


def _trimap_from_mask(mask: np.ndarray, radius: int = 4) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    kernel = np.ones((radius * 2 + 1, radius * 2 + 1), dtype=np.uint8)
    return np.clip(cv2.dilate(mask, kernel, iterations=1) - cv2.erode(mask, kernel, iterations=1), 0, 1).astype(np.uint8)


def _second_order_jitter(points, scale):
    points = np.asarray(points, dtype=np.float32)
    if len(points) < 3:
        return 0.0
    second = points[2:] - 2.0 * points[1:-1] + points[:-2]
    jitter = np.linalg.norm(second, axis=-1)
    scale = np.asarray(scale, dtype=np.float32)
    if scale.ndim == 1:
        scale = scale[1:-1, None]
    return float((jitter / np.maximum(scale, 1e-6)).mean())


def _continuity_score(points, states, spike_mask, scale):
    if points.size == 0:
        return 0.0
    jitter = _second_order_jitter(points, scale)
    visible_ratio = float(np.mean(states == 0))
    interpolated_ratio = float(np.mean(states == 3))
    occluded_ratio = float(np.mean(states == 2))
    low_conf_ratio = float(np.mean(states == 1))
    spike_ratio = float(np.mean(spike_mask))
    jitter_penalty = np.clip(jitter / 0.02, 0.0, 1.0)
    score = (
        0.50 * visible_ratio
        + 0.20 * interpolated_ratio
        + 0.30 * (1.0 - spike_ratio)
        - 0.20 * occluded_ratio
        - 0.15 * low_conf_ratio
        - 0.20 * jitter_penalty
    )
    return float(np.clip(score, 0.0, 1.0))


def _robust_spike_threshold(values):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median))) * 1.4826
    mad = max(mad, 1e-6)
    return float(median + 3.68 * mad)


def main():
    parser = argparse.ArgumentParser(description="Compute pose precision proxy metrics for optimization3.")
    parser.add_argument("--src_root_path", required=True, type=str)
    parser.add_argument("--label_json", type=str, default=None)
    parser.add_argument("--output_json", required=True, type=str)
    args = parser.parse_args()

    src_root = Path(args.src_root_path)
    curve = _read_json(src_root / "pose_conf_curve.json")
    frames = curve.get("frames", [])
    aggregate = curve.get("aggregate_stats", {})

    raw_body = np.array([f.get("raw_body_mean_conf", 0.0) for f in frames], dtype=np.float32)
    raw_face = np.array([f.get("raw_face_mean_conf", 0.0) for f in frames], dtype=np.float32)
    raw_hand = np.array([f.get("raw_hand_mean_conf", 0.0) for f in frames], dtype=np.float32)
    smooth_body = np.array([f.get("smoothed_body_mean_conf", 0.0) for f in frames], dtype=np.float32)

    body_delta = np.abs(np.diff(smooth_body)) if len(smooth_body) > 1 else np.zeros((0,), dtype=np.float32)
    result = {
        "mode": "proxy",
        "frame_count": len(frames),
        "raw_body_mean_conf": float(raw_body.mean()) if len(raw_body) else 0.0,
        "raw_face_mean_conf": float(raw_face.mean()) if len(raw_face) else 0.0,
        "raw_hand_mean_conf": float(raw_hand.mean()) if len(raw_hand) else 0.0,
        "smoothed_body_mean_conf": float(smooth_body.mean()) if len(smooth_body) else 0.0,
        "body_conf_delta_mean": float(body_delta.mean()) if len(body_delta) else 0.0,
        "aggregate_stats": aggregate,
        "pose_tracks_available": False,
        "limb_tracks_available": False,
        "hand_tracks_available": False,
        "pose_visibility_available": False,
        "pose_uncertainty_available": False,
    }

    pose_tracks_path = src_root / "src_pose_tracks.json"
    limb_tracks_path = src_root / "src_limb_tracks.json"
    hand_tracks_path = src_root / "src_hand_tracks.json"
    visibility_path = src_root / "src_pose_visibility.json"
    pose_uncertainty_path = src_root / "src_pose_uncertainty.npz"

    if pose_tracks_path.exists():
        pose_tracks = _read_json(pose_tracks_path)
        pose_frames = pose_tracks.get("frames", [])
        body_points = _as_points(pose_frames)
        body_velocity = np.asarray([frame.get("velocity", []) for frame in pose_frames], dtype=np.float32)
        body_states = _as_states(pose_frames)
        body_scale = None
        if body_points.size > 0:
            core_indices = [1, 2, 5, 8, 11]
            core_points = body_points[:, core_indices, :2]
            shoulders = 0.5 * (body_points[:, 2, :2] + body_points[:, 5, :2])
            hips = 0.5 * (body_points[:, 8, :2] + body_points[:, 11, :2])
            body_scale = np.linalg.norm(shoulders - hips, axis=-1)
            result["body_jitter_mean"] = _second_order_jitter(core_points, np.maximum(body_scale, 1e-4))
            if body_velocity.size > 0:
                body_speed = np.linalg.norm(body_velocity, axis=-1)
                result["body_velocity_mean"] = float(body_speed.mean())
        result["pose_tracks_available"] = True
        result.update(pose_tracks.get("aggregate_stats", {}))

    if hand_tracks_path.exists():
        hand_tracks = _read_json(hand_tracks_path)
        left_points = _as_points(hand_tracks.get("left", []))
        right_points = _as_points(hand_tracks.get("right", []))
        left_states = _as_states(hand_tracks.get("left", []))
        right_states = _as_states(hand_tracks.get("right", []))
        left_velocity = np.asarray([frame.get("velocity", []) for frame in hand_tracks.get("left", [])], dtype=np.float32)
        right_velocity = np.asarray([frame.get("velocity", []) for frame in hand_tracks.get("right", [])], dtype=np.float32)
        if left_points.size > 0 and right_points.size > 0 and len(left_points) >= 3:
            left_second = left_points[2:, :, :2] - 2.0 * left_points[1:-1, :, :2] + left_points[:-2, :, :2]
            right_second = right_points[2:, :, :2] - 2.0 * right_points[1:-1, :, :2] + right_points[:-2, :, :2]
            left_span = np.maximum(left_points[:, :, 0].max(axis=1) - left_points[:, :, 0].min(axis=1), 1e-4)
            right_span = np.maximum(right_points[:, :, 0].max(axis=1) - right_points[:, :, 0].min(axis=1), 1e-4)
            span = 0.5 * (left_span[1:-1] + right_span[1:-1])
            jitter = np.concatenate([np.linalg.norm(left_second, axis=-1), np.linalg.norm(right_second, axis=-1)], axis=1)
            result["hand_jitter_mean"] = float((jitter / span[:, None]).mean())
            if left_velocity.size > 0 and right_velocity.size > 0:
                result["hand_velocity_mean"] = float(
                    0.5 * (
                        np.linalg.norm(left_velocity, axis=-1).mean()
                        + np.linalg.norm(right_velocity, axis=-1).mean()
                    )
                )
        result["hand_tracks_available"] = True
        result.update(hand_tracks.get("aggregate_stats", {}))

    if limb_tracks_path.exists():
        limb_tracks = _read_json(limb_tracks_path)
        result["limb_tracks_available"] = True
        result.update(limb_tracks.get("aggregate_stats", {}))

    if visibility_path.exists():
        visibility = _read_json(visibility_path)
        result["pose_visibility_available"] = True
        result["visibility_counts"] = visibility.get("aggregate_counts", {})

    if pose_uncertainty_path.exists():
        pose_uncertainty = _load_npz_array(pose_uncertainty_path).astype(np.float32)
        result["pose_uncertainty_available"] = True
        result["pose_uncertainty_mean"] = float(pose_uncertainty.mean())
        if pose_tracks_path.exists():
            pose_tracks = _read_json(pose_tracks_path)
            pose_frames = pose_tracks.get("frames", [])
            body_points = _as_points(pose_frames)
            if body_points.size > 0:
                transition = np.zeros_like(pose_uncertainty, dtype=np.uint8)
                height, width = pose_uncertainty.shape[1:3]
                for frame_index in range(min(len(body_points), pose_uncertainty.shape[0])):
                    for point in body_points[frame_index, :, :2]:
                        x = int(np.clip(round(float(point[0]) * (width - 1)), 0, width - 1))
                        y = int(np.clip(round(float(point[1]) * (height - 1)), 0, height - 1))
                        cv2.circle(transition[frame_index], (x, y), 8, 1, -1)
                transition = np.stack([_trimap_from_mask(frame, radius=3) for frame in transition]).astype(np.uint8)
                high_uncertainty = (pose_uncertainty >= np.quantile(pose_uncertainty, 0.8)).astype(np.uint8)
                result["pose_uncertainty_focus_ratio"] = float(
                    np.logical_and(high_uncertainty > 0, transition > 0).sum() / max(high_uncertainty.sum(), 1)
                )

    if pose_tracks_path.exists() and hand_tracks_path.exists():
        pose_tracks = _read_json(pose_tracks_path)
        pose_frames = pose_tracks.get("frames", [])
        body_points = _as_points(pose_frames)
        body_states = _as_states(pose_frames)
        body_velocity = np.asarray([frame.get("velocity", []) for frame in pose_frames], dtype=np.float32)
        left_points = _as_points(_read_json(hand_tracks_path).get("left", []))
        right_points = _as_points(_read_json(hand_tracks_path).get("right", []))
        left_states = _as_states(_read_json(hand_tracks_path).get("left", []))
        right_states = _as_states(_read_json(hand_tracks_path).get("right", []))
        left_velocity = np.asarray([frame.get("velocity", []) for frame in _read_json(hand_tracks_path).get("left", [])], dtype=np.float32)
        right_velocity = np.asarray([frame.get("velocity", []) for frame in _read_json(hand_tracks_path).get("right", [])], dtype=np.float32)
        if body_points.size > 0 and left_points.size > 0 and right_points.size > 0:
            shoulders = 0.5 * (body_points[:, 2, :2] + body_points[:, 5, :2])
            hips = 0.5 * (body_points[:, 8, :2] + body_points[:, 11, :2])
            body_scale = np.maximum(np.linalg.norm(shoulders - hips, axis=-1), 1e-4)
            body_speed_norm = np.linalg.norm(body_velocity, axis=-1) / body_scale[:, None]
            left_speed_norm = np.linalg.norm(left_velocity, axis=-1) / body_scale[:, None]
            right_speed_norm = np.linalg.norm(right_velocity, axis=-1) / body_scale[:, None]
            threshold = _robust_spike_threshold(np.concatenate([body_speed_norm.reshape(-1), left_speed_norm.reshape(-1), right_speed_norm.reshape(-1)]))
            body_spike = (body_speed_norm >= threshold).astype(np.float32)
            left_spike = (left_speed_norm >= threshold).astype(np.float32)
            right_spike = (right_speed_norm >= threshold).astype(np.float32)
            result["velocity_spike_rate"] = float(
                np.mean(np.concatenate([body_spike.reshape(-1), left_spike.reshape(-1), right_spike.reshape(-1)]))
            )
            limb_scores = []
            for indices in ([5, 6, 7], [2, 3, 4], [11, 12, 13], [8, 9, 10]):
                limb_scores.append(_continuity_score(body_points[:, indices, :2], body_states[:, indices], body_spike[:, indices], body_scale))
            limb_continuity = float(np.mean(limb_scores)) if limb_scores else 0.0
            limb_coverage = float(result.get("limb_roi_coverage_ratio", 0.0))
            result["limb_continuity_score"] = float(np.clip(0.8 * limb_continuity + 0.2 * limb_coverage, 0.0, 1.0))

    if args.label_json:
        result["mode"] = "labelled"
        result["label_available"] = True
        result["pck"] = None
    else:
        result["label_available"] = False

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
