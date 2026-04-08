#!/usr/bin/env python
import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _load_npz_array(path: Path):
    data = np.load(path)
    for key in ("mask", "labels", "alpha", "uncertainty", "arr_0"):
        if key in data:
            return np.asarray(data[key])
    return np.asarray(data[data.files[0]])


def _trimap_from_mask(mask: np.ndarray, radius: int = 4) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    kernel = np.ones((radius * 2 + 1, radius * 2 + 1), dtype=np.uint8)
    return np.clip(cv2.dilate(mask, kernel, iterations=1) - cv2.erode(mask, kernel, iterations=1), 0, 1).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Compute face precision proxy metrics for optimization3.")
    parser.add_argument("--src_root_path", required=True, type=str)
    parser.add_argument("--label_json", type=str, default=None)
    parser.add_argument("--output_json", required=True, type=str)
    args = parser.parse_args()

    src_root = Path(args.src_root_path)
    curve = _read_json(src_root / "face_bbox_curve.json")
    frames = curve.get("frames", [])
    centers = np.array([[f.get("center_x", 0.0), f.get("center_y", 0.0)] for f in frames], dtype=np.float32)
    widths = np.array([f.get("width", 0.0) for f in frames], dtype=np.float32)
    heights = np.array([f.get("height", 0.0) for f in frames], dtype=np.float32)
    valid_points = np.array([f.get("valid_face_points", 0.0) for f in frames], dtype=np.float32)
    difficulty = np.array([f.get("difficulty_score", 0.0) for f in frames], dtype=np.float32)
    rerun_ratio = float(np.mean([1.0 if f.get("rerun_applied") else 0.0 for f in frames])) if frames else 0.0

    landmarks_path = src_root / "src_face_landmarks.json"
    pose_path = src_root / "src_face_pose.json"
    expression_path = src_root / "src_face_expression.json"
    face_alpha_path = src_root / "src_face_alpha.npz"
    face_parsing_path = src_root / "src_face_parsing.npz"
    face_uncertainty_path = src_root / "src_face_uncertainty.npz"
    landmarks = _read_json(landmarks_path) if landmarks_path.exists() else None
    head_pose = _read_json(pose_path) if pose_path.exists() else None
    expression = _read_json(expression_path) if expression_path.exists() else None
    face_alpha = _load_npz_array(face_alpha_path).astype(np.float32) if face_alpha_path.exists() else None
    face_parsing = _load_npz_array(face_parsing_path).astype(np.uint8) if face_parsing_path.exists() else None
    face_uncertainty = _load_npz_array(face_uncertainty_path).astype(np.float32) if face_uncertainty_path.exists() else None

    center_step = np.linalg.norm(np.diff(centers, axis=0), axis=1) if len(centers) > 1 else np.zeros((0,), dtype=np.float32)
    width_step = np.abs(np.diff(widths)) if len(widths) > 1 else np.zeros((0,), dtype=np.float32)
    height_step = np.abs(np.diff(heights)) if len(heights) > 1 else np.zeros((0,), dtype=np.float32)

    landmark_nstep = np.zeros((0,), dtype=np.float32)
    landmark_conf = np.zeros((0,), dtype=np.float32)
    if landmarks is not None:
        landmark_frames = landmarks.get("frames", [])
        landmark_conf = np.array([f.get("mean_confidence", 0.0) for f in landmark_frames], dtype=np.float32)
        per_frame = []
        for frame in landmark_frames:
            pts = np.asarray(frame.get("landmarks", []), dtype=np.float32)
            per_frame.append(pts)
        if len(per_frame) > 1:
            step_values = []
            for prev, curr, bbox in zip(per_frame[:-1], per_frame[1:], frames[1:]):
                if prev.size == 0 or curr.size == 0:
                    continue
                count = min(len(prev), len(curr))
                prev_xy = prev[:count, :2]
                curr_xy = curr[:count, :2]
                diag = max(float(np.hypot(bbox.get("width", 0.0), bbox.get("height", 0.0))), 1.0)
                step_values.append(float(np.linalg.norm(curr_xy - prev_xy, axis=1).mean() / diag))
            landmark_nstep = np.asarray(step_values, dtype=np.float32)

    pose_yaw_step = np.zeros((0,), dtype=np.float32)
    pose_pitch_step = np.zeros((0,), dtype=np.float32)
    pose_roll_step = np.zeros((0,), dtype=np.float32)
    if head_pose is not None:
        pose_frames = head_pose.get("frames", [])
        yaw = np.array([f.get("yaw", 0.0) for f in pose_frames], dtype=np.float32)
        pitch = np.array([f.get("pitch", 0.0) for f in pose_frames], dtype=np.float32)
        roll = np.array([f.get("roll", 0.0) for f in pose_frames], dtype=np.float32)
        pose_yaw_step = np.abs(np.diff(yaw)) if len(yaw) > 1 else np.zeros((0,), dtype=np.float32)
        pose_pitch_step = np.abs(np.diff(pitch)) if len(pitch) > 1 else np.zeros((0,), dtype=np.float32)
        pose_roll_step = np.abs(np.diff(roll)) if len(roll) > 1 else np.zeros((0,), dtype=np.float32)

    expr_step = np.zeros((0,), dtype=np.float32)
    if expression is not None:
        expr_frames = expression.get("frames", [])
        intensity = np.array([f.get("expression_intensity", 0.0) for f in expr_frames], dtype=np.float32)
        expr_step = np.abs(np.diff(intensity)) if len(intensity) > 1 else np.zeros((0,), dtype=np.float32)

    result = {
        "mode": "proxy",
        "frame_count": len(frames),
        "center_jitter_mean": float(center_step.mean()) if len(center_step) else 0.0,
        "center_jitter_max": float(center_step.max()) if len(center_step) else 0.0,
        "width_jitter_mean": float(width_step.mean()) if len(width_step) else 0.0,
        "height_jitter_mean": float(height_step.mean()) if len(height_step) else 0.0,
        "valid_face_points_mean": float(valid_points.mean()) if len(valid_points) else 0.0,
        "valid_face_points_min": float(valid_points.min()) if len(valid_points) else 0.0,
        "difficulty_mean": float(difficulty.mean()) if len(difficulty) else 0.0,
        "rerun_ratio": rerun_ratio,
        "source_counts": dict(Counter(f.get("source", "unknown") for f in frames)),
        "face_landmarks_available": landmarks is not None,
        "head_pose_available": head_pose is not None,
        "expression_available": expression is not None,
        "face_alpha_available": face_alpha is not None,
        "face_parsing_available": face_parsing is not None,
        "face_uncertainty_available": face_uncertainty is not None,
        "landmark_confidence_mean": float(landmark_conf.mean()) if len(landmark_conf) else 0.0,
        "landmark_norm_step_mean": float(landmark_nstep.mean()) if len(landmark_nstep) else 0.0,
        "landmark_norm_step_p95": float(np.quantile(landmark_nstep, 0.95)) if len(landmark_nstep) else 0.0,
        "head_pose_yaw_jitter_mean": float(pose_yaw_step.mean()) if len(pose_yaw_step) else 0.0,
        "head_pose_pitch_jitter_mean": float(pose_pitch_step.mean()) if len(pose_pitch_step) else 0.0,
        "head_pose_roll_jitter_mean": float(pose_roll_step.mean()) if len(pose_roll_step) else 0.0,
        "expression_step_mean": float(expr_step.mean()) if len(expr_step) else 0.0,
    }

    if face_alpha is not None:
        result["face_alpha_mean"] = float(face_alpha.mean())
    if face_uncertainty is not None and face_alpha is not None:
        hard_face = (face_alpha >= 0.60).astype(np.uint8)
        transition = np.stack([_trimap_from_mask(frame > 0.60, radius=3) for frame in face_alpha]).astype(np.uint8)
        high_uncertainty = (face_uncertainty >= np.quantile(face_uncertainty, 0.8)).astype(np.uint8)
        result["face_uncertainty_transition_focus_ratio"] = float(
            np.logical_and(high_uncertainty > 0, transition > 0).sum() / max(high_uncertainty.sum(), 1)
        )
        result["face_uncertainty_transition_coverage"] = float(
            np.logical_and(high_uncertainty > 0, transition > 0).sum() / max(transition.sum(), 1)
        )
        interior = (hard_face > 0) & (transition == 0)
        result["face_uncertainty_transition_mean"] = float(face_uncertainty[transition > 0].mean()) if transition.any() else 0.0
        result["face_uncertainty_interior_mean"] = float(face_uncertainty[interior > 0].mean()) if interior.any() else 0.0
        result["face_uncertainty_transition_to_interior_ratio"] = float(
            result["face_uncertainty_transition_mean"] / max(result["face_uncertainty_interior_mean"], 1e-6)
        )

    if args.label_json:
        label = _read_json(Path(args.label_json))
        result["mode"] = "labelled"
        result["label_available"] = True
        frame_index = int(label.get("frame_index", 0))
        result["landmark_nme"] = None
        annotations = label.get("annotations", {})
        landmark_path = annotations.get("face_landmarks", {}).get("path")
        alpha_path = annotations.get("face_alpha", {}).get("path")
        pose_path = annotations.get("face_pose", {}).get("path")
        if landmark_path and landmarks is not None:
            label_landmarks = np.asarray(_read_json(Path(args.label_json).parent / landmark_path)["landmarks"], dtype=np.float32)
            pred_frames = landmarks.get("frames", [])
            if frame_index < len(pred_frames):
                pred_landmarks = np.asarray(pred_frames[frame_index].get("landmarks", []), dtype=np.float32)
                count = min(len(label_landmarks), len(pred_landmarks))
                if count > 0:
                    pred_xy = pred_landmarks[:count, :2]
                    label_xy = label_landmarks[:count, :2]
                    bbox = frames[frame_index]
                    norm = max(float(np.hypot(bbox.get("width", 0.0), bbox.get("height", 0.0))), 1.0)
                    result["landmark_nme"] = float(np.linalg.norm(pred_xy - label_xy, axis=1).mean() / norm)
        if alpha_path and face_alpha is not None:
            label_alpha = _load_npz_array(Path(args.label_json).parent / alpha_path).astype(np.float32)
            if label_alpha.max() > 1.0:
                label_alpha = label_alpha / 255.0
            pred_alpha = face_alpha[frame_index].astype(np.float32)
            result["face_alpha_mae"] = float(np.abs(pred_alpha - label_alpha).mean())
            trimap = _trimap_from_mask(label_alpha > 0.5, radius=3)
            result["face_alpha_trimap_error"] = float((np.abs(pred_alpha - label_alpha) * trimap.astype(np.float32)).sum() / max(trimap.sum(), 1))
        if pose_path and head_pose is not None:
            label_pose = _read_json(Path(args.label_json).parent / pose_path)
            pose_frames = head_pose.get("frames", [])
            if frame_index < len(pose_frames):
                pred_pose = pose_frames[frame_index]
                result["head_pose_abs_error"] = {
                    "yaw": float(abs(pred_pose.get("yaw", 0.0) - label_pose.get("yaw", 0.0))),
                    "pitch": float(abs(pred_pose.get("pitch", 0.0) - label_pose.get("pitch", 0.0))),
                    "roll": float(abs(pred_pose.get("roll", 0.0) - label_pose.get("roll", 0.0))),
                }
    else:
        result["label_available"] = False

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
