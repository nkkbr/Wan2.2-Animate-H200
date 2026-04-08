import json
from pathlib import Path

import numpy as np


def load_json_if_exists(path: str | Path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _clip01(value):
    return float(np.clip(float(value), 0.0, 1.0))


def build_face_conditioning_maps(
    *,
    face_alpha: np.ndarray | None,
    face_uncertainty: np.ndarray | None,
    face_bbox_curve: dict | None,
    face_pose: dict | None,
    face_expression: dict | None,
) -> dict:
    if face_alpha is None:
        return {
            "face_confidence_map": None,
            "face_preserve_map": None,
            "frame_confidence": None,
            "summary": {
                "available": False,
                "frame_confidence_mean": None,
                "face_preserve_mean": None,
            },
        }

    face_alpha = np.clip(np.asarray(face_alpha, dtype=np.float32), 0.0, 1.0)
    frame_count = int(face_alpha.shape[0])

    bbox_frames = (face_bbox_curve or {}).get("frames", [])
    pose_frames = (face_pose or {}).get("frames", [])
    expr_frames = (face_expression or {}).get("frames", [])

    frame_conf = []
    for index in range(frame_count):
        bbox_frame = bbox_frames[index] if index < len(bbox_frames) else {}
        pose_frame = pose_frames[index] if index < len(pose_frames) else {}
        expr_frame = expr_frames[index] if index < len(expr_frames) else {}

        valid_face_points = _clip01(float(bbox_frame.get("valid_face_points", 0.0)) / 68.0)
        bbox_conf = _clip01(1.0 - float(bbox_frame.get("difficulty_score", 1.0)))
        pose_conf = _clip01(pose_frame.get("confidence", 0.0))
        expr_conf = _clip01(expr_frame.get("confidence", 0.0))
        mouth_open = _clip01(expr_frame.get("mouth_open", 0.0))
        expression_intensity = _clip01(expr_frame.get("expression_intensity", 0.0))

        # Favor stable, high-confidence face geometry while allowing expressive frames to keep enough influence.
        confidence = (
            0.30 * pose_conf
            + 0.20 * expr_conf
            + 0.25 * valid_face_points
            + 0.15 * bbox_conf
            + 0.10 * (1.0 - 0.35 * mouth_open + 0.15 * expression_intensity)
        )
        frame_conf.append(_clip01(confidence))

    frame_conf = np.asarray(frame_conf, dtype=np.float32)
    face_confidence_map = face_alpha * frame_conf[:, None, None]

    if face_uncertainty is not None:
        face_uncertainty = np.clip(np.asarray(face_uncertainty, dtype=np.float32), 0.0, 1.0)
        face_preserve_map = face_confidence_map * (1.0 - face_uncertainty)
    else:
        face_preserve_map = face_confidence_map.copy()

    return {
        "face_confidence_map": np.clip(face_confidence_map, 0.0, 1.0).astype(np.float32),
        "face_preserve_map": np.clip(face_preserve_map, 0.0, 1.0).astype(np.float32),
        "frame_confidence": frame_conf.astype(np.float32),
        "summary": {
            "available": True,
            "frame_confidence_mean": float(frame_conf.mean()) if frame_conf.size else None,
            "face_preserve_mean": float(face_preserve_map.mean()),
            "face_confidence_mean": float(face_confidence_map.mean()),
        },
    }


def summarize_reference_structure_guard(preprocess_metadata: dict | None) -> dict:
    reference_cfg = (preprocess_metadata or {}).get("processing", {}).get("reference_normalization", {})
    if not reference_cfg:
        return {
            "available": False,
            "guard_strength": 1.0,
        }

    guard = 1.0
    width_budget_triggered = bool(reference_cfg.get("width_budget_triggered", False))
    height_budget_triggered = bool(reference_cfg.get("height_budget_triggered", False))
    if width_budget_triggered:
        guard *= 0.93
    if height_budget_triggered:
        guard *= 0.93

    segment_scales = reference_cfg.get("applied_segment_scales", {}) or {}
    deviations = []
    for key in ("head", "torso", "legs", "width"):
        if key in segment_scales:
            deviations.append(abs(float(segment_scales[key]) - 1.0))
    if deviations:
        max_deviation = max(deviations)
        if max_deviation >= 0.20:
            guard *= 0.94
        elif max_deviation >= 0.10:
            guard *= 0.97
    else:
        max_deviation = 0.0

    return {
        "available": True,
        "guard_strength": float(np.clip(guard, 0.75, 1.0)),
        "width_budget_triggered": width_budget_triggered,
        "height_budget_triggered": height_budget_triggered,
        "max_segment_scale_deviation": float(max_deviation),
        "applied_segment_scales": segment_scales,
    }
