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


def build_boundary_conditioning_maps(
    *,
    soft_alpha: np.ndarray | None,
    alpha_v2: np.ndarray | None,
    trimap_v2: np.ndarray | None,
    boundary_band: np.ndarray | None,
    fine_boundary_mask: np.ndarray | None,
    hair_edge_mask: np.ndarray | None,
    alpha_uncertainty_v2: np.ndarray | None,
    alpha_confidence: np.ndarray | None,
    alpha_source_provenance: np.ndarray | None,
    uncertainty_map: np.ndarray | None,
    occlusion_band: np.ndarray | None,
) -> dict:
    base_alpha = None
    if soft_alpha is not None:
        base_alpha = np.clip(np.asarray(soft_alpha, dtype=np.float32), 0.0, 1.0)
    elif alpha_v2 is not None:
        base_alpha = np.clip(np.asarray(alpha_v2, dtype=np.float32), 0.0, 1.0)
    else:
        return {
            "conditioning_soft_alpha": None,
            "conditioning_boundary_band": None,
            "detail_release_map": None,
            "trimap_unknown_map": None,
            "edge_detail_map": None,
            "summary": {
                "available": False,
                "detail_release_mean": None,
                "trimap_unknown_mean": None,
                "edge_detail_mean": None,
                "conditioning_alpha_delta_mean": None,
            },
        }

    alpha_v2 = base_alpha if alpha_v2 is None else np.clip(np.asarray(alpha_v2, dtype=np.float32), 0.0, 1.0)
    boundary_band = np.zeros_like(base_alpha, dtype=np.float32) if boundary_band is None else np.clip(np.asarray(boundary_band, dtype=np.float32), 0.0, 1.0)
    fine_boundary_mask = np.zeros_like(base_alpha, dtype=np.float32) if fine_boundary_mask is None else np.clip(np.asarray(fine_boundary_mask, dtype=np.float32), 0.0, 1.0)
    hair_edge_mask = np.zeros_like(base_alpha, dtype=np.float32) if hair_edge_mask is None else np.clip(np.asarray(hair_edge_mask, dtype=np.float32), 0.0, 1.0)
    alpha_uncertainty_v2 = np.zeros_like(base_alpha, dtype=np.float32) if alpha_uncertainty_v2 is None else np.clip(np.asarray(alpha_uncertainty_v2, dtype=np.float32), 0.0, 1.0)
    uncertainty_map = np.zeros_like(base_alpha, dtype=np.float32) if uncertainty_map is None else np.clip(np.asarray(uncertainty_map, dtype=np.float32), 0.0, 1.0)
    occlusion_band = np.zeros_like(base_alpha, dtype=np.float32) if occlusion_band is None else np.clip(np.asarray(occlusion_band, dtype=np.float32), 0.0, 1.0)

    if alpha_confidence is None:
        alpha_confidence = np.clip(1.0 - alpha_uncertainty_v2, 0.0, 1.0).astype(np.float32)
    else:
        alpha_confidence = np.clip(np.asarray(alpha_confidence, dtype=np.float32), 0.0, 1.0)

    if trimap_v2 is None:
        trimap_unknown_map = np.zeros_like(base_alpha, dtype=np.float32)
    else:
        trimap_v2 = np.asarray(trimap_v2, dtype=np.float32)
        trimap_unknown_map = (np.abs(trimap_v2 - 0.5) <= 0.25).astype(np.float32)

    if alpha_source_provenance is None:
        hair_provenance = np.zeros_like(base_alpha, dtype=np.float32)
        boundary_provenance = np.zeros_like(base_alpha, dtype=np.float32)
    else:
        alpha_source_provenance = np.asarray(alpha_source_provenance, dtype=np.float32)
        hair_provenance = (alpha_source_provenance >= 0.75).astype(np.float32)
        boundary_provenance = (alpha_source_provenance >= 0.25).astype(np.float32)

    edge_detail_map = np.clip(
        np.maximum.reduce([
            boundary_band,
            0.80 * fine_boundary_mask,
            1.10 * hair_edge_mask,
            0.45 * trimap_unknown_map,
            0.35 * occlusion_band,
        ]),
        0.0,
        1.0,
    ).astype(np.float32)

    detail_release_map = np.clip(
        edge_detail_map
        * (0.45 + 0.55 * alpha_confidence)
        * (1.0 - 0.35 * np.maximum(alpha_uncertainty_v2, uncertainty_map)),
        0.0,
        1.0,
    ).astype(np.float32)

    conditioning_boundary_band = np.clip(
        np.maximum.reduce([
            boundary_band,
            0.55 * trimap_unknown_map,
            0.80 * fine_boundary_mask,
            1.05 * hair_edge_mask,
            0.35 * occlusion_band,
        ]),
        0.0,
        1.0,
    ).astype(np.float32)

    alpha_delta = np.clip(alpha_v2 - base_alpha, -0.05, 0.05).astype(np.float32)
    delta_gate = np.clip(
        0.35 * detail_release_map
        + 0.25 * trimap_unknown_map
        + 0.20 * hair_provenance
        + 0.20 * boundary_provenance,
        0.0,
        1.0,
    ).astype(np.float32)
    conditioning_soft_alpha = np.clip(base_alpha + alpha_delta * delta_gate, 0.0, 1.0).astype(np.float32)

    return {
        "conditioning_soft_alpha": conditioning_soft_alpha,
        "conditioning_boundary_band": conditioning_boundary_band,
        "detail_release_map": detail_release_map,
        "trimap_unknown_map": trimap_unknown_map.astype(np.float32),
        "edge_detail_map": edge_detail_map.astype(np.float32),
        "summary": {
            "available": True,
            "detail_release_mean": float(detail_release_map.mean()),
            "trimap_unknown_mean": float(trimap_unknown_map.mean()),
            "edge_detail_mean": float(edge_detail_map.mean()),
            "conditioning_alpha_delta_mean": float(np.abs(conditioning_soft_alpha - base_alpha).mean()),
        },
    }


def build_core_condition_rgb(
    *,
    background_rgb: np.ndarray,
    foreground_rgb: np.ndarray | None,
    foreground_alpha: np.ndarray | None,
    foreground_confidence: np.ndarray | None,
    soft_alpha: np.ndarray | None,
    trimap_unknown: np.ndarray | None,
    hair_alpha: np.ndarray | None,
    uncertainty_map: np.ndarray | None,
    occlusion_band: np.ndarray | None,
    face_preserve: np.ndarray | None,
    composite_roi_mask: np.ndarray | None,
    mode: str = "core_rich_v1",
) -> dict:
    background_rgb = np.asarray(background_rgb, dtype=np.float32)
    if foreground_rgb is None:
        return {
            "core_condition_rgb": background_rgb.astype(np.uint8),
            "summary": {
                "available": False,
                "fg_weight_mean": 0.0,
                "roi_mean": 0.0,
                "confidence_mean": 0.0,
            },
        }

    foreground_rgb = np.asarray(foreground_rgb, dtype=np.float32)
    if foreground_alpha is None:
        foreground_alpha = np.zeros(background_rgb.shape[:3], dtype=np.float32)
    else:
        foreground_alpha = np.clip(np.asarray(foreground_alpha, dtype=np.float32), 0.0, 1.0)

    def _mask(value):
        if value is None:
            return np.zeros(background_rgb.shape[:3], dtype=np.float32)
        return np.clip(np.asarray(value, dtype=np.float32), 0.0, 1.0)

    soft_alpha = _mask(soft_alpha)
    trimap_unknown = _mask(trimap_unknown)
    hair_alpha = _mask(hair_alpha)
    uncertainty_map = _mask(uncertainty_map)
    occlusion_band = _mask(occlusion_band)
    face_preserve = _mask(face_preserve)
    composite_roi_mask = _mask(composite_roi_mask)
    foreground_confidence = _mask(foreground_confidence)

    alpha_den = np.clip(foreground_alpha, 1e-3, 1.0)[..., None]
    foreground_unpremul = np.where(
        foreground_alpha[..., None] > 1e-3,
        np.clip(foreground_rgb / alpha_den, 0.0, 255.0),
        background_rgb,
    )

    roi = np.clip(
        np.maximum.reduce([
            composite_roi_mask,
            0.65 * trimap_unknown,
            0.55 * hair_alpha,
        ]),
        0.0,
        1.0,
    )
    confidence = np.clip(
        1.0 - 0.55 * uncertainty_map - 0.25 * occlusion_band + 0.10 * face_preserve,
        0.0,
        1.0,
    )
    if mode == "decoupled_v2":
        confidence = np.clip(
            0.60 * confidence + 0.40 * foreground_confidence,
            0.0,
            1.0,
        )
        fg_weight = np.clip(
            0.08 * soft_alpha
            + 0.14 * roi * (0.20 + 0.80 * confidence)
            + 0.06 * hair_alpha
            + 0.04 * face_preserve
            + 0.05 * foreground_confidence,
            0.0,
            0.18,
        )
        foreground_proxy = 0.72 * foreground_unpremul + 0.28 * background_rgb
    else:
        fg_weight = np.clip(
            0.02 * soft_alpha
            + 0.06 * roi * (0.25 + 0.75 * confidence)
            + 0.04 * hair_alpha
            + 0.02 * face_preserve,
            0.0,
            0.08,
        )
        foreground_proxy = 0.55 * foreground_unpremul + 0.45 * background_rgb
    core_rgb = background_rgb * (1.0 - fg_weight[..., None]) + foreground_proxy * fg_weight[..., None]
    core_rgb = np.clip(core_rgb, 0.0, 255.0).astype(np.uint8)

    return {
        "core_condition_rgb": core_rgb,
        "summary": {
            "available": True,
            "fg_weight_mean": float(fg_weight.mean()),
            "roi_mean": float(roi.mean()),
            "confidence_mean": float(confidence.mean()),
            "mode": mode,
        },
    }
