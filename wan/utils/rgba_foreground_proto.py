from __future__ import annotations

import numpy as np


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)


def _gradient_band(alpha: np.ndarray) -> np.ndarray:
    alpha = _clip01(alpha)
    gy, gx = np.gradient(alpha.astype(np.float32))
    grad = np.sqrt(gx * gx + gy * gy)
    scale = float(np.percentile(grad, 99.0)) if grad.size else 0.0
    if scale > 1e-6:
        grad = grad / scale
    return _clip01(grad)


ROUND_CONFIGS = {
    "round1": {
        "alpha_mix": 0.35,
        "solve_mix": 0.55,
        "hair_boost": 1.00,
        "hand_boost": 0.70,
        "cloth_boost": 0.70,
        "conf_guard": 0.35,
    },
    "round2": {
        "alpha_mix": 0.65,
        "solve_mix": 0.85,
        "hair_boost": 1.35,
        "hand_boost": 0.95,
        "cloth_boost": 0.85,
        "conf_guard": 0.10,
    },
    "round3": {
        "alpha_mix": 0.28,
        "solve_mix": 0.60,
        "hair_boost": 1.20,
        "hand_boost": 0.80,
        "cloth_boost": 0.75,
        "conf_guard": 0.45,
    },
}


def build_rgba_foreground(
    *,
    source_rgb: np.ndarray,
    foreground_rgb: np.ndarray,
    foreground_alpha: np.ndarray,
    soft_alpha: np.ndarray,
    hard_foreground: np.ndarray,
    boundary_band: np.ndarray,
    trimap_unknown: np.ndarray,
    composite_roi_mask: np.ndarray,
    foreground_confidence: np.ndarray,
    background_rgb: np.ndarray,
    background_visible_support: np.ndarray,
    background_unresolved: np.ndarray,
    hair_alpha: np.ndarray | None = None,
    hair_boundary: np.ndarray | None = None,
    hand_boundary: np.ndarray | None = None,
    cloth_boundary: np.ndarray | None = None,
    uncertainty_map: np.ndarray | None = None,
    mode: str,
) -> dict[str, np.ndarray]:
    if mode not in ROUND_CONFIGS:
        raise ValueError(f"Unsupported RGBA foreground mode: {mode}")

    cfg = ROUND_CONFIGS[mode]
    src = np.asarray(source_rgb, dtype=np.float32)
    fg = np.asarray(foreground_rgb, dtype=np.float32)
    bg = np.asarray(background_rgb, dtype=np.float32)

    fg_alpha = _clip01(foreground_alpha)
    soft_alpha = _clip01(soft_alpha)
    hard_foreground = _clip01(hard_foreground)
    boundary_band = _clip01(boundary_band)
    trimap_unknown = _clip01(trimap_unknown)
    composite_roi_mask = _clip01(composite_roi_mask)
    fg_conf = _clip01(foreground_confidence)
    bg_visible = _clip01(background_visible_support)
    bg_unresolved = _clip01(background_unresolved)
    hair_alpha = _clip01(hair_alpha) if hair_alpha is not None else np.zeros_like(fg_alpha)
    hair_boundary = _clip01(hair_boundary) if hair_boundary is not None else np.zeros_like(fg_alpha)
    hand_boundary = _clip01(hand_boundary) if hand_boundary is not None else np.zeros_like(fg_alpha)
    cloth_boundary = _clip01(cloth_boundary) if cloth_boundary is not None else np.zeros_like(fg_alpha)
    uncertainty = _clip01(uncertainty_map) if uncertainty_map is not None else np.zeros_like(fg_alpha)

    semantic_roi = np.maximum.reduce(
        [
            hair_boundary * cfg["hair_boost"],
            hand_boundary * cfg["hand_boost"],
            cloth_boundary * cfg["cloth_boost"],
        ]
    )
    base_roi = np.maximum.reduce(
        [
            boundary_band,
            trimap_unknown,
            composite_roi_mask * 0.65,
            bg_unresolved * 0.45,
        ]
    )
    low_conf = (1.0 - fg_conf) * 0.55 + uncertainty * 0.35 + (1.0 - bg_visible) * 0.20
    rgba_roi_mask = _clip01(np.maximum(base_roi, semantic_roi + low_conf * (base_roi > 0.02)))

    alpha_anchor = np.maximum(fg_alpha, soft_alpha)
    alpha_anchor = np.maximum(alpha_anchor, hard_foreground * 0.96)
    if np.any(hair_alpha > 0):
        alpha_anchor = np.maximum(alpha_anchor, hair_alpha * (0.88 if mode == "round1" else 0.95))

    alpha_delta = np.clip(alpha_anchor - fg_alpha, 0.0, 1.0)
    alpha_sync = _clip01(
        fg_alpha
        + cfg["alpha_mix"] * rgba_roi_mask * alpha_delta
        + 0.08 * trimap_unknown * (1.0 - fg_conf)
    )
    alpha_sync = np.maximum(alpha_sync, hard_foreground * 0.92)

    premul_solved = np.clip(src - bg * (1.0 - alpha_sync[..., None]), 0.0, 255.0)
    premul_ceiling = alpha_sync[..., None] * 255.0
    premul_solved = np.minimum(premul_solved, premul_ceiling)

    solve_gate = _clip01(
        cfg["solve_mix"] * rgba_roi_mask * (1.0 - cfg["conf_guard"] * fg_conf)
        + 0.22 * bg_unresolved
        + 0.10 * hair_boundary
    )
    rgba_foreground_rgb = np.clip(
        fg * (1.0 - solve_gate[..., None]) + premul_solved * solve_gate[..., None],
        0.0,
        255.0,
    )
    rgba_composite_rgb = np.clip(
        rgba_foreground_rgb + bg * (1.0 - alpha_sync[..., None]),
        0.0,
        255.0,
    ).astype(np.uint8)

    rgba_boundary_band = _gradient_band(alpha_sync)
    rgba_boundary_band = _clip01(
        np.maximum.reduce(
            [
                rgba_boundary_band,
                boundary_band * 0.55,
                hair_boundary * 0.75,
                hand_boundary * 0.45,
                cloth_boundary * 0.45,
                trimap_unknown * 0.40,
            ]
        )
    )

    rgba_trimap_unknown = _clip01(
        np.maximum.reduce(
            [
                trimap_unknown * 0.60,
                rgba_roi_mask * ((alpha_sync > 0.03) & (alpha_sync < 0.97)).astype(np.float32),
                rgba_boundary_band * 0.70,
            ]
        )
    )
    rgba_person_mask = _clip01((alpha_sync > 0.50).astype(np.float32))
    rgba_person_mask = np.maximum(rgba_person_mask, hard_foreground)

    return {
        "rgba_roi_mask": rgba_roi_mask.astype(np.float32),
        "rgba_foreground_alpha": alpha_sync.astype(np.float32),
        "rgba_foreground_rgb": rgba_foreground_rgb.astype(np.uint8),
        "rgba_composite_rgb": rgba_composite_rgb.astype(np.uint8),
        "rgba_boundary_band": rgba_boundary_band.astype(np.float32),
        "rgba_trimap_unknown": rgba_trimap_unknown.astype(np.float32),
        "rgba_person_mask": rgba_person_mask.astype(np.float32),
        "rgba_semantic_roi": semantic_roi.astype(np.float32),
    }
