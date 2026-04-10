from __future__ import annotations

import numpy as np


def _clip01(arr: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(arr, dtype=np.float32), 0.0, 1.0).astype(np.float32)


def build_layer_roi_mask(
    *,
    composite_roi_mask: np.ndarray,
    background_unresolved: np.ndarray,
    occlusion_band: np.ndarray,
    occluded_boundary: np.ndarray | None,
    uncertainty_map: np.ndarray | None,
) -> np.ndarray:
    components = [
        _clip01(composite_roi_mask),
        _clip01(background_unresolved),
        _clip01(occlusion_band),
    ]
    if occluded_boundary is not None:
        components.append(_clip01(occluded_boundary))
    if uncertainty_map is not None:
        components.append(np.clip(1.25 * _clip01(uncertainty_map), 0.0, 1.0).astype(np.float32))
    return np.clip(np.maximum.reduce(components), 0.0, 1.0).astype(np.float32)


def _signed_residual_rgb(
    *,
    source_rgb: np.ndarray,
    foreground_rgb: np.ndarray,
    background_rgb: np.ndarray,
    foreground_alpha: np.ndarray,
) -> np.ndarray:
    source = np.asarray(source_rgb, dtype=np.float32)
    fg = np.asarray(foreground_rgb, dtype=np.float32)
    bg = np.asarray(background_rgb, dtype=np.float32)
    alpha = _clip01(foreground_alpha)[..., None]
    baseline = np.clip(fg + bg * (1.0 - alpha), 0.0, 255.0)
    return source - baseline


def _build_occlusion_alpha(
    *,
    foreground_alpha: np.ndarray,
    foreground_confidence: np.ndarray,
    background_visible_support: np.ndarray,
    background_unresolved: np.ndarray,
    composite_roi_mask: np.ndarray,
    occlusion_band: np.ndarray,
    occluded_boundary: np.ndarray | None,
    uncertainty_map: np.ndarray | None,
    mode: str,
    occlusion_strength: float,
    alpha_mix: float,
) -> tuple[np.ndarray, np.ndarray]:
    fg_alpha = _clip01(foreground_alpha)
    fg_conf = _clip01(foreground_confidence)
    visible = _clip01(background_visible_support)
    unresolved = _clip01(background_unresolved)
    composite_roi = _clip01(composite_roi_mask)
    occ_band = _clip01(occlusion_band)
    occ_boundary = _clip01(occluded_boundary) if occluded_boundary is not None else np.zeros_like(fg_alpha, dtype=np.float32)
    uncertainty = _clip01(uncertainty_map) if uncertainty_map is not None else np.zeros_like(fg_alpha, dtype=np.float32)

    if mode == "round1":
        seed = np.maximum.reduce([
            unresolved,
            occ_band,
            0.65 * composite_roi,
            0.85 * occ_boundary,
            0.50 * uncertainty,
        ])
        gating = np.clip((1.0 - fg_alpha) * (0.65 + 0.35 * (1.0 - visible)) * (1.0 - 0.15 * fg_conf), 0.0, 1.0)
    elif mode == "round2":
        seed = np.maximum.reduce([
            1.10 * unresolved,
            1.00 * occ_band,
            0.85 * composite_roi,
            1.15 * occ_boundary,
            0.75 * uncertainty,
        ])
        gating = np.clip((1.0 - fg_alpha) * (0.45 + 0.55 * (1.0 - visible)) * (1.0 - 0.10 * fg_conf), 0.0, 1.0)
    elif mode == "round3":
        seed = np.maximum.reduce([
            1.00 * unresolved,
            0.90 * occ_band,
            0.75 * composite_roi,
            1.00 * occ_boundary,
            0.70 * uncertainty,
        ])
        gating = np.clip((1.0 - fg_alpha) * (0.55 + 0.45 * (1.0 - visible)) * (1.0 - 0.25 * fg_conf), 0.0, 1.0)
    else:
        raise ValueError(f"Unsupported layer decomposition mode: {mode}")

    occlusion_alpha = np.clip(occlusion_strength * np.clip(seed, 0.0, 1.0) * gating, 0.0, 1.0).astype(np.float32)
    total_alpha = np.clip(fg_alpha + alpha_mix * occlusion_alpha * (1.0 - fg_alpha), 0.0, 1.0).astype(np.float32)
    return occlusion_alpha, total_alpha


def decompose_layers(
    *,
    source_rgb: np.ndarray,
    foreground_rgb: np.ndarray,
    foreground_alpha: np.ndarray,
    foreground_confidence: np.ndarray,
    background_rgb: np.ndarray,
    background_visible_support: np.ndarray,
    background_unresolved: np.ndarray,
    composite_roi_mask: np.ndarray,
    occlusion_band: np.ndarray,
    occluded_boundary: np.ndarray | None,
    uncertainty_map: np.ndarray | None,
    mode: str,
    occlusion_strength: float,
    alpha_mix: float,
    residual_mix: float,
) -> dict[str, np.ndarray]:
    fg_rgb = np.asarray(foreground_rgb, dtype=np.float32)
    fg_alpha = _clip01(foreground_alpha)
    bg_rgb = np.asarray(background_rgb, dtype=np.float32)
    src_rgb = np.asarray(source_rgb, dtype=np.float32)

    layer_roi = build_layer_roi_mask(
        composite_roi_mask=composite_roi_mask,
        background_unresolved=background_unresolved,
        occlusion_band=occlusion_band,
        occluded_boundary=occluded_boundary,
        uncertainty_map=uncertainty_map,
    )
    occlusion_alpha, total_alpha = _build_occlusion_alpha(
        foreground_alpha=fg_alpha,
        foreground_confidence=foreground_confidence,
        background_visible_support=background_visible_support,
        background_unresolved=background_unresolved,
        composite_roi_mask=composite_roi_mask,
        occlusion_band=occlusion_band,
        occluded_boundary=occluded_boundary,
        uncertainty_map=uncertainty_map,
        mode=mode,
        occlusion_strength=occlusion_strength,
        alpha_mix=alpha_mix,
    )
    residual = _signed_residual_rgb(
        source_rgb=src_rgb,
        foreground_rgb=fg_rgb,
        background_rgb=bg_rgb,
        foreground_alpha=fg_alpha,
    )
    residual_rgb = np.clip(np.abs(residual), 0.0, 255.0)
    source_occ_rgb = np.clip(src_rgb * occlusion_alpha[..., None], 0.0, 255.0)
    residual_occ_rgb = np.clip(residual_rgb * occlusion_alpha[..., None], 0.0, 255.0)
    occlusion_rgb = np.clip(
        residual_mix * residual_occ_rgb + (1.0 - residual_mix) * source_occ_rgb,
        0.0,
        255.0,
    ).astype(np.float32)

    composite_rgb = np.clip(
        fg_rgb + occlusion_rgb + bg_rgb * (1.0 - total_alpha[..., None]),
        0.0,
        255.0,
    ).astype(np.uint8)
    person_mask = (total_alpha >= 0.5).astype(np.float32)
    trimap_unknown = np.logical_or(
        np.logical_and(total_alpha > 0.08, total_alpha < 0.92),
        np.logical_or(layer_roi > 0.10, occlusion_alpha > 0.08),
    ).astype(np.float32)
    return {
        "layer_roi_mask": layer_roi.astype(np.float32),
        "foreground_alpha": fg_alpha.astype(np.float32),
        "occlusion_alpha": occlusion_alpha.astype(np.float32),
        "total_alpha": total_alpha.astype(np.float32),
        "occlusion_rgb": occlusion_rgb.astype(np.uint8),
        "composite_rgb": composite_rgb,
        "person_mask": person_mask.astype(np.float32),
        "trimap_unknown": trimap_unknown.astype(np.float32),
    }
