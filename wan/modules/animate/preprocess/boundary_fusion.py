import cv2
import numpy as np

from wan.utils.replacement_masks import compose_background_keep_mask


def _dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(np.float32)
    kernel = np.ones((radius * 2 + 1, radius * 2 + 1), dtype=np.uint8)
    return cv2.dilate((mask > 0.5).astype(np.uint8), kernel, iterations=1).astype(np.float32)


def _normalize_map(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    max_value = float(values.max()) if values.size > 0 else 0.0
    if max_value <= 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip(values / max_value, 0.0, 1.0).astype(np.float32)


def _temporal_instability(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError(f"values must have shape [T, H, W]. Got {values.shape}.")
    if values.shape[0] <= 1:
        return np.zeros_like(values, dtype=np.float32)
    instability = np.zeros_like(values, dtype=np.float32)
    for index in range(values.shape[0]):
        if index == 0:
            reference = values[index + 1]
        elif index == values.shape[0] - 1:
            reference = values[index - 1]
        else:
            reference = 0.5 * (values[index - 1] + values[index + 1])
        instability[index] = np.abs(values[index] - reference)
    return _normalize_map(instability)


def _quantile(value_map: np.ndarray, q: float) -> float:
    value_map = np.asarray(value_map, dtype=np.float32)
    if value_map.size == 0:
        return 0.0
    return float(np.quantile(value_map, q))


def fuse_boundary_signals(
    *,
    hard_mask: np.ndarray,
    soft_band: np.ndarray | None = None,
    parsing_output: dict | None = None,
    matting_output: dict | None = None,
    mode: str = "heuristic",
    support_expand: int = 10,
    alpha_floor: float = 0.92,
    parsing_boundary_weight: float = 0.45,
    matting_boundary_weight: float = 0.55,
    background_boundary_strength: float = 0.7,
) -> dict:
    hard_mask = np.asarray(hard_mask, dtype=np.float32)
    if hard_mask.ndim != 3:
        raise ValueError(f"hard_mask must have shape [T, H, W]. Got {hard_mask.shape}.")
    frame_count, height, width = hard_mask.shape
    zeros = np.zeros((frame_count, height, width), dtype=np.float32)

    if soft_band is None:
        soft_band = zeros
    else:
        soft_band = np.asarray(soft_band, dtype=np.float32)
    parsing_boundary = (
        np.asarray(parsing_output.get("semantic_boundary_prior"), dtype=np.float32)
        if parsing_output is not None and parsing_output.get("semantic_boundary_prior") is not None
        else zeros
    )
    parsing_foreground = (
        np.asarray(parsing_output.get("part_foreground_prior"), dtype=np.float32)
        if parsing_output is not None and parsing_output.get("part_foreground_prior") is not None
        else zeros
    )
    parsing_occlusion = (
        np.asarray(parsing_output.get("occlusion_prior"), dtype=np.float32)
        if parsing_output is not None and parsing_output.get("occlusion_prior") is not None
        else zeros
    )
    soft_alpha = (
        np.asarray(matting_output.get("soft_alpha"), dtype=np.float32)
        if matting_output is not None and matting_output.get("soft_alpha") is not None
        else None
    )
    matting_unknown = (
        np.asarray(matting_output.get("unknown_region"), dtype=np.float32)
        if matting_output is not None and matting_output.get("unknown_region") is not None
        else zeros
    )
    matting_uncertainty = (
        np.asarray(matting_output.get("uncertainty_prior"), dtype=np.float32)
        if matting_output is not None and matting_output.get("uncertainty_prior") is not None
        else zeros
    )
    support_region = (
        np.asarray(matting_output.get("support_region"), dtype=np.float32)
        if matting_output is not None and matting_output.get("support_region") is not None
        else zeros
    )
    refined_hard_foreground = (
        np.asarray(matting_output.get("refined_hard_foreground"), dtype=np.float32)
        if matting_output is not None and matting_output.get("refined_hard_foreground") is not None
        else None
    )

    if mode == "heuristic":
        mode = "legacy"
    if mode not in {"none", "legacy", "v2"}:
        raise ValueError(f"Unsupported boundary fusion mode: {mode}")

    hard_foreground = np.clip(hard_mask, 0.0, 1.0).astype(np.float32)
    if refined_hard_foreground is not None:
        hard_foreground = np.clip(refined_hard_foreground, 0.0, 1.0).astype(np.float32)
    fallback_soft_alpha = np.clip(hard_foreground + soft_band, 0.0, 1.0).astype(np.float32)
    if soft_alpha is None:
        soft_alpha = fallback_soft_alpha
    else:
        soft_alpha = np.clip(soft_alpha, 0.0, 1.0).astype(np.float32)
    occlusion_band = zeros
    uncertainty_map = zeros

    if mode == "legacy":
        support = np.stack([_dilate_mask(mask, support_expand) for mask in hard_foreground]).astype(np.float32)
        support = np.clip(np.maximum(support, soft_band), 0.0, 1.0)
        boundary_from_alpha = np.clip(soft_alpha - hard_foreground, 0.0, 1.0)
        boundary_band = np.maximum(soft_band, boundary_from_alpha)
        boundary_band = np.maximum(boundary_band, parsing_boundary_weight * parsing_boundary)
        boundary_band = np.maximum(boundary_band, matting_boundary_weight * boundary_from_alpha)
        boundary_band = np.clip(boundary_band * support, 0.0, 1.0).astype(np.float32)

        boosted_alpha = np.maximum(soft_alpha, hard_foreground * float(alpha_floor))
        boosted_alpha = np.maximum(boosted_alpha, np.clip(hard_foreground + 0.35 * parsing_foreground * boundary_band, 0.0, 1.0))
        fused_soft_alpha = np.clip(boosted_alpha * support, 0.0, 1.0).astype(np.float32)
    elif mode == "v2":
        alpha_base_hard = np.clip(hard_mask, 0.0, 1.0).astype(np.float32)
        support = np.stack([_dilate_mask(mask, support_expand) for mask in hard_foreground]).astype(np.float32)
        support = np.clip(np.maximum.reduce([support, soft_band, support_region, parsing_foreground]), 0.0, 1.0)
        legacy_support = np.stack([_dilate_mask(mask, support_expand) for mask in alpha_base_hard]).astype(np.float32)
        legacy_support = np.clip(np.maximum(legacy_support, soft_band), 0.0, 1.0)
        boundary_from_alpha = np.clip(soft_alpha - hard_foreground, 0.0, 1.0)
        consensus_boundary = np.maximum.reduce([
            soft_band,
            parsing_boundary_weight * parsing_boundary,
            matting_boundary_weight * boundary_from_alpha,
        ])
        legacy_boundary = np.maximum(soft_band, boundary_from_alpha)
        legacy_boundary = np.maximum(legacy_boundary, parsing_boundary_weight * parsing_boundary)
        legacy_boundary = np.maximum(legacy_boundary, matting_boundary_weight * boundary_from_alpha)
        disagreement = np.maximum.reduce([
            np.abs(soft_alpha - fallback_soft_alpha),
            np.abs(parsing_boundary - soft_band),
            np.abs(parsing_foreground - hard_foreground) * np.clip(consensus_boundary + soft_band, 0.0, 1.0),
        ])
        temporal_uncertainty = np.maximum(
            _temporal_instability(soft_alpha),
            0.7 * _temporal_instability(consensus_boundary),
        )
        occlusion_band = np.clip(
            np.maximum(
                parsing_occlusion,
                np.maximum(parsing_foreground, 0.75 * consensus_boundary) * np.clip(matting_unknown + boundary_from_alpha, 0.0, 1.0),
            ) * support,
            0.0,
            1.0,
        ).astype(np.float32)
        boundary_focus = np.clip(
            np.maximum.reduce([
                consensus_boundary,
                0.85 * occlusion_band,
                0.60 * matting_unknown,
                0.35 * soft_band,
            ]),
            0.0,
            1.0,
        ).astype(np.float32)
        uncertainty_raw = np.maximum.reduce([
            0.80 * matting_unknown,
            0.70 * matting_uncertainty,
            0.65 * disagreement,
            0.45 * temporal_uncertainty,
            0.55 * occlusion_band,
        ])
        uncertainty_map = np.clip(
            np.maximum(
                uncertainty_raw * np.maximum(boundary_focus, 0.15 * np.clip(1.0 - soft_alpha, 0.0, 1.0)),
                0.35 * occlusion_band,
            ),
            0.0,
            1.0,
        ).astype(np.float32)
        boundary_band = np.clip(
            np.maximum.reduce([
                consensus_boundary,
                0.65 * occlusion_band,
                0.55 * uncertainty_map,
            ]) * support,
            0.0,
            1.0,
        ).astype(np.float32)
        confident_hard = np.clip(
            hard_foreground * (soft_alpha >= 0.88).astype(np.float32)
            + hard_foreground * (1.0 - 0.25 * uncertainty_map) * (soft_alpha >= 0.75).astype(np.float32),
            0.0,
            1.0,
        ).astype(np.float32)
        hard_foreground = np.maximum(confident_hard, np.stack([_dilate_mask(mask, 1) for mask in confident_hard]).astype(np.float32) * 0.0 + confident_hard)
        boosted_alpha = np.maximum(soft_alpha, alpha_base_hard * float(alpha_floor))
        boosted_alpha = np.maximum(
            boosted_alpha,
            np.clip(alpha_base_hard + 0.38 * parsing_foreground * np.clip(1.0 - uncertainty_map, 0.0, 1.0), 0.0, 1.0),
        )
        legacy_like_alpha = np.maximum(
            boosted_alpha,
            np.clip(alpha_base_hard + 0.35 * parsing_foreground * legacy_boundary, 0.0, 1.0),
        )
        fused_soft_alpha = np.clip(legacy_like_alpha * legacy_support, 0.0, 1.0).astype(np.float32)
    else:
        boundary_band = np.clip(soft_band, 0.0, 1.0).astype(np.float32)
        fused_soft_alpha = fallback_soft_alpha

    background_keep_prior = compose_background_keep_mask(
        hard_foreground,
        soft_band=boundary_band,
        background_keep_prior=None,
        mode="soft_band",
        boundary_strength=background_boundary_strength,
    ).cpu().numpy().astype(np.float32)
    background_keep_prior = np.maximum(background_keep_prior, np.clip(1.0 - fused_soft_alpha, 0.0, 1.0))
    if mode == "v2":
        background_keep_prior = np.clip(
            background_keep_prior * (1.0 - 0.45 * uncertainty_map) * (1.0 - 0.25 * occlusion_band),
            0.0,
            1.0,
        ).astype(np.float32)
    background_keep_prior = np.clip(background_keep_prior, 0.0, 1.0).astype(np.float32)

    return {
        "mode": mode,
        "hard_foreground": hard_foreground,
        "soft_alpha": fused_soft_alpha,
        "boundary_band": boundary_band,
        "occlusion_band": occlusion_band.astype(np.float32),
        "uncertainty_map": uncertainty_map.astype(np.float32),
        "background_keep_prior": background_keep_prior,
        "stats": {
            "hard_foreground_mean": float(hard_foreground.mean()),
            "soft_alpha_mean": float(fused_soft_alpha.mean()),
            "boundary_band_mean": float(boundary_band.mean()),
            "occlusion_band_mean": float(occlusion_band.mean()),
            "uncertainty_map_mean": float(uncertainty_map.mean()),
            "uncertainty_map_p80": _quantile(uncertainty_map, 0.80),
            "uncertainty_map_p95": _quantile(uncertainty_map, 0.95),
            "background_keep_prior_mean": float(background_keep_prior.mean()),
            "parsing_boundary_mean": float(parsing_boundary.mean()),
            "parsing_occlusion_mean": float(parsing_occlusion.mean()),
            "matting_unknown_mean": float(matting_unknown.mean()),
            "matting_uncertainty_mean": float(matting_uncertainty.mean()),
        },
    }


def make_fused_boundary_preview(frames: np.ndarray, fusion_output: dict) -> np.ndarray:
    frames = np.asarray(frames, dtype=np.uint8)
    hard = np.asarray(fusion_output["hard_foreground"], dtype=np.float32)
    alpha = np.asarray(fusion_output["soft_alpha"], dtype=np.float32)
    boundary = np.asarray(fusion_output["boundary_band"], dtype=np.float32)
    occlusion = np.asarray(fusion_output.get("occlusion_band"), dtype=np.float32) if fusion_output.get("occlusion_band") is not None else np.zeros_like(boundary)
    uncertainty = np.asarray(fusion_output.get("uncertainty_map"), dtype=np.float32) if fusion_output.get("uncertainty_map") is not None else np.zeros_like(boundary)
    if hard.shape != frames.shape[:3]:
        raise ValueError(f"fusion hard_foreground must match frames. Got {hard.shape} vs {frames.shape[:3]}.")
    overlays = []
    for frame, hard_frame, alpha_frame, boundary_frame, occlusion_frame, uncertainty_frame in zip(
        frames, hard, alpha, boundary, occlusion, uncertainty
    ):
        overlay = frame.astype(np.float32).copy()
        overlay[..., 0] = np.clip(overlay[..., 0] + boundary_frame * 110.0 + uncertainty_frame * 90.0, 0.0, 255.0)
        overlay[..., 1] = np.clip(overlay[..., 1] + hard_frame * 40.0 + occlusion_frame * 120.0, 0.0, 255.0)
        overlay[..., 2] = np.clip(overlay[..., 2] + alpha_frame * 95.0, 0.0, 255.0)
        overlays.append(overlay.astype(np.uint8))
    return np.stack(overlays).astype(np.uint8)


def make_uncertainty_heatmap_preview(uncertainty_map: np.ndarray) -> np.ndarray:
    uncertainty_map = np.asarray(uncertainty_map, dtype=np.float32)
    if uncertainty_map.ndim != 3:
        raise ValueError(f"uncertainty_map must have shape [T, H, W]. Got {uncertainty_map.shape}.")
    previews = []
    for frame in uncertainty_map:
        frame_u8 = np.clip(np.rint(frame * 255.0), 0, 255).astype(np.uint8)
        previews.append(cv2.cvtColor(cv2.applyColorMap(frame_u8, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB))
    return np.stack(previews).astype(np.uint8)


def make_alpha_hard_compare_preview(hard_foreground: np.ndarray, soft_alpha: np.ndarray) -> np.ndarray:
    hard_foreground = np.asarray(hard_foreground, dtype=np.float32)
    soft_alpha = np.asarray(soft_alpha, dtype=np.float32)
    if hard_foreground.shape != soft_alpha.shape or hard_foreground.ndim != 3:
        raise ValueError(
            f"hard_foreground and soft_alpha must match with shape [T, H, W]. Got {hard_foreground.shape} vs {soft_alpha.shape}."
        )
    previews = []
    for hard_frame, alpha_frame in zip(hard_foreground, soft_alpha):
        hard_u8 = np.repeat(np.clip(np.rint(hard_frame[..., None] * 255.0), 0, 255).astype(np.uint8), 3, axis=2)
        alpha_u8 = np.repeat(np.clip(np.rint(alpha_frame[..., None] * 255.0), 0, 255).astype(np.uint8), 3, axis=2)
        previews.append(np.concatenate([hard_u8, alpha_u8], axis=1))
    return np.stack(previews).astype(np.uint8)
