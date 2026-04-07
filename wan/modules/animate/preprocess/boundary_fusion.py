import cv2
import numpy as np

from wan.utils.replacement_masks import compose_background_keep_mask


def _dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(np.float32)
    kernel = np.ones((radius * 2 + 1, radius * 2 + 1), dtype=np.uint8)
    return cv2.dilate((mask > 0.5).astype(np.uint8), kernel, iterations=1).astype(np.float32)


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
    soft_alpha = (
        np.asarray(matting_output.get("soft_alpha"), dtype=np.float32)
        if matting_output is not None and matting_output.get("soft_alpha") is not None
        else None
    )

    if mode not in {"none", "heuristic"}:
        raise ValueError(f"Unsupported boundary fusion mode: {mode}")

    hard_foreground = np.clip(hard_mask, 0.0, 1.0).astype(np.float32)
    fallback_soft_alpha = np.clip(hard_foreground + soft_band, 0.0, 1.0).astype(np.float32)
    if soft_alpha is None:
        soft_alpha = fallback_soft_alpha
    else:
        soft_alpha = np.clip(soft_alpha, 0.0, 1.0).astype(np.float32)

    if mode == "heuristic":
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
    background_keep_prior = np.clip(background_keep_prior, 0.0, 1.0).astype(np.float32)

    return {
        "mode": mode,
        "hard_foreground": hard_foreground,
        "soft_alpha": fused_soft_alpha,
        "boundary_band": boundary_band,
        "background_keep_prior": background_keep_prior,
        "stats": {
            "hard_foreground_mean": float(hard_foreground.mean()),
            "soft_alpha_mean": float(fused_soft_alpha.mean()),
            "boundary_band_mean": float(boundary_band.mean()),
            "background_keep_prior_mean": float(background_keep_prior.mean()),
            "parsing_boundary_mean": float(parsing_boundary.mean()),
        },
    }


def make_fused_boundary_preview(frames: np.ndarray, fusion_output: dict) -> np.ndarray:
    frames = np.asarray(frames, dtype=np.uint8)
    hard = np.asarray(fusion_output["hard_foreground"], dtype=np.float32)
    alpha = np.asarray(fusion_output["soft_alpha"], dtype=np.float32)
    boundary = np.asarray(fusion_output["boundary_band"], dtype=np.float32)
    if hard.shape != frames.shape[:3]:
        raise ValueError(f"fusion hard_foreground must match frames. Got {hard.shape} vs {frames.shape[:3]}.")
    overlays = []
    for frame, hard_frame, alpha_frame, boundary_frame in zip(frames, hard, alpha, boundary):
        overlay = frame.astype(np.float32).copy()
        overlay[..., 0] = np.clip(overlay[..., 0] + boundary_frame * 135.0, 0.0, 255.0)
        overlay[..., 1] = np.clip(overlay[..., 1] + hard_frame * 40.0, 0.0, 255.0)
        overlay[..., 2] = np.clip(overlay[..., 2] + alpha_frame * 95.0, 0.0, 255.0)
        overlays.append(overlay.astype(np.uint8))
    return np.stack(overlays).astype(np.uint8)
