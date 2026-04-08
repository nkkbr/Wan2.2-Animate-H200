import cv2
import numpy as np


def _normalize_map(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    max_value = float(values.max()) if values.size > 0 else 0.0
    if max_value <= 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip(values / max_value, 0.0, 1.0).astype(np.float32)


def _mean_color(frame: np.ndarray, mask: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    pixels = frame[mask > 0]
    if pixels.size == 0:
        return fallback.astype(np.float32)
    return pixels.reshape(-1, 3).mean(axis=0).astype(np.float32)


def _distance_alpha(frame: np.ndarray, fg_color: np.ndarray, bg_color: np.ndarray) -> np.ndarray:
    frame_f = frame.astype(np.float32)
    d_fg = np.linalg.norm(frame_f - fg_color[None, None, :], axis=-1)
    d_bg = np.linalg.norm(frame_f - bg_color[None, None, :], axis=-1)
    return np.clip(d_bg / (d_fg + d_bg + 1e-6), 0.0, 1.0).astype(np.float32)


def _top_band_from_prior(prior: np.ndarray, top_ratio: float = 0.65) -> np.ndarray:
    prior = np.asarray(prior, dtype=np.float32)
    if prior.ndim != 2:
        raise ValueError(f"prior must have shape [H, W]. Got {prior.shape}.")
    coords = np.argwhere(prior > 1e-3)
    if coords.size == 0:
        return np.zeros_like(prior, dtype=np.float32)
    y0 = int(coords[:, 0].min())
    y1 = int(coords[:, 0].max())
    cutoff = y0 + int(round((y1 - y0 + 1) * float(np.clip(top_ratio, 0.0, 1.0))))
    top_band = np.zeros_like(prior, dtype=np.float32)
    top_band[: max(cutoff, y0 + 1), :] = 1.0
    return np.clip(top_band * (prior > 0).astype(np.float32), 0.0, 1.0).astype(np.float32)


def _build_trimap(sure_fg: np.ndarray, sure_bg: np.ndarray) -> np.ndarray:
    trimap = np.full_like(sure_fg, 0.5, dtype=np.float32)
    trimap[sure_bg > 0] = 0.0
    trimap[sure_fg > 0] = 1.0
    return trimap.astype(np.float32)


def _ensure_odd(kernel: int) -> int:
    kernel = max(1, int(kernel))
    if kernel % 2 == 0:
        kernel += 1
    return kernel


def run_alpha_refinement_v2(
    *,
    frames: np.ndarray,
    hard_mask: np.ndarray,
    base_soft_alpha: np.ndarray,
    unknown_region: np.ndarray,
    uncertainty_prior: np.ndarray,
    support_region: np.ndarray,
    soft_band: np.ndarray,
    parsing_boundary_prior: np.ndarray,
    head_prior: np.ndarray,
    hand_prior: np.ndarray,
    occlusion_prior: np.ndarray,
    part_foreground_prior: np.ndarray,
    trimap_inner_erode: int,
    trimap_outer_dilate: int,
    blur_kernel: int,
    detail_boost: float = 0.28,
    shrink_strength: float = 0.34,
    hair_boost: float = 0.42,
    hard_threshold: float = 0.68,
    bilateral_sigma_color: float = 0.12,
    bilateral_sigma_space: float = 5.0,
) -> dict:
    frames = np.asarray(frames, dtype=np.uint8)
    hard_mask = np.asarray(hard_mask, dtype=np.float32)
    base_soft_alpha = np.asarray(base_soft_alpha, dtype=np.float32)
    unknown_region = np.asarray(unknown_region, dtype=np.float32)
    uncertainty_prior = np.asarray(uncertainty_prior, dtype=np.float32)
    support_region = np.asarray(support_region, dtype=np.float32)
    soft_band = np.asarray(soft_band, dtype=np.float32)
    parsing_boundary_prior = np.asarray(parsing_boundary_prior, dtype=np.float32)
    head_prior = np.asarray(head_prior, dtype=np.float32)
    hand_prior = np.asarray(hand_prior, dtype=np.float32)
    occlusion_prior = np.asarray(occlusion_prior, dtype=np.float32)
    part_foreground_prior = np.asarray(part_foreground_prior, dtype=np.float32)

    erode_kernel = max(1, int(trimap_inner_erode))
    dilate_kernel = max(1, int(trimap_outer_dilate))
    erode_element = np.ones((erode_kernel * 2 + 1, erode_kernel * 2 + 1), dtype=np.uint8)
    dilate_element = np.ones((dilate_kernel * 2 + 1, dilate_kernel * 2 + 1), dtype=np.uint8)
    blur_kernel = _ensure_odd(blur_kernel)

    alpha_v2_frames = []
    uncertainty_frames = []
    trimap_frames = []
    fine_boundary_frames = []
    hair_edge_frames = []
    confidence_frames = []
    provenance_frames = []
    refined_hard_frames = []

    for frame, mask, base_alpha, unknown, uncertainty, support, band, parsing_boundary, head, hand, occlusion, part_fg in zip(
        frames,
        hard_mask,
        base_soft_alpha,
        unknown_region,
        uncertainty_prior,
        support_region,
        soft_band,
        parsing_boundary_prior,
        head_prior,
        hand_prior,
        occlusion_prior,
        part_foreground_prior,
    ):
        hard = (mask > 0.5).astype(np.uint8)
        sure_fg = cv2.erode(hard, erode_element, iterations=1)
        support_binary = (np.clip(np.maximum.reduce([support, band, parsing_boundary, part_fg]), 0.0, 1.0) > 0.15).astype(np.uint8)
        sure_bg = (1 - cv2.dilate(support_binary, dilate_element, iterations=1)).astype(np.uint8)
        trimap = _build_trimap(sure_fg, sure_bg)

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if blur_kernel > 1:
            gray = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), sigmaX=0)
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge_strength = _normalize_map(np.sqrt(grad_x * grad_x + grad_y * grad_y))
        canny = cv2.Canny(gray, 64, 128).astype(np.float32) / 255.0
        fine_edge_energy = np.clip(np.maximum(edge_strength, canny), 0.0, 1.0).astype(np.float32)

        fine_boundary_support = np.clip(
            np.maximum.reduce([
                band,
                parsing_boundary,
                unknown,
                np.abs(base_alpha - mask),
                0.6 * occlusion,
            ]),
            0.0,
            1.0,
        ).astype(np.float32)
        boundary_region = cv2.dilate((fine_boundary_support > 0.05).astype(np.uint8), np.ones((5, 5), dtype=np.uint8), iterations=1).astype(np.float32)
        fine_boundary_mask = np.clip(
            fine_edge_energy * boundary_region * (0.45 + 0.55 * np.clip(1.0 - uncertainty, 0.0, 1.0)),
            0.0,
            1.0,
        ).astype(np.float32)

        hair_top_band = _top_band_from_prior(head, top_ratio=0.68)
        hair_edge_mask = np.clip(
            hair_top_band
            * fine_edge_energy
            * np.clip(
                np.maximum.reduce([
                    fine_boundary_support,
                    0.8 * np.clip(1.0 - mask, 0.0, 1.0),
                    0.6 * unknown,
                ]),
                0.0,
                1.0,
            ),
            0.0,
            1.0,
        ).astype(np.float32)

        fg_color = _mean_color(frame, sure_fg, fallback=np.array([192.0, 160.0, 144.0], dtype=np.float32))
        bg_color = _mean_color(frame, sure_bg, fallback=frame.reshape(-1, 3).mean(axis=0))
        color_alpha = _distance_alpha(frame, fg_color, bg_color)

        shrink_mask = np.clip(
            fine_boundary_support
            * (0.25 + 0.75 * fine_edge_energy)
            * (0.50 + 0.50 * np.clip(base_alpha - mask * 0.85, 0.0, 1.0))
            * (0.75 + 0.25 * np.clip(1.0 - 0.6 * head - 0.35 * hand, 0.0, 1.0)),
            0.0,
            1.0,
        ).astype(np.float32)

        alpha = np.clip(
            0.84 * base_alpha
            + 0.08 * color_alpha
            + 0.05 * part_fg
            + 0.03 * mask,
            0.0,
            1.0,
        ).astype(np.float32)
        alpha = np.clip(
            alpha - 0.45 * float(shrink_strength) * shrink_mask * np.clip(alpha - mask * 0.92, 0.0, 1.0),
            0.0,
            1.0,
        ).astype(np.float32)

        detail_gain = float(detail_boost) * fine_boundary_mask * (0.25 + 0.75 * fine_edge_energy) * np.clip(1.0 - uncertainty, 0.0, 1.0)
        hair_gain = float(hair_boost) * hair_edge_mask * np.clip(1.0 - 0.5 * uncertainty, 0.0, 1.0)
        alpha = np.clip(
            alpha
            + (1.0 - alpha) * 0.10 * detail_gain
            + (1.0 - alpha) * 0.14 * hair_gain,
            0.0,
            1.0,
        ).astype(np.float32)

        alpha = alpha * (1.0 - 0.16 * fine_boundary_mask) + color_alpha * (0.16 * fine_boundary_mask)
        max_delta = 0.08
        delta = np.clip(alpha - base_alpha, -max_delta, max_delta)
        delta_gate = np.clip(
            0.55 * fine_boundary_support
            + 0.40 * hair_edge_mask
            + 0.35 * unknown,
            0.0,
            1.0,
        ).astype(np.float32)
        alpha = np.clip(base_alpha + delta * delta_gate, 0.0, 1.0).astype(np.float32)
        alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)

        if bilateral_sigma_space > 0:
            alpha_filtered = cv2.bilateralFilter(
                alpha.astype(np.float32),
                d=5,
                sigmaColor=max(0.01, float(bilateral_sigma_color)) * 255.0,
                sigmaSpace=float(bilateral_sigma_space),
            )
            alpha = np.where(
                trimap == 0.5,
                alpha_filtered,
                alpha,
            ).astype(np.float32)

        alpha = np.where(sure_fg > 0, 1.0, alpha)
        alpha = np.where(sure_bg > 0, 0.0, alpha)
        alpha = np.maximum(alpha, mask * (1.0 - 0.12 * band))
        alpha = np.minimum(alpha, np.clip(np.maximum(support, fine_boundary_support), 0.0, 1.0))
        alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)

        alpha_uncertainty = np.clip(
            0.45 * uncertainty
            + 0.20 * np.abs(alpha - base_alpha)
            + 0.15 * fine_boundary_mask
            + 0.10 * hair_edge_mask
            + 0.10 * occlusion,
            0.0,
            1.0,
        ).astype(np.float32)
        alpha_confidence = np.clip(1.0 - alpha_uncertainty, 0.0, 1.0).astype(np.float32)

        provenance = np.zeros_like(alpha, dtype=np.float32)
        provenance = np.where(fine_boundary_mask > 0.25, 0.5, provenance).astype(np.float32)
        provenance = np.where(hair_edge_mask > 0.20, 1.0, provenance).astype(np.float32)

        refined_hard = np.where(sure_fg > 0, 1.0, (alpha >= float(hard_threshold)).astype(np.float32))
        refined_hard = np.clip(refined_hard * np.clip(np.maximum(support, mask), 0.0, 1.0), 0.0, 1.0).astype(np.float32)

        alpha_v2_frames.append(alpha.astype(np.float32))
        uncertainty_frames.append(alpha_uncertainty.astype(np.float32))
        trimap_frames.append(trimap.astype(np.float32))
        fine_boundary_frames.append(fine_boundary_mask.astype(np.float32))
        hair_edge_frames.append(hair_edge_mask.astype(np.float32))
        confidence_frames.append(alpha_confidence.astype(np.float32))
        provenance_frames.append(provenance.astype(np.float32))
        refined_hard_frames.append(refined_hard.astype(np.float32))

    alpha_v2 = np.stack(alpha_v2_frames).astype(np.float32)
    alpha_uncertainty_v2 = np.stack(uncertainty_frames).astype(np.float32)
    trimap_v2 = np.stack(trimap_frames).astype(np.float32)
    fine_boundary_mask = np.stack(fine_boundary_frames).astype(np.float32)
    hair_edge_mask = np.stack(hair_edge_frames).astype(np.float32)
    alpha_confidence = np.stack(confidence_frames).astype(np.float32)
    alpha_source_provenance = np.stack(provenance_frames).astype(np.float32)
    refined_hard_foreground = np.stack(refined_hard_frames).astype(np.float32)

    return {
        "alpha_v2": alpha_v2,
        "trimap_v2": trimap_v2,
        "alpha_uncertainty_v2": alpha_uncertainty_v2,
        "fine_boundary_mask": fine_boundary_mask,
        "hair_edge_mask": hair_edge_mask,
        "alpha_confidence": alpha_confidence,
        "alpha_source_provenance": alpha_source_provenance,
        "refined_hard_foreground": refined_hard_foreground,
        "stats": {
            "alpha_v2_mean": float(alpha_v2.mean()),
            "trimap_unknown_mean": float((trimap_v2 == 0.5).astype(np.float32).mean()),
            "alpha_uncertainty_v2_mean": float(alpha_uncertainty_v2.mean()),
            "fine_boundary_mask_mean": float(fine_boundary_mask.mean()),
            "hair_edge_mask_mean": float(hair_edge_mask.mean()),
            "alpha_confidence_mean": float(alpha_confidence.mean()),
            "refined_hard_foreground_mean": float(refined_hard_foreground.mean()),
        },
    }


def run_alpha_refinement_v3(
    *,
    frames: np.ndarray,
    legacy_soft_alpha: np.ndarray,
    alpha_v2: np.ndarray,
    trimap_v2: np.ndarray,
    alpha_uncertainty_v2: np.ndarray,
    fine_boundary_mask: np.ndarray,
    hair_edge_mask: np.ndarray,
    refined_hard_foreground: np.ndarray,
    support_region: np.ndarray,
    unknown_region: np.ndarray,
    head_prior: np.ndarray,
    hand_prior: np.ndarray,
    occlusion_prior: np.ndarray,
    detail_boost: float = 0.24,
    color_mix: float = 0.42,
    active_blend: float = 0.55,
    delta_clip: float = 0.10,
) -> dict:
    frames = np.asarray(frames, dtype=np.uint8)
    legacy_soft_alpha = np.asarray(legacy_soft_alpha, dtype=np.float32)
    alpha_v2 = np.asarray(alpha_v2, dtype=np.float32)
    trimap_v2 = np.asarray(trimap_v2, dtype=np.float32)
    alpha_uncertainty_v2 = np.asarray(alpha_uncertainty_v2, dtype=np.float32)
    fine_boundary_mask = np.asarray(fine_boundary_mask, dtype=np.float32)
    hair_edge_mask = np.asarray(hair_edge_mask, dtype=np.float32)
    refined_hard_foreground = np.asarray(refined_hard_foreground, dtype=np.float32)
    support_region = np.asarray(support_region, dtype=np.float32)
    unknown_region = np.asarray(unknown_region, dtype=np.float32)
    head_prior = np.asarray(head_prior, dtype=np.float32)
    hand_prior = np.asarray(hand_prior, dtype=np.float32)
    occlusion_prior = np.asarray(occlusion_prior, dtype=np.float32)

    alpha_v3_frames = []
    trimap_unknown_frames = []
    hair_alpha_frames = []
    confidence_frames = []
    provenance_frames = []

    active_blend = float(np.clip(active_blend, 0.0, 1.0))
    color_mix = float(np.clip(color_mix, 0.0, 1.0))
    detail_boost = float(np.clip(detail_boost, 0.0, 1.0))
    delta_clip = float(max(0.01, delta_clip))

    for frame, legacy_alpha, alpha, trimap, uncertainty, fine_boundary, hair_edge, refined_hard, support, unknown, head, hand, occlusion in zip(
        frames,
        legacy_soft_alpha,
        alpha_v2,
        trimap_v2,
        alpha_uncertainty_v2,
        fine_boundary_mask,
        hair_edge_mask,
        refined_hard_foreground,
        support_region,
        unknown_region,
        head_prior,
        hand_prior,
        occlusion_prior,
    ):
        hard = (refined_hard > 0.5).astype(np.uint8)
        sure_fg = cv2.erode(hard, np.ones((5, 5), dtype=np.uint8), iterations=1)
        support_binary = (np.clip(np.maximum.reduce([support, fine_boundary, unknown, 0.75 * head]), 0.0, 1.0) > 0.12).astype(np.uint8)
        sure_bg = (1 - cv2.dilate(support_binary, np.ones((9, 9), dtype=np.uint8), iterations=1)).astype(np.uint8)

        fg_color = _mean_color(frame, sure_fg, fallback=np.array([192.0, 160.0, 144.0], dtype=np.float32))
        bg_color = _mean_color(frame, sure_bg, fallback=frame.reshape(-1, 3).mean(axis=0))
        color_alpha = _distance_alpha(frame, fg_color, bg_color)

        hair_top_band = _top_band_from_prior(head, top_ratio=0.72)
        hair_region = np.clip(
            np.maximum.reduce([
                hair_edge,
                hair_top_band * np.clip(0.65 * fine_boundary + 0.35 * unknown, 0.0, 1.0),
            ]),
            0.0,
            1.0,
        ).astype(np.float32)
        trimap_unknown = np.clip(
            np.maximum.reduce([
                (trimap == 0.5).astype(np.float32),
                unknown,
                0.85 * fine_boundary,
                0.65 * hair_region,
                0.45 * occlusion,
                ((alpha > 0.08) & (alpha < 0.92)).astype(np.float32) * 0.35,
            ]),
            0.0,
            1.0,
        ).astype(np.float32)
        trimap_unknown = cv2.GaussianBlur(trimap_unknown, (5, 5), sigmaX=0)
        trimap_unknown = np.clip(trimap_unknown, 0.0, 1.0).astype(np.float32)

        blend_support = np.clip(
            0.45 * trimap_unknown
            + 0.30 * hair_region
            + 0.15 * hand
            + 0.10 * fine_boundary,
            0.0,
            1.0,
        ).astype(np.float32)
        certainty_gate = np.clip(1.0 - 0.75 * uncertainty, 0.0, 1.0).astype(np.float32)
        color_delta = np.clip(color_alpha - alpha, -delta_clip, delta_clip).astype(np.float32)
        alpha_v3 = np.clip(
            alpha + color_mix * color_delta * blend_support * certainty_gate,
            0.0,
            1.0,
        ).astype(np.float32)
        alpha_v3 = np.clip(
            alpha_v3 + detail_boost * hair_region * (1.0 - alpha_v3) * certainty_gate * 0.10,
            0.0,
            1.0,
        ).astype(np.float32)
        alpha_v3 = np.where(sure_fg > 0, 1.0, alpha_v3)
        alpha_v3 = np.where(sure_bg > 0, 0.0, alpha_v3)
        alpha_v3 = np.clip(alpha_v3, 0.0, 1.0).astype(np.float32)

        active_soft_alpha = np.clip(
            legacy_alpha * (1.0 - active_blend * blend_support) + alpha_v3 * (active_blend * blend_support),
            0.0,
            1.0,
        ).astype(np.float32)
        active_soft_alpha = np.where(sure_fg > 0, 1.0, active_soft_alpha)
        active_soft_alpha = np.where(sure_bg > 0, 0.0, active_soft_alpha)
        active_soft_alpha = np.clip(active_soft_alpha, 0.0, 1.0).astype(np.float32)

        hair_alpha = np.clip(alpha_v3 * hair_region, 0.0, 1.0).astype(np.float32)
        alpha_confidence = np.clip(
            1.0 - np.clip(0.55 * uncertainty + 0.25 * trimap_unknown + 0.20 * np.abs(alpha_v3 - legacy_alpha), 0.0, 1.0),
            0.0,
            1.0,
        ).astype(np.float32)
        provenance = np.zeros_like(alpha_v3, dtype=np.float32)
        provenance = np.where(fine_boundary > 0.20, 0.33, provenance).astype(np.float32)
        provenance = np.where(hair_region > 0.20, 0.66, provenance).astype(np.float32)
        provenance = np.where(hand > 0.15, 1.0, provenance).astype(np.float32)

        alpha_v3_frames.append(active_soft_alpha)
        trimap_unknown_frames.append(trimap_unknown)
        hair_alpha_frames.append(hair_alpha)
        confidence_frames.append(alpha_confidence)
        provenance_frames.append(provenance)

    alpha_v3 = np.stack(alpha_v3_frames).astype(np.float32)
    trimap_unknown = np.stack(trimap_unknown_frames).astype(np.float32)
    hair_alpha = np.stack(hair_alpha_frames).astype(np.float32)
    alpha_confidence = np.stack(confidence_frames).astype(np.float32)
    alpha_source_provenance = np.stack(provenance_frames).astype(np.float32)

    return {
        "soft_alpha": alpha_v3,
        "trimap_unknown": trimap_unknown,
        "hair_alpha": hair_alpha,
        "alpha_confidence": alpha_confidence,
        "alpha_source_provenance": alpha_source_provenance,
        "stats": {
            "alpha_v3_mean": float(alpha_v3.mean()),
            "trimap_unknown_mean": float(trimap_unknown.mean()),
            "hair_alpha_mean": float(hair_alpha.mean()),
            "alpha_confidence_v3_mean": float(alpha_confidence.mean()),
        },
    }


def make_trimap_preview(trimap: np.ndarray) -> np.ndarray:
    trimap = np.asarray(trimap, dtype=np.float32)
    if trimap.ndim != 3:
        raise ValueError(f"trimap must have shape [T, H, W]. Got {trimap.shape}.")
    preview = np.clip(np.rint(trimap[..., None] * 255.0), 0, 255).astype(np.uint8)
    return np.repeat(preview, repeats=3, axis=3)


def make_binary_mask_preview(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=np.float32)
    if mask.ndim != 3:
        raise ValueError(f"mask must have shape [T, H, W]. Got {mask.shape}.")
    preview = np.clip(np.rint(mask[..., None] * 255.0), 0, 255).astype(np.uint8)
    return np.repeat(preview, repeats=3, axis=3)
