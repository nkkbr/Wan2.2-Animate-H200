import cv2
import numpy as np


def build_hole_background(frames: np.ndarray, person_mask: np.ndarray) -> np.ndarray:
    frames = np.asarray(frames, dtype=np.uint8)
    person_mask = np.asarray(person_mask, dtype=np.float32)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"frames must have shape [T, H, W, 3]. Got {frames.shape}.")
    if person_mask.shape != frames.shape[:3]:
        raise ValueError(
            f"person_mask must have shape [T, H, W] matching frames. Got {person_mask.shape} vs {frames.shape[:3]}."
        )
    background = frames.astype(np.float32) * (1.0 - person_mask[:, :, :, None])
    return np.clip(np.rint(background), 0, 255).astype(np.uint8)


def _expand_mask(mask: np.ndarray, expand_pixels: int) -> np.ndarray:
    mask = (mask > 0.0).astype(np.uint8)
    if expand_pixels <= 0:
        return mask
    kernel_size = max(1, int(expand_pixels) * 2 + 1)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return cv2.dilate(mask, kernel, iterations=1)


def _make_inpaint_region(
    hard_mask: np.ndarray,
    soft_band: np.ndarray | None,
    background_keep_prior: np.ndarray | None,
    expand_pixels: int,
) -> np.ndarray:
    combined = np.asarray(hard_mask, dtype=np.float32)
    if soft_band is not None:
        combined = np.maximum(combined, np.asarray(soft_band, dtype=np.float32))
    if background_keep_prior is not None:
        uncertain_background = np.clip(1.0 - np.asarray(background_keep_prior, dtype=np.float32), 0.0, 1.0)
        combined = np.maximum(combined, uncertain_background * np.maximum(combined, 0.35))
    region = _expand_mask(combined, expand_pixels)
    return (region > 0).astype(np.uint8)


def _inpaint_single_frame(frame: np.ndarray, inpaint_mask: np.ndarray, radius: float, method: str) -> np.ndarray:
    if method == "telea":
        inpaint_flag = cv2.INPAINT_TELEA
    elif method == "ns":
        inpaint_flag = cv2.INPAINT_NS
    else:
        raise ValueError(f"Unsupported bg inpaint method: {method}")
    inpaint_mask_u8 = np.clip(inpaint_mask * 255, 0, 255).astype(np.uint8)
    if np.count_nonzero(inpaint_mask_u8) == 0:
        return frame.copy()
    return cv2.inpaint(frame, inpaint_mask_u8, float(radius), inpaint_flag)


def _temporal_smooth_clean_plate(
    clean_frames: np.ndarray,
    inpaint_regions: np.ndarray,
    temporal_smooth_strength: float,
) -> np.ndarray:
    strength = float(max(0.0, min(temporal_smooth_strength, 1.0)))
    if strength <= 0.0 or len(clean_frames) <= 1:
        return clean_frames.astype(np.uint8)

    smoothed = clean_frames.astype(np.float32).copy()
    previous_region = inpaint_regions[0]
    for idx in range(1, len(clean_frames)):
        region = np.maximum(previous_region, inpaint_regions[idx]).astype(bool)
        smoothed[idx][region] = (
            strength * smoothed[idx - 1][region]
            + (1.0 - strength) * smoothed[idx][region]
        )
        previous_region = inpaint_regions[idx]
    return np.clip(np.rint(smoothed), 0, 255).astype(np.uint8)


def _temporal_smooth_weighted_bidirectional(
    clean_frames: np.ndarray,
    weight_maps: np.ndarray,
    temporal_smooth_strength: float,
) -> np.ndarray:
    strength = float(max(0.0, min(temporal_smooth_strength, 1.0)))
    if strength <= 0.0 or len(clean_frames) <= 1:
        return clean_frames.astype(np.uint8)

    weights = np.clip(np.asarray(weight_maps, dtype=np.float32), 0.0, 1.0)
    if weights.shape != clean_frames.shape[:3]:
        raise ValueError(f"weight_maps must match clean_frames spatial shape. Got {weights.shape} vs {clean_frames.shape[:3]}.")

    forward = clean_frames.astype(np.float32).copy()
    for idx in range(1, len(clean_frames)):
        alpha = (strength * weights[idx])[..., None]
        forward[idx] = alpha * forward[idx - 1] + (1.0 - alpha) * forward[idx]

    backward = clean_frames.astype(np.float32).copy()
    for idx in range(len(clean_frames) - 2, -1, -1):
        alpha = (strength * weights[idx])[..., None]
        backward[idx] = alpha * backward[idx + 1] + (1.0 - alpha) * backward[idx]

    blended = 0.5 * (forward + backward)
    return np.clip(np.rint(blended), 0, 255).astype(np.uint8)


def _build_image_clean_plate(
    frames: np.ndarray,
    person_mask: np.ndarray,
    *,
    soft_band: np.ndarray | None,
    background_keep_prior: np.ndarray | None,
    bg_inpaint_mask_expand: int,
    bg_inpaint_radius: float,
    bg_inpaint_method: str,
    bg_temporal_smooth_strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    inpaint_regions = []
    clean_frames = []
    soft_band_seq = soft_band if soft_band is not None else [None] * len(frames)
    background_prior_seq = background_keep_prior if background_keep_prior is not None else [None] * len(frames)
    for frame, hard_mask, soft_mask, background_prior in zip(frames, person_mask, soft_band_seq, background_prior_seq):
        inpaint_region = _make_inpaint_region(hard_mask, soft_mask, background_prior, bg_inpaint_mask_expand)
        clean_frame = _inpaint_single_frame(frame, inpaint_region, bg_inpaint_radius, bg_inpaint_method)
        inpaint_regions.append(inpaint_region)
        clean_frames.append(clean_frame)

    inpaint_regions = np.stack(inpaint_regions).astype(np.float32)
    clean_frames = np.stack(clean_frames).astype(np.uint8)
    clean_frames = _temporal_smooth_clean_plate(
        clean_frames,
        inpaint_regions=inpaint_regions,
        temporal_smooth_strength=bg_temporal_smooth_strength,
    )
    return clean_frames, inpaint_regions


def _masked_sum(frames: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return (frames.astype(np.float32) * mask[..., None].astype(np.float32)).sum(axis=0)


def _masked_count(mask: np.ndarray) -> np.ndarray:
    return mask.astype(np.float32).sum(axis=0)


def _temporal_difference_preview(frames: np.ndarray) -> tuple[np.ndarray, float]:
    frames_f = frames.astype(np.float32)
    diffs = [np.zeros_like(frames[0], dtype=np.uint8)]
    mean_values = []
    for idx in range(1, len(frames)):
        diff = np.abs(frames_f[idx] - frames_f[idx - 1]).mean(axis=-1)
        mean_values.append(float(diff.mean()))
        diff_u8 = np.clip(np.rint(diff), 0, 255).astype(np.uint8)
        diffs.append(np.repeat(diff_u8[..., None], repeats=3, axis=2))
    return np.stack(diffs).astype(np.uint8), (float(np.mean(mean_values)) if mean_values else 0.0)


def _region_temporal_fluctuation(frames: np.ndarray, region: np.ndarray) -> float:
    if len(frames) <= 1:
        return 0.0
    frames_f = frames.astype(np.float32)
    diffs = np.abs(frames_f[1:] - frames_f[:-1]).mean(axis=-1)
    region_union = np.maximum(region[1:], region[:-1]).astype(np.float32)
    denom = float(region_union.sum())
    if denom <= 1e-6:
        return 0.0
    return float((diffs * region_union).sum() / denom)


def _compute_visibility_span_ratio(visible_masks: np.ndarray) -> np.ndarray:
    frame_count = int(visible_masks.shape[0])
    height, width = visible_masks.shape[1:3]
    first_visible = np.full((height, width), frame_count, dtype=np.int32)
    last_visible = np.full((height, width), -1, dtype=np.int32)
    seen = np.zeros((height, width), dtype=bool)
    for frame_index in range(frame_count):
        visible = visible_masks[frame_index] > 0.5
        first_visible = np.where(visible & (~seen), frame_index, first_visible)
        last_visible = np.where(visible, frame_index, last_visible)
        seen |= visible
    span = np.zeros((height, width), dtype=np.float32)
    if frame_count <= 1:
        return span
    valid = (last_visible >= 0) & (first_visible < frame_count)
    span[valid] = (last_visible[valid] - first_visible[valid]).astype(np.float32) / float(frame_count - 1)
    return np.clip(span, 0.0, 1.0).astype(np.float32)


def _masked_mean_abs_deviation(
    frames: np.ndarray,
    mask: np.ndarray,
    mean_frame: np.ndarray,
) -> np.ndarray:
    count = np.maximum(mask.sum(axis=0), 1.0)
    deviation = np.abs(frames - mean_frame[None]).mean(axis=-1)
    mad = (deviation * mask).sum(axis=0) / count
    return mad.astype(np.float32)


def _confidence_from_deviation(deviation: np.ndarray, scale: float) -> np.ndarray:
    scale = max(float(scale), 1e-3)
    confidence = np.exp(-np.asarray(deviation, dtype=np.float32) / scale)
    return np.clip(confidence, 0.0, 1.0).astype(np.float32)


def _encode_provenance(
    region_mask: np.ndarray,
    *,
    local_weight: np.ndarray,
    global_weight: np.ndarray,
    image_weight: np.ndarray,
) -> np.ndarray:
    provenance = np.zeros_like(region_mask, dtype=np.float32)
    if np.count_nonzero(region_mask) == 0:
        return provenance

    stacked = np.stack([image_weight, global_weight, local_weight], axis=0)
    dominant = np.argmax(stacked, axis=0)
    provenance_codes = np.array([1.0 / 3.0, 2.0 / 3.0, 1.0], dtype=np.float32)
    provenance[region_mask] = provenance_codes[dominant[region_mask]]
    return provenance


def _build_video_clean_plate(
    frames: np.ndarray,
    image_clean_frames: np.ndarray,
    inpaint_regions: np.ndarray,
    *,
    bg_video_window_radius: int,
    bg_video_min_visible_count: int,
    bg_video_blend_strength: float,
    bg_temporal_smooth_strength: float,
) -> tuple[np.ndarray, dict]:
    frame_count = len(frames)
    frames_f = frames.astype(np.float32)
    visible_masks = (1.0 - np.clip(inpaint_regions, 0.0, 1.0)).astype(np.float32)
    global_visible_sum = _masked_sum(frames_f, visible_masks)
    global_visible_count = _masked_count(visible_masks)
    global_visible_count_safe = np.maximum(global_visible_count, 1.0)
    global_mean = global_visible_sum / global_visible_count_safe[..., None]

    blend_strength = float(np.clip(bg_video_blend_strength, 0.0, 1.0))
    min_visible_count = max(1, int(bg_video_min_visible_count))
    window_radius = max(1, int(bg_video_window_radius))

    support_masks = []
    unresolved_masks = []
    video_frames = []
    for index in range(frame_count):
        start = max(0, index - window_radius)
        end = min(frame_count, index + window_radius + 1)
        window_frames = frames_f[start:end]
        window_visible = visible_masks[start:end]
        local_visible_count = _masked_count(window_visible)
        local_visible_count_safe = np.maximum(local_visible_count, 1.0)
        local_mean = _masked_sum(window_frames, window_visible) / local_visible_count_safe[..., None]
        support = (local_visible_count >= float(min_visible_count)).astype(np.float32)
        support_masks.append(support)
        unresolved = np.clip(inpaint_regions[index].astype(np.float32) - support, 0.0, 1.0)
        unresolved_masks.append(unresolved)

        image_frame = image_clean_frames[index].astype(np.float32)
        temporal_prefill = (
            blend_strength * local_mean
            + (1.0 - blend_strength) * image_frame
        )
        temporal_prefill = np.where(
            support[..., None] > 0.0,
            temporal_prefill,
            global_mean,
        )
        final_frame = image_frame.copy()
        region = inpaint_regions[index].astype(bool)
        final_frame[region] = temporal_prefill[region]
        video_frames.append(np.clip(np.rint(final_frame), 0, 255).astype(np.uint8))

    support_masks = np.stack(support_masks).astype(np.float32)
    unresolved_masks = np.stack(unresolved_masks).astype(np.float32)
    video_frames = np.stack(video_frames).astype(np.uint8)
    video_frames = _temporal_smooth_clean_plate(
        video_frames,
        inpaint_regions=inpaint_regions,
        temporal_smooth_strength=bg_temporal_smooth_strength,
    )
    return video_frames, {
        "support_mask": support_masks,
        "unresolved_mask": unresolved_masks,
        "support_ratio_mean": float((support_masks * inpaint_regions).sum() / max(inpaint_regions.sum(), 1.0)),
        "unresolved_ratio_mean": float((unresolved_masks.sum()) / max(inpaint_regions.sum(), 1.0)),
    }


def _build_video_clean_plate_v2(
    frames: np.ndarray,
    image_clean_frames: np.ndarray,
    inpaint_regions: np.ndarray,
    *,
    soft_band: np.ndarray | None,
    bg_video_window_radius: int,
    bg_video_min_visible_count: int,
    bg_video_blend_strength: float,
    bg_temporal_smooth_strength: float,
    bg_video_global_min_visible_count: int,
    bg_video_confidence_threshold: float,
    bg_video_global_blend_strength: float,
    bg_video_consistency_scale: float,
) -> tuple[np.ndarray, dict]:
    base_frames, base_debug = _build_video_clean_plate(
        frames,
        image_clean_frames,
        inpaint_regions,
        bg_video_window_radius=bg_video_window_radius,
        bg_video_min_visible_count=bg_video_min_visible_count,
        bg_video_blend_strength=bg_video_blend_strength,
        bg_temporal_smooth_strength=0.0,
    )

    frame_count = len(frames)
    frames_f = frames.astype(np.float32)
    image_clean_f = image_clean_frames.astype(np.float32)
    base_frames_f = base_frames.astype(np.float32)
    visible_masks = (1.0 - np.clip(inpaint_regions, 0.0, 1.0)).astype(np.float32)

    global_visible_sum = _masked_sum(frames_f, visible_masks)
    global_visible_count = _masked_count(visible_masks)
    global_visible_count_safe = np.maximum(global_visible_count, 1.0)
    global_mean = global_visible_sum / global_visible_count_safe[..., None]
    global_support = np.clip(global_visible_count / max(float(frame_count), 1.0), 0.0, 1.0).astype(np.float32)
    global_span_ratio = _compute_visibility_span_ratio(visible_masks)
    global_visible_support = np.clip(
        0.65 * global_support + 0.35 * np.sqrt(np.clip(global_span_ratio, 0.0, 1.0)),
        0.0,
        1.0,
    ).astype(np.float32)
    global_mad = _masked_mean_abs_deviation(frames_f, visible_masks, global_mean)
    global_consistency = _confidence_from_deviation(global_mad, bg_video_consistency_scale)
    global_confidence = np.clip(global_visible_support * global_consistency, 0.0, 1.0).astype(np.float32)
    global_available = global_visible_count >= float(max(1, int(bg_video_global_min_visible_count)))
    unresolved_threshold = float(np.clip(bg_video_confidence_threshold, 0.0, 1.0))
    global_blend_strength = float(np.clip(bg_video_global_blend_strength, 0.0, 1.0))
    soft_band_maps = (
        np.zeros_like(inpaint_regions, dtype=np.float32)
        if soft_band is None
        else np.clip(np.asarray(soft_band, dtype=np.float32), 0.0, 1.0)
    )

    base_support = np.asarray(base_debug["support_mask"], dtype=np.float32)
    base_unresolved = np.asarray(base_debug["unresolved_mask"], dtype=np.float32)
    visible_support_maps = []
    unresolved_masks = []
    confidence_maps = []
    provenance_maps = []
    video_frames = []
    global_sources = []
    image_fallbacks = []

    for index in range(frame_count):
        region = inpaint_regions[index].astype(bool)
        image_frame = image_clean_f[index]
        base_frame = base_frames_f[index]

        visible_support = np.maximum(base_support[index], global_visible_support)
        confidence = np.maximum(base_support[index], global_confidence)
        confidence = np.where(inpaint_regions[index] > 0.0, confidence, 1.0).astype(np.float32)
        visible_support = np.where(inpaint_regions[index] > 0.0, visible_support, 1.0).astype(np.float32)
        support_norm = np.clip(visible_support / max(unresolved_threshold, 1e-3), 0.0, 1.0)
        confidence_norm = np.clip(confidence / max(unresolved_threshold, 1e-3), 0.0, 1.0)
        availability = np.where(
            global_available,
            1.0,
            np.clip(0.35 + 0.65 * global_visible_support, 0.0, 1.0),
        ).astype(np.float32)
        base_resolved = np.clip(0.30 + 0.70 * base_support[index], 0.0, 1.0)
        resolved_score = np.clip(
            0.35 * support_norm + 0.30 * confidence_norm + 0.20 * availability + 0.15 * base_resolved,
            0.0,
            1.0,
        )
        unresolved = np.clip(
            inpaint_regions[index].astype(np.float32) * (1.0 - resolved_score),
            0.0,
            1.0,
        )
        dynamic_global_blend = np.clip(global_blend_strength + 0.08 * confidence + 0.04 * visible_support, 0.0, 0.98)
        global_prefill = dynamic_global_blend[..., None] * global_mean + (1.0 - dynamic_global_blend[..., None]) * image_frame

        refine_mask = region & global_available & (
            (base_unresolved[index] > 0.0)
            | ((global_visible_support > (base_support[index] + 0.08)) & (global_confidence > unresolved_threshold))
        )
        refine_alpha = np.clip(0.18 + 0.42 * global_confidence, 0.0, 0.70).astype(np.float32)
        final_frame = base_frame.copy()
        if np.count_nonzero(refine_mask) > 0:
            refined = (
                refine_alpha[..., None] * global_prefill
                + (1.0 - refine_alpha[..., None]) * base_frame
            )
            final_frame[refine_mask] = refined[refine_mask]

        local_weight = np.where(region, base_support[index], 0.0).astype(np.float32)
        global_weight = np.where(refine_mask, global_confidence, 0.0).astype(np.float32)
        image_weight = np.where(
            region & (~refine_mask) & (local_weight <= 1e-6),
            1.0,
            np.where(region & (~refine_mask), 0.25, 0.0),
        ).astype(np.float32)
        provenance = _encode_provenance(
            region,
            local_weight=local_weight,
            global_weight=global_weight,
            image_weight=image_weight,
        )

        visible_support_maps.append(visible_support.astype(np.float32))
        unresolved_masks.append(unresolved.astype(np.float32))
        confidence_maps.append(confidence.astype(np.float32))
        provenance_maps.append(provenance.astype(np.float32))
        global_sources.append(global_prefill.astype(np.float32))
        image_fallbacks.append(image_frame.astype(np.float32))
        video_frames.append(np.clip(np.rint(final_frame), 0, 255).astype(np.uint8))

    visible_support_maps = np.stack(visible_support_maps).astype(np.float32)
    unresolved_masks = np.stack(unresolved_masks).astype(np.float32)
    confidence_maps = np.stack(confidence_maps).astype(np.float32)
    provenance_maps = np.stack(provenance_maps).astype(np.float32)
    video_frames = np.stack(video_frames).astype(np.uint8)
    video_frames = _temporal_smooth_clean_plate(
        video_frames,
        inpaint_regions=np.maximum(inpaint_regions, 0.35 * unresolved_masks),
        temporal_smooth_strength=min(1.0, bg_temporal_smooth_strength * 1.35),
    )
    boundary_focus = np.clip(
        np.maximum(soft_band_maps, np.maximum(0.80 * unresolved_masks, 0.35 * (1.0 - confidence_maps))),
        0.0,
        1.0,
    )
    smoothing_weight = np.clip(
        boundary_focus * (0.68 + 0.32 * (1.0 - confidence_maps)),
        0.0,
        1.0,
    )
    video_frames = _temporal_smooth_weighted_bidirectional(
        video_frames,
        weight_maps=smoothing_weight,
        temporal_smooth_strength=min(1.0, bg_temporal_smooth_strength * 2.8 + 0.32),
    )

    return video_frames, {
        "visible_support_map": visible_support_maps,
        "unresolved_region": unresolved_masks,
        "background_confidence": confidence_maps,
        "background_source_provenance": provenance_maps,
        "support_mask": visible_support_maps,
        "unresolved_mask": unresolved_masks,
        "support_ratio_mean": float((visible_support_maps * inpaint_regions).sum() / max(inpaint_regions.sum(), 1.0)),
        "unresolved_ratio_mean": float(unresolved_masks.sum() / max(inpaint_regions.sum(), 1.0)),
        "background_confidence_mean": float((confidence_maps * inpaint_regions).sum() / max(inpaint_regions.sum(), 1.0)),
        "high_confidence_ratio_mean": float(
            (((confidence_maps >= 0.7).astype(np.float32)) * inpaint_regions).sum() / max(inpaint_regions.sum(), 1.0)
        ),
        "global_prefill": np.stack(global_sources).astype(np.float32),
        "image_fallback": np.stack(image_fallbacks).astype(np.float32),
        "global_visible_support": global_visible_support.astype(np.float32),
        "global_confidence": global_confidence.astype(np.float32),
    }


def build_clean_plate_background(
    frames: np.ndarray,
    person_mask: np.ndarray,
    *,
    bg_inpaint_mode: str = "none",
    soft_band: np.ndarray | None = None,
    background_keep_prior: np.ndarray | None = None,
    bg_inpaint_mask_expand: int = 16,
    bg_inpaint_radius: float = 5.0,
    bg_inpaint_method: str = "telea",
    bg_temporal_smooth_strength: float = 0.14,
    bg_video_window_radius: int = 4,
    bg_video_min_visible_count: int = 2,
    bg_video_blend_strength: float = 0.7,
    bg_video_global_min_visible_count: int = 3,
    bg_video_confidence_threshold: float = 0.30,
    bg_video_global_blend_strength: float = 0.95,
    bg_video_consistency_scale: float = 18.0,
) -> tuple[np.ndarray, dict]:
    frames = np.asarray(frames, dtype=np.uint8)
    person_mask = np.asarray(person_mask, dtype=np.float32)
    hole_background = build_hole_background(frames, person_mask)

    if soft_band is not None:
        soft_band = np.asarray(soft_band, dtype=np.float32)
        if soft_band.shape != person_mask.shape:
            raise ValueError(f"soft_band must match person_mask shape. Got {soft_band.shape} vs {person_mask.shape}.")
    if background_keep_prior is not None:
        background_keep_prior = np.asarray(background_keep_prior, dtype=np.float32)
        if background_keep_prior.shape != person_mask.shape:
            raise ValueError(
                f"background_keep_prior must match person_mask shape. Got {background_keep_prior.shape} vs {person_mask.shape}."
            )

    if bg_inpaint_mode == "none":
        diff = np.abs(hole_background.astype(np.int16) - frames.astype(np.int16)).astype(np.uint8)
        temporal_diff_preview, temporal_fluctuation = _temporal_difference_preview(hole_background)
        return hole_background, {
            "background_mode": "hole",
            "hole_background": hole_background,
            "clean_plate_background": hole_background,
            "clean_plate_image_background": hole_background,
            "clean_plate_video_background": None,
            "clean_plate_video_v2_background": None,
            "background_diff": diff,
            "background_temporal_diff": temporal_diff_preview,
            "inpaint_mask": person_mask.astype(np.float32),
            "support_mask": np.zeros_like(person_mask, dtype=np.float32),
            "unresolved_mask": np.zeros_like(person_mask, dtype=np.float32),
            "visible_support_map": np.ones_like(person_mask, dtype=np.float32),
            "unresolved_region": np.zeros_like(person_mask, dtype=np.float32),
            "background_confidence": np.ones_like(person_mask, dtype=np.float32),
            "background_source_provenance": np.zeros_like(person_mask, dtype=np.float32),
            "stats": {
                "inpainted_area_ratio_mean": float(person_mask.mean()),
                "temporal_fluctuation_mean": temporal_fluctuation,
                "band_adjacent_background_stability": _region_temporal_fluctuation(
                    hole_background,
                    np.clip(soft_band, 0.0, 1.0) if soft_band is not None else person_mask,
                ),
                "support_ratio_mean": 0.0,
                "unresolved_ratio_mean": 0.0,
                "background_confidence_mean": 1.0,
                "high_confidence_ratio_mean": 1.0,
            },
        }

    if bg_inpaint_mode not in {"image", "video", "video_v2"}:
        raise ValueError(f"Unsupported bg_inpaint_mode: {bg_inpaint_mode}")

    image_clean_frames, inpaint_regions = _build_image_clean_plate(
        frames,
        person_mask,
        soft_band=soft_band,
        background_keep_prior=background_keep_prior,
        bg_inpaint_mask_expand=bg_inpaint_mask_expand,
        bg_inpaint_radius=bg_inpaint_radius,
        bg_inpaint_method=bg_inpaint_method,
        bg_temporal_smooth_strength=bg_temporal_smooth_strength if bg_inpaint_mode == "image" else 0.0,
    )

    if bg_inpaint_mode == "image":
        clean_frames = image_clean_frames
        background_mode = "clean_plate_image"
        support_mask = np.zeros_like(inpaint_regions, dtype=np.float32)
        unresolved_mask = np.zeros_like(inpaint_regions, dtype=np.float32)
        visible_support_map = np.ones_like(inpaint_regions, dtype=np.float32)
        background_confidence = np.ones_like(inpaint_regions, dtype=np.float32)
        background_source_provenance = np.where(inpaint_regions > 0.0, 1.0 / 3.0, 0.0).astype(np.float32)
        video_clean_frames = None
        video_v2_clean_frames = None
    elif bg_inpaint_mode == "video":
        clean_frames, video_debug = _build_video_clean_plate(
            frames,
            image_clean_frames,
            inpaint_regions,
            bg_video_window_radius=bg_video_window_radius,
            bg_video_min_visible_count=bg_video_min_visible_count,
            bg_video_blend_strength=bg_video_blend_strength,
            bg_temporal_smooth_strength=bg_temporal_smooth_strength,
        )
        background_mode = "clean_plate_video"
        support_mask = video_debug["support_mask"]
        unresolved_mask = video_debug["unresolved_mask"]
        visible_support_map = support_mask
        background_confidence = np.where(inpaint_regions > 0.0, np.clip(support_mask, 0.0, 1.0), 1.0).astype(np.float32)
        background_source_provenance = np.where(
            inpaint_regions > 0.0,
            np.where(support_mask > 0.0, 2.0 / 3.0, 1.0 / 3.0),
            0.0,
        ).astype(np.float32)
        video_clean_frames = clean_frames
        video_v2_clean_frames = None
    else:
        clean_frames, video_debug = _build_video_clean_plate_v2(
            frames,
            image_clean_frames,
            inpaint_regions,
            soft_band=soft_band,
            bg_video_window_radius=bg_video_window_radius,
            bg_video_min_visible_count=bg_video_min_visible_count,
            bg_video_blend_strength=bg_video_blend_strength,
            bg_temporal_smooth_strength=bg_temporal_smooth_strength,
            bg_video_global_min_visible_count=bg_video_global_min_visible_count,
            bg_video_confidence_threshold=bg_video_confidence_threshold,
            bg_video_global_blend_strength=bg_video_global_blend_strength,
            bg_video_consistency_scale=bg_video_consistency_scale,
        )
        background_mode = "clean_plate_video_v2"
        support_mask = video_debug["support_mask"]
        unresolved_mask = video_debug["unresolved_mask"]
        visible_support_map = video_debug["visible_support_map"]
        background_confidence = video_debug["background_confidence"]
        background_source_provenance = video_debug["background_source_provenance"]
        video_clean_frames = None
        video_v2_clean_frames = clean_frames

    diff = np.abs(clean_frames.astype(np.int16) - hole_background.astype(np.int16)).astype(np.uint8)
    temporal_diff_preview, temporal_fluctuation = _temporal_difference_preview(clean_frames)
    stats = {
        "inpainted_area_ratio_mean": float(inpaint_regions.mean()),
        "temporal_fluctuation_mean": temporal_fluctuation,
        "band_adjacent_background_stability": _region_temporal_fluctuation(
            clean_frames,
            np.clip(soft_band, 0.0, 1.0) if soft_band is not None else person_mask,
        ),
        "support_ratio_mean": float((visible_support_map * inpaint_regions).sum() / max(inpaint_regions.sum(), 1.0)),
        "unresolved_ratio_mean": float(unresolved_mask.sum() / max(inpaint_regions.sum(), 1.0)),
        "background_confidence_mean": float((background_confidence * inpaint_regions).sum() / max(inpaint_regions.sum(), 1.0)),
        "high_confidence_ratio_mean": float(
            (((background_confidence >= 0.7).astype(np.float32)) * inpaint_regions).sum() / max(inpaint_regions.sum(), 1.0)
        ),
    }
    if bg_inpaint_mode == "video_v2":
        stats["support_ratio_mean"] = float(video_debug["support_ratio_mean"])
        stats["unresolved_ratio_mean"] = float(video_debug["unresolved_ratio_mean"])
        stats["background_confidence_mean"] = float(video_debug["background_confidence_mean"])
        stats["high_confidence_ratio_mean"] = float(video_debug["high_confidence_ratio_mean"])
    return clean_frames, {
        "background_mode": background_mode,
        "hole_background": hole_background,
        "clean_plate_background": clean_frames,
        "clean_plate_image_background": image_clean_frames,
        "clean_plate_video_background": video_clean_frames,
        "clean_plate_video_v2_background": video_v2_clean_frames,
        "background_diff": diff,
        "background_temporal_diff": temporal_diff_preview,
        "inpaint_mask": inpaint_regions.astype(np.float32),
        "support_mask": support_mask.astype(np.float32),
        "unresolved_mask": unresolved_mask.astype(np.float32),
        "visible_support_map": visible_support_map.astype(np.float32),
        "unresolved_region": unresolved_mask.astype(np.float32),
        "background_confidence": background_confidence.astype(np.float32),
        "background_source_provenance": background_source_provenance.astype(np.float32),
        "stats": stats,
    }
