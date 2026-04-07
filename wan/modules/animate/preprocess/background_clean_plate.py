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
    bg_temporal_smooth_strength: float = 0.0,
    bg_video_window_radius: int = 4,
    bg_video_min_visible_count: int = 2,
    bg_video_blend_strength: float = 0.7,
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
            "background_diff": diff,
            "background_temporal_diff": temporal_diff_preview,
            "inpaint_mask": person_mask.astype(np.float32),
            "support_mask": np.zeros_like(person_mask, dtype=np.float32),
            "unresolved_mask": np.zeros_like(person_mask, dtype=np.float32),
            "stats": {
                "inpainted_area_ratio_mean": float(person_mask.mean()),
                "temporal_fluctuation_mean": temporal_fluctuation,
                "band_adjacent_background_stability": _region_temporal_fluctuation(
                    hole_background,
                    np.clip(soft_band, 0.0, 1.0) if soft_band is not None else person_mask,
                ),
                "support_ratio_mean": 0.0,
                "unresolved_ratio_mean": 0.0,
            },
        }

    if bg_inpaint_mode not in {"image", "video"}:
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
        video_clean_frames = None
    else:
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
        video_clean_frames = clean_frames

    diff = np.abs(clean_frames.astype(np.int16) - hole_background.astype(np.int16)).astype(np.uint8)
    temporal_diff_preview, temporal_fluctuation = _temporal_difference_preview(clean_frames)
    stats = {
        "inpainted_area_ratio_mean": float(inpaint_regions.mean()),
        "temporal_fluctuation_mean": temporal_fluctuation,
        "band_adjacent_background_stability": _region_temporal_fluctuation(
            clean_frames,
            np.clip(soft_band, 0.0, 1.0) if soft_band is not None else person_mask,
        ),
        "support_ratio_mean": float((support_mask * inpaint_regions).sum() / max(inpaint_regions.sum(), 1.0)),
        "unresolved_ratio_mean": float(unresolved_mask.sum() / max(inpaint_regions.sum(), 1.0)),
    }
    return clean_frames, {
        "background_mode": background_mode,
        "hole_background": hole_background,
        "clean_plate_background": clean_frames,
        "clean_plate_image_background": image_clean_frames,
        "clean_plate_video_background": video_clean_frames,
        "background_diff": diff,
        "background_temporal_diff": temporal_diff_preview,
        "inpaint_mask": inpaint_regions.astype(np.float32),
        "support_mask": support_mask.astype(np.float32),
        "unresolved_mask": unresolved_mask.astype(np.float32),
        "stats": stats,
    }
