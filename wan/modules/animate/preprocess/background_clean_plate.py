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


def _make_inpaint_region(hard_mask: np.ndarray, soft_band: np.ndarray | None, expand_pixels: int) -> np.ndarray:
    if soft_band is not None:
        combined = np.clip(hard_mask + soft_band, 0.0, 1.0)
    else:
        combined = np.asarray(hard_mask, dtype=np.float32)
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


def build_clean_plate_background(
    frames: np.ndarray,
    person_mask: np.ndarray,
    *,
    bg_inpaint_mode: str = "none",
    soft_band: np.ndarray | None = None,
    bg_inpaint_mask_expand: int = 16,
    bg_inpaint_radius: float = 5.0,
    bg_inpaint_method: str = "telea",
    bg_temporal_smooth_strength: float = 0.0,
) -> tuple[np.ndarray, dict]:
    frames = np.asarray(frames, dtype=np.uint8)
    person_mask = np.asarray(person_mask, dtype=np.float32)
    hole_background = build_hole_background(frames, person_mask)

    if bg_inpaint_mode == "none":
        diff = np.abs(hole_background.astype(np.int16) - frames.astype(np.int16)).astype(np.uint8)
        return hole_background, {
            "background_mode": "hole",
            "hole_background": hole_background,
            "clean_plate_background": hole_background,
            "background_diff": diff,
            "inpaint_mask": person_mask.astype(np.float32),
        }

    if bg_inpaint_mode == "video":
        raise NotImplementedError("bg_inpaint_mode=video is not implemented yet. Use 'image' for the current clean-plate path.")

    if bg_inpaint_mode != "image":
        raise ValueError(f"Unsupported bg_inpaint_mode: {bg_inpaint_mode}")

    if soft_band is not None:
        soft_band = np.asarray(soft_band, dtype=np.float32)
        if soft_band.shape != person_mask.shape:
            raise ValueError(f"soft_band must match person_mask shape. Got {soft_band.shape} vs {person_mask.shape}.")

    inpaint_regions = []
    clean_frames = []
    for frame, hard_mask, soft_mask in zip(frames, person_mask, soft_band if soft_band is not None else [None] * len(frames)):
        inpaint_region = _make_inpaint_region(hard_mask, soft_mask, bg_inpaint_mask_expand)
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
    diff = np.abs(clean_frames.astype(np.int16) - hole_background.astype(np.int16)).astype(np.uint8)
    return clean_frames, {
        "background_mode": "clean_plate_image",
        "hole_background": hole_background,
        "clean_plate_background": clean_frames,
        "background_diff": diff,
        "inpaint_mask": inpaint_regions.astype(np.float32),
    }
