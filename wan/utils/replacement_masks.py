import numpy as np
import torch
import torch.nn.functional as F


def build_soft_boundary_band(mask_frames: np.ndarray, band_width: int, blur_kernel_size: int = 0) -> np.ndarray:
    mask_frames = np.asarray(mask_frames, dtype=np.float32)
    if mask_frames.ndim != 3:
        raise ValueError(f"mask_frames must have shape [T, H, W]. Got {mask_frames.shape}.")
    if band_width <= 0:
        return np.zeros_like(mask_frames, dtype=np.float32)

    import cv2

    soft_bands = []
    kernel = None
    if blur_kernel_size and blur_kernel_size > 1:
        kernel = int(blur_kernel_size)
        if kernel % 2 == 0:
            kernel += 1
    for mask in mask_frames:
        hard = (mask > 0.5).astype(np.uint8)
        outside = 1 - hard
        outside_dist = cv2.distanceTransform(outside, cv2.DIST_L2, 5)
        band = np.zeros_like(mask, dtype=np.float32)
        inside_outer_band = (hard == 0) & (outside_dist <= float(band_width))
        band[inside_outer_band] = 1.0 - (outside_dist[inside_outer_band] / float(band_width))
        if kernel is not None:
            band = cv2.GaussianBlur(band, (kernel, kernel), sigmaX=0)
        band = np.clip(band, 0.0, 1.0).astype(np.float32)
        soft_bands.append(band)
    return np.stack(soft_bands).astype(np.float32)


def compose_background_keep_mask(
    person_mask: torch.Tensor | np.ndarray,
    soft_band: torch.Tensor | np.ndarray | None = None,
    *,
    soft_alpha: torch.Tensor | np.ndarray | None = None,
    background_keep_prior: torch.Tensor | np.ndarray | None = None,
    mode: str = "soft_band",
    boundary_strength: float = 0.5,
) -> torch.Tensor:
    if not isinstance(person_mask, torch.Tensor):
        person_mask = torch.as_tensor(person_mask, dtype=torch.float32)
    person_mask = person_mask.to(dtype=torch.float32)
    background_keep = 1.0 - torch.clamp(person_mask, 0.0, 1.0)
    if mode == "hard":
        return torch.clamp(background_keep, 0.0, 1.0)
    if background_keep_prior is not None:
        if not isinstance(background_keep_prior, torch.Tensor):
            background_keep_prior = torch.as_tensor(background_keep_prior, dtype=torch.float32)
        return torch.clamp(background_keep_prior.to(dtype=torch.float32, device=background_keep.device), 0.0, 1.0)
    if mode != "soft_band":
        raise ValueError(f"Unsupported replacement mask mode: {mode}")
    if soft_alpha is not None:
        if not isinstance(soft_alpha, torch.Tensor):
            soft_alpha = torch.as_tensor(soft_alpha, dtype=torch.float32)
        soft_alpha = torch.clamp(soft_alpha.to(dtype=torch.float32, device=background_keep.device), 0.0, 1.0)
        background_keep = torch.maximum(background_keep, 1.0 - soft_alpha)
    if soft_band is None:
        return torch.clamp(background_keep, 0.0, 1.0)
    if not isinstance(soft_band, torch.Tensor):
        soft_band = torch.as_tensor(soft_band, dtype=torch.float32)
    soft_band = soft_band.to(dtype=torch.float32, device=background_keep.device)
    boundary_strength = max(0.0, min(float(boundary_strength), 1.0))
    background_keep = background_keep - boundary_strength * torch.clamp(soft_band, 0.0, 1.0)
    return torch.clamp(background_keep, 0.0, 1.0)


def resize_mask_volume(
    mask_frames: torch.Tensor | np.ndarray,
    *,
    output_size: tuple[int, int],
    mode: str = "area",
) -> torch.Tensor:
    if not isinstance(mask_frames, torch.Tensor):
        mask_frames = torch.as_tensor(mask_frames, dtype=torch.float32)
    mask_frames = mask_frames.to(dtype=torch.float32)
    if mask_frames.ndim == 3:
        mask_frames = mask_frames[:, None]
        squeeze_channel = True
    elif mask_frames.ndim == 4:
        squeeze_channel = False
    else:
        raise ValueError(f"mask_frames must have shape [T, H, W] or [T, C, H, W]. Got {mask_frames.shape}.")

    if mode == "nearest":
        resized = F.interpolate(mask_frames, size=output_size, mode="nearest")
    elif mode == "area":
        resized = F.interpolate(mask_frames, size=output_size, mode="area")
    elif mode == "bilinear":
        resized = F.interpolate(mask_frames, size=output_size, mode="bilinear", align_corners=False)
    else:
        raise ValueError(f"Unsupported mask resize mode: {mode}")

    if squeeze_channel:
        resized = resized[:, 0]
    return torch.clamp(resized, 0.0, 1.0)


def derive_replacement_regions(
    background_keep_mask: torch.Tensor | np.ndarray,
    *,
    transition_low: float = 0.1,
    transition_high: float = 0.9,
) -> dict[str, torch.Tensor]:
    if not isinstance(background_keep_mask, torch.Tensor):
        background_keep_mask = torch.as_tensor(background_keep_mask, dtype=torch.float32)
    background_keep_mask = torch.clamp(background_keep_mask.to(dtype=torch.float32), 0.0, 1.0)
    hard_background_keep = (background_keep_mask >= transition_high).to(torch.float32)
    transition_band = ((background_keep_mask > transition_low) & (background_keep_mask < transition_high)).to(torch.float32)
    free_replacement = (background_keep_mask <= transition_low).to(torch.float32)
    replacement_strength = 1.0 - background_keep_mask
    return {
        "hard_background_keep": hard_background_keep,
        "transition_band": transition_band,
        "free_replacement": free_replacement,
        "replacement_strength": torch.clamp(replacement_strength, 0.0, 1.0),
    }
