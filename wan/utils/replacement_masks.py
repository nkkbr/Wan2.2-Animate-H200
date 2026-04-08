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
    detail_release_map: torch.Tensor | np.ndarray | None = None,
    trimap_unknown_map: torch.Tensor | np.ndarray | None = None,
    edge_detail_map: torch.Tensor | np.ndarray | None = None,
    background_keep_prior: torch.Tensor | np.ndarray | None = None,
    visible_support: torch.Tensor | np.ndarray | None = None,
    unresolved_region: torch.Tensor | np.ndarray | None = None,
    background_confidence: torch.Tensor | np.ndarray | None = None,
    occlusion_band: torch.Tensor | np.ndarray | None = None,
    uncertainty_map: torch.Tensor | np.ndarray | None = None,
    face_preserve: torch.Tensor | np.ndarray | None = None,
    face_confidence: torch.Tensor | np.ndarray | None = None,
    face_boundary: torch.Tensor | np.ndarray | None = None,
    hair_boundary: torch.Tensor | np.ndarray | None = None,
    hand_boundary: torch.Tensor | np.ndarray | None = None,
    cloth_boundary: torch.Tensor | np.ndarray | None = None,
    occluded_boundary: torch.Tensor | np.ndarray | None = None,
    conditioning_mode: str = "legacy",
    structure_guard_strength: float = 1.0,
    mode: str = "soft_band",
    boundary_strength: float = 0.5,
) -> torch.Tensor:
    if not isinstance(person_mask, torch.Tensor):
        person_mask = torch.as_tensor(person_mask, dtype=torch.float32)
    person_mask = person_mask.to(dtype=torch.float32)
    background_keep = 1.0 - torch.clamp(person_mask, 0.0, 1.0)
    if mode == "hard":
        return torch.clamp(background_keep, 0.0, 1.0)
    if conditioning_mode not in {"legacy", "rich", "rich_v1", "semantic_v1"}:
        raise ValueError(f"Unsupported replacement conditioning mode: {conditioning_mode}")

    def _to_tensor(value):
        if value is None:
            return None
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value, dtype=torch.float32)
        return torch.clamp(value.to(dtype=torch.float32, device=background_keep.device), 0.0, 1.0)

    face_boundary = _to_tensor(face_boundary)
    hair_boundary = _to_tensor(hair_boundary)
    hand_boundary = _to_tensor(hand_boundary)
    cloth_boundary = _to_tensor(cloth_boundary)
    occluded_boundary = _to_tensor(occluded_boundary)

    if background_keep_prior is not None:
        if not isinstance(background_keep_prior, torch.Tensor):
            background_keep_prior = torch.as_tensor(background_keep_prior, dtype=torch.float32)
        background_keep = background_keep_prior.to(dtype=torch.float32, device=background_keep.device)
        if visible_support is not None:
            if not isinstance(visible_support, torch.Tensor):
                visible_support = torch.as_tensor(visible_support, dtype=torch.float32)
            background_keep = background_keep * (
                0.25 + 0.75 * torch.clamp(visible_support.to(dtype=torch.float32, device=background_keep.device), 0.0, 1.0)
            )
        if background_confidence is not None:
            if not isinstance(background_confidence, torch.Tensor):
                background_confidence = torch.as_tensor(background_confidence, dtype=torch.float32)
            background_keep = background_keep * torch.clamp(
                background_confidence.to(dtype=torch.float32, device=background_keep.device),
                0.0,
                1.0,
            )
        if unresolved_region is not None:
            if not isinstance(unresolved_region, torch.Tensor):
                unresolved_region = torch.as_tensor(unresolved_region, dtype=torch.float32)
            background_keep = background_keep * (
                1.0 - 0.75 * torch.clamp(unresolved_region.to(dtype=torch.float32, device=background_keep.device), 0.0, 1.0)
            )
        if conditioning_mode in {"rich", "rich_v1", "semantic_v1"}:
            if uncertainty_map is not None:
                if not isinstance(uncertainty_map, torch.Tensor):
                    uncertainty_map = torch.as_tensor(uncertainty_map, dtype=torch.float32)
                background_keep = background_keep * (
                    1.0 - 0.85 * torch.clamp(uncertainty_map.to(dtype=torch.float32, device=background_keep.device), 0.0, 1.0)
                )
            if occlusion_band is not None:
                if not isinstance(occlusion_band, torch.Tensor):
                    occlusion_band = torch.as_tensor(occlusion_band, dtype=torch.float32)
                background_keep = background_keep * (
                    1.0 - 0.65 * torch.clamp(occlusion_band.to(dtype=torch.float32, device=background_keep.device), 0.0, 1.0)
                )
            if face_preserve is not None:
                if not isinstance(face_preserve, torch.Tensor):
                    face_preserve = torch.as_tensor(face_preserve, dtype=torch.float32)
                background_keep = background_keep * (
                    1.0 - 0.55 * torch.clamp(face_preserve.to(dtype=torch.float32, device=background_keep.device), 0.0, 1.0)
                )
            if face_confidence is not None and soft_band is not None:
                if not isinstance(face_confidence, torch.Tensor):
                    face_confidence = torch.as_tensor(face_confidence, dtype=torch.float32)
                if not isinstance(soft_band, torch.Tensor):
                    soft_band = torch.as_tensor(soft_band, dtype=torch.float32)
                background_keep = background_keep * (
                    1.0 - 0.20
                    * torch.clamp(face_confidence.to(dtype=torch.float32, device=background_keep.device), 0.0, 1.0)
                    * torch.clamp(soft_band.to(dtype=torch.float32, device=background_keep.device), 0.0, 1.0)
                )
            if conditioning_mode == "semantic_v1":
                if face_boundary is not None:
                    background_keep = background_keep * (1.0 - 0.18 * face_boundary)
                if hair_boundary is not None:
                    background_keep = background_keep * (1.0 - 0.28 * hair_boundary)
                if hand_boundary is not None:
                    background_keep = background_keep * (1.0 - 0.22 * hand_boundary)
                if cloth_boundary is not None:
                    background_keep = background_keep * (1.0 - 0.12 * cloth_boundary)
                if occluded_boundary is not None:
                    background_keep = torch.clamp(
                        background_keep + 0.16 * occluded_boundary * torch.clamp(background_keep, 0.0, 1.0),
                        0.0,
                        1.0,
                    )
            background_keep = background_keep * float(np.clip(structure_guard_strength, 0.0, 1.0))
        return torch.clamp(background_keep, 0.0, 1.0)
    if mode != "soft_band":
        raise ValueError(f"Unsupported replacement mask mode: {mode}")
    if soft_alpha is not None:
        if not isinstance(soft_alpha, torch.Tensor):
            soft_alpha = torch.as_tensor(soft_alpha, dtype=torch.float32)
        soft_alpha = torch.clamp(soft_alpha.to(dtype=torch.float32, device=background_keep.device), 0.0, 1.0)
        background_keep = torch.maximum(background_keep, 1.0 - soft_alpha)
    if detail_release_map is not None:
        if not isinstance(detail_release_map, torch.Tensor):
            detail_release_map = torch.as_tensor(detail_release_map, dtype=torch.float32)
        detail_release_map = torch.clamp(detail_release_map.to(dtype=torch.float32, device=background_keep.device), 0.0, 1.0)
    if trimap_unknown_map is not None:
        if not isinstance(trimap_unknown_map, torch.Tensor):
            trimap_unknown_map = torch.as_tensor(trimap_unknown_map, dtype=torch.float32)
        trimap_unknown_map = torch.clamp(trimap_unknown_map.to(dtype=torch.float32, device=background_keep.device), 0.0, 1.0)
    if edge_detail_map is not None:
        if not isinstance(edge_detail_map, torch.Tensor):
            edge_detail_map = torch.as_tensor(edge_detail_map, dtype=torch.float32)
        edge_detail_map = torch.clamp(edge_detail_map.to(dtype=torch.float32, device=background_keep.device), 0.0, 1.0)
    if soft_band is None:
        return torch.clamp(background_keep, 0.0, 1.0)
    if not isinstance(soft_band, torch.Tensor):
        soft_band = torch.as_tensor(soft_band, dtype=torch.float32)
    soft_band = soft_band.to(dtype=torch.float32, device=background_keep.device)
    boundary_strength = max(0.0, min(float(boundary_strength), 1.0))
    if conditioning_mode in {"rich", "rich_v1", "semantic_v1"}:
        if uncertainty_map is not None:
            if not isinstance(uncertainty_map, torch.Tensor):
                uncertainty_map = torch.as_tensor(uncertainty_map, dtype=torch.float32)
            boundary_strength = boundary_strength * (1.0 - 0.30 * torch.clamp(
                uncertainty_map.to(dtype=torch.float32, device=background_keep.device), 0.0, 1.0
            ))
        if occlusion_band is not None:
            if not isinstance(occlusion_band, torch.Tensor):
                occlusion_band = torch.as_tensor(occlusion_band, dtype=torch.float32)
            boundary_strength = boundary_strength * (1.0 - 0.20 * torch.clamp(
                occlusion_band.to(dtype=torch.float32, device=background_keep.device), 0.0, 1.0
            ))
        if face_preserve is not None:
            if not isinstance(face_preserve, torch.Tensor):
                face_preserve = torch.as_tensor(face_preserve, dtype=torch.float32)
            background_keep = background_keep * (
                1.0 - 0.45 * torch.clamp(face_preserve.to(dtype=torch.float32, device=background_keep.device), 0.0, 1.0)
            )
        if detail_release_map is not None:
            background_keep = background_keep * (1.0 - 0.40 * detail_release_map)
        if trimap_unknown_map is not None:
            background_keep = background_keep * (1.0 - 0.18 * trimap_unknown_map)
        if edge_detail_map is not None:
            background_keep = background_keep * (1.0 - 0.10 * edge_detail_map)
        boundary_strength = boundary_strength * float(np.clip(structure_guard_strength, 0.0, 1.0))
        if detail_release_map is not None:
            boundary_strength = boundary_strength + 0.10 * detail_release_map
        if trimap_unknown_map is not None:
            boundary_strength = boundary_strength + 0.08 * trimap_unknown_map
        if conditioning_mode == "semantic_v1":
            semantic_release = torch.zeros_like(background_keep, dtype=torch.float32)
            if face_boundary is not None:
                semantic_release = semantic_release + 0.18 * face_boundary
            if hair_boundary is not None:
                semantic_release = semantic_release + 0.26 * hair_boundary
            if hand_boundary is not None:
                semantic_release = semantic_release + 0.22 * hand_boundary
            if cloth_boundary is not None:
                semantic_release = semantic_release + 0.12 * cloth_boundary
            if occluded_boundary is not None:
                semantic_release = semantic_release - 0.24 * occluded_boundary
            boundary_strength = torch.clamp(boundary_strength * (1.0 + semantic_release), 0.0, 1.25)
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
