import json
from pathlib import Path

import cv2
import numpy as np
import torch

from .local_edge_restoration import restore_local_edge_roi
from .media_io import write_output_frames, write_person_mask_artifact


def tensor_video_to_rgb_frames(video: torch.Tensor) -> np.ndarray:
    if video.ndim != 4:
        raise ValueError(f"Expected video tensor with shape [C, T, H, W]. Got {video.shape}.")
    frames = video.detach().cpu().to(dtype=torch.float32).permute(1, 2, 3, 0).numpy()
    return np.clip((frames + 1.0) * 127.5, 0.0, 255.0).astype(np.uint8)


def rgb_frames_to_tensor_video(frames: np.ndarray) -> torch.Tensor:
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected RGB frames with shape [T, H, W, 3]. Got {frames.shape}.")
    normalized = (frames / 127.5) - 1.0
    return torch.from_numpy(normalized).permute(3, 0, 1, 2).contiguous()


def build_inner_boundary_band(person_mask: np.ndarray, inner_width: int) -> np.ndarray:
    person_mask = np.asarray(person_mask, dtype=np.float32)
    if person_mask.ndim != 3:
        raise ValueError(f"person_mask must have shape [T, H, W]. Got {person_mask.shape}.")
    if inner_width <= 0:
        return np.zeros_like(person_mask, dtype=np.float32)

    kernel_size = max(1, int(inner_width) * 2 + 1)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    bands = []
    for mask in person_mask:
        hard = (mask > 0.5).astype(np.uint8)
        eroded = cv2.erode(hard, kernel, iterations=1)
        band = np.clip(hard - eroded, 0, 1).astype(np.float32)
        bands.append(band)
    return np.stack(bands).astype(np.float32)


def _gaussian_kernel(sigma: float) -> tuple[int, int]:
    radius = max(1, int(round(float(sigma) * 3.0)))
    kernel = radius * 2 + 1
    return kernel, kernel


def apply_unsharp_mask(frames: np.ndarray, *, sigma: float, amount: float) -> np.ndarray:
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"frames must have shape [T, H, W, 3]. Got {frames.shape}.")
    if amount <= 0.0:
        return frames.copy()

    ksize = _gaussian_kernel(max(0.1, sigma))
    sharpened = []
    for frame in frames:
        blurred = cv2.GaussianBlur(frame, ksize, sigmaX=sigma, sigmaY=sigma)
        sharp = np.clip(frame + float(amount) * (frame - blurred), 0.0, 1.0)
        sharpened.append(sharp.astype(np.float32))
    return np.stack(sharpened).astype(np.float32)


def _grayscale(frames: np.ndarray) -> np.ndarray:
    return np.dot(frames[..., :3], np.array([0.299, 0.587, 0.114], dtype=np.float32))


def compute_boundary_refinement_metrics(
    *,
    before_frames: np.ndarray,
    after_frames: np.ndarray,
    background_frames: np.ndarray,
    outer_band: np.ndarray,
    inner_band: np.ndarray,
) -> dict:
    before = np.asarray(before_frames, dtype=np.float32)
    after = np.asarray(after_frames, dtype=np.float32)
    background = np.asarray(background_frames, dtype=np.float32)
    outer_band = np.asarray(outer_band, dtype=np.float32)
    inner_band = np.asarray(inner_band, dtype=np.float32)
    band_mask = np.clip(np.maximum(outer_band, inner_band), 0.0, 1.0)

    before_gray = _grayscale(before)
    after_gray = _grayscale(after)
    background_gray = _grayscale(background)

    def _gradient(gray: np.ndarray) -> np.ndarray:
        grads = []
        for frame in gray:
            dx = cv2.Sobel(frame, cv2.CV_32F, 1, 0, ksize=3)
            dy = cv2.Sobel(frame, cv2.CV_32F, 0, 1, ksize=3)
            grads.append(np.sqrt(dx * dx + dy * dy))
        return np.stack(grads).astype(np.float32)

    gradient_before = _gradient(before_gray)
    gradient_after = _gradient(after_gray)

    def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
        denom = float(weights.sum())
        if denom <= 1e-6:
            return 0.0
        return float((values * weights).sum() / denom)

    band_mad = np.abs(after - before).mean(axis=-1)
    contrast_before = np.abs(before_gray - background_gray)
    contrast_after = np.abs(after_gray - background_gray)
    halo_weights = np.clip(outer_band * (1.0 - outer_band), 0.0, 1.0)
    halo_threshold = 0.08
    halo_before = ((contrast_before > halo_threshold).astype(np.float32) * halo_weights)
    halo_after = ((contrast_after > halo_threshold).astype(np.float32) * halo_weights)

    return {
        "band_gradient_before_mean": _weighted_mean(gradient_before, band_mask),
        "band_gradient_after_mean": _weighted_mean(gradient_after, band_mask),
        "band_edge_contrast_before_mean": _weighted_mean(contrast_before, outer_band),
        "band_edge_contrast_after_mean": _weighted_mean(contrast_after, outer_band),
        "band_mad_mean": _weighted_mean(band_mad, band_mask),
        "halo_ratio_before": _weighted_mean((halo_before > 0).astype(np.float32), halo_weights),
        "halo_ratio_after": _weighted_mean((halo_after > 0).astype(np.float32), halo_weights),
        "outer_band_mean": float(outer_band.mean()),
        "inner_band_mean": float(inner_band.mean()),
    }


def _score_roi_candidate(
    *,
    before_rgb: np.ndarray,
    candidate_rgb: np.ndarray,
    background_rgb: np.ndarray,
    outer_band: np.ndarray,
    inner_band: np.ndarray,
) -> tuple[float, dict]:
    metrics = compute_boundary_refinement_metrics(
        before_frames=before_rgb[None].astype(np.float32),
        after_frames=candidate_rgb[None].astype(np.float32),
        background_frames=background_rgb[None].astype(np.float32),
        outer_band=outer_band[None].astype(np.float32),
        inner_band=inner_band[None].astype(np.float32),
    )
    score = (
        1.20 * float(metrics["band_gradient_after_mean"])
        + 1.55 * float(metrics["band_edge_contrast_after_mean"])
        - 0.85 * float(metrics["halo_ratio_after"])
        - 0.30 * float(metrics["band_mad_mean"])
    )
    return score, metrics


def _apply_semantic_roi_experts(
    *,
    generated_frames: np.ndarray,
    background_frames: np.ndarray,
    person_mask: np.ndarray,
    outer_band: np.ndarray,
    inner_band: np.ndarray,
    soft_alpha: np.ndarray | None,
    background_confidence: np.ndarray,
    uncertainty_map: np.ndarray,
    occlusion_band: np.ndarray,
    face_preserve_map: np.ndarray,
    face_confidence_map: np.ndarray,
    detail_release_map: np.ndarray,
    trimap_unknown_map: np.ndarray,
    edge_detail_map: np.ndarray,
    face_boundary_map: np.ndarray,
    hair_boundary_map: np.ndarray,
    hand_boundary_map: np.ndarray,
    cloth_boundary_map: np.ndarray,
    occluded_boundary_map: np.ndarray,
    strength: float,
    sharpen: float,
    inner_width: int,
    sharpen_sigma: float,
) -> tuple[np.ndarray, dict]:
    refined_frames = generated_frames.copy()
    roi_blend_alpha = np.zeros_like(person_mask, dtype=np.float32)
    expert_coverages: dict[str, float] = {}
    expert_strengths: dict[str, float] = {}
    local_edge_focus_frames = np.zeros_like(person_mask, dtype=np.float32)
    local_edge_gain_frames = np.zeros_like(person_mask, dtype=np.float32)
    local_edge_feather_frames = np.zeros_like(person_mask, dtype=np.float32)

    expert_specs = [
        {
            "name": "face",
            "mask": np.clip(np.maximum(face_boundary_map, 0.35 * face_preserve_map) * np.maximum(outer_band, 0.40 * inner_band), 0.0, 1.0),
            "scale": 1.7,
            "min_size": 112,
            "pad": 16,
            "sharpen": min(1.0, sharpen + 0.04),
            "detail": min(1.0, strength + 0.02),
            "core_w": 0.58,
            "feather_w": 0.26,
            "extra_unsharp": 0.04,
            "dilate_kernel": 9,
        },
        {
            "name": "hair",
            "mask": np.clip(np.maximum(hair_boundary_map, 0.55 * trimap_unknown_map) * np.maximum(outer_band, 0.55 * inner_band), 0.0, 1.0),
            "scale": 2.6,
            "min_size": 96,
            "pad": 22,
            "sharpen": min(1.0, sharpen + 0.22),
            "detail": min(1.0, strength + 0.24),
            "core_w": 0.78,
            "feather_w": 0.36,
            "extra_unsharp": 0.16,
            "dilate_kernel": 13,
        },
        {
            "name": "hand",
            "mask": np.clip(np.maximum(hand_boundary_map, 0.45 * detail_release_map) * np.maximum(outer_band, 0.45 * inner_band), 0.0, 1.0),
            "scale": 2.6,
            "min_size": 92,
            "pad": 20,
            "sharpen": min(1.0, sharpen + 0.28),
            "detail": min(1.0, strength + 0.26),
            "core_w": 0.82,
            "feather_w": 0.36,
            "extra_unsharp": 0.22,
            "dilate_kernel": 17,
        },
        {
            "name": "cloth",
            "mask": np.clip(np.maximum(cloth_boundary_map, 0.50 * edge_detail_map) * np.maximum(outer_band, 0.40 * inner_band), 0.0, 1.0),
            "scale": 2.05,
            "min_size": 96,
            "pad": 20,
            "sharpen": min(1.0, sharpen + 0.18),
            "detail": min(1.0, strength + 0.18),
            "core_w": 0.72,
            "feather_w": 0.30,
            "extra_unsharp": 0.22,
            "dilate_kernel": 13,
        },
        {
            "name": "occluded",
            "mask": np.clip(occluded_boundary_map * np.maximum(outer_band, 0.50 * inner_band), 0.0, 1.0),
            "scale": 1.6,
            "min_size": 80,
            "pad": 16,
            "sharpen": min(1.0, sharpen + 0.02),
            "detail": min(1.0, strength + 0.02),
            "core_w": 0.34,
            "feather_w": 0.18,
            "extra_unsharp": 0.0,
            "dilate_kernel": 7,
        },
    ]

    for spec in expert_specs:
        dilated_mask = np.stack(
            [_dilate_mask(frame, kernel_size=spec["dilate_kernel"], iterations=1) for frame in spec["mask"]],
            axis=0,
        ).astype(np.float32)
        expert_mask = np.clip(
            dilated_mask + 0.15 * spec["mask"],
            0.0,
            1.0,
        ).astype(np.float32)
        expert_coverages[f"{spec['name']}_coverage_mean"] = float(expert_mask.mean())
        expert_strengths[f"{spec['name']}_detail_strength"] = float(spec["detail"])
        if expert_mask.max() <= 1e-4:
            continue
        boxes = _compute_roi_boxes(expert_mask, min_size=spec["min_size"], pad=spec["pad"])
        for frame_idx, (x0, y0, x1, y1) in enumerate(boxes):
            crop_w = max(1, x1 - x0)
            crop_h = max(1, y1 - y0)
            if crop_w <= 1 or crop_h <= 1:
                continue
            if float(expert_mask[frame_idx, y0:y1, x0:x1].mean()) <= 1e-4:
                continue
            target_size = (max(32, int(round(crop_w * spec["scale"]))), max(32, int(round(crop_h * spec["scale"]))))

            def _crop_resize_rgb(arr):
                return _resize_rgb_frame(arr[frame_idx, y0:y1, x0:x1], target_size)

            def _crop_resize_mask(arr):
                return _resize_mask_frame(arr[frame_idx, y0:y1, x0:x1], target_size)

            crop_generated_float = _crop_resize_rgb(generated_frames).astype(np.float32) / 255.0
            crop_background_float = _crop_resize_rgb(background_frames).astype(np.float32) / 255.0
            crop_person_mask = _crop_resize_mask(person_mask).astype(np.float32)
            crop_soft_band = _crop_resize_mask(outer_band).astype(np.float32)
            crop_soft_alpha = _crop_resize_mask(soft_alpha).astype(np.float32) if soft_alpha is not None else crop_person_mask.astype(np.float32)
            crop_background_confidence = _crop_resize_mask(background_confidence).astype(np.float32)
            crop_uncertainty_map = _crop_resize_mask(uncertainty_map).astype(np.float32)
            crop_occlusion_band = _crop_resize_mask(occlusion_band).astype(np.float32)
            crop_face_preserve = _crop_resize_mask(face_preserve_map).astype(np.float32)
            crop_face_confidence = _crop_resize_mask(face_confidence_map).astype(np.float32)
            crop_detail_release = _crop_resize_mask(detail_release_map).astype(np.float32)
            crop_trimap_unknown = _crop_resize_mask(trimap_unknown_map).astype(np.float32)
            crop_edge_detail = _crop_resize_mask(edge_detail_map).astype(np.float32)
            crop_face_boundary = _crop_resize_mask(face_boundary_map).astype(np.float32)
            crop_hair_boundary = _crop_resize_mask(hair_boundary_map).astype(np.float32)
            crop_hand_boundary = _crop_resize_mask(hand_boundary_map).astype(np.float32)
            crop_cloth_boundary = _crop_resize_mask(cloth_boundary_map).astype(np.float32)
            crop_occluded_boundary = _crop_resize_mask(occluded_boundary_map).astype(np.float32)
            crop_expert_mask = _crop_resize_mask(expert_mask).astype(np.float32)

            crop_refined, local_edge_debug = restore_local_edge_roi(
                original_rgb=crop_generated_float,
                refined_rgb=crop_generated_float,
                soft_alpha=crop_soft_alpha,
                outer_band=crop_soft_band,
                uncertainty_map=crop_uncertainty_map,
                detail_release_map=crop_detail_release,
                trimap_unknown_map=crop_trimap_unknown,
                edge_detail_map=crop_edge_detail,
                face_boundary_map=crop_face_boundary,
                hair_boundary_map=crop_hair_boundary,
                hand_boundary_map=crop_hand_boundary,
                cloth_boundary_map=crop_cloth_boundary,
                occluded_boundary_map=crop_occluded_boundary,
                sharpen=spec["sharpen"],
                detail_strength=spec["detail"],
                scale_factor=spec["scale"],
            )

            if spec["extra_unsharp"] > 0.0:
                crop_refined = apply_unsharp_mask(
                    crop_refined[None],
                    sigma=max(0.6, sharpen_sigma * 0.8),
                    amount=spec["extra_unsharp"],
                )[0]

            focus_full = _resize_mask_frame(local_edge_debug["local_edge_focus"], (crop_w, crop_h))
            gain_full = _resize_mask_frame(local_edge_debug["local_edge_gain"], (crop_w, crop_h))
            feather_full = _resize_mask_frame(local_edge_debug["local_edge_feather"], (crop_w, crop_h))
            expert_full = _resize_mask_frame(crop_expert_mask, (crop_w, crop_h))
            blend_full = np.clip(
                np.maximum(spec["core_w"] * expert_full, spec["feather_w"] * feather_full) * (0.55 + 0.45 * gain_full),
                0.0,
                1.0,
            ).astype(np.float32)
            if spec["name"] == "occluded":
                blend_full = np.clip(blend_full * (1.0 - 0.45 * _resize_mask_frame(crop_uncertainty_map, (crop_w, crop_h))), 0.0, 1.0)

            local_edge_focus_frames[frame_idx, y0:y1, x0:x1] = np.maximum(
                local_edge_focus_frames[frame_idx, y0:y1, x0:x1], focus_full
            )
            local_edge_gain_frames[frame_idx, y0:y1, x0:x1] = np.maximum(
                local_edge_gain_frames[frame_idx, y0:y1, x0:x1], gain_full
            )
            local_edge_feather_frames[frame_idx, y0:y1, x0:x1] = np.maximum(
                local_edge_feather_frames[frame_idx, y0:y1, x0:x1], feather_full
            )
            roi_blend_alpha[frame_idx, y0:y1, x0:x1] = np.maximum(
                roi_blend_alpha[frame_idx, y0:y1, x0:x1], blend_full
            )

            crop_refined_small = _resize_rgb_frame(
                np.clip(np.rint(crop_refined * 255.0), 0, 255).astype(np.uint8),
                (crop_w, crop_h),
            ).astype(np.float32) / 255.0
            current = refined_frames[frame_idx, y0:y1, x0:x1].astype(np.float32) / 255.0
            merged = np.clip(current * (1.0 - blend_full[..., None]) + crop_refined_small * blend_full[..., None], 0.0, 1.0)
            refined_frames[frame_idx, y0:y1, x0:x1] = np.clip(np.rint(merged * 255.0), 0, 255).astype(np.uint8)

    metrics = compute_boundary_refinement_metrics(
        before_frames=generated_frames.astype(np.float32) / 255.0,
        after_frames=refined_frames.astype(np.float32) / 255.0,
        background_frames=background_frames.astype(np.float32) / 255.0,
        outer_band=outer_band,
        inner_band=inner_band,
    )
    metrics.update(compute_boundary_roi_metrics(
        before_frames=generated_frames.astype(np.float32) / 255.0,
        after_frames=refined_frames.astype(np.float32) / 255.0,
        background_frames=background_frames.astype(np.float32) / 255.0,
        outer_band=outer_band,
        inner_band=inner_band,
        roi_mask=np.clip(np.maximum.reduce([face_boundary_map, hair_boundary_map, hand_boundary_map, cloth_boundary_map, occluded_boundary_map]), 0.0, 1.0),
    ))
    metrics.update(expert_coverages)
    metrics.update(expert_strengths)
    debug = {
        "outer_band": outer_band.astype(np.float32),
        "inner_band": inner_band.astype(np.float32),
        "soft_alpha": (soft_alpha if soft_alpha is not None else np.clip(person_mask + outer_band, 0.0, 1.0)).astype(np.float32),
        "sharpen_alpha": np.zeros_like(person_mask, dtype=np.float32),
        "band_blend": roi_blend_alpha.astype(np.float32),
        "target_foreground_alpha": (soft_alpha if soft_alpha is not None else np.clip(person_mask + outer_band, 0.0, 1.0)).astype(np.float32),
        "mode": "semantic_experts_v1",
        "metrics": metrics,
        "roi_mask": np.clip(np.maximum.reduce([face_boundary_map, hair_boundary_map, hand_boundary_map, cloth_boundary_map, occluded_boundary_map]), 0.0, 1.0).astype(np.float32),
        "roi_blend_alpha": roi_blend_alpha.astype(np.float32),
        "roi_area_ratio": float(np.maximum.reduce([face_boundary_map, hair_boundary_map, hand_boundary_map, cloth_boundary_map, occluded_boundary_map]).mean()),
        "local_edge_focus": local_edge_focus_frames.astype(np.float32),
        "local_edge_gain": local_edge_gain_frames.astype(np.float32),
        "local_edge_feather": local_edge_feather_frames.astype(np.float32),
        "face_boundary_map": face_boundary_map.astype(np.float32),
        "hair_boundary_map": hair_boundary_map.astype(np.float32),
        "hand_boundary_map": hand_boundary_map.astype(np.float32),
        "cloth_boundary_map": cloth_boundary_map.astype(np.float32),
        "occluded_boundary_map": occluded_boundary_map.astype(np.float32),
    }
    return refined_frames.astype(np.uint8), debug


def refine_boundary_frames(
    *,
    generated_frames: np.ndarray,
    background_frames: np.ndarray,
    person_mask: np.ndarray,
    soft_band: np.ndarray | None = None,
    soft_alpha: np.ndarray | None = None,
    background_confidence: np.ndarray | None = None,
    uncertainty_map: np.ndarray | None = None,
    occlusion_band: np.ndarray | None = None,
    face_preserve_map: np.ndarray | None = None,
    face_confidence_map: np.ndarray | None = None,
    detail_release_map: np.ndarray | None = None,
    trimap_unknown_map: np.ndarray | None = None,
    edge_detail_map: np.ndarray | None = None,
    face_boundary_map: np.ndarray | None = None,
    hair_boundary_map: np.ndarray | None = None,
    hand_boundary_map: np.ndarray | None = None,
    cloth_boundary_map: np.ndarray | None = None,
    occluded_boundary_map: np.ndarray | None = None,
    structure_guard_strength: float = 1.0,
    mode: str = "deterministic",
    strength: float = 0.35,
    sharpen: float = 0.15,
    inner_width: int = 2,
    sharpen_sigma: float = 1.0,
) -> tuple[np.ndarray, dict]:
    generated_frames = np.asarray(generated_frames, dtype=np.uint8)
    background_frames = np.asarray(background_frames, dtype=np.uint8)
    person_mask = np.asarray(person_mask, dtype=np.float32)
    soft_band = np.zeros_like(person_mask, dtype=np.float32) if soft_band is None else np.asarray(soft_band, dtype=np.float32)
    soft_alpha = None if soft_alpha is None else np.asarray(soft_alpha, dtype=np.float32)

    if generated_frames.shape != background_frames.shape:
        raise ValueError(f"Generated and background frames must match. Got {generated_frames.shape} vs {background_frames.shape}.")
    if generated_frames.shape[:3] != person_mask.shape:
        raise ValueError(f"person_mask must match frame shape. Got {person_mask.shape} vs {generated_frames.shape[:3]}.")
    if soft_band.shape != person_mask.shape:
        raise ValueError(f"soft_band must match person_mask shape. Got {soft_band.shape} vs {person_mask.shape}.")
    if soft_alpha is not None and soft_alpha.shape != person_mask.shape:
        raise ValueError(f"soft_alpha must match person_mask shape. Got {soft_alpha.shape} vs {person_mask.shape}.")
    for name, value in (
        ("background_confidence", background_confidence),
        ("uncertainty_map", uncertainty_map),
        ("occlusion_band", occlusion_band),
        ("face_preserve_map", face_preserve_map),
        ("face_confidence_map", face_confidence_map),
        ("detail_release_map", detail_release_map),
        ("trimap_unknown_map", trimap_unknown_map),
        ("edge_detail_map", edge_detail_map),
        ("face_boundary_map", face_boundary_map),
        ("hair_boundary_map", hair_boundary_map),
        ("hand_boundary_map", hand_boundary_map),
        ("cloth_boundary_map", cloth_boundary_map),
        ("occluded_boundary_map", occluded_boundary_map),
    ):
        if value is not None and np.asarray(value).shape != person_mask.shape:
            raise ValueError(f"{name} must match person_mask shape. Got {np.asarray(value).shape} vs {person_mask.shape}.")

    strength = max(0.0, min(float(strength), 1.0))
    sharpen = max(0.0, min(float(sharpen), 1.0))
    person_mask = np.clip(person_mask, 0.0, 1.0).astype(np.float32)
    outer_band = np.clip(soft_band, 0.0, 1.0).astype(np.float32)
    if soft_alpha is not None:
        soft_alpha = np.clip(soft_alpha, 0.0, 1.0).astype(np.float32)
    inner_band = build_inner_boundary_band(person_mask, inner_width=inner_width)

    generated = generated_frames.astype(np.float32) / 255.0
    background = background_frames.astype(np.float32) / 255.0
    sharpened = apply_unsharp_mask(generated, sigma=sharpen_sigma, amount=sharpen)

    target_foreground_alpha = (
        np.clip(soft_alpha, 0.0, 1.0)[..., None]
        if soft_alpha is not None
        else np.clip(person_mask + outer_band, 0.0, 1.0)[..., None]
    )

    if mode not in {"deterministic", "v2", "roi_v1", "semantic_v1", "semantic_experts_v1", "local_edge_v1", "roi_gen_v1"}:
        raise ValueError(f"Unsupported boundary refinement mode: {mode}")

    if mode == "semantic_experts_v1":
        background_confidence_local = (
            np.ones_like(person_mask, dtype=np.float32)
            if background_confidence is None else np.clip(np.asarray(background_confidence, dtype=np.float32), 0.0, 1.0)
        )
        uncertainty_map_local = (
            np.zeros_like(person_mask, dtype=np.float32)
            if uncertainty_map is None else np.clip(np.asarray(uncertainty_map, dtype=np.float32), 0.0, 1.0)
        )
        occlusion_band_local = (
            np.zeros_like(person_mask, dtype=np.float32)
            if occlusion_band is None else np.clip(np.asarray(occlusion_band, dtype=np.float32), 0.0, 1.0)
        )
        face_preserve_local = (
            np.zeros_like(person_mask, dtype=np.float32)
            if face_preserve_map is None else np.clip(np.asarray(face_preserve_map, dtype=np.float32), 0.0, 1.0)
        )
        face_confidence_local = (
            np.zeros_like(person_mask, dtype=np.float32)
            if face_confidence_map is None else np.clip(np.asarray(face_confidence_map, dtype=np.float32), 0.0, 1.0)
        )
        detail_release_local = (
            np.zeros_like(person_mask, dtype=np.float32)
            if detail_release_map is None else np.clip(np.asarray(detail_release_map, dtype=np.float32), 0.0, 1.0)
        )
        trimap_unknown_local = (
            np.zeros_like(person_mask, dtype=np.float32)
            if trimap_unknown_map is None else np.clip(np.asarray(trimap_unknown_map, dtype=np.float32), 0.0, 1.0)
        )
        edge_detail_local = (
            np.zeros_like(person_mask, dtype=np.float32)
            if edge_detail_map is None else np.clip(np.asarray(edge_detail_map, dtype=np.float32), 0.0, 1.0)
        )
        face_boundary_local = (
            np.zeros_like(person_mask, dtype=np.float32)
            if face_boundary_map is None else np.clip(np.asarray(face_boundary_map, dtype=np.float32), 0.0, 1.0)
        )
        hair_boundary_local = (
            np.zeros_like(person_mask, dtype=np.float32)
            if hair_boundary_map is None else np.clip(np.asarray(hair_boundary_map, dtype=np.float32), 0.0, 1.0)
        )
        hand_boundary_local = (
            np.zeros_like(person_mask, dtype=np.float32)
            if hand_boundary_map is None else np.clip(np.asarray(hand_boundary_map, dtype=np.float32), 0.0, 1.0)
        )
        cloth_boundary_local = (
            np.zeros_like(person_mask, dtype=np.float32)
            if cloth_boundary_map is None else np.clip(np.asarray(cloth_boundary_map, dtype=np.float32), 0.0, 1.0)
        )
        occluded_boundary_local = (
            np.zeros_like(person_mask, dtype=np.float32)
            if occluded_boundary_map is None else np.clip(np.asarray(occluded_boundary_map, dtype=np.float32), 0.0, 1.0)
        )
        return _apply_semantic_roi_experts(
            generated_frames=generated_frames,
            background_frames=background_frames,
            person_mask=person_mask,
            outer_band=outer_band,
            inner_band=inner_band,
            soft_alpha=soft_alpha,
            background_confidence=background_confidence_local,
            uncertainty_map=uncertainty_map_local,
            occlusion_band=occlusion_band_local,
            face_preserve_map=face_preserve_local,
            face_confidence_map=face_confidence_local,
            detail_release_map=detail_release_local,
            trimap_unknown_map=trimap_unknown_local,
            edge_detail_map=edge_detail_local,
            face_boundary_map=face_boundary_local,
            hair_boundary_map=hair_boundary_local,
            hand_boundary_map=hand_boundary_local,
            cloth_boundary_map=cloth_boundary_local,
            occluded_boundary_map=occluded_boundary_local,
            strength=strength,
            sharpen=sharpen,
            inner_width=inner_width,
            sharpen_sigma=sharpen_sigma,
        )

    if mode in {"roi_v1", "local_edge_v1", "roi_gen_v1"}:
        background_confidence_local = (
            np.ones_like(person_mask, dtype=np.float32)
            if background_confidence is None else np.clip(np.asarray(background_confidence, dtype=np.float32), 0.0, 1.0)
        )
        uncertainty_map_local = (
            np.zeros_like(person_mask, dtype=np.float32)
            if uncertainty_map is None else np.clip(np.asarray(uncertainty_map, dtype=np.float32), 0.0, 1.0)
        )
        occlusion_band_local = (
            np.zeros_like(person_mask, dtype=np.float32)
            if occlusion_band is None else np.clip(np.asarray(occlusion_band, dtype=np.float32), 0.0, 1.0)
        )
        face_preserve_local = (
            np.zeros_like(person_mask, dtype=np.float32)
            if face_preserve_map is None else np.clip(np.asarray(face_preserve_map, dtype=np.float32), 0.0, 1.0)
        )
        face_confidence_local = (
            np.zeros_like(person_mask, dtype=np.float32)
            if face_confidence_map is None else np.clip(np.asarray(face_confidence_map, dtype=np.float32), 0.0, 1.0)
        )
        detail_release_local = (
            np.zeros_like(person_mask, dtype=np.float32)
            if detail_release_map is None else np.clip(np.asarray(detail_release_map, dtype=np.float32), 0.0, 1.0)
        )
        trimap_unknown_local = (
            np.zeros_like(person_mask, dtype=np.float32)
            if trimap_unknown_map is None else np.clip(np.asarray(trimap_unknown_map, dtype=np.float32), 0.0, 1.0)
        )
        edge_detail_local = (
            np.zeros_like(person_mask, dtype=np.float32)
            if edge_detail_map is None else np.clip(np.asarray(edge_detail_map, dtype=np.float32), 0.0, 1.0)
        )
        face_boundary_local = (
            np.zeros_like(person_mask, dtype=np.float32)
            if face_boundary_map is None else np.clip(np.asarray(face_boundary_map, dtype=np.float32), 0.0, 1.0)
        )
        hair_boundary_local = (
            np.zeros_like(person_mask, dtype=np.float32)
            if hair_boundary_map is None else np.clip(np.asarray(hair_boundary_map, dtype=np.float32), 0.0, 1.0)
        )
        hand_boundary_local = (
            np.zeros_like(person_mask, dtype=np.float32)
            if hand_boundary_map is None else np.clip(np.asarray(hand_boundary_map, dtype=np.float32), 0.0, 1.0)
        )
        cloth_boundary_local = (
            np.zeros_like(person_mask, dtype=np.float32)
            if cloth_boundary_map is None else np.clip(np.asarray(cloth_boundary_map, dtype=np.float32), 0.0, 1.0)
        )
        occluded_boundary_local = (
            np.zeros_like(person_mask, dtype=np.float32)
            if occluded_boundary_map is None else np.clip(np.asarray(occluded_boundary_map, dtype=np.float32), 0.0, 1.0)
        )

        roi_mask = build_boundary_roi_mask(
            person_mask=person_mask,
            outer_band=outer_band,
            inner_band=inner_band,
            soft_alpha=soft_alpha,
            occlusion_band=occlusion_band_local,
            uncertainty_map=uncertainty_map_local,
            face_preserve_map=face_preserve_local,
            detail_release_map=detail_release_local,
            trimap_unknown_map=trimap_unknown_local,
            edge_detail_map=edge_detail_local,
        )
        roi_boxes = _compute_roi_boxes(roi_mask, min_size=96, pad=20)
        refined_frames = generated_frames.copy()
        roi_blend_alpha = np.zeros_like(person_mask, dtype=np.float32)
        roi_scale_values = []
        local_edge_focus_frames = np.zeros_like(person_mask, dtype=np.float32)
        local_edge_gain_frames = np.zeros_like(person_mask, dtype=np.float32)
        local_edge_feather_frames = np.zeros_like(person_mask, dtype=np.float32)

        for frame_idx, (x0, y0, x1, y1) in enumerate(roi_boxes):
            crop_w = max(1, x1 - x0)
            crop_h = max(1, y1 - y0)
            if crop_w <= 1 or crop_h <= 1:
                continue
            scale = 2.1 if max(crop_w, crop_h) <= 256 else 1.6
            roi_scale_values.append(float(scale))
            target_size = (max(32, int(round(crop_w * scale))), max(32, int(round(crop_h * scale))))

            def _crop_resize_rgb(arr):
                return _resize_rgb_frame(arr[frame_idx, y0:y1, x0:x1], target_size)

            def _crop_resize_mask(arr):
                return _resize_mask_frame(arr[frame_idx, y0:y1, x0:x1], target_size)

            crop_generated = _crop_resize_rgb(generated_frames)[None]
            crop_background = _crop_resize_rgb(background_frames)[None]
            crop_person_mask = _crop_resize_mask(person_mask)[None]
            crop_soft_band = _crop_resize_mask(outer_band)[None]
            crop_soft_alpha = _crop_resize_mask(soft_alpha)[None] if soft_alpha is not None else None
            crop_background_confidence = _crop_resize_mask(background_confidence_local)[None]
            crop_uncertainty_map = _crop_resize_mask(uncertainty_map_local)[None]
            crop_occlusion_band = _crop_resize_mask(occlusion_band_local)[None]
            crop_face_preserve = _crop_resize_mask(face_preserve_local)[None]
            crop_face_confidence = _crop_resize_mask(face_confidence_local)[None]
            crop_detail_release = _crop_resize_mask(detail_release_local)[None]
            crop_trimap_unknown = _crop_resize_mask(trimap_unknown_local)[None]
            crop_edge_detail = _crop_resize_mask(edge_detail_local)[None]
            crop_face_boundary = _crop_resize_mask(face_boundary_local)[None]
            crop_hair_boundary = _crop_resize_mask(hair_boundary_local)[None]
            crop_hand_boundary = _crop_resize_mask(hand_boundary_local)[None]
            crop_cloth_boundary = _crop_resize_mask(cloth_boundary_local)[None]
            crop_occluded_boundary = _crop_resize_mask(occluded_boundary_local)[None]

            crop_refined, crop_debug = refine_boundary_frames(
                generated_frames=crop_generated,
                background_frames=crop_background,
                person_mask=crop_person_mask,
                soft_band=crop_soft_band,
                soft_alpha=crop_soft_alpha,
                background_confidence=crop_background_confidence,
                uncertainty_map=crop_uncertainty_map,
                occlusion_band=crop_occlusion_band,
                face_preserve_map=crop_face_preserve,
                face_confidence_map=crop_face_confidence,
                detail_release_map=crop_detail_release if mode == "local_edge_v1" else None,
                trimap_unknown_map=crop_trimap_unknown if mode == "local_edge_v1" else None,
                edge_detail_map=crop_edge_detail if mode == "local_edge_v1" else None,
                face_boundary_map=crop_face_boundary if mode in {"local_edge_v1", "roi_gen_v1"} else None,
                hair_boundary_map=crop_hair_boundary if mode in {"local_edge_v1", "roi_gen_v1"} else None,
                hand_boundary_map=crop_hand_boundary if mode in {"local_edge_v1", "roi_gen_v1"} else None,
                cloth_boundary_map=crop_cloth_boundary if mode in {"local_edge_v1", "roi_gen_v1"} else None,
                occluded_boundary_map=crop_occluded_boundary if mode in {"local_edge_v1", "roi_gen_v1"} else None,
                structure_guard_strength=structure_guard_strength,
                mode="semantic_v1",
                strength=min(1.0, strength + 0.14),
                sharpen=min(1.0, sharpen + 0.14),
                inner_width=max(inner_width, 2),
                sharpen_sigma=max(0.8, sharpen_sigma),
            )
            crop_refined = crop_refined[0].astype(np.float32) / 255.0
            crop_generated_float = crop_generated[0].astype(np.float32) / 255.0

            if mode in {"local_edge_v1", "roi_gen_v1"}:
                crop_refined, local_edge_debug = restore_local_edge_roi(
                    original_rgb=crop_generated_float,
                    refined_rgb=crop_refined,
                    soft_alpha=crop_soft_alpha[0] if crop_soft_alpha is not None else crop_person_mask[0],
                    outer_band=crop_soft_band[0],
                    uncertainty_map=crop_uncertainty_map[0],
                    detail_release_map=crop_detail_release[0],
                    trimap_unknown_map=crop_trimap_unknown[0],
                    edge_detail_map=crop_edge_detail[0],
                    face_boundary_map=crop_face_boundary[0],
                    hair_boundary_map=crop_hair_boundary[0],
                    hand_boundary_map=crop_hand_boundary[0],
                    cloth_boundary_map=crop_cloth_boundary[0],
                    occluded_boundary_map=crop_occluded_boundary[0],
                    sharpen=min(1.0, sharpen + 0.08),
                    detail_strength=min(1.0, strength + 0.06),
                    scale_factor=2.0 if max(crop_w, crop_h) <= 280 else 1.7,
                )
                local_edge_focus_full = _resize_mask_frame(local_edge_debug["local_edge_focus"], (crop_w, crop_h))
                local_edge_gain_full = _resize_mask_frame(local_edge_debug["local_edge_gain"], (crop_w, crop_h))
                local_edge_feather_full = _resize_mask_frame(local_edge_debug["local_edge_feather"], (crop_w, crop_h))
                local_edge_focus_frames[frame_idx, y0:y1, x0:x1] = np.maximum(
                    local_edge_focus_frames[frame_idx, y0:y1, x0:x1],
                    local_edge_focus_full,
                )
                local_edge_gain_frames[frame_idx, y0:y1, x0:x1] = np.maximum(
                    local_edge_gain_frames[frame_idx, y0:y1, x0:x1],
                    local_edge_gain_full,
                )
                local_edge_feather_frames[frame_idx, y0:y1, x0:x1] = np.maximum(
                    local_edge_feather_frames[frame_idx, y0:y1, x0:x1],
                    local_edge_feather_full,
                )

            if mode == "roi_gen_v1":
                crop_background_float = crop_background[0].astype(np.float32) / 255.0
                crop_outer_band = crop_soft_band[0].astype(np.float32)
                crop_inner_band = build_inner_boundary_band(crop_person_mask, inner_width=max(inner_width, 2))[0]
                candidate_specs = [
                    ("base", crop_refined.astype(np.float32)),
                    (
                        "detail_mid",
                        restore_local_edge_roi(
                            original_rgb=crop_generated_float,
                            refined_rgb=crop_refined.astype(np.float32),
                            soft_alpha=crop_soft_alpha[0] if crop_soft_alpha is not None else crop_person_mask[0],
                            outer_band=crop_outer_band,
                            uncertainty_map=crop_uncertainty_map[0],
                            detail_release_map=crop_detail_release[0],
                            trimap_unknown_map=crop_trimap_unknown[0],
                            edge_detail_map=crop_edge_detail[0],
                            face_boundary_map=crop_face_boundary[0],
                            hair_boundary_map=crop_hair_boundary[0],
                            hand_boundary_map=crop_hand_boundary[0],
                            cloth_boundary_map=crop_cloth_boundary[0],
                            occluded_boundary_map=crop_occluded_boundary[0],
                            sharpen=min(1.0, sharpen + 0.12),
                            detail_strength=min(1.0, strength + 0.12),
                            scale_factor=2.2 if max(crop_w, crop_h) <= 280 else 1.8,
                        )[0].astype(np.float32),
                    ),
                    (
                        "detail_high",
                        restore_local_edge_roi(
                            original_rgb=crop_generated_float,
                            refined_rgb=crop_refined.astype(np.float32),
                            soft_alpha=crop_soft_alpha[0] if crop_soft_alpha is not None else crop_person_mask[0],
                            outer_band=crop_outer_band,
                            uncertainty_map=crop_uncertainty_map[0],
                            detail_release_map=crop_detail_release[0],
                            trimap_unknown_map=crop_trimap_unknown[0],
                            edge_detail_map=crop_edge_detail[0],
                            face_boundary_map=crop_face_boundary[0],
                            hair_boundary_map=crop_hair_boundary[0],
                            hand_boundary_map=crop_hand_boundary[0],
                            cloth_boundary_map=crop_cloth_boundary[0],
                            occluded_boundary_map=crop_occluded_boundary[0],
                            sharpen=min(1.0, sharpen + 0.18),
                            detail_strength=min(1.0, strength + 0.20),
                            scale_factor=2.5 if max(crop_w, crop_h) <= 280 else 2.0,
                        )[0].astype(np.float32),
                    ),
                    (
                        "contrast_high",
                        np.clip(
                            apply_unsharp_mask(
                                restore_local_edge_roi(
                                    original_rgb=crop_generated_float,
                                    refined_rgb=crop_refined.astype(np.float32),
                                    soft_alpha=crop_soft_alpha[0] if crop_soft_alpha is not None else crop_person_mask[0],
                                    outer_band=crop_outer_band,
                                    uncertainty_map=crop_uncertainty_map[0],
                                    detail_release_map=crop_detail_release[0],
                                    trimap_unknown_map=crop_trimap_unknown[0],
                                    edge_detail_map=crop_edge_detail[0],
                                    face_boundary_map=crop_face_boundary[0],
                                    hair_boundary_map=crop_hair_boundary[0],
                                    hand_boundary_map=crop_hand_boundary[0],
                                    cloth_boundary_map=crop_cloth_boundary[0],
                                    occluded_boundary_map=crop_occluded_boundary[0],
                                    sharpen=min(1.0, sharpen + 0.22),
                                    detail_strength=min(1.0, strength + 0.26),
                                    scale_factor=2.6 if max(crop_w, crop_h) <= 280 else 2.1,
                                )[0][None],
                                sigma=max(0.7, sharpen_sigma * 0.8),
                                amount=min(1.0, sharpen + 0.28),
                            )[0],
                            0.0,
                            1.0,
                        ).astype(np.float32),
                    ),
                ]
                scored = []
                for candidate_name, candidate_rgb in candidate_specs:
                    candidate_score, candidate_metrics = _score_roi_candidate(
                        before_rgb=crop_generated_float,
                        candidate_rgb=candidate_rgb,
                        background_rgb=crop_background_float,
                        outer_band=crop_outer_band,
                        inner_band=crop_inner_band,
                    )
                    scored.append((candidate_score, candidate_name, candidate_rgb, candidate_metrics))
                scored.sort(key=lambda item: item[0], reverse=True)
                crop_refined = scored[0][2]

            crop_edge_focus = np.clip(
                np.maximum.reduce(
                    [
                        0.95 * crop_soft_band[0],
                        0.75 * crop_person_mask[0],
                        0.70 * crop_detail_release[0],
                        0.65 * crop_edge_detail[0],
                        0.55 * crop_trimap_unknown[0],
                    ]
                ),
                0.0,
                1.0,
            ).astype(np.float32)
            crop_edge_focus = crop_edge_focus * np.clip(1.0 - 0.25 * crop_uncertainty_map[0], 0.0, 1.0)
            crop_generated_sharp = apply_unsharp_mask(
                crop_generated_float[None],
                sigma=max(0.7, sharpen_sigma * 0.85),
                amount=min(1.0, sharpen + 0.32),
            )[0]
            crop_detail_residual = np.clip(crop_generated_sharp - crop_generated_float, -0.5, 0.5)
            crop_refined = np.clip(
                crop_refined + crop_edge_focus[..., None] * 0.24 * crop_detail_residual,
                0.0,
                1.0,
            )

            core_focus = np.clip(
                np.maximum.reduce(
                    [
                        roi_mask[frame_idx, y0:y1, x0:x1],
                        0.85 * outer_band[frame_idx, y0:y1, x0:x1],
                        0.60 * inner_band[frame_idx, y0:y1, x0:x1],
                    ]
                ),
                0.0,
                1.0,
            ).astype(np.float32)
            feather = core_focus.copy()
            feather = _dilate_mask(feather, kernel_size=9, iterations=1)
            feather = cv2.GaussianBlur(feather, (11, 11), sigmaX=0)
            feather = np.clip(feather, 0.0, 1.0).astype(np.float32)
            feather_small = _resize_mask_frame(feather, target_size)
            feather_full = _resize_mask_frame(feather_small, (crop_w, crop_h))
            core_small = _resize_mask_frame(core_focus, target_size)
            core_full = _resize_mask_frame(core_small, (crop_w, crop_h))
            blend_full = np.clip(np.maximum(0.80 * core_full, 0.40 * feather_full), 0.0, 1.0).astype(np.float32)
            roi_blend_alpha[frame_idx, y0:y1, x0:x1] = np.maximum(
                roi_blend_alpha[frame_idx, y0:y1, x0:x1],
                blend_full,
            )

            crop_refined_small = _resize_rgb_frame(
                np.clip(np.rint(crop_refined * 255.0), 0, 255).astype(np.uint8),
                (crop_w, crop_h),
            ).astype(np.float32) / 255.0
            current = refined_frames[frame_idx, y0:y1, x0:x1].astype(np.float32) / 255.0
            blend = blend_full[..., None]
            merged = np.clip(current * (1.0 - blend) + crop_refined_small * blend, 0.0, 1.0)
            refined_frames[frame_idx, y0:y1, x0:x1] = np.clip(np.rint(merged * 255.0), 0, 255).astype(np.uint8)

        metrics = compute_boundary_refinement_metrics(
            before_frames=generated,
            after_frames=refined_frames.astype(np.float32) / 255.0,
            background_frames=background,
            outer_band=outer_band,
            inner_band=inner_band,
        )
        roi_metrics = compute_boundary_roi_metrics(
            before_frames=generated,
            after_frames=refined_frames.astype(np.float32) / 255.0,
            background_frames=background,
            outer_band=outer_band,
            inner_band=inner_band,
            roi_mask=roi_mask,
        )
        metrics.update(roi_metrics)
        debug = {
            "outer_band": outer_band,
            "inner_band": inner_band,
            "soft_alpha": target_foreground_alpha[..., 0],
            "sharpen_alpha": np.zeros_like(person_mask, dtype=np.float32),
            "band_blend": roi_blend_alpha.astype(np.float32),
            "target_foreground_alpha": target_foreground_alpha[..., 0],
            "mode": mode,
            "metrics": metrics,
            "roi_mask": roi_mask.astype(np.float32),
            "roi_blend_alpha": roi_blend_alpha.astype(np.float32),
            "roi_area_ratio": float(roi_mask.mean()),
            "roi_scale_mean": float(np.mean(roi_scale_values)) if roi_scale_values else 1.0,
            "local_edge_focus": local_edge_focus_frames.astype(np.float32),
            "local_edge_gain": local_edge_gain_frames.astype(np.float32),
            "local_edge_feather": local_edge_feather_frames.astype(np.float32),
            "roi_boxes": [
                {"frame_index": int(i), "x0": int(box[0]), "y0": int(box[1]), "x1": int(box[2]), "y1": int(box[3])}
                for i, box in enumerate(roi_boxes)
            ],
        }
        return refined_frames.astype(np.uint8), debug

    if mode == "deterministic":
        sharpen_alpha = np.clip(inner_band * sharpen, 0.0, 1.0)[..., None]
        refined = generated * (1.0 - sharpen_alpha) + sharpened * sharpen_alpha
        band_blend = np.clip(outer_band * strength, 0.0, 1.0)[..., None]
        composite = refined * target_foreground_alpha + background * (1.0 - target_foreground_alpha)
        refined = refined * (1.0 - band_blend) + composite * band_blend
        extra_debug = {
            "background_confidence": np.ones_like(person_mask, dtype=np.float32),
            "uncertainty_map": np.zeros_like(person_mask, dtype=np.float32),
            "occlusion_band": np.zeros_like(person_mask, dtype=np.float32),
            "face_preserve_map": np.zeros_like(person_mask, dtype=np.float32),
            "face_confidence_map": np.zeros_like(person_mask, dtype=np.float32),
            "structure_guard_map": np.full_like(person_mask, float(np.clip(structure_guard_strength, 0.0, 1.0)), dtype=np.float32),
            "adaptive_composite_strength": band_blend[..., 0],
            "adaptive_sharpen_alpha": sharpen_alpha[..., 0],
        }
    else:
        background_confidence = (
            np.ones_like(person_mask, dtype=np.float32)
            if background_confidence is None else np.clip(np.asarray(background_confidence, dtype=np.float32), 0.0, 1.0)
        )
        uncertainty_map = (
            np.zeros_like(person_mask, dtype=np.float32)
            if uncertainty_map is None else np.clip(np.asarray(uncertainty_map, dtype=np.float32), 0.0, 1.0)
        )
        occlusion_band = (
            np.zeros_like(person_mask, dtype=np.float32)
            if occlusion_band is None else np.clip(np.asarray(occlusion_band, dtype=np.float32), 0.0, 1.0)
        )
        face_preserve_map = (
            np.zeros_like(person_mask, dtype=np.float32)
            if face_preserve_map is None else np.clip(np.asarray(face_preserve_map, dtype=np.float32), 0.0, 1.0)
        )
        face_confidence_map = (
            np.zeros_like(person_mask, dtype=np.float32)
            if face_confidence_map is None else np.clip(np.asarray(face_confidence_map, dtype=np.float32), 0.0, 1.0)
        )
        face_boundary_map = (
            np.zeros_like(person_mask, dtype=np.float32)
            if face_boundary_map is None else np.clip(np.asarray(face_boundary_map, dtype=np.float32), 0.0, 1.0)
        )
        hair_boundary_map = (
            np.zeros_like(person_mask, dtype=np.float32)
            if hair_boundary_map is None else np.clip(np.asarray(hair_boundary_map, dtype=np.float32), 0.0, 1.0)
        )
        hand_boundary_map = (
            np.zeros_like(person_mask, dtype=np.float32)
            if hand_boundary_map is None else np.clip(np.asarray(hand_boundary_map, dtype=np.float32), 0.0, 1.0)
        )
        cloth_boundary_map = (
            np.zeros_like(person_mask, dtype=np.float32)
            if cloth_boundary_map is None else np.clip(np.asarray(cloth_boundary_map, dtype=np.float32), 0.0, 1.0)
        )
        occluded_boundary_map = (
            np.zeros_like(person_mask, dtype=np.float32)
            if occluded_boundary_map is None else np.clip(np.asarray(occluded_boundary_map, dtype=np.float32), 0.0, 1.0)
        )
        structure_guard = float(np.clip(structure_guard_strength, 0.70, 1.0))
        target_alpha_2d = target_foreground_alpha[..., 0]
        outer_focus = np.clip(outer_band * (1.0 - target_alpha_2d), 0.0, 1.0)
        edge_focus = np.clip(outer_band * target_alpha_2d, 0.0, 1.0)

        adaptive_sharpen = np.clip(0.85 * inner_band + 0.65 * edge_focus, 0.0, 1.0)
        adaptive_sharpen = adaptive_sharpen * (1.0 - 0.85 * uncertainty_map)
        adaptive_sharpen = adaptive_sharpen * (0.35 + 0.65 * background_confidence)
        adaptive_sharpen = adaptive_sharpen * (1.0 - 0.45 * occlusion_band)
        adaptive_sharpen = adaptive_sharpen * structure_guard
        adaptive_sharpen = np.clip(
            adaptive_sharpen + 0.18 * np.clip(face_preserve_map * (0.4 + 0.6 * face_confidence_map), 0.0, 1.0),
            0.0,
            1.0,
        )
        adaptive_blend = np.clip(outer_focus * strength, 0.0, 1.0)
        adaptive_blend = adaptive_blend * (0.25 + 0.75 * background_confidence)
        adaptive_blend = adaptive_blend * (1.0 - 0.80 * uncertainty_map)
        adaptive_blend = adaptive_blend * (1.0 - 0.55 * occlusion_band)
        adaptive_blend = adaptive_blend * (1.0 - 0.50 * face_preserve_map)
        adaptive_blend = np.clip(adaptive_blend * structure_guard, 0.0, 1.0)
        edge_boost = np.clip(edge_focus * (0.55 + 0.45 * target_alpha_2d), 0.0, 1.0)
        edge_boost = edge_boost * (1.0 - 0.45 * uncertainty_map)
        edge_boost = edge_boost * (0.35 + 0.65 * background_confidence)
        edge_boost = edge_boost * (1.0 - 0.40 * occlusion_band)
        edge_boost = edge_boost * (1.0 - 0.35 * face_preserve_map)
        edge_boost = np.clip(edge_boost * structure_guard, 0.0, 1.0)
        if mode == "semantic_v1":
            hair_focus = np.clip(hair_boundary_map, 0.0, 1.0)
            face_focus = np.clip(face_boundary_map, 0.0, 1.0)
            hand_focus = np.clip(hand_boundary_map, 0.0, 1.0)
            cloth_focus = np.clip(cloth_boundary_map, 0.0, 1.0)
            occluded_focus = np.clip(occluded_boundary_map, 0.0, 1.0)

            adaptive_sharpen = np.clip(
                adaptive_sharpen
                * (1.0 + 0.22 * face_focus + 0.18 * hand_focus + 0.16 * cloth_focus)
                * (1.0 - 0.12 * hair_focus)
                * (1.0 - 0.45 * occluded_focus),
                0.0,
                1.0,
            )
            adaptive_blend = np.clip(
                adaptive_blend
                * (1.0 - 0.28 * face_focus - 0.06 * hair_focus - 0.16 * hand_focus)
                * (1.0 + 0.26 * occluded_focus),
                0.0,
                1.0,
            )
            band_blend = adaptive_blend[..., None]
            edge_boost = np.clip(
                edge_boost
                * (1.0 + 0.28 * face_focus + 0.08 * hair_focus + 0.28 * hand_focus + 0.20 * cloth_focus)
                * (1.0 - 0.50 * occluded_focus),
                0.0,
                1.0,
            )
        sharpen_alpha = np.clip(adaptive_sharpen * sharpen, 0.0, 1.0)[..., None]
        refined = generated * (1.0 - sharpen_alpha) + sharpened * sharpen_alpha
        band_blend = adaptive_blend[..., None]
        composite = refined * target_foreground_alpha + background * (1.0 - target_foreground_alpha)
        refined = refined * (1.0 - band_blend) + composite * band_blend
        edge_boost_alpha = np.clip(edge_boost * (0.30 + 0.70 * sharpen), 0.0, 1.0)[..., None]
        edge_detail = np.clip(sharpened - background, -1.0, 1.0)
        refined = np.clip(refined + edge_boost_alpha * 0.24 * edge_detail, 0.0, 1.0)
        extra_debug = {
            "background_confidence": background_confidence,
            "uncertainty_map": uncertainty_map,
            "occlusion_band": occlusion_band,
            "face_preserve_map": face_preserve_map,
            "face_confidence_map": face_confidence_map,
            "structure_guard_map": np.full_like(person_mask, structure_guard, dtype=np.float32),
            "adaptive_composite_strength": adaptive_blend.astype(np.float32),
            "adaptive_sharpen_alpha": sharpen_alpha[..., 0].astype(np.float32),
            "outer_focus": outer_focus.astype(np.float32),
            "edge_focus": edge_focus.astype(np.float32),
            "edge_boost_alpha": edge_boost_alpha[..., 0].astype(np.float32),
            "face_boundary_map": face_boundary_map.astype(np.float32),
            "hair_boundary_map": hair_boundary_map.astype(np.float32),
            "hand_boundary_map": hand_boundary_map.astype(np.float32),
            "cloth_boundary_map": cloth_boundary_map.astype(np.float32),
            "occluded_boundary_map": occluded_boundary_map.astype(np.float32),
        }

    refined = np.clip(refined, 0.0, 1.0)
    refined_u8 = np.clip(np.rint(refined * 255.0), 0, 255).astype(np.uint8)

    metrics = compute_boundary_refinement_metrics(
        before_frames=generated,
        after_frames=refined,
        background_frames=background,
        outer_band=outer_band,
        inner_band=inner_band,
    )
    debug = {
        "outer_band": outer_band,
        "inner_band": inner_band,
        "soft_alpha": target_foreground_alpha[..., 0],
        "sharpen_alpha": sharpen_alpha[..., 0],
        "band_blend": band_blend[..., 0],
        "target_foreground_alpha": target_foreground_alpha[..., 0],
        "mode": mode,
        "metrics": metrics,
    }
    debug.update(extra_debug)
    return refined_u8, debug


def _write_rgb_png(path: Path, frame_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError(f"Failed to write PNG frame: {path}")


def _dilate_mask(mask: np.ndarray, kernel_size: int, iterations: int = 1) -> np.ndarray:
    if kernel_size <= 1:
        return mask.astype(np.float32)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    hard = (mask > 1e-3).astype(np.uint8)
    dilated = cv2.dilate(hard, kernel, iterations=max(1, int(iterations)))
    return dilated.astype(np.float32)


def build_boundary_roi_mask(
    *,
    person_mask: np.ndarray,
    outer_band: np.ndarray,
    inner_band: np.ndarray,
    soft_alpha: np.ndarray | None = None,
    occlusion_band: np.ndarray | None = None,
    uncertainty_map: np.ndarray | None = None,
    face_preserve_map: np.ndarray | None = None,
    detail_release_map: np.ndarray | None = None,
    trimap_unknown_map: np.ndarray | None = None,
    edge_detail_map: np.ndarray | None = None,
    dilate_kernel: int = 17,
) -> np.ndarray:
    person_mask = np.asarray(person_mask, dtype=np.float32)
    outer_band = np.asarray(outer_band, dtype=np.float32)
    inner_band = np.asarray(inner_band, dtype=np.float32)
    soft_alpha = np.zeros_like(person_mask, dtype=np.float32) if soft_alpha is None else np.asarray(soft_alpha, dtype=np.float32)
    occlusion_band = np.zeros_like(person_mask, dtype=np.float32) if occlusion_band is None else np.asarray(occlusion_band, dtype=np.float32)
    uncertainty_map = np.zeros_like(person_mask, dtype=np.float32) if uncertainty_map is None else np.asarray(uncertainty_map, dtype=np.float32)
    face_preserve_map = np.zeros_like(person_mask, dtype=np.float32) if face_preserve_map is None else np.asarray(face_preserve_map, dtype=np.float32)
    detail_release_map = np.zeros_like(person_mask, dtype=np.float32) if detail_release_map is None else np.asarray(detail_release_map, dtype=np.float32)
    trimap_unknown_map = np.zeros_like(person_mask, dtype=np.float32) if trimap_unknown_map is None else np.asarray(trimap_unknown_map, dtype=np.float32)
    edge_detail_map = np.zeros_like(person_mask, dtype=np.float32) if edge_detail_map is None else np.asarray(edge_detail_map, dtype=np.float32)

    alpha_transition = np.clip(4.0 * soft_alpha * (1.0 - soft_alpha), 0.0, 1.0).astype(np.float32)
    base = np.clip(
        np.maximum.reduce(
            [
                outer_band,
                0.85 * inner_band,
                0.75 * alpha_transition,
                0.60 * occlusion_band,
                0.45 * detail_release_map,
                0.45 * edge_detail_map,
                0.35 * trimap_unknown_map,
                0.30 * face_preserve_map,
            ]
        ),
        0.0,
        1.0,
    ).astype(np.float32)
    base = base * np.clip(1.0 - 0.20 * uncertainty_map, 0.0, 1.0)

    roi_masks = []
    for idx in range(base.shape[0]):
        roi = _dilate_mask(base[idx], kernel_size=dilate_kernel, iterations=1)
        roi = np.clip(np.maximum(roi, (outer_band[idx] > 0.02).astype(np.float32)), 0.0, 1.0)
        # Keep ROI local to the person neighborhood to avoid background-only spill.
        person_neighborhood = _dilate_mask(person_mask[idx], kernel_size=max(9, dilate_kernel + 8), iterations=1)
        roi = np.clip(roi * person_neighborhood, 0.0, 1.0)
        roi_masks.append(roi.astype(np.float32))
    return np.stack(roi_masks).astype(np.float32)


def _compute_roi_boxes(roi_mask: np.ndarray, *, min_size: int = 96, pad: int = 20) -> list[tuple[int, int, int, int]]:
    boxes = []
    height, width = roi_mask.shape[1:]
    for mask in roi_mask:
        ys, xs = np.nonzero(mask > 0.01)
        if len(xs) == 0 or len(ys) == 0:
            boxes.append((0, 0, width, height))
            continue
        x0 = max(0, int(xs.min()) - pad)
        y0 = max(0, int(ys.min()) - pad)
        x1 = min(width, int(xs.max()) + pad + 1)
        y1 = min(height, int(ys.max()) + pad + 1)

        bw = x1 - x0
        bh = y1 - y0
        if bw < min_size:
            extra = min_size - bw
            x0 = max(0, x0 - extra // 2)
            x1 = min(width, x1 + extra - extra // 2)
        if bh < min_size:
            extra = min_size - bh
            y0 = max(0, y0 - extra // 2)
            y1 = min(height, y1 + extra - extra // 2)
        boxes.append((int(x0), int(y0), int(x1), int(y1)))
    return boxes


def _resize_rgb_frame(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)


def _resize_mask_frame(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    return cv2.resize(frame.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def compute_boundary_roi_metrics(
    *,
    before_frames: np.ndarray,
    after_frames: np.ndarray,
    background_frames: np.ndarray,
    outer_band: np.ndarray,
    inner_band: np.ndarray,
    roi_mask: np.ndarray,
) -> dict:
    roi_mask = np.asarray(roi_mask, dtype=np.float32)
    roi_outer_band = np.clip(np.asarray(outer_band, dtype=np.float32) * roi_mask, 0.0, 1.0)
    roi_inner_band = np.clip(np.asarray(inner_band, dtype=np.float32) * roi_mask, 0.0, 1.0)
    metrics = compute_boundary_refinement_metrics(
        before_frames=before_frames,
        after_frames=after_frames,
        background_frames=background_frames,
        outer_band=roi_outer_band,
        inner_band=roi_inner_band,
    )
    return {
        "roi_band_gradient_before_mean": metrics["band_gradient_before_mean"],
        "roi_band_gradient_after_mean": metrics["band_gradient_after_mean"],
        "roi_band_edge_contrast_before_mean": metrics["band_edge_contrast_before_mean"],
        "roi_band_edge_contrast_after_mean": metrics["band_edge_contrast_after_mean"],
        "roi_band_mad_mean": metrics["band_mad_mean"],
        "roi_halo_ratio_before": metrics["halo_ratio_before"],
        "roi_halo_ratio_after": metrics["halo_ratio_after"],
        "roi_outer_band_mean": metrics["outer_band_mean"],
        "roi_inner_band_mean": metrics["inner_band_mean"],
        "roi_area_ratio": float(roi_mask.mean()),
    }


def _compute_crop_boxes(person_mask: np.ndarray, outer_band: np.ndarray, frame_indices: list[int]) -> dict[int, tuple[int, int, int, int]]:
    boxes = {}
    height, width = person_mask.shape[1:]
    for index in frame_indices:
        union = np.clip(person_mask[index] + outer_band[index], 0.0, 1.0) > 0.0
        ys, xs = np.nonzero(union)
        if len(xs) == 0 or len(ys) == 0:
            boxes[index] = (0, 0, width, height)
            continue
        x0 = max(0, int(xs.min()) - 24)
        y0 = max(0, int(ys.min()) - 24)
        x1 = min(width, int(xs.max()) + 25)
        y1 = min(height, int(ys.max()) + 25)
        boxes[index] = (x0, y0, x1, y1)
    return boxes


def write_boundary_refinement_debug_artifacts(
    *,
    save_debug_dir: str | Path,
    fps: float,
    generated_frames: np.ndarray,
    refined_frames: np.ndarray,
    background_frames: np.ndarray,
    person_mask: np.ndarray,
    debug_data: dict,
) -> dict[str, str]:
    debug_dir = Path(save_debug_dir) / "boundary_refinement"
    debug_dir.mkdir(parents=True, exist_ok=True)

    outer_band = np.asarray(debug_data["outer_band"], dtype=np.float32)
    inner_band = np.asarray(debug_data["inner_band"], dtype=np.float32)
    soft_alpha = np.asarray(debug_data.get("soft_alpha", debug_data["target_foreground_alpha"]), dtype=np.float32)
    sharpen_alpha = np.asarray(debug_data["sharpen_alpha"], dtype=np.float32)
    band_blend = np.asarray(debug_data["band_blend"], dtype=np.float32)
    target_alpha = np.asarray(debug_data["target_foreground_alpha"], dtype=np.float32)

    comparison = np.concatenate(
        [
            generated_frames,
            refined_frames,
            background_frames,
            np.repeat(np.clip(np.rint(target_alpha[..., None] * 255.0), 0, 255).astype(np.uint8), 3, axis=3),
        ],
        axis=2,
    )
    comparison_path = write_output_frames(
        comparison,
        debug_dir / "boundary_refinement_comparison.mp4",
        fps=fps,
        output_format="mp4",
    )

    masks = {
        "outer_band": outer_band,
        "inner_band": inner_band,
        "soft_alpha": soft_alpha,
        "sharpen_alpha": sharpen_alpha,
        "blend_alpha": band_blend,
        "target_foreground_alpha": target_alpha,
    }
    for key in (
        "background_confidence",
        "uncertainty_map",
        "occlusion_band",
        "face_preserve_map",
        "face_confidence_map",
        "structure_guard_map",
        "adaptive_composite_strength",
        "adaptive_sharpen_alpha",
        "outer_focus",
        "edge_focus",
        "edge_boost_alpha",
        "roi_mask",
        "roi_blend_alpha",
        "local_edge_focus",
        "local_edge_gain",
        "local_edge_feather",
        "face_boundary_map",
        "hair_boundary_map",
        "hand_boundary_map",
        "cloth_boundary_map",
        "occluded_boundary_map",
    ):
        if key in debug_data:
            masks[key] = np.asarray(debug_data[key], dtype=np.float32)
    artifacts = {
        "comparison": str(Path(comparison_path).resolve()),
    }
    for stem, mask_frames in masks.items():
        artifact = write_person_mask_artifact(
            mask_frames=np.clip(mask_frames.astype(np.float32), 0.0, 1.0),
            output_root=debug_dir,
            stem=stem,
            artifact_format="mp4",
            fps=fps,
            mask_semantics=stem,
        )
        artifacts[stem] = str((debug_dir / artifact["path"]).resolve())

    frame_count = generated_frames.shape[0]
    keyframes = sorted({0, max(0, frame_count // 2), max(0, frame_count - 1)})
    crop_boxes = _compute_crop_boxes(person_mask, outer_band, keyframes)
    crops_dir = debug_dir / "crops"
    for index in keyframes:
        x0, y0, x1, y1 = crop_boxes[index]
        crop_strip = np.concatenate(
            [
                generated_frames[index, y0:y1, x0:x1],
                refined_frames[index, y0:y1, x0:x1],
                background_frames[index, y0:y1, x0:x1],
            ],
            axis=1,
        )
        _write_rgb_png(crops_dir / f"frame_{index:06d}_crop.png", crop_strip)

    if "roi_boxes" in debug_data:
        roi_boxes_path = debug_dir / "roi_boxes.json"
        with roi_boxes_path.open("w", encoding="utf-8") as handle:
            json.dump(debug_data["roi_boxes"], handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        artifacts["roi_boxes"] = str(roi_boxes_path.resolve())

    metrics_path = debug_dir / "boundary_refinement_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(debug_data["metrics"], handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    artifacts["metrics"] = str(metrics_path.resolve())
    artifacts["crops_dir"] = str(crops_dir.resolve())
    return artifacts
