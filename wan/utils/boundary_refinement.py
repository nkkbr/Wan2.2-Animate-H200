import json
from pathlib import Path

import cv2
import numpy as np
import torch

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

    if mode not in {"deterministic", "v2"}:
        raise ValueError(f"Unsupported boundary refinement mode: {mode}")

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
        sharpen_alpha = np.clip(adaptive_sharpen * sharpen, 0.0, 1.0)[..., None]
        refined = generated * (1.0 - sharpen_alpha) + sharpened * sharpen_alpha

        adaptive_blend = np.clip(outer_focus * strength, 0.0, 1.0)
        adaptive_blend = adaptive_blend * (0.25 + 0.75 * background_confidence)
        adaptive_blend = adaptive_blend * (1.0 - 0.80 * uncertainty_map)
        adaptive_blend = adaptive_blend * (1.0 - 0.55 * occlusion_band)
        adaptive_blend = adaptive_blend * (1.0 - 0.50 * face_preserve_map)
        adaptive_blend = np.clip(adaptive_blend * structure_guard, 0.0, 1.0)
        band_blend = adaptive_blend[..., None]

        composite = refined * target_foreground_alpha + background * (1.0 - target_foreground_alpha)
        refined = refined * (1.0 - band_blend) + composite * band_blend
        edge_boost = np.clip(edge_focus * (0.55 + 0.45 * target_alpha_2d), 0.0, 1.0)
        edge_boost = edge_boost * (1.0 - 0.45 * uncertainty_map)
        edge_boost = edge_boost * (0.35 + 0.65 * background_confidence)
        edge_boost = edge_boost * (1.0 - 0.40 * occlusion_band)
        edge_boost = edge_boost * (1.0 - 0.35 * face_preserve_map)
        edge_boost = np.clip(edge_boost * structure_guard, 0.0, 1.0)
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

    metrics_path = debug_dir / "boundary_refinement_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(debug_data["metrics"], handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    artifacts["metrics"] = str(metrics_path.resolve())
    artifacts["crops_dir"] = str(crops_dir.resolve())
    return artifacts
