import json
from pathlib import Path

import numpy as np
import torch

from .media_io import write_output_frames, write_person_mask_artifact


def _ensure_5d(frames: torch.Tensor) -> torch.Tensor:
    if frames.ndim == 4:
        return frames.unsqueeze(0)
    if frames.ndim == 5:
        return frames
    raise ValueError(f"Expected frames with shape [C, T, H, W] or [B, C, T, H, W]. Got {frames.shape}.")


def _smoothstep(alpha: torch.Tensor) -> torch.Tensor:
    return alpha * alpha * (3.0 - 2.0 * alpha)


def _build_time_alpha(overlap_len: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if overlap_len <= 0:
        raise ValueError(f"overlap_len must be > 0. Got {overlap_len}.")
    if overlap_len == 1:
        alpha = torch.tensor([0.5], device=device, dtype=dtype)
    else:
        alpha = torch.linspace(0.0, 1.0, steps=overlap_len, device=device, dtype=dtype)
    return alpha.view(1, 1, overlap_len, 1, 1)


def mean_abs_difference(
    left: torch.Tensor,
    right: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> float:
    left = _ensure_5d(left).to(dtype=torch.float32)
    right = _ensure_5d(right).to(dtype=torch.float32, device=left.device)
    if left.shape != right.shape:
        raise ValueError(f"Expected matching tensor shapes. Got {left.shape} and {right.shape}.")
    diff = (left - right).abs()
    if mask is None:
        return float(diff.mean().item())
    mask = torch.as_tensor(mask, dtype=torch.float32, device=left.device)
    if mask.ndim == 3:
        mask = mask.unsqueeze(0).unsqueeze(0)
    if mask.ndim == 4:
        if mask.shape[0] == left.shape[2]:
            mask = mask.permute(1, 0, 2, 3).unsqueeze(0)
        elif mask.shape[1] == left.shape[2]:
            mask = mask.unsqueeze(0)
        else:
            raise ValueError(f"Could not align 4D mask with frames. Got mask={mask.shape}, frames={left.shape}.")
    if mask.ndim != 5:
        raise ValueError(f"Mask must have shape [T, H, W], [T, 1, H, W] or [B, 1, T, H, W]. Got {mask.shape}.")
    weighted = diff * mask
    denom = float(mask.sum().item() * diff.shape[1])
    if denom <= 1e-6:
        return 0.0
    return float(weighted.sum().item() / denom)


def blend_clip_overlap(
    prev_overlap: torch.Tensor,
    curr_overlap: torch.Tensor,
    *,
    mode: str = "mask_aware",
    pixel_regions: dict[str, torch.Tensor] | None = None,
    background_current_strength: float = 0.35,
) -> dict[str, torch.Tensor | dict]:
    prev_overlap = _ensure_5d(prev_overlap).to(dtype=torch.float32)
    curr_overlap = _ensure_5d(curr_overlap).to(dtype=torch.float32, device=prev_overlap.device)
    if prev_overlap.shape != curr_overlap.shape:
        raise ValueError(f"Overlap tensors must match. Got {prev_overlap.shape} and {curr_overlap.shape}.")

    _, _, overlap_len, height, width = prev_overlap.shape
    time_alpha = _build_time_alpha(overlap_len, device=prev_overlap.device, dtype=prev_overlap.dtype)
    alpha_map = torch.zeros((1, 1, overlap_len, height, width), device=prev_overlap.device, dtype=prev_overlap.dtype)
    mode_used = mode

    if mode == "none":
        alpha_map.zero_()
    elif mode == "linear":
        alpha_map = time_alpha.expand_as(alpha_map)
    elif mode == "mask_aware":
        if pixel_regions is None:
            alpha_map = time_alpha.expand_as(alpha_map)
            mode_used = "linear_fallback"
        else:
            hard_background_keep = torch.as_tensor(
                pixel_regions["hard_background_keep"],
                dtype=prev_overlap.dtype,
                device=prev_overlap.device,
            )
            transition_band = torch.as_tensor(
                pixel_regions["transition_band"],
                dtype=prev_overlap.dtype,
                device=prev_overlap.device,
            )
            free_replacement = torch.as_tensor(
                pixel_regions["free_replacement"],
                dtype=prev_overlap.dtype,
                device=prev_overlap.device,
            )
            if hard_background_keep.ndim == 3:
                hard_background_keep = hard_background_keep.unsqueeze(0).unsqueeze(0)
                transition_band = transition_band.unsqueeze(0).unsqueeze(0)
                free_replacement = free_replacement.unsqueeze(0).unsqueeze(0)
            background_current_strength = max(0.0, min(float(background_current_strength), 1.0))
            background_alpha = time_alpha * background_current_strength
            boundary_alpha = _smoothstep(time_alpha)
            person_alpha = time_alpha
            alpha_map = (
                hard_background_keep * background_alpha
                + transition_band * boundary_alpha
                + free_replacement * person_alpha
            )
    else:
        raise ValueError(f"Unsupported overlap blend mode: {mode}")

    alpha_map = torch.clamp(alpha_map, 0.0, 1.0)
    blended = prev_overlap * (1.0 - alpha_map) + curr_overlap * alpha_map

    stats = {
        "mode_used": mode_used,
        "overlap_mad_before": mean_abs_difference(prev_overlap, curr_overlap),
        "overlap_mad_after_prev": mean_abs_difference(prev_overlap, blended),
        "overlap_mad_after_curr": mean_abs_difference(curr_overlap, blended),
        "alpha_mean": float(alpha_map.mean().item()),
        "alpha_min": float(alpha_map.min().item()),
        "alpha_max": float(alpha_map.max().item()),
    }
    if pixel_regions is not None:
        hard_background_keep = torch.as_tensor(pixel_regions["hard_background_keep"], dtype=torch.float32, device=alpha_map.device)
        transition_band = torch.as_tensor(pixel_regions["transition_band"], dtype=torch.float32, device=alpha_map.device)
        free_replacement = torch.as_tensor(pixel_regions["free_replacement"], dtype=torch.float32, device=alpha_map.device)
        stats.update({
            "alpha_background_mean": mean_abs_difference(alpha_map, torch.zeros_like(alpha_map), mask=hard_background_keep),
            "alpha_transition_mean": mean_abs_difference(alpha_map, torch.zeros_like(alpha_map), mask=transition_band),
            "alpha_person_mean": mean_abs_difference(alpha_map, torch.zeros_like(alpha_map), mask=free_replacement),
        })
    return {
        "blended": blended,
        "alpha_map": alpha_map,
        "time_alpha": time_alpha,
        "stats": stats,
    }


def summarize_scalar_series(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None}
    array = np.asarray(values, dtype=np.float32)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _tensor_video_to_uint8(frames: torch.Tensor) -> np.ndarray:
    frames = _ensure_5d(frames).detach().cpu().to(dtype=torch.float32)
    if frames.shape[0] != 1:
        raise ValueError(f"Debug export expects a batch size of 1. Got {frames.shape}.")
    frames = frames[0].permute(1, 2, 3, 0).numpy()
    frames = np.clip((frames + 1.0) * 127.5, 0, 255).astype(np.uint8)
    return frames


def write_seam_debug_artifacts(
    *,
    save_debug_dir: str | Path,
    seam_index: int,
    fps: float,
    prev_overlap: torch.Tensor,
    curr_overlap: torch.Tensor,
    blended_overlap: torch.Tensor,
    alpha_map: torch.Tensor,
) -> dict[str, str]:
    seams_dir = Path(save_debug_dir) / "seams"
    seams_dir.mkdir(parents=True, exist_ok=True)
    stem = f"seam_{seam_index:03d}"

    prev_rgb = _tensor_video_to_uint8(prev_overlap)
    curr_rgb = _tensor_video_to_uint8(curr_overlap)
    blended_rgb = _tensor_video_to_uint8(blended_overlap)
    alpha_frames = _ensure_5d(alpha_map).detach().cpu().numpy()[0, 0]
    alpha_rgb = np.repeat(np.clip(np.rint(alpha_frames[:, :, :, None] * 255.0), 0, 255).astype(np.uint8), repeats=3, axis=3)
    comparison = np.concatenate([prev_rgb, curr_rgb, blended_rgb, alpha_rgb], axis=2)

    comparison_path = write_output_frames(
        comparison,
        seams_dir / f"{stem}_comparison.mp4",
        fps=fps,
        output_format="mp4",
    )
    alpha_artifact = write_person_mask_artifact(
        mask_frames=np.clip(alpha_frames.astype(np.float32), 0.0, 1.0),
        output_root=seams_dir,
        stem=f"{stem}_alpha",
        artifact_format="mp4",
        fps=fps,
        mask_semantics="seam_blend_alpha",
    )

    return {
        "comparison": str(Path(comparison_path).resolve()),
        "alpha": str((seams_dir / alpha_artifact["path"]).resolve()),
    }


def write_seam_summary(save_debug_dir: str | Path, seam_stats: list[dict]) -> str:
    seams_dir = Path(save_debug_dir) / "seams"
    seams_dir.mkdir(parents=True, exist_ok=True)
    summary_path = seams_dir / "seam_debug.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump({"seams": seam_stats}, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    return str(summary_path.resolve())
