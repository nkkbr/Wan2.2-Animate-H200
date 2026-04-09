from __future__ import annotations

import torch
import torch.nn.functional as F


LOSS_STACKS = {
    "pixel_v1": {
        "composite_weight": 0.0,
        "gradient_weight": 0.0,
        "contrast_weight": 0.0,
    },
    "composite_v1": {
        "composite_weight": 0.35,
        "gradient_weight": 0.0,
        "contrast_weight": 0.0,
    },
    "composite_grad_v1": {
        "composite_weight": 0.35,
        "gradient_weight": 0.20,
        "contrast_weight": 0.0,
    },
    "composite_grad_contrast_v1": {
        "composite_weight": 0.35,
        "gradient_weight": 0.20,
        "contrast_weight": 0.12,
    },
}


def resolve_loss_stack(
    name: str,
    *,
    composite_weight: float | None = None,
    gradient_weight: float | None = None,
    contrast_weight: float | None = None,
) -> dict[str, float]:
    if name not in LOSS_STACKS:
        raise ValueError(f"Unknown loss stack: {name}")
    weights = dict(LOSS_STACKS[name])
    if composite_weight is not None and composite_weight >= 0.0:
        weights["composite_weight"] = float(composite_weight)
    if gradient_weight is not None and gradient_weight >= 0.0:
        weights["gradient_weight"] = float(gradient_weight)
    if contrast_weight is not None and contrast_weight >= 0.0:
        weights["contrast_weight"] = float(contrast_weight)
    weights["loss_stack"] = name
    return weights


def _ensure_nchw(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:
        return x[:, None]
    return x


def rgb_to_luma(rgb: torch.Tensor) -> torch.Tensor:
    rgb = _ensure_nchw(rgb)
    if rgb.shape[1] == 1:
        return rgb[:, 0]
    return 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]


def sobel_magnitude(image: torch.Tensor) -> torch.Tensor:
    image = _ensure_nchw(image)
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=image.dtype,
        device=image.device,
    ).view(1, 1, 3, 3)
    kernel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        dtype=image.dtype,
        device=image.device,
    ).view(1, 1, 3, 3)
    gx = F.conv2d(image, kernel_x, padding=1)
    gy = F.conv2d(image, kernel_y, padding=1)
    return torch.sqrt(torch.clamp(gx * gx + gy * gy, min=1e-12))[:, 0]


def composite_from_premultiplied(
    foreground_rgb: torch.Tensor,
    background_rgb: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    foreground_rgb = _ensure_nchw(foreground_rgb)
    background_rgb = _ensure_nchw(background_rgb)
    if foreground_rgb.shape[1] == 1:
        foreground_rgb = foreground_rgb.repeat(1, 3, 1, 1)
    if background_rgb.shape[1] == 1:
        background_rgb = background_rgb.repeat(1, 3, 1, 1)
    alpha = _ensure_nchw(alpha)
    return foreground_rgb + (1.0 - alpha) * background_rgb


def weighted_l1(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    diff = (pred - target).abs()
    if weight is None:
        return diff.mean()
    return (diff * weight).sum() / torch.clamp(weight.sum(), min=1.0)


def compositing_reconstruction_loss(
    pred_alpha: torch.Tensor,
    target_alpha: torch.Tensor,
    foreground_rgb: torch.Tensor,
    background_rgb: torch.Tensor,
    focus_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    pred_comp = composite_from_premultiplied(foreground_rgb, background_rgb, pred_alpha)
    target_comp = composite_from_premultiplied(foreground_rgb, background_rgb, target_alpha)
    diff = (pred_comp - target_comp).abs().mean(dim=1)
    if focus_mask is None:
        return diff.mean()
    focus_mask = focus_mask.to(diff.dtype)
    weight = 1.0 + 3.0 * focus_mask
    return (diff * weight).sum() / torch.clamp(weight.sum(), min=1.0)


def gradient_preservation_loss(
    pred_alpha: torch.Tensor,
    target_alpha: torch.Tensor,
    focus_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    pred_grad = sobel_magnitude(pred_alpha)
    target_grad = sobel_magnitude(target_alpha)
    diff = (pred_grad - target_grad).abs()
    if focus_mask is None:
        return diff.mean()
    focus_mask = focus_mask.to(diff.dtype)
    weight = 1.0 + 2.0 * focus_mask
    return (diff * weight).sum() / torch.clamp(weight.sum(), min=1.0)


def contrast_preservation_loss(
    pred_alpha: torch.Tensor,
    target_alpha: torch.Tensor,
    foreground_rgb: torch.Tensor,
    background_rgb: torch.Tensor,
    focus_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    pred_comp = composite_from_premultiplied(foreground_rgb, background_rgb, pred_alpha)
    target_comp = composite_from_premultiplied(foreground_rgb, background_rgb, target_alpha)
    pred_luma = rgb_to_luma(pred_comp)[:, None]
    target_luma = rgb_to_luma(target_comp)[:, None]
    pred_local_mean = F.avg_pool2d(pred_luma, kernel_size=5, stride=1, padding=2)[:, 0]
    target_local_mean = F.avg_pool2d(target_luma, kernel_size=5, stride=1, padding=2)[:, 0]
    pred_contrast = (pred_luma[:, 0] - pred_local_mean).abs()
    target_contrast = (target_luma[:, 0] - target_local_mean).abs()
    diff = (pred_contrast - target_contrast).abs()
    if focus_mask is None:
        return diff.mean()
    focus_mask = focus_mask.to(diff.dtype)
    weight = 1.0 + 2.0 * focus_mask
    return (diff * weight).sum() / torch.clamp(weight.sum(), min=1.0)
