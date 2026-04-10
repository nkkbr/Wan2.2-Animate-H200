from __future__ import annotations

import cv2
import numpy as np


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)


ROUND_CONFIGS = {
    "round1": {
        "temporal_alpha_blend": 0.32,
        "temporal_rgb_blend": 0.24,
        "silhouette_gain": 0.42,
        "depth_gain": 0.35,
        "conf_guard": 0.30,
    },
    "round2": {
        "temporal_alpha_blend": 0.58,
        "temporal_rgb_blend": 0.44,
        "silhouette_gain": 0.75,
        "depth_gain": 0.65,
        "conf_guard": 0.12,
    },
    "round3": {
        "temporal_alpha_blend": 0.42,
        "temporal_rgb_blend": 0.30,
        "silhouette_gain": 0.55,
        "depth_gain": 0.48,
        "conf_guard": 0.24,
    },
}


def _mask_centroid(mask: np.ndarray) -> tuple[float, float]:
    m = _clip01(mask)
    denom = float(m.sum())
    if denom <= 1e-6:
        h, w = m.shape
        return float(w - 1) * 0.5, float(h - 1) * 0.5
    ys, xs = np.indices(m.shape, dtype=np.float32)
    cx = float((xs * m).sum() / denom)
    cy = float((ys * m).sum() / denom)
    return cx, cy


def _warp_frame(frame: np.ndarray, dx: float, dy: float, *, interpolation: int, border_value: float | tuple[float, float, float]) -> np.ndarray:
    h, w = frame.shape[:2]
    matrix = np.float32([[1.0, 0.0, dx], [0.0, 1.0, dy]])
    return cv2.warpAffine(
        frame,
        matrix,
        (w, h),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def _distance_depth(mask: np.ndarray) -> np.ndarray:
    mask_u8 = (_clip01(mask) > 0.05).astype(np.uint8)
    if int(mask_u8.sum()) == 0:
        return np.zeros(mask.shape, dtype=np.float32)
    dt = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 3).astype(np.float32)
    scale = float(np.percentile(dt[mask_u8 > 0], 95.0)) if int(mask_u8.sum()) > 0 else 0.0
    if scale > 1e-6:
        dt = dt / scale
    return _clip01(dt)


def build_renderable_foreground_frame(
    *,
    source_rgb: np.ndarray,
    background_rgb: np.ndarray,
    foreground_rgb: np.ndarray,
    foreground_alpha: np.ndarray,
    soft_alpha: np.ndarray,
    hard_foreground: np.ndarray,
    boundary_band: np.ndarray,
    composite_roi_mask: np.ndarray,
    foreground_confidence: np.ndarray,
    occlusion_band: np.ndarray,
    background_unresolved: np.ndarray,
    uncertainty_map: np.ndarray | None,
    prev_state: dict[str, np.ndarray] | None,
    mode: str,
) -> dict[str, np.ndarray]:
    if mode not in ROUND_CONFIGS:
        raise ValueError(f"Unsupported renderable foreground mode: {mode}")
    cfg = ROUND_CONFIGS[mode]

    src = np.asarray(source_rgb, dtype=np.float32)
    bg = np.asarray(background_rgb, dtype=np.float32)
    fg = np.asarray(foreground_rgb, dtype=np.float32)
    fg_alpha = _clip01(foreground_alpha)
    soft_alpha = _clip01(soft_alpha)
    hard = _clip01(hard_foreground)
    boundary = _clip01(boundary_band)
    roi = _clip01(composite_roi_mask)
    fg_conf = _clip01(foreground_confidence)
    occ = _clip01(occlusion_band)
    unresolved = _clip01(background_unresolved)
    uncertainty = _clip01(uncertainty_map) if uncertainty_map is not None else np.zeros_like(fg_alpha)

    semantic_support = np.maximum.reduce([
        boundary,
        roi * 0.65,
        occ * 0.85,
        unresolved * 0.35,
    ])
    depth_like = _distance_depth(np.maximum(hard, soft_alpha))

    render_alpha = np.maximum(fg_alpha, hard * 0.95)
    render_alpha = np.maximum(render_alpha, soft_alpha * (0.72 + 0.18 * cfg["silhouette_gain"]))
    render_alpha = np.clip(
        render_alpha + boundary * cfg["silhouette_gain"] * (1.0 - fg_conf) * 0.18,
        0.0,
        1.0,
    )

    premul_current = np.clip(src - bg * (1.0 - render_alpha[..., None]), 0.0, 255.0)
    premul_current = np.minimum(premul_current, render_alpha[..., None] * 255.0)
    render_rgb = np.clip(
        fg * 0.55 + premul_current * 0.45 + depth_like[..., None] * cfg["depth_gain"] * 8.0,
        0.0,
        255.0,
    )

    if prev_state is not None:
        prev_mask = prev_state["render_person_mask"]
        prev_alpha = prev_state["render_alpha"]
        prev_rgb = prev_state["render_rgb"]
        prev_depth = prev_state["render_depth"]
        curr_mask = np.maximum(hard, render_alpha)
        prev_cx, prev_cy = _mask_centroid(prev_mask)
        curr_cx, curr_cy = _mask_centroid(curr_mask)
        dx = curr_cx - prev_cx
        dy = curr_cy - prev_cy
        prev_alpha_warp = _warp_frame(prev_alpha, dx, dy, interpolation=cv2.INTER_LINEAR, border_value=0.0).astype(np.float32)
        prev_mask_warp = _warp_frame(prev_mask, dx, dy, interpolation=cv2.INTER_LINEAR, border_value=0.0).astype(np.float32)
        prev_depth_warp = _warp_frame(prev_depth, dx, dy, interpolation=cv2.INTER_LINEAR, border_value=0.0).astype(np.float32)
        prev_rgb_warp = _warp_frame(prev_rgb, dx, dy, interpolation=cv2.INTER_LINEAR, border_value=(0.0, 0.0, 0.0)).astype(np.float32)

        temporal_gate = _clip01(
            semantic_support * (1.0 - cfg["conf_guard"] * fg_conf)
            + occ * 0.30
            + uncertainty * 0.20
        )
        depth_sync = np.maximum(depth_like, prev_depth_warp * temporal_gate * 0.25)
        render_alpha = _clip01(
            render_alpha * (1.0 - cfg["temporal_alpha_blend"] * temporal_gate)
            + prev_alpha_warp * (cfg["temporal_alpha_blend"] * temporal_gate)
        )
        render_alpha = np.maximum(render_alpha, hard * 0.95)
        render_rgb = np.clip(
            render_rgb * (1.0 - cfg["temporal_rgb_blend"] * temporal_gate[..., None])
            + prev_rgb_warp * (cfg["temporal_rgb_blend"] * temporal_gate[..., None])
            + depth_sync[..., None] * 6.0,
            0.0,
            255.0,
        )
        render_mask = _clip01(np.maximum(prev_mask_warp * temporal_gate, (render_alpha > 0.5).astype(np.float32)))
    else:
        depth_sync = depth_like
        render_mask = (render_alpha > 0.5).astype(np.float32)

    composite_rgb = np.clip(render_rgb + bg * (1.0 - render_alpha[..., None]), 0.0, 255.0).astype(np.uint8)
    silhouette_band = _clip01(np.maximum(boundary * 0.75, _distance_depth(render_mask) * 0.20))

    return {
        "render_rgb": render_rgb.astype(np.uint8),
        "render_alpha": render_alpha.astype(np.float32),
        "render_depth": depth_sync.astype(np.float32),
        "render_person_mask": _clip01(np.maximum(render_mask, hard)).astype(np.float32),
        "render_silhouette_band": silhouette_band.astype(np.float32),
        "render_composite_rgb": composite_rgb,
        "render_roi_mask": semantic_support.astype(np.float32),
    }
