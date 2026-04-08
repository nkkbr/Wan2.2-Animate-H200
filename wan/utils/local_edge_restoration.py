import cv2
import numpy as np


def _clip01(array: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(array, dtype=np.float32), 0.0, 1.0).astype(np.float32)


def _resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(_clip01(mask), (int(width), int(height)), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def _resize_rgb(image: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(np.asarray(image, dtype=np.float32), (int(width), int(height)), interpolation=cv2.INTER_CUBIC).astype(np.float32)


def _gaussian_ksize(sigma: float) -> tuple[int, int]:
    radius = max(1, int(round(float(sigma) * 3.0)))
    kernel = radius * 2 + 1
    return kernel, kernel


def _apply_unsharp(image: np.ndarray, sigma: float, amount: float) -> np.ndarray:
    if amount <= 0.0:
        return np.asarray(image, dtype=np.float32).copy()
    blurred = cv2.GaussianBlur(np.asarray(image, dtype=np.float32), _gaussian_ksize(max(0.1, sigma)), sigmaX=sigma, sigmaY=sigma)
    sharpened = image + float(amount) * (image - blurred)
    return np.clip(sharpened, 0.0, 1.0).astype(np.float32)


def _apply_clahe_rgb(image: np.ndarray, clip_limit: float = 1.8, tile_grid_size: int = 8) -> np.ndarray:
    image_u8 = np.clip(np.rint(np.asarray(image, dtype=np.float32) * 255.0), 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(image_u8, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile_grid_size), int(tile_grid_size)))
    l_enhanced = clahe.apply(l_channel)
    merged = cv2.merge((l_enhanced, a_channel, b_channel))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
    return np.clip(enhanced, 0.0, 1.0).astype(np.float32)


def build_local_edge_focus_map(
    *,
    outer_band: np.ndarray,
    soft_alpha: np.ndarray,
    uncertainty_map: np.ndarray,
    detail_release_map: np.ndarray | None = None,
    trimap_unknown_map: np.ndarray | None = None,
    edge_detail_map: np.ndarray | None = None,
    face_boundary_map: np.ndarray | None = None,
    hair_boundary_map: np.ndarray | None = None,
    hand_boundary_map: np.ndarray | None = None,
    cloth_boundary_map: np.ndarray | None = None,
    occluded_boundary_map: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    outer_band = _clip01(outer_band)
    soft_alpha = _clip01(soft_alpha)
    uncertainty_map = _clip01(uncertainty_map)
    detail_release_map = np.zeros_like(outer_band, dtype=np.float32) if detail_release_map is None else _clip01(detail_release_map)
    trimap_unknown_map = np.zeros_like(outer_band, dtype=np.float32) if trimap_unknown_map is None else _clip01(trimap_unknown_map)
    edge_detail_map = np.zeros_like(outer_band, dtype=np.float32) if edge_detail_map is None else _clip01(edge_detail_map)
    face_boundary_map = np.zeros_like(outer_band, dtype=np.float32) if face_boundary_map is None else _clip01(face_boundary_map)
    hair_boundary_map = np.zeros_like(outer_band, dtype=np.float32) if hair_boundary_map is None else _clip01(hair_boundary_map)
    hand_boundary_map = np.zeros_like(outer_band, dtype=np.float32) if hand_boundary_map is None else _clip01(hand_boundary_map)
    cloth_boundary_map = np.zeros_like(outer_band, dtype=np.float32) if cloth_boundary_map is None else _clip01(cloth_boundary_map)
    occluded_boundary_map = np.zeros_like(outer_band, dtype=np.float32) if occluded_boundary_map is None else _clip01(occluded_boundary_map)

    alpha_transition = np.clip(4.0 * soft_alpha * (1.0 - soft_alpha), 0.0, 1.0).astype(np.float32)
    base_focus = np.clip(
        np.maximum.reduce([
            1.15 * outer_band,
            0.95 * alpha_transition,
            0.82 * detail_release_map,
            0.80 * edge_detail_map,
            0.70 * trimap_unknown_map,
            0.90 * hair_boundary_map,
            0.76 * cloth_boundary_map,
            0.74 * hand_boundary_map,
            0.48 * face_boundary_map,
        ]),
        0.0,
        1.0,
    ).astype(np.float32)
    base_focus = np.clip(base_focus * (1.0 - 0.42 * uncertainty_map) * (1.0 - 0.55 * occluded_boundary_map), 0.0, 1.0)

    semantic_gain = np.clip(
        0.14 * face_boundary_map
        + 0.72 * hair_boundary_map
        + 0.42 * hand_boundary_map
        + 0.58 * cloth_boundary_map
        + 0.24 * detail_release_map
        + 0.18 * edge_detail_map,
        0.0,
        1.0,
    ).astype(np.float32)
    face_guard = np.clip(0.55 + 0.45 * (1.0 - face_boundary_map), 0.55, 1.0).astype(np.float32)
    local_gain = np.clip((0.20 + 0.80 * semantic_gain) * face_guard * (1.0 - 0.40 * uncertainty_map), 0.0, 1.0)

    return {
        "focus": base_focus.astype(np.float32),
        "semantic_gain": semantic_gain.astype(np.float32),
        "face_guard": face_guard.astype(np.float32),
        "local_gain": local_gain.astype(np.float32),
    }


def restore_local_edge_roi(
    *,
    original_rgb: np.ndarray,
    refined_rgb: np.ndarray,
    soft_alpha: np.ndarray,
    outer_band: np.ndarray,
    uncertainty_map: np.ndarray,
    detail_release_map: np.ndarray | None = None,
    trimap_unknown_map: np.ndarray | None = None,
    edge_detail_map: np.ndarray | None = None,
    face_boundary_map: np.ndarray | None = None,
    hair_boundary_map: np.ndarray | None = None,
    hand_boundary_map: np.ndarray | None = None,
    cloth_boundary_map: np.ndarray | None = None,
    occluded_boundary_map: np.ndarray | None = None,
    sharpen: float = 0.25,
    detail_strength: float = 0.35,
    scale_factor: float = 2.0,
) -> tuple[np.ndarray, dict[str, np.ndarray | float]]:
    original_rgb = _clip01(original_rgb)
    refined_rgb = _clip01(refined_rgb)
    soft_alpha = _clip01(soft_alpha)
    outer_band = _clip01(outer_band)
    uncertainty_map = _clip01(uncertainty_map)
    focus_pack = build_local_edge_focus_map(
        outer_band=outer_band,
        soft_alpha=soft_alpha,
        uncertainty_map=uncertainty_map,
        detail_release_map=detail_release_map,
        trimap_unknown_map=trimap_unknown_map,
        edge_detail_map=edge_detail_map,
        face_boundary_map=face_boundary_map,
        hair_boundary_map=hair_boundary_map,
        hand_boundary_map=hand_boundary_map,
        cloth_boundary_map=cloth_boundary_map,
        occluded_boundary_map=occluded_boundary_map,
    )

    height, width = original_rgb.shape[:2]
    target_w = max(32, int(round(width * float(scale_factor))))
    target_h = max(32, int(round(height * float(scale_factor))))

    refined_up = _resize_rgb(refined_rgb, target_w, target_h)
    original_up = _resize_rgb(original_rgb, target_w, target_h)

    focus_up = _resize_mask(focus_pack["focus"], target_w, target_h)
    local_gain_up = _resize_mask(focus_pack["local_gain"], target_w, target_h)
    face_guard_up = _resize_mask(focus_pack["face_guard"], target_w, target_h)
    semantic_gain_up = _resize_mask(focus_pack["semantic_gain"], target_w, target_h)

    base_up = cv2.bilateralFilter(refined_up, d=0, sigmaColor=18, sigmaSpace=7)
    original_sharp_up = _apply_unsharp(original_up, sigma=0.75, amount=min(1.0, float(sharpen) + 0.42))
    refined_sharp_up = _apply_unsharp(refined_up, sigma=0.85, amount=min(1.0, float(sharpen) + 0.30))
    detail_residual_up = np.clip(original_sharp_up - base_up, -0.35, 0.35).astype(np.float32)
    refined_residual_up = np.clip(refined_sharp_up - refined_up, -0.25, 0.25).astype(np.float32)

    clahe_up = _apply_clahe_rgb(refined_up, clip_limit=1.9, tile_grid_size=8)
    clahe_delta_up = np.clip(clahe_up - refined_up, -0.18, 0.18).astype(np.float32)

    laplacian = cv2.Laplacian(cv2.cvtColor(np.clip(np.rint(refined_up * 255.0), 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY), cv2.CV_32F, ksize=3)
    laplacian = np.clip(laplacian / 255.0, -0.18, 0.18).astype(np.float32)
    laplacian = np.repeat(laplacian[..., None], 3, axis=2)

    detail_gain_up = np.clip(
        float(detail_strength) * focus_up * (0.28 + 0.72 * local_gain_up) * (0.50 + 0.50 * face_guard_up),
        0.0,
        1.0,
    ).astype(np.float32)
    contrast_gain_up = np.clip(
        focus_up * np.clip(0.10 + 0.18 * semantic_gain_up, 0.0, 0.32),
        0.0,
        1.0,
    ).astype(np.float32)
    high_freq = np.clip(refined_sharp_up - base_up, -0.30, 0.30).astype(np.float32)
    semantic_high_freq = np.clip(
        high_freq * np.clip(0.15 + 0.85 * semantic_gain_up, 0.0, 1.0)[..., None],
        -0.30,
        0.30,
    ).astype(np.float32)

    restored_up = np.clip(
        refined_up
        + detail_gain_up[..., None] * (0.44 * detail_residual_up + 0.38 * refined_residual_up + 0.26 * high_freq + 0.22 * semantic_high_freq)
        + contrast_gain_up[..., None] * 0.55 * clahe_delta_up
        + 0.18 * detail_gain_up[..., None] * laplacian,
        0.0,
        1.0,
    ).astype(np.float32)

    restored_down = _resize_rgb(restored_up, width, height)
    focus_down = _resize_mask(focus_up, width, height)
    local_gain_down = _resize_mask(local_gain_up, width, height)

    feather = cv2.GaussianBlur(np.clip(focus_down * (0.80 + 0.20 * local_gain_down), 0.0, 1.0), (9, 9), sigmaX=0)
    feather = np.clip(feather * (0.40 + 0.60 * soft_alpha), 0.0, 1.0).astype(np.float32)

    merged = np.clip(refined_rgb * (1.0 - feather[..., None]) + restored_down * feather[..., None], 0.0, 1.0).astype(np.float32)
    debug = {
        "local_edge_focus": focus_pack["focus"].astype(np.float32),
        "local_edge_gain": local_gain_down.astype(np.float32),
        "local_edge_feather": feather.astype(np.float32),
        "local_edge_scale_factor": float(scale_factor),
    }
    return merged, debug
