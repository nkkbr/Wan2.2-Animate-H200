import cv2
import numpy as np


def _normalize_map(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    max_value = float(values.max()) if values.size > 0 else 0.0
    if max_value <= 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip(values / max_value, 0.0, 1.0).astype(np.float32)


def _draw_circle_mask(canvas: np.ndarray, center_xy: tuple[float, float], radius: float, value: float = 1.0) -> None:
    height, width = canvas.shape
    cx, cy = center_xy
    if radius <= 0:
        return
    cv2.circle(
        canvas,
        (int(round(np.clip(cx, 0, width - 1))), int(round(np.clip(cy, 0, height - 1)))),
        max(1, int(round(radius))),
        float(value),
        thickness=-1,
    )


def _extract_keypoint_array(meta: dict | None, key: str) -> np.ndarray:
    if not meta or key not in meta or meta[key] is None:
        return np.zeros((0, 3), dtype=np.float32)
    points = np.asarray(meta[key], dtype=np.float32)
    if points.ndim != 2 or points.shape[-1] < 2:
        return np.zeros((0, 3), dtype=np.float32)
    if points.shape[-1] == 2:
        confidence = np.ones((points.shape[0], 1), dtype=np.float32)
        points = np.concatenate([points, confidence], axis=1)
    return points[:, :3].astype(np.float32)


def _normalized_points_to_pixels(points: np.ndarray, image_shape: tuple[int, int], conf_thresh: float) -> np.ndarray:
    if points.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    height, width = image_shape
    valid = points[:, 2] >= float(conf_thresh)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32)
    coords = points[valid].copy()
    coords[:, 0] *= float(width)
    coords[:, 1] *= float(height)
    return coords.astype(np.float32)


def _make_head_prior(
    image_shape: tuple[int, int],
    face_bbox: tuple[int, int, int, int] | list[int] | np.ndarray | None,
    face_points: np.ndarray,
    *,
    head_expand: float,
) -> np.ndarray:
    height, width = image_shape
    prior = np.zeros((height, width), dtype=np.float32)
    if face_bbox is not None:
        x1, x2, y1, y2 = [float(v) for v in face_bbox]
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        rx = max(2.0, 0.5 * (x2 - x1) * float(head_expand))
        ry = max(2.0, 0.6 * (y2 - y1) * float(head_expand))
        cv2.ellipse(
            prior,
            (int(round(cx)), int(round(cy))),
            (max(1, int(round(rx))), max(1, int(round(ry)))),
            0.0,
            0.0,
            360.0,
            1.0,
            thickness=-1,
        )
    if face_points.size > 0:
        x0 = float(face_points[:, 0].min())
        x1 = float(face_points[:, 0].max())
        y0 = float(face_points[:, 1].min())
        y1 = float(face_points[:, 1].max())
        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)
        rx = max(2.0, 0.65 * (x1 - x0))
        ry = max(2.0, 0.8 * (y1 - y0))
        cv2.ellipse(
            prior,
            (int(round(cx)), int(round(cy))),
            (max(1, int(round(rx))), max(1, int(round(ry)))),
            0.0,
            0.0,
            360.0,
            1.0,
            thickness=-1,
        )
    return np.clip(prior, 0.0, 1.0).astype(np.float32)


def _make_hand_prior(image_shape: tuple[int, int], hand_points: np.ndarray, *, radius_ratio: float) -> np.ndarray:
    height, width = image_shape
    prior = np.zeros((height, width), dtype=np.float32)
    radius = max(2.0, float(max(height, width)) * float(radius_ratio))
    for point in hand_points:
        _draw_circle_mask(prior, (float(point[0]), float(point[1])), radius)
    if hand_points.shape[0] >= 3:
        hull_points = hand_points[:, :2].astype(np.int32)
        if hull_points.shape[0] >= 3:
            hull = cv2.convexHull(hull_points)
            cv2.fillConvexPoly(prior, hull, 1.0)
    return np.clip(prior, 0.0, 1.0).astype(np.float32)


def run_parsing_adapter(
    *,
    frames: np.ndarray,
    hard_mask: np.ndarray,
    pose_metas: list[dict] | None,
    face_bboxes: list[tuple[int, int, int, int]] | None,
    mode: str = "heuristic",
    face_conf_thresh: float = 0.45,
    hand_conf_thresh: float = 0.35,
    head_expand: float = 1.2,
    hand_radius_ratio: float = 0.025,
    boundary_kernel: int = 11,
    gradient_blur_kernel: int = 3,
) -> dict:
    frames = np.asarray(frames, dtype=np.uint8)
    hard_mask = np.asarray(hard_mask, dtype=np.float32)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"frames must have shape [T, H, W, 3]. Got {frames.shape}.")
    if hard_mask.shape != frames.shape[:3]:
        raise ValueError(f"hard_mask must match frames. Got {hard_mask.shape} vs {frames.shape[:3]}.")

    frame_count, height, width = hard_mask.shape
    zeros = np.zeros((frame_count, height, width), dtype=np.float32)
    if mode == "none":
        return {
            "mode": "none",
            "semantic_boundary_prior": zeros,
            "head_prior": zeros,
            "hand_prior": zeros,
            "part_foreground_prior": zeros,
            "stats": {
                "head_prior_mean": 0.0,
                "hand_prior_mean": 0.0,
                "semantic_boundary_mean": 0.0,
            },
        }
    if mode != "heuristic":
        raise ValueError(f"Unsupported parsing adapter mode: {mode}")

    kernel_size = max(3, int(boundary_kernel))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    blur_kernel = max(1, int(gradient_blur_kernel))
    if blur_kernel % 2 == 0:
        blur_kernel += 1

    semantic_boundary_prior = []
    head_priors = []
    hand_priors = []
    part_foreground_priors = []

    for index in range(frame_count):
        frame = frames[index]
        mask = np.clip(hard_mask[index], 0.0, 1.0)
        hard = (mask > 0.5).astype(np.uint8)
        dilated = cv2.dilate(hard, kernel, iterations=1)
        eroded = cv2.erode(hard, kernel, iterations=1)
        boundary_ring = np.clip(dilated - eroded, 0, 1).astype(np.float32)

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if blur_kernel > 1:
            gray = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), sigmaX=0)
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = _normalize_map(np.sqrt(grad_x * grad_x + grad_y * grad_y))
        canny = cv2.Canny(gray, 64, 128).astype(np.float32) / 255.0

        pose_meta = pose_metas[index] if pose_metas is not None and index < len(pose_metas) else None
        face_points = _normalized_points_to_pixels(
            _extract_keypoint_array(pose_meta, "keypoints_face"),
            (height, width),
            face_conf_thresh,
        )
        left_hand_points = _normalized_points_to_pixels(
            _extract_keypoint_array(pose_meta, "keypoints_left_hand"),
            (height, width),
            hand_conf_thresh,
        )
        right_hand_points = _normalized_points_to_pixels(
            _extract_keypoint_array(pose_meta, "keypoints_right_hand"),
            (height, width),
            hand_conf_thresh,
        )
        face_bbox = face_bboxes[index] if face_bboxes is not None and index < len(face_bboxes) else None
        head_prior = _make_head_prior((height, width), face_bbox, face_points, head_expand=head_expand)
        hand_prior = np.maximum(
            _make_hand_prior((height, width), left_hand_points, radius_ratio=hand_radius_ratio),
            _make_hand_prior((height, width), right_hand_points, radius_ratio=hand_radius_ratio),
        )
        part_foreground_prior = np.clip(np.maximum(head_prior, hand_prior) * dilated, 0.0, 1.0).astype(np.float32)
        semantic = np.clip(0.6 * grad_mag + 0.4 * canny, 0.0, 1.0).astype(np.float32)
        semantic = np.clip(semantic * dilated, 0.0, 1.0)
        semantic = np.maximum(semantic * boundary_ring, 0.6 * part_foreground_prior * (1.0 - mask))
        semantic_boundary = np.clip(semantic + 0.35 * part_foreground_prior * boundary_ring, 0.0, 1.0).astype(np.float32)

        semantic_boundary_prior.append(semantic_boundary)
        head_priors.append(head_prior)
        hand_priors.append(hand_prior)
        part_foreground_priors.append(part_foreground_prior)

    semantic_boundary_prior = np.stack(semantic_boundary_prior).astype(np.float32)
    head_priors = np.stack(head_priors).astype(np.float32)
    hand_priors = np.stack(hand_priors).astype(np.float32)
    part_foreground_priors = np.stack(part_foreground_priors).astype(np.float32)
    return {
        "mode": mode,
        "semantic_boundary_prior": semantic_boundary_prior,
        "head_prior": head_priors,
        "hand_prior": hand_priors,
        "part_foreground_prior": part_foreground_priors,
        "stats": {
            "head_prior_mean": float(head_priors.mean()),
            "hand_prior_mean": float(hand_priors.mean()),
            "semantic_boundary_mean": float(semantic_boundary_prior.mean()),
        },
    }


def make_parsing_overlay(frames: np.ndarray, parsing_output: dict) -> np.ndarray:
    frames = np.asarray(frames, dtype=np.uint8)
    semantic = np.asarray(parsing_output.get("semantic_boundary_prior"), dtype=np.float32)
    head = np.asarray(parsing_output.get("head_prior"), dtype=np.float32)
    hand = np.asarray(parsing_output.get("hand_prior"), dtype=np.float32)
    if semantic.shape != frames.shape[:3]:
        raise ValueError(f"parsing_output semantic boundary must match frames. Got {semantic.shape} vs {frames.shape[:3]}.")
    overlays = []
    for frame, semantic_frame, head_frame, hand_frame in zip(frames, semantic, head, hand):
        overlay = frame.astype(np.float32).copy()
        overlay[..., 0] = np.clip(overlay[..., 0] + semantic_frame * 120.0, 0.0, 255.0)
        overlay[..., 1] = np.clip(overlay[..., 1] + head_frame * 110.0, 0.0, 255.0)
        overlay[..., 2] = np.clip(overlay[..., 2] + hand_frame * 110.0, 0.0, 255.0)
        overlays.append(overlay.astype(np.uint8))
    return np.stack(overlays).astype(np.uint8)
