import cv2
import numpy as np


def _copy_meta(meta: dict) -> dict:
    return {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in meta.items()
    }


def _extract_valid_points(meta: dict, key: str, conf_thresh: float) -> np.ndarray:
    points = np.asarray(meta.get(key), dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        return np.zeros((0, 3), dtype=np.float32)
    valid = (
        np.isfinite(points[:, 0])
        & np.isfinite(points[:, 1])
        & np.isfinite(points[:, 2])
        & (points[:, 2] >= float(conf_thresh))
        & (points[:, 0] >= 0.0)
        & (points[:, 0] <= 1.0)
        & (points[:, 1] >= 0.0)
        & (points[:, 1] <= 1.0)
    )
    return points[valid].astype(np.float32)


def _bbox_from_normalized_points(points: np.ndarray, *, expand_ratio: float, min_size_ratio: float) -> np.ndarray | None:
    if points.size == 0:
        return None
    x1 = float(points[:, 0].min())
    x2 = float(points[:, 0].max())
    y1 = float(points[:, 1].min())
    y2 = float(points[:, 1].max())
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    width = max(float(x2 - x1), float(min_size_ratio))
    height = max(float(y2 - y1), float(min_size_ratio))
    width *= float(expand_ratio)
    height *= float(expand_ratio)
    x1 = np.clip(cx - width / 2.0, 0.0, 1.0)
    x2 = np.clip(cx + width / 2.0, 0.0, 1.0)
    y1 = np.clip(cy - height / 2.0, 0.0, 1.0)
    y2 = np.clip(cy + height / 2.0, 0.0, 1.0)
    if x2 - x1 < min_size_ratio:
        half = float(min_size_ratio) / 2.0
        x1 = np.clip(cx - half, 0.0, 1.0)
        x2 = np.clip(cx + half, 0.0, 1.0)
    if y2 - y1 < min_size_ratio:
        half = float(min_size_ratio) / 2.0
        y1 = np.clip(cy - half, 0.0, 1.0)
        y2 = np.clip(cy + half, 0.0, 1.0)
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _bbox_pixels_from_normalized(bbox_norm: np.ndarray, frame_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    height, width = frame_shape
    x1 = int(np.floor(np.clip(float(bbox_norm[0]), 0.0, 1.0) * width))
    y1 = int(np.floor(np.clip(float(bbox_norm[1]), 0.0, 1.0) * height))
    x2 = int(np.ceil(np.clip(float(bbox_norm[2]), 0.0, 1.0) * width))
    y2 = int(np.ceil(np.clip(float(bbox_norm[3]), 0.0, 1.0) * height))
    x1 = max(0, min(x1, width - 2))
    y1 = max(0, min(y1, height - 2))
    x2 = max(x1 + 2, min(x2, width))
    y2 = max(y1 + 2, min(y2, height))
    return x1, y1, x2, y2


def _resize_crop_if_needed(crop: np.ndarray, *, target_long_side: int | None) -> np.ndarray:
    if target_long_side is None or target_long_side <= 0:
        return crop
    height, width = crop.shape[:2]
    long_side = max(height, width)
    if long_side == target_long_side:
        return crop
    scale = float(target_long_side) / float(long_side)
    new_width = max(2, int(round(width * scale)))
    new_height = max(2, int(round(height * scale)))
    interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    return cv2.resize(crop, (new_width, new_height), interpolation=interpolation)


def _map_meta_from_roi_to_global(meta: dict, bbox_norm: np.ndarray, stage_shape: tuple[int, int]) -> dict:
    mapped = _copy_meta(meta)
    x1, y1, x2, y2 = [float(v) for v in bbox_norm]
    width_scale = max(x2 - x1, 1e-6)
    height_scale = max(y2 - y1, 1e-6)
    for key in ("keypoints_body", "keypoints_left_hand", "keypoints_right_hand", "keypoints_face"):
        points = np.asarray(mapped[key], dtype=np.float32).copy()
        if points.ndim == 2 and points.shape[1] >= 2:
            points[:, 0] = np.clip(x1 + points[:, 0] * width_scale, 0.0, 1.0)
            points[:, 1] = np.clip(y1 + points[:, 1] * height_scale, 0.0, 1.0)
        mapped[key] = points
    mapped["width"] = int(stage_shape[1])
    mapped["height"] = int(stage_shape[0])
    return mapped


def propose_person_roi_bboxes(
    metas: list[dict],
    *,
    body_conf_thresh: float = 0.35,
    hand_conf_thresh: float = 0.25,
    face_conf_thresh: float = 0.35,
    expand_ratio: float = 1.18,
    min_size_ratio: float = 0.20,
    fallback_mode: str = "full_frame",
) -> tuple[list[np.ndarray], dict]:
    bboxes = []
    sources = []
    for meta in metas:
        points = []
        for key, thresh in (
            ("keypoints_body", body_conf_thresh),
            ("keypoints_left_hand", hand_conf_thresh),
            ("keypoints_right_hand", hand_conf_thresh),
            ("keypoints_face", face_conf_thresh),
        ):
            pts = _extract_valid_points(meta, key, thresh)
            if pts.size > 0:
                points.append(pts[:, :2])
        if points:
            bbox = _bbox_from_normalized_points(
                np.concatenate(points, axis=0),
                expand_ratio=expand_ratio,
                min_size_ratio=min_size_ratio,
            )
            source = "pose_points"
        else:
            bbox = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
            source = fallback_mode
        bboxes.append(bbox.astype(np.float32))
        sources.append(source)
    widths = np.array([bbox[2] - bbox[0] for bbox in bboxes], dtype=np.float32)
    heights = np.array([bbox[3] - bbox[1] for bbox in bboxes], dtype=np.float32)
    stats = {
        "frame_count": int(len(bboxes)),
        "coverage_ratio": float(sum(source != fallback_mode for source in sources) / max(len(sources), 1)),
        "fallback_ratio": float(sum(source == fallback_mode for source in sources) / max(len(sources), 1)),
        "mean_width_ratio": float(widths.mean()) if len(widths) else 0.0,
        "mean_height_ratio": float(heights.mean()) if len(heights) else 0.0,
        "sources": {
            "pose_points": int(sum(source == "pose_points" for source in sources)),
            fallback_mode: int(sum(source == fallback_mode for source in sources)),
        },
    }
    return bboxes, stats


def propose_face_roi_bboxes(
    metas: list[dict],
    face_bboxes: list[list[int]] | list[tuple[int, int, int, int]],
    *,
    image_shape: tuple[int, int],
    expand_ratio: float = 1.65,
    min_size_ratio: float = 0.12,
    face_conf_thresh: float = 0.35,
) -> tuple[list[np.ndarray], dict]:
    height, width = image_shape
    bboxes = []
    sources = []
    for meta, face_bbox in zip(metas, face_bboxes):
        if face_bbox is not None:
            x1, x2, y1, y2 = [float(v) for v in face_bbox]
            bbox_norm = np.array([x1 / width, y1 / height, x2 / width, y2 / height], dtype=np.float32)
            cx = 0.5 * (bbox_norm[0] + bbox_norm[2])
            cy = 0.5 * (bbox_norm[1] + bbox_norm[3])
            face_width = max(float(bbox_norm[2] - bbox_norm[0]), min_size_ratio)
            face_height = max(float(bbox_norm[3] - bbox_norm[1]), min_size_ratio)
            roi_width = face_width * float(expand_ratio)
            roi_height = face_height * (float(expand_ratio) + 0.35)
            shoulders = _extract_valid_points(meta, "keypoints_body", face_conf_thresh)
            if shoulders.size > 0:
                shoulder_indices = [2, 5]
                shoulder_points = []
                body = np.asarray(meta["keypoints_body"], dtype=np.float32)
                for idx in shoulder_indices:
                    if idx < body.shape[0] and body[idx, 2] >= face_conf_thresh:
                        shoulder_points.append(body[idx, :2])
                if shoulder_points:
                    shoulder_y = max(float(point[1]) for point in shoulder_points)
                    cy = 0.55 * cy + 0.45 * min(shoulder_y, 0.98)
            bbox = _bbox_from_normalized_points(
                np.array(
                    [
                        [cx - roi_width / 2.0, cy - roi_height / 2.0],
                        [cx + roi_width / 2.0, cy + roi_height / 2.0],
                    ],
                    dtype=np.float32,
                ),
                expand_ratio=1.0,
                min_size_ratio=min_size_ratio,
            )
            source = "face_bbox"
        else:
            bbox = np.array([0.25, 0.05, 0.75, 0.55], dtype=np.float32)
            source = "fallback"
        bboxes.append(bbox.astype(np.float32))
        sources.append(source)
    widths = np.array([bbox[2] - bbox[0] for bbox in bboxes], dtype=np.float32)
    heights = np.array([bbox[3] - bbox[1] for bbox in bboxes], dtype=np.float32)
    stats = {
        "frame_count": int(len(bboxes)),
        "coverage_ratio": float(sum(source != "fallback" for source in sources) / max(len(sources), 1)),
        "fallback_ratio": float(sum(source == "fallback" for source in sources) / max(len(sources), 1)),
        "mean_width_ratio": float(widths.mean()) if len(widths) else 0.0,
        "mean_height_ratio": float(heights.mean()) if len(heights) else 0.0,
    }
    return bboxes, stats


def run_roi_pose_refinement(
    *,
    pose2d,
    stage_frames: list[np.ndarray],
    roi_bboxes_norm: list[np.ndarray],
    target_long_side: int | None = None,
) -> tuple[list[dict], dict]:
    if len(stage_frames) != len(roi_bboxes_norm):
        raise ValueError("stage_frames and roi_bboxes_norm must have identical length.")
    crops = []
    mappings = []
    skipped = 0
    for frame, bbox_norm in zip(stage_frames, roi_bboxes_norm):
        x1, y1, x2, y2 = _bbox_pixels_from_normalized(bbox_norm, frame.shape[:2])
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            skipped += 1
            crop = np.zeros((32, 32, 3), dtype=np.uint8)
        crop = _resize_crop_if_needed(crop, target_long_side=target_long_side)
        crops.append(crop)
        mappings.append(np.asarray(bbox_norm, dtype=np.float32))
    refined = pose2d(crops)
    mapped_metas = []
    for meta, bbox_norm in zip(refined, mappings):
        mapped_metas.append(_map_meta_from_roi_to_global(meta, bbox_norm, stage_frames[0].shape[:2]))
    stats = {
        "frame_count": int(len(stage_frames)),
        "skipped_frame_count": int(skipped),
        "coverage_ratio": float((len(stage_frames) - skipped) / max(len(stage_frames), 1)),
        "target_long_side": None if target_long_side is None else int(target_long_side),
        "mean_crop_height": float(np.mean([crop.shape[0] for crop in crops])) if crops else 0.0,
        "mean_crop_width": float(np.mean([crop.shape[1] for crop in crops])) if crops else 0.0,
    }
    return mapped_metas, stats


def _fuse_point_sets(
    base_points: np.ndarray,
    refine_points: np.ndarray,
    *,
    prefer_weight: float,
    conf_margin: float,
    max_coordinate_shift: float,
) -> np.ndarray:
    fused = np.asarray(base_points, dtype=np.float32).copy()
    refine_points = np.asarray(refine_points, dtype=np.float32)
    for index in range(min(len(fused), len(refine_points))):
        base = fused[index]
        refine = refine_points[index]
        base_valid = bool(np.isfinite(base[:3]).all() and base[2] > 0.0)
        refine_valid = bool(np.isfinite(refine[:3]).all() and refine[2] > 0.0)
        if refine_valid and not base_valid:
            fused[index] = refine
            continue
        if not (base_valid and refine_valid):
            continue
        delta = refine[:2] - base[:2]
        distance = float(np.linalg.norm(delta))
        if max_coordinate_shift > 0.0 and distance > max_coordinate_shift:
            refine_xy = base[:2] + delta / max(distance, 1e-6) * max_coordinate_shift
        else:
            refine_xy = refine[:2]
        if refine[2] >= base[2] + conf_margin:
            alpha = float(prefer_weight)
        elif base[2] >= refine[2] + conf_margin:
            alpha = float(1.0 - prefer_weight) * 0.5
        else:
            alpha = 0.5
        fused[index, :2] = np.clip((1.0 - alpha) * base[:2] + alpha * refine_xy, 0.0, 1.0)
        fused[index, 2] = max(float(base[2]), float(refine[2]))
    return fused.astype(np.float32)


def fuse_multistage_pose_metas(
    global_metas: list[dict],
    *,
    person_metas: list[dict] | None = None,
    face_metas: list[dict] | None = None,
    person_weight: float = 0.7,
    face_weight: float = 0.85,
    conf_margin: float = 0.03,
    person_max_coordinate_shift: float = 0.08,
    face_max_coordinate_shift: float = 0.05,
) -> tuple[list[dict], dict]:
    fused = [_copy_meta(meta) for meta in global_metas]
    person_updates = 0
    face_updates = 0
    if person_metas is not None:
        for fused_meta, refine_meta in zip(fused, person_metas):
            for key in ("keypoints_body", "keypoints_left_hand", "keypoints_right_hand", "keypoints_face"):
                before = np.asarray(fused_meta[key], dtype=np.float32)
                after = _fuse_point_sets(
                    before,
                    np.asarray(refine_meta[key], dtype=np.float32),
                    prefer_weight=person_weight,
                    conf_margin=conf_margin,
                    max_coordinate_shift=person_max_coordinate_shift,
                )
                person_updates += int(np.any(np.abs(after[:, :2] - before[:, :2]) > 1e-6))
                fused_meta[key] = after
    if face_metas is not None:
        for fused_meta, refine_meta in zip(fused, face_metas):
            before = np.asarray(fused_meta["keypoints_face"], dtype=np.float32)
            after = _fuse_point_sets(
                before,
                np.asarray(refine_meta["keypoints_face"], dtype=np.float32),
                prefer_weight=face_weight,
                conf_margin=conf_margin,
                max_coordinate_shift=face_max_coordinate_shift,
            )
            face_updates += int(np.any(np.abs(after[:, :2] - before[:, :2]) > 1e-6))
            fused_meta["keypoints_face"] = after
    stats = {
        "frame_count": int(len(fused)),
        "person_update_frames": int(person_updates),
        "face_update_frames": int(face_updates),
    }
    return fused, stats
