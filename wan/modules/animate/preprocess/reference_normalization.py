from pathlib import Path

import cv2
import numpy as np


BODY_NOSE = 0
BODY_NECK = 1
BODY_RIGHT_SHOULDER = 2
BODY_LEFT_SHOULDER = 5
BODY_RIGHT_HIP = 8
BODY_LEFT_HIP = 11
BODY_RIGHT_ANKLE = 10
BODY_LEFT_ANKLE = 13


def _as_array(points):
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return np.zeros((0, 3), dtype=np.float32)
    if arr.shape[1] == 2:
        arr = np.concatenate([arr, np.ones((arr.shape[0], 1), dtype=np.float32)], axis=1)
    return arr[:, :3]


def _valid_points(points, image_shape, conf_thresh):
    height, width = image_shape
    arr = _as_array(points)
    if arr.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    valid = (
        np.isfinite(arr[:, 0])
        & np.isfinite(arr[:, 1])
        & np.isfinite(arr[:, 2])
        & (arr[:, 2] >= conf_thresh)
        & (arr[:, 0] >= 0.0)
        & (arr[:, 0] <= 1.0)
        & (arr[:, 1] >= 0.0)
        & (arr[:, 1] <= 1.0)
    )
    coords = arr[valid, :2].copy()
    if coords.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    coords[:, 0] *= float(width)
    coords[:, 1] *= float(height)
    return coords


def _valid_body_point(meta, index, image_shape, conf_thresh):
    arr = _as_array(meta.get("keypoints_body", []))
    if arr.shape[0] <= index:
        return None
    point = arr[index]
    x, y, conf = float(point[0]), float(point[1]), float(point[2])
    if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(conf)):
        return None
    if conf < float(conf_thresh):
        return None
    height, width = image_shape
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        return None
    return np.asarray([x * float(width), y * float(height)], dtype=np.float32)


def _line_midpoint(lhs, rhs):
    if lhs is not None and rhs is not None:
        return (lhs + rhs) / 2.0
    return lhs if lhs is not None else rhs


def _distance(lhs, rhs):
    if lhs is None or rhs is None:
        return None
    return float(np.linalg.norm(np.asarray(lhs, dtype=np.float32) - np.asarray(rhs, dtype=np.float32)))


def _clip_structure_anchor_order(top_y, shoulder_y, hip_y, foot_y, bbox):
    bbox = np.asarray(bbox, dtype=np.float32)
    top = float(np.clip(top_y, bbox[1], bbox[3]))
    shoulder = float(np.clip(shoulder_y, top + 1.0, bbox[3]))
    hip = float(np.clip(hip_y, shoulder + 1.0, bbox[3]))
    foot = float(np.clip(foot_y, hip + 1.0, bbox[3]))
    return top, shoulder, hip, foot


def _face_bbox_from_meta(meta, image_shape, conf_thresh):
    points = _valid_points(meta.get("keypoints_face", []), image_shape=image_shape, conf_thresh=conf_thresh)
    if points.size == 0:
        return None
    x1 = float(points[:, 0].min())
    y1 = float(points[:, 1].min())
    x2 = float(points[:, 0].max())
    y2 = float(points[:, 1].max())
    if x2 <= x1 or y2 <= y1:
        return None
    return np.asarray([x1, y1, x2, y2], dtype=np.float32)


def bbox_from_pose_meta(meta, image_shape, conf_thresh=0.35):
    point_groups = []
    for key in ("keypoints_body", "keypoints_face", "keypoints_left_hand", "keypoints_right_hand"):
        if key not in meta:
            continue
        points = _valid_points(meta[key], image_shape=image_shape, conf_thresh=conf_thresh)
        if points.size > 0:
            point_groups.append(points)
    if not point_groups:
        return None
    points = np.concatenate(point_groups, axis=0)
    x1 = float(points[:, 0].min())
    y1 = float(points[:, 1].min())
    x2 = float(points[:, 0].max())
    y2 = float(points[:, 1].max())
    if x2 <= x1 or y2 <= y1:
        return None
    return np.asarray([x1, y1, x2, y2], dtype=np.float32)


def structure_from_pose_meta(meta, image_shape, conf_thresh=0.35):
    bbox = bbox_from_pose_meta(meta, image_shape=image_shape, conf_thresh=conf_thresh)
    if bbox is None:
        return None

    bbox = np.asarray(bbox, dtype=np.float32)
    bbox_w = max(float(bbox[2] - bbox[0]), 1.0)
    bbox_h = max(float(bbox[3] - bbox[1]), 1.0)

    face_bbox = _face_bbox_from_meta(meta, image_shape=image_shape, conf_thresh=conf_thresh)
    nose = _valid_body_point(meta, BODY_NOSE, image_shape, conf_thresh)
    neck = _valid_body_point(meta, BODY_NECK, image_shape, conf_thresh)
    right_shoulder = _valid_body_point(meta, BODY_RIGHT_SHOULDER, image_shape, conf_thresh)
    left_shoulder = _valid_body_point(meta, BODY_LEFT_SHOULDER, image_shape, conf_thresh)
    right_hip = _valid_body_point(meta, BODY_RIGHT_HIP, image_shape, conf_thresh)
    left_hip = _valid_body_point(meta, BODY_LEFT_HIP, image_shape, conf_thresh)
    right_ankle = _valid_body_point(meta, BODY_RIGHT_ANKLE, image_shape, conf_thresh)
    left_ankle = _valid_body_point(meta, BODY_LEFT_ANKLE, image_shape, conf_thresh)

    shoulder_mid = _line_midpoint(left_shoulder, right_shoulder)
    hip_mid = _line_midpoint(left_hip, right_hip)
    ankle_candidates = [point[1] for point in (left_ankle, right_ankle) if point is not None]

    top_y = float(face_bbox[1]) if face_bbox is not None else float(bbox[1])
    if nose is not None:
        top_y = min(top_y, float(nose[1] - 0.18 * bbox_h))
    shoulder_y = None
    if shoulder_mid is not None:
        shoulder_y = float(shoulder_mid[1])
    elif neck is not None:
        shoulder_y = float(neck[1] + 0.05 * bbox_h)
    else:
        shoulder_y = float(bbox[1] + 0.22 * bbox_h)

    if face_bbox is not None:
        shoulder_y = max(shoulder_y, float(face_bbox[3] + 0.02 * bbox_h))

    hip_y = float(hip_mid[1]) if hip_mid is not None else float(bbox[1] + 0.60 * bbox_h)
    foot_y = float(max(ankle_candidates)) if ankle_candidates else float(bbox[3])

    top_y, shoulder_y, hip_y, foot_y = _clip_structure_anchor_order(
        top_y,
        shoulder_y,
        hip_y,
        foot_y,
        bbox,
    )

    head_h = max(shoulder_y - top_y, 0.12 * bbox_h)
    torso_h = max(hip_y - shoulder_y, 0.22 * bbox_h)
    leg_h = max(foot_y - hip_y, 0.28 * bbox_h)
    total_h = max(foot_y - top_y, head_h + torso_h + leg_h, 1.0)

    shoulder_w = _distance(left_shoulder, right_shoulder)
    hip_w = _distance(left_hip, right_hip)
    if shoulder_w is None:
        shoulder_w = 0.55 * bbox_w
    if hip_w is None:
        hip_w = 0.45 * bbox_w

    return {
        "bbox": [float(v) for v in bbox.tolist()],
        "face_bbox": None if face_bbox is None else [float(v) for v in face_bbox.tolist()],
        "center_x": float((bbox[0] + bbox[2]) / 2.0),
        "center_y": float((bbox[1] + bbox[3]) / 2.0),
        "top_y": float(top_y),
        "shoulder_y": float(shoulder_y),
        "hip_y": float(hip_y),
        "foot_y": float(foot_y),
        "bbox_width": float(bbox_w),
        "bbox_height": float(bbox_h),
        "shoulder_width": float(shoulder_w),
        "hip_width": float(hip_w),
        "head_height": float(head_h),
        "torso_height": float(torso_h),
        "leg_height": float(leg_h),
        "person_height": float(total_h),
        "width_height_ratio": float(bbox_w / max(total_h, 1.0)),
        "head_ratio": float(head_h / max(total_h, 1.0)),
        "torso_ratio": float(torso_h / max(total_h, 1.0)),
        "leg_ratio": float(leg_h / max(total_h, 1.0)),
    }


def scale_bbox_between_shapes(bbox, from_shape, to_shape):
    if bbox is None:
        return None
    from_h, from_w = from_shape
    to_h, to_w = to_shape
    scale_x = float(to_w) / max(float(from_w), 1.0)
    scale_y = float(to_h) / max(float(from_h), 1.0)
    x1, y1, x2, y2 = np.asarray(bbox, dtype=np.float32)
    return np.asarray([x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y], dtype=np.float32)


def scale_structure_between_shapes(structure, from_shape, to_shape):
    if structure is None:
        return None
    from_h, from_w = from_shape
    to_h, to_w = to_shape
    scale_x = float(to_w) / max(float(from_w), 1.0)
    scale_y = float(to_h) / max(float(from_h), 1.0)
    scaled = dict(structure)
    if structure.get("bbox") is not None:
        scaled["bbox"] = [
            float(structure["bbox"][0] * scale_x),
            float(structure["bbox"][1] * scale_y),
            float(structure["bbox"][2] * scale_x),
            float(structure["bbox"][3] * scale_y),
        ]
    if structure.get("face_bbox") is not None:
        scaled["face_bbox"] = [
            float(structure["face_bbox"][0] * scale_x),
            float(structure["face_bbox"][1] * scale_y),
            float(structure["face_bbox"][2] * scale_x),
            float(structure["face_bbox"][3] * scale_y),
        ]
    for key in ("center_x", "bbox_width", "shoulder_width", "hip_width"):
        if key in structure:
            scaled[key] = float(structure[key]) * scale_x
    for key in (
        "center_y",
        "top_y",
        "shoulder_y",
        "hip_y",
        "foot_y",
        "bbox_height",
        "head_height",
        "torso_height",
        "leg_height",
        "person_height",
    ):
        if key in structure:
            scaled[key] = float(structure[key]) * scale_y
    return scaled


def _bbox_to_cxcywh(bbox):
    x1, y1, x2, y2 = np.asarray(bbox, dtype=np.float32)
    return np.asarray([(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1], dtype=np.float32)


def _cxcywh_to_bbox(cxcywh):
    cx, cy, width, height = np.asarray(cxcywh, dtype=np.float32)
    return np.asarray([cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0], dtype=np.float32)


def estimate_driver_target_bbox(
    pose_metas,
    image_shape,
    *,
    source="median_first_n",
    num_frames=16,
    conf_thresh=0.35,
):
    if source not in {"first_frame", "median_first_n"}:
        raise ValueError(f"Unsupported reference_target_bbox_source: {source}")
    if not pose_metas:
        return None, {"source": source, "used_frames": 0, "valid_frames": 0}
    frame_limit = 1 if source == "first_frame" else max(1, min(int(num_frames), len(pose_metas)))
    boxes = []
    used_indices = []
    for frame_index, meta in enumerate(pose_metas[:frame_limit]):
        bbox = bbox_from_pose_meta(meta, image_shape=image_shape, conf_thresh=conf_thresh)
        if bbox is None:
            continue
        boxes.append(bbox)
        used_indices.append(frame_index)
        if source == "first_frame":
            break
    if not boxes:
        return None, {
            "source": source,
            "used_frames": frame_limit,
            "valid_frames": 0,
            "frame_indices": [],
        }
    boxes = np.stack(boxes, axis=0)
    target_bbox = np.median(boxes, axis=0).astype(np.float32)
    return target_bbox, {
        "source": source,
        "used_frames": frame_limit,
        "valid_frames": int(len(boxes)),
        "frame_indices": used_indices,
        "median_bbox": [float(v) for v in target_bbox.tolist()],
    }


def estimate_driver_target_structure(
    pose_metas,
    image_shape,
    *,
    source="median_first_n",
    num_frames=16,
    conf_thresh=0.35,
):
    if source not in {"first_frame", "median_first_n"}:
        raise ValueError(f"Unsupported reference_target_bbox_source: {source}")
    if not pose_metas:
        return None, {"source": source, "used_frames": 0, "valid_frames": 0, "frame_indices": []}
    frame_limit = 1 if source == "first_frame" else max(1, min(int(num_frames), len(pose_metas)))
    structures = []
    frame_indices = []
    for frame_index, meta in enumerate(pose_metas[:frame_limit]):
        structure = structure_from_pose_meta(meta, image_shape=image_shape, conf_thresh=conf_thresh)
        if structure is None:
            continue
        structures.append(structure)
        frame_indices.append(frame_index)
        if source == "first_frame":
            break
    if not structures:
        return None, {
            "source": source,
            "used_frames": frame_limit,
            "valid_frames": 0,
            "frame_indices": [],
        }

    keys = (
        "center_x",
        "center_y",
        "top_y",
        "shoulder_y",
        "hip_y",
        "foot_y",
        "bbox_width",
        "bbox_height",
        "shoulder_width",
        "hip_width",
        "head_height",
        "torso_height",
        "leg_height",
        "person_height",
        "width_height_ratio",
        "head_ratio",
        "torso_ratio",
        "leg_ratio",
    )
    target = {}
    for key in keys:
        target[key] = float(np.median([structure[key] for structure in structures]))
    target_bbox = _cxcywh_to_bbox(
        [
            target["center_x"],
            target["center_y"],
            target["bbox_width"],
            target["bbox_height"],
        ]
    )
    target["bbox"] = [float(v) for v in target_bbox.tolist()]
    target["face_bbox"] = None
    return target, {
        "source": source,
        "used_frames": frame_limit,
        "valid_frames": int(len(structures)),
        "frame_indices": frame_indices,
        "median_structure": target,
    }


def _resize_full_image(image, scale_x, scale_y=None):
    if scale_y is None:
        scale_y = scale_x
    height, width = image.shape[:2]
    new_width = max(1, int(round(width * float(scale_x))))
    new_height = max(1, int(round(height * float(scale_y))))
    interpolation = cv2.INTER_AREA if max(scale_x, scale_y) < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(image, (new_width, new_height), interpolation=interpolation)


def project_bbox_with_letterbox(bbox, image_shape, canvas_shape):
    if bbox is None:
        return None
    image_h, image_w = image_shape
    canvas_h, canvas_w = canvas_shape
    scale = min(float(canvas_w) / max(float(image_w), 1.0), float(canvas_h) / max(float(image_h), 1.0))
    new_w = float(image_w) * scale
    new_h = float(image_h) * scale
    pad_x = (float(canvas_w) - new_w) / 2.0
    pad_y = (float(canvas_h) - new_h) / 2.0
    x1, y1, x2, y2 = np.asarray(bbox, dtype=np.float32)
    return np.asarray(
        [
            x1 * scale + pad_x,
            y1 * scale + pad_y,
            x2 * scale + pad_x,
            y2 * scale + pad_y,
        ],
        dtype=np.float32,
    )


def project_structure_with_letterbox(structure, image_shape, canvas_shape):
    if structure is None:
        return None
    image_h, image_w = image_shape
    canvas_h, canvas_w = canvas_shape
    scale = min(float(canvas_w) / max(float(image_w), 1.0), float(canvas_h) / max(float(image_h), 1.0))
    new_w = float(image_w) * scale
    new_h = float(image_h) * scale
    pad_x = (float(canvas_w) - new_w) / 2.0
    pad_y = (float(canvas_h) - new_h) / 2.0
    projected = dict(structure)
    if structure.get("bbox") is not None:
        projected["bbox"] = project_bbox_with_letterbox(structure["bbox"], image_shape, canvas_shape).tolist()
    if structure.get("face_bbox") is not None:
        projected["face_bbox"] = project_bbox_with_letterbox(structure["face_bbox"], image_shape, canvas_shape).tolist()
    for key in ("center_x", "bbox_width", "shoulder_width", "hip_width"):
        if key in structure:
            projected[key] = float(structure[key]) * scale + (pad_x if key == "center_x" else 0.0)
    for key in (
        "center_y",
        "top_y",
        "shoulder_y",
        "hip_y",
        "foot_y",
        "bbox_height",
        "head_height",
        "torso_height",
        "leg_height",
        "person_height",
    ):
        if key in structure:
            projected[key] = float(structure[key]) * scale + (pad_y if key == "center_y" else 0.0)
    return projected


def normalize_reference_image(
    reference_image,
    *,
    reference_bbox,
    target_bbox,
    canvas_shape,
    scale_clamp_min=0.75,
    scale_clamp_max=1.6,
):
    canvas_h, canvas_w = canvas_shape
    if reference_bbox is None or target_bbox is None:
        return None, {
            "enabled": False,
            "applied": False,
            "reason": "missing_bbox",
        }

    ref_cxcywh = _bbox_to_cxcywh(reference_bbox)
    target_cxcywh = _bbox_to_cxcywh(target_bbox)
    ref_area = max(ref_cxcywh[2] * ref_cxcywh[3], 1.0)
    target_area = max(target_cxcywh[2] * target_cxcywh[3], 1.0)
    raw_scale = float(np.sqrt(target_area / ref_area))
    scale = float(np.clip(raw_scale, float(scale_clamp_min), float(scale_clamp_max)))

    scaled_image = _resize_full_image(reference_image, scale)
    scaled_bbox = np.asarray(reference_bbox, dtype=np.float32) * scale
    scaled_cxcywh = _bbox_to_cxcywh(scaled_bbox)
    target_center = target_cxcywh[:2]

    offset_x = int(round(float(target_center[0] - scaled_cxcywh[0])))
    offset_y = int(round(float(target_center[1] - scaled_cxcywh[1])))

    canvas = np.zeros((canvas_h, canvas_w, reference_image.shape[2]), dtype=np.uint8)
    src_h, src_w = scaled_image.shape[:2]

    dst_x1 = max(0, offset_x)
    dst_y1 = max(0, offset_y)
    dst_x2 = min(canvas_w, offset_x + src_w)
    dst_y2 = min(canvas_h, offset_y + src_h)

    src_x1 = max(0, -offset_x)
    src_y1 = max(0, -offset_y)
    src_x2 = src_x1 + max(0, dst_x2 - dst_x1)
    src_y2 = src_y1 + max(0, dst_y2 - dst_y1)

    if dst_x2 > dst_x1 and dst_y2 > dst_y1:
        canvas[dst_y1:dst_y2, dst_x1:dst_x2] = scaled_image[src_y1:src_y2, src_x1:src_x2]

    normalized_bbox = scaled_bbox.copy()
    normalized_bbox[[0, 2]] += float(offset_x)
    normalized_bbox[[1, 3]] += float(offset_y)
    normalized_bbox[0] = np.clip(normalized_bbox[0], 0.0, float(canvas_w))
    normalized_bbox[2] = np.clip(normalized_bbox[2], 0.0, float(canvas_w))
    normalized_bbox[1] = np.clip(normalized_bbox[1], 0.0, float(canvas_h))
    normalized_bbox[3] = np.clip(normalized_bbox[3], 0.0, float(canvas_h))

    return canvas, {
        "enabled": True,
        "applied": True,
        "reason": "ok",
        "reference_bbox": [float(v) for v in np.asarray(reference_bbox, dtype=np.float32).tolist()],
        "target_bbox": [float(v) for v in np.asarray(target_bbox, dtype=np.float32).tolist()],
        "normalized_bbox": [float(v) for v in normalized_bbox.tolist()],
        "raw_scale_factor": raw_scale,
        "applied_scale_factor": scale,
        "scale_clamp_min": float(scale_clamp_min),
        "scale_clamp_max": float(scale_clamp_max),
        "offset_x": int(offset_x),
        "offset_y": int(offset_y),
    }


def _piecewise_map(values, dst_anchors, src_anchors):
    dst_anchors = np.asarray(dst_anchors, dtype=np.float32)
    src_anchors = np.asarray(src_anchors, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    mapped = np.empty_like(values, dtype=np.float32)
    for index, value in enumerate(values):
        if value <= dst_anchors[0]:
            dst_span = max(dst_anchors[1] - dst_anchors[0], 1.0)
            src_span = src_anchors[1] - src_anchors[0]
            mapped[index] = src_anchors[0] + (value - dst_anchors[0]) * (src_span / dst_span)
            continue
        if value >= dst_anchors[-1]:
            dst_span = max(dst_anchors[-1] - dst_anchors[-2], 1.0)
            src_span = src_anchors[-1] - src_anchors[-2]
            mapped[index] = src_anchors[-1] + (value - dst_anchors[-1]) * (src_span / dst_span)
            continue
        for segment in range(len(dst_anchors) - 1):
            left = dst_anchors[segment]
            right = dst_anchors[segment + 1]
            if left <= value <= right:
                ratio = (value - left) / max(right - left, 1.0)
                mapped[index] = src_anchors[segment] + ratio * (src_anchors[segment + 1] - src_anchors[segment])
                break
    return mapped


def normalize_reference_image_structure_aware(
    reference_image,
    *,
    reference_structure,
    target_structure,
    canvas_shape,
    scale_clamp_min=0.75,
    scale_clamp_max=1.6,
    segment_clamp_min=0.8,
    segment_clamp_max=1.25,
    width_budget_ratio=1.05,
    height_budget_ratio=1.05,
):
    canvas_h, canvas_w = canvas_shape
    if reference_structure is None or target_structure is None:
        return None, {
            "enabled": False,
            "applied": False,
            "reason": "missing_structure",
        }

    ref_bbox = np.asarray(reference_structure["bbox"], dtype=np.float32)
    target_bbox = np.asarray(target_structure["bbox"], dtype=np.float32)
    ref_bbox_w = max(float(ref_bbox[2] - ref_bbox[0]), 1.0)
    ref_bbox_h = max(float(ref_bbox[3] - ref_bbox[1]), 1.0)
    target_bbox_w = max(float(target_bbox[2] - target_bbox[0]), 1.0)
    target_bbox_h = max(float(target_bbox[3] - target_bbox[1]), 1.0)

    raw_bbox_scale = float(np.sqrt((target_bbox_w * target_bbox_h) / max(ref_bbox_w * ref_bbox_h, 1.0)))
    raw_shoulder_scale = float(target_structure["shoulder_width"] / max(reference_structure["shoulder_width"], 1.0))
    raw_width_scale = float(0.35 * raw_bbox_scale + 0.65 * raw_shoulder_scale)
    applied_width_scale = float(np.clip(raw_width_scale, float(scale_clamp_min), float(scale_clamp_max)))

    width_budget_scale = float((target_bbox_w * float(width_budget_ratio)) / ref_bbox_w)
    width_budget_triggered = applied_width_scale > width_budget_scale
    applied_width_scale = min(applied_width_scale, width_budget_scale)

    raw_head_scale = float(target_structure["head_height"] / max(reference_structure["head_height"], 1.0))
    raw_torso_scale = float(target_structure["torso_height"] / max(reference_structure["torso_height"], 1.0))
    raw_leg_scale = float(target_structure["leg_height"] / max(reference_structure["leg_height"], 1.0))

    head_scale = float(np.clip(raw_head_scale, float(segment_clamp_min), float(segment_clamp_max)))
    torso_scale = float(np.clip(raw_torso_scale, float(segment_clamp_min), float(segment_clamp_max)))
    leg_scale = float(np.clip(raw_leg_scale, float(segment_clamp_min), float(segment_clamp_max)))

    dest_head_h = float(reference_structure["head_height"] * head_scale)
    dest_torso_h = float(reference_structure["torso_height"] * torso_scale)
    dest_leg_h = float(reference_structure["leg_height"] * leg_scale)
    raw_total_h = dest_head_h + dest_torso_h + dest_leg_h
    height_budget = float(target_bbox_h * float(height_budget_ratio))
    height_budget_triggered = raw_total_h > height_budget
    if raw_total_h > height_budget:
        reduce_scale = height_budget / max(raw_total_h, 1.0)
        dest_head_h *= reduce_scale
        dest_torso_h *= reduce_scale
        dest_leg_h *= reduce_scale
    applied_total_h = dest_head_h + dest_torso_h + dest_leg_h

    target_foot_y = float(target_structure["foot_y"])
    dest_top_y = target_foot_y - applied_total_h
    dest_shoulder_y = dest_top_y + dest_head_h
    dest_hip_y = dest_shoulder_y + dest_torso_h
    dest_foot_y = dest_hip_y + dest_leg_h

    shift_y = 0.0
    if dest_top_y < 0.0:
        shift_y = -dest_top_y
    if dest_foot_y + shift_y > float(canvas_h):
        shift_y += float(canvas_h) - (dest_foot_y + shift_y)
    dest_top_y += shift_y
    dest_shoulder_y += shift_y
    dest_hip_y += shift_y
    dest_foot_y += shift_y

    dst_anchors = np.asarray([dest_top_y, dest_shoulder_y, dest_hip_y, dest_foot_y], dtype=np.float32)
    src_anchors = np.asarray(
        [
            float(reference_structure["top_y"]),
            float(reference_structure["shoulder_y"]),
            float(reference_structure["hip_y"]),
            float(reference_structure["foot_y"]),
        ],
        dtype=np.float32,
    )

    target_center_x = float(target_structure["center_x"])
    ref_center_x = float(reference_structure["center_x"])
    dst_x, dst_y = np.meshgrid(np.arange(canvas_w, dtype=np.float32), np.arange(canvas_h, dtype=np.float32))
    src_x = ((dst_x - target_center_x) / max(applied_width_scale, 1e-6)) + ref_center_x
    src_y = _piecewise_map(dst_y.reshape(-1), dst_anchors=dst_anchors, src_anchors=src_anchors).reshape(canvas_h, canvas_w)

    canvas = cv2.remap(
        reference_image,
        src_x.astype(np.float32),
        src_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    normalized_structure = {
        "bbox": [
            float(target_center_x - ref_bbox_w * applied_width_scale / 2.0),
            float(dest_top_y),
            float(target_center_x + ref_bbox_w * applied_width_scale / 2.0),
            float(dest_foot_y),
        ],
        "center_x": float(target_center_x),
        "center_y": float((dest_top_y + dest_foot_y) / 2.0),
        "top_y": float(dest_top_y),
        "shoulder_y": float(dest_shoulder_y),
        "hip_y": float(dest_hip_y),
        "foot_y": float(dest_foot_y),
        "bbox_width": float(ref_bbox_w * applied_width_scale),
        "bbox_height": float(dest_foot_y - dest_top_y),
        "shoulder_width": float(reference_structure["shoulder_width"] * applied_width_scale),
        "hip_width": float(reference_structure["hip_width"] * applied_width_scale),
        "head_height": float(dest_head_h),
        "torso_height": float(dest_torso_h),
        "leg_height": float(dest_leg_h),
        "person_height": float(dest_foot_y - dest_top_y),
        "width_height_ratio": float((ref_bbox_w * applied_width_scale) / max(dest_foot_y - dest_top_y, 1.0)),
        "head_ratio": float(dest_head_h / max(dest_foot_y - dest_top_y, 1.0)),
        "torso_ratio": float(dest_torso_h / max(dest_foot_y - dest_top_y, 1.0)),
        "leg_ratio": float(dest_leg_h / max(dest_foot_y - dest_top_y, 1.0)),
        "face_bbox": None,
    }

    return canvas, {
        "enabled": True,
        "applied": True,
        "reason": "ok",
        "reference_bbox": [float(v) for v in ref_bbox.tolist()],
        "target_bbox": [float(v) for v in target_bbox.tolist()],
        "normalized_bbox": [float(v) for v in normalized_structure["bbox"]],
        "reference_structure": reference_structure,
        "target_structure": target_structure,
        "normalized_structure": normalized_structure,
        "raw_bbox_scale_factor": raw_bbox_scale,
        "raw_shoulder_scale_factor": raw_shoulder_scale,
        "applied_width_scale_factor": float(applied_width_scale),
        "raw_segment_scales": {
            "head": float(raw_head_scale),
            "torso": float(raw_torso_scale),
            "legs": float(raw_leg_scale),
        },
        "applied_segment_scales": {
            "head": float(head_scale),
            "torso": float(torso_scale),
            "legs": float(leg_scale),
        },
        "segment_clamp_min": float(segment_clamp_min),
        "segment_clamp_max": float(segment_clamp_max),
        "scale_clamp_min": float(scale_clamp_min),
        "scale_clamp_max": float(scale_clamp_max),
        "width_budget_ratio": float(width_budget_ratio),
        "height_budget_ratio": float(height_budget_ratio),
        "width_budget_triggered": bool(width_budget_triggered),
        "height_budget_triggered": bool(height_budget_triggered),
        "replacement_region_budget_bbox": [float(v) for v in target_bbox.tolist()],
        "target_center_x": float(target_center_x),
        "shift_y": float(shift_y),
    }


def _draw_bbox(canvas, bbox, color, thickness=2):
    if bbox is None:
        return
    x1, y1, x2, y2 = np.rint(np.asarray(bbox)).astype(np.int32)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)


def _draw_structure_overlay(canvas, structure, color, label=None):
    if structure is None:
        return
    bbox = structure.get("bbox")
    _draw_bbox(canvas, bbox, color, 2)
    center_x = int(round(float(structure.get("center_x", 0.0))))
    for key in ("top_y", "shoulder_y", "hip_y", "foot_y"):
        if key not in structure:
            continue
        y = int(round(float(structure[key])))
        cv2.line(canvas, (max(0, center_x - 28), y), (min(canvas.shape[1] - 1, center_x + 28), y), color, 2)
    if label:
        top_y = int(round(float(structure.get("top_y", 18.0))))
        cv2.putText(canvas, label, (10, max(24, top_y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


def make_reference_normalization_preview(
    *,
    original_canvas,
    normalized_canvas,
    original_bbox=None,
    target_bbox=None,
    normalized_bbox=None,
    original_structure=None,
    target_structure=None,
    normalized_structure=None,
):
    preview_left = original_canvas.copy()
    preview_right = normalized_canvas.copy()
    if original_bbox is not None:
        _draw_bbox(preview_left, original_bbox, (255, 196, 0), 3)
    if target_bbox is not None:
        _draw_bbox(preview_left, target_bbox, (0, 255, 255), 2)
        _draw_bbox(preview_right, target_bbox, (0, 255, 255), 2)
    if normalized_bbox is not None:
        _draw_bbox(preview_right, normalized_bbox, (0, 255, 0), 3)
    if original_structure is not None:
        _draw_structure_overlay(preview_left, original_structure, (255, 96, 0))
    if target_structure is not None:
        _draw_structure_overlay(preview_left, target_structure, (0, 255, 255))
        _draw_structure_overlay(preview_right, target_structure, (0, 255, 255))
    if normalized_structure is not None:
        _draw_structure_overlay(preview_right, normalized_structure, (0, 255, 0))

    combined = np.concatenate([preview_left, preview_right], axis=1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, "original / target", (20, 36), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        combined,
        "normalized structure-aware",
        (original_canvas.shape[1] + 20, 36),
        font,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return combined


def write_reference_image(path, image_rgb):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), cv2.cvtColor(np.asarray(image_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError(f"Failed to write reference image: {path}")
