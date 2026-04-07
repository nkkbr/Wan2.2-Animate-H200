from pathlib import Path

import cv2
import numpy as np


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


def scale_bbox_between_shapes(bbox, from_shape, to_shape):
    if bbox is None:
        return None
    from_h, from_w = from_shape
    to_h, to_w = to_shape
    scale_x = float(to_w) / max(float(from_w), 1.0)
    scale_y = float(to_h) / max(float(from_h), 1.0)
    x1, y1, x2, y2 = np.asarray(bbox, dtype=np.float32)
    return np.asarray([x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y], dtype=np.float32)


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


def _resize_full_image(image, scale):
    height, width = image.shape[:2]
    new_width = max(1, int(round(width * float(scale))))
    new_height = max(1, int(round(height * float(scale))))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
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


def make_reference_normalization_preview(
    *,
    original_canvas,
    normalized_canvas,
    original_bbox=None,
    target_bbox=None,
    normalized_bbox=None,
):
    preview_left = original_canvas.copy()
    preview_right = normalized_canvas.copy()
    if original_bbox is not None:
        x1, y1, x2, y2 = np.rint(np.asarray(original_bbox)).astype(np.int32)
        cv2.rectangle(preview_left, (x1, y1), (x2, y2), (255, 196, 0), 3)
    if target_bbox is not None:
        x1, y1, x2, y2 = np.rint(np.asarray(target_bbox)).astype(np.int32)
        cv2.rectangle(preview_left, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.rectangle(preview_right, (x1, y1), (x2, y2), (0, 255, 255), 2)
    if normalized_bbox is not None:
        x1, y1, x2, y2 = np.rint(np.asarray(normalized_bbox)).astype(np.int32)
        cv2.rectangle(preview_right, (x1, y1), (x2, y2), (0, 255, 0), 3)

    combined = np.concatenate([preview_left, preview_right], axis=1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, "original letterbox", (20, 36), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(combined, "normalized reference", (original_canvas.shape[1] + 20, 36), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return combined


def write_reference_image(path, image_rgb):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), cv2.cvtColor(np.asarray(image_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError(f"Failed to write reference image: {path}")

