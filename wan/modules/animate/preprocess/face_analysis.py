import json
from pathlib import Path

import cv2
import numpy as np


def _clamp_bbox_xyxy(bbox, image_shape):
    height, width = image_shape
    x1, x2, y1, y2 = [float(v) for v in bbox]
    x1 = max(0.0, min(x1, width - 2.0))
    x2 = max(x1 + 2.0, min(x2, width * 1.0))
    y1 = max(0.0, min(y1, height - 2.0))
    y2 = max(y1 + 2.0, min(y2, height * 1.0))
    return np.array([x1, x2, y1, y2], dtype=np.float32)


def _bbox_to_cxcywh(bbox):
    x1, x2, y1, y2 = [float(v) for v in bbox]
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1], dtype=np.float32)


def _cxcywh_to_bbox(cxcywh, image_shape):
    cx, cy, width, height = [float(v) for v in cxcywh]
    return _clamp_bbox_xyxy(
        [cx - width / 2.0, cx + width / 2.0, cy - height / 2.0, cy + height / 2.0],
        image_shape,
    )


def _scale_bbox(bbox, source_shape, target_shape):
    if bbox is None:
        return None
    src_h, src_w = source_shape
    tgt_h, tgt_w = target_shape
    scale_x = float(tgt_w) / max(float(src_w), 1.0)
    scale_y = float(tgt_h) / max(float(src_h), 1.0)
    x1, x2, y1, y2 = [float(v) for v in bbox]
    return _clamp_bbox_xyxy([x1 * scale_x, x2 * scale_x, y1 * scale_y, y2 * scale_y], target_shape)


def _scale_bbox_with_expand(bbox, source_shape, target_shape, expand_ratio=1.0):
    scaled = _scale_bbox(bbox, source_shape, target_shape)
    if scaled is None:
        return None
    cxcywh = _bbox_to_cxcywh(scaled)
    cxcywh[2] *= float(expand_ratio)
    cxcywh[3] *= float(expand_ratio)
    return _cxcywh_to_bbox(cxcywh, target_shape)


def _extract_face_points(meta, image_shape, conf_thresh):
    height, width = image_shape
    points = np.asarray(meta.get("keypoints_face"), dtype=np.float32)
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
    points = points[valid].astype(np.float32)
    if points.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    points[:, 0] *= float(width)
    points[:, 1] *= float(height)
    return points


def _bbox_from_points(points, image_shape, expand_ratio=1.22, min_size_ratio=0.12):
    if points.size == 0:
        return None
    height, width = image_shape
    xy = points[:, :2]
    min_xy = xy.min(axis=0)
    max_xy = xy.max(axis=0)
    cx = float((min_xy[0] + max_xy[0]) * 0.5)
    cy = float((min_xy[1] + max_xy[1]) * 0.5)
    box_w = max(float(max_xy[0] - min_xy[0]), float(width) * float(min_size_ratio))
    box_h = max(float(max_xy[1] - min_xy[1]), float(height) * float(min_size_ratio))
    box_w *= float(expand_ratio)
    box_h *= float(expand_ratio) * 1.10
    return _cxcywh_to_bbox([cx, cy, box_w, box_h], image_shape)


def _blend_bboxes(primary, secondary, weight, image_shape):
    if primary is None and secondary is None:
        return None
    if primary is None:
        return _clamp_bbox_xyxy(secondary, image_shape)
    if secondary is None:
        return _clamp_bbox_xyxy(primary, image_shape)
    a = _bbox_to_cxcywh(primary)
    b = _bbox_to_cxcywh(secondary)
    fused = float(weight) * a + (1.0 - float(weight)) * b
    return _cxcywh_to_bbox(fused, image_shape)


def _estimate_head_pose(points, bbox):
    if points.shape[0] < 8:
        return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0, "confidence": 0.0}
    x1, x2, y1, y2 = [float(v) for v in bbox]
    bbox_w = max(x2 - x1, 1.0)
    bbox_h = max(y2 - y1, 1.0)
    xy = points[:, :2]
    conf = points[:, 2]
    center = xy.mean(axis=0)
    left_extent = max(center[0] - float(xy[:, 0].min()), 1e-3)
    right_extent = max(float(xy[:, 0].max()) - center[0], 1e-3)
    yaw = np.clip(38.0 * (right_extent - left_extent) / max(right_extent + left_extent, 1e-3), -45.0, 45.0)
    pitch_center = y1 + 0.48 * bbox_h
    pitch = np.clip(30.0 * (center[1] - pitch_center) / max(0.25 * bbox_h, 1e-3), -35.0, 35.0)

    upper = xy[xy[:, 1] <= y1 + 0.52 * bbox_h]
    if upper.shape[0] >= 6:
        left = upper[upper[:, 0] <= np.median(upper[:, 0])]
        right = upper[upper[:, 0] > np.median(upper[:, 0])]
        if left.shape[0] >= 2 and right.shape[0] >= 2:
            left_center = left.mean(axis=0)
            right_center = right.mean(axis=0)
            roll = np.degrees(np.arctan2(right_center[1] - left_center[1], max(right_center[0] - left_center[0], 1e-3)))
        else:
            roll = 0.0
    else:
        roll = 0.0
    roll = float(np.clip(roll, -35.0, 35.0))

    confidence = float(np.clip(conf.mean() * min(points.shape[0] / 40.0, 1.0), 0.0, 1.0))
    return {"yaw": float(yaw), "pitch": float(pitch), "roll": roll, "confidence": confidence}


def _estimate_expression(points, bbox):
    if points.shape[0] < 8:
        return {
            "mouth_open": 0.0,
            "left_eye_open": 0.0,
            "right_eye_open": 0.0,
            "smile": 0.0,
            "expression_intensity": 0.0,
            "confidence": 0.0,
        }
    x1, x2, y1, y2 = [float(v) for v in bbox]
    bbox_w = max(x2 - x1, 1.0)
    bbox_h = max(y2 - y1, 1.0)
    xy = points[:, :2]
    conf = points[:, 2]
    lower = xy[xy[:, 1] >= y1 + 0.58 * bbox_h]
    upper = xy[xy[:, 1] <= y1 + 0.50 * bbox_h]
    mouth_open = float(np.clip((lower[:, 1].max() - lower[:, 1].min()) / max(bbox_h, 1.0), 0.0, 1.0)) if lower.shape[0] >= 4 else 0.0
    mouth_width = float(np.clip((lower[:, 0].max() - lower[:, 0].min()) / max(bbox_w, 1.0), 0.0, 1.0)) if lower.shape[0] >= 4 else 0.0
    smile = float(np.clip(mouth_width - 0.25, 0.0, 1.0))
    if upper.shape[0] >= 6:
        mid_x = np.median(upper[:, 0])
        left_eye = upper[upper[:, 0] <= mid_x]
        right_eye = upper[upper[:, 0] > mid_x]
        left_eye_open = float(np.clip((left_eye[:, 1].max() - left_eye[:, 1].min()) / max(0.25 * bbox_h, 1.0), 0.0, 1.0)) if left_eye.shape[0] >= 2 else 0.0
        right_eye_open = float(np.clip((right_eye[:, 1].max() - right_eye[:, 1].min()) / max(0.25 * bbox_h, 1.0), 0.0, 1.0)) if right_eye.shape[0] >= 2 else 0.0
    else:
        left_eye_open = 0.0
        right_eye_open = 0.0
    confidence = float(np.clip(conf.mean() * min(points.shape[0] / 40.0, 1.0), 0.0, 1.0))
    expression_intensity = float(np.clip(0.55 * mouth_open + 0.25 * smile + 0.20 * abs(left_eye_open - right_eye_open), 0.0, 1.0))
    return {
        "mouth_open": mouth_open,
        "left_eye_open": left_eye_open,
        "right_eye_open": right_eye_open,
        "smile": smile,
        "expression_intensity": expression_intensity,
        "confidence": confidence,
    }


def _smooth_scalar_sequence(values, strength):
    values = np.asarray(values, dtype=np.float32).copy()
    if values.size == 0:
        return values
    out = values.copy()
    for i in range(1, len(values)):
        out[i] = float(strength) * out[i - 1] + (1.0 - float(strength)) * values[i]
    return out


def _face_hull_mask(points, bbox, image_shape):
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.float32)
    if points.shape[0] >= 6:
        xy = np.round(points[:, :2]).astype(np.int32)
        xy[:, 0] = np.clip(xy[:, 0], 0, width - 1)
        xy[:, 1] = np.clip(xy[:, 1], 0, height - 1)
        hull = cv2.convexHull(xy)
        cv2.fillConvexPoly(mask, hull, 1.0)
    else:
        x1, x2, y1, y2 = [float(v) for v in bbox]
        center = (int(round((x1 + x2) / 2.0)), int(round((y1 + y2) / 2.0)))
        axes = (max(2, int(round((x2 - x1) * 0.48))), max(2, int(round((y2 - y1) * 0.60))))
        cv2.ellipse(mask, center, axes, 0.0, 0.0, 360.0, 1.0, -1)
    return mask.astype(np.float32)


def _ellipse_mask(image_shape, center, axes):
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.float32)
    center_i = (int(round(center[0])), int(round(center[1])))
    axes_i = (max(1, int(round(axes[0]))), max(1, int(round(axes[1]))))
    cv2.ellipse(mask, center_i, axes_i, 0.0, 0.0, 360.0, 1.0, -1)
    return mask


def _build_face_parsing(points, bbox, image_shape):
    height, width = image_shape
    labels = np.zeros((height, width), dtype=np.uint8)
    face_mask = _face_hull_mask(points, bbox, image_shape)
    labels[face_mask > 0.5] = 1
    x1, x2, y1, y2 = [float(v) for v in bbox]
    bbox_w = max(x2 - x1, 1.0)
    bbox_h = max(y2 - y1, 1.0)

    upper_points = points[points[:, 1] <= y1 + 0.50 * bbox_h] if points.shape[0] > 0 else np.zeros((0, 3), dtype=np.float32)
    if upper_points.shape[0] >= 6:
        mid_x = np.median(upper_points[:, 0])
        left_eye_points = upper_points[upper_points[:, 0] <= mid_x]
        right_eye_points = upper_points[upper_points[:, 0] > mid_x]
        for eye_points, label in ((left_eye_points, 2), (right_eye_points, 2)):
            if eye_points.shape[0] >= 2:
                center = eye_points[:, :2].mean(axis=0)
                span_x = max(eye_points[:, 0].max() - eye_points[:, 0].min(), 1.0)
                span_y = max(eye_points[:, 1].max() - eye_points[:, 1].min(), 1.0)
                eye_mask = _ellipse_mask(image_shape, center, (0.75 * span_x, 1.2 * span_y))
                labels[eye_mask > 0.5] = label

    mouth_points = points[points[:, 1] >= y1 + 0.58 * bbox_h] if points.shape[0] > 0 else np.zeros((0, 3), dtype=np.float32)
    if mouth_points.shape[0] >= 4:
        center = mouth_points[:, :2].mean(axis=0)
        span_x = max(mouth_points[:, 0].max() - mouth_points[:, 0].min(), 1.0)
        span_y = max(mouth_points[:, 1].max() - mouth_points[:, 1].min(), 1.0)
        mouth_mask = _ellipse_mask(image_shape, center, (0.65 * span_x, 1.1 * span_y))
        labels[mouth_mask > 0.5] = 3

    forehead_mask = np.zeros((height, width), dtype=np.float32)
    forehead_center = ((x1 + x2) / 2.0, y1 + 0.18 * bbox_h)
    forehead_axes = (0.42 * bbox_w, 0.18 * bbox_h)
    forehead_mask += _ellipse_mask(image_shape, forehead_center, forehead_axes)
    labels[(forehead_mask > 0.5) & (labels == 1)] = 4
    return labels


def _smooth_mask_sequence(mask_seq, kernel_size=5):
    if kernel_size <= 1:
        return mask_seq.astype(np.float32)
    smoothed = []
    for frame in np.asarray(mask_seq, dtype=np.float32):
        smoothed.append(cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0))
    return np.stack(smoothed).astype(np.float32)


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _write_npz(path, **payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def _build_face_crop(frame, bbox, target_size=512):
    x1, x2, y1, y2 = [int(round(v)) for v in bbox]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = max(x1 + 2, min(x2, frame.shape[1]))
    y2 = max(y1 + 2, min(y2, frame.shape[0]))
    crop = frame[y1:y2, x1:x2]
    return cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_CUBIC if crop.shape[0] < target_size else cv2.INTER_AREA)


def run_face_analysis(
    *,
    export_frames,
    face_source_frames,
    pose_metas,
    face_bboxes,
    export_shape,
    analysis_shape,
    face_source_shape,
    hard_foreground=None,
    soft_alpha=None,
    occlusion_band=None,
    uncertainty_map=None,
    conf_thresh=0.45,
    tracking_smooth_strength=0.86,
    tracking_max_scale_change=1.10,
    tracking_max_center_shift=0.018,
    tracking_hold_frames=8,
    landmark_expand_ratio=1.22,
    difficulty_expand_ratio=1.18,
    rerun_difficulty_threshold=0.48,
    alpha_blur_kernel=7,
):
    export_h, export_w = export_shape
    source_h, source_w = face_source_shape
    base_bboxes_export = [_scale_bbox(bbox, analysis_shape, export_shape) for bbox in face_bboxes]
    landmarks_frames = []
    track_frames = []
    tracked_bboxes_export = []
    track_state = None
    hold_run = 0
    max_center_shift_px = float(max(export_h, export_w)) * float(tracking_max_center_shift)
    prev_pose = None
    prev_expr = None
    prev_velocity = np.zeros(2, dtype=np.float32)
    head_pose_frames = []
    expression_frames = []
    face_alpha_frames = []
    face_parsing_frames = []
    face_uncertainty_frames = []
    face_images = []
    source_distribution = {}

    for frame_index, (frame, source_frame, meta, base_bbox_export) in enumerate(
        zip(export_frames, face_source_frames, pose_metas, base_bboxes_export)
    ):
        points = _extract_face_points(meta, export_shape, conf_thresh=conf_thresh)
        landmark_bbox = _bbox_from_points(points, export_shape, expand_ratio=landmark_expand_ratio, min_size_ratio=0.10)
        detection_bbox = _blend_bboxes(landmark_bbox, base_bbox_export, weight=0.68, image_shape=export_shape)
        valid_ratio = float(min(points.shape[0] / 68.0, 1.0)) if points.shape[0] > 0 else 0.0
        track_source = "tracked"
        if track_state is None:
            if detection_bbox is None:
                detection_bbox = np.array([0.30 * export_w, 0.70 * export_w, 0.12 * export_h, 0.56 * export_h], dtype=np.float32)
                track_source = "default"
            track_state = detection_bbox.copy()
        else:
            prev_state = _bbox_to_cxcywh(track_state)
            predicted = prev_state.copy()
            predicted[:2] = predicted[:2] + prev_velocity
            if detection_bbox is None:
                hold_run += 1
                detection_bbox = _cxcywh_to_bbox(predicted, export_shape)
                track_source = "hold" if hold_run <= tracking_hold_frames else "predict"
            else:
                hold_run = 0
                det_cxcywh = _bbox_to_cxcywh(detection_bbox)
                det_cxcywh[2] = np.clip(det_cxcywh[2], prev_state[2] / tracking_max_scale_change, prev_state[2] * tracking_max_scale_change)
                det_cxcywh[3] = np.clip(det_cxcywh[3], prev_state[3] / tracking_max_scale_change, prev_state[3] * tracking_max_scale_change)
                center_delta = det_cxcywh[:2] - predicted[:2]
                center_distance = float(np.linalg.norm(center_delta))
                if center_distance > max_center_shift_px:
                    det_cxcywh[:2] = predicted[:2] + center_delta / max(center_distance, 1e-6) * max_center_shift_px
                    track_source = "clamped"
                track_state = _cxcywh_to_bbox(
                    tracking_smooth_strength * predicted + (1.0 - tracking_smooth_strength) * det_cxcywh,
                    export_shape,
                )
            velocity_target = _bbox_to_cxcywh(track_state)[:2] - prev_state[:2]
            prev_velocity = 0.70 * prev_velocity + 0.30 * velocity_target
        if track_source == "tracked":
            track_state = _clamp_bbox_xyxy(detection_bbox, export_shape)
        tracked_bboxes_export.append(track_state.astype(np.float32))

        pose = _estimate_head_pose(points, track_state)
        expr = _estimate_expression(points, track_state)
        if prev_pose is not None:
            yaw_jump = abs(pose["yaw"] - prev_pose["yaw"]) / 45.0
            pitch_jump = abs(pose["pitch"] - prev_pose["pitch"]) / 35.0
            roll_jump = abs(pose["roll"] - prev_pose["roll"]) / 35.0
            pose_jump = float(np.clip((yaw_jump + pitch_jump + roll_jump) / 3.0, 0.0, 1.0))
        else:
            pose_jump = 0.0
        if prev_expr is not None:
            expr_jump = abs(expr["expression_intensity"] - prev_expr["expression_intensity"])
        else:
            expr_jump = 0.0

        face_region = _face_hull_mask(points, track_state, export_shape)
        local_alpha = cv2.GaussianBlur(face_region, (alpha_blur_kernel, alpha_blur_kernel), 0)
        if soft_alpha is not None:
            local_alpha = np.maximum(local_alpha, 0.80 * soft_alpha[frame_index] * face_region)
        local_alpha = np.clip(local_alpha, 0.0, 1.0).astype(np.float32)
        parsing = _build_face_parsing(points, track_state, export_shape)

        occ_map = occlusion_band[frame_index] if occlusion_band is not None else np.zeros(export_shape, dtype=np.float32)
        global_unc = uncertainty_map[frame_index] if uncertainty_map is not None else np.zeros(export_shape, dtype=np.float32)
        face_band = np.clip(local_alpha - face_region, 0.0, 1.0)
        bbox_mask = np.zeros(export_shape, dtype=np.float32)
        x1, x2, y1, y2 = [int(round(v)) for v in track_state]
        bbox_mask[y1:y2, x1:x2] = 1.0
        difficulty_score = float(np.clip(
            0.32 * (1.0 - valid_ratio)
            + 0.22 * pose_jump
            + 0.12 * expr_jump
            + 0.18 * float((occ_map * bbox_mask).mean())
            + 0.16 * float((global_unc * np.maximum(face_band, bbox_mask * 0.25)).mean()),
            0.0,
            1.0,
        ))
        face_uncertainty = np.clip(
            np.maximum.reduce([
                0.60 * global_unc * np.maximum(face_band, bbox_mask * 0.15),
                0.45 * occ_map * np.maximum(face_region, face_band),
                0.35 * face_band,
                np.full(export_shape, 0.0, dtype=np.float32),
            ])
            + bbox_mask * (0.18 * difficulty_score)
            - face_region * (0.10 * max(valid_ratio - 0.65, 0.0)),
            0.0,
            1.0,
        ).astype(np.float32)

        rerun_applied = difficulty_score >= float(rerun_difficulty_threshold)
        crop_bbox_export = track_state.copy()
        if rerun_applied:
            crop_bbox_export = _scale_bbox_with_expand(crop_bbox_export, export_shape, export_shape, expand_ratio=difficulty_expand_ratio)
            track_source = f"{track_source}+rerun"

        crop_bbox_source = _scale_bbox(crop_bbox_export, export_shape, face_source_shape)
        face_images.append(_build_face_crop(source_frame, crop_bbox_source, target_size=512))

        points_list = []
        for idx in range(int(points.shape[0])):
            points_list.append([float(points[idx, 0]), float(points[idx, 1]), float(points[idx, 2])])

        bbox_int = [int(round(v)) for v in track_state.tolist()]
        track_frames.append({
            "frame_index": frame_index,
            "bbox": bbox_int,
            "center_x": float((bbox_int[0] + bbox_int[1]) / 2.0),
            "center_y": float((bbox_int[2] + bbox_int[3]) / 2.0),
            "width": float(bbox_int[1] - bbox_int[0]),
            "height": float(bbox_int[3] - bbox_int[2]),
            "source": track_source,
            "valid_face_points": int(points.shape[0]),
            "track_confidence": float(np.clip(0.65 * valid_ratio + 0.35 * pose["confidence"], 0.0, 1.0)),
            "difficulty_score": difficulty_score,
            "rerun_applied": bool(rerun_applied),
        })
        landmarks_frames.append({
            "frame_index": frame_index,
            "bbox": bbox_int,
            "landmarks": points_list,
            "valid_points": int(points.shape[0]),
            "mean_confidence": float(points[:, 2].mean()) if points.shape[0] > 0 else 0.0,
            "difficulty_score": difficulty_score,
        })
        head_pose_frames.append({
            "frame_index": frame_index,
            **pose,
            "difficulty_score": difficulty_score,
        })
        expression_frames.append({
            "frame_index": frame_index,
            **expr,
            "difficulty_score": difficulty_score,
        })
        face_alpha_frames.append(local_alpha.astype(np.float32))
        face_parsing_frames.append(parsing.astype(np.uint8))
        face_uncertainty_frames.append(face_uncertainty.astype(np.float32))
        prev_pose = pose
        prev_expr = expr
        source_distribution[track_source] = source_distribution.get(track_source, 0) + 1

    yaw = _smooth_scalar_sequence([frame["yaw"] for frame in head_pose_frames], 0.55)
    pitch = _smooth_scalar_sequence([frame["pitch"] for frame in head_pose_frames], 0.55)
    roll = _smooth_scalar_sequence([frame["roll"] for frame in head_pose_frames], 0.60)
    expr_intensity = _smooth_scalar_sequence([frame["expression_intensity"] for frame in expression_frames], 0.35)
    mouth_open = _smooth_scalar_sequence([frame["mouth_open"] for frame in expression_frames], 0.30)
    for idx, frame in enumerate(head_pose_frames):
        frame["yaw"] = float(yaw[idx])
        frame["pitch"] = float(pitch[idx])
        frame["roll"] = float(roll[idx])
    for idx, frame in enumerate(expression_frames):
        frame["expression_intensity"] = float(expr_intensity[idx])
        frame["mouth_open"] = float(mouth_open[idx])

    face_alpha = _smooth_mask_sequence(np.stack(face_alpha_frames).astype(np.float32), kernel_size=5)
    face_parsing = np.stack(face_parsing_frames).astype(np.uint8)
    face_uncertainty = _smooth_mask_sequence(np.stack(face_uncertainty_frames).astype(np.float32), kernel_size=5)
    face_images = np.stack(face_images).astype(np.uint8)

    center_curve = np.array([[frame["center_x"], frame["center_y"]] for frame in track_frames], dtype=np.float32)
    center_step = np.linalg.norm(np.diff(center_curve, axis=0), axis=1) if len(center_curve) > 1 else np.zeros((0,), dtype=np.float32)
    yaw_step = np.abs(np.diff(yaw)) if len(yaw) > 1 else np.zeros((0,), dtype=np.float32)
    pitch_step = np.abs(np.diff(pitch)) if len(pitch) > 1 else np.zeros((0,), dtype=np.float32)
    roll_step = np.abs(np.diff(roll)) if len(roll) > 1 else np.zeros((0,), dtype=np.float32)
    landmark_conf = np.array([frame["mean_confidence"] for frame in landmarks_frames], dtype=np.float32)
    valid_points = np.array([frame["valid_points"] for frame in landmarks_frames], dtype=np.float32)
    difficulty = np.array([frame["difficulty_score"] for frame in track_frames], dtype=np.float32)

    landmarks_json = {
        "frame_count": int(len(landmarks_frames)),
        "landmark_count": 69,
        "frames": landmarks_frames,
        "aggregate_stats": {
            "valid_points_mean": float(valid_points.mean()) if len(valid_points) else 0.0,
            "valid_points_min": float(valid_points.min()) if len(valid_points) else 0.0,
            "mean_confidence": float(landmark_conf.mean()) if len(landmark_conf) else 0.0,
        },
    }
    head_pose_json = {
        "frame_count": int(len(head_pose_frames)),
        "frames": head_pose_frames,
        "aggregate_stats": {
            "yaw_step_mean": float(yaw_step.mean()) if len(yaw_step) else 0.0,
            "pitch_step_mean": float(pitch_step.mean()) if len(pitch_step) else 0.0,
            "roll_step_mean": float(roll_step.mean()) if len(roll_step) else 0.0,
            "mean_confidence": float(np.mean([f["confidence"] for f in head_pose_frames])) if head_pose_frames else 0.0,
        },
    }
    expression_json = {
        "frame_count": int(len(expression_frames)),
        "frames": expression_frames,
        "aggregate_stats": {
            "expression_intensity_mean": float(expr_intensity.mean()) if len(expr_intensity) else 0.0,
            "mouth_open_mean": float(mouth_open.mean()) if len(mouth_open) else 0.0,
            "mean_confidence": float(np.mean([f["confidence"] for f in expression_frames])) if expression_frames else 0.0,
        },
    }
    bbox_curve = {
        "frames": track_frames,
        "aggregate_stats": {
            "center_jitter_mean": float(center_step.mean()) if len(center_step) else 0.0,
            "difficulty_mean": float(difficulty.mean()) if len(difficulty) else 0.0,
            "rerun_ratio": float(np.mean([1.0 if frame["rerun_applied"] else 0.0 for frame in track_frames])) if track_frames else 0.0,
            "source_counts": source_distribution,
        },
    }
    stats = {
        "center_jitter_mean": float(center_step.mean()) if len(center_step) else 0.0,
        "valid_landmark_points_mean": float(valid_points.mean()) if len(valid_points) else 0.0,
        "landmark_confidence_mean": float(landmark_conf.mean()) if len(landmark_conf) else 0.0,
        "head_pose_jitter_mean": float((yaw_step.mean() + pitch_step.mean() + roll_step.mean()) / 3.0) if len(yaw_step) else 0.0,
        "expression_intensity_mean": float(expr_intensity.mean()) if len(expr_intensity) else 0.0,
        "alpha_mean": float(face_alpha.mean()),
        "uncertainty_mean": float(face_uncertainty.mean()),
        "rerun_ratio": bbox_curve["aggregate_stats"]["rerun_ratio"],
        "source_counts": source_distribution,
    }
    return {
        "face_images": face_images,
        "tracked_bboxes_export": [frame["bbox"] for frame in track_frames],
        "bbox_curve": bbox_curve,
        "landmarks_json": landmarks_json,
        "head_pose_json": head_pose_json,
        "expression_json": expression_json,
        "face_alpha": face_alpha.astype(np.float32),
        "face_parsing": face_parsing.astype(np.uint8),
        "face_uncertainty": face_uncertainty.astype(np.float32),
        "stats": stats,
    }


def make_face_landmark_overlay(frames, bbox_curve, landmarks_json, head_pose_json=None):
    frames = np.asarray(frames, dtype=np.uint8)
    overlays = []
    pose_frames = {frame["frame_index"]: frame for frame in (head_pose_json or {}).get("frames", [])}
    landmark_frames = {frame["frame_index"]: frame for frame in landmarks_json.get("frames", [])}
    for frame_entry in bbox_curve.get("frames", []):
        frame_index = int(frame_entry["frame_index"])
        image = frames[frame_index].copy()
        x1, x2, y1, y2 = [int(v) for v in frame_entry["bbox"]]
        color = (0, 255, 0) if not frame_entry.get("rerun_applied") else (255, 180, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        landmarks = landmark_frames.get(frame_index, {}).get("landmarks", [])
        for point in landmarks:
            px, py, conf = point
            cv2.circle(image, (int(round(px)), int(round(py))), 1, (0, 255, 255) if conf >= 0.5 else (255, 0, 0), -1)
        pose = pose_frames.get(frame_index)
        text = f"pts={frame_entry.get('valid_face_points', 0)} diff={frame_entry.get('difficulty_score', 0.0):.2f}"
        if pose is not None:
            text += f" yaw={pose.get('yaw', 0.0):.1f}"
        cv2.putText(image, text, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        overlays.append(image)
    return np.stack(overlays).astype(np.uint8)


def make_face_parsing_preview(face_parsing):
    face_parsing = np.asarray(face_parsing, dtype=np.uint8)
    palette = np.array(
        [
            [0, 0, 0],
            [240, 182, 160],
            [80, 220, 255],
            [255, 120, 120],
            [180, 120, 255],
        ],
        dtype=np.uint8,
    )
    face_parsing = np.clip(face_parsing, 0, len(palette) - 1)
    return palette[face_parsing]


def write_face_analysis_artifacts(output_root, analysis, fps, write_json_fn=None):
    output_root = Path(output_root)
    if write_json_fn is None:
        write_json_fn = _write_json
    write_json_fn(output_root / "src_face_landmarks.json", analysis["landmarks_json"])
    write_json_fn(output_root / "src_face_pose.json", analysis["head_pose_json"])
    write_json_fn(output_root / "src_face_expression.json", analysis["expression_json"])
    _write_npz(output_root / "src_face_alpha.npz", mask=analysis["face_alpha"].astype(np.float32), fps=np.array([fps], dtype=np.float32))
    _write_npz(output_root / "src_face_uncertainty.npz", mask=analysis["face_uncertainty"].astype(np.float32), fps=np.array([fps], dtype=np.float32))
    _write_npz(output_root / "src_face_parsing.npz", labels=analysis["face_parsing"].astype(np.uint8), fps=np.array([fps], dtype=np.float32))
    return {
        "face_landmarks": {
            "path": "src_face_landmarks.json",
            "type": "json",
            "format": "json",
            "frame_count": int(analysis["landmarks_json"]["frame_count"]),
            "landmark_count": int(analysis["landmarks_json"].get("landmark_count", 0)),
        },
        "face_pose": {
            "path": "src_face_pose.json",
            "type": "json",
            "format": "json",
            "frame_count": int(analysis["head_pose_json"]["frame_count"]),
        },
        "face_expression": {
            "path": "src_face_expression.json",
            "type": "json",
            "format": "json",
            "frame_count": int(analysis["expression_json"]["frame_count"]),
        },
        "face_alpha": {
            "path": "src_face_alpha.npz",
            "type": "video",
            "format": "npz",
            "frame_count": int(analysis["face_alpha"].shape[0]),
            "height": int(analysis["face_alpha"].shape[1]),
            "width": int(analysis["face_alpha"].shape[2]),
            "channels": 1,
            "dtype": str(analysis["face_alpha"].dtype),
            "shape": list(analysis["face_alpha"].shape),
            "fps": float(fps),
            "value_range": [0.0, 1.0],
            "mask_semantics": "face_alpha",
        },
        "face_uncertainty": {
            "path": "src_face_uncertainty.npz",
            "type": "video",
            "format": "npz",
            "frame_count": int(analysis["face_uncertainty"].shape[0]),
            "height": int(analysis["face_uncertainty"].shape[1]),
            "width": int(analysis["face_uncertainty"].shape[2]),
            "channels": 1,
            "dtype": str(analysis["face_uncertainty"].dtype),
            "shape": list(analysis["face_uncertainty"].shape),
            "fps": float(fps),
            "value_range": [0.0, 1.0],
            "mask_semantics": "face_uncertainty",
        },
        "face_parsing": {
            "path": "src_face_parsing.npz",
            "type": "video",
            "format": "npz",
            "frame_count": int(analysis["face_parsing"].shape[0]),
            "height": int(analysis["face_parsing"].shape[1]),
            "width": int(analysis["face_parsing"].shape[2]),
            "channels": 1,
            "dtype": str(analysis["face_parsing"].dtype),
            "shape": list(analysis["face_parsing"].shape),
            "fps": float(fps),
            "value_range": [0, 4],
            "label_semantics": "face_parsing_v1",
        },
    }
