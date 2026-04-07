import json
from pathlib import Path

import cv2
import numpy as np

from human_visualization import draw_aapose_by_meta_new
from pose2d_utils import AAPoseMeta


def _clamp01(points):
    return np.clip(points, 0.0, 1.0)


def _is_valid_points(points, conf_thresh):
    return (
        np.isfinite(points[:, 0])
        & np.isfinite(points[:, 1])
        & np.isfinite(points[:, 2])
        & (points[:, 2] >= conf_thresh)
        & (points[:, 0] >= 0.0)
        & (points[:, 0] <= 1.0)
        & (points[:, 1] >= 0.0)
        & (points[:, 1] <= 1.0)
    )


def _smooth_track(points, conf_thresh, smooth_strength, max_velocity, interp_max_gap):
    points = points.copy().astype(np.float32)
    repaired_mask = np.zeros(points.shape[0], dtype=bool)
    valid = _is_valid_points(points, conf_thresh)

    valid_indices = np.where(valid)[0]
    if len(valid_indices) > 0:
        for left, right in zip(valid_indices[:-1], valid_indices[1:]):
            gap = right - left - 1
            if gap <= 0 or gap > interp_max_gap:
                continue
            for offset, index in enumerate(range(left + 1, right), start=1):
                ratio = offset / (gap + 1)
                points[index, :2] = (1.0 - ratio) * points[left, :2] + ratio * points[right, :2]
                points[index, 2] = max(points[index, 2], min(points[left, 2], points[right, 2]) * 0.95)
                repaired_mask[index] = True
                valid[index] = True

    smoothed = points.copy()
    stats = {
        "interpolated_points": int(repaired_mask.sum()),
        "held_points": 0,
        "clamped_points": 0,
        "dropped_points": 0,
    }

    previous = None
    previous_conf = 0.0
    missing_run = 0
    for index in range(points.shape[0]):
        if valid[index]:
            current_xy = points[index, :2]
            if previous is None:
                smoothed_xy = current_xy
            else:
                smoothed_xy = smooth_strength * previous + (1.0 - smooth_strength) * current_xy
                delta = smoothed_xy - previous
                distance = float(np.linalg.norm(delta))
                if distance > max_velocity > 0:
                    smoothed_xy = previous + delta / max(distance, 1e-6) * max_velocity
                    stats["clamped_points"] += 1
            smoothed[index, :2] = _clamp01(smoothed_xy)
            smoothed[index, 2] = max(points[index, 2], conf_thresh)
            previous = smoothed[index, :2]
            previous_conf = smoothed[index, 2]
            missing_run = 0
            continue

        missing_run += 1
        if previous is not None and missing_run <= interp_max_gap:
            smoothed[index, :2] = previous
            smoothed[index, 2] = max(previous_conf * 0.95, conf_thresh)
            repaired_mask[index] = True
            stats["held_points"] += 1
            continue

        smoothed[index, 2] = 0.0
        stats["dropped_points"] += 1

    return smoothed, repaired_mask, stats


def stabilize_pose_metas(
    metas,
    *,
    method="ema",
    pose_conf_thresh_body=0.5,
    pose_conf_thresh_hand=0.35,
    pose_conf_thresh_face=0.45,
    pose_smooth_strength_body=0.65,
    pose_smooth_strength_hand=0.35,
    pose_smooth_strength_face=0.7,
    pose_max_velocity_body=0.05,
    pose_max_velocity_hand=0.08,
    pose_max_velocity_face=0.04,
    pose_interp_max_gap=3,
):
    if method != "ema":
        raise ValueError(f"Unsupported pose_smooth_method: {method}")

    stabilized_metas = []
    for meta in metas:
        stabilized_metas.append({
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in meta.items()
        })

    key_configs = [
        ("keypoints_body", pose_conf_thresh_body, pose_smooth_strength_body, pose_max_velocity_body, "body"),
        ("keypoints_left_hand", pose_conf_thresh_hand, pose_smooth_strength_hand, pose_max_velocity_hand, "left_hand"),
        ("keypoints_right_hand", pose_conf_thresh_hand, pose_smooth_strength_hand, pose_max_velocity_hand, "right_hand"),
        ("keypoints_face", pose_conf_thresh_face, pose_smooth_strength_face, pose_max_velocity_face, "face"),
    ]

    pose_conf_curve = []
    repair_masks = {}
    aggregate_stats = {}

    for key, conf_thresh, smooth_strength, max_velocity, stats_key in key_configs:
        sequence = np.stack([meta[key] for meta in stabilized_metas], axis=0).astype(np.float32)
        smoothed = np.zeros_like(sequence)
        repaired = np.zeros(sequence.shape[:2], dtype=bool)
        category_stats = {
            "interpolated_points": 0,
            "held_points": 0,
            "clamped_points": 0,
            "dropped_points": 0,
        }
        for kp_index in range(sequence.shape[1]):
            smoothed_track, repaired_track, track_stats = _smooth_track(
                sequence[:, kp_index],
                conf_thresh=conf_thresh,
                smooth_strength=smooth_strength,
                max_velocity=max_velocity,
                interp_max_gap=pose_interp_max_gap,
            )
            smoothed[:, kp_index] = smoothed_track
            repaired[:, kp_index] = repaired_track
            for stat_name, stat_value in track_stats.items():
                category_stats[stat_name] += int(stat_value)

        for frame_index, meta in enumerate(stabilized_metas):
            meta[key] = smoothed[frame_index]

        repair_masks[stats_key] = repaired
        aggregate_stats[stats_key] = category_stats

    for frame_index, (raw_meta, stabilized_meta) in enumerate(zip(metas, stabilized_metas)):
        pose_conf_curve.append({
            "frame_index": frame_index,
            "raw_body_mean_conf": float(np.mean(raw_meta["keypoints_body"][:, 2])),
            "smoothed_body_mean_conf": float(np.mean(stabilized_meta["keypoints_body"][:, 2])),
            "raw_hand_mean_conf": float(
                np.mean(
                    np.concatenate(
                        [
                            raw_meta["keypoints_left_hand"][:, 2],
                            raw_meta["keypoints_right_hand"][:, 2],
                        ]
                    )
                )
            ),
            "smoothed_hand_mean_conf": float(
                np.mean(
                    np.concatenate(
                        [
                            stabilized_meta["keypoints_left_hand"][:, 2],
                            stabilized_meta["keypoints_right_hand"][:, 2],
                        ]
                    )
                )
            ),
            "raw_face_mean_conf": float(np.mean(raw_meta["keypoints_face"][:, 2])),
            "smoothed_face_mean_conf": float(np.mean(stabilized_meta["keypoints_face"][:, 2])),
            "repaired_body_points": int(repair_masks["body"][frame_index].sum()),
            "repaired_hand_points": int(
                repair_masks["left_hand"][frame_index].sum() + repair_masks["right_hand"][frame_index].sum()
            ),
            "repaired_face_points": int(repair_masks["face"][frame_index].sum()),
        })

    return stabilized_metas, {
        "frames": pose_conf_curve,
        "aggregate_stats": aggregate_stats,
    }


def _bbox_from_face_points(face_points, image_shape, scale):
    image_h, image_w = image_shape
    min_x, min_y = np.min(face_points, axis=0)
    max_x, max_y = np.max(face_points, axis=0)
    initial_width = max(max_x - min_x, 1.0)
    initial_height = max(max_y - min_y, 1.0)
    initial_area = max(initial_width * initial_height, 1.0)
    expanded_area = initial_area * scale
    new_width = np.sqrt(expanded_area * (initial_width / initial_height))
    new_height = np.sqrt(expanded_area * (initial_height / initial_width))
    delta_width = (new_width - initial_width) / 2.0
    delta_height = (new_height - initial_height) / 4.0
    expanded_min_x = max(min_x - delta_width, 0.0)
    expanded_max_x = min(max_x + delta_width, float(image_w))
    expanded_min_y = max(min_y - 3.0 * delta_height, 0.0)
    expanded_max_y = min(max_y + delta_height, float(image_h))
    return np.array([expanded_min_x, expanded_max_x, expanded_min_y, expanded_max_y], dtype=np.float32)


def _bbox_to_cxcywh(bbox):
    x1, x2, y1, y2 = bbox
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1], dtype=np.float32)


def _cxcywh_to_bbox(cxcywh, image_shape):
    image_h, image_w = image_shape
    cx, cy, width, height = cxcywh
    x1 = np.clip(cx - width / 2.0, 0.0, image_w)
    x2 = np.clip(cx + width / 2.0, 0.0, image_w)
    y1 = np.clip(cy - height, 0.0, image_h)
    y2 = np.clip(cy + height / 3.0, 0.0, image_h)
    if x2 <= x1:
        x2 = min(float(image_w), x1 + 1.0)
    if y2 <= y1:
        y2 = min(float(image_h), y1 + 1.0)
    return np.array([x1, x2, y1, y2], dtype=np.float32)


def stabilize_face_bboxes(
    metas,
    image_shape,
    *,
    scale=1.3,
    conf_thresh=0.45,
    min_valid_points=15,
    smooth_method="ema",
    smooth_strength=0.7,
    max_scale_change=1.15,
    max_center_shift=0.04,
    hold_frames=6,
):
    if smooth_method != "ema":
        raise ValueError(f"Unsupported face_bbox_smooth_method: {smooth_method}")

    image_h, image_w = image_shape
    default_size = float(min(image_h, image_w) * 0.35)
    default_bbox = np.array(
        [
            image_w / 2.0 - default_size / 2.0,
            image_w / 2.0 + default_size / 2.0,
            image_h / 2.0 - default_size * 0.75,
            image_h / 2.0 + default_size * 0.25,
        ],
        dtype=np.float32,
    )

    max_center_shift_px = float(max(image_h, image_w)) * max_center_shift
    smoothed_bbox = None
    hold_run = 0
    bbox_curves = []
    bboxes = []

    for frame_index, meta in enumerate(metas):
        keypoints_face = meta["keypoints_face"].copy()
        valid_mask = _is_valid_points(keypoints_face[1:], conf_thresh)
        valid_points = keypoints_face[1:][valid_mask][:, :2] * np.array([image_w, image_h], dtype=np.float32)

        source = "strict"
        raw_bbox = None
        if valid_points.shape[0] >= min_valid_points:
            raw_bbox = _bbox_from_face_points(valid_points, (image_h, image_w), scale=scale)
            hold_run = 0
        else:
            relaxed_valid_mask = np.isfinite(keypoints_face[1:, 0]) & np.isfinite(keypoints_face[1:, 1]) & (keypoints_face[1:, 2] > 0.0)
            relaxed_points = keypoints_face[1:][relaxed_valid_mask][:, :2] * np.array([image_w, image_h], dtype=np.float32)
            if relaxed_points.shape[0] >= max(5, min_valid_points // 2):
                raw_bbox = _bbox_from_face_points(relaxed_points, (image_h, image_w), scale=scale)
                source = "relaxed"
                hold_run = 0
            elif smoothed_bbox is not None and hold_run < hold_frames:
                raw_bbox = smoothed_bbox.copy()
                source = "hold"
                hold_run += 1
            else:
                raw_bbox = default_bbox.copy()
                source = "default"

        raw_cxcywh = _bbox_to_cxcywh(raw_bbox)
        if smoothed_bbox is None or source in {"hold", "default"}:
            smoothed_bbox = raw_bbox.copy()
        else:
            prev_cxcywh = _bbox_to_cxcywh(smoothed_bbox)
            raw_cxcywh[2] = np.clip(raw_cxcywh[2], prev_cxcywh[2] / max_scale_change, prev_cxcywh[2] * max_scale_change)
            raw_cxcywh[3] = np.clip(raw_cxcywh[3], prev_cxcywh[3] / max_scale_change, prev_cxcywh[3] * max_scale_change)

            center_delta = raw_cxcywh[:2] - prev_cxcywh[:2]
            center_distance = float(np.linalg.norm(center_delta))
            if center_distance > max_center_shift_px:
                raw_cxcywh[:2] = prev_cxcywh[:2] + center_delta / max(center_distance, 1e-6) * max_center_shift_px

            smoothed_cxcywh = smooth_strength * prev_cxcywh + (1.0 - smooth_strength) * raw_cxcywh
            smoothed_bbox = _cxcywh_to_bbox(smoothed_cxcywh, (image_h, image_w))

        x1, x2, y1, y2 = smoothed_bbox.astype(np.int32).tolist()
        x1 = max(0, min(x1, image_w - 1))
        x2 = max(x1 + 1, min(x2, image_w))
        y1 = max(0, min(y1, image_h - 1))
        y2 = max(y1 + 1, min(y2, image_h))
        clipped_bbox = [x1, x2, y1, y2]
        smoothed_bbox = np.array(clipped_bbox, dtype=np.float32)
        bboxes.append(clipped_bbox)
        bbox_curves.append({
            "frame_index": frame_index,
            "valid_face_points": int(valid_points.shape[0]),
            "source": source,
            "bbox": clipped_bbox,
            "center_x": float((x1 + x2) / 2.0),
            "center_y": float((y1 + y2) / 2.0),
            "width": float(x2 - x1),
            "height": float(y2 - y1),
        })

    return bboxes, {"frames": bbox_curves}


def make_face_bbox_overlay(frames, bboxes, bbox_curve):
    overlay_frames = []
    for frame, bbox, curve in zip(frames, bboxes, bbox_curve["frames"]):
        image = frame.copy()
        x1, x2, y1, y2 = bbox
        color = (0, 255, 0) if curve["source"] == "strict" else (255, 180, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image,
            f"{curve['source']} pts={curve['valid_face_points']}",
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
        overlay_frames.append(image)
    return np.stack(overlay_frames).astype(np.uint8)


def make_pose_overlay(frames, metas):
    overlay_frames = []
    for frame, meta in zip(frames, metas):
        canvas = frame.copy()
        pose_meta = AAPoseMeta.from_humanapi_meta(meta)
        overlay_frames.append(draw_aapose_by_meta_new(canvas, pose_meta))
    return np.stack(overlay_frames).astype(np.uint8)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
