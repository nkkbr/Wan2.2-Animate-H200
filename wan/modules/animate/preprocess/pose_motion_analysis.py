import json
from pathlib import Path

import cv2
import numpy as np


BODY_NOSE = 0
BODY_NECK = 1
BODY_RIGHT_SHOULDER = 2
BODY_RIGHT_ELBOW = 3
BODY_RIGHT_WRIST = 4
BODY_LEFT_SHOULDER = 5
BODY_LEFT_ELBOW = 6
BODY_LEFT_WRIST = 7
BODY_RIGHT_HIP = 8
BODY_RIGHT_KNEE = 9
BODY_RIGHT_ANKLE = 10
BODY_LEFT_HIP = 11
BODY_LEFT_KNEE = 12
BODY_LEFT_ANKLE = 13

CORE_BODY_INDICES = [BODY_NECK, BODY_RIGHT_SHOULDER, BODY_LEFT_SHOULDER, BODY_RIGHT_HIP, BODY_LEFT_HIP]
LEFT_ARM_INDICES = [BODY_LEFT_SHOULDER, BODY_LEFT_ELBOW, BODY_LEFT_WRIST]
RIGHT_ARM_INDICES = [BODY_RIGHT_SHOULDER, BODY_RIGHT_ELBOW, BODY_RIGHT_WRIST]
LEFT_LEG_INDICES = [BODY_LEFT_HIP, BODY_LEFT_KNEE, BODY_LEFT_ANKLE]
RIGHT_LEG_INDICES = [BODY_RIGHT_HIP, BODY_RIGHT_KNEE, BODY_RIGHT_ANKLE]

POSE_STATUS_VISIBLE = 0
POSE_STATUS_LOW_CONFIDENCE = 1
POSE_STATUS_OCCLUDED = 2
POSE_STATUS_INTERPOLATED = 3

POSE_STATUS_LABELS = {
    POSE_STATUS_VISIBLE: "visible",
    POSE_STATUS_LOW_CONFIDENCE: "low_confidence",
    POSE_STATUS_OCCLUDED: "occluded",
    POSE_STATUS_INTERPOLATED: "interpolated",
}


def _copy_meta(meta: dict) -> dict:
    return {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in meta.items()
    }


def _as_track(metas, key):
    return np.stack([np.asarray(meta[key], dtype=np.float32) for meta in metas], axis=0).astype(np.float32)


def _point_valid(track, conf_thresh):
    return (
        np.isfinite(track[:, :, 0])
        & np.isfinite(track[:, :, 1])
        & np.isfinite(track[:, :, 2])
        & (track[:, :, 2] >= float(conf_thresh))
        & (track[:, :, 0] >= 0.0)
        & (track[:, :, 0] <= 1.0)
        & (track[:, :, 1] >= 0.0)
        & (track[:, :, 1] <= 1.0)
    )


def _smooth_track_bidirectional(
    track,
    *,
    conf_thresh,
    smooth_strength,
    max_velocity,
    interp_max_gap,
):
    track = np.asarray(track, dtype=np.float32).copy()
    smoothed = track.copy()
    interpolated = np.zeros(track.shape[:2], dtype=bool)
    dropped = np.zeros(track.shape[:2], dtype=bool)
    valid = _point_valid(track, conf_thresh)

    for kp_index in range(track.shape[1]):
        points = track[:, kp_index].copy()
        point_valid = valid[:, kp_index].copy()
        valid_indices = np.where(point_valid)[0]
        if len(valid_indices) > 1:
            for left, right in zip(valid_indices[:-1], valid_indices[1:]):
                gap = int(right - left - 1)
                if gap <= 0 or gap > interp_max_gap:
                    continue
                for offset, frame_index in enumerate(range(left + 1, right), start=1):
                    ratio = offset / (gap + 1)
                    points[frame_index, :2] = (1.0 - ratio) * points[left, :2] + ratio * points[right, :2]
                    points[frame_index, 2] = max(points[frame_index, 2], min(points[left, 2], points[right, 2]) * 0.95)
                    point_valid[frame_index] = True
                    interpolated[frame_index, kp_index] = True

        def _pass(sequence):
            out = sequence.copy()
            previous_xy = None
            previous_conf = 0.0
            missing = 0
            for frame_index in range(sequence.shape[0]):
                if point_valid[frame_index]:
                    current_xy = sequence[frame_index, :2]
                    if previous_xy is None:
                        smooth_xy = current_xy
                    else:
                        smooth_xy = float(smooth_strength) * previous_xy + (1.0 - float(smooth_strength)) * current_xy
                        delta = smooth_xy - previous_xy
                        distance = float(np.linalg.norm(delta))
                        if max_velocity > 0 and distance > max_velocity:
                            smooth_xy = previous_xy + delta / max(distance, 1e-6) * max_velocity
                    out[frame_index, :2] = np.clip(smooth_xy, 0.0, 1.0)
                    out[frame_index, 2] = max(sequence[frame_index, 2], conf_thresh)
                    previous_xy = out[frame_index, :2]
                    previous_conf = out[frame_index, 2]
                    missing = 0
                else:
                    missing += 1
                    if previous_xy is not None and missing <= interp_max_gap:
                        out[frame_index, :2] = previous_xy
                        out[frame_index, 2] = max(previous_conf * 0.92, conf_thresh * 0.92)
                        interpolated[frame_index, kp_index] = True
                    else:
                        out[frame_index, 2] = 0.0
                        dropped[frame_index, kp_index] = True
            return out

        forward = _pass(points)
        backward = _pass(points[::-1])[::-1]
        merged = points.copy()
        merged[:, :2] = 0.5 * forward[:, :2] + 0.5 * backward[:, :2]
        merged[:, 2] = np.maximum(points[:, 2], np.maximum(forward[:, 2], backward[:, 2]))
        smoothed[:, kp_index] = merged

    return smoothed.astype(np.float32), interpolated, dropped


def _group_bbox_from_points(points_xy, expand_ratio, min_size_ratio):
    if points_xy.size == 0:
        return None
    x1 = float(points_xy[:, 0].min())
    x2 = float(points_xy[:, 0].max())
    y1 = float(points_xy[:, 1].min())
    y2 = float(points_xy[:, 1].max())
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
    return np.asarray([x1, y1, x2, y2], dtype=np.float32)


def _smooth_roi_sequence(boxes, smooth_strength=0.82):
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    arr = np.asarray(boxes, dtype=np.float32).copy()
    out = arr.copy()
    for i in range(1, len(arr)):
        out[i] = float(smooth_strength) * out[i - 1] + (1.0 - float(smooth_strength)) * arr[i]
    for i in range(len(arr) - 2, -1, -1):
        out[i] = 0.5 * out[i] + 0.5 * (float(smooth_strength) * out[i + 1] + (1.0 - float(smooth_strength)) * arr[i])
    out[:, 0::2] = np.clip(out[:, 0::2], 0.0, 1.0)
    out[:, 1::2] = np.clip(out[:, 1::2], 0.0, 1.0)
    return out.astype(np.float32)


def _local_refine_group(track, indices, *, conf_thresh, smooth_strength, expand_ratio, min_size_ratio):
    refined = track.copy()
    valid = _point_valid(track, conf_thresh)
    roi_boxes = []
    covered = 0
    for frame_index in range(track.shape[0]):
        points = track[frame_index, indices]
        points_valid = valid[frame_index, indices]
        if points_valid.any():
            roi = _group_bbox_from_points(points[points_valid, :2], expand_ratio=expand_ratio, min_size_ratio=min_size_ratio)
            covered += 1
        else:
            roi = np.asarray([0.25, 0.25, 0.75, 0.75], dtype=np.float32)
        roi_boxes.append(roi)
    roi_boxes = _smooth_roi_sequence(roi_boxes, smooth_strength=max(0.6, smooth_strength))
    local_track = np.zeros((track.shape[0], len(indices), 3), dtype=np.float32)
    for frame_index, roi in enumerate(roi_boxes):
        x1, y1, x2, y2 = roi
        width = max(float(x2 - x1), 1e-6)
        height = max(float(y2 - y1), 1e-6)
        local_track[frame_index, :, 0] = np.clip((track[frame_index, indices, 0] - x1) / width, 0.0, 1.0)
        local_track[frame_index, :, 1] = np.clip((track[frame_index, indices, 1] - y1) / height, 0.0, 1.0)
        local_track[frame_index, :, 2] = track[frame_index, indices, 2]
    local_smoothed, _, _ = _smooth_track_bidirectional(
        local_track,
        conf_thresh=conf_thresh,
        smooth_strength=smooth_strength,
        max_velocity=0.12,
        interp_max_gap=4,
    )
    for frame_index, roi in enumerate(roi_boxes):
        x1, y1, x2, y2 = roi
        width = max(float(x2 - x1), 1e-6)
        height = max(float(y2 - y1), 1e-6)
        refined[frame_index, indices, 0] = np.clip(x1 + local_smoothed[frame_index, :, 0] * width, 0.0, 1.0)
        refined[frame_index, indices, 1] = np.clip(y1 + local_smoothed[frame_index, :, 1] * height, 0.0, 1.0)
        refined[frame_index, indices, 2] = np.maximum(refined[frame_index, indices, 2], local_smoothed[frame_index, :, 2])
    stats = {
        "frame_count": int(track.shape[0]),
        "coverage_ratio": float(covered / max(track.shape[0], 1)),
        "mean_width_ratio": float(np.mean(roi_boxes[:, 2] - roi_boxes[:, 0])) if len(roi_boxes) else 0.0,
        "mean_height_ratio": float(np.mean(roi_boxes[:, 3] - roi_boxes[:, 1])) if len(roi_boxes) else 0.0,
    }
    return refined.astype(np.float32), roi_boxes.astype(np.float32), stats


def _sample_mask(mask_frames, frame_index, point_xy):
    if mask_frames is None:
        return 0.0
    height, width = mask_frames.shape[1:3]
    x = int(np.clip(round(float(point_xy[0]) * (width - 1)), 0, width - 1))
    y = int(np.clip(round(float(point_xy[1]) * (height - 1)), 0, height - 1))
    return float(mask_frames[frame_index, y, x])


def _compute_velocity(track):
    velocity = np.zeros_like(track[:, :, :2], dtype=np.float32)
    if len(track) > 1:
        velocity[1:] = track[1:, :, :2] - track[:-1, :, :2]
    acceleration = np.zeros_like(velocity, dtype=np.float32)
    if len(velocity) > 1:
        acceleration[1:] = velocity[1:] - velocity[:-1]
    speed = np.linalg.norm(velocity, axis=-1)
    accel = np.linalg.norm(acceleration, axis=-1)
    return velocity, acceleration, speed.astype(np.float32), accel.astype(np.float32)


def _robust_spike_threshold(values, quantile):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return 0.0
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median))) * 1.4826
    mad = max(mad, 1e-6)
    multiplier = 2.0 + 4.0 * float(np.clip(quantile, 0.5, 0.99) - 0.5)
    return float(median + multiplier * mad)


def _suppress_velocity_spikes(track, *, conf_thresh, max_velocity_norm, scale, blend_strength, passes=1):
    track = np.asarray(track, dtype=np.float32).copy()
    if len(track) < 3:
        return track
    scale = np.maximum(np.asarray(scale, dtype=np.float32), 1e-6)
    for _ in range(max(int(passes), 1)):
        velocity = track[1:, :, :2] - track[:-1, :, :2]
        speed_norm = np.linalg.norm(velocity, axis=-1) / scale[1:, None]
        spikes = speed_norm > float(max_velocity_norm)
        for frame_index in range(1, track.shape[0] - 1):
            for kp_index in range(track.shape[1]):
                if not spikes[frame_index, kp_index]:
                    continue
                if track[frame_index, kp_index, 2] < conf_thresh:
                    continue
                prev_xy = track[frame_index - 1, kp_index, :2]
                cur_xy = track[frame_index, kp_index, :2]
                next_xy = track[frame_index + 1, kp_index, :2]
                target_xy = 0.5 * (prev_xy + next_xy)
                blended_xy = float(blend_strength) * cur_xy + (1.0 - float(blend_strength)) * target_xy
                track[frame_index, kp_index, :2] = np.clip(blended_xy, 0.0, 1.0)
                track[frame_index, kp_index, 2] = max(track[frame_index, kp_index, 2], conf_thresh)
    return track.astype(np.float32)


def _estimate_body_scale(track):
    shoulder = 0.5 * (track[:, BODY_RIGHT_SHOULDER, :2] + track[:, BODY_LEFT_SHOULDER, :2])
    hip = 0.5 * (track[:, BODY_RIGHT_HIP, :2] + track[:, BODY_LEFT_HIP, :2])
    scale = np.linalg.norm(shoulder - hip, axis=-1)
    return np.maximum(scale, 1e-4).astype(np.float32)


def _track_second_order_jitter(track_xy, scale):
    if len(track_xy) < 3:
        return 0.0
    second = track_xy[2:] - 2.0 * track_xy[1:-1] + track_xy[:-2]
    jitter = np.linalg.norm(second, axis=-1)
    ref = np.maximum(scale[1:-1], 1e-6)
    return float((jitter / ref[:, None]).mean())


def _group_continuity_score(track, states, spike_mask, scale, indices):
    points = np.asarray(track[:, indices, :2], dtype=np.float32)
    if points.size == 0:
        return 0.0
    states = np.asarray(states[:, indices], dtype=np.uint8)
    spike_mask = np.asarray(spike_mask[:, indices], dtype=np.float32)
    scale = np.maximum(np.asarray(scale, dtype=np.float32), 1e-6)
    jitter = _track_second_order_jitter(points, scale)
    visible_ratio = float(np.mean(states == POSE_STATUS_VISIBLE))
    interpolated_ratio = float(np.mean(states == POSE_STATUS_INTERPOLATED))
    occluded_ratio = float(np.mean(states == POSE_STATUS_OCCLUDED))
    low_conf_ratio = float(np.mean(states == POSE_STATUS_LOW_CONFIDENCE))
    spike_ratio = float(np.mean(spike_mask))
    jitter_penalty = np.clip(jitter / 0.02, 0.0, 1.0)
    score = (
        0.50 * visible_ratio
        + 0.20 * interpolated_ratio
        + 0.30 * (1.0 - spike_ratio)
        - 0.20 * occluded_ratio
        - 0.15 * low_conf_ratio
        - 0.20 * jitter_penalty
    )
    return float(np.clip(score, 0.0, 1.0))


def _point_status(raw_conf, refined_conf, occ_value):
    if raw_conf >= 0.35 and occ_value < 0.35:
        return POSE_STATUS_VISIBLE
    if occ_value >= 0.35:
        return POSE_STATUS_OCCLUDED
    if raw_conf < 0.35 and refined_conf >= 0.30:
        return POSE_STATUS_INTERPOLATED
    return POSE_STATUS_LOW_CONFIDENCE


def _make_uncertainty_map(image_shape, body_track, left_hand_track, right_hand_track, speed_scores, occ_map, disagreement_score):
    height, width = image_shape
    maps = []
    body_speed, left_speed, right_speed = speed_scores
    for frame_index in range(body_track.shape[0]):
        canvas = np.zeros((height, width), dtype=np.float32)
        for group_track, group_speed, radius in (
            (body_track, body_speed, 9),
            (left_hand_track, left_speed, 5),
            (right_hand_track, right_speed, 5),
        ):
            for kp_index in range(group_track.shape[1]):
                point = group_track[frame_index, kp_index]
                if point[2] <= 0.05:
                    continue
                px = int(np.clip(round(float(point[0]) * (width - 1)), 0, width - 1))
                py = int(np.clip(round(float(point[1]) * (height - 1)), 0, height - 1))
                score = float(np.clip(0.45 * point[2] + 0.65 * group_speed[frame_index, kp_index] + 0.85 * disagreement_score[frame_index], 0.0, 1.0))
                cv2.circle(canvas, (px, py), radius, score, -1)
        if occ_map is not None:
            canvas = np.maximum(canvas, 0.55 * occ_map[frame_index])
        maps.append(np.clip(canvas, 0.0, 1.0))
    uncertainty = np.stack(maps).astype(np.float32)
    return uncertainty


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported type for json serialization: {type(value)}")


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _write_npz(path, **payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def _build_track_frames(track, velocity, acceleration, states):
    frames = []
    for frame_index in range(track.shape[0]):
        points = np.asarray(track[frame_index], dtype=np.float32)
        velocities = np.asarray(velocity[frame_index], dtype=np.float32)
        accels = np.asarray(acceleration[frame_index], dtype=np.float32)
        frame_states = np.asarray(states[frame_index], dtype=np.int32)
        frames.append({
            "frame_index": int(frame_index),
            "points": points.tolist(),
            "velocity": velocities.tolist(),
            "acceleration": accels.tolist(),
            "states": frame_states.tolist(),
        })
    return frames


def run_pose_motion_stack(
    *,
    export_frames,
    pose_metas,
    raw_pose_metas,
    image_shape,
    occlusion_band=None,
    uncertainty_map=None,
    mode="v1",
    body_conf_thresh=0.5,
    hand_conf_thresh=0.35,
    face_conf_thresh=0.45,
    body_bidirectional_strength=0.84,
    hand_bidirectional_strength=0.72,
    face_bidirectional_strength=0.86,
    local_refine_strength=0.74,
    limb_roi_expand_ratio=1.28,
    hand_roi_expand_ratio=1.65,
    velocity_spike_quantile=0.92,
    uncertainty_blur_kernel=21,
):
    body_track = _as_track(pose_metas, "keypoints_body")
    left_hand_track = _as_track(pose_metas, "keypoints_left_hand")
    right_hand_track = _as_track(pose_metas, "keypoints_right_hand")
    face_track = _as_track(pose_metas, "keypoints_face")
    raw_body_track = _as_track(raw_pose_metas, "keypoints_body")
    raw_left_hand_track = _as_track(raw_pose_metas, "keypoints_left_hand")
    raw_right_hand_track = _as_track(raw_pose_metas, "keypoints_right_hand")
    raw_face_track = _as_track(raw_pose_metas, "keypoints_face")

    limb_roi_stats = {}
    hand_roi_stats = {}
    disagreement_score = np.zeros((body_track.shape[0],), dtype=np.float32)

    if mode != "none":
        body_track, _, _ = _smooth_track_bidirectional(
            body_track,
            conf_thresh=body_conf_thresh,
            smooth_strength=body_bidirectional_strength,
            max_velocity=0.075,
            interp_max_gap=4,
        )
        face_track, _, _ = _smooth_track_bidirectional(
            face_track,
            conf_thresh=face_conf_thresh,
            smooth_strength=face_bidirectional_strength,
            max_velocity=0.05,
            interp_max_gap=4,
        )
        left_hand_track, _, _ = _smooth_track_bidirectional(
            left_hand_track,
            conf_thresh=hand_conf_thresh,
            smooth_strength=hand_bidirectional_strength,
            max_velocity=0.11,
            interp_max_gap=4,
        )
        right_hand_track, _, _ = _smooth_track_bidirectional(
            right_hand_track,
            conf_thresh=hand_conf_thresh,
            smooth_strength=hand_bidirectional_strength,
            max_velocity=0.11,
            interp_max_gap=4,
        )

        for name, indices in (
            ("left_arm", LEFT_ARM_INDICES),
            ("right_arm", RIGHT_ARM_INDICES),
            ("left_leg", LEFT_LEG_INDICES),
            ("right_leg", RIGHT_LEG_INDICES),
        ):
            before = body_track.copy()
            body_track, roi_boxes, stats = _local_refine_group(
                body_track,
                indices,
                conf_thresh=body_conf_thresh,
                smooth_strength=local_refine_strength,
                expand_ratio=limb_roi_expand_ratio,
                min_size_ratio=0.10,
            )
            limb_roi_stats[name] = {**stats, "roi_boxes": roi_boxes.tolist()}
            disagreement_score += np.mean(np.linalg.norm(body_track[:, indices, :2] - before[:, indices, :2], axis=-1), axis=-1)

        for name, hand_track in (("left_hand", left_hand_track), ("right_hand", right_hand_track)):
            before = hand_track.copy()
            refined_hand, roi_boxes, stats = _local_refine_group(
                hand_track,
                list(range(hand_track.shape[1])),
                conf_thresh=hand_conf_thresh,
                smooth_strength=min(0.9, local_refine_strength + 0.08),
                expand_ratio=hand_roi_expand_ratio,
                min_size_ratio=0.08,
            )
            hand_roi_stats[name] = {**stats, "roi_boxes": roi_boxes.tolist()}
            disagreement_score += np.mean(np.linalg.norm(refined_hand[:, :, :2] - before[:, :, :2], axis=-1), axis=-1)
            if name == "left_hand":
                left_hand_track = refined_hand
            else:
                right_hand_track = refined_hand

        body_scale_for_clamp = _estimate_body_scale(body_track)
        body_speed_reference = _compute_velocity(body_track)[2] / np.maximum(body_scale_for_clamp[:, None], 1e-6)
        hand_speed_reference = np.concatenate(
            [
                _compute_velocity(left_hand_track)[2].reshape(-1),
                _compute_velocity(right_hand_track)[2].reshape(-1),
            ]
        ) / max(float(np.mean(body_scale_for_clamp)), 1e-6)
        body_velocity_threshold = _robust_spike_threshold(body_speed_reference, velocity_spike_quantile)
        hand_velocity_threshold = _robust_spike_threshold(hand_speed_reference, max(0.88, velocity_spike_quantile))
        body_track = _suppress_velocity_spikes(
            body_track,
            conf_thresh=body_conf_thresh,
            max_velocity_norm=body_velocity_threshold,
            scale=body_scale_for_clamp,
            blend_strength=0.56,
            passes=2,
        )
        left_hand_track = _suppress_velocity_spikes(
            left_hand_track,
            conf_thresh=hand_conf_thresh,
            max_velocity_norm=hand_velocity_threshold,
            scale=body_scale_for_clamp,
            blend_strength=0.38,
            passes=2,
        )
        right_hand_track = _suppress_velocity_spikes(
            right_hand_track,
            conf_thresh=hand_conf_thresh,
            max_velocity_norm=hand_velocity_threshold,
            scale=body_scale_for_clamp,
            blend_strength=0.38,
            passes=2,
        )

    disagreement_score = np.clip(disagreement_score, 0.0, 1.0).astype(np.float32)

    body_velocity, body_accel, body_speed, body_accel_mag = _compute_velocity(body_track)
    left_velocity, left_accel, left_speed, left_accel_mag = _compute_velocity(left_hand_track)
    right_velocity, right_accel, right_speed, right_accel_mag = _compute_velocity(right_hand_track)
    face_velocity, face_accel, face_speed, face_accel_mag = _compute_velocity(face_track)

    body_scale = _estimate_body_scale(body_track)
    body_speed_norm = body_speed / np.maximum(body_scale[:, None], 1e-6)
    left_speed_norm = left_speed / np.maximum(body_scale[:, None], 1e-6)
    right_speed_norm = right_speed / np.maximum(body_scale[:, None], 1e-6)
    velocity_reference = np.concatenate(
        [
            body_speed_norm.reshape(-1),
            left_speed_norm.reshape(-1),
            right_speed_norm.reshape(-1),
        ]
    )
    velocity_threshold = _robust_spike_threshold(velocity_reference, velocity_spike_quantile)
    body_spike = (body_speed_norm >= velocity_threshold).astype(np.float32)
    left_spike = (left_speed_norm >= velocity_threshold).astype(np.float32)
    right_spike = (right_speed_norm >= velocity_threshold).astype(np.float32)

    body_states = np.zeros(body_track.shape[:2], dtype=np.uint8)
    left_states = np.zeros(left_hand_track.shape[:2], dtype=np.uint8)
    right_states = np.zeros(right_hand_track.shape[:2], dtype=np.uint8)
    face_states = np.zeros(face_track.shape[:2], dtype=np.uint8)
    for frame_index in range(body_track.shape[0]):
        for kp_index in range(body_track.shape[1]):
            occ = _sample_mask(occlusion_band, frame_index, body_track[frame_index, kp_index, :2])
            body_states[frame_index, kp_index] = _point_status(
                raw_body_track[frame_index, kp_index, 2],
                body_track[frame_index, kp_index, 2],
                max(occ, body_spike[frame_index, kp_index]),
            )
        for kp_index in range(left_hand_track.shape[1]):
            occ = _sample_mask(occlusion_band, frame_index, left_hand_track[frame_index, kp_index, :2])
            left_states[frame_index, kp_index] = _point_status(
                raw_left_hand_track[frame_index, kp_index, 2],
                left_hand_track[frame_index, kp_index, 2],
                max(occ, left_spike[frame_index, kp_index]),
            )
        for kp_index in range(right_hand_track.shape[1]):
            occ = _sample_mask(occlusion_band, frame_index, right_hand_track[frame_index, kp_index, :2])
            right_states[frame_index, kp_index] = _point_status(
                raw_right_hand_track[frame_index, kp_index, 2],
                right_hand_track[frame_index, kp_index, 2],
                max(occ, right_spike[frame_index, kp_index]),
            )
        for kp_index in range(face_track.shape[1]):
            occ = _sample_mask(occlusion_band, frame_index, face_track[frame_index, kp_index, :2])
            face_states[frame_index, kp_index] = _point_status(
                raw_face_track[frame_index, kp_index, 2],
                face_track[frame_index, kp_index, 2],
                occ,
            )

    pose_uncertainty = _make_uncertainty_map(
        image_shape,
        body_track,
        left_hand_track,
        right_hand_track,
        (body_spike, left_spike, right_spike),
        occ_map=occlusion_band if occlusion_band is not None else uncertainty_map,
        disagreement_score=disagreement_score,
    )
    if uncertainty_blur_kernel > 1:
        pose_uncertainty = np.stack(
            [cv2.GaussianBlur(frame, (uncertainty_blur_kernel, uncertainty_blur_kernel), 0) for frame in pose_uncertainty]
        ).astype(np.float32)
    pose_uncertainty = np.clip(pose_uncertainty, 0.0, 1.0).astype(np.float32)

    optimized_metas = []
    for frame_index, meta in enumerate(pose_metas):
        updated = _copy_meta(meta)
        updated["keypoints_body"] = body_track[frame_index]
        updated["keypoints_left_hand"] = left_hand_track[frame_index]
        updated["keypoints_right_hand"] = right_hand_track[frame_index]
        updated["keypoints_face"] = face_track[frame_index]
        optimized_metas.append(updated)

    limb_continuity_components = []
    for indices in (LEFT_ARM_INDICES, RIGHT_ARM_INDICES, LEFT_LEG_INDICES, RIGHT_LEG_INDICES):
        limb_continuity_components.append(
            _group_continuity_score(body_track, body_states, body_spike, body_scale, indices)
        )
    limb_continuity_score = float(np.mean(limb_continuity_components)) if limb_continuity_components else 0.0
    limb_roi_coverage_ratio = float(
        np.mean([stats.get("coverage_ratio", 0.0) for stats in limb_roi_stats.values()])
    ) if limb_roi_stats else 0.0
    limb_continuity_score = float(
        np.clip(0.8 * limb_continuity_score + 0.2 * limb_roi_coverage_ratio, 0.0, 1.0)
    )

    body_jitter = _track_second_order_jitter(body_track[:, CORE_BODY_INDICES, :2], body_scale)
    hand_scale = np.maximum(np.concatenate([left_speed_norm.reshape(-1), right_speed_norm.reshape(-1)]).mean() + 1e-4, 1e-4)
    if len(left_hand_track) >= 3:
        left_second = left_hand_track[2:, :, :2] - 2.0 * left_hand_track[1:-1, :, :2] + left_hand_track[:-2, :, :2]
        right_second = right_hand_track[2:, :, :2] - 2.0 * right_hand_track[1:-1, :, :2] + right_hand_track[:-2, :, :2]
        hand_jitter = float(
            (
                np.linalg.norm(left_second, axis=-1).mean()
                + np.linalg.norm(right_second, axis=-1).mean()
            )
            / max(2.0 * hand_scale, 1e-6)
        )
    else:
        hand_jitter = 0.0
    velocity_spike_rate = float(
        np.mean(
            np.concatenate(
                [
                    body_spike.reshape(-1),
                    left_spike.reshape(-1),
                    right_spike.reshape(-1),
                ]
            )
        )
    )

    pose_tracks_json = {
        "frame_count": int(body_track.shape[0]),
        "body_keypoint_count": int(body_track.shape[1]),
        "frames": _build_track_frames(body_track, body_velocity, body_accel, body_states),
        "aggregate_stats": {
            "body_jitter_mean": body_jitter,
            "body_velocity_spike_rate": float(np.mean(body_spike)),
            "velocity_spike_rate": velocity_spike_rate,
            "body_scale_mean": float(body_scale.mean()),
            "raw_body_mean_conf": float(raw_body_track[:, :, 2].mean()),
            "smoothed_body_mean_conf": float(body_track[:, :, 2].mean()),
        },
    }
    limb_tracks_json = {
        "frame_count": int(body_track.shape[0]),
        "groups": {
            "left_arm": LEFT_ARM_INDICES,
            "right_arm": RIGHT_ARM_INDICES,
            "left_leg": LEFT_LEG_INDICES,
            "right_leg": RIGHT_LEG_INDICES,
        },
        "aggregate_stats": {
            "limb_continuity_score": limb_continuity_score,
            "limb_roi_coverage_ratio": limb_roi_coverage_ratio,
            "limb_roi_stats": limb_roi_stats,
        },
    }
    hand_tracks_json = {
        "frame_count": int(left_hand_track.shape[0]),
        "left": _build_track_frames(left_hand_track, left_velocity, left_accel, left_states),
        "right": _build_track_frames(right_hand_track, right_velocity, right_accel, right_states),
        "aggregate_stats": {
            "hand_jitter_mean": hand_jitter,
            "hand_roi_coverage_ratio": float(
                np.mean([stats.get("coverage_ratio", 0.0) for stats in hand_roi_stats.values()])
            ) if hand_roi_stats else 0.0,
            "left_hand_mean_conf": float(left_hand_track[:, :, 2].mean()),
            "right_hand_mean_conf": float(right_hand_track[:, :, 2].mean()),
            "hand_roi_stats": hand_roi_stats,
        },
    }
    pose_visibility_json = {
        "frame_count": int(body_track.shape[0]),
        "status_labels": {str(k): v for k, v in POSE_STATUS_LABELS.items()},
        "frames": [
            {
                "frame_index": int(frame_index),
                "body": body_states[frame_index].tolist(),
                "left_hand": left_states[frame_index].tolist(),
                "right_hand": right_states[frame_index].tolist(),
                "face": face_states[frame_index].tolist(),
            }
            for frame_index in range(body_track.shape[0])
        ],
        "aggregate_counts": {
            "body": {label: int(np.sum(body_states == code)) for code, label in POSE_STATUS_LABELS.items()},
            "left_hand": {label: int(np.sum(left_states == code)) for code, label in POSE_STATUS_LABELS.items()},
            "right_hand": {label: int(np.sum(right_states == code)) for code, label in POSE_STATUS_LABELS.items()},
            "face": {label: int(np.sum(face_states == code)) for code, label in POSE_STATUS_LABELS.items()},
        },
    }
    stats = {
        "body_jitter_mean": body_jitter,
        "hand_jitter_mean": hand_jitter,
        "limb_continuity_score": limb_continuity_score,
        "velocity_spike_rate": velocity_spike_rate,
        "limb_roi_coverage_ratio": limb_roi_coverage_ratio,
        "hand_roi_coverage_ratio": hand_tracks_json["aggregate_stats"]["hand_roi_coverage_ratio"],
        "pose_uncertainty_mean": float(pose_uncertainty.mean()),
        "mode": mode,
    }
    return {
        "optimized_pose_metas": optimized_metas,
        "pose_tracks_json": pose_tracks_json,
        "limb_tracks_json": limb_tracks_json,
        "hand_tracks_json": hand_tracks_json,
        "pose_visibility_json": pose_visibility_json,
        "pose_uncertainty": pose_uncertainty,
        "stats": stats,
    }


def write_pose_motion_artifacts(output_root, analysis, fps, write_json_fn=None):
    output_root = Path(output_root)
    if write_json_fn is None:
        write_json_fn = _write_json
    write_json_fn(output_root / "src_pose_tracks.json", analysis["pose_tracks_json"])
    write_json_fn(output_root / "src_limb_tracks.json", analysis["limb_tracks_json"])
    write_json_fn(output_root / "src_hand_tracks.json", analysis["hand_tracks_json"])
    write_json_fn(output_root / "src_pose_visibility.json", analysis["pose_visibility_json"])
    _write_npz(output_root / "src_pose_uncertainty.npz", mask=analysis["pose_uncertainty"].astype(np.float32), fps=np.array([fps], dtype=np.float32))
    return {
        "pose_tracks": {
            "path": "src_pose_tracks.json",
            "type": "json",
            "format": "json",
            "frame_count": int(analysis["pose_tracks_json"]["frame_count"]),
        },
        "limb_tracks": {
            "path": "src_limb_tracks.json",
            "type": "json",
            "format": "json",
            "frame_count": int(analysis["limb_tracks_json"]["frame_count"]),
        },
        "hand_tracks": {
            "path": "src_hand_tracks.json",
            "type": "json",
            "format": "json",
            "frame_count": int(analysis["hand_tracks_json"]["frame_count"]),
        },
        "pose_visibility": {
            "path": "src_pose_visibility.json",
            "type": "json",
            "format": "json",
            "frame_count": int(analysis["pose_visibility_json"]["frame_count"]),
        },
        "pose_uncertainty": {
            "path": "src_pose_uncertainty.npz",
            "type": "video",
            "format": "npz",
            "frame_count": int(analysis["pose_uncertainty"].shape[0]),
            "height": int(analysis["pose_uncertainty"].shape[1]),
            "width": int(analysis["pose_uncertainty"].shape[2]),
            "channels": 1,
            "dtype": str(analysis["pose_uncertainty"].dtype),
            "shape": list(analysis["pose_uncertainty"].shape),
            "fps": float(fps),
            "value_range": [0.0, 1.0],
            "mask_semantics": "pose_uncertainty",
        },
    }


def make_pose_uncertainty_preview(pose_uncertainty):
    mask = np.asarray(pose_uncertainty, dtype=np.float32)
    overlays = []
    for frame in mask:
        heat = cv2.applyColorMap(np.clip(np.round(frame * 255.0), 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        overlays.append(cv2.cvtColor(heat, cv2.COLOR_BGR2RGB))
    return np.stack(overlays).astype(np.uint8)
