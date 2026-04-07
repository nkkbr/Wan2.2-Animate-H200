import json
from pathlib import Path

import cv2
import numpy as np


BODY_PROMPT_INDICES = list(range(14))


def _as_array(points):
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return np.zeros((0, 3), dtype=np.float32)
    if arr.shape[1] == 2:
        scores = np.ones((arr.shape[0], 1), dtype=np.float32)
        arr = np.concatenate([arr, scores], axis=1)
    return arr[:, :3]


def _valid_points(points, width, height, conf_thresh):
    arr = _as_array(points)
    if arr.size == 0:
        return np.zeros((0, 2), dtype=np.int32), np.zeros((0,), dtype=np.float32)
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
    coords = arr[valid, :2]
    if coords.size == 0:
        return np.zeros((0, 2), dtype=np.int32), np.zeros((0,), dtype=np.float32)
    coords[:, 0] *= width
    coords[:, 1] *= height
    coords = np.rint(coords).astype(np.int32)
    coords[:, 0] = np.clip(coords[:, 0], 0, width - 1)
    coords[:, 1] = np.clip(coords[:, 1], 0, height - 1)
    return coords, arr[valid, 2]


def _dedupe_points(points):
    if points.size == 0:
        return points.reshape(0, 2)
    unique = []
    seen = set()
    for x, y in points.tolist():
        key = (int(x), int(y))
        if key in seen:
            continue
        seen.add(key)
        unique.append(key)
    return np.asarray(unique, dtype=np.int32)


def _center_of_points(points):
    if points.size == 0:
        return None
    center = np.mean(points, axis=0)
    return np.rint(center).astype(np.int32)


def _bbox_from_points(points, image_shape):
    height, width = image_shape
    if points.size == 0:
        return None
    x1 = int(np.clip(points[:, 0].min(), 0, width - 1))
    y1 = int(np.clip(points[:, 1].min(), 0, height - 1))
    x2 = int(np.clip(points[:, 0].max(), x1 + 1, width))
    y2 = int(np.clip(points[:, 1].max(), y1 + 1, height))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def compute_prompt_frames(frame_count, keyframes_per_chunk, reprompt_interval):
    if frame_count <= 0:
        return [], {}, []
    keyframes_per_chunk = max(1, min(int(keyframes_per_chunk), frame_count))
    base_frames = np.linspace(0, frame_count - 1, num=keyframes_per_chunk, dtype=np.int32).tolist()
    tags = {int(frame_idx): {"keyframe"} for frame_idx in base_frames}
    reprompt_frames = []
    if reprompt_interval is not None and int(reprompt_interval) > 0:
        reprompt_step = int(reprompt_interval)
        reprompt_frames = list(range(0, frame_count, reprompt_step))
        if reprompt_frames[-1] != frame_count - 1:
            reprompt_frames.append(frame_count - 1)
        for frame_idx in reprompt_frames:
            tags.setdefault(int(frame_idx), set()).add("reprompt")
    prompt_frames = sorted(set(base_frames) | set(reprompt_frames) | {0, frame_count - 1})
    tags[0].add("boundary")
    tags[frame_count - 1].add("boundary")
    normalized_tags = {frame_idx: sorted(frame_tags) for frame_idx, frame_tags in tags.items()}
    return prompt_frames, normalized_tags, sorted(set(reprompt_frames))


def _collect_person_points(meta, image_shape, body_conf_thresh, face_conf_thresh, hand_conf_thresh):
    height, width = image_shape
    collected = []
    body_points, _ = _valid_points(meta["keypoints_body"], width, height, body_conf_thresh)
    if body_points.size > 0:
        collected.append(body_points)
    face_points, _ = _valid_points(meta["keypoints_face"], width, height, face_conf_thresh)
    if face_points.size > 0:
        collected.append(face_points)
    left_hand_points, _ = _valid_points(meta["keypoints_left_hand"], width, height, hand_conf_thresh)
    if left_hand_points.size > 0:
        collected.append(left_hand_points)
    right_hand_points, _ = _valid_points(meta["keypoints_right_hand"], width, height, hand_conf_thresh)
    if right_hand_points.size > 0:
        collected.append(right_hand_points)
    if not collected:
        return np.zeros((0, 2), dtype=np.int32)
    return _dedupe_points(np.concatenate(collected, axis=0))


def build_prompt_for_frame(
    meta,
    *,
    image_shape,
    body_conf_thresh=0.35,
    face_conf_thresh=0.45,
    hand_conf_thresh=0.35,
    face_min_points=8,
    hand_min_points=6,
    use_negative_points=True,
    negative_margin=0.08,
):
    height, width = image_shape
    positive_parts = []
    positive_sources = {}

    body_array = _as_array(meta["keypoints_body"])
    selected_body = []
    for index in BODY_PROMPT_INDICES:
        if index >= len(body_array):
            continue
        keypoint = body_array[index]
        if not np.isfinite(keypoint).all() or keypoint[2] < body_conf_thresh:
            continue
        x = int(np.clip(round(keypoint[0] * width), 0, width - 1))
        y = int(np.clip(round(keypoint[1] * height), 0, height - 1))
        selected_body.append([x, y])
    if selected_body:
        body_points = _dedupe_points(np.asarray(selected_body, dtype=np.int32))
        positive_parts.append(body_points)
        positive_sources["body_points"] = int(len(body_points))

    face_points, _ = _valid_points(meta["keypoints_face"], width, height, face_conf_thresh)
    if len(face_points) >= int(face_min_points):
        face_center = _center_of_points(face_points)
        if face_center is not None:
            positive_parts.append(face_center[None, :])
            positive_sources["face_center"] = 1

    for hand_name, hand_key in [("left_hand_center", "keypoints_left_hand"), ("right_hand_center", "keypoints_right_hand")]:
        hand_points, _ = _valid_points(meta[hand_key], width, height, hand_conf_thresh)
        if len(hand_points) >= int(hand_min_points):
            hand_center = _center_of_points(hand_points)
            if hand_center is not None:
                positive_parts.append(hand_center[None, :])
                positive_sources[hand_name] = 1

    person_points = _collect_person_points(
        meta,
        image_shape=image_shape,
        body_conf_thresh=min(body_conf_thresh, 0.2),
        face_conf_thresh=min(face_conf_thresh, 0.2),
        hand_conf_thresh=min(hand_conf_thresh, 0.2),
    )
    person_bbox = _bbox_from_points(person_points, image_shape)

    if not positive_parts and person_bbox is not None:
        x1, y1, x2, y2 = person_bbox
        bbox_center = np.asarray([[int(round((x1 + x2) / 2.0)), int(round((y1 + y2) / 2.0))]], dtype=np.int32)
        positive_parts.append(bbox_center)
        positive_sources["bbox_center_fallback"] = 1

    positive_points = _dedupe_points(np.concatenate(positive_parts, axis=0)) if positive_parts else np.zeros((0, 2), dtype=np.int32)

    negative_points = np.zeros((0, 2), dtype=np.int32)
    if use_negative_points and person_bbox is not None:
        x1, y1, x2, y2 = person_bbox
        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)
        margin_px = max(4, int(round(max(box_w, box_h) * negative_margin)))
        ex1 = max(0, x1 - margin_px)
        ey1 = max(0, y1 - margin_px)
        ex2 = min(width - 1, x2 + margin_px)
        ey2 = min(height - 1, y2 + margin_px)
        cx = int(round((x1 + x2) / 2.0))
        cy = int(round((y1 + y2) / 2.0))
        candidates = np.asarray(
            [
                [ex1, ey1],
                [cx, ey1],
                [ex2, ey1],
                [ex1, cy],
                [ex2, cy],
                [ex1, ey2],
                [cx, ey2],
                [ex2, ey2],
            ],
            dtype=np.int32,
        )
        outside = (
            (candidates[:, 0] < x1)
            | (candidates[:, 0] > x2)
            | (candidates[:, 1] < y1)
            | (candidates[:, 1] > y2)
        )
        negative_points = _dedupe_points(candidates[outside])

    points = positive_points
    labels = np.ones((len(positive_points),), dtype=np.int32)
    if len(negative_points) > 0:
        points = np.concatenate([positive_points, negative_points], axis=0)
        labels = np.concatenate(
            [
                np.ones((len(positive_points),), dtype=np.int32),
                np.zeros((len(negative_points),), dtype=np.int32),
            ],
            axis=0,
        )

    return {
        "points": points.astype(np.int32),
        "labels": labels.astype(np.int32),
        "positive_points": positive_points.astype(np.int32),
        "negative_points": negative_points.astype(np.int32),
        "positive_count": int(len(positive_points)),
        "negative_count": int(len(negative_points)),
        "positive_sources": positive_sources,
        "person_bbox": person_bbox,
    }


def plan_chunk_prompts(
    chunk_metas,
    *,
    image_shape,
    keyframes_per_chunk,
    reprompt_interval,
    body_conf_thresh=0.35,
    face_conf_thresh=0.45,
    hand_conf_thresh=0.35,
    face_min_points=8,
    hand_min_points=6,
    use_negative_points=True,
    negative_margin=0.08,
):
    prompt_frames, prompt_tags, reprompt_frames = compute_prompt_frames(
        frame_count=len(chunk_metas),
        keyframes_per_chunk=keyframes_per_chunk,
        reprompt_interval=reprompt_interval,
    )
    prompt_entries = []
    for frame_idx in prompt_frames:
        prompt = build_prompt_for_frame(
            chunk_metas[frame_idx],
            image_shape=image_shape,
            body_conf_thresh=body_conf_thresh,
            face_conf_thresh=face_conf_thresh,
            hand_conf_thresh=hand_conf_thresh,
            face_min_points=face_min_points,
            hand_min_points=hand_min_points,
            use_negative_points=use_negative_points,
            negative_margin=negative_margin,
        )
        if prompt["positive_count"] <= 0:
            continue
        prompt_entries.append(
            {
                "frame_idx": int(frame_idx),
                "tags": prompt_tags.get(int(frame_idx), []),
                "positive_points": prompt["positive_points"],
                "negative_points": prompt["negative_points"],
                "points": prompt["points"],
                "labels": prompt["labels"],
                "positive_count": prompt["positive_count"],
                "negative_count": prompt["negative_count"],
                "positive_sources": prompt["positive_sources"],
                "person_bbox": prompt["person_bbox"],
            }
        )
    return {
        "prompt_entries": prompt_entries,
        "prompt_frames": [entry["frame_idx"] for entry in prompt_entries],
        "reprompt_frames": [frame_idx for frame_idx in reprompt_frames if frame_idx in {entry["frame_idx"] for entry in prompt_entries}],
    }


def _overlay_mask(frame, mask, color=(64, 255, 96), alpha=0.35):
    output = frame.copy()
    if mask.dtype != np.uint8:
        mask_u8 = (mask > 0).astype(np.uint8)
    else:
        mask_u8 = (mask > 0).astype(np.uint8)
    if np.any(mask_u8):
        overlay = np.zeros_like(output)
        overlay[:, :] = np.asarray(color, dtype=np.uint8)
        output = np.where(mask_u8[:, :, None].astype(bool), cv2.addWeighted(output, 1.0 - alpha, overlay, alpha, 0), output)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(output, contours, -1, (0, 255, 0), 2)
    return output


def make_mask_overlay(frames, masks, prompt_entries=None):
    overlays = []
    prompt_by_frame = {}
    for entry in prompt_entries or []:
        frame_idx = int(entry.get("global_frame_idx", entry["frame_idx"]))
        prompt_by_frame[frame_idx] = entry
    for frame_idx, (frame, mask) in enumerate(zip(frames, masks)):
        overlay = _overlay_mask(frame, mask)
        if frame_idx in prompt_by_frame:
            cv2.putText(overlay, "SAM prompt frame", (24, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
        overlays.append(overlay.astype(np.uint8))
    return np.stack(overlays)


def make_sam_prompts_overlay(frames, prompt_entries):
    overlays = []
    prompt_by_frame = {
        int(entry.get("global_frame_idx", entry["frame_idx"])): entry
        for entry in prompt_entries
    }
    for frame_idx, frame in enumerate(frames):
        overlay = frame.copy()
        entry = prompt_by_frame.get(frame_idx)
        if entry is not None:
            bbox = entry.get("person_bbox")
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 0), 2)
            for x, y in entry["positive_points"].tolist():
                cv2.circle(overlay, (int(x), int(y)), 6, (0, 255, 0), thickness=-1)
            for x, y in entry["negative_points"].tolist():
                cv2.circle(overlay, (int(x), int(y)), 6, (0, 0, 255), thickness=2)
                cv2.line(overlay, (int(x) - 5, int(y) - 5), (int(x) + 5, int(y) + 5), (0, 0, 255), 2)
                cv2.line(overlay, (int(x) - 5, int(y) + 5), (int(x) + 5, int(y) - 5), (0, 0, 255), 2)
            tag_text = ", ".join(entry.get("tags", [])) or "prompt"
            label = f"{tag_text} +{entry['positive_count']} -{entry['negative_count']}"
            cv2.putText(overlay, label, (24, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
        overlays.append(overlay.astype(np.uint8))
    return np.stack(overlays)


def write_prompt_keyframes(output_dir, frames, prompt_entries, prefix="sam_prompt_keyframes"):
    root = Path(output_dir) / prefix
    root.mkdir(parents=True, exist_ok=True)
    overlays = make_sam_prompts_overlay(frames, prompt_entries)
    written = []
    for entry in prompt_entries:
        frame_idx = int(entry.get("global_frame_idx", entry["frame_idx"]))
        path = root / f"frame_{frame_idx:06d}.png"
        ok = cv2.imwrite(str(path), cv2.cvtColor(overlays[frame_idx], cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError(f"Failed to write SAM prompt keyframe: {path}")
        written.append(str(Path(prefix) / path.name))
    return {"path": prefix, "frames": written}


def build_mask_stats(
    *,
    masks,
    chunk_plans,
    sam_chunk_len,
    sam_keyframes_per_chunk,
    sam_reprompt_interval,
    sam_use_negative_points,
    sam_negative_margin,
):
    masks = np.asarray(masks, dtype=np.float32)
    frame_count, height, width = masks.shape
    area_pixels = masks.reshape(frame_count, -1).sum(axis=1)
    area_ratio = area_pixels / float(height * width)
    area_delta_ratio = np.zeros_like(area_ratio)
    if frame_count > 1:
        area_delta_ratio[1:] = np.abs(np.diff(area_ratio))

    chunks = []
    prompt_entries = []
    for chunk in chunk_plans:
        chunk_prompt_entries = []
        for entry in chunk["prompt_entries"]:
            global_entry = {
                "frame_idx": int(entry["global_frame_idx"]),
                "chunk_index": int(chunk["chunk_index"]),
                "chunk_frame_idx": int(entry["frame_idx"]),
                "tags": entry.get("tags", []),
                "positive_count": int(entry["positive_count"]),
                "negative_count": int(entry["negative_count"]),
                "positive_sources": entry.get("positive_sources", {}),
                "person_bbox": entry.get("person_bbox"),
            }
            prompt_entries.append(global_entry)
            chunk_prompt_entries.append(global_entry)
        start = int(chunk["start_frame"])
        end = int(chunk["end_frame"])
        chunk_area_ratio = area_ratio[start:end + 1]
        chunks.append(
            {
                "chunk_index": int(chunk["chunk_index"]),
                "start_frame": start,
                "end_frame": end,
                "frame_count": int(end - start + 1),
                "prompt_frames": [entry["frame_idx"] for entry in chunk_prompt_entries],
                "reprompt_frames": [int(start + frame_idx) for frame_idx in chunk.get("reprompt_frames", [])],
                "prompt_count": int(len(chunk_prompt_entries)),
                "positive_points": int(sum(entry["positive_count"] for entry in chunk_prompt_entries)),
                "negative_points": int(sum(entry["negative_count"] for entry in chunk_prompt_entries)),
                "area_ratio_mean": float(chunk_area_ratio.mean()) if len(chunk_area_ratio) else 0.0,
                "area_ratio_min": float(chunk_area_ratio.min()) if len(chunk_area_ratio) else 0.0,
                "area_ratio_max": float(chunk_area_ratio.max()) if len(chunk_area_ratio) else 0.0,
                "area_ratio_std": float(chunk_area_ratio.std()) if len(chunk_area_ratio) else 0.0,
            }
        )

    return {
        "frame_count": int(frame_count),
        "height": int(height),
        "width": int(width),
        "sam_chunk_len": int(sam_chunk_len),
        "sam_keyframes_per_chunk": int(sam_keyframes_per_chunk),
        "sam_reprompt_interval": int(sam_reprompt_interval),
        "sam_use_negative_points": bool(sam_use_negative_points),
        "sam_negative_margin": float(sam_negative_margin),
        "mask_area_pixels": area_pixels.astype(np.float32).round(2).tolist(),
        "mask_area_ratio": area_ratio.astype(np.float32).round(6).tolist(),
        "mask_area_delta_ratio": area_delta_ratio.astype(np.float32).round(6).tolist(),
        "total_prompt_frames": int(len(prompt_entries)),
        "total_positive_points": int(sum(entry["positive_count"] for entry in prompt_entries)),
        "total_negative_points": int(sum(entry["negative_count"] for entry in prompt_entries)),
        "prompt_entries": prompt_entries,
        "chunks": chunks,
    }


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
