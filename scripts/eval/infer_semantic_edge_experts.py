#!/usr/bin/env python
import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan.utils.media_io import load_mask_artifact, load_person_mask_artifact, load_rgb_artifact, write_person_mask_artifact
from wan.utils.semantic_edge_experts import build_model_from_checkpoint, predict_patches
from wan.utils.trainable_alpha_model import stack_input_channels


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _load_video_frames(video_path: Path, frame_count: int, shape_hw: tuple[int, int]) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or frame_count)
    if total_frames <= 0:
        total_frames = frame_count
    frame_indices = np.linspace(0, max(total_frames - 1, 0), num=frame_count).round().astype(int)
    frames = []
    target_h, target_w = shape_hw
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read frame {idx} from {video_path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release()
    return np.stack(frames, axis=0).astype(np.uint8)


def _input_trimap(alpha: np.ndarray, boundary: np.ndarray, uncertainty: np.ndarray) -> np.ndarray:
    return np.logical_or(
        np.logical_and(alpha > 0.08, alpha < 0.92),
        np.logical_or(boundary > 0.10, uncertainty > 0.10),
    ).astype(np.uint8)


def _crop_with_padding(arr: np.ndarray, center_x: int, center_y: int, patch_size: int):
    half = patch_size // 2
    h, w = arr.shape[:2]
    x0 = center_x - half
    y0 = center_y - half
    x1 = x0 + patch_size
    y1 = y0 + patch_size
    pad_left = max(0, -x0)
    pad_top = max(0, -y0)
    pad_right = max(0, x1 - w)
    pad_bottom = max(0, y1 - h)
    x0_clip = max(0, x0)
    y0_clip = max(0, y0)
    x1_clip = min(w, x1)
    y1_clip = min(h, y1)
    cropped = arr[y0_clip:y1_clip, x0_clip:x1_clip]
    if arr.ndim == 2:
        padded = np.pad(cropped, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="edge")
    else:
        padded = np.pad(cropped, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode="edge")
    return padded[:patch_size, :patch_size], (x0, y0, x1, y1), (pad_left, pad_top)


def _paste_patch(base: np.ndarray, patch: np.ndarray, box_xyxy, pad_xy, blend: np.ndarray) -> np.ndarray:
    h, w = base.shape
    x0, y0, x1, y1 = box_xyxy
    pad_left, pad_top = pad_xy
    x0_clip = max(0, x0)
    y0_clip = max(0, y0)
    x1_clip = min(w, x1)
    y1_clip = min(h, y1)
    px0 = pad_left
    py0 = pad_top
    px1 = px0 + (x1_clip - x0_clip)
    py1 = py0 + (y1_clip - y0_clip)
    patch_crop = patch[py0:py1, px0:px1]
    blend_crop = blend[py0:py1, px0:px1]
    base[y0_clip:y1_clip, x0_clip:x1_clip] = (
        patch_crop * blend_crop + base[y0_clip:y1_clip, x0_clip:x1_clip] * (1.0 - blend_crop)
    )
    return base


def _component_centers(mask: np.ndarray, max_rois: int, min_area: int) -> list[tuple[int, int]]:
    mask_u8 = (mask > 0).astype(np.uint8)
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    components = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        cx, cy = centroids[label]
        components.append((area, int(round(cx)), int(round(cy))))
    components.sort(reverse=True)
    return [(cx, cy) for _, cx, cy in components[:max_rois]]


def _dilate(mask: np.ndarray, k: int) -> np.ndarray:
    kernel = np.ones((k, k), np.uint8)
    return cv2.dilate((mask > 0).astype(np.uint8), kernel, iterations=1).astype(np.float32)


def _face_route(face_alpha: np.ndarray, activity: np.ndarray) -> np.ndarray:
    return np.clip(_dilate(face_alpha > 0.05, 9) * activity, 0.0, 1.0).astype(np.float32)


def _hair_route(frame_idx: int, face_bbox_frames: list[dict], person_mask: np.ndarray, activity: np.ndarray) -> np.ndarray:
    h, w = person_mask.shape
    route = np.zeros((h, w), dtype=np.float32)
    if frame_idx >= len(face_bbox_frames):
        return route
    frame = face_bbox_frames[frame_idx]
    bbox = frame.get("bbox")
    if not bbox:
        return route
    x1, x2, y1, y2 = [float(v) for v in bbox]
    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)
    hx1 = int(max(0, np.floor(x1 - 0.35 * bw)))
    hx2 = int(min(w, np.ceil(x2 + 0.35 * bw)))
    hy1 = int(max(0, np.floor(y1 - 0.45 * bh)))
    hy2 = int(min(h, np.ceil(y1 + 0.20 * bh)))
    route[hy1:hy2, hx1:hx2] = 1.0
    return np.clip(route * person_mask * activity, 0.0, 1.0).astype(np.float32)


def _hand_route(frame_idx: int, hand_tracks: dict, shape_hw: tuple[int, int], activity: np.ndarray) -> np.ndarray:
    h, w = shape_hw
    route = np.zeros((h, w), dtype=np.float32)
    agg = hand_tracks.get("aggregate_stats") or {}
    hand_roi_stats = agg.get("hand_roi_stats") or {}
    for side_key in ("left_hand", "right_hand"):
        roi_stats = hand_roi_stats.get(side_key) or {}
        roi_boxes = roi_stats.get("roi_boxes") or []
        if frame_idx >= len(roi_boxes):
            continue
        x1n, y1n, x2n, y2n = roi_boxes[frame_idx]
        x1 = int(max(0, np.floor(x1n * w)))
        x2 = int(min(w, np.ceil(x2n * w)))
        y1 = int(max(0, np.floor(y1n * h)))
        y2 = int(min(h, np.ceil(y2n * h)))
        route[y1:y2, x1:x2] = 1.0
    route = _dilate(route > 0, 7)
    return np.clip(route * activity, 0.0, 1.0).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Apply semantic edge experts to a preprocess bundle.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--src_root_path", required=True)
    parser.add_argument("--source_video_path", required=True)
    parser.add_argument("--output_root_path", required=True)
    parser.add_argument("--patch_size", type=int, default=192)
    parser.add_argument("--max_rois_per_frame", type=int, default=3)
    parser.add_argument("--min_roi_area", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = build_model_from_checkpoint(checkpoint)
    enabled_tags = tuple(checkpoint["semantic_tags"])
    semantic_to_id = {tag: idx for idx, tag in enumerate(enabled_tags)}
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)

    src_root = Path(args.src_root_path)
    output_root = Path(args.output_root_path)
    if output_root.exists():
        shutil.rmtree(output_root)
    shutil.copytree(src_root, output_root)

    metadata_path = output_root / "metadata.json"
    metadata = _read_json(metadata_path)
    fps = float(metadata.get("fps", 5.0) or 5.0)

    def load_mask(name: str):
        artifact = metadata["src_files"][name]
        loader = load_person_mask_artifact if name == "person_mask" else load_mask_artifact
        return loader(output_root / artifact["path"], artifact.get("format")).astype(np.float32)

    background = load_rgb_artifact(
        output_root / metadata["src_files"]["background"]["path"],
        metadata["src_files"]["background"].get("format"),
    ).astype(np.uint8)
    base_alpha = load_mask("soft_alpha")
    person_mask = load_mask("person_mask")
    boundary_band = load_mask("boundary_band")
    uncertainty = load_mask("uncertainty_map")
    face_alpha = load_mask("face_alpha") if "face_alpha" in metadata["src_files"] else np.zeros_like(base_alpha)
    frame_count, height, width = base_alpha.shape
    video_rgb = _load_video_frames(Path(args.source_video_path), frame_count, (height, width))
    face_bbox_curve = _read_json(output_root / "face_bbox_curve.json")
    face_bbox_frames = face_bbox_curve.get("frames") or []
    hand_tracks = _read_json(output_root / "src_hand_tracks.json") if (output_root / "src_hand_tracks.json").exists() else {}

    semantic_route_masks = {tag: np.zeros_like(base_alpha, dtype=np.float32) for tag in enabled_tags}
    activity_all = np.clip(np.maximum(boundary_band, uncertainty), 0.0, 1.0)
    trimap_all = _input_trimap(base_alpha, boundary_band, uncertainty).astype(np.float32)
    for frame_idx in range(frame_count):
        activity = np.clip(np.maximum(activity_all[frame_idx], trimap_all[frame_idx]), 0.0, 1.0)
        face_route = _face_route(face_alpha[frame_idx], activity)
        hand_route = _hand_route(frame_idx, hand_tracks, (height, width), activity)
        hair_route = _hair_route(frame_idx, face_bbox_frames, person_mask[frame_idx], activity)
        cloth_route = np.clip(person_mask[frame_idx] * activity - np.maximum.reduce([face_route, hand_route, hair_route]), 0.0, 1.0)
        route_map = {
            "face": face_route,
            "hair": hair_route,
            "hand": hand_route,
            "cloth": cloth_route,
        }
        for tag in enabled_tags:
            semantic_route_masks[tag][frame_idx] = route_map.get(tag, np.zeros((height, width), dtype=np.float32))

    inputs = []
    mappings = []
    for frame_idx in range(frame_count):
        input_trimap = trimap_all[frame_idx]
        foreground_rgb = np.clip(video_rgb[frame_idx].astype(np.float32) * base_alpha[frame_idx][..., None], 0.0, 255.0).astype(np.uint8)
        for tag in enabled_tags:
            route_mask = semantic_route_masks[tag][frame_idx]
            centers = _component_centers(route_mask, max_rois=int(args.max_rois_per_frame), min_area=int(args.min_roi_area))
            for center_x, center_y in centers:
                fg_patch, box_xyxy, pad_xy = _crop_with_padding(foreground_rgb, center_x, center_y, int(args.patch_size))
                bg_patch, _, _ = _crop_with_padding(background[frame_idx], center_x, center_y, int(args.patch_size))
                alpha_patch, _, _ = _crop_with_padding(base_alpha[frame_idx], center_x, center_y, int(args.patch_size))
                trimap_patch, _, _ = _crop_with_padding(input_trimap, center_x, center_y, int(args.patch_size))
                route_patch, _, _ = _crop_with_padding(route_mask, center_x, center_y, int(args.patch_size))
                person_patch, _, _ = _crop_with_padding(person_mask[frame_idx], center_x, center_y, int(args.patch_size))
                boundary_patch, _, _ = _crop_with_padding(boundary_band[frame_idx], center_x, center_y, int(args.patch_size))
                uncertainty_patch, _, _ = _crop_with_padding(uncertainty[frame_idx], center_x, center_y, int(args.patch_size))
                inputs.append(stack_input_channels(
                    foreground_patch=fg_patch,
                    background_patch=bg_patch,
                    input_soft_alpha=alpha_patch,
                    input_trimap_unknown=trimap_patch,
                    input_boundary_roi_mask=route_patch,
                    input_person_mask=person_patch,
                    input_boundary_band=boundary_patch,
                    input_uncertainty=uncertainty_patch,
                ))
                mappings.append({
                    "frame_idx": frame_idx,
                    "semantic_tag": tag,
                    "semantic_id": semantic_to_id[tag],
                    "box_xyxy": box_xyxy,
                    "pad_xy": pad_xy,
                    "base_alpha_patch": alpha_patch.astype(np.float32),
                    "blend_patch": np.clip(np.maximum.reduce([
                        route_patch.astype(np.float32),
                        0.75 * boundary_patch.astype(np.float32),
                        0.50 * trimap_patch.astype(np.float32),
                        0.35 * uncertainty_patch.astype(np.float32),
                    ]), 0.0, 1.0).astype(np.float32),
                })

    if not inputs:
        raise RuntimeError("No semantic ROI patches were generated for inference.")

    input_array = np.stack(inputs).astype(np.float32)
    base_alpha_patches = np.stack([m["base_alpha_patch"] for m in mappings]).astype(np.float32)
    semantic_ids = np.asarray([m["semantic_id"] for m in mappings], dtype=np.int64)
    pred_patches = predict_patches(
        model=model,
        inputs=input_array,
        base_alpha=base_alpha_patches,
        semantic_ids=semantic_ids,
        device=device,
        batch_size=int(args.batch_size),
    )

    adapted_alpha = base_alpha.copy()
    route_counts = {tag: 0 for tag in enabled_tags}
    for pred_patch, mapping in zip(pred_patches, mappings):
        route_counts[mapping["semantic_tag"]] += 1
        frame_idx = int(mapping["frame_idx"])
        adapted_alpha[frame_idx] = _paste_patch(
            adapted_alpha[frame_idx],
            np.clip(pred_patch.astype(np.float32), 0.0, 1.0),
            mapping["box_xyxy"],
            mapping["pad_xy"],
            mapping["blend_patch"],
        )

    adapted_alpha = np.clip(adapted_alpha, 0.0, 1.0).astype(np.float32)
    threshold = 0.35
    adapted_person = (adapted_alpha >= threshold).astype(np.float32)
    adapted_trimap = _input_trimap(adapted_alpha, boundary_band, uncertainty).astype(np.float32)

    person_info = write_person_mask_artifact(
        mask_frames=adapted_person,
        output_root=output_root,
        stem="src_mask_semantic_experts",
        artifact_format="npz",
        fps=fps,
        mask_semantics=metadata["src_files"]["person_mask"].get("mask_semantics", "person_foreground"),
    )
    soft_alpha_info = write_person_mask_artifact(
        mask_frames=adapted_alpha,
        output_root=output_root,
        stem="src_soft_alpha_semantic_experts",
        artifact_format="npz",
        fps=fps,
        mask_semantics=metadata["src_files"]["soft_alpha"].get("mask_semantics", "soft_alpha"),
    )
    trimap_info = write_person_mask_artifact(
        mask_frames=adapted_trimap,
        output_root=output_root,
        stem="src_trimap_unknown_semantic_experts",
        artifact_format="npz",
        fps=fps,
        mask_semantics=metadata["src_files"].get("trimap_unknown", {}).get("mask_semantics", "trimap_unknown"),
    )

    metadata["src_files"]["person_mask"] = person_info
    metadata["src_files"]["soft_alpha"] = soft_alpha_info
    metadata["src_files"]["trimap_unknown"] = trimap_info
    metadata.setdefault("processing", {})["semantic_edge_experts"] = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "mode": "semantic_experts_v1",
        "enabled_tags": list(enabled_tags),
        "patch_size": int(args.patch_size),
        "max_rois_per_frame": int(args.max_rois_per_frame),
        "min_roi_area": int(args.min_roi_area),
        "route_counts": route_counts,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary = {
        "src_root_path": str(src_root.resolve()),
        "output_root_path": str(output_root.resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "source_video_path": str(Path(args.source_video_path).resolve()),
        "enabled_tags": list(enabled_tags),
        "route_counts": route_counts,
        "roi_patch_count": int(len(mappings)),
    }
    (output_root / "semantic_edge_experts_infer_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
