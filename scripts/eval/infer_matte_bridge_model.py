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

from wan.utils.matte_bridge_model import (
    build_bridge_focus_map,
    build_model_from_checkpoint,
    predict_bridge_patches,
    stack_bridge_input_channels,
)
from wan.utils.media_io import load_mask_artifact, load_rgb_artifact, write_person_mask_artifact


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


def _paste_patch(base: np.ndarray, patch: np.ndarray, box_xyxy: tuple[int, int, int, int], pad_xy: tuple[int, int], blend: np.ndarray) -> np.ndarray:
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


def _input_trimap(alpha: np.ndarray, boundary: np.ndarray, uncertainty: np.ndarray) -> np.ndarray:
    return np.logical_or(
        np.logical_and(alpha > 0.08, alpha < 0.92),
        np.logical_or(boundary > 0.10, uncertainty > 0.10),
    ).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply matte bridge model to a preprocess bundle.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--src_root_path", required=True)
    parser.add_argument("--source_video_path", required=True)
    parser.add_argument("--output_root_path", required=True)
    parser.add_argument("--patch_size", type=int, default=192)
    parser.add_argument("--max_rois_per_frame", type=int, default=4)
    parser.add_argument("--min_roi_area", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = build_model_from_checkpoint(checkpoint)
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

    background_artifact = metadata["src_files"]["background"]
    background = load_rgb_artifact(output_root / background_artifact["path"], background_artifact.get("format")).astype(np.uint8)
    base_alpha_artifact = metadata["src_files"]["soft_alpha"]
    base_alpha = load_mask_artifact(output_root / base_alpha_artifact["path"], base_alpha_artifact.get("format")).astype(np.float32)
    person_mask_artifact = metadata["src_files"]["person_mask"]
    person_mask = load_mask_artifact(output_root / person_mask_artifact["path"], person_mask_artifact.get("format")).astype(np.float32)
    boundary_artifact = metadata["src_files"]["boundary_band"]
    boundary_band = load_mask_artifact(output_root / boundary_artifact["path"], boundary_artifact.get("format")).astype(np.float32)
    uncertainty_artifact = metadata["src_files"]["uncertainty_map"]
    uncertainty = load_mask_artifact(output_root / uncertainty_artifact["path"], uncertainty_artifact.get("format")).astype(np.float32)

    frame_count, height, width = base_alpha.shape
    video_rgb = _load_video_frames(Path(args.source_video_path), frame_count, (height, width))

    inputs = []
    mappings = []
    for frame_idx in range(frame_count):
        roi_activity = np.logical_or(boundary_band[frame_idx] > 0.10, uncertainty[frame_idx] > 0.10).astype(np.uint8)
        centers = _component_centers(roi_activity, max_rois=int(args.max_rois_per_frame), min_area=int(args.min_roi_area))
        if not centers:
            continue
        input_trimap = _input_trimap(base_alpha[frame_idx], boundary_band[frame_idx], uncertainty[frame_idx])
        foreground_rgb = np.clip(video_rgb[frame_idx].astype(np.float32) * base_alpha[frame_idx][..., None], 0.0, 255.0).astype(np.uint8)
        for center_x, center_y in centers:
            fg_patch, box_xyxy, pad_xy = _crop_with_padding(foreground_rgb, center_x, center_y, int(args.patch_size))
            bg_patch, _, _ = _crop_with_padding(background[frame_idx], center_x, center_y, int(args.patch_size))
            alpha_patch, _, _ = _crop_with_padding(base_alpha[frame_idx], center_x, center_y, int(args.patch_size))
            trimap_patch, _, _ = _crop_with_padding(input_trimap, center_x, center_y, int(args.patch_size))
            roi_patch, _, _ = _crop_with_padding((roi_activity > 0).astype(np.uint8), center_x, center_y, int(args.patch_size))
            person_patch, _, _ = _crop_with_padding(person_mask[frame_idx], center_x, center_y, int(args.patch_size))
            boundary_patch, _, _ = _crop_with_padding(boundary_band[frame_idx], center_x, center_y, int(args.patch_size))
            uncertainty_patch, _, _ = _crop_with_padding(uncertainty[frame_idx], center_x, center_y, int(args.patch_size))
            focus_patch = build_bridge_focus_map(
                input_trimap_unknown=trimap_patch.astype(np.float32),
                input_boundary_roi_mask=roi_patch.astype(np.float32),
                input_boundary_band=boundary_patch.astype(np.float32),
                input_uncertainty=uncertainty_patch.astype(np.float32),
            )
            inputs.append(
                stack_bridge_input_channels(
                    foreground_patch=fg_patch,
                    background_patch=bg_patch,
                    input_soft_alpha=alpha_patch,
                    input_trimap_unknown=trimap_patch,
                    input_boundary_roi_mask=roi_patch,
                    input_person_mask=person_patch,
                    input_boundary_band=boundary_patch,
                    input_uncertainty=uncertainty_patch,
                )
            )
            mappings.append(
                {
                    "frame_idx": frame_idx,
                    "box_xyxy": box_xyxy,
                    "pad_xy": pad_xy,
                    "focus_patch": focus_patch.astype(np.float32),
                    "base_alpha_patch": alpha_patch.astype(np.float32),
                }
            )

    if not inputs:
        raise RuntimeError("No ROI patches were generated for inference.")

    input_array = np.stack(inputs).astype(np.float32)
    base_alpha_patches = np.stack([mapping["base_alpha_patch"] for mapping in mappings]).astype(np.float32)
    focus_patches = np.stack([mapping["focus_patch"] for mapping in mappings]).astype(np.float32)
    _, bridge_gate_patches, final_alpha_patches = predict_bridge_patches(
        model=model,
        inputs=input_array,
        base_alpha=base_alpha_patches,
        focus_map=focus_patches,
        device=device,
        batch_size=int(args.batch_size),
        gate_strength=float(checkpoint.get("gate_strength", 1.0)),
    )

    adapted_alpha = base_alpha.copy()
    bridge_gate_full = np.zeros_like(base_alpha, dtype=np.float32)
    roi_count_by_frame = {}
    for final_patch, gate_patch, mapping in zip(final_alpha_patches, bridge_gate_patches, mappings):
        frame_idx = int(mapping["frame_idx"])
        roi_count_by_frame[frame_idx] = roi_count_by_frame.get(frame_idx, 0) + 1
        adapted_alpha[frame_idx] = _paste_patch(
            adapted_alpha[frame_idx],
            np.clip(final_patch, 0.0, 1.0),
            mapping["box_xyxy"],
            mapping["pad_xy"],
            mapping["focus_patch"],
        )
        bridge_gate_full[frame_idx] = _paste_patch(
            bridge_gate_full[frame_idx],
            np.clip(gate_patch, 0.0, 1.0),
            mapping["box_xyxy"],
            mapping["pad_xy"],
            mapping["focus_patch"],
        )

    adapted_alpha = np.clip(adapted_alpha, 0.0, 1.0).astype(np.float32)
    threshold = float((checkpoint.get("best_val_metrics") or {}).get("best_threshold", 0.5))
    adapted_person = (adapted_alpha >= threshold).astype(np.float32)
    adapted_trimap = _input_trimap(adapted_alpha, boundary_band, uncertainty).astype(np.float32)

    person_info = write_person_mask_artifact(
        mask_frames=adapted_person,
        output_root=output_root,
        stem="src_mask_matte_bridge",
        artifact_format="npz",
        fps=fps,
        mask_semantics=metadata["src_files"]["person_mask"].get("mask_semantics", "person_foreground"),
    )
    soft_alpha_info = write_person_mask_artifact(
        mask_frames=adapted_alpha,
        output_root=output_root,
        stem="src_soft_alpha_matte_bridge",
        artifact_format="npz",
        fps=fps,
        mask_semantics=metadata["src_files"]["soft_alpha"].get("mask_semantics", "soft_alpha"),
    )
    trimap_info = write_person_mask_artifact(
        mask_frames=adapted_trimap,
        output_root=output_root,
        stem="src_trimap_unknown_matte_bridge",
        artifact_format="npz",
        fps=fps,
        mask_semantics=metadata["src_files"].get("trimap_unknown", {}).get("mask_semantics", "trimap_unknown"),
    )
    bridge_gate_info = write_person_mask_artifact(
        mask_frames=np.clip(bridge_gate_full, 0.0, 1.0).astype(np.float32),
        output_root=output_root,
        stem="src_bridge_gate_matte_bridge",
        artifact_format="npz",
        fps=fps,
        mask_semantics="bridge_gate",
    )

    metadata["src_files"]["person_mask"] = person_info
    metadata["src_files"]["soft_alpha"] = soft_alpha_info
    metadata["src_files"]["trimap_unknown"] = trimap_info
    metadata.setdefault("processing", {})["matte_bridge_model"] = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "patch_size": int(args.patch_size),
        "max_rois_per_frame": int(args.max_rois_per_frame),
        "min_roi_area": int(args.min_roi_area),
        "roi_frames_touched": int(sum(1 for count in roi_count_by_frame.values() if count > 0)),
        "roi_patch_count": int(len(mappings)),
        "mode": "matte_bridge_v1",
        "bridge_gate_artifact": bridge_gate_info,
        "gate_strength": float(checkpoint.get("gate_strength", 1.0)),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary = {
        "src_root_path": str(src_root.resolve()),
        "output_root_path": str(output_root.resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "source_video_path": str(Path(args.source_video_path).resolve()),
        "roi_patch_count": int(len(mappings)),
        "roi_frames_touched": int(sum(1 for count in roi_count_by_frame.values() if count > 0)),
        "avg_rois_per_touched_frame": float(len(mappings) / max(sum(1 for count in roi_count_by_frame.values() if count > 0), 1)),
        "person_mask_threshold": threshold,
        "bridge_gate_mean": float(bridge_gate_full.mean()),
        "bridge_gate_max": float(bridge_gate_full.max()),
    }
    (output_root / "matte_bridge_infer_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
