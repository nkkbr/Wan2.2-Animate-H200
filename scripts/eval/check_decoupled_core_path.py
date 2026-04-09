#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan.utils.animate_contract import load_preprocess_metadata, resolve_preprocess_artifacts
from wan.utils.replacement_masks import compose_background_keep_mask
from wan.utils.rich_conditioning import (
    build_boundary_conditioning_maps,
    build_core_condition_rgb,
    build_face_conditioning_maps,
    load_json_if_exists,
    summarize_reference_structure_guard,
)


def _read_video_frame_indices(path: Path, indices: list[int], *, rgb: bool) -> np.ndarray:
    try:
        import decord

        vr = decord.VideoReader(str(path))
        frames = vr.get_batch(indices).asnumpy()
        if not rgb:
            frames = frames[..., 0]
        return frames
    except Exception:
        import cv2

        cap = cv2.VideoCapture(str(path))
        frames = []
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Failed to read frame {index} from {path}")
            if rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame = frame[..., 0]
            frames.append(frame)
        cap.release()
        return np.stack(frames)


def _sample_artifact(artifact: dict, indices: list[int], *, rgb: bool) -> np.ndarray:
    path = Path(artifact["path"])
    artifact_format = artifact.get("format")
    if artifact_format == "npz":
        data = np.load(path)
        if "mask" in data:
            arr = data["mask"]
        elif "frames" in data:
            arr = data["frames"]
        else:
            first_key = next(iter(data.files))
            arr = data[first_key]
        sampled = arr[indices]
        if rgb and sampled.ndim == 3:
            sampled = np.repeat(sampled[..., None], 3, axis=-1)
        if not rgb and sampled.ndim == 4:
            sampled = sampled[..., 0]
        return sampled
    if artifact_format == "png_seq":
        frame_dir = path
        frame_paths = sorted(frame_dir.glob("*.png"))
        import imageio.v3 as iio

        sampled = [iio.imread(frame_paths[index]) for index in indices]
        sampled = np.stack(sampled)
        if not rgb and sampled.ndim == 4:
            sampled = sampled[..., 0]
        return sampled
    return _read_video_frame_indices(path, indices, rgb=rgb)


def _load_optional_mask(artifacts: dict, name: str):
    artifact = artifacts.get(name)
    if not artifact:
        return None
    return load_mask_artifact(artifact["path"], artifact.get("format"))


def _load_optional_rgb(artifacts: dict, name: str):
    artifact = artifacts.get(name)
    if not artifact:
        return None
    return load_rgb_artifact(artifact["path"], artifact.get("format"))


def main():
    parser = argparse.ArgumentParser(description="Validate Optimization6 Step03 decoupled core conditioning path.")
    parser.add_argument("--src_root_path", type=str, required=True)
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    src_root = Path(args.src_root_path).resolve()
    artifacts, metadata = resolve_preprocess_artifacts(src_root, replace_flag=True)
    if metadata is None:
        raise SystemExit("metadata.json is required for decoupled core-path validation.")

    required = [
        "foreground_rgb",
        "foreground_alpha",
        "foreground_confidence",
        "background_rgb",
        "background_visible_support",
        "background_unresolved",
        "composite_roi_mask",
        "soft_alpha",
        "boundary_band",
        "background_keep_prior",
    ]
    missing = [key for key in required if key not in artifacts]
    if missing:
        raise SystemExit(f"Missing required decoupled core artifacts: {missing}")

    frame_count = int(metadata.get("frame_count") or metadata.get("runtime", {}).get("frame_count"))
    sample_indices = sorted({0, max(0, frame_count // 2), max(0, frame_count - 1)})

    background_rgb = _sample_artifact(artifacts["background_rgb"], sample_indices, rgb=True)
    foreground_rgb = _sample_artifact(artifacts["foreground_rgb"], sample_indices, rgb=True)
    foreground_alpha = _sample_artifact(artifacts["foreground_alpha"], sample_indices, rgb=False)
    foreground_confidence = _sample_artifact(artifacts["foreground_confidence"], sample_indices, rgb=False)
    background_keep_prior = _sample_artifact(artifacts["background_keep_prior"], sample_indices, rgb=False)
    visible_support = _sample_artifact(artifacts["background_visible_support"], sample_indices, rgb=False)
    unresolved_region = _sample_artifact(artifacts["background_unresolved"], sample_indices, rgb=False)
    composite_roi_mask = _sample_artifact(artifacts["composite_roi_mask"], sample_indices, rgb=False)
    person_mask = _sample_artifact(artifacts["person_mask"], sample_indices, rgb=False)

    soft_alpha = _sample_artifact(artifacts["soft_alpha"], sample_indices, rgb=False) if "soft_alpha" in artifacts else None
    alpha_v2 = _sample_artifact(artifacts["alpha_v2"], sample_indices, rgb=False) if "alpha_v2" in artifacts else None
    trimap_v2 = _sample_artifact(artifacts["trimap_v2"], sample_indices, rgb=False) if "trimap_v2" in artifacts else None
    trimap_unknown = _sample_artifact(artifacts["trimap_unknown"], sample_indices, rgb=False) if "trimap_unknown" in artifacts else None
    boundary_band = _sample_artifact(artifacts["boundary_band"], sample_indices, rgb=False) if "boundary_band" in artifacts else None
    fine_boundary_mask = _sample_artifact(artifacts["fine_boundary_mask"], sample_indices, rgb=False) if "fine_boundary_mask" in artifacts else None
    hair_alpha = _sample_artifact(artifacts["hair_alpha"], sample_indices, rgb=False) if "hair_alpha" in artifacts else None
    hair_edge_mask = _sample_artifact(artifacts["hair_edge_mask"], sample_indices, rgb=False) if "hair_edge_mask" in artifacts else None
    alpha_uncertainty_v2 = _sample_artifact(artifacts["alpha_uncertainty_v2"], sample_indices, rgb=False) if "alpha_uncertainty_v2" in artifacts else None
    alpha_confidence = _sample_artifact(artifacts["alpha_confidence_v2"], sample_indices, rgb=False) if "alpha_confidence_v2" in artifacts else None
    alpha_source_provenance = _sample_artifact(artifacts["alpha_source_provenance_v2"], sample_indices, rgb=False) if "alpha_source_provenance_v2" in artifacts else None
    uncertainty_map = _sample_artifact(artifacts["uncertainty_map"], sample_indices, rgb=False) if "uncertainty_map" in artifacts else None
    occlusion_band = _sample_artifact(artifacts["occlusion_band"], sample_indices, rgb=False) if "occlusion_band" in artifacts else None
    face_alpha = _sample_artifact(artifacts["face_alpha"], sample_indices, rgb=False) if "face_alpha" in artifacts else None
    face_uncertainty = _sample_artifact(artifacts["face_uncertainty"], sample_indices, rgb=False) if "face_uncertainty" in artifacts else None
    face_boundary = _sample_artifact(artifacts["face_boundary"], sample_indices, rgb=False) if "face_boundary" in artifacts else None
    hair_boundary = _sample_artifact(artifacts["hair_boundary"], sample_indices, rgb=False) if "hair_boundary" in artifacts else None
    hand_boundary = _sample_artifact(artifacts["hand_boundary"], sample_indices, rgb=False) if "hand_boundary" in artifacts else None
    cloth_boundary = _sample_artifact(artifacts["cloth_boundary"], sample_indices, rgb=False) if "cloth_boundary" in artifacts else None
    occluded_boundary = _sample_artifact(artifacts["occluded_boundary"], sample_indices, rgb=False) if "occluded_boundary" in artifacts else None

    face_bbox_curve = load_json_if_exists(src_root / "face_bbox_curve.json")
    face_pose = load_json_if_exists(src_root / "src_face_pose.json")
    face_expression = load_json_if_exists(src_root / "src_face_expression.json")

    face_maps = build_face_conditioning_maps(
        face_alpha=face_alpha,
        face_uncertainty=face_uncertainty,
        face_bbox_curve=face_bbox_curve,
        face_pose=face_pose,
        face_expression=face_expression,
    )
    boundary_maps = build_boundary_conditioning_maps(
        soft_alpha=soft_alpha,
        alpha_v2=alpha_v2,
        trimap_v2=trimap_v2,
        boundary_band=boundary_band,
        fine_boundary_mask=fine_boundary_mask,
        hair_edge_mask=hair_edge_mask,
        alpha_uncertainty_v2=alpha_uncertainty_v2,
        alpha_confidence=alpha_confidence,
        alpha_source_provenance=alpha_source_provenance,
        uncertainty_map=uncertainty_map,
        occlusion_band=occlusion_band,
    )
    structure_guard = summarize_reference_structure_guard(metadata)

    rows = []
    for local_index, source_index in enumerate(sample_indices):
        legacy_keep = compose_background_keep_mask(
            person_mask[local_index],
            soft_band=None,
            background_keep_prior=background_keep_prior[local_index],
            visible_support=visible_support[local_index],
            unresolved_region=unresolved_region[local_index],
            background_confidence=None,
            occlusion_band=occlusion_band[local_index] if occlusion_band is not None else None,
            uncertainty_map=uncertainty_map[local_index] if uncertainty_map is not None else None,
            face_preserve=face_maps["face_preserve_map"][local_index] if face_maps["face_preserve_map"] is not None else None,
            face_confidence=face_maps["face_confidence_map"][local_index] if face_maps["face_confidence_map"] is not None else None,
            conditioning_mode="decoupled_v1",
            mode="soft_band",
            boundary_strength=0.35,
            decoupled_release_strength=0.35,
        )
        v2_keep = compose_background_keep_mask(
            person_mask[local_index],
            soft_band=None,
            foreground_alpha=foreground_alpha[local_index],
            foreground_confidence=foreground_confidence[local_index],
            composite_roi_mask=composite_roi_mask[local_index],
            background_keep_prior=background_keep_prior[local_index],
            visible_support=visible_support[local_index],
            unresolved_region=unresolved_region[local_index],
            background_confidence=None,
            occlusion_band=occlusion_band[local_index] if occlusion_band is not None else None,
            uncertainty_map=uncertainty_map[local_index] if uncertainty_map is not None else None,
            face_preserve=face_maps["face_preserve_map"][local_index] if face_maps["face_preserve_map"] is not None else None,
            face_confidence=face_maps["face_confidence_map"][local_index] if face_maps["face_confidence_map"] is not None else None,
            face_boundary=face_boundary[local_index] if face_boundary is not None else None,
            hair_boundary=hair_boundary[local_index] if hair_boundary is not None else None,
            hand_boundary=hand_boundary[local_index] if hand_boundary is not None else None,
            cloth_boundary=cloth_boundary[local_index] if cloth_boundary is not None else None,
            occluded_boundary=occluded_boundary[local_index] if occluded_boundary is not None else None,
            conditioning_mode="decoupled_v2",
            mode="soft_band",
            boundary_strength=0.50,
            decoupled_release_strength=0.35,
        )
        v1_core = build_core_condition_rgb(
            background_rgb=background_rgb[local_index],
            foreground_rgb=foreground_rgb[local_index],
            foreground_alpha=foreground_alpha[local_index],
            foreground_confidence=foreground_confidence[local_index],
            soft_alpha=boundary_maps["conditioning_soft_alpha"][local_index] if boundary_maps["conditioning_soft_alpha"] is not None else None,
            trimap_unknown=trimap_unknown[local_index] if trimap_unknown is not None else None,
            hair_alpha=hair_alpha[local_index] if hair_alpha is not None else None,
            uncertainty_map=uncertainty_map[local_index] if uncertainty_map is not None else None,
            occlusion_band=occlusion_band[local_index] if occlusion_band is not None else None,
            face_preserve=face_maps["face_preserve_map"][local_index] if face_maps["face_preserve_map"] is not None else None,
            composite_roi_mask=composite_roi_mask[local_index],
            mode="core_rich_v1",
        )
        v2_core = build_core_condition_rgb(
            background_rgb=background_rgb[local_index],
            foreground_rgb=foreground_rgb[local_index],
            foreground_alpha=foreground_alpha[local_index],
            foreground_confidence=foreground_confidence[local_index],
            soft_alpha=boundary_maps["conditioning_soft_alpha"][local_index] if boundary_maps["conditioning_soft_alpha"] is not None else None,
            trimap_unknown=trimap_unknown[local_index] if trimap_unknown is not None else None,
            hair_alpha=hair_alpha[local_index] if hair_alpha is not None else None,
            uncertainty_map=uncertainty_map[local_index] if uncertainty_map is not None else None,
            occlusion_band=occlusion_band[local_index] if occlusion_band is not None else None,
            face_preserve=face_maps["face_preserve_map"][local_index] if face_maps["face_preserve_map"] is not None else None,
            composite_roi_mask=composite_roi_mask[local_index],
            mode="decoupled_v2",
        )
        rows.append({
            "frame_index": int(source_index),
            "legacy_keep_mean": float(torch.as_tensor(legacy_keep).mean().item()),
            "decoupled_v2_keep_mean": float(torch.as_tensor(v2_keep).mean().item()),
            "keep_delta_mean": float((torch.as_tensor(v2_keep) - torch.as_tensor(legacy_keep)).mean().item()),
            "core_rich_v1_fg_weight_mean": v1_core["summary"]["fg_weight_mean"],
            "decoupled_v2_fg_weight_mean": v2_core["summary"]["fg_weight_mean"],
            "core_fg_weight_delta_mean": float(v2_core["summary"]["fg_weight_mean"] - v1_core["summary"]["fg_weight_mean"]),
        })

    payload = {
        "status": "PASS",
        "frame_count": int(background_rgb.shape[0]),
        "face_maps_available": bool(face_maps["summary"]["available"]),
        "boundary_maps_available": bool(boundary_maps["summary"]["available"]),
        "structure_guard_strength": structure_guard["guard_strength"],
        "face_summary": face_maps["summary"],
        "boundary_summary": boundary_maps["summary"],
        "sample_rows": rows,
    }
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
