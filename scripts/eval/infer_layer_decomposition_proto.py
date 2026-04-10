#!/usr/bin/env python
import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan.utils.animate_contract import resolve_preprocess_artifacts
from wan.utils.layer_decomposition_proto import decompose_layers
from wan.utils.media_io import load_mask_artifact, load_rgb_artifact, write_person_mask_artifact, write_rgb_artifact


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply Optimization9 Step03 layer decomposition proto to a decoupled preprocess bundle.")
    parser.add_argument("--src_root_path", required=True)
    parser.add_argument("--source_video_path", required=True)
    parser.add_argument("--output_root_path", required=True)
    parser.add_argument("--mode", choices=("round1", "round2", "round3"), required=True)
    parser.add_argument("--occlusion_strength", type=float, default=0.65)
    parser.add_argument("--alpha_mix", type=float, default=0.40)
    parser.add_argument("--residual_mix", type=float, default=0.35)
    args = parser.parse_args()

    src_root = Path(args.src_root_path).resolve()
    output_root = Path(args.output_root_path).resolve()
    if output_root.exists():
        shutil.rmtree(output_root)
    shutil.copytree(src_root, output_root)

    artifacts, metadata = resolve_preprocess_artifacts(output_root, replace_flag=True)
    fps = float(metadata.get("fps") or 5.0)

    foreground_rgb = load_rgb_artifact(artifacts["foreground_rgb"]["path"], artifacts["foreground_rgb"].get("format")).astype(np.uint8)
    foreground_alpha = load_mask_artifact(artifacts["foreground_alpha"]["path"], artifacts["foreground_alpha"].get("format")).astype(np.float32)
    foreground_confidence = load_mask_artifact(artifacts["foreground_confidence"]["path"], artifacts["foreground_confidence"].get("format")).astype(np.float32)
    background_rgb = load_rgb_artifact(artifacts["background_rgb"]["path"], artifacts["background_rgb"].get("format")).astype(np.uint8)
    background_visible_support = load_mask_artifact(artifacts["background_visible_support"]["path"], artifacts["background_visible_support"].get("format")).astype(np.float32)
    background_unresolved = load_mask_artifact(artifacts["background_unresolved"]["path"], artifacts["background_unresolved"].get("format")).astype(np.float32)
    composite_roi_mask = load_mask_artifact(artifacts["composite_roi_mask"]["path"], artifacts["composite_roi_mask"].get("format")).astype(np.float32)
    occlusion_band = load_mask_artifact(artifacts["occlusion_band"]["path"], artifacts["occlusion_band"].get("format")).astype(np.float32)
    uncertainty_map = load_mask_artifact(artifacts["uncertainty_map"]["path"], artifacts["uncertainty_map"].get("format")).astype(np.float32) if "uncertainty_map" in artifacts else None
    occluded_boundary = load_mask_artifact(artifacts["occluded_boundary"]["path"], artifacts["occluded_boundary"].get("format")).astype(np.float32) if "occluded_boundary" in artifacts else None

    frame_count, height, width = foreground_alpha.shape
    source_rgb = _load_video_frames(Path(args.source_video_path).resolve(), frame_count, (height, width))

    layer_roi_frames = []
    occlusion_alpha_frames = []
    total_alpha_frames = []
    occlusion_rgb_frames = []
    composite_frames = []
    person_mask_frames = []
    trimap_unknown_frames = []

    for idx in range(frame_count):
        layer = decompose_layers(
            source_rgb=source_rgb[idx],
            foreground_rgb=foreground_rgb[idx],
            foreground_alpha=foreground_alpha[idx],
            foreground_confidence=foreground_confidence[idx],
            background_rgb=background_rgb[idx],
            background_visible_support=background_visible_support[idx],
            background_unresolved=background_unresolved[idx],
            composite_roi_mask=composite_roi_mask[idx],
            occlusion_band=occlusion_band[idx],
            occluded_boundary=occluded_boundary[idx] if occluded_boundary is not None else None,
            uncertainty_map=uncertainty_map[idx] if uncertainty_map is not None else None,
            mode=args.mode,
            occlusion_strength=float(args.occlusion_strength),
            alpha_mix=float(args.alpha_mix),
            residual_mix=float(args.residual_mix),
        )
        layer_roi_frames.append(layer["layer_roi_mask"])
        occlusion_alpha_frames.append(layer["occlusion_alpha"])
        total_alpha_frames.append(layer["total_alpha"])
        occlusion_rgb_frames.append(layer["occlusion_rgb"])
        composite_frames.append(layer["composite_rgb"])
        person_mask_frames.append(layer["person_mask"])
        trimap_unknown_frames.append(layer["trimap_unknown"])

    layer_roi_frames = np.stack(layer_roi_frames).astype(np.float32)
    occlusion_alpha_frames = np.stack(occlusion_alpha_frames).astype(np.float32)
    total_alpha_frames = np.stack(total_alpha_frames).astype(np.float32)
    occlusion_rgb_frames = np.stack(occlusion_rgb_frames).astype(np.uint8)
    composite_frames = np.stack(composite_frames).astype(np.uint8)
    person_mask_frames = np.stack(person_mask_frames).astype(np.float32)
    trimap_unknown_frames = np.stack(trimap_unknown_frames).astype(np.float32)

    layer_roi_info = write_person_mask_artifact(
        mask_frames=layer_roi_frames,
        output_root=output_root,
        stem="src_layer_roi_mask",
        artifact_format="npz",
        fps=fps,
        mask_semantics="layer_roi_mask",
    )
    occ_alpha_info = write_person_mask_artifact(
        mask_frames=occlusion_alpha_frames,
        output_root=output_root,
        stem="src_layer_occlusion_alpha",
        artifact_format="npz",
        fps=fps,
        mask_semantics="layer_occlusion_alpha",
    )
    occ_band_info = write_person_mask_artifact(
        mask_frames=occlusion_alpha_frames,
        output_root=output_root,
        stem="src_occlusion_band_layer_decomposition",
        artifact_format="npz",
        fps=fps,
        mask_semantics=metadata["src_files"]["occlusion_band"].get("mask_semantics", "occlusion_band"),
    )
    total_alpha_info = write_person_mask_artifact(
        mask_frames=total_alpha_frames,
        output_root=output_root,
        stem="src_soft_alpha_layer_decomposition",
        artifact_format="npz",
        fps=fps,
        mask_semantics=metadata["src_files"]["soft_alpha"].get("mask_semantics", "soft_alpha"),
    )
    person_info = write_person_mask_artifact(
        mask_frames=person_mask_frames,
        output_root=output_root,
        stem="src_mask_layer_decomposition",
        artifact_format="npz",
        fps=fps,
        mask_semantics=metadata["src_files"]["person_mask"].get("mask_semantics", "person_foreground"),
    )
    trimap_info = write_person_mask_artifact(
        mask_frames=trimap_unknown_frames,
        output_root=output_root,
        stem="src_trimap_unknown_layer_decomposition",
        artifact_format="npz",
        fps=fps,
        mask_semantics=metadata["src_files"].get("trimap_unknown", {}).get("mask_semantics", "trimap_unknown"),
    )
    occ_rgb_info = write_rgb_artifact(
        frames=occlusion_rgb_frames,
        output_root=output_root,
        stem="src_layer_occlusion_rgb",
        artifact_format="npz",
        fps=fps,
    )
    composite_info = write_rgb_artifact(
        frames=composite_frames,
        output_root=output_root,
        stem="src_layer_composite_preview",
        artifact_format="png_seq",
        fps=fps,
    )

    metadata["src_files"]["soft_alpha"] = total_alpha_info
    metadata["src_files"]["person_mask"] = person_info
    metadata["src_files"]["trimap_unknown"] = trimap_info
    metadata["src_files"]["occlusion_band"] = occ_band_info
    metadata["src_files"]["layer_roi_mask"] = layer_roi_info
    metadata["src_files"]["layer_occlusion_alpha"] = occ_alpha_info
    metadata["src_files"]["layer_occlusion_rgb"] = occ_rgb_info
    metadata["src_files"]["layer_composite_preview"] = composite_info
    metadata.setdefault("processing", {})["layer_decomposition_proto"] = {
        "mode": args.mode,
        "occlusion_strength": float(args.occlusion_strength),
        "alpha_mix": float(args.alpha_mix),
        "residual_mix": float(args.residual_mix),
        "layer_roi_mask_artifact": layer_roi_info,
        "layer_occlusion_alpha_artifact": occ_alpha_info,
        "layer_occlusion_band_artifact": occ_band_info,
        "layer_occlusion_rgb_artifact": occ_rgb_info,
        "layer_composite_preview_artifact": composite_info,
    }
    metadata_path = output_root / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary = {
        "src_root_path": str(src_root),
        "output_root_path": str(output_root),
        "source_video_path": str(Path(args.source_video_path).resolve()),
        "mode": args.mode,
        "occlusion_strength": float(args.occlusion_strength),
        "alpha_mix": float(args.alpha_mix),
        "residual_mix": float(args.residual_mix),
        "layer_roi_mean": float(layer_roi_frames.mean()),
        "occlusion_alpha_mean": float(occlusion_alpha_frames.mean()),
        "total_alpha_mean": float(total_alpha_frames.mean()),
    }
    (output_root / "layer_decomposition_infer_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
