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
from wan.utils.media_io import load_mask_artifact, load_rgb_artifact, write_person_mask_artifact, write_rgb_artifact
from wan.utils.renderable_foreground_proto import build_renderable_foreground_frame


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


def _load_optional_mask(artifacts: dict, name: str):
    if name not in artifacts:
        return None
    return load_mask_artifact(artifacts[name]["path"], artifacts[name].get("format")).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply Optimization9 Step05 renderable foreground proto to a decoupled preprocess bundle.")
    parser.add_argument("--src_root_path", required=True)
    parser.add_argument("--source_video_path", required=True)
    parser.add_argument("--output_root_path", required=True)
    parser.add_argument("--mode", choices=("round1", "round2", "round3"), required=True)
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
    soft_alpha = load_mask_artifact(artifacts["soft_alpha"]["path"], artifacts["soft_alpha"].get("format")).astype(np.float32)
    hard_foreground = load_mask_artifact(artifacts["hard_foreground"]["path"], artifacts["hard_foreground"].get("format")).astype(np.float32)
    boundary_band = load_mask_artifact(artifacts["boundary_band"]["path"], artifacts["boundary_band"].get("format")).astype(np.float32)
    composite_roi_mask = load_mask_artifact(artifacts["composite_roi_mask"]["path"], artifacts["composite_roi_mask"].get("format")).astype(np.float32)
    foreground_confidence = load_mask_artifact(artifacts["foreground_confidence"]["path"], artifacts["foreground_confidence"].get("format")).astype(np.float32)
    occlusion_band = load_mask_artifact(artifacts["occlusion_band"]["path"], artifacts["occlusion_band"].get("format")).astype(np.float32)
    background_unresolved = load_mask_artifact(artifacts["background_unresolved"]["path"], artifacts["background_unresolved"].get("format")).astype(np.float32)
    background_rgb = load_rgb_artifact(artifacts["background_rgb"]["path"], artifacts["background_rgb"].get("format")).astype(np.uint8)
    uncertainty_map = _load_optional_mask(artifacts, "uncertainty_map")

    frame_count, height, width = foreground_alpha.shape
    source_rgb = _load_video_frames(Path(args.source_video_path).resolve(), frame_count, (height, width))

    render_rgb_frames = []
    render_alpha_frames = []
    render_depth_frames = []
    render_mask_frames = []
    render_silhouette_frames = []
    render_roi_frames = []
    composite_frames = []
    prev_state = None
    for idx in range(frame_count):
        result = build_renderable_foreground_frame(
            source_rgb=source_rgb[idx],
            background_rgb=background_rgb[idx],
            foreground_rgb=foreground_rgb[idx],
            foreground_alpha=foreground_alpha[idx],
            soft_alpha=soft_alpha[idx],
            hard_foreground=hard_foreground[idx],
            boundary_band=boundary_band[idx],
            composite_roi_mask=composite_roi_mask[idx],
            foreground_confidence=foreground_confidence[idx],
            occlusion_band=occlusion_band[idx],
            background_unresolved=background_unresolved[idx],
            uncertainty_map=uncertainty_map[idx] if uncertainty_map is not None else None,
            prev_state=prev_state,
            mode=args.mode,
        )
        render_rgb_frames.append(result["render_rgb"])
        render_alpha_frames.append(result["render_alpha"])
        render_depth_frames.append(result["render_depth"])
        render_mask_frames.append(result["render_person_mask"])
        render_silhouette_frames.append(result["render_silhouette_band"])
        render_roi_frames.append(result["render_roi_mask"])
        composite_frames.append(result["render_composite_rgb"])
        prev_state = result

    render_rgb_frames = np.stack(render_rgb_frames).astype(np.uint8)
    render_alpha_frames = np.stack(render_alpha_frames).astype(np.float32)
    render_depth_frames = np.stack(render_depth_frames).astype(np.float32)
    render_mask_frames = np.stack(render_mask_frames).astype(np.float32)
    render_silhouette_frames = np.stack(render_silhouette_frames).astype(np.float32)
    render_roi_frames = np.stack(render_roi_frames).astype(np.float32)
    composite_frames = np.stack(composite_frames).astype(np.uint8)

    rgb_info = write_rgb_artifact(frames=render_rgb_frames, output_root=output_root, stem="src_renderable_foreground_rgb", artifact_format="npz", fps=fps)
    alpha_info = write_person_mask_artifact(mask_frames=render_alpha_frames, output_root=output_root, stem="src_renderable_foreground_alpha", artifact_format="npz", fps=fps, mask_semantics=metadata["src_files"]["foreground_alpha"].get("mask_semantics", "foreground_alpha"))
    soft_info = write_person_mask_artifact(mask_frames=render_alpha_frames, output_root=output_root, stem="src_renderable_soft_alpha", artifact_format="npz", fps=fps, mask_semantics=metadata["src_files"]["soft_alpha"].get("mask_semantics", "soft_alpha"))
    depth_info = write_person_mask_artifact(mask_frames=render_depth_frames, output_root=output_root, stem="src_renderable_depth_like", artifact_format="npz", fps=fps, mask_semantics="renderable_depth_like")
    mask_info = write_person_mask_artifact(mask_frames=render_mask_frames, output_root=output_root, stem="src_renderable_person_mask", artifact_format="npz", fps=fps, mask_semantics=metadata["src_files"]["person_mask"].get("mask_semantics", "person_foreground"))
    silhouette_info = write_person_mask_artifact(
        mask_frames=render_silhouette_frames,
        output_root=output_root,
        stem="src_renderable_silhouette_band",
        artifact_format="npz",
        fps=fps,
        mask_semantics=metadata["src_files"]["boundary_band"].get("mask_semantics", "boundary_band"),
    )
    roi_info = write_person_mask_artifact(mask_frames=render_roi_frames, output_root=output_root, stem="src_renderable_roi_mask", artifact_format="npz", fps=fps, mask_semantics="renderable_roi_mask")
    composite_info = write_rgb_artifact(frames=composite_frames, output_root=output_root, stem="src_renderable_composite_preview", artifact_format="png_seq", fps=fps)

    metadata["src_files"]["foreground_rgb"] = rgb_info
    metadata["src_files"]["foreground_alpha"] = alpha_info
    metadata["src_files"]["soft_alpha"] = soft_info
    metadata["src_files"]["person_mask"] = mask_info
    metadata["src_files"]["boundary_band"] = silhouette_info
    metadata["src_files"]["renderable_depth_like"] = depth_info
    metadata["src_files"]["renderable_roi_mask"] = roi_info
    metadata["src_files"]["renderable_composite_preview"] = composite_info
    metadata.setdefault("processing", {})["renderable_foreground_proto"] = {
        "mode": args.mode,
        "renderable_foreground_rgb_artifact": rgb_info,
        "renderable_foreground_alpha_artifact": alpha_info,
        "renderable_soft_alpha_artifact": soft_info,
        "renderable_depth_like_artifact": depth_info,
        "renderable_person_mask_artifact": mask_info,
        "renderable_silhouette_band_artifact": silhouette_info,
        "renderable_roi_mask_artifact": roi_info,
        "renderable_composite_preview_artifact": composite_info,
    }
    (output_root / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {
        "src_root_path": str(src_root),
        "output_root_path": str(output_root),
        "source_video_path": str(Path(args.source_video_path).resolve()),
        "mode": args.mode,
        "render_alpha_mean": float(render_alpha_frames.mean()),
        "render_depth_mean": float(render_depth_frames.mean()),
        "render_roi_mean": float(render_roi_frames.mean()),
    }
    (output_root / "renderable_foreground_infer_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
