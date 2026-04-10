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
from wan.utils.rgba_foreground_proto import build_rgba_foreground


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


def _load_optional_mask(artifacts: dict, name: str):
    if name not in artifacts:
        return None
    return load_mask_artifact(artifacts[name]["path"], artifacts[name].get("format")).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply Optimization9 Step04 RGBA foreground proto to a decoupled preprocess bundle.")
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
    trimap_unknown = load_mask_artifact(artifacts["trimap_unknown"]["path"], artifacts["trimap_unknown"].get("format")).astype(np.float32)
    composite_roi_mask = load_mask_artifact(artifacts["composite_roi_mask"]["path"], artifacts["composite_roi_mask"].get("format")).astype(np.float32)
    foreground_confidence = load_mask_artifact(artifacts["foreground_confidence"]["path"], artifacts["foreground_confidence"].get("format")).astype(np.float32)
    background_rgb = load_rgb_artifact(artifacts["background_rgb"]["path"], artifacts["background_rgb"].get("format")).astype(np.uint8)
    background_visible_support = load_mask_artifact(artifacts["background_visible_support"]["path"], artifacts["background_visible_support"].get("format")).astype(np.float32)
    background_unresolved = load_mask_artifact(artifacts["background_unresolved"]["path"], artifacts["background_unresolved"].get("format")).astype(np.float32)
    hair_alpha = _load_optional_mask(artifacts, "hair_alpha")
    hair_boundary = _load_optional_mask(artifacts, "hair_boundary")
    hand_boundary = _load_optional_mask(artifacts, "hand_boundary")
    cloth_boundary = _load_optional_mask(artifacts, "cloth_boundary")
    uncertainty_map = _load_optional_mask(artifacts, "uncertainty_map")

    frame_count, height, width = foreground_alpha.shape
    source_rgb = _load_video_frames(Path(args.source_video_path).resolve(), frame_count, (height, width))

    rgba_roi_frames = []
    rgba_fg_rgb_frames = []
    rgba_fg_alpha_frames = []
    rgba_composite_frames = []
    rgba_boundary_frames = []
    rgba_trimap_frames = []
    rgba_person_frames = []

    for idx in range(frame_count):
        result = build_rgba_foreground(
            source_rgb=source_rgb[idx],
            foreground_rgb=foreground_rgb[idx],
            foreground_alpha=foreground_alpha[idx],
            soft_alpha=soft_alpha[idx],
            hard_foreground=hard_foreground[idx],
            boundary_band=boundary_band[idx],
            trimap_unknown=trimap_unknown[idx],
            composite_roi_mask=composite_roi_mask[idx],
            foreground_confidence=foreground_confidence[idx],
            background_rgb=background_rgb[idx],
            background_visible_support=background_visible_support[idx],
            background_unresolved=background_unresolved[idx],
            hair_alpha=hair_alpha[idx] if hair_alpha is not None else None,
            hair_boundary=hair_boundary[idx] if hair_boundary is not None else None,
            hand_boundary=hand_boundary[idx] if hand_boundary is not None else None,
            cloth_boundary=cloth_boundary[idx] if cloth_boundary is not None else None,
            uncertainty_map=uncertainty_map[idx] if uncertainty_map is not None else None,
            mode=args.mode,
        )
        rgba_roi_frames.append(result["rgba_roi_mask"])
        rgba_fg_rgb_frames.append(result["rgba_foreground_rgb"])
        rgba_fg_alpha_frames.append(result["rgba_foreground_alpha"])
        rgba_composite_frames.append(result["rgba_composite_rgb"])
        rgba_boundary_frames.append(result["rgba_boundary_band"])
        rgba_trimap_frames.append(result["rgba_trimap_unknown"])
        rgba_person_frames.append(result["rgba_person_mask"])

    rgba_roi_frames = np.stack(rgba_roi_frames).astype(np.float32)
    rgba_fg_rgb_frames = np.stack(rgba_fg_rgb_frames).astype(np.uint8)
    rgba_fg_alpha_frames = np.stack(rgba_fg_alpha_frames).astype(np.float32)
    rgba_composite_frames = np.stack(rgba_composite_frames).astype(np.uint8)
    rgba_boundary_frames = np.stack(rgba_boundary_frames).astype(np.float32)
    rgba_trimap_frames = np.stack(rgba_trimap_frames).astype(np.float32)
    rgba_person_frames = np.stack(rgba_person_frames).astype(np.float32)

    roi_info = write_person_mask_artifact(
        mask_frames=rgba_roi_frames,
        output_root=output_root,
        stem="src_rgba_roi_mask",
        artifact_format="npz",
        fps=fps,
        mask_semantics="rgba_roi_mask",
    )
    fg_rgb_info = write_rgb_artifact(
        frames=rgba_fg_rgb_frames,
        output_root=output_root,
        stem="src_rgba_foreground_rgb",
        artifact_format="npz",
        fps=fps,
    )
    fg_alpha_info = write_person_mask_artifact(
        mask_frames=rgba_fg_alpha_frames,
        output_root=output_root,
        stem="src_rgba_foreground_alpha",
        artifact_format="npz",
        fps=fps,
        mask_semantics=metadata["src_files"]["foreground_alpha"].get("mask_semantics", "foreground_alpha"),
    )
    soft_alpha_info = write_person_mask_artifact(
        mask_frames=rgba_fg_alpha_frames,
        output_root=output_root,
        stem="src_rgba_soft_alpha",
        artifact_format="npz",
        fps=fps,
        mask_semantics=metadata["src_files"]["soft_alpha"].get("mask_semantics", "soft_alpha"),
    )
    composite_info = write_rgb_artifact(
        frames=rgba_composite_frames,
        output_root=output_root,
        stem="src_rgba_composite_preview",
        artifact_format="png_seq",
        fps=fps,
    )
    boundary_info = write_person_mask_artifact(
        mask_frames=rgba_boundary_frames,
        output_root=output_root,
        stem="src_rgba_boundary_band",
        artifact_format="npz",
        fps=fps,
        mask_semantics=metadata["src_files"]["boundary_band"].get("mask_semantics", "boundary_band"),
    )
    trimap_info = write_person_mask_artifact(
        mask_frames=rgba_trimap_frames,
        output_root=output_root,
        stem="src_rgba_trimap_unknown",
        artifact_format="npz",
        fps=fps,
        mask_semantics=metadata["src_files"]["trimap_unknown"].get("mask_semantics", "trimap_unknown"),
    )
    person_info = write_person_mask_artifact(
        mask_frames=rgba_person_frames,
        output_root=output_root,
        stem="src_rgba_person_mask",
        artifact_format="npz",
        fps=fps,
        mask_semantics=metadata["src_files"]["person_mask"].get("mask_semantics", "person_foreground"),
    )

    metadata["src_files"]["foreground_rgb"] = fg_rgb_info
    metadata["src_files"]["foreground_alpha"] = fg_alpha_info
    metadata["src_files"]["soft_alpha"] = soft_alpha_info
    metadata["src_files"]["person_mask"] = person_info
    metadata["src_files"]["boundary_band"] = boundary_info
    metadata["src_files"]["trimap_unknown"] = trimap_info
    metadata["src_files"]["rgba_roi_mask"] = roi_info
    metadata["src_files"]["rgba_composite_preview"] = composite_info
    metadata["src_files"]["rgba_foreground_rgb"] = fg_rgb_info
    metadata["src_files"]["rgba_foreground_alpha"] = fg_alpha_info
    metadata.setdefault("processing", {})["rgba_foreground_proto"] = {
        "mode": args.mode,
        "rgba_roi_mask_artifact": roi_info,
        "rgba_foreground_rgb_artifact": fg_rgb_info,
        "rgba_foreground_alpha_artifact": fg_alpha_info,
        "rgba_soft_alpha_artifact": soft_alpha_info,
        "rgba_boundary_band_artifact": boundary_info,
        "rgba_trimap_unknown_artifact": trimap_info,
        "rgba_person_mask_artifact": person_info,
        "rgba_composite_preview_artifact": composite_info,
    }
    (output_root / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary = {
        "src_root_path": str(src_root),
        "output_root_path": str(output_root),
        "source_video_path": str(Path(args.source_video_path).resolve()),
        "mode": args.mode,
        "rgba_roi_mean": float(rgba_roi_frames.mean()),
        "rgba_alpha_mean": float(rgba_fg_alpha_frames.mean()),
        "rgba_boundary_mean": float(rgba_boundary_frames.mean()),
    }
    (output_root / "rgba_foreground_infer_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
