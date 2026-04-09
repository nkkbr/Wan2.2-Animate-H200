#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
import sys

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan.utils.animate_contract import load_preprocess_metadata
from wan.utils.media_io import load_mask_artifact, load_rgb_artifact, write_person_mask_artifact
from wan.utils.trainable_edge_model import EdgeRefinementNet, split_outputs


def _resize_stack(frames: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.stack([cv2.resize(frame, size, interpolation=cv2.INTER_CUBIC) for frame in frames]).astype(np.float32)


def _resize_mask_stack(frames: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.stack([cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR) for frame in frames]).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Infer reviewed-data trainable edge model and write a derived preprocess bundle.")
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--baseline_preprocess_dir", required=True, type=str)
    parser.add_argument("--output_preprocess_dir", required=True, type=str)
    parser.add_argument("--reviewed_dataset_dir", type=str, default=None)
    args = parser.parse_args()

    baseline_dir = Path(args.baseline_preprocess_dir).resolve()
    output_dir = Path(args.output_preprocess_dir).resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.copytree(baseline_dir, output_dir)

    metadata = load_preprocess_metadata(baseline_dir)
    artifacts = metadata["src_files"]
    fps = float(metadata.get("fps") or 5.0)
    soft_alpha = load_mask_artifact(baseline_dir / artifacts["soft_alpha"]["path"], artifacts["soft_alpha"].get("format"))
    boundary_band = load_mask_artifact(baseline_dir / artifacts["boundary_band"]["path"], artifacts["boundary_band"].get("format"))
    background_rgb = load_rgb_artifact(baseline_dir / artifacts["background"]["path"], artifacts["background"].get("format")).astype(np.float32) / 255.0
    foreground_rgb_artifact = artifacts.get("foreground_rgb")
    foreground_alpha = load_mask_artifact(baseline_dir / artifacts["foreground_alpha"]["path"], artifacts["foreground_alpha"].get("format")) if "foreground_alpha" in artifacts else None
    if foreground_rgb_artifact is not None and foreground_alpha is not None:
        foreground_rgb = load_rgb_artifact(baseline_dir / foreground_rgb_artifact["path"], foreground_rgb_artifact.get("format")).astype(np.float32) / 255.0
        source_rgb = np.clip(foreground_rgb * foreground_alpha[..., None] + background_rgb * (1.0 - foreground_alpha[..., None]), 0.0, 1.0)
    else:
        source_rgb = background_rgb.copy()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    size = tuple(checkpoint["size"])
    model = EdgeRefinementNet()
    model.load_state_dict(checkpoint["model"])
    model.eval()

    resized_rgb = _resize_stack(source_rgb, size)
    resized_alpha = _resize_mask_stack(soft_alpha, size)
    resized_boundary = _resize_mask_stack(boundary_band, size)
    frame_indices = np.arange(resized_rgb.shape[0], dtype=int)
    if args.reviewed_dataset_dir:
        summary = json.loads((Path(args.reviewed_dataset_dir).resolve() / "summary.json").read_text(encoding="utf-8"))
        reviewed_indices: list[int] = []
        for case in summary.get("cases", []):
            for record in case.get("records", []):
                reviewed_indices.append(int(record["preprocess_frame_index"]))
        frame_indices = np.asarray(sorted(set(reviewed_indices)), dtype=int)
        if frame_indices.size == 0:
            raise RuntimeError(f"No reviewed frame indices found in dataset: {args.reviewed_dataset_dir}")

    inputs = np.concatenate([
        resized_rgb[frame_indices],
        resized_alpha[frame_indices, ..., None],
        resized_boundary[frame_indices, ..., None],
    ], axis=-1)
    inputs = torch.from_numpy(inputs).permute(0, 3, 1, 2).float()

    with torch.no_grad():
        outputs = split_outputs(model(inputs))
    pred_alpha = outputs.alpha.squeeze(1).cpu().numpy().astype(np.float32)
    pred_boundary = outputs.boundary.squeeze(1).cpu().numpy().astype(np.float32)

    target_size = (int(metadata["width"]), int(metadata["height"]))
    pred_alpha_full = soft_alpha.copy()
    pred_boundary_full = boundary_band.copy()
    pred_alpha_resized = np.stack([cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR) for frame in pred_alpha]).astype(np.float32)
    pred_boundary_resized = np.stack([cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR) for frame in pred_boundary]).astype(np.float32)
    pred_alpha_full[frame_indices] = pred_alpha_resized
    pred_boundary_full[frame_indices] = pred_boundary_resized

    soft_alpha_artifact = write_person_mask_artifact(
        mask_frames=np.clip(pred_alpha_full, 0.0, 1.0),
        output_root=output_dir,
        stem="src_soft_alpha",
        artifact_format="npz",
        fps=fps,
        mask_semantics="soft_alpha",
    )
    boundary_artifact = write_person_mask_artifact(
        mask_frames=np.clip(pred_boundary_full, 0.0, 1.0),
        output_root=output_dir,
        stem="src_boundary_band",
        artifact_format="npz",
        fps=fps,
        mask_semantics="boundary_transition_band",
    )
    metadata["src_files"]["soft_alpha"] = soft_alpha_artifact
    metadata["src_files"]["boundary_band"] = boundary_artifact
    processing = metadata.setdefault("processing", {})
    processing["trainable_edge_model"] = {
        "enabled": True,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "output_preprocess_dir": str(output_dir),
        "reviewed_dataset_dir": str(Path(args.reviewed_dataset_dir).resolve()) if args.reviewed_dataset_dir else None,
        "reviewed_frame_count": int(frame_indices.size),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output_preprocess_dir": str(output_dir),
        "soft_alpha_path": soft_alpha_artifact["path"],
        "boundary_band_path": boundary_artifact["path"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
