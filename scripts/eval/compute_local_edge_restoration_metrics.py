import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan.utils.animate_contract import load_preprocess_metadata, read_video_rgb
from wan.utils.boundary_refinement import compute_boundary_refinement_metrics, compute_boundary_roi_metrics
from wan.utils.media_io import load_mask_artifact, load_rgb_artifact


def _load_output_frames(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.is_dir():
        return load_rgb_artifact(path, "png_seq")
    return read_video_rgb(path)


def _masked_metrics(*, before, after, background, mask: np.ndarray) -> dict:
    mask = np.asarray(mask, dtype=np.float32)
    metrics = compute_boundary_refinement_metrics(
        before_frames=before,
        after_frames=after,
        background_frames=background,
        outer_band=mask,
        inner_band=np.zeros_like(mask, dtype=np.float32),
    )
    return {
        "coverage": float(mask.mean()),
        "gradient_before_mean": metrics["band_gradient_before_mean"],
        "gradient_after_mean": metrics["band_gradient_after_mean"],
        "contrast_before_mean": metrics["band_edge_contrast_before_mean"],
        "contrast_after_mean": metrics["band_edge_contrast_after_mean"],
        "halo_before": metrics["halo_ratio_before"],
        "halo_after": metrics["halo_ratio_after"],
        "mad_mean": metrics["band_mad_mean"],
    }


def main():
    parser = argparse.ArgumentParser(description="Compute optimization4 Step 07 local edge restoration metrics.")
    parser.add_argument("--before", type=str, required=True)
    parser.add_argument("--after", type=str, required=True)
    parser.add_argument("--src_root_path", type=str, required=True)
    parser.add_argument("--debug_dir", type=str, required=True)
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    metadata = load_preprocess_metadata(args.src_root_path)
    before = _load_output_frames(args.before).astype(np.float32) / 255.0
    after = _load_output_frames(args.after).astype(np.float32) / 255.0
    if before.shape != after.shape:
        raise ValueError(f"before and after outputs must match. Got {before.shape} vs {after.shape}.")

    artifacts = metadata["src_files"]
    background = load_rgb_artifact(
        Path(args.src_root_path) / artifacts["background"]["path"],
        artifacts["background"]["format"],
    ).astype(np.float32) / 255.0
    debug_dir = Path(args.debug_dir)
    outer_band = load_mask_artifact(debug_dir / "outer_band.mp4", "mp4").astype(np.float32)
    inner_band = load_mask_artifact(debug_dir / "inner_band.mp4", "mp4").astype(np.float32)
    roi_mask = load_mask_artifact(debug_dir / "roi_mask.mp4", "mp4").astype(np.float32)

    frame_count = min(len(before), len(after), len(background), len(outer_band), len(inner_band), len(roi_mask))
    before = before[:frame_count]
    after = after[:frame_count]
    background = background[:frame_count]
    outer_band = outer_band[:frame_count]
    inner_band = inner_band[:frame_count]
    roi_mask = roi_mask[:frame_count]

    roi_metrics = compute_boundary_roi_metrics(
        before_frames=before,
        after_frames=after,
        background_frames=background,
        outer_band=outer_band,
        inner_band=inner_band,
        roi_mask=roi_mask,
    )

    semantic_metrics = {}
    for key in ("face_boundary", "hair_boundary", "hand_boundary", "cloth_boundary", "occluded_boundary"):
        if key not in artifacts:
            continue
        mask = load_mask_artifact(
            Path(args.src_root_path) / artifacts[key]["path"],
            artifacts[key]["format"],
        ).astype(np.float32)[:frame_count]
        semantic_metrics[key] = _masked_metrics(
            before=before,
            after=after,
            background=background,
            mask=mask,
        )

    local_edge_focus = None
    local_edge_gain = None
    local_edge_feather = None
    if (debug_dir / "local_edge_focus.mp4").exists():
        local_edge_focus = load_mask_artifact(debug_dir / "local_edge_focus.mp4", "mp4").astype(np.float32)[:frame_count]
    if (debug_dir / "local_edge_gain.mp4").exists():
        local_edge_gain = load_mask_artifact(debug_dir / "local_edge_gain.mp4", "mp4").astype(np.float32)[:frame_count]
    if (debug_dir / "local_edge_feather.mp4").exists():
        local_edge_feather = load_mask_artifact(debug_dir / "local_edge_feather.mp4", "mp4").astype(np.float32)[:frame_count]

    result = {
        "frame_count": int(frame_count),
        "before_path": str(Path(args.before).resolve()),
        "after_path": str(Path(args.after).resolve()),
        "src_root_path": str(Path(args.src_root_path).resolve()),
        "debug_dir": str(debug_dir.resolve()),
        "roi_metrics": roi_metrics,
        "semantic_metrics": semantic_metrics,
        "local_edge_focus_mean": float(local_edge_focus.mean()) if local_edge_focus is not None else None,
        "local_edge_gain_mean": float(local_edge_gain.mean()) if local_edge_gain is not None else None,
        "local_edge_feather_mean": float(local_edge_feather.mean()) if local_edge_feather is not None else None,
    }
    payload = json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
