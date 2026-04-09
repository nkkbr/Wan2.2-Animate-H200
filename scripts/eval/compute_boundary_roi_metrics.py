import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan.utils.animate_contract import load_preprocess_metadata, read_video_rgb
from wan.utils.boundary_refinement import (
    build_boundary_roi_mask,
    build_inner_boundary_band,
    compute_boundary_roi_metrics,
)
from wan.utils.media_io import load_mask_artifact, load_rgb_artifact


def _load_output_frames(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.is_dir():
        return load_rgb_artifact(path, "png_seq")
    return read_video_rgb(path)


def main():
    parser = argparse.ArgumentParser(description="Compute boundary ROI refine metrics from before/after outputs.")
    parser.add_argument("--before", type=str, required=True)
    parser.add_argument("--after", type=str, required=True)
    parser.add_argument("--src_root_path", type=str, required=True)
    parser.add_argument(
        "--debug_dir",
        type=str,
        default=None,
        help="boundary_refinement debug directory for the refined run. If omitted, ROI bands are derived from preprocess artifacts.",
    )
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    metadata = load_preprocess_metadata(args.src_root_path)
    before = _load_output_frames(args.before)
    after = _load_output_frames(args.after)
    if before.shape != after.shape:
        raise ValueError(f"before and after outputs must match. Got {before.shape} vs {after.shape}.")

    artifacts = metadata["src_files"]
    background = load_rgb_artifact(
        Path(args.src_root_path) / artifacts["background"]["path"],
        artifacts["background"]["format"],
    )
    debug_dir = Path(args.debug_dir).resolve() if args.debug_dir else None
    if debug_dir is not None and all((debug_dir / name).exists() for name in ["outer_band.mp4", "inner_band.mp4", "roi_mask.mp4"]):
        outer_band = load_mask_artifact(debug_dir / "outer_band.mp4", "mp4")
        inner_band = load_mask_artifact(debug_dir / "inner_band.mp4", "mp4")
        roi_mask = load_mask_artifact(debug_dir / "roi_mask.mp4", "mp4")
        roi_source = "boundary_refinement_debug"
    else:
        person_mask = load_mask_artifact(
            Path(args.src_root_path) / artifacts["person_mask"]["path"],
            artifacts["person_mask"].get("format"),
        )
        outer_band = load_mask_artifact(
            Path(args.src_root_path) / artifacts["boundary_band"]["path"],
            artifacts["boundary_band"].get("format"),
        ).astype(np.float32)
        soft_alpha = None
        if "soft_alpha" in artifacts:
            soft_alpha = load_mask_artifact(
                Path(args.src_root_path) / artifacts["soft_alpha"]["path"],
                artifacts["soft_alpha"].get("format"),
            ).astype(np.float32)
        occlusion_band = None
        if "occlusion_band" in artifacts:
            occlusion_band = load_mask_artifact(
                Path(args.src_root_path) / artifacts["occlusion_band"]["path"],
                artifacts["occlusion_band"].get("format"),
            ).astype(np.float32)
        uncertainty_map = None
        if "uncertainty_map" in artifacts:
            uncertainty_map = load_mask_artifact(
                Path(args.src_root_path) / artifacts["uncertainty_map"]["path"],
                artifacts["uncertainty_map"].get("format"),
            ).astype(np.float32)
        inner_band = build_inner_boundary_band(person_mask.astype(np.float32), inner_width=3)
        roi_mask = build_boundary_roi_mask(
            person_mask=person_mask.astype(np.float32),
            outer_band=outer_band,
            inner_band=inner_band,
            soft_alpha=soft_alpha,
            occlusion_band=occlusion_band,
            uncertainty_map=uncertainty_map,
        ).astype(np.float32)
        roi_source = "preprocess_boundary_artifacts"

    frame_count = min(len(before), len(after), len(background), len(outer_band), len(inner_band), len(roi_mask))
    metrics = compute_boundary_roi_metrics(
        before_frames=before[:frame_count].astype(np.float32) / 255.0,
        after_frames=after[:frame_count].astype(np.float32) / 255.0,
        background_frames=background[:frame_count].astype(np.float32) / 255.0,
        outer_band=outer_band[:frame_count].astype(np.float32),
        inner_band=inner_band[:frame_count].astype(np.float32),
        roi_mask=roi_mask[:frame_count].astype(np.float32),
    )
    metrics.update({
        "frame_count": int(frame_count),
        "before_path": str(Path(args.before).resolve()),
        "after_path": str(Path(args.after).resolve()),
        "debug_dir": str(debug_dir) if debug_dir is not None else None,
        "src_root_path": str(Path(args.src_root_path).resolve()),
        "roi_source": roi_source,
    })
    payload = json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
