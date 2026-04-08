import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan.utils.animate_contract import load_preprocess_metadata, read_video_rgb
from wan.utils.boundary_refinement import build_inner_boundary_band, compute_boundary_refinement_metrics
from wan.utils.media_io import load_mask_artifact, load_person_mask_artifact, load_rgb_artifact


SEMANTIC_KEYS = (
    "face_boundary",
    "hair_boundary",
    "hand_boundary",
    "cloth_boundary",
    "occluded_boundary",
)


def _load_output_frames(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.is_dir():
        from wan.utils.media_io import load_rgb_artifact
        return load_rgb_artifact(path, "png_seq")
    return read_video_rgb(path)


def _safe_pct(after: float, before: float) -> float:
    if before is None or abs(before) < 1e-9:
        return 0.0
    return float((after / before - 1.0) * 100.0)


def main():
    parser = argparse.ArgumentParser(description="Compute semantic boundary metrics from before/after outputs.")
    parser.add_argument("--before", type=str, required=True)
    parser.add_argument("--after", type=str, required=True)
    parser.add_argument("--src_root_path", type=str, required=True)
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    src_root = Path(args.src_root_path)
    metadata = load_preprocess_metadata(src_root)
    before = _load_output_frames(args.before)
    after = _load_output_frames(args.after)
    if before.shape != after.shape:
        raise ValueError(f"before and after outputs must match. Got {before.shape} vs {after.shape}.")

    artifacts = metadata["src_files"]
    background = load_rgb_artifact(src_root / artifacts["background"]["path"], artifacts["background"]["format"])
    person_mask = load_person_mask_artifact(src_root / artifacts["person_mask"]["path"], artifacts["person_mask"]["format"])
    boundary_band = load_mask_artifact(src_root / artifacts["boundary_band"]["path"], artifacts["boundary_band"]["format"])
    inner_band = build_inner_boundary_band(person_mask, inner_width=2)

    frame_count = min(len(before), len(after), len(background), len(person_mask), len(boundary_band), len(inner_band))
    before_f = before[:frame_count].astype(np.float32) / 255.0
    after_f = after[:frame_count].astype(np.float32) / 255.0
    background_f = background[:frame_count].astype(np.float32) / 255.0
    boundary_band = boundary_band[:frame_count].astype(np.float32)
    inner_band = inner_band[:frame_count].astype(np.float32)

    per_class = {}
    for key in SEMANTIC_KEYS:
        if key not in artifacts:
            continue
        semantic_mask = load_mask_artifact(src_root / artifacts[key]["path"], artifacts[key]["format"])[:frame_count].astype(np.float32)
        outer = np.clip(boundary_band * semantic_mask, 0.0, 1.0)
        inner = np.clip(inner_band * semantic_mask, 0.0, 1.0)
        metrics = compute_boundary_refinement_metrics(
            before_frames=before_f,
            after_frames=after_f,
            background_frames=background_f,
            outer_band=outer,
            inner_band=inner,
        )
        metrics.update({
            "coverage_mean": float(semantic_mask.mean()),
            "outer_coverage_mean": float(outer.mean()),
            "inner_coverage_mean": float(inner.mean()),
            "gradient_gain_pct": _safe_pct(metrics["band_gradient_after_mean"], metrics["band_gradient_before_mean"]),
            "contrast_gain_pct": _safe_pct(metrics["band_edge_contrast_after_mean"], metrics["band_edge_contrast_before_mean"]),
            "halo_reduction_pct": 0.0 if metrics["halo_ratio_before"] <= 1e-9 else float(
                (metrics["halo_ratio_before"] - metrics["halo_ratio_after"]) / metrics["halo_ratio_before"] * 100.0
            ),
        })
        per_class[key] = metrics

    payload = {
        "frame_count": int(frame_count),
        "before_path": str(Path(args.before).resolve()),
        "after_path": str(Path(args.after).resolve()),
        "src_root_path": str(src_root.resolve()),
        "per_class": per_class,
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
