import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan.utils.animate_contract import load_preprocess_metadata, read_video_rgb
from wan.utils.boundary_refinement import compute_boundary_refinement_metrics
from wan.utils.media_io import load_mask_artifact, load_person_mask_artifact, load_rgb_artifact


def _load_output_frames(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.is_dir():
        return load_rgb_artifact(path, "png_seq")
    return read_video_rgb(path)


def main():
    parser = argparse.ArgumentParser(description="Compute pixel-domain boundary refinement metrics from before/after outputs.")
    parser.add_argument("--before", type=str, required=True, help="Baseline output video path or png sequence directory.")
    parser.add_argument("--after", type=str, required=True, help="Refined output video path or png sequence directory.")
    parser.add_argument("--src_root_path", type=str, required=True, help="Preprocess bundle root used for the output.")
    parser.add_argument("--output_json", type=str, default=None, help="Optional path to save the metrics JSON.")
    args = parser.parse_args()

    metadata = load_preprocess_metadata(args.src_root_path)
    before = _load_output_frames(args.before)
    after = _load_output_frames(args.after)
    if before.shape != after.shape:
        raise ValueError(f"before and after outputs must match. Got {before.shape} vs {after.shape}.")

    artifacts = metadata["src_files"]
    person_mask = load_person_mask_artifact(
        Path(args.src_root_path) / artifacts["person_mask"]["path"],
        artifacts["person_mask"]["format"],
    )
    soft_band = None
    if "soft_band" in artifacts:
        soft_band = load_mask_artifact(
            Path(args.src_root_path) / artifacts["soft_band"]["path"],
            artifacts["soft_band"]["format"],
        )
    background = load_rgb_artifact(
        Path(args.src_root_path) / artifacts["background"]["path"],
        artifacts["background"]["format"],
    )

    frame_count = min(len(before), len(person_mask), len(background))
    if soft_band is not None:
        frame_count = min(frame_count, len(soft_band))

    if soft_band is None:
        soft_band = np.zeros_like(person_mask[:frame_count], dtype=np.float32)

    metrics = compute_boundary_refinement_metrics(
        before_frames=before[:frame_count].astype(np.float32) / 255.0,
        after_frames=after[:frame_count].astype(np.float32) / 255.0,
        background_frames=background[:frame_count].astype(np.float32) / 255.0,
        outer_band=soft_band[:frame_count].astype(np.float32),
        inner_band=np.zeros_like(person_mask[:frame_count], dtype=np.float32),
    )
    metrics.update({
        "frame_count": int(frame_count),
        "before_path": str(Path(args.before).resolve()),
        "after_path": str(Path(args.after).resolve()),
        "src_root_path": str(Path(args.src_root_path).resolve()),
    })

    payload = json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True)
    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
