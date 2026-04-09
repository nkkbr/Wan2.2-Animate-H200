#!/usr/bin/env python
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan.utils.animate_contract import resolve_preprocess_artifacts, read_video_rgb
from wan.utils.boundary_refinement import refine_boundary_frames, write_boundary_refinement_debug_artifacts
from wan.utils.media_io import load_mask_artifact, load_rgb_artifact, write_output_frames
from wan.utils.rich_conditioning import build_boundary_conditioning_maps, build_face_conditioning_maps, load_json_if_exists


RUNS_ROOT = REPO_ROOT / "runs"
REPLACEMENT_METRICS_SCRIPT = REPO_ROOT / "scripts" / "eval" / "compute_replacement_metrics.py"
ROI_METRICS_SCRIPT = REPO_ROOT / "scripts" / "eval" / "compute_boundary_roi_metrics.py"

DEFAULT_SRC_ROOT = str(REPO_ROOT / "runs" / "optimization5_step03_round1_preprocess" / "preprocess")
DEFAULT_BEFORE_VIDEO = str(REPO_ROOT / "runs" / "optimization5_step03_round1_ab" / "legacy" / "outputs" / "legacy.mkv")


def _default_python_bin() -> str:
    override = os.environ.get("WAN_PYTHON")
    for candidate in [override, "/home/user1/miniconda3/envs/wan/bin/python", sys.executable]:
        if candidate and Path(candidate).exists():
            return str(Path(candidate).resolve())
    return sys.executable


PYTHON = _default_python_bin()


def _run(command: list[str], *, cwd: Path = REPO_ROOT):
    return subprocess.run(command, cwd=str(cwd), text=True, capture_output=True)


def _write_log(directory: Path, stem: str, result: subprocess.CompletedProcess):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{stem}.stdout.log").write_text(result.stdout, encoding="utf-8")
    (directory / f"{stem}.stderr.log").write_text(result.stderr, encoding="utf-8")


def _read_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_optional_mask(artifacts: dict, root: Path, name: str):
    artifact = artifacts.get(name)
    if not artifact:
        return None
    return load_mask_artifact(root / artifact["path"], artifact.get("format"))


def _load_optional_rgb(artifacts: dict, root: Path, name: str):
    artifact = artifacts.get(name)
    if not artifact:
        return None
    return load_rgb_artifact(root / artifact["path"], artifact.get("format"))


def _build_face_maps(src_root: Path, artifacts: dict):
    face_alpha = _load_optional_mask(artifacts, src_root, "face_alpha")
    face_uncertainty = _load_optional_mask(artifacts, src_root, "face_uncertainty")
    face_bbox_curve = load_json_if_exists(src_root / "face_bbox_curve.json")
    face_pose = load_json_if_exists(src_root / "src_face_pose.json")
    face_expression = load_json_if_exists(src_root / "src_face_expression.json")
    return build_face_conditioning_maps(
        face_alpha=face_alpha,
        face_uncertainty=face_uncertainty,
        face_bbox_curve=face_bbox_curve,
        face_pose=face_pose,
        face_expression=face_expression,
    )


def _build_boundary_maps(src_root: Path, artifacts: dict):
    return build_boundary_conditioning_maps(
        soft_alpha=_load_optional_mask(artifacts, src_root, "soft_alpha"),
        alpha_v2=_load_optional_mask(artifacts, src_root, "alpha_v2"),
        trimap_v2=_load_optional_mask(artifacts, src_root, "trimap_v2"),
        boundary_band=_load_optional_mask(artifacts, src_root, "boundary_band"),
        fine_boundary_mask=_load_optional_mask(artifacts, src_root, "fine_boundary_mask"),
        hair_edge_mask=_load_optional_mask(artifacts, src_root, "hair_edge_mask"),
        alpha_uncertainty_v2=_load_optional_mask(artifacts, src_root, "alpha_uncertainty_v2"),
        alpha_confidence=_load_optional_mask(artifacts, src_root, "alpha_confidence_v2"),
        alpha_source_provenance=_load_optional_mask(artifacts, src_root, "alpha_source_provenance_v2"),
        uncertainty_map=_load_optional_mask(artifacts, src_root, "uncertainty_map"),
        occlusion_band=_load_optional_mask(artifacts, src_root, "occlusion_band"),
    )


def main():
    parser = argparse.ArgumentParser(description="Run Optimization6 Step04 boundary ROI generative reconstruction benchmark.")
    parser.add_argument("--suite_name", type=str, default=None)
    parser.add_argument("--src_root_path", type=str, default=DEFAULT_SRC_ROOT)
    parser.add_argument("--before_video", type=str, default=DEFAULT_BEFORE_VIDEO)
    parser.add_argument("--clip_len", type=int, default=45)
    parser.add_argument("--refert_num", type=int, default=5)
    parser.add_argument("--max_frames", type=int, default=9)
    parser.add_argument("--sample_evenly", action="store_true")
    parser.add_argument("--boundary_refine_strength", type=float, default=0.38)
    parser.add_argument("--boundary_refine_sharpen", type=float, default=0.20)
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suite_name = args.suite_name or f"optimization6_step04_ab_{timestamp}"
    suite_dir = RUNS_ROOT / suite_name
    suite_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = suite_dir / "logs"

    src_root = Path(args.src_root_path).resolve()
    artifacts, metadata = resolve_preprocess_artifacts(src_root, replace_flag=True)
    fps = float(metadata.get("fps") or metadata.get("runtime", {}).get("fps") or 5.0)

    before_video = Path(args.before_video).resolve()
    before_frames = read_video_rgb(before_video)
    background_artifact = artifacts["background"]
    background_frames = load_rgb_artifact(src_root / background_artifact["path"], background_artifact.get("format"))
    person_mask = load_mask_artifact(src_root / artifacts["person_mask"]["path"], artifacts["person_mask"].get("format"))

    total_frame_count = min(len(before_frames), len(background_frames), len(person_mask))
    frame_count = total_frame_count
    if args.max_frames > 0:
        frame_count = min(total_frame_count, int(args.max_frames))
    if args.sample_evenly and frame_count > 0:
        if frame_count == 1:
            sample_indices = np.asarray([total_frame_count // 2], dtype=int)
        else:
            sample_indices = np.linspace(0, total_frame_count - 1, num=frame_count, dtype=int)
        before_frames = before_frames[sample_indices]
        background_frames = background_frames[sample_indices]
        person_mask = person_mask[sample_indices]
    else:
        before_frames = before_frames[:frame_count]
        background_frames = background_frames[:frame_count]
        person_mask = person_mask[:frame_count]

    face_maps = _build_face_maps(src_root, artifacts)
    boundary_maps = _build_boundary_maps(src_root, artifacts)

    baseline_dir = suite_dir / "none" / "outputs"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    before_sampled_path = write_output_frames(before_frames, baseline_dir / "none.mkv", fps=fps, output_format="ffv1")

    start = time.perf_counter()
    after_frames, debug_data = refine_boundary_frames(
        generated_frames=before_frames,
        background_frames=background_frames,
        person_mask=person_mask,
        soft_band=_load_optional_mask(artifacts, src_root, "boundary_band")[:frame_count]
        if "boundary_band" in artifacts else _load_optional_mask(artifacts, src_root, "soft_band")[:frame_count],
        soft_alpha=boundary_maps["conditioning_soft_alpha"][:frame_count] if boundary_maps["conditioning_soft_alpha"] is not None else None,
        background_confidence=_load_optional_mask(artifacts, src_root, "background_confidence")[:frame_count] if "background_confidence" in artifacts else None,
        uncertainty_map=_load_optional_mask(artifacts, src_root, "uncertainty_map")[:frame_count] if "uncertainty_map" in artifacts else None,
        occlusion_band=_load_optional_mask(artifacts, src_root, "occlusion_band")[:frame_count] if "occlusion_band" in artifacts else None,
        face_preserve_map=face_maps["face_preserve_map"][:frame_count] if face_maps["face_preserve_map"] is not None else None,
        face_confidence_map=face_maps["face_confidence_map"][:frame_count] if face_maps["face_confidence_map"] is not None else None,
        detail_release_map=boundary_maps["detail_release_map"][:frame_count] if boundary_maps["detail_release_map"] is not None else None,
        trimap_unknown_map=boundary_maps["trimap_unknown_map"][:frame_count] if boundary_maps["trimap_unknown_map"] is not None else None,
        edge_detail_map=boundary_maps["edge_detail_map"][:frame_count] if boundary_maps["edge_detail_map"] is not None else None,
        face_boundary_map=_load_optional_mask(artifacts, src_root, "face_boundary")[:frame_count] if "face_boundary" in artifacts else None,
        hair_boundary_map=_load_optional_mask(artifacts, src_root, "hair_boundary")[:frame_count] if "hair_boundary" in artifacts else None,
        hand_boundary_map=_load_optional_mask(artifacts, src_root, "hand_boundary")[:frame_count] if "hand_boundary" in artifacts else None,
        cloth_boundary_map=_load_optional_mask(artifacts, src_root, "cloth_boundary")[:frame_count] if "cloth_boundary" in artifacts else None,
        occluded_boundary_map=_load_optional_mask(artifacts, src_root, "occluded_boundary")[:frame_count] if "occluded_boundary" in artifacts else None,
        structure_guard_strength=1.0,
        mode="roi_gen_v2",
        strength=args.boundary_refine_strength,
        sharpen=args.boundary_refine_sharpen,
    )
    runtime_sec = time.perf_counter() - start

    outputs_dir = suite_dir / "roi_gen_v2" / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    after_path = write_output_frames(after_frames, outputs_dir / "roi_gen_v2.mkv", fps=fps, output_format="ffv1")
    debug_paths = write_boundary_refinement_debug_artifacts(
        save_debug_dir=suite_dir / "roi_gen_v2" / "debug" / "generate",
        fps=fps,
        generated_frames=before_frames,
        refined_frames=after_frames,
        background_frames=background_frames,
        person_mask=person_mask,
        debug_data=debug_data,
    )

    repl_before_path = suite_dir / "none_replacement_metrics.json"
    repl_after_path = suite_dir / "roi_gen_v2_replacement_metrics.json"
    for stem, video_path, output_json in (
        ("none", Path(before_sampled_path), repl_before_path),
        ("roi_gen_v2", Path(after_path), repl_after_path),
    ):
        result = _run([
            PYTHON,
            str(REPLACEMENT_METRICS_SCRIPT),
            "--video_path", str(video_path),
            "--mask_path", str(src_root / artifacts["person_mask"]["path"]),
            "--clip_len", str(min(args.clip_len, frame_count)),
            "--refert_num", str(args.refert_num),
            "--output_json", str(output_json),
        ])
        _write_log(logs_dir, f"{stem}_replacement_metrics", result)

    roi_metrics_path = suite_dir / "roi_gen_v2_vs_none_roi_metrics.json"
    roi_metrics = _run([
        PYTHON,
        str(ROI_METRICS_SCRIPT),
        "--before", str(before_sampled_path),
        "--after", str(after_path),
        "--src_root_path", str(src_root),
        "--debug_dir", str(Path(debug_paths["metrics"]).parent),
        "--output_json", str(roi_metrics_path),
    ])
    _write_log(logs_dir, "roi_gen_v2_vs_none_roi_metrics", roi_metrics)

    summary = {
        "suite_dir": str(suite_dir.resolve()),
        "src_root_path": str(src_root),
        "rows": [
            {
                "boundary_refine_mode": "none",
                "output_video": str(Path(before_sampled_path).resolve()),
                "replacement_metrics": _read_json(repl_before_path),
                "seam_score_mean": ((_read_json(repl_before_path) or {}).get("seam_score") or {}).get("mean"),
                "background_fluctuation_mean": ((_read_json(repl_before_path) or {}).get("background_fluctuation") or {}).get("mean"),
            },
            {
                "boundary_refine_mode": "roi_gen_v2",
                "output_video": str(Path(after_path).resolve()),
                "replacement_metrics": _read_json(repl_after_path),
                "seam_score_mean": ((_read_json(repl_after_path) or {}).get("seam_score") or {}).get("mean"),
                "background_fluctuation_mean": ((_read_json(repl_after_path) or {}).get("background_fluctuation") or {}).get("mean"),
                "runtime_sec": runtime_sec,
                "debug_paths": debug_paths,
                "metrics": debug_data.get("metrics"),
            },
        ],
        "roi_metrics": _read_json(roi_metrics_path),
    }
    summary_json = suite_dir / "summary.json"
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"suite_dir": str(suite_dir.resolve()), "summary_json": str(summary_json.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
