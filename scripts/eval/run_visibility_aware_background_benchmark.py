#!/usr/bin/env python
import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = REPO_ROOT / "runs"
PREPROCESS_SCRIPT = REPO_ROOT / "wan" / "modules" / "animate" / "preprocess" / "preprocess_data.py"
CONTRACT_SCRIPT = REPO_ROOT / "scripts" / "eval" / "check_animate_contract.py"
BACKGROUND_METRICS_SCRIPT = REPO_ROOT / "scripts" / "eval" / "compute_background_precision_metrics.py"

DEFAULT_VIDEO_PATH = "/home/user1/wan2.2/test1/260306_01_step_1_0000_0010.mp4"
DEFAULT_REFERENCE_PATH = "/home/user1/wan2.2/test1/Gemini_Generated_Image_2.png"
DEFAULT_CKPT_PATH = "/home/user1/wan2.2/Wan2.2/Wan2.2-Animate-14B/process_checkpoint"


def _default_python_bin() -> str:
    override = os.environ.get("WAN_PYTHON")
    candidates = [override, "/home/user1/miniconda3/envs/wan/bin/python", sys.executable]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(Path(candidate).resolve())
    return sys.executable


PYTHON = _default_python_bin()


def _read_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _run(command: list[str], *, cwd: Path = REPO_ROOT):
    return subprocess.run(command, cwd=str(cwd), text=True, capture_output=True)


def _write_log(directory: Path, stem: str, result: subprocess.CompletedProcess):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{stem}.stdout.log").write_text(result.stdout, encoding="utf-8")
    (directory / f"{stem}.stderr.log").write_text(result.stderr, encoding="utf-8")


def _build_preprocess_command(args, run_dir: Path, mode: str):
    command = [
        PYTHON,
        "-X",
        "faulthandler",
        str(PREPROCESS_SCRIPT),
        "--ckpt_path",
        args.ckpt_path,
        "--video_path",
        args.video_path,
        "--refer_path",
        args.reference_path,
        "--run_dir",
        str(run_dir),
        "--save_manifest",
        "--replace_flag",
        "--lossless_intermediate",
        "--resolution_area",
        str(args.resolution_area[0]),
        str(args.resolution_area[1]),
        "--fps",
        str(args.fps),
        "--soft_mask_mode",
        "soft_band",
        "--boundary_fusion_mode",
        "v2",
        "--reference_normalization_mode",
        "structure_match",
        "--preprocess_runtime_profile",
        args.preprocess_runtime_profile,
        "--sam_runtime_profile",
        "h200_safe" if args.preprocess_runtime_profile == "h200_extreme" else args.preprocess_runtime_profile,
        "--multistage_preprocess_mode",
        args.multistage_preprocess_mode,
        "--bg_inpaint_mode",
        mode,
        "--bg_temporal_smooth_strength",
        str(args.bg_temporal_smooth_strength),
        "--bg_video_window_radius",
        str(args.bg_video_window_radius),
        "--bg_video_min_visible_count",
        str(args.bg_video_min_visible_count),
        "--bg_video_blend_strength",
        str(args.bg_video_blend_strength),
        "--bg_video_global_min_visible_count",
        str(args.bg_video_global_min_visible_count),
        "--bg_video_confidence_threshold",
        str(args.bg_video_confidence_threshold),
        "--bg_video_global_blend_strength",
        str(args.bg_video_global_blend_strength),
        "--bg_video_consistency_scale",
        str(args.bg_video_consistency_scale),
        "--sam_chunk_len",
        str(args.sam_chunk_len),
        "--sam_keyframes_per_chunk",
        str(args.sam_keyframes_per_chunk),
        "--sam_reprompt_interval",
        "0",
        "--no-sam_apply_postprocessing",
        "--no-sam_use_negative_points",
        "--sam_prompt_mode",
        "mask_seed",
        "--sam_prompt_body_conf_thresh",
        "0.999",
        "--sam_prompt_face_conf_thresh",
        "0.999",
        "--sam_prompt_hand_conf_thresh",
        "0.999",
        "--sam_prompt_face_min_points",
        "100",
        "--sam_prompt_hand_min_points",
        "100",
        "--iterations",
        "2",
        "--k",
        "7",
        "--w_len",
        "4",
        "--h_len",
        "8",
    ]
    return command


def main():
    parser = argparse.ArgumentParser(description="Run image/video/video_v2 clean-plate background benchmark.")
    parser.add_argument("--suite_name", type=str, default=None)
    parser.add_argument("--video_path", type=str, default=DEFAULT_VIDEO_PATH)
    parser.add_argument("--reference_path", type=str, default=DEFAULT_REFERENCE_PATH)
    parser.add_argument("--ckpt_path", type=str, default=DEFAULT_CKPT_PATH)
    parser.add_argument("--resolution_area", nargs=2, type=int, default=[640, 360])
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--preprocess_runtime_profile", choices=["h200_safe", "h200_aggressive", "h200_extreme"], default="h200_extreme")
    parser.add_argument("--multistage_preprocess_mode", choices=["none", "h200_extreme"], default="h200_extreme")
    parser.add_argument("--sam_chunk_len", type=int, default=12)
    parser.add_argument("--sam_keyframes_per_chunk", type=int, default=3)
    parser.add_argument("--bg_temporal_smooth_strength", type=float, default=0.14)
    parser.add_argument("--bg_video_window_radius", type=int, default=4)
    parser.add_argument("--bg_video_min_visible_count", type=int, default=2)
    parser.add_argument("--bg_video_blend_strength", type=float, default=0.7)
    parser.add_argument("--bg_video_global_min_visible_count", type=int, default=3)
    parser.add_argument("--bg_video_confidence_threshold", type=float, default=0.30)
    parser.add_argument("--bg_video_global_blend_strength", type=float, default=0.95)
    parser.add_argument("--bg_video_consistency_scale", type=float, default=18.0)
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suite_name = args.suite_name or f"optimization3_step06_background_ab_{timestamp}"
    suite_dir = RUNS_ROOT / suite_name
    suite_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    logs_dir = suite_dir / "logs"
    for mode in ("image", "video", "video_v2"):
        run_dir = suite_dir / f"preprocess_{mode}"
        preprocess = _run(_build_preprocess_command(args, run_dir, mode))
        _write_log(logs_dir, f"{mode}_preprocess", preprocess)

        row = {
            "mode": mode,
            "run_dir": str(run_dir.resolve()),
            "preprocess_returncode": int(preprocess.returncode),
        }
        preprocess_dir = run_dir / "preprocess"
        if preprocess.returncode == 0 and preprocess_dir.exists():
            contract = _run([
                PYTHON,
                str(CONTRACT_SCRIPT),
                "--src_root_path",
                str(preprocess_dir),
                "--replace_flag",
                "--skip_synthetic",
            ])
            _write_log(logs_dir, f"{mode}_contract", contract)
            metrics_path = suite_dir / f"{mode}_background_metrics.json"
            metrics = _run([
                PYTHON,
                str(BACKGROUND_METRICS_SCRIPT),
                "--src_root_path",
                str(preprocess_dir),
                "--output_json",
                str(metrics_path),
            ])
            _write_log(logs_dir, f"{mode}_background_metrics", metrics)
            metrics_payload = _read_json(metrics_path)
            row["contract_returncode"] = int(contract.returncode)
            row["metrics"] = metrics_payload
            if metrics_payload is not None:
                stats = metrics_payload.get("background_stats", {})
                row["background_mode"] = metrics_payload.get("background_mode")
                row["temporal_fluctuation_mean"] = stats.get("temporal_fluctuation_mean")
                row["band_adjacent_background_stability"] = stats.get("band_adjacent_background_stability")
                row["support_ratio_mean"] = stats.get("support_ratio_mean")
                row["unresolved_ratio_mean"] = stats.get("unresolved_ratio_mean")
                row["background_confidence_mean"] = stats.get("background_confidence_mean")
        rows.append(row)

    summary_json = suite_dir / "summary.json"
    summary_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary_csv = suite_dir / "summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "mode",
                "run_dir",
                "preprocess_returncode",
                "contract_returncode",
                "background_mode",
                "temporal_fluctuation_mean",
                "band_adjacent_background_stability",
                "support_ratio_mean",
                "unresolved_ratio_mean",
                "background_confidence_mean",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in writer.fieldnames})

    print(json.dumps({"suite_dir": str(suite_dir.resolve()), "summary_json": str(summary_json.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
