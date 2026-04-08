#!/usr/bin/env python
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = REPO_ROOT / "runs"
GENERATE_SCRIPT = REPO_ROOT / "generate.py"
REPLACEMENT_METRICS_SCRIPT = REPO_ROOT / "scripts" / "eval" / "compute_replacement_metrics.py"
ROI_METRICS_SCRIPT = REPO_ROOT / "scripts" / "eval" / "compute_boundary_roi_metrics.py"

DEFAULT_CKPT_DIR = "/home/user1/wan2.2/Wan2.2/Wan2.2-Animate-14B"
DEFAULT_SRC_ROOT = str(REPO_ROOT / "runs" / "optimization5_step03_round1_preprocess" / "preprocess")


def _default_python_bin() -> str:
    override = os.environ.get("WAN_PYTHON")
    candidates = [override, "/home/user1/miniconda3/envs/wan/bin/python", sys.executable]
    for candidate in candidates:
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


def _find_output_video(run_dir: Path):
    for suffix in ("*.mkv", "*.mp4"):
        matches = sorted((run_dir / "outputs").glob(suffix))
        if matches:
            return matches[0]
    return None


def _build_generate_command(args, run_dir: Path, boundary_refine_mode: str):
    output_name = f"{boundary_refine_mode}.mkv"
    return [
        PYTHON,
        str(GENERATE_SCRIPT),
        "--task", "animate-14B",
        "--ckpt_dir", args.ckpt_dir,
        "--src_root_path", args.src_root_path,
        "--run_dir", str(run_dir),
        "--save_manifest",
        "--frame_num", str(args.frame_num),
        "--refert_num", str(args.refert_num),
        "--replace_flag",
        "--sample_solver", "dpm++",
        "--sample_steps", str(args.sample_steps),
        "--sample_shift", str(args.sample_shift),
        "--offload_model", "False",
        "--sample_guide_scale", "1.0",
        "--replacement_conditioning_mode", "legacy",
        "--boundary_refine_mode", boundary_refine_mode,
        "--boundary_refine_strength", str(args.boundary_refine_strength),
        "--boundary_refine_sharpen", str(args.boundary_refine_sharpen),
        "--save_debug_dir", str(run_dir / "debug" / "generate"),
        "--save_file", str(run_dir / "outputs" / output_name),
        "--output_format", "ffv1",
        "--base_seed", str(args.base_seed),
    ]


def main():
    parser = argparse.ArgumentParser(description="Run Optimization5 Step05 boundary ROI reconstruction benchmark.")
    parser.add_argument("--suite_name", type=str, default=None)
    parser.add_argument("--ckpt_dir", type=str, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--src_root_path", type=str, default=DEFAULT_SRC_ROOT)
    parser.add_argument("--frame_num", type=int, default=45)
    parser.add_argument("--refert_num", type=int, default=5)
    parser.add_argument("--sample_steps", type=int, default=4)
    parser.add_argument("--sample_shift", type=float, default=5.0)
    parser.add_argument("--base_seed", type=int, default=123)
    parser.add_argument("--boundary_refine_strength", type=float, default=0.35)
    parser.add_argument("--boundary_refine_sharpen", type=float, default=0.15)
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suite_name = args.suite_name or f"optimization5_step05_ab_{timestamp}"
    suite_dir = RUNS_ROOT / suite_name
    suite_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = suite_dir / "logs"

    metadata = _read_json(Path(args.src_root_path) / "metadata.json")
    person_mask_path = Path(args.src_root_path) / metadata["src_files"]["person_mask"]["path"]

    rows = []
    outputs = {}
    debug_dirs = {}
    for boundary_refine_mode in ("none", "roi_gen_v1"):
        run_dir = suite_dir / boundary_refine_mode
        generate = _run(_build_generate_command(args, run_dir, boundary_refine_mode))
        _write_log(logs_dir, f"{boundary_refine_mode}_generate", generate)
        row = {
            "boundary_refine_mode": boundary_refine_mode,
            "run_dir": str(run_dir.resolve()),
            "generate_returncode": int(generate.returncode),
        }
        output_video = _find_output_video(run_dir)
        if generate.returncode == 0 and output_video is not None:
            repl_metrics_path = suite_dir / f"{boundary_refine_mode}_replacement_metrics.json"
            repl = _run([
                PYTHON,
                str(REPLACEMENT_METRICS_SCRIPT),
                "--video_path", str(output_video),
                "--mask_path", str(person_mask_path),
                "--clip_len", str(args.frame_num),
                "--refert_num", str(args.refert_num),
                "--output_json", str(repl_metrics_path),
            ])
            _write_log(logs_dir, f"{boundary_refine_mode}_replacement_metrics", repl)
            repl_metrics = _read_json(repl_metrics_path) or {}
            runtime_stats = _read_json(run_dir / "debug" / "generate" / "wan_animate_runtime_stats.json") or {}
            row.update({
                "output_video": str(output_video.resolve()),
                "replacement_metrics": repl_metrics,
                "runtime_stats": runtime_stats,
                "seam_score_mean": ((repl_metrics.get("seam_score") or {}).get("mean")),
                "background_fluctuation_mean": ((repl_metrics.get("background_fluctuation") or {}).get("mean")),
                "total_generate_sec": runtime_stats.get("total_generate_sec"),
                "peak_memory_gb": runtime_stats.get("peak_memory_gb"),
            })
            outputs[boundary_refine_mode] = output_video
            debug_dirs[boundary_refine_mode] = run_dir / "debug" / "generate" / "boundary_refinement"
        rows.append(row)

    roi_metrics = None
    if "none" in outputs and "roi_gen_v1" in outputs and debug_dirs.get("roi_gen_v1"):
        roi_metrics_path = suite_dir / "roi_gen_v1_vs_none_roi_metrics.json"
        roi = _run([
            PYTHON,
            str(ROI_METRICS_SCRIPT),
            "--before", str(outputs["none"]),
            "--after", str(outputs["roi_gen_v1"]),
            "--src_root_path", str(args.src_root_path),
            "--debug_dir", str(debug_dirs["roi_gen_v1"]),
            "--output_json", str(roi_metrics_path),
        ])
        _write_log(logs_dir, "roi_gen_v1_vs_none_roi_metrics", roi)
        roi_metrics = _read_json(roi_metrics_path)

    payload = {
        "suite_dir": str(suite_dir.resolve()),
        "src_root_path": str(Path(args.src_root_path).resolve()),
        "rows": rows,
        "roi_metrics": roi_metrics,
    }
    summary_json = suite_dir / "summary.json"
    summary_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"suite_dir": str(suite_dir.resolve()), "summary_json": str(summary_json.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
