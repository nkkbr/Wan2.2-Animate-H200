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
GENERATE_SCRIPT = REPO_ROOT / "generate.py"
REPLACEMENT_METRICS_SCRIPT = REPO_ROOT / "scripts" / "eval" / "compute_replacement_metrics.py"
BOUNDARY_METRICS_SCRIPT = REPO_ROOT / "scripts" / "eval" / "compute_boundary_refinement_metrics.py"

DEFAULT_CKPT_DIR = "/home/user1/wan2.2/Wan2.2/Wan2.2-Animate-14B"
DEFAULT_SRC_ROOT = str(REPO_ROOT / "runs" / "optimization3_step06_round5_ab" / "preprocess_video_v2" / "preprocess")


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


def _build_generate_command(args, run_dir: Path, conditioning_mode: str, boundary_refine_mode: str):
    output_name = f"{conditioning_mode}_{boundary_refine_mode}.mkv"
    return [
        PYTHON,
        str(GENERATE_SCRIPT),
        "--task",
        "animate-14B",
        "--ckpt_dir",
        args.ckpt_dir,
        "--src_root_path",
        args.src_root_path,
        "--run_dir",
        str(run_dir),
        "--save_manifest",
        "--frame_num",
        str(args.frame_num),
        "--refert_num",
        str(args.refert_num),
        "--replace_flag",
        "--sample_steps",
        str(args.sample_steps),
        "--sample_shift",
        str(args.sample_shift),
        "--offload_model",
        "False",
        "--sample_guide_scale",
        "1.0",
        "--replacement_conditioning_mode",
        conditioning_mode,
        "--boundary_refine_mode",
        boundary_refine_mode,
        "--boundary_refine_strength",
        str(args.boundary_refine_strength),
        "--boundary_refine_sharpen",
        str(args.boundary_refine_sharpen),
        "--save_debug_dir",
        str(run_dir / "debug" / "generate"),
        "--save_file",
        str(run_dir / "outputs" / output_name),
        "--output_format",
        "ffv1",
        "--base_seed",
        str(args.base_seed),
    ]


def main():
    parser = argparse.ArgumentParser(description="Run legacy/rich + deterministic/v2 Wan-Animate generate AB.")
    parser.add_argument("--suite_name", type=str, default=None)
    parser.add_argument("--ckpt_dir", type=str, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--src_root_path", type=str, default=DEFAULT_SRC_ROOT)
    parser.add_argument("--frame_num", type=int, default=45)
    parser.add_argument("--refert_num", type=int, default=9)
    parser.add_argument("--sample_steps", type=int, default=4)
    parser.add_argument("--sample_shift", type=float, default=5.0)
    parser.add_argument("--boundary_refine_strength", type=float, default=0.35)
    parser.add_argument("--boundary_refine_sharpen", type=float, default=0.15)
    parser.add_argument("--base_seed", type=int, default=123456)
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suite_name = args.suite_name or f"optimization3_step07_generate_ab_{timestamp}"
    suite_dir = RUNS_ROOT / suite_name
    suite_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = suite_dir / "logs"

    metadata = _read_json(Path(args.src_root_path) / "metadata.json")
    person_mask_rel = metadata["src_files"]["person_mask"]["path"]
    person_mask_path = Path(args.src_root_path) / person_mask_rel

    rows = []
    base_outputs = {}
    for conditioning_mode in ("legacy", "rich"):
        for boundary_refine_mode in ("none", "deterministic", "v2"):
            stem = f"{conditioning_mode}_{boundary_refine_mode}"
            run_dir = suite_dir / stem
            generate = _run(_build_generate_command(args, run_dir, conditioning_mode, boundary_refine_mode))
            _write_log(logs_dir, f"{stem}_generate", generate)

            row = {
                "conditioning_mode": conditioning_mode,
                "boundary_refine_mode": boundary_refine_mode,
                "run_dir": str(run_dir.resolve()),
                "generate_returncode": int(generate.returncode),
            }
            output_video = _find_output_video(run_dir)
            if generate.returncode == 0 and output_video is not None:
                repl_metrics_path = suite_dir / f"{stem}_replacement_metrics.json"
                repl = _run([
                    PYTHON,
                    str(REPLACEMENT_METRICS_SCRIPT),
                    "--video_path",
                    str(output_video),
                    "--mask_path",
                    str(person_mask_path),
                    "--clip_len",
                    str(args.frame_num),
                    "--refert_num",
                    str(args.refert_num),
                    "--output_json",
                    str(repl_metrics_path),
                ])
                _write_log(logs_dir, f"{stem}_replacement_metrics", repl)
                repl_metrics = _read_json(repl_metrics_path) or {}
                runtime_stats = _read_json(run_dir / "debug" / "generate" / "wan_animate_runtime_stats.json") or {}
                row.update({
                    "output_video": str(output_video.resolve()),
                    "replacement_metrics": repl_metrics,
                    "runtime_stats": runtime_stats,
                    "seam_score_mean": ((repl_metrics.get("seam_score") or {}).get("mean")),
                    "total_generate_sec": runtime_stats.get("total_generate_sec"),
                    "peak_memory_gb": runtime_stats.get("peak_memory_gb"),
                })
                if boundary_refine_mode == "none":
                    base_outputs[conditioning_mode] = output_video
                else:
                    before_path = base_outputs.get(conditioning_mode)
                    if before_path is not None:
                        boundary_metrics_path = suite_dir / f"{stem}_boundary_metrics.json"
                        boundary = _run([
                            PYTHON,
                            str(BOUNDARY_METRICS_SCRIPT),
                            "--before",
                            str(before_path),
                            "--after",
                            str(output_video),
                            "--src_root_path",
                            str(args.src_root_path),
                            "--output_json",
                            str(boundary_metrics_path),
                        ])
                        _write_log(logs_dir, f"{stem}_boundary_metrics", boundary)
                        row["boundary_metrics"] = _read_json(boundary_metrics_path)
            rows.append(row)

    summary_json = suite_dir / "summary.json"
    summary_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary_csv = suite_dir / "summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "conditioning_mode",
                "boundary_refine_mode",
                "generate_returncode",
                "seam_score_mean",
                "total_generate_sec",
                "peak_memory_gb",
                "halo_ratio_before",
                "halo_ratio_after",
                "band_gradient_before_mean",
                "band_gradient_after_mean",
                "band_edge_contrast_before_mean",
                "band_edge_contrast_after_mean",
            ],
        )
        writer.writeheader()
        for row in rows:
            boundary_metrics = row.get("boundary_metrics") or {}
            writer.writerow({
                "conditioning_mode": row.get("conditioning_mode"),
                "boundary_refine_mode": row.get("boundary_refine_mode"),
                "generate_returncode": row.get("generate_returncode"),
                "seam_score_mean": row.get("seam_score_mean"),
                "total_generate_sec": row.get("total_generate_sec"),
                "peak_memory_gb": row.get("peak_memory_gb"),
                "halo_ratio_before": boundary_metrics.get("halo_ratio_before"),
                "halo_ratio_after": boundary_metrics.get("halo_ratio_after"),
                "band_gradient_before_mean": boundary_metrics.get("band_gradient_before_mean"),
                "band_gradient_after_mean": boundary_metrics.get("band_gradient_after_mean"),
                "band_edge_contrast_before_mean": boundary_metrics.get("band_edge_contrast_before_mean"),
                "band_edge_contrast_after_mean": boundary_metrics.get("band_edge_contrast_after_mean"),
            })

    print(json.dumps({"suite_dir": str(suite_dir.resolve()), "summary_json": str(summary_json.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
