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
BOUNDARY_METRICS_SCRIPT = REPO_ROOT / "scripts" / "eval" / "compute_boundary_refinement_metrics.py"

DEFAULT_SRC_ROOT = str(REPO_ROOT / "runs" / "optimization3_step06_round5_ab" / "preprocess_video_v2" / "preprocess")
DEFAULT_CKPT_DIR = "/home/user1/wan2.2/Wan2.2/Wan2.2-Animate-14B"


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


def _build_generate_command(*, args, run_dir: Path, sample_steps: int, output_name: str):
    return [
        PYTHON, str(GENERATE_SCRIPT),
        "--task", "animate-14B",
        "--ckpt_dir", args.ckpt_dir,
        "--src_root_path", args.src_root_path,
        "--run_dir", str(run_dir),
        "--save_manifest",
        "--frame_num", str(args.frame_num),
        "--refert_num", str(args.refert_num),
        "--replace_flag",
        "--sample_steps", str(sample_steps),
        "--sample_shift", str(args.sample_shift),
        "--offload_model", "False",
        "--sample_guide_scale", "1.0",
        "--replacement_conditioning_mode", "legacy",
        "--boundary_refine_mode", "none",
        "--save_debug_dir", str(run_dir / "debug" / "generate"),
        "--save_file", str(run_dir / "outputs" / output_name),
        "--output_format", "ffv1",
        "--base_seed", str(args.base_seed),
    ]


def main():
    parser = argparse.ArgumentParser(description="Run optimization5 Step 08 H200 quality tier benchmark.")
    parser.add_argument("--suite_name", type=str, default=None)
    parser.add_argument("--src_root_path", type=str, default=DEFAULT_SRC_ROOT)
    parser.add_argument("--ckpt_dir", type=str, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--frame_num", type=int, default=45)
    parser.add_argument("--refert_num", type=int, default=9)
    parser.add_argument("--sample_shift", type=float, default=5.0)
    parser.add_argument("--base_seed", type=int, default=123456)
    parser.add_argument("--default_steps", type=int, default=4)
    parser.add_argument("--high_steps", type=int, default=8)
    parser.add_argument("--extreme_steps", type=int, default=12)
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suite_name = args.suite_name or f"optimization5_step08_{timestamp}"
    suite_dir = RUNS_ROOT / suite_name
    suite_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = suite_dir / "logs"

    mask_path = Path(args.src_root_path) / "src_mask.npz"
    rows = []
    videos = {}
    for tier_name, sample_steps in (
        ("default", args.default_steps),
        ("high", args.high_steps),
        ("extreme", args.extreme_steps),
    ):
        run_dir = suite_dir / tier_name
        generate = _run(_build_generate_command(args=args, run_dir=run_dir, sample_steps=sample_steps, output_name=f"{tier_name}.mkv"))
        _write_log(logs_dir, f"{tier_name}_generate", generate)
        row = {
            "tier": tier_name,
            "sample_steps": int(sample_steps),
            "run_dir": str(run_dir.resolve()),
            "generate_returncode": int(generate.returncode),
        }
        output_video = _find_output_video(run_dir)
        if generate.returncode == 0 and output_video is not None:
            videos[tier_name] = output_video
            repl_json = suite_dir / f"{tier_name}_replacement_metrics.json"
            repl = _run([
                PYTHON, str(REPLACEMENT_METRICS_SCRIPT),
                "--video_path", str(output_video),
                "--mask_path", str(mask_path),
                "--clip_len", str(args.frame_num),
                "--refert_num", str(args.refert_num),
                "--output_json", str(repl_json),
            ])
            _write_log(logs_dir, f"{tier_name}_replacement_metrics", repl)
            runtime = _read_json(run_dir / "debug" / "generate" / "wan_animate_runtime_stats.json")
            row.update({
                "output_video": str(output_video.resolve()),
                "replacement_metrics": _read_json(repl_json),
                "runtime_stats": runtime,
                "seam_score_mean": ((_read_json(repl_json) or {}).get("seam_score") or {}).get("mean"),
                "background_fluctuation_mean": ((_read_json(repl_json) or {}).get("background_fluctuation") or {}).get("mean"),
                "total_generate_sec": (runtime or {}).get("total_generate_sec"),
                "peak_memory_gb": (runtime or {}).get("peak_memory_gb"),
            })
        rows.append(row)

    pair_metrics = {}
    for before_tier, after_tier in (("default", "high"), ("high", "extreme"), ("default", "extreme")):
        if before_tier in videos and after_tier in videos:
            out_json = suite_dir / f"{before_tier}_to_{after_tier}_boundary_metrics.json"
            result = _run([
                PYTHON, str(BOUNDARY_METRICS_SCRIPT),
                "--before", str(videos[before_tier]),
                "--after", str(videos[after_tier]),
                "--src_root_path", args.src_root_path,
                "--output_json", str(out_json),
            ])
            _write_log(logs_dir, f"{before_tier}_to_{after_tier}_boundary_metrics", result)
            pair_metrics[f"{before_tier}_to_{after_tier}"] = _read_json(out_json)

    summary = {
        "suite_dir": str(suite_dir.resolve()),
        "src_root_path": str(Path(args.src_root_path).resolve()),
        "rows": rows,
        "pair_metrics": pair_metrics,
    }
    summary_json = suite_dir / "summary.json"
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"suite_dir": str(suite_dir.resolve()), "summary_json": str(summary_json.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
