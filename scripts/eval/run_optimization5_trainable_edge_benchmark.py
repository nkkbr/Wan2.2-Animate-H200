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
BUILD_DATASET_SCRIPT = REPO_ROOT / "scripts" / "eval" / "build_edge_adapter_dataset.py"
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "eval" / "train_edge_alpha_adapter.py"
APPLY_SCRIPT = REPO_ROOT / "scripts" / "eval" / "apply_edge_alpha_adapter.py"
REVIEWED_METRICS_SCRIPT = REPO_ROOT / "scripts" / "eval" / "compute_reviewed_edge_metrics.py"
ALPHA_METRICS_SCRIPT = REPO_ROOT / "scripts" / "eval" / "compute_alpha_precision_metrics.py"
REPLACEMENT_METRICS_SCRIPT = REPO_ROOT / "scripts" / "eval" / "compute_replacement_metrics.py"
BOUNDARY_METRICS_SCRIPT = REPO_ROOT / "scripts" / "eval" / "compute_boundary_refinement_metrics.py"

DEFAULT_REVIEWED_DATASET = str(REPO_ROOT / "runs" / "optimization5_step01_round3" / "reviewed_edge_benchmark")
DEFAULT_SOURCE_BUNDLE = str(REPO_ROOT / "runs" / "optimization3_step06_round5_ab" / "preprocess_video_v2" / "preprocess")
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


def _build_generate_command(*, args, run_dir: Path, src_root_path: str, output_name: str):
    return [
        PYTHON, str(GENERATE_SCRIPT),
        "--task", "animate-14B",
        "--ckpt_dir", args.ckpt_dir,
        "--src_root_path", src_root_path,
        "--run_dir", str(run_dir),
        "--save_manifest",
        "--frame_num", str(args.frame_num),
        "--refert_num", str(args.refert_num),
        "--replace_flag",
        "--sample_steps", str(args.sample_steps),
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


def _metric_row(prefix: str, reviewed_json: Path, alpha_json: Path):
    reviewed = _read_json(reviewed_json) or {}
    alpha = _read_json(alpha_json) or {}
    return {
        f"{prefix}_reviewed_metrics": reviewed,
        f"{prefix}_alpha_metrics": alpha,
        f"{prefix}_boundary_f1_mean": reviewed.get("boundary_f1_mean"),
        f"{prefix}_alpha_mae_mean": reviewed.get("alpha_mae_mean"),
        f"{prefix}_trimap_error_mean": reviewed.get("trimap_error_mean"),
        f"{prefix}_fine_boundary_iou_mean": alpha.get("fine_boundary_iou_mean"),
        f"{prefix}_trimap_unknown_iou_mean": alpha.get("trimap_unknown_iou_mean"),
    }


def main():
    parser = argparse.ArgumentParser(description="Run optimization5 Step 07 trainable edge benchmark.")
    parser.add_argument("--suite_name", type=str, default=None)
    parser.add_argument("--reviewed_dataset_dir", type=str, default=DEFAULT_REVIEWED_DATASET)
    parser.add_argument("--source_preprocess_dir", type=str, default=DEFAULT_SOURCE_BUNDLE)
    parser.add_argument("--ckpt_dir", type=str, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--frame_num", type=int, default=45)
    parser.add_argument("--refert_num", type=int, default=9)
    parser.add_argument("--sample_steps", type=int, default=4)
    parser.add_argument("--sample_shift", type=float, default=5.0)
    parser.add_argument("--base_seed", type=int, default=123456)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--residual_scale", type=float, default=0.16)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--boundary_weight", type=float, default=4.0)
    parser.add_argument("--hair_weight", type=float, default=2.0)
    parser.add_argument("--hand_weight", type=float, default=1.5)
    parser.add_argument("--bce_weight", type=float, default=0.15)
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suite_name = args.suite_name or f"optimization5_step07_{timestamp}"
    suite_dir = RUNS_ROOT / suite_name
    suite_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = suite_dir / "logs"

    dataset_dir = suite_dir / "dataset"
    build = _run([
        PYTHON, str(BUILD_DATASET_SCRIPT),
        "--reviewed_dataset_dir", args.reviewed_dataset_dir,
        "--output_dir", str(dataset_dir),
    ])
    _write_log(logs_dir, "build_dataset", build)

    train_dir = suite_dir / "train"
    train = _run([
        PYTHON, str(TRAIN_SCRIPT),
        "--dataset_npz", str(dataset_dir / "edge_adapter_dataset.npz"),
        "--dataset_json", str(dataset_dir / "edge_adapter_dataset.json"),
        "--output_dir", str(train_dir),
        "--width", str(args.width),
        "--residual_scale", str(args.residual_scale),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--learning_rate", str(args.learning_rate),
        "--boundary_weight", str(args.boundary_weight),
        "--hair_weight", str(args.hair_weight),
        "--hand_weight", str(args.hand_weight),
        "--bce_weight", str(args.bce_weight),
    ])
    _write_log(logs_dir, "train", train)

    adapted_bundle_dir = suite_dir / "preprocess_adapter"
    apply = _run([
        PYTHON, str(APPLY_SCRIPT),
        "--source_preprocess_dir", args.source_preprocess_dir,
        "--checkpoint", str(train_dir / "edge_alpha_adapter.pt"),
        "--output_preprocess_dir", str(adapted_bundle_dir),
    ])
    _write_log(logs_dir, "apply_adapter", apply)

    baseline_reviewed_json = suite_dir / "baseline_reviewed_metrics.json"
    baseline_reviewed = _run([
        PYTHON, str(REVIEWED_METRICS_SCRIPT),
        "--dataset_dir", args.reviewed_dataset_dir,
        "--prediction_preprocess_dir", args.source_preprocess_dir,
        "--output_json", str(baseline_reviewed_json),
    ])
    _write_log(logs_dir, "baseline_reviewed_metrics", baseline_reviewed)

    candidate_reviewed_json = suite_dir / "candidate_reviewed_metrics.json"
    candidate_reviewed = _run([
        PYTHON, str(REVIEWED_METRICS_SCRIPT),
        "--dataset_dir", args.reviewed_dataset_dir,
        "--prediction_preprocess_dir", str(adapted_bundle_dir),
        "--output_json", str(candidate_reviewed_json),
    ])
    _write_log(logs_dir, "candidate_reviewed_metrics", candidate_reviewed)

    baseline_alpha_json = suite_dir / "baseline_alpha_metrics.json"
    baseline_alpha = _run([
        PYTHON, str(ALPHA_METRICS_SCRIPT),
        "--dataset_dir", args.reviewed_dataset_dir,
        "--prediction_preprocess_dir", args.source_preprocess_dir,
        "--output_json", str(baseline_alpha_json),
    ])
    _write_log(logs_dir, "baseline_alpha_metrics", baseline_alpha)

    candidate_alpha_json = suite_dir / "candidate_alpha_metrics.json"
    candidate_alpha = _run([
        PYTHON, str(ALPHA_METRICS_SCRIPT),
        "--dataset_dir", args.reviewed_dataset_dir,
        "--prediction_preprocess_dir", str(adapted_bundle_dir),
        "--output_json", str(candidate_alpha_json),
    ])
    _write_log(logs_dir, "candidate_alpha_metrics", candidate_alpha)

    baseline_run = suite_dir / "baseline_generate"
    baseline_gen = _run(_build_generate_command(args=args, run_dir=baseline_run, src_root_path=args.source_preprocess_dir, output_name="baseline.mkv"))
    _write_log(logs_dir, "baseline_generate", baseline_gen)

    candidate_run = suite_dir / "candidate_generate"
    candidate_gen = _run(_build_generate_command(args=args, run_dir=candidate_run, src_root_path=str(adapted_bundle_dir), output_name="candidate.mkv"))
    _write_log(logs_dir, "candidate_generate", candidate_gen)

    baseline_video = _find_output_video(baseline_run)
    candidate_video = _find_output_video(candidate_run)

    baseline_replacement_json = suite_dir / "baseline_replacement_metrics.json"
    if baseline_video is not None:
        baseline_repl = _run([
            PYTHON, str(REPLACEMENT_METRICS_SCRIPT),
            "--video_path", str(baseline_video),
            "--mask_path", str(Path(args.source_preprocess_dir) / "src_mask.npz"),
            "--clip_len", str(args.frame_num),
            "--refert_num", str(args.refert_num),
            "--output_json", str(baseline_replacement_json),
        ])
        _write_log(logs_dir, "baseline_replacement_metrics", baseline_repl)

    candidate_replacement_json = suite_dir / "candidate_replacement_metrics.json"
    if candidate_video is not None:
        candidate_repl = _run([
            PYTHON, str(REPLACEMENT_METRICS_SCRIPT),
            "--video_path", str(candidate_video),
            "--mask_path", str(adapted_bundle_dir / "src_mask.npz"),
            "--clip_len", str(args.frame_num),
            "--refert_num", str(args.refert_num),
            "--output_json", str(candidate_replacement_json),
        ])
        _write_log(logs_dir, "candidate_replacement_metrics", candidate_repl)

    boundary_json = suite_dir / "generate_boundary_metrics.json"
    if baseline_video is not None and candidate_video is not None:
        boundary = _run([
            PYTHON, str(BOUNDARY_METRICS_SCRIPT),
            "--before", str(baseline_video),
            "--after", str(candidate_video),
            "--src_root_path", str(adapted_bundle_dir),
            "--output_json", str(boundary_json),
        ])
        _write_log(logs_dir, "generate_boundary_metrics", boundary)

    summary = {
        "suite_dir": str(suite_dir.resolve()),
        "reviewed_dataset_dir": str(Path(args.reviewed_dataset_dir).resolve()),
        "source_preprocess_dir": str(Path(args.source_preprocess_dir).resolve()),
        "adapted_preprocess_dir": str(adapted_bundle_dir.resolve()),
        "train_summary": _read_json(train_dir / "train_summary.json"),
        **_metric_row("baseline", baseline_reviewed_json, baseline_alpha_json),
        **_metric_row("candidate", candidate_reviewed_json, candidate_alpha_json),
        "baseline_generate_runtime": _read_json(baseline_run / "debug" / "generate" / "wan_animate_runtime_stats.json"),
        "candidate_generate_runtime": _read_json(candidate_run / "debug" / "generate" / "wan_animate_runtime_stats.json"),
        "baseline_replacement_metrics": _read_json(baseline_replacement_json),
        "candidate_replacement_metrics": _read_json(candidate_replacement_json),
        "generate_boundary_metrics": _read_json(boundary_json),
    }
    summary_json = suite_dir / "summary.json"
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"suite_dir": str(suite_dir.resolve()), "summary_json": str(summary_json.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
