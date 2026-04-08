#!/usr/bin/env python
import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = REPO_ROOT / "runs"
GENERATE_SCRIPT = REPO_ROOT / "generate.py"
REPLACEMENT_METRICS_SCRIPT = REPO_ROOT / "scripts" / "eval" / "compute_replacement_metrics.py"
BOUNDARY_METRICS_SCRIPT = REPO_ROOT / "scripts" / "eval" / "compute_boundary_refinement_metrics.py"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan.utils.generate_candidate_selection import load_candidate_manifest


DEFAULT_CKPT_DIR = "/home/user1/wan2.2/Wan2.2/Wan2.2-Animate-14B"
DEFAULT_SRC_ROOT = str(REPO_ROOT / "runs" / "optimization4_step06_round1_preprocess" / "preprocess")
DEFAULT_MANIFEST = REPO_ROOT / "docs" / "optimization4" / "benchmark" / "candidate_manifest.step05.json"


def _default_python_bin() -> str:
    override = os.environ.get("WAN_PYTHON")
    candidates = [override, "/home/user1/miniconda3/envs/wan/bin/python", sys.executable]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(Path(candidate).resolve())
    return sys.executable


PYTHON = _default_python_bin()


def _signal_name(returncode: int):
    if returncode >= 0:
        return None
    try:
        return signal.Signals(-returncode).name
    except Exception:
        return f"SIG{-returncode}"


def _run(command: list[str], *, cwd: Path = REPO_ROOT):
    started = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        env={**os.environ, "PYTHONFAULTHANDLER": "1"},
    )
    duration = time.perf_counter() - started
    return result, duration


def _write_log(directory: Path, stem: str, result: subprocess.CompletedProcess, *, duration_sec: float | None = None):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{stem}.stdout.log").write_text(result.stdout, encoding="utf-8")
    stderr_text = result.stderr
    if duration_sec is not None:
        stderr_text += f"\n[duration_sec] {duration_sec:.6f}\n"
    (directory / f"{stem}.stderr.log").write_text(stderr_text, encoding="utf-8")


def _read_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _find_output_video(run_dir: Path):
    outputs_dir = run_dir / "outputs"
    for suffix in ("*.mkv", "*.mp4"):
        matches = sorted(outputs_dir.glob(suffix))
        if matches:
            return matches[0]
    return None


def _build_generate_command(args, run_dir: Path, case_cfg: dict, candidate: dict):
    candidate_name = candidate["name"]
    output_name = f"{candidate_name}.mkv"
    command = [
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
        str(case_cfg["frame_num"]),
        "--refert_num",
        str(case_cfg["refert_num"]),
        "--replace_flag",
        "--sample_steps",
        str(case_cfg.get("sample_steps", args.sample_steps)),
        "--sample_shift",
        str(case_cfg.get("sample_shift", args.sample_shift)),
        "--offload_model",
        "False",
        "--sample_guide_scale",
        "1.0",
        "--replacement_conditioning_mode",
        candidate["replacement_conditioning_mode"],
        "--boundary_refine_mode",
        candidate["boundary_refine_mode"],
        "--replacement_boundary_strength",
        str(candidate["replacement_boundary_strength"]),
        "--replacement_transition_low",
        str(candidate["replacement_transition_low"]),
        "--replacement_transition_high",
        str(candidate["replacement_transition_high"]),
        "--boundary_refine_strength",
        str(candidate["boundary_refine_strength"]),
        "--boundary_refine_sharpen",
        str(candidate["boundary_refine_sharpen"]),
        "--save_debug_dir",
        str(run_dir / "debug" / "generate"),
        "--save_file",
        str(run_dir / "outputs" / output_name),
        "--output_format",
        "ffv1",
        "--base_seed",
        str(case_cfg.get("base_seed", args.base_seed)),
    ]
    command.extend(candidate.get("extra_args", []))
    return command


def _run_metrics(output_video: Path, src_root_path: Path, case_cfg: dict, metrics_path: Path, logs_dir: Path, stem: str):
    person_mask_path = Path(src_root_path) / "src_mask.npz"
    command = [
        PYTHON,
        str(REPLACEMENT_METRICS_SCRIPT),
        "--video_path",
        str(output_video),
        "--mask_path",
        str(person_mask_path),
        "--clip_len",
        str(case_cfg["frame_num"]),
        "--refert_num",
        str(case_cfg["refert_num"]),
        "--output_json",
        str(metrics_path),
    ]
    result, duration_sec = _run(command)
    _write_log(logs_dir, f"{stem}_replacement_metrics", result, duration_sec=duration_sec)
    return result.returncode == 0, _read_json(metrics_path)


def _run_boundary_metrics(default_output: Path, candidate_output: Path, src_root_path: Path, metrics_path: Path, logs_dir: Path, stem: str):
    command = [
        PYTHON,
        str(BOUNDARY_METRICS_SCRIPT),
        "--before",
        str(default_output),
        "--after",
        str(candidate_output),
        "--src_root_path",
        str(src_root_path),
        "--output_json",
        str(metrics_path),
    ]
    result, duration_sec = _run(command)
    _write_log(logs_dir, f"{stem}_boundary_metrics", result, duration_sec=duration_sec)
    return result.returncode == 0, _read_json(metrics_path)


def main():
    parser = argparse.ArgumentParser(description="Run Optimization4 Step05 generate-side edge candidate search.")
    parser.add_argument("--suite_name", type=str, default=None)
    parser.add_argument("--ckpt_dir", type=str, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--src_root_path", type=str, default=DEFAULT_SRC_ROOT)
    parser.add_argument("--candidate_manifest_json", type=str, default=str(DEFAULT_MANIFEST))
    parser.add_argument("--case_names", type=str, default=None, help="Comma-separated case names to run.")
    parser.add_argument("--candidate_names", type=str, default=None, help="Comma-separated candidate names to run.")
    parser.add_argument("--sample_steps", type=int, default=4)
    parser.add_argument("--sample_shift", type=float, default=5.0)
    parser.add_argument("--base_seed", type=int, default=123456)
    args = parser.parse_args()

    manifest = load_candidate_manifest(args.candidate_manifest_json)
    cases = manifest["cases"]
    candidates = manifest["candidates"]

    if args.case_names:
        allowed = {item.strip() for item in args.case_names.split(",") if item.strip()}
        cases = [case for case in cases if case["name"] in allowed]
    if args.candidate_names:
        allowed = {item.strip() for item in args.candidate_names.split(",") if item.strip()}
        candidates = [candidate for candidate in candidates if candidate["name"] in allowed]

    if not cases:
        raise ValueError("No cases selected for candidate search.")
    if not candidates:
        raise ValueError("No candidates selected for candidate search.")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suite_name = args.suite_name or f"optimization4_step05_candidate_search_{timestamp}"
    suite_dir = RUNS_ROOT / suite_name
    logs_dir = suite_dir / "logs"
    suite_dir.mkdir(parents=True, exist_ok=True)

    summary_cases = []
    default_candidate = manifest["default_candidate"]

    for case_cfg in cases:
        case_name = case_cfg["name"]
        case_dir = suite_dir / case_name
        case_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        outputs = {}

        for candidate in candidates:
            candidate_name = candidate["name"]
            run_dir = case_dir / candidate_name
            command = _build_generate_command(args, run_dir, case_cfg, candidate)
            result, duration_sec = _run(command)
            _write_log(logs_dir, f"{case_name}_{candidate_name}_generate", result, duration_sec=duration_sec)
            row = {
                "candidate_name": candidate_name,
                "is_default": candidate_name == default_candidate,
                "replacement_conditioning_mode": candidate["replacement_conditioning_mode"],
                "boundary_refine_mode": candidate["boundary_refine_mode"],
                "candidate_config": candidate,
                "run_dir": str(run_dir.resolve()),
                "generate_returncode": int(result.returncode),
                "generate_signal": _signal_name(result.returncode),
                "generate_duration_sec": float(duration_sec),
                "command_pretty": shlex.join(command),
            }
            output_video = _find_output_video(run_dir)
            if result.returncode == 0 and output_video is not None:
                repl_metrics_path = case_dir / f"{candidate_name}_replacement_metrics.json"
                _, repl_metrics = _run_metrics(
                    output_video,
                    Path(args.src_root_path),
                    case_cfg,
                    repl_metrics_path,
                    logs_dir,
                    f"{case_name}_{candidate_name}",
                )
                runtime_stats = _read_json(run_dir / "debug" / "generate" / "wan_animate_runtime_stats.json") or {}
                row.update(
                    {
                        "output_video": str(output_video.resolve()),
                        "replacement_metrics": repl_metrics or {},
                        "runtime_stats": runtime_stats,
                        "seam_score_mean": ((repl_metrics or {}).get("seam_score") or {}).get("mean"),
                        "background_fluctuation_mean": ((repl_metrics or {}).get("background_fluctuation") or {}).get("mean"),
                        "total_generate_sec": runtime_stats.get("total_generate_sec"),
                        "peak_memory_gb": runtime_stats.get("peak_memory_gb"),
                    }
                )
                outputs[candidate_name] = output_video
            rows.append(row)

        default_output = outputs.get(default_candidate)
        if default_output is not None:
            for row in rows:
                candidate_name = row["candidate_name"]
                if candidate_name == default_candidate:
                    continue
                candidate_output = outputs.get(candidate_name)
                if candidate_output is None:
                    continue
                boundary_metrics_path = case_dir / f"{candidate_name}_vs_{default_candidate}_boundary_metrics.json"
                _, boundary_metrics = _run_boundary_metrics(
                    default_output,
                    candidate_output,
                    Path(args.src_root_path),
                    boundary_metrics_path,
                    logs_dir,
                    f"{case_name}_{candidate_name}_vs_default",
                )
                row["boundary_metrics"] = boundary_metrics or {}

        summary_cases.append(
            {
                "case_name": case_name,
                "case_config": case_cfg,
                "rows": rows,
            }
        )

    summary = {
        "suite_dir": str(suite_dir.resolve()),
        "src_root_path": str(Path(args.src_root_path).resolve()),
        "candidate_manifest_json": str(Path(args.candidate_manifest_json).resolve()),
        "default_candidate": default_candidate,
        "cases": summary_cases,
    }
    summary_json = suite_dir / "summary.json"
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"suite_dir": str(suite_dir.resolve()), "summary_json": str(summary_json.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
