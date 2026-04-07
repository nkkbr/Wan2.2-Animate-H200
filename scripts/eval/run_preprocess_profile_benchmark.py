import argparse
import csv
import json
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
PREPROCESS_SCRIPT = REPO_ROOT / "wan" / "modules" / "animate" / "preprocess" / "preprocess_data.py"
CONTRACT_SCRIPT = REPO_ROOT / "scripts" / "eval" / "check_animate_contract.py"

DEFAULT_VIDEO_PATH = "/home/user1/wan2.2/test1/260306_01_step_1_0000_0010.mp4"
DEFAULT_REFERENCE_PATH = "/home/user1/wan2.2/test1/Gemini_Generated_Image_2.png"
DEFAULT_CKPT_PATH = "/home/user1/wan2.2/Wan2.2/Wan2.2-Animate-14B/process_checkpoint"


def _read_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _signal_name(returncode: int):
    if returncode >= 0:
        return None
    try:
        return signal.Signals(-returncode).name
    except Exception:
        return f"SIG{-returncode}"


def _run_contract_check(preprocess_dir: Path):
    command = [
        sys.executable,
        "-X",
        "faulthandler",
        str(CONTRACT_SCRIPT),
        "--src_root_path",
        str(preprocess_dir),
        "--replace_flag",
        "--skip_synthetic",
    ]
    result = subprocess.run(command, cwd=str(REPO_ROOT), text=True, capture_output=True)
    return {
        "passed": result.returncode == 0,
        "returncode": int(result.returncode),
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def _load_mask_np_stats(preprocess_dir: Path, metadata: dict, key: str):
    artifact = metadata.get("src_files", {}).get(key)
    if not artifact:
        return None
    path = preprocess_dir / artifact["path"]
    if artifact.get("format") != "npz" or not path.exists():
        return None
    data = np.load(path)
    payload = data["mask"].astype(np.float32)
    return {
        "mean": float(payload.mean()),
        "std": float(payload.std()),
        "max": float(payload.max()),
    }


def _load_background_diff_stats(preprocess_dir: Path):
    path = preprocess_dir / "background_diff.mp4"
    if not path.exists():
        return None
    from wan.utils.media_io import load_rgb_artifact

    frames = load_rgb_artifact(path, "mp4").astype(np.float32)
    gray = frames.mean(axis=3)
    return {
        "mean": float(gray.mean()),
        "std": float(gray.std()),
        "max": float(gray.max()),
    }


def _build_command(args, run_dir: Path, profile: str):
    command = [
        sys.executable,
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
        "--export_qa_visuals",
        "--sam_debug_trace",
        "--bg_inpaint_mode",
        "image",
        "--soft_mask_mode",
        "soft_band",
        "--reference_normalization_mode",
        "bbox_match",
        "--resolution_area",
        str(args.resolution_area[0]),
        str(args.resolution_area[1]),
        "--fps",
        str(args.fps),
        "--preprocess_runtime_profile",
        profile,
    ]
    if args.analysis_resolution_area is not None:
        command.extend(["--analysis_resolution_area", str(args.analysis_resolution_area[0]), str(args.analysis_resolution_area[1])])
    if args.analysis_min_short_side is not None:
        command.extend(["--analysis_min_short_side", str(args.analysis_min_short_side)])
    return command


def _summarize_case(run_dir: Path, profile: str, result: subprocess.CompletedProcess):
    manifest = _read_json(run_dir / "manifest.json")
    preprocess_dir = run_dir / "preprocess"
    metadata = _read_json(preprocess_dir / "metadata.json")
    runtime_stats = _read_json(preprocess_dir / "preprocess_runtime_stats.json")
    mask_stats = _read_json(preprocess_dir / "mask_stats.json")
    contract = _run_contract_check(preprocess_dir) if metadata is not None else None
    soft_band_stats = _load_mask_np_stats(preprocess_dir, metadata or {}, "soft_band") if metadata is not None else None
    background_diff_stats = _load_background_diff_stats(preprocess_dir) if metadata is not None else None
    stage_entry = None
    if manifest is not None:
        entries = manifest.get("stages", {}).get("preprocess", [])
        if entries:
            stage_entry = entries[-1]

    mask_area_ratio = None
    if mask_stats is not None and "mask_area_ratio" in mask_stats:
        ratios = np.asarray(mask_stats["mask_area_ratio"], dtype=np.float32)
        mask_area_ratio = {
            "mean": float(ratios.mean()),
            "std": float(ratios.std()),
            "min": float(ratios.min()),
            "max": float(ratios.max()),
        }

    summary = {
        "profile": profile,
        "run_dir": str(run_dir),
        "returncode": int(result.returncode),
        "signal": _signal_name(result.returncode),
        "manifest_status": None if stage_entry is None else stage_entry.get("status"),
        "runtime_stats": runtime_stats,
        "contract_check": contract,
        "mask_area_ratio": mask_area_ratio,
        "soft_band_stats": soft_band_stats,
        "background_diff_stats": background_diff_stats,
    }
    return summary


def _summary_row(case_name: str, summary: dict):
    runtime_stats = summary.get("runtime_stats") or {}
    stage_seconds = runtime_stats.get("stage_seconds") or {}
    analysis_shape = runtime_stats.get("analysis_shape") or [None, None]
    export_shape = runtime_stats.get("export_shape") or [None, None]
    mask_area_ratio = summary.get("mask_area_ratio") or {}
    soft_band_stats = summary.get("soft_band_stats") or {}
    background_diff_stats = summary.get("background_diff_stats") or {}
    contract = summary.get("contract_check") or {}
    return {
        "case_name": case_name,
        "profile": summary["profile"],
        "manifest_status": summary.get("manifest_status"),
        "returncode": summary.get("returncode"),
        "signal": summary.get("signal"),
        "preprocess_total_seconds": stage_seconds.get("total"),
        "peak_memory_gb": runtime_stats.get("peak_memory_gb"),
        "sam_chunk_seconds_mean": runtime_stats.get("sam_chunk_seconds_mean"),
        "sam_chunk_peak_memory_gb_max": runtime_stats.get("sam_chunk_peak_memory_gb_max"),
        "analysis_height": analysis_shape[0],
        "analysis_width": analysis_shape[1],
        "export_height": export_shape[0],
        "export_width": export_shape[1],
        "mask_area_ratio_mean": mask_area_ratio.get("mean"),
        "mask_area_ratio_std": mask_area_ratio.get("std"),
        "soft_band_mean": soft_band_stats.get("mean"),
        "background_diff_mean": background_diff_stats.get("mean"),
        "contract_passed": contract.get("passed"),
    }


def main():
    parser = argparse.ArgumentParser(description="Run preprocess runtime profile benchmarks for Wan-Animate replacement.")
    parser.add_argument("--video_path", type=str, default=DEFAULT_VIDEO_PATH)
    parser.add_argument("--reference_path", type=str, default=DEFAULT_REFERENCE_PATH)
    parser.add_argument("--ckpt_path", type=str, default=DEFAULT_CKPT_PATH)
    parser.add_argument(
        "--profile",
        action="append",
        choices=["legacy_safe", "h200_safe", "h200_aggressive"],
        default=None,
        help="Preprocess runtime profile to benchmark. Can be repeated. Defaults to all three profiles.",
    )
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--resolution_area", type=int, nargs=2, default=[640, 360])
    parser.add_argument("--analysis_resolution_area", type=int, nargs=2, default=None)
    parser.add_argument("--analysis_min_short_side", type=int, default=None)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--suite_name", type=str, default=None)
    args = parser.parse_args()

    profiles = args.profile or ["legacy_safe", "h200_safe", "h200_aggressive"]
    suite_name = args.suite_name or f"preprocess_profile_benchmark_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    suite_dir = REPO_ROOT / "runs" / suite_name
    suite_dir.mkdir(parents=True, exist_ok=True)

    cases = []
    csv_rows = []
    for profile in profiles:
        for repeat_index in range(args.repeat):
            case_name = f"{profile}_r{repeat_index + 1:02d}"
            run_dir = suite_dir / case_name
            run_dir.mkdir(parents=True, exist_ok=True)
            command = _build_command(args, run_dir, profile)
            print(f"[run] {case_name}")
            result = subprocess.run(
                command,
                cwd=str(REPO_ROOT),
                text=True,
                capture_output=True,
                env={**os.environ, "PYTHONFAULTHANDLER": "1"},
            )
            (run_dir / "runner_stdout.log").write_text(result.stdout, encoding="utf-8")
            (run_dir / "runner_stderr.log").write_text(result.stderr, encoding="utf-8")
            summary = _summarize_case(run_dir, profile, result)
            summary["case_name"] = case_name
            summary["command"] = command
            cases.append(summary)
            csv_rows.append(_summary_row(case_name, summary))
            print(f"[done] {case_name}: {summary.get('manifest_status') or summary.get('returncode')}")

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "suite_dir": str(suite_dir),
        "video_path": args.video_path,
        "reference_path": args.reference_path,
        "ckpt_path": args.ckpt_path,
        "resolution_area": list(args.resolution_area),
        "analysis_resolution_area": None if args.analysis_resolution_area is None else list(args.analysis_resolution_area),
        "analysis_min_short_side": args.analysis_min_short_side,
        "fps": args.fps,
        "cases": cases,
    }
    summary_path = suite_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    csv_path = suite_dir / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()) if csv_rows else [])
        if csv_rows:
            writer.writeheader()
            writer.writerows(csv_rows)
    print(f"Summary written to {summary_path}")
    print(f"CSV written to {csv_path}")


if __name__ == "__main__":
    main()
