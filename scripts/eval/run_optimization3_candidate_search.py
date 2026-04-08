#!/usr/bin/env python
import argparse
import csv
import json
import math
import os
import shlex
import signal
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan.utils.preprocess_candidate_selection import (
    load_candidate_manifest,
    load_score_policy,
    score_candidates,
)


def _default_python_bin() -> str:
    override = os.environ.get("WAN_PYTHON")
    candidates = [override, "/home/user1/miniconda3/envs/wan/bin/python", sys.executable]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(Path(candidate).resolve())
    return sys.executable


PYTHON = _default_python_bin()

PREPROCESS_SCRIPT = REPO_ROOT / "wan" / "modules" / "animate" / "preprocess" / "preprocess_data.py"
GENERATE_SCRIPT = REPO_ROOT / "generate.py"
BOUNDARY_METRICS = REPO_ROOT / "scripts" / "eval" / "compute_boundary_precision_metrics.py"
FACE_METRICS = REPO_ROOT / "scripts" / "eval" / "compute_face_precision_metrics.py"
POSE_METRICS = REPO_ROOT / "scripts" / "eval" / "compute_pose_precision_metrics.py"
BACKGROUND_METRICS = REPO_ROOT / "scripts" / "eval" / "compute_background_precision_metrics.py"
REPLACEMENT_METRICS = REPO_ROOT / "scripts" / "eval" / "compute_replacement_metrics.py"
CONTRACT_CHECK = REPO_ROOT / "scripts" / "eval" / "check_animate_contract.py"

DEFAULT_CANDIDATE_MANIFEST = REPO_ROOT / "docs" / "optimization3" / "benchmark" / "candidate_manifest.step08.json"
DEFAULT_SCORE_POLICY = REPO_ROOT / "docs" / "optimization3" / "benchmark" / "candidate_score_policy.step08.json"

DEFAULT_VIDEO_PATH = "/home/user1/wan2.2/test1/260306_01_step_1_0000_0010.mp4"
DEFAULT_REFERENCE_PATH = "/home/user1/wan2.2/test1/Gemini_Generated_Image_2.png"
DEFAULT_CKPT_PATH = "/home/user1/wan2.2/Wan2.2/Wan2.2-Animate-14B/process_checkpoint"
DEFAULT_GENERATE_CKPT_DIR = "/home/user1/wan2.2/Wan2.2/Wan2.2-Animate-14B"


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


def _run_command(name: str, command: list[str], *, workdir: Path, log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"{name}.stdout.log"
    stderr_path = log_dir / f"{name}.stderr.log"
    started = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=str(workdir),
        text=True,
        capture_output=True,
        env={**os.environ, "PYTHONFAULTHANDLER": "1"},
    )
    duration = time.perf_counter() - started
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    return {
        "returncode": int(result.returncode),
        "signal": _signal_name(result.returncode),
        "duration_sec": float(duration),
        "stdout_path": str(stdout_path.resolve()),
        "stderr_path": str(stderr_path.resolve()),
        "command": command,
        "command_pretty": shlex.join(command),
        "passed": result.returncode == 0,
    }


def _metric_command(script: Path, src_root_path: Path, output_json: Path):
    return [
        PYTHON,
        str(script),
        "--src_root_path",
        str(src_root_path),
        "--output_json",
        str(output_json),
    ]


def _build_preprocess_command(args, candidate: dict, run_dir: Path):
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
        "640",
        "360",
        "--fps",
        "5",
        "--preprocess_runtime_profile",
        "h200_extreme",
        "--sam_runtime_profile",
        "h200_safe",
        "--sam_prompt_mode",
        "mask_seed",
        "--no-sam_apply_postprocessing",
        "--no-sam_use_negative_points",
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
        "--sam_chunk_len",
        "12",
        "--sam_keyframes_per_chunk",
        "3",
        "--sam_reprompt_interval",
        "0",
        "--multistage_preprocess_mode",
        "h200_extreme",
        "--face_analysis_mode",
        "heuristic",
        "--pose_motion_stack_mode",
        "v1",
        "--boundary_fusion_mode",
        "v2",
        "--parsing_mode",
        "heuristic",
        "--matting_mode",
        "heuristic",
        "--bg_inpaint_mode",
        "video_v2",
        "--soft_mask_mode",
        "soft_band",
        "--reference_normalization_mode",
        "structure_match",
        "--reference_structure_segment_clamp_min",
        "0.8",
        "--reference_structure_segment_clamp_max",
        "1.25",
        "--reference_structure_width_budget_ratio",
        "1.05",
        "--reference_structure_height_budget_ratio",
        "1.05",
    ]
    command.extend(candidate.get("extra_args", []))
    return command


def _build_generate_command(run_dir: Path, src_root_path: Path, args):
    output_path = run_dir / "outputs" / "replacement_smoke_none.mkv"
    return [
        PYTHON,
        str(GENERATE_SCRIPT),
        "--task",
        "animate-14B",
        "--ckpt_dir",
        args.generate_ckpt_dir,
        "--src_root_path",
        str(src_root_path),
        "--run_dir",
        str(run_dir),
        "--save_manifest",
        "--frame_num",
        "45",
        "--refert_num",
        "9",
        "--replace_flag",
        "--sample_steps",
        "4",
        "--sample_shift",
        "5.0",
        "--sample_guide_scale",
        "1.0",
        "--replacement_conditioning_mode",
        "rich",
        "--boundary_refine_mode",
        "none",
        "--save_debug_dir",
        str(run_dir / "debug" / "generate"),
        "--save_file",
        str(output_path),
        "--output_format",
        "ffv1",
        "--base_seed",
        "123456",
        "--offload_model",
        "False",
    ]


def _run_contract(preprocess_dir: Path, log_dir: Path):
    command = [
        PYTHON,
        str(CONTRACT_CHECK),
        "--src_root_path",
        str(preprocess_dir),
        "--replace_flag",
        "--skip_synthetic",
    ]
    return _run_command("contract_check", command, workdir=REPO_ROOT, log_dir=log_dir)


def _run_metrics(preprocess_dir: Path, metrics_dir: Path, log_dir: Path):
    metrics_dir.mkdir(parents=True, exist_ok=True)
    boundary_path = metrics_dir / "boundary_precision.json"
    face_path = metrics_dir / "face_precision.json"
    pose_path = metrics_dir / "pose_precision.json"
    background_path = metrics_dir / "background_precision.json"

    boundary_run = _run_command("boundary_metrics", _metric_command(BOUNDARY_METRICS, preprocess_dir, boundary_path), workdir=REPO_ROOT, log_dir=log_dir)
    face_run = _run_command("face_metrics", _metric_command(FACE_METRICS, preprocess_dir, face_path), workdir=REPO_ROOT, log_dir=log_dir)
    pose_run = _run_command("pose_metrics", _metric_command(POSE_METRICS, preprocess_dir, pose_path), workdir=REPO_ROOT, log_dir=log_dir)
    background_run = _run_command("background_metrics", _metric_command(BACKGROUND_METRICS, preprocess_dir, background_path), workdir=REPO_ROOT, log_dir=log_dir)

    return {
        "boundary": _read_json(boundary_path) if boundary_run["passed"] else None,
        "face": _read_json(face_path) if face_run["passed"] else None,
        "pose": _read_json(pose_path) if pose_run["passed"] else None,
        "background": _read_json(background_path) if background_run["passed"] else None,
        "runs": {
            "boundary": boundary_run,
            "face": face_run,
            "pose": pose_run,
            "background": background_run,
        },
    }


def _summarize_candidate(candidate: dict, run_dir: Path, preprocess_run: dict, contract_run: dict | None, metrics_payload: dict | None):
    preprocess_dir = run_dir / "preprocess"
    manifest = _read_json(run_dir / "manifest.json")
    runtime_stats = _read_json(preprocess_dir / "preprocess_runtime_stats.json")
    stage_entry = None
    if manifest is not None:
        entries = manifest.get("stages", {}).get("preprocess", [])
        if entries:
            stage_entry = entries[-1]
    contract_passed = bool(contract_run and contract_run["passed"])
    metrics = metrics_payload or {}
    all_metric_runs_passed = all((metrics.get("runs") or {}).get(name, {}).get("passed", False) for name in ("boundary", "face", "pose", "background"))
    passed = bool(preprocess_run["passed"] and contract_passed and all_metric_runs_passed)
    return {
        "name": candidate["name"],
        "description": candidate.get("description"),
        "is_default": bool(candidate.get("is_default", False)),
        "tags": candidate.get("tags", []),
        "extra_args": candidate.get("extra_args", []),
        "run_dir": str(run_dir.resolve()),
        "preprocess_dir": str(preprocess_dir.resolve()),
        "passed": passed,
        "preprocess_returncode": preprocess_run["returncode"],
        "preprocess_signal": preprocess_run["signal"],
        "manifest_status": None if stage_entry is None else stage_entry.get("status"),
        "contract_passed": contract_passed,
        "preprocess_duration_sec": preprocess_run["duration_sec"],
        "runtime_stats": runtime_stats,
        "metrics": {
            "boundary": metrics.get("boundary"),
            "face": metrics.get("face"),
            "pose": metrics.get("pose"),
            "background": metrics.get("background"),
            "runtime": runtime_stats or {},
        },
        "logs": {
            "preprocess": preprocess_run,
            "contract": contract_run,
            "metric_runs": None if metrics_payload is None else metrics_payload.get("runs"),
        },
    }


def _run_generate_metrics(run_dir: Path, preprocess_dir: Path, log_dir: Path, generate_ckpt_dir: str):
    generate_run_dir = run_dir
    generate_run = _run_command(
        "generate_smoke",
        _build_generate_command(
            generate_run_dir,
            preprocess_dir,
            argparse.Namespace(generate_ckpt_dir=generate_ckpt_dir),
        ),
        workdir=REPO_ROOT,
        log_dir=log_dir,
    )
    metrics_path = generate_run_dir / "metrics" / "replacement_metrics.json"
    metrics_run = None
    metrics_payload = None
    if generate_run["passed"]:
        output_path = generate_run_dir / "outputs" / "replacement_smoke_none.mkv"
        mask_path = preprocess_dir / "src_mask.npz"
        if not mask_path.exists():
            mask_path = preprocess_dir / "src_mask.mp4"
        metrics_command = [
            PYTHON,
            str(REPLACEMENT_METRICS),
            "--video_path",
            str(output_path),
            "--mask_path",
            str(mask_path),
            "--clip_len",
            "45",
            "--refert_num",
            "9",
            "--output_json",
            str(metrics_path),
        ]
        metrics_run = _run_command("generate_metrics", metrics_command, workdir=REPO_ROOT, log_dir=log_dir)
        if metrics_run["passed"]:
            metrics_payload = _read_json(metrics_path)
    return {
        "generate_run": generate_run,
        "metrics_run": metrics_run,
        "metrics": metrics_payload,
    }


def _summary_row(repeat_index: int, candidate: dict, selection_result: dict | None):
    metrics = candidate.get("metrics", {})
    boundary = metrics.get("boundary") or {}
    face = metrics.get("face") or {}
    pose = metrics.get("pose") or {}
    background = metrics.get("background") or {}
    runtime = metrics.get("runtime") or {}
    stage_seconds = runtime.get("stage_seconds") or {}
    selected_name = None if selection_result is None else selection_result.get("selected_candidate")
    selected_total_score = None
    candidate_total_score = None
    for scored in (selection_result or {}).get("scored_candidates", []):
        if scored.get("name") == candidate["name"]:
            candidate_total_score = scored.get("total_score")
        if scored.get("name") == selected_name:
            selected_total_score = scored.get("total_score")
    return {
        "repeat_index": repeat_index,
        "candidate_name": candidate["name"],
        "is_default": candidate.get("is_default"),
        "passed": candidate.get("passed"),
        "selected_candidate": selected_name,
        "candidate_total_score": candidate_total_score,
        "selected_total_score": selected_total_score,
        "preprocess_total_seconds": stage_seconds.get("total"),
        "peak_memory_gb": runtime.get("peak_memory_gb"),
        "boundary_focus_ratio": boundary.get("uncertainty_transition_focus_ratio_dilated"),
        "boundary_to_interior_ratio": boundary.get("uncertainty_transition_to_interior_ratio"),
        "face_center_jitter_mean": face.get("center_jitter_mean"),
        "face_valid_points_mean": face.get("valid_face_points_mean"),
        "pose_body_jitter_mean": pose.get("body_jitter_mean"),
        "pose_hand_jitter_mean": pose.get("hand_jitter_mean"),
        "pose_velocity_spike_rate": pose.get("velocity_spike_rate"),
        "pose_limb_continuity_score": pose.get("limb_continuity_score"),
        "bg_temporal_fluctuation_mean": ((background.get("background_stats") or {}).get("temporal_fluctuation_mean")),
        "bg_band_stability": ((background.get("background_stats") or {}).get("band_adjacent_background_stability")),
        "bg_unresolved_ratio_mean": ((background.get("background_stats") or {}).get("unresolved_ratio_mean")),
    }


def _write_summary_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _compare_selected_vs_default(selection_result: dict, generate_records: dict):
    default_name = selection_result.get("default_candidate")
    selected_name = selection_result.get("selected_candidate")
    default_metrics = ((generate_records.get(default_name) or {}).get("metrics")) if default_name else None
    selected_metrics = ((generate_records.get(selected_name) or {}).get("metrics")) if selected_name else None
    if default_metrics is None or selected_metrics is None:
        return None
    default_seam = ((default_metrics.get("seam_score") or {}).get("mean"))
    selected_seam = ((selected_metrics.get("seam_score") or {}).get("mean"))
    default_bg = ((default_metrics.get("background_fluctuation") or {}).get("mean"))
    selected_bg = ((selected_metrics.get("background_fluctuation") or {}).get("mean"))
    return {
        "default_candidate": default_name,
        "selected_candidate": selected_name,
        "seam_delta": None if default_seam is None or selected_seam is None else float(selected_seam - default_seam),
        "background_fluctuation_delta": None if default_bg is None or selected_bg is None else float(selected_bg - default_bg),
    }


def _generate_nonregression_passed(compare: dict | None):
    if compare is None:
        return None
    seam_delta = compare.get("seam_delta")
    bg_delta = compare.get("background_fluctuation_delta")
    if seam_delta is None or bg_delta is None:
        return False
    return bool(seam_delta <= 0.0 and bg_delta <= 0.0)


def main():
    parser = argparse.ArgumentParser(description="Run optimization3 multi-candidate preprocess search and auto-selection.")
    parser.add_argument("--video_path", type=str, default=DEFAULT_VIDEO_PATH)
    parser.add_argument("--reference_path", type=str, default=DEFAULT_REFERENCE_PATH)
    parser.add_argument("--ckpt_path", type=str, default=DEFAULT_CKPT_PATH)
    parser.add_argument("--generate_ckpt_dir", type=str, default=DEFAULT_GENERATE_CKPT_DIR)
    parser.add_argument("--candidate_manifest", type=str, default=str(DEFAULT_CANDIDATE_MANIFEST))
    parser.add_argument("--score_policy_json", type=str, default=str(DEFAULT_SCORE_POLICY))
    parser.add_argument("--suite_name", type=str, default=None)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--run_generate_smoke", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    candidate_manifest = load_candidate_manifest(args.candidate_manifest)
    score_policy = load_score_policy(args.score_policy_json)

    suite_name = args.suite_name or f"optimization3_step08_candidate_search_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    suite_dir = REPO_ROOT / "runs" / suite_name
    suite_dir.mkdir(parents=True, exist_ok=True)

    repeats = []
    csv_rows = []
    for repeat_index in range(1, args.repeat + 1):
        repeat_dir = suite_dir / f"repeat_{repeat_index:02d}"
        repeat_dir.mkdir(parents=True, exist_ok=True)
        repeat_summary = {
            "repeat_index": repeat_index,
            "candidates": [],
            "selection_result": None,
            "generate_records": {},
            "selected_vs_default_generate": None,
        }

        for candidate in candidate_manifest["candidates"]:
            candidate_run_dir = repeat_dir / candidate["name"]
            candidate_log_dir = candidate_run_dir / "logs"
            command = _build_preprocess_command(args, candidate, candidate_run_dir)
            preprocess_run = _run_command("preprocess", command, workdir=REPO_ROOT, log_dir=candidate_log_dir)
            contract_run = None
            metrics_payload = None
            if preprocess_run["passed"]:
                contract_run = _run_contract(candidate_run_dir / "preprocess", candidate_log_dir)
                if contract_run["passed"]:
                    metrics_payload = _run_metrics(candidate_run_dir / "preprocess", candidate_run_dir / "metrics", candidate_log_dir)
            candidate_summary = _summarize_candidate(candidate, candidate_run_dir, preprocess_run, contract_run, metrics_payload)
            repeat_summary["candidates"].append(candidate_summary)

        repeat_payload_path = repeat_dir / "candidate_summary.json"
        repeat_payload_path.write_text(json.dumps({"candidates": repeat_summary["candidates"]}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        selection_result = score_candidates(repeat_summary["candidates"], score_policy)
        selection_path = repeat_dir / "selection_result.json"
        selection_path.write_text(json.dumps(selection_result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        repeat_summary["selection_result"] = selection_result

        if args.run_generate_smoke and selection_result.get("selected_candidate") is not None:
            chosen_names = [selection_result["default_candidate"], selection_result["selected_candidate"]]
            for name in dict.fromkeys(chosen_names):
                if name is None:
                    continue
                candidate_obj = next((item for item in repeat_summary["candidates"] if item["name"] == name), None)
                if candidate_obj is None or not candidate_obj.get("passed"):
                    continue
                generate_run_dir = repeat_dir / f"generate_{name}"
                generate_record = _run_generate_metrics(
                    generate_run_dir,
                    Path(candidate_obj["preprocess_dir"]),
                    generate_run_dir / "logs",
                    args.generate_ckpt_dir,
                )
                repeat_summary["generate_records"][name] = generate_record
            repeat_summary["selected_vs_default_generate"] = _compare_selected_vs_default(selection_result, repeat_summary["generate_records"])

        repeats.append(repeat_summary)
        for candidate in repeat_summary["candidates"]:
            csv_rows.append(_summary_row(repeat_index, candidate, selection_result))

    selected_names = [repeat["selection_result"]["selected_candidate"] for repeat in repeats if repeat["selection_result"]]
    selected_counter = Counter(selected_names)
    most_common_selected = selected_counter.most_common(1)[0][0] if selected_counter else None
    stable_ratio = float(selected_counter[most_common_selected] / len(repeats)) if repeats and most_common_selected is not None else 0.0
    better_flags = [
        bool((repeat["selection_result"] or {}).get("selected_better_than_default"))
        for repeat in repeats
        if repeat.get("selection_result") is not None
    ]
    selected_better_ratio = float(sum(1 for flag in better_flags if flag) / len(better_flags)) if better_flags else 0.0
    generate_flags = [
        _generate_nonregression_passed(repeat.get("selected_vs_default_generate"))
        for repeat in repeats
        if repeat.get("selected_vs_default_generate") is not None
    ]
    generate_flags = [flag for flag in generate_flags if flag is not None]
    generate_nonregression_ratio = (
        float(sum(1 for flag in generate_flags if flag) / len(generate_flags)) if generate_flags else None
    )
    generate_gate_passed = (
        True if not args.run_generate_smoke else bool(generate_nonregression_ratio is not None and math.isclose(generate_nonregression_ratio, 1.0))
    )
    gate = {
        "selected_better_than_default_ratio": selected_better_ratio,
        "selection_stable_ratio": stable_ratio,
        "selection_stable_3_of_3": bool(len(repeats) >= 3 and math.isclose(stable_ratio, 1.0)),
        "selected_better_gate_passed": selected_better_ratio >= 0.70,
        "generate_nonregression_ratio": generate_nonregression_ratio,
        "generate_nonregression_passed": generate_gate_passed,
        "overall_passed": bool(
            selected_better_ratio >= 0.70
            and (len(repeats) < 3 or math.isclose(stable_ratio, 1.0))
            and generate_gate_passed
        ),
    }

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "suite_dir": str(suite_dir.resolve()),
        "video_path": args.video_path,
        "reference_path": args.reference_path,
        "ckpt_path": args.ckpt_path,
        "generate_ckpt_dir": args.generate_ckpt_dir,
        "candidate_manifest": str(Path(args.candidate_manifest).resolve()),
        "score_policy_json": str(Path(args.score_policy_json).resolve()) if args.score_policy_json else None,
        "repeat": args.repeat,
        "repeats": repeats,
        "gate": gate,
    }
    summary_json = suite_dir / "summary.json"
    summary_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    gate_result = {
        "suite_dir": str(suite_dir.resolve()),
        "candidate_manifest": str(Path(args.candidate_manifest).resolve()),
        "score_policy_json": str(Path(args.score_policy_json).resolve()) if args.score_policy_json else None,
        "selected_better_than_default_ratio": selected_better_ratio,
        "selection_stable_ratio": stable_ratio,
        "selection_stable_3_of_3": gate["selection_stable_3_of_3"],
        "generate_nonregression_ratio": generate_nonregression_ratio,
        "generate_nonregression_passed": generate_gate_passed,
        "overall_passed": gate["overall_passed"],
        "selected_candidates": selected_names,
    }
    (suite_dir / "gate_result.json").write_text(
        json.dumps(gate_result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_summary_csv(suite_dir / "summary.csv", csv_rows)

    lines = [
        "# Optimization3 Step08 Candidate Search",
        "",
        f"- suite_dir: `{suite_dir}`",
        f"- repeats: `{args.repeat}`",
        f"- selected_better_than_default_ratio: `{selected_better_ratio:.4f}`",
        f"- selection_stable_ratio: `{stable_ratio:.4f}`",
        f"- generate_nonregression_ratio: `{generate_nonregression_ratio}`",
        f"- overall_passed: `{gate['overall_passed']}`",
        "",
        "## Repeat Selection",
        "",
    ]
    for repeat in repeats:
        selection = repeat["selection_result"] or {}
        lines.append(
            f"- `repeat_{repeat['repeat_index']:02d}`: selected=`{selection.get('selected_candidate')}` default=`{selection.get('default_candidate')}` margin=`{selection.get('score_margin_vs_default')}` better=`{selection.get('selected_better_than_default')}`"
        )
    (suite_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "suite_dir": str(suite_dir.resolve()),
                "selected_better_than_default_ratio": selected_better_ratio,
                "selection_stable_ratio": stable_ratio,
                "gate_result": str((suite_dir / "gate_result.json").resolve()),
                "overall_passed": gate["overall_passed"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
