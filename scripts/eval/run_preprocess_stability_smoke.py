import argparse
import json
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PREPROCESS_SCRIPT = REPO_ROOT / "wan" / "modules" / "animate" / "preprocess" / "preprocess_data.py"
CONTRACT_SCRIPT = REPO_ROOT / "scripts" / "eval" / "check_animate_contract.py"

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

PRESETS = {
    "failed_high_legacy": {
        "resolution_area": (1280, 720),
        "fps": 15,
        "sam_chunk_len": 60,
        "sam_keyframes_per_chunk": 6,
        "sam_use_negative_points": True,
        "sam_reprompt_interval": 20,
        "sam_runtime_profile": "legacy_safe",
        "sam_apply_postprocessing": True,
    },
    "failed_low_legacy": {
        "resolution_area": (832, 480),
        "fps": 5,
        "sam_chunk_len": 20,
        "sam_keyframes_per_chunk": 4,
        "sam_use_negative_points": True,
        "sam_reprompt_interval": 10,
        "sam_runtime_profile": "legacy_safe",
        "sam_apply_postprocessing": True,
    },
    "stable_h200_safe": {
        "resolution_area": (640, 360),
        "fps": 5,
        "sam_chunk_len": 12,
        "sam_keyframes_per_chunk": 3,
        "sam_use_negative_points": False,
        "sam_reprompt_interval": 0,
        "sam_runtime_profile": "h200_safe",
        "sam_apply_postprocessing": False,
        "extra_args": [
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
        ],
    },
    "lowload_h200_safe": {
        "resolution_area": (832, 480),
        "fps": 5,
        "sam_chunk_len": 20,
        "sam_keyframes_per_chunk": 4,
        "sam_use_negative_points": False,
        "sam_reprompt_interval": 0,
        "sam_runtime_profile": "h200_safe",
        "sam_apply_postprocessing": False,
        "extra_args": [
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
        ],
    },
    "highload_h200_safe": {
        "resolution_area": (1280, 720),
        "fps": 15,
        "sam_chunk_len": 60,
        "sam_keyframes_per_chunk": 6,
        "sam_use_negative_points": False,
        "sam_reprompt_interval": 0,
        "sam_runtime_profile": "h200_safe",
        "sam_apply_postprocessing": False,
        "extra_args": [
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
        ],
    },
    "stable_h200_extreme": {
        "resolution_area": (640, 360),
        "fps": 5,
        "sam_chunk_len": 20,
        "sam_keyframes_per_chunk": 4,
        "sam_use_negative_points": False,
        "sam_reprompt_interval": 0,
        "sam_runtime_profile": "h200_safe",
        "sam_apply_postprocessing": False,
        "extra_args": [
            "--preprocess_runtime_profile",
            "h200_extreme",
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
        ],
    },
}


def _read_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_stage_status(manifest_data):
    if not manifest_data:
        return None
    stage_entries = manifest_data.get("stages", {}).get("preprocess", [])
    if not stage_entries:
        return None
    return stage_entries[-1].get("status")


def _signal_name(returncode: int):
    if returncode >= 0:
        return None
    try:
        return signal.Signals(-returncode).name
    except Exception:
        return f"SIG{-returncode}"


def _verify_outputs(run_dir: Path):
    preprocess_dir = run_dir / "preprocess"
    required = {
        "metadata": preprocess_dir / "metadata.json",
        "person_mask": None,
        "background": None,
    }
    if (preprocess_dir / "metadata.json").exists():
        metadata = _read_json(preprocess_dir / "metadata.json")
        if metadata is not None:
            artifacts = metadata.get("src_files", {})
            if "person_mask" in artifacts:
                required["person_mask"] = preprocess_dir / artifacts["person_mask"]["path"]
            if "background" in artifacts:
                required["background"] = preprocess_dir / artifacts["background"]["path"]
    qa_candidates = [
        preprocess_dir / "mask_overlay.mp4",
        preprocess_dir / "mask_overlay",
    ]
    required["qa_mask_overlay"] = next((path for path in qa_candidates if path.exists()), None)
    checks = {}
    for key, path in required.items():
        checks[key] = bool(path is not None and path.exists())
    return checks


def _run_contract_check(preprocess_dir: Path):
    command = [
        PYTHON,
        "-X",
        "faulthandler",
        str(CONTRACT_SCRIPT),
        "--src_root_path",
        str(preprocess_dir),
        "--replace_flag",
        "--skip_synthetic",
    ]
    result = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    return {
        "returncode": int(result.returncode),
        "stdout": result.stdout,
        "stderr": result.stderr,
        "passed": result.returncode == 0,
    }


def _build_command(run_dir: Path, preset_name: str, preset: dict, args):
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
        "--export_qa_visuals",
        "--sam_debug_trace",
        "--bg_inpaint_mode",
        "image",
        "--soft_mask_mode",
        "soft_band",
        "--reference_normalization_mode",
        "bbox_match",
        "--iterations",
        "2",
        "--k",
        "7",
        "--w_len",
        "4",
        "--h_len",
        "8",
        "--resolution_area",
        str(preset["resolution_area"][0]),
        str(preset["resolution_area"][1]),
        "--fps",
        str(preset["fps"]),
        "--sam_chunk_len",
        str(preset["sam_chunk_len"]),
        "--sam_keyframes_per_chunk",
        str(preset["sam_keyframes_per_chunk"]),
        "--sam_reprompt_interval",
        str(preset["sam_reprompt_interval"]),
        "--sam_runtime_profile",
        preset["sam_runtime_profile"],
    ]
    if preset.get("sam_apply_postprocessing", True):
        command.append("--sam_apply_postprocessing")
    else:
        command.append("--no-sam_apply_postprocessing")
    if preset["sam_use_negative_points"]:
        command.append("--sam_use_negative_points")
    else:
        command.append("--no-sam_use_negative_points")
    command.extend(preset.get("extra_args", []))
    return command


def run_case(suite_dir: Path, case_name: str, preset_name: str, preset: dict, args):
    run_dir = suite_dir / case_name
    run_dir.mkdir(parents=True, exist_ok=True)
    command = _build_command(run_dir, preset_name, preset, args)
    result = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        env={**os.environ, "PYTHONFAULTHANDLER": "1"},
    )
    (run_dir / "runner_stdout.log").write_text(result.stdout, encoding="utf-8")
    (run_dir / "runner_stderr.log").write_text(result.stderr, encoding="utf-8")

    manifest_path = run_dir / "manifest.json"
    preprocess_dir = run_dir / "preprocess"
    trace_dir = preprocess_dir / "sam2_debug"
    manifest_data = _read_json(manifest_path)
    latest_trace = _read_json(trace_dir / "latest_state.json")
    output_checks = _verify_outputs(run_dir)
    contract_result = None
    if preprocess_dir.exists() and (preprocess_dir / "metadata.json").exists():
        contract_result = _run_contract_check(preprocess_dir)

    summary = {
        "case_name": case_name,
        "preset_name": preset_name,
        "run_dir": str(run_dir),
        "command": command,
        "returncode": int(result.returncode),
        "signal": _signal_name(result.returncode),
        "manifest_status": _resolve_stage_status(manifest_data),
        "manifest_path": str(manifest_path) if manifest_path.exists() else None,
        "trace_dir": str(trace_dir) if trace_dir.exists() else None,
        "latest_trace": latest_trace,
        "output_checks": output_checks,
        "contract_check": contract_result,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run a real-video SAM2 preprocess stability smoke suite.")
    parser.add_argument("--video_path", type=str, default=DEFAULT_VIDEO_PATH)
    parser.add_argument("--reference_path", type=str, default=DEFAULT_REFERENCE_PATH)
    parser.add_argument("--ckpt_path", type=str, default=DEFAULT_CKPT_PATH)
    parser.add_argument(
        "--preset",
        action="append",
        choices=sorted(PRESETS.keys()),
        default=None,
        help="Smoke preset to execute. Can be repeated. Defaults to stable_h200_safe.",
    )
    parser.add_argument("--repeat", type=int, default=1, help="Repeat each selected preset this many times.")
    parser.add_argument("--suite_name", type=str, default=None, help="Optional suite directory name under ./runs.")
    args = parser.parse_args()

    selected_presets = args.preset or ["stable_h200_safe"]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suite_name = args.suite_name or f"sam2_preprocess_stability_smoke_{timestamp}"
    suite_dir = REPO_ROOT / "runs" / suite_name
    suite_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for preset_name in selected_presets:
        preset = PRESETS[preset_name]
        for repeat_index in range(args.repeat):
            case_name = f"{preset_name}_r{repeat_index + 1:02d}"
            print(f"[run] {case_name}")
            summary = run_case(suite_dir, case_name, preset_name, preset, args)
            summaries.append(summary)
            status = summary["manifest_status"] or summary["returncode"]
            print(f"[done] {case_name}: {status}")

    suite_summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "suite_dir": str(suite_dir),
        "video_path": args.video_path,
        "reference_path": args.reference_path,
        "ckpt_path": args.ckpt_path,
        "cases": summaries,
    }
    summary_path = suite_dir / "summary.json"
    summary_path.write_text(json.dumps(suite_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
