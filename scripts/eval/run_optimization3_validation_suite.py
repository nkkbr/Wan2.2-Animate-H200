#!/usr/bin/env python
import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = REPO_ROOT / "runs"


def _default_python_bin() -> str:
    override = os.environ.get("WAN_PYTHON")
    candidates = [override, "/home/user1/miniconda3/envs/wan/bin/python", sys.executable]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(Path(candidate).resolve())
    return sys.executable


PYTHON = _default_python_bin()

DEFAULT_MANIFEST = REPO_ROOT / "docs" / "optimization3" / "benchmark" / "benchmark_manifest.example.json"
DEFAULT_GATE_POLICY = REPO_ROOT / "docs" / "optimization3" / "benchmark" / "gate_policy.step01.json"

PREPROCESS_SCRIPT = REPO_ROOT / "wan" / "modules" / "animate" / "preprocess" / "preprocess_data.py"
GENERATE_SCRIPT = REPO_ROOT / "generate.py"
COMPUTE_REPLACEMENT_METRICS = REPO_ROOT / "scripts" / "eval" / "compute_replacement_metrics.py"
COMPUTE_BOUNDARY_REFINEMENT = REPO_ROOT / "scripts" / "eval" / "compute_boundary_refinement_metrics.py"
COMPUTE_BOUNDARY_PRECISION = REPO_ROOT / "scripts" / "eval" / "compute_boundary_precision_metrics.py"
COMPUTE_FACE_PRECISION = REPO_ROOT / "scripts" / "eval" / "compute_face_precision_metrics.py"
COMPUTE_POSE_PRECISION = REPO_ROOT / "scripts" / "eval" / "compute_pose_precision_metrics.py"
COMPUTE_BACKGROUND_PRECISION = REPO_ROOT / "scripts" / "eval" / "compute_background_precision_metrics.py"
CONTRACT_CHECK = REPO_ROOT / "scripts" / "eval" / "check_animate_contract.py"
SUMMARIZE = REPO_ROOT / "scripts" / "eval" / "summarize_optimization3_validation.py"
EXTRACT_KEYFRAMES = REPO_ROOT / "scripts" / "eval" / "extract_benchmark_keyframes.py"


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _run_command(name: str, command: list[str], *, suite_dir: Path, cwd: Path = REPO_ROOT):
    log_dir = suite_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"{name}.stdout.log"
    stderr_path = log_dir / f"{name}.stderr.log"
    started = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        env={**os.environ, "PYTHONFAULTHANDLER": "1"},
    )
    elapsed = time.perf_counter() - started
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    return {
        "name": name,
        "command": command,
        "command_pretty": shlex.join(command),
        "returncode": int(result.returncode),
        "duration_sec": float(elapsed),
        "stdout_path": str(stdout_path.resolve()),
        "stderr_path": str(stderr_path.resolve()),
        "passed": result.returncode == 0,
    }


def _assert_in_help(result: dict, path: Path, expected_tokens: list[str]):
    text = path.read_text(encoding="utf-8")
    missing = [token for token in expected_tokens if token not in text]
    result["missing_tokens"] = missing
    result["passed"] = result["passed"] and not missing
    return result


def _write_summary(suite_dir: Path, payload: dict):
    summary_json = suite_dir / "summary.json"
    summary_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Optimization3 Validation Summary",
        "",
        f"- tier: `{payload['tier']}`",
        f"- suite_dir: `{suite_dir}`",
        f"- created_at_utc: `{payload['created_at_utc']}`",
        "",
        "## Cases",
        "",
    ]
    for case in payload["cases"]:
        status = "PASS" if case.get("passed") else "FAIL"
        lines.append(f"- `{case['name']}`: {status} ({case.get('duration_sec', 0.0):.2f}s)")
    if payload.get("benchmark"):
        lines.extend(["", "## Benchmark Sets", ""])
        for key, items in payload["benchmark"].items():
            lines.append(f"- `{key}`: `{len(items)}` cases")
    if payload.get("comparisons"):
        lines.extend(["", "## Comparisons", ""])
        for key, value in payload["comparisons"].items():
            lines.append(f"- `{key}`: `{json.dumps(value, ensure_ascii=False, sort_keys=True)}`")
    (suite_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_preprocess_command(
    *,
    run_dir: Path,
    ckpt_path: str,
    video_path: str,
    reference_path: str,
    preprocess_runtime_profile: str,
    bg_inpaint_mode: str,
    multistage_preprocess_mode: str,
    disable_person_roi_refine: bool,
    disable_face_roi_refine: bool,
):
    command = [
        PYTHON,
        "-X",
        "faulthandler",
        str(PREPROCESS_SCRIPT),
        "--ckpt_path",
        ckpt_path,
        "--video_path",
        video_path,
        "--refer_path",
        reference_path,
        "--run_dir",
        str(run_dir),
        "--save_manifest",
        "--replace_flag",
        "--lossless_intermediate",
        "--bg_inpaint_mode",
        bg_inpaint_mode,
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
        "--iterations",
        "2",
        "--k",
        "7",
        "--w_len",
        "4",
        "--h_len",
        "8",
        "--resolution_area",
        "640",
        "360",
        "--fps",
        "5",
        "--sam_chunk_len",
        "12",
        "--sam_keyframes_per_chunk",
        "3",
        "--sam_reprompt_interval",
        "0",
        "--preprocess_runtime_profile",
        preprocess_runtime_profile,
        "--sam_runtime_profile",
        "h200_safe" if preprocess_runtime_profile == "h200_extreme" else preprocess_runtime_profile,
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
        "--boundary_fusion_mode",
        "v2",
        "--parsing_mode",
        "heuristic",
        "--matting_mode",
        "heuristic",
        "--bg_temporal_smooth_strength",
        "0.1",
    ]
    if multistage_preprocess_mode != "none":
        command.extend(["--multistage_preprocess_mode", multistage_preprocess_mode])
        if disable_person_roi_refine:
            command.append("--disable_person_roi_refine")
        if disable_face_roi_refine:
            command.append("--disable_face_roi_refine")
    return command


def _build_generate_command(*, run_dir: Path, ckpt_dir: str, src_root_path: Path, boundary_refine_mode: str):
    output_name = "replacement_smoke_refined.mkv" if boundary_refine_mode == "deterministic" else "replacement_smoke_none.mkv"
    return [
        PYTHON,
        str(GENERATE_SCRIPT),
        "--task",
        "animate-14B",
        "--ckpt_dir",
        ckpt_dir,
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
        "--boundary_refine_mode",
        boundary_refine_mode,
        "--save_debug_dir",
        str(run_dir / "debug" / "generate"),
        "--save_file",
        str(run_dir / "outputs" / output_name),
        "--output_format",
        "ffv1",
        "--base_seed",
        "123456",
    ]


def _contract_check_command(preprocess_dir: Path):
    return [
        PYTHON,
        str(CONTRACT_CHECK),
        "--src_root_path",
        str(preprocess_dir),
        "--replace_flag",
        "--skip_synthetic",
    ]


def _find_output_video(run_dir: Path):
    for candidate in sorted((run_dir / "outputs").glob("*.mkv")):
        return candidate
    for candidate in sorted((run_dir / "outputs").glob("*.mp4")):
        return candidate
    return None


def main():
    parser = argparse.ArgumentParser(description="Run an optimization3 validation suite.")
    parser.add_argument("--tier", choices=["core", "extended"], default="core")
    parser.add_argument("--suite_name", type=str, default=None)
    parser.add_argument("--manifest", type=str, default=str(DEFAULT_MANIFEST))
    parser.add_argument("--gate_policy", type=str, default=str(DEFAULT_GATE_POLICY))
    parser.add_argument("--preprocess_runtime_profile", choices=["h200_safe", "h200_aggressive", "h200_extreme"], default="h200_safe")
    parser.add_argument("--bg_inpaint_mode", choices=["image", "video"], default="image")
    parser.add_argument("--multistage_preprocess_mode", choices=["none", "h200_extreme"], default="none")
    parser.add_argument("--disable_person_roi_refine", action="store_true", default=False)
    parser.add_argument("--disable_face_roi_refine", action="store_true", default=False)
    args = parser.parse_args()

    manifest = _read_json(Path(args.manifest))
    shared = manifest["shared"]
    smoke_case = manifest["smoke_cases"][0]

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suite_name = args.suite_name or f"optimization3_validation_{args.tier}_{timestamp}"
    suite_dir = RUNS_ROOT / suite_name
    suite_dir.mkdir(parents=True, exist_ok=True)

    cases = []
    comparisons = {}
    benchmark = {
        "smoke_cases": manifest.get("smoke_cases", []),
        "labeled_cases": manifest.get("labeled_cases", []),
        "stress_cases": manifest.get("stress_cases", []),
    }

    compile_targets = [
        "generate.py",
        "wan/animate.py",
        "wan/modules/animate/preprocess/preprocess_data.py",
        "wan/modules/animate/preprocess/process_pipepline.py",
        "wan/modules/animate/preprocess/face_analysis.py",
        "wan/modules/animate/preprocess/background_clean_plate.py",
        "wan/modules/animate/preprocess/reference_normalization.py",
        "wan/utils/boundary_refinement.py",
        "scripts/eval/check_face_analysis_stack.py",
        "scripts/eval/check_sam_prompting.py",
        "scripts/eval/check_control_stability.py",
        "scripts/eval/check_boundary_fusion.py",
        "scripts/eval/check_clean_plate_background.py",
        "scripts/eval/check_video_clean_plate_background.py",
        "scripts/eval/check_reference_normalization.py",
        "scripts/eval/check_boundary_refinement.py",
        "scripts/eval/check_soft_mask_pipeline.py",
        "scripts/eval/compute_boundary_precision_metrics.py",
        "scripts/eval/compute_face_precision_metrics.py",
        "scripts/eval/compute_pose_precision_metrics.py",
        "scripts/eval/compute_background_precision_metrics.py",
        "scripts/eval/summarize_optimization3_validation.py",
    ]
    cases.append(_run_command("compileall", [PYTHON, "-m", "compileall", *compile_targets], suite_dir=suite_dir))

    generate_help = _run_command("generate_help", [PYTHON, str(GENERATE_SCRIPT), "--help"], suite_dir=suite_dir)
    cases.append(_assert_in_help(generate_help, Path(generate_help["stdout_path"]), ["boundary_refine_mode", "guidance_uncond_mode", "temporal_handoff_mode"]))

    preprocess_help = _run_command("preprocess_help", [PYTHON, str(PREPROCESS_SCRIPT), "--help"], suite_dir=suite_dir)
    cases.append(_assert_in_help(preprocess_help, Path(preprocess_help["stdout_path"]), ["sam_runtime_profile", "bg_inpaint_mode", "reference_normalization_mode", "face_analysis_mode"]))

    for script_name in [
        "check_face_analysis_stack.py",
        "check_sam_prompting.py",
        "check_control_stability.py",
        "check_boundary_fusion.py",
        "check_clean_plate_background.py",
        "check_video_clean_plate_background.py",
        "check_reference_normalization.py",
        "check_boundary_refinement.py",
        "check_soft_mask_pipeline.py",
    ]:
        cases.append(_run_command(script_name.replace(".py", ""), [PYTHON, str(REPO_ROOT / "scripts" / "eval" / script_name)], suite_dir=suite_dir))

    keyframe_dir = suite_dir / "labeled_keyframes"
    cases.append(
        _run_command(
            "extract_benchmark_keyframes",
            [
                PYTHON,
                str(EXTRACT_KEYFRAMES),
                "--manifest",
                str(Path(args.manifest).resolve()),
                "--output_dir",
                str(keyframe_dir),
            ],
            suite_dir=suite_dir,
        )
    )

    preprocess_run = suite_dir / "preprocess_structure_smoke"
    cases.append(
        _run_command(
            "preprocess_structure_smoke",
            _build_preprocess_command(
                run_dir=preprocess_run,
                ckpt_path=shared["process_checkpoint"],
                video_path=smoke_case["video_path"],
                reference_path=shared["reference_image"],
                preprocess_runtime_profile=args.preprocess_runtime_profile,
                bg_inpaint_mode=args.bg_inpaint_mode,
                multistage_preprocess_mode=args.multistage_preprocess_mode,
                disable_person_roi_refine=args.disable_person_roi_refine,
                disable_face_roi_refine=args.disable_face_roi_refine,
            ),
            suite_dir=suite_dir,
        )
    )
    preprocess_dir = preprocess_run / "preprocess"
    cases.append(_run_command("contract_preprocess_structure_smoke", _contract_check_command(preprocess_dir), suite_dir=suite_dir))

    none_run = suite_dir / "generate_none"
    refined_run = suite_dir / "generate_refined"
    cases.append(_run_command("generate_none", _build_generate_command(run_dir=none_run, ckpt_dir=shared["generate_ckpt_dir"], src_root_path=preprocess_dir, boundary_refine_mode="none"), suite_dir=suite_dir))
    cases.append(_run_command("generate_refined", _build_generate_command(run_dir=refined_run, ckpt_dir=shared["generate_ckpt_dir"], src_root_path=preprocess_dir, boundary_refine_mode="deterministic"), suite_dir=suite_dir))

    metrics_dir = suite_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metadata = _read_json(preprocess_dir / "metadata.json")
    person_mask_path = preprocess_dir / metadata["src_files"]["person_mask"]["path"]
    refined_output = _find_output_video(refined_run)
    none_output = _find_output_video(none_run)

    cases.append(
        _run_command(
            "compute_replacement_metrics_refined",
            [PYTHON, str(COMPUTE_REPLACEMENT_METRICS), "--video_path", str(refined_output), "--mask_path", str(person_mask_path), "--clip_len", "45", "--refert_num", "9", "--output_json", str(metrics_dir / "replacement_metrics.json")],
            suite_dir=suite_dir,
        )
    )
    cases.append(
        _run_command(
            "compute_boundary_refinement_metrics",
            [PYTHON, str(COMPUTE_BOUNDARY_REFINEMENT), "--before", str(none_output), "--after", str(refined_output), "--src_root_path", str(preprocess_dir), "--output_json", str(metrics_dir / "boundary_refinement_metrics.json")],
            suite_dir=suite_dir,
        )
    )
    cases.append(_run_command("compute_boundary_precision_metrics", [PYTHON, str(COMPUTE_BOUNDARY_PRECISION), "--src_root_path", str(preprocess_dir), "--output_json", str(metrics_dir / "boundary_precision_metrics.json")], suite_dir=suite_dir))
    cases.append(_run_command("compute_face_precision_metrics", [PYTHON, str(COMPUTE_FACE_PRECISION), "--src_root_path", str(preprocess_dir), "--output_json", str(metrics_dir / "face_precision_metrics.json")], suite_dir=suite_dir))
    cases.append(_run_command("compute_pose_precision_metrics", [PYTHON, str(COMPUTE_POSE_PRECISION), "--src_root_path", str(preprocess_dir), "--output_json", str(metrics_dir / "pose_precision_metrics.json")], suite_dir=suite_dir))
    cases.append(_run_command("compute_background_precision_metrics", [PYTHON, str(COMPUTE_BACKGROUND_PRECISION), "--src_root_path", str(preprocess_dir), "--output_json", str(metrics_dir / "background_precision_metrics.json")], suite_dir=suite_dir))

    if args.tier == "extended":
        from pathlib import Path as _P
        PROFILE_BENCHMARK = REPO_ROOT / "scripts" / "eval" / "run_preprocess_profile_benchmark.py"
        STABILITY_SMOKE = REPO_ROOT / "scripts" / "eval" / "run_preprocess_stability_smoke.py"
        cases.append(
            _run_command(
                "profile_benchmark",
                [
                    PYTHON,
                    str(PROFILE_BENCHMARK),
                    "--video_path",
                    smoke_case["video_path"],
                    "--reference_path",
                    shared["reference_image"],
                    "--ckpt_path",
                    shared["process_checkpoint"],
                    "--profile",
                    "h200_safe",
                    "--profile",
                    "h200_aggressive",
                    "--repeat",
                    "1",
                    "--resolution_area",
                    "640",
                    "360",
                    "--fps",
                    "5",
                    "--suite_name",
                    f"{suite_name}_profile_matrix",
                ],
                suite_dir=suite_dir,
            )
        )
        cases.append(
            _run_command(
                "stability_smoke",
                [
                    PYTHON,
                    str(STABILITY_SMOKE),
                    "--video_path",
                    smoke_case["video_path"],
                    "--reference_path",
                    shared["reference_image"],
                    "--ckpt_path",
                    shared["process_checkpoint"],
                    "--preset",
                    "stable_h200_safe",
                    "--repeat",
                    "2",
                    "--suite_name",
                    f"{suite_name}_stability",
                ],
                suite_dir=suite_dir,
            )
        )

    runtime_stats = json.loads((refined_run / "debug" / "generate" / "wan_animate_runtime_stats.json").read_text(encoding="utf-8"))
    comparisons["generate_runtime"] = {
        "background_mode": runtime_stats.get("background_mode"),
        "reference_normalization_mode": runtime_stats.get("reference_normalization_mode"),
        "total_generate_sec": runtime_stats.get("total_generate_sec"),
        "peak_memory_gb": runtime_stats.get("peak_memory_gb"),
    }

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "suite_dir": str(suite_dir.resolve()),
        "tier": args.tier,
        "manifest_path": str(Path(args.manifest).resolve()),
        "gate_policy_path": str(Path(args.gate_policy).resolve()),
        "cases": cases,
        "benchmark": benchmark,
        "comparisons": comparisons,
        "all_passed": all(case.get("passed", False) for case in cases),
    }
    _write_summary(suite_dir, payload)

    gate_result = suite_dir / "gate_result.json"
    _run_command(
        "gate_summary",
        [PYTHON, str(SUMMARIZE), "--summary_json", str(suite_dir / "summary.json"), "--gate_policy", str(args.gate_policy), "--output_json", str(gate_result)],
        suite_dir=suite_dir,
    )
    payload = _read_json(suite_dir / "summary.json")
    payload["gate_result_path"] = str(gate_result.resolve())
    _write_summary(suite_dir, payload)
    print(json.dumps({"suite_dir": str(suite_dir.resolve()), "all_passed": payload["all_passed"], "gate_result": str(gate_result.resolve())}))


if __name__ == "__main__":
    main()
