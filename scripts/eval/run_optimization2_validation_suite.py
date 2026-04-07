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
PYTHON = sys.executable

DEFAULT_VIDEO_PATH = "/home/user1/wan2.2/test1/260306_01_step_1_0000_0010.mp4"
DEFAULT_REFERENCE_PATH = "/home/user1/wan2.2/test1/Gemini_Generated_Image_2.png"
DEFAULT_CKPT_PATH = "/home/user1/wan2.2/Wan2.2/Wan2.2-Animate-14B/process_checkpoint"

PREPROCESS_SCRIPT = REPO_ROOT / "wan" / "modules" / "animate" / "preprocess" / "preprocess_data.py"
GENERATE_SCRIPT = REPO_ROOT / "generate.py"
COMPUTE_REPLACEMENT_METRICS = REPO_ROOT / "scripts" / "eval" / "compute_replacement_metrics.py"
COMPUTE_BOUNDARY_METRICS = REPO_ROOT / "scripts" / "eval" / "compute_boundary_refinement_metrics.py"
PROFILE_BENCHMARK = REPO_ROOT / "scripts" / "eval" / "run_preprocess_profile_benchmark.py"
STABILITY_SMOKE = REPO_ROOT / "scripts" / "eval" / "run_preprocess_stability_smoke.py"
CONTRACT_CHECK = REPO_ROOT / "scripts" / "eval" / "check_animate_contract.py"


def _read_json(path: Path):
    if not path.exists():
        return None
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


def _load_preprocess_metadata(preprocess_dir: Path):
    return _read_json(preprocess_dir / "metadata.json")


def _load_preprocess_runtime(preprocess_dir: Path):
    return _read_json(preprocess_dir / "preprocess_runtime_stats.json")


def _load_generate_runtime(run_dir: Path):
    return _read_json(run_dir / "debug" / "generate" / "wan_animate_runtime_stats.json")


def _build_preprocess_command(
    *,
    run_dir: Path,
    ckpt_path: str,
    video_path: str,
    reference_path: str,
    bg_inpaint_mode: str,
    reference_mode: str,
):
    return [
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
        "--export_qa_visuals",
        "--sam_debug_trace",
        "--bg_inpaint_mode",
        bg_inpaint_mode,
        "--soft_mask_mode",
        "soft_band",
        "--reference_normalization_mode",
        reference_mode,
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
        "--sam_runtime_profile",
        "h200_safe",
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
        "heuristic",
        "--parsing_mode",
        "heuristic",
        "--matting_mode",
        "heuristic",
        "--bg_video_window_radius",
        "4",
        "--bg_video_min_visible_count",
        "2",
        "--bg_video_blend_strength",
        "0.7",
        "--bg_temporal_smooth_strength",
        "0.1",
    ]


def _build_generate_command(
    *,
    run_dir: Path,
    ckpt_dir: str,
    src_root_path: Path,
    boundary_refine_mode: str,
):
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


def _find_output_video(run_dir: Path):
    outputs_dir = run_dir / "outputs"
    for candidate in sorted(outputs_dir.glob("*.mkv")):
        return candidate
    for candidate in sorted(outputs_dir.glob("*.mp4")):
        return candidate
    return None


def _contract_check_command(preprocess_dir: Path):
    return [
        PYTHON,
        str(CONTRACT_CHECK),
        "--src_root_path",
        str(preprocess_dir),
        "--replace_flag",
        "--skip_synthetic",
    ]


def _case_passed(case):
    return bool(case.get("passed", False))


def _compare_background_stats(image_meta: dict, video_meta: dict):
    image_stats = image_meta["processing"]["background"]["stats"]["stats"]
    video_stats = video_meta["processing"]["background"]["stats"]["stats"]
    return {
        "image_mode": image_meta["src_files"]["background"]["background_mode"],
        "video_mode": video_meta["src_files"]["background"]["background_mode"],
        "image_temporal_fluctuation_mean": image_stats.get("temporal_fluctuation_mean"),
        "video_temporal_fluctuation_mean": video_stats.get("temporal_fluctuation_mean"),
        "image_band_adjacent_background_stability": image_stats.get("band_adjacent_background_stability"),
        "video_band_adjacent_background_stability": video_stats.get("band_adjacent_background_stability"),
        "video_better_temporal_fluctuation": (
            video_stats.get("temporal_fluctuation_mean") is not None
            and image_stats.get("temporal_fluctuation_mean") is not None
            and video_stats["temporal_fluctuation_mean"] < image_stats["temporal_fluctuation_mean"]
        ),
    }


def _compare_reference_stats(bbox_meta: dict, structure_meta: dict):
    bbox_stats = bbox_meta["processing"]["reference_normalization"]["stats"]
    structure_stats = structure_meta["processing"]["reference_normalization"]["stats"]

    def _bbox_ratio(bbox):
        if bbox is None:
            return None
        return float((bbox[2] - bbox[0]) / max((bbox[3] - bbox[1]), 1e-6))

    return {
        "bbox_mode": bbox_meta["processing"]["reference_normalization"]["reference_normalization_mode"],
        "structure_mode": structure_meta["processing"]["reference_normalization"]["reference_normalization_mode"],
        "bbox_target_ratio": _bbox_ratio(bbox_stats.get("target_bbox")),
        "bbox_normalized_ratio": _bbox_ratio(bbox_stats.get("normalized_bbox")),
        "structure_target_ratio": structure_stats.get("target_structure", {}).get("width_height_ratio"),
        "structure_normalized_ratio": structure_stats.get("normalized_structure", {}).get("width_height_ratio"),
        "structure_budget_flags": {
            "width_budget_triggered": structure_stats.get("width_budget_triggered"),
            "height_budget_triggered": structure_stats.get("height_budget_triggered"),
        },
    }


def _write_summary(suite_dir: Path, payload: dict):
    summary_json = suite_dir / "summary.json"
    summary_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Optimization2 Validation Summary",
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
    if payload.get("comparisons"):
        lines.extend(["", "## Comparisons", ""])
        for key, value in payload["comparisons"].items():
            lines.append(f"- `{key}`: `{json.dumps(value, ensure_ascii=False, sort_keys=True)}`")
    summary_md = suite_dir / "summary.md"
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Run an optimization2 validation suite.")
    parser.add_argument("--tier", choices=["core", "extended"], default="core")
    parser.add_argument("--suite_name", type=str, default=None)
    parser.add_argument("--video_path", type=str, default=DEFAULT_VIDEO_PATH)
    parser.add_argument("--reference_path", type=str, default=DEFAULT_REFERENCE_PATH)
    parser.add_argument("--ckpt_path", type=str, default=DEFAULT_CKPT_PATH)
    parser.add_argument("--ckpt_dir", type=str, default=str(Path(DEFAULT_CKPT_PATH).parent))
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suite_name = args.suite_name or f"optimization2_validation_{args.tier}_{timestamp}"
    suite_dir = RUNS_ROOT / suite_name
    suite_dir.mkdir(parents=True, exist_ok=True)

    cases = []
    comparisons = {}

    compile_targets = [
        "generate.py",
        "wan/animate.py",
        "wan/modules/animate/preprocess/preprocess_data.py",
        "wan/modules/animate/preprocess/process_pipepline.py",
        "wan/modules/animate/preprocess/background_clean_plate.py",
        "wan/modules/animate/preprocess/reference_normalization.py",
        "wan/utils/boundary_refinement.py",
        "scripts/eval/check_sam_prompting.py",
        "scripts/eval/check_control_stability.py",
        "scripts/eval/check_boundary_fusion.py",
        "scripts/eval/check_clean_plate_background.py",
        "scripts/eval/check_video_clean_plate_background.py",
        "scripts/eval/check_reference_normalization.py",
        "scripts/eval/check_boundary_refinement.py",
        "scripts/eval/check_soft_mask_pipeline.py",
    ]
    cases.append(
        _run_command(
            "compileall",
            [PYTHON, "-m", "compileall", *compile_targets],
            suite_dir=suite_dir,
        )
    )

    generate_help = _run_command("generate_help", [PYTHON, str(GENERATE_SCRIPT), "--help"], suite_dir=suite_dir)
    generate_help = _assert_in_help(
        generate_help,
        Path(generate_help["stdout_path"]),
        ["boundary_refine_mode", "guidance_uncond_mode", "temporal_handoff_mode", "replacement_mask_mode"],
    )
    cases.append(generate_help)

    preprocess_help = _run_command("preprocess_help", [PYTHON, str(PREPROCESS_SCRIPT), "--help"], suite_dir=suite_dir)
    preprocess_help = _assert_in_help(
        preprocess_help,
        Path(preprocess_help["stdout_path"]),
        ["sam_runtime_profile", "bg_inpaint_mode", "reference_normalization_mode", "reference_structure_segment_clamp_min"],
    )
    cases.append(preprocess_help)

    synthetic_scripts = [
        "check_sam_prompting.py",
        "check_control_stability.py",
        "check_boundary_fusion.py",
        "check_clean_plate_background.py",
        "check_video_clean_plate_background.py",
        "check_reference_normalization.py",
        "check_boundary_refinement.py",
        "check_soft_mask_pipeline.py",
    ]
    for script_name in synthetic_scripts:
        cases.append(
            _run_command(
                script_name.replace(".py", ""),
                [PYTHON, str(REPO_ROOT / "scripts" / "eval" / script_name)],
                suite_dir=suite_dir,
            )
        )

    preprocess_structure_run = suite_dir / "preprocess_structure_video"
    preprocess_structure = _run_command(
        "preprocess_structure_video",
        _build_preprocess_command(
            run_dir=preprocess_structure_run,
            ckpt_path=args.ckpt_path,
            video_path=args.video_path,
            reference_path=args.reference_path,
            bg_inpaint_mode="video",
            reference_mode="structure_match",
        ),
        suite_dir=suite_dir,
    )
    cases.append(preprocess_structure)

    preprocess_dir = preprocess_structure_run / "preprocess"
    structure_contract = _run_command(
        "contract_preprocess_structure_video",
        _contract_check_command(preprocess_dir),
        suite_dir=suite_dir,
    )
    cases.append(structure_contract)

    none_run = suite_dir / "generate_none"
    refined_run = suite_dir / "generate_refined"
    gen_none = _run_command(
        "generate_none",
        _build_generate_command(
            run_dir=none_run,
            ckpt_dir=args.ckpt_dir,
            src_root_path=preprocess_dir,
            boundary_refine_mode="none",
        ),
        suite_dir=suite_dir,
    )
    cases.append(gen_none)
    gen_refined = _run_command(
        "generate_refined",
        _build_generate_command(
            run_dir=refined_run,
            ckpt_dir=args.ckpt_dir,
            src_root_path=preprocess_dir,
            boundary_refine_mode="deterministic",
        ),
        suite_dir=suite_dir,
    )
    cases.append(gen_refined)

    metadata = _load_preprocess_metadata(preprocess_dir)
    person_mask_artifact = metadata["src_files"]["person_mask"]
    person_mask_path = preprocess_dir / person_mask_artifact["path"]

    refined_output = _find_output_video(refined_run)
    none_output = _find_output_video(none_run)
    replacement_metrics_json = suite_dir / "metrics" / "replacement_metrics.json"
    replacement_metrics_json.parent.mkdir(parents=True, exist_ok=True)
    replacement_metrics = _run_command(
        "compute_replacement_metrics_refined",
        [
            PYTHON,
            str(COMPUTE_REPLACEMENT_METRICS),
            "--video_path",
            str(refined_output),
            "--mask_path",
            str(person_mask_path),
            "--clip_len",
            "45",
            "--refert_num",
            "9",
            "--output_json",
            str(replacement_metrics_json),
        ],
        suite_dir=suite_dir,
    )
    cases.append(replacement_metrics)

    boundary_metrics_json = suite_dir / "metrics" / "boundary_refinement_metrics.json"
    boundary_metrics = _run_command(
        "compute_boundary_refinement_metrics",
        [
            PYTHON,
            str(COMPUTE_BOUNDARY_METRICS),
            "--before",
            str(none_output),
            "--after",
            str(refined_output),
            "--src_root_path",
            str(preprocess_dir),
            "--output_json",
            str(boundary_metrics_json),
        ],
        suite_dir=suite_dir,
    )
    cases.append(boundary_metrics)

    structure_runtime = _load_generate_runtime(refined_run)
    comparisons["structure_generate_runtime"] = {
        "background_mode": None if structure_runtime is None else structure_runtime.get("background_mode"),
        "reference_normalization_mode": None if structure_runtime is None else structure_runtime.get("reference_normalization_mode"),
        "total_generate_sec": None if structure_runtime is None else structure_runtime.get("total_generate_sec"),
    }

    if args.tier == "extended":
        preprocess_image_run = suite_dir / "preprocess_structure_image"
        preprocess_bbox_run = suite_dir / "preprocess_bbox_video"

        cases.append(
            _run_command(
                "preprocess_structure_image",
                _build_preprocess_command(
                    run_dir=preprocess_image_run,
                    ckpt_path=args.ckpt_path,
                    video_path=args.video_path,
                    reference_path=args.reference_path,
                    bg_inpaint_mode="image",
                    reference_mode="structure_match",
                ),
                suite_dir=suite_dir,
            )
        )
        cases.append(
            _run_command(
                "preprocess_bbox_video",
                _build_preprocess_command(
                    run_dir=preprocess_bbox_run,
                    ckpt_path=args.ckpt_path,
                    video_path=args.video_path,
                    reference_path=args.reference_path,
                    bg_inpaint_mode="video",
                    reference_mode="bbox_match",
                ),
                suite_dir=suite_dir,
            )
        )

        image_meta = _load_preprocess_metadata(preprocess_image_run / "preprocess")
        video_meta = _load_preprocess_metadata(preprocess_dir)
        bbox_meta = _load_preprocess_metadata(preprocess_bbox_run / "preprocess")
        structure_meta = _load_preprocess_metadata(preprocess_dir)
        comparisons["clean_plate_ab"] = _compare_background_stats(image_meta, video_meta)
        comparisons["reference_normalization_ab"] = _compare_reference_stats(bbox_meta, structure_meta)

        profile_suite_name = f"{suite_name}_profile_matrix"
        stability_suite_name = f"{suite_name}_stability"
        cases.append(
            _run_command(
                "profile_benchmark",
                [
                    PYTHON,
                    str(PROFILE_BENCHMARK),
                    "--video_path",
                    args.video_path,
                    "--reference_path",
                    args.reference_path,
                    "--ckpt_path",
                    args.ckpt_path,
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
                    profile_suite_name,
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
                    args.video_path,
                    "--reference_path",
                    args.reference_path,
                    "--ckpt_path",
                    args.ckpt_path,
                    "--preset",
                    "stable_h200_safe",
                    "--repeat",
                    "2",
                    "--suite_name",
                    stability_suite_name,
                ],
                suite_dir=suite_dir,
            )
        )

        comparisons["profile_benchmark_summary"] = {
            "summary_json": str((RUNS_ROOT / profile_suite_name / "summary.json").resolve()),
        }
        comparisons["stability_smoke_summary"] = {
            "summary_json": str((RUNS_ROOT / stability_suite_name / "summary.json").resolve()),
        }

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "suite_dir": str(suite_dir.resolve()),
        "tier": args.tier,
        "video_path": args.video_path,
        "reference_path": args.reference_path,
        "ckpt_path": args.ckpt_path,
        "cases": cases,
        "comparisons": comparisons,
        "all_passed": all(_case_passed(case) for case in cases),
    }
    _write_summary(suite_dir, payload)
    print(json.dumps({"suite_dir": str(suite_dir), "all_passed": payload["all_passed"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
