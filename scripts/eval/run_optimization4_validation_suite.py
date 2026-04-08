#!/usr/bin/env python
import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VIDEO = "/home/user1/wan2.2/test1/260306_01_step_1_0000_0010.mp4"
DEFAULT_REFER = "/home/user1/wan2.2/test1/Gemini_Generated_Image_2.png"
DEFAULT_CKPT = "/home/user1/wan2.2/Wan2.2/Wan2.2-Animate-14B/process_checkpoint"
DEFAULT_MANIFEST = ROOT / "docs/optimization4/benchmark/benchmark_manifest.step01.json"
DEFAULT_GATE = ROOT / "docs/optimization4/benchmark/gate_policy.step01.json"


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _run(cmd, cwd=ROOT):
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Run optimization4 Step 01 validation suite.")
    parser.add_argument("--suite_name", required=True, type=str)
    parser.add_argument("--manifest", type=str, default=str(DEFAULT_MANIFEST))
    parser.add_argument("--gate_policy", type=str, default=str(DEFAULT_GATE))
    parser.add_argument("--video_path", type=str, default=DEFAULT_VIDEO)
    parser.add_argument("--refer_path", type=str, default=DEFAULT_REFER)
    parser.add_argument("--ckpt_path", type=str, default=DEFAULT_CKPT)
    parser.add_argument("--rerun_baseline_preprocess", action="store_true")
    args = parser.parse_args()

    suite_dir = ROOT / "runs" / args.suite_name
    suite_dir.mkdir(parents=True, exist_ok=True)
    manifest = _read_json(Path(args.manifest))
    case_cfg = manifest["cases"][0]
    bootstrap_dir = suite_dir / "edge_mini_set"
    baseline_run_dir = suite_dir / "preprocess_edge_baseline"
    baseline_preprocess_dir = baseline_run_dir / "preprocess"
    frozen_baseline_preprocess_dir = Path(case_cfg["baseline_prediction_preprocess_dir"])

    cases = []

    extract_cmd = [
        "python",
        "scripts/eval/extract_edge_benchmark_keyframes.py",
        "--manifest",
        args.manifest,
        "--output_dir",
        str(bootstrap_dir),
    ]
    extract_res = _run(extract_cmd)
    extract_passed = extract_res["returncode"] == 0 and (bootstrap_dir / "summary.json").exists()
    cases.append({"name": "edge_mini_dataset_bootstrap", "passed": extract_passed, "returncode": extract_res["returncode"]})

    if args.rerun_baseline_preprocess:
        preprocess_cmd = [
            "python",
            "wan/modules/animate/preprocess/preprocess_data.py",
            "--ckpt_path", args.ckpt_path,
            "--video_path", args.video_path,
            "--refer_path", args.refer_path,
            "--run_dir", str(baseline_run_dir),
            "--save_manifest",
            "--replace_flag",
            "--lossless_intermediate",
            "--resolution_area", "640", "360",
            "--fps", "5",
            "--preprocess_runtime_profile", "h200_extreme",
            "--sam_runtime_profile", "h200_safe",
            "--sam_prompt_mode", "mask_seed",
            "--no-sam_apply_postprocessing",
            "--no-sam_use_negative_points",
            "--sam_prompt_body_conf_thresh", "0.999",
            "--sam_prompt_face_conf_thresh", "0.999",
            "--sam_prompt_hand_conf_thresh", "0.999",
            "--sam_prompt_face_min_points", "100",
            "--sam_prompt_hand_min_points", "100",
            "--sam_chunk_len", "20",
            "--sam_keyframes_per_chunk", "4",
            "--sam_reprompt_interval", "0",
            "--multistage_preprocess_mode", "h200_extreme",
            "--face_analysis_mode", "heuristic",
            "--pose_motion_stack_mode", "v1",
            "--boundary_fusion_mode", "v2",
            "--parsing_mode", "heuristic",
            "--matting_mode", "heuristic",
            "--bg_inpaint_mode", "video_v2",
            "--soft_mask_mode", "soft_band",
            "--reference_normalization_mode", "structure_match",
            "--reference_structure_segment_clamp_min", "0.8",
            "--reference_structure_segment_clamp_max", "1.25",
            "--reference_structure_width_budget_ratio", "1.05",
            "--reference_structure_height_budget_ratio", "1.05",
            "--analysis_resolution_area", "960", "540",
            "--soft_mask_band_width", "28",
            "--soft_mask_blur_kernel", "7",
            "--matting_trimap_outer_dilate", "16",
            "--parsing_boundary_kernel", "13",
            "--bg_temporal_smooth_strength", "0.18",
            "--bg_video_global_blend_strength", "0.97",
            "--face_rerun_difficulty_threshold", "0.36",
            "--face_difficulty_expand_ratio", "1.32",
            "--face_tracking_smooth_strength", "0.92",
            "--face_tracking_max_center_shift", "0.014",
            "--multistage_face_bbox_extra_smooth", "0.10",
        ]
        preprocess_res = _run(preprocess_cmd)
        preprocess_passed = preprocess_res["returncode"] == 0 and (baseline_preprocess_dir / "metadata.json").exists()
        cases.append({
            "name": "baseline_preprocess_contract",
            "passed": preprocess_passed,
            "returncode": preprocess_res["returncode"],
            "mode": "rerun",
        })
    else:
        preprocess_res = {"cmd": None, "returncode": 0, "stdout": "", "stderr": ""}
        baseline_preprocess_dir = frozen_baseline_preprocess_dir
        preprocess_passed = (baseline_preprocess_dir / "metadata.json").exists()
        cases.append({
            "name": "baseline_preprocess_contract",
            "passed": preprocess_passed,
            "returncode": 0 if preprocess_passed else 1,
            "mode": "frozen_baseline",
        })

    contract_cmd = [
        "python",
        "scripts/eval/check_animate_contract.py",
        "--src_root_path",
        str(baseline_preprocess_dir),
    ]
    contract_res = _run(contract_cmd)
    if not preprocess_passed:
        contract_passed = False
    else:
        contract_passed = contract_res["returncode"] == 0
    cases[-1]["contract_passed"] = contract_passed

    edge_metrics_path = suite_dir / "metrics" / "edge_groundtruth_metrics.json"
    edge_metrics_cmd = [
        "python",
        "scripts/eval/compute_edge_groundtruth_metrics.py",
        "--dataset_dir",
        str(bootstrap_dir),
        "--prediction_preprocess_dir",
        str(baseline_preprocess_dir),
        "--output_json",
        str(edge_metrics_path),
    ]
    edge_metrics_res = _run(edge_metrics_cmd)
    edge_metrics_passed = edge_metrics_res["returncode"] == 0 and edge_metrics_path.exists()
    cases.append({"name": "baseline_edge_groundtruth_metrics", "passed": edge_metrics_passed, "returncode": edge_metrics_res["returncode"]})

    edge_dataset_summary = _read_json(bootstrap_dir / "summary.json") if (bootstrap_dir / "summary.json").exists() else {}
    edge_metrics_summary = _read_json(edge_metrics_path) if edge_metrics_path.exists() else {}

    summary = {
        "suite_name": args.suite_name,
        "manifest": str(Path(args.manifest).resolve()),
        "gate_policy": str(Path(args.gate_policy).resolve()),
        "cases": cases,
        "all_passed": all(case.get("passed", False) for case in cases),
        "edge_dataset": edge_dataset_summary,
        "edge_groundtruth_metrics": edge_metrics_summary,
        "artifacts": {
            "edge_dataset_dir": str(bootstrap_dir.resolve()),
            "baseline_preprocess_dir": str(baseline_preprocess_dir.resolve()),
            "baseline_mode": cases[1].get("mode"),
            "edge_groundtruth_metrics_json": str(edge_metrics_path.resolve()) if edge_metrics_path.exists() else None,
        },
    }
    summary_path = suite_dir / "summary.json"
    _write_json(summary_path, summary)

    summary_md = [
        f"# {args.suite_name}",
        "",
        f"- all_passed: `{summary['all_passed']}`",
        f"- total_keyframe_count: `{edge_dataset_summary.get('total_keyframe_count')}`",
        f"- boundary_f1_mean: `{edge_metrics_summary.get('boundary_f1_mean')}`",
        f"- alpha_mae_mean: `{edge_metrics_summary.get('alpha_mae_mean')}`",
        f"- trimap_error_mean: `{edge_metrics_summary.get('trimap_error_mean')}`",
        "",
        "## Cases",
    ]
    for case in cases:
        summary_md.append(f"- `{case['name']}`: passed=`{case['passed']}` returncode=`{case['returncode']}`")
    (suite_dir / "summary.md").write_text("\n".join(summary_md) + "\n", encoding="utf-8")

    gate_cmd = [
        "python",
        "scripts/eval/summarize_optimization4_validation.py",
        "--summary_json",
        str(summary_path),
        "--gate_policy",
        args.gate_policy,
        "--output_json",
        str(suite_dir / "gate_result.json"),
    ]
    gate_res = _run(gate_cmd)
    if gate_res["returncode"] != 0:
        raise SystemExit(gate_res["stderr"] or gate_res["stdout"] or gate_res["returncode"])

    print(json.dumps({
        "suite_dir": str(suite_dir.resolve()),
        "all_passed": summary["all_passed"],
        "total_keyframe_count": edge_dataset_summary.get("total_keyframe_count"),
        "boundary_f1_mean": edge_metrics_summary.get("boundary_f1_mean"),
        "alpha_mae_mean": edge_metrics_summary.get("alpha_mae_mean"),
        "trimap_error_mean": edge_metrics_summary.get("trimap_error_mean"),
        "gate_result": str((suite_dir / 'gate_result.json').resolve()),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
