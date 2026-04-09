#!/usr/bin/env python
import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "docs/optimization6/benchmark/benchmark_manifest.step01.v2.json"
DEFAULT_SCHEMA = ROOT / "docs/optimization6/benchmark/label_schema.edge_reviewed_v2.json"
DEFAULT_GATE = ROOT / "docs/optimization6/benchmark/gate_policy.step01.v2.json"


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _run(cmd, cwd=ROOT):
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    return {"cmd": cmd, "returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}


def main():
    parser = argparse.ArgumentParser(description="Run optimization6 Step 01 validation suite.")
    parser.add_argument("--suite_name", required=True, type=str)
    parser.add_argument("--manifest", type=str, default=str(DEFAULT_MANIFEST))
    parser.add_argument("--schema", type=str, default=str(DEFAULT_SCHEMA))
    parser.add_argument("--gate_policy", type=str, default=str(DEFAULT_GATE))
    args = parser.parse_args()

    suite_dir = ROOT / "runs" / args.suite_name
    suite_dir.mkdir(parents=True, exist_ok=True)
    reviewed_dir = suite_dir / "reviewed_edge_benchmark_v2"

    manifest = _read_json(Path(args.manifest))
    case_cfg = manifest["cases"][0]
    baseline_preprocess_dir = Path(case_cfg["baseline_prediction_preprocess_dir"])
    cases = []

    build_cmd = [
        "python", "scripts/eval/build_reviewed_edge_benchmark_v2.py",
        "--manifest", args.manifest,
        "--output_dir", str(reviewed_dir),
    ]
    build_res = _run(build_cmd)
    build_passed = build_res["returncode"] == 0 and (reviewed_dir / "summary.json").exists()
    cases.append({"name": "reviewed_edge_dataset_v2_build", "passed": build_passed, "returncode": build_res["returncode"]})

    validate_cmd = [
        "python", "scripts/eval/validate_reviewed_edge_labels.py",
        "--dataset_dir", str(reviewed_dir),
        "--schema", args.schema,
        "--output_json", str(suite_dir / "validation.json"),
    ]
    validate_res = _run(validate_cmd)
    validation_passed = validate_res["returncode"] == 0
    cases.append({"name": "reviewed_edge_dataset_v2_validate", "passed": validation_passed, "returncode": validate_res["returncode"]})

    contract_cmd = [
        "python", "scripts/eval/check_animate_contract.py",
        "--src_root_path", str(baseline_preprocess_dir),
    ]
    contract_res = _run(contract_cmd)
    contract_passed = contract_res["returncode"] == 0
    cases.append({"name": "baseline_preprocess_contract", "passed": contract_passed, "returncode": contract_res["returncode"]})

    metric_paths = []
    metric_runs = []
    for idx in range(3):
        out = suite_dir / "metrics" / f"edge_groundtruth_metrics_v2_run{idx+1}.json"
        cmd = [
            "python", "scripts/eval/compute_reviewed_edge_metrics_v2.py",
            "--dataset_dir", str(reviewed_dir),
            "--prediction_preprocess_dir", str(baseline_preprocess_dir),
            "--output_json", str(out),
        ]
        res = _run(cmd)
        metric_runs.append(res)
        metric_paths.append(out)
    metrics_passed = all(run["returncode"] == 0 for run in metric_runs) and all(p.exists() for p in metric_paths)
    cases.append({"name": "reviewed_edge_v2_groundtruth_metrics", "passed": metrics_passed, "returncode": 0 if metrics_passed else 1})

    metric_payloads = [_read_json(p) for p in metric_paths if p.exists()]
    keys = [
        "boundary_f1_mean", "alpha_mae_mean", "trimap_error_mean",
        "face_boundary_f1_mean", "hair_boundary_f1_mean", "hand_boundary_f1_mean",
        "cloth_boundary_f1_mean", "occluded_boundary_f1_mean",
    ]
    max_metric_delta = 0.0
    if len(metric_payloads) >= 2:
        baseline = metric_payloads[0]
        for other in metric_payloads[1:]:
            for key in keys:
                a = baseline.get(key)
                b = other.get(key)
                if a is None or b is None:
                    continue
                max_metric_delta = max(max_metric_delta, abs(float(a) - float(b)))

    dataset_summary = _read_json(reviewed_dir / "summary.json") if (reviewed_dir / "summary.json").exists() else {}
    validation_payload = _read_json(suite_dir / "validation.json") if (suite_dir / "validation.json").exists() else {}
    metrics_payload = metric_payloads[0] if metric_payloads else {}

    summary = {
        "suite_name": args.suite_name,
        "suite_dir": str(suite_dir.resolve()),
        "manifest": str(Path(args.manifest).resolve()),
        "schema": str(Path(args.schema).resolve()),
        "gate_policy": str(Path(args.gate_policy).resolve()),
        "cases": cases,
        "all_cases_passed": all(case["passed"] for case in cases),
        "reviewed_edge_dataset": dataset_summary,
        "validation": validation_payload,
        "edge_groundtruth_metrics": metrics_payload,
        "repeat_stability": {
            "repeat_count": len(metric_payloads),
            "max_metric_delta": max_metric_delta,
        },
        "artifacts": {
            "reviewed_edge_dataset_dir": str(reviewed_dir.resolve()),
            "baseline_preprocess_dir": str(baseline_preprocess_dir.resolve()),
            "validation_json": str((suite_dir / "validation.json").resolve()) if (suite_dir / "validation.json").exists() else None,
            "metrics": [str(p.resolve()) for p in metric_paths if p.exists()],
            "review_spotcheck_contact_sheet": str((reviewed_dir / "review_spotcheck_contact_sheet.jpg").resolve()) if (reviewed_dir / "review_spotcheck_contact_sheet.jpg").exists() else None,
        },
    }
    summary_path = suite_dir / "summary.json"
    _write_json(summary_path, summary)

    md = [
        f"# {args.suite_name}",
        "",
        f"- all_cases_passed: `{summary['all_cases_passed']}`",
        f"- total_keyframe_count: `{dataset_summary.get('total_keyframe_count')}`",
        f"- reviewed_fraction: `{validation_payload.get('reviewed_fraction')}`",
        f"- boundary_f1_mean: `{metrics_payload.get('boundary_f1_mean')}`",
        f"- alpha_mae_mean: `{metrics_payload.get('alpha_mae_mean')}`",
        f"- trimap_error_mean: `{metrics_payload.get('trimap_error_mean')}`",
        f"- face_boundary_f1_mean: `{metrics_payload.get('face_boundary_f1_mean')}`",
        f"- hair_boundary_f1_mean: `{metrics_payload.get('hair_boundary_f1_mean')}`",
        f"- hand_boundary_f1_mean: `{metrics_payload.get('hand_boundary_f1_mean')}`",
        f"- cloth_boundary_f1_mean: `{metrics_payload.get('cloth_boundary_f1_mean')}`",
        f"- occluded_boundary_f1_mean: `{metrics_payload.get('occluded_boundary_f1_mean')}`",
        f"- repeat_max_metric_delta: `{max_metric_delta}`",
        "",
        "## Category Coverage",
    ]
    for category, count in dataset_summary.get("category_coverage", {}).items():
        md.append(f"- `{category}`: `{count}`")
    md.extend(["", "## Cases"])
    for case in cases:
        md.append(f"- `{case['name']}`: passed=`{case['passed']}` returncode=`{case['returncode']}`")
    (suite_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    gate_policy = _read_json(Path(args.gate_policy))
    category_cov = dataset_summary.get("category_coverage", {})
    gate_result = {
        "suite_name": args.suite_name,
        "all_cases_passed": summary["all_cases_passed"],
        "minimum_keyframe_count_passed": dataset_summary.get("total_keyframe_count", 0) >= gate_policy["minimum_keyframe_count"],
        "reviewed_ratio_passed": validation_payload.get("reviewed_fraction", 0.0) >= gate_policy["minimum_reviewed_ratio"],
        "category_coverage_passed": all(category_cov.get(k, 0) >= v for k, v in gate_policy["minimum_category_coverage"].items()),
        "boundary_f1_passed": (metrics_payload.get("boundary_f1_mean") or 0.0) >= gate_policy["metric_thresholds"]["boundary_f1_mean_min"],
        "alpha_mae_passed": (metrics_payload.get("alpha_mae_mean") or 999.0) <= gate_policy["metric_thresholds"]["alpha_mae_mean_max"],
        "trimap_error_passed": (metrics_payload.get("trimap_error_mean") or 999.0) <= gate_policy["metric_thresholds"]["trimap_error_mean_max"],
        "repeat_delta_passed": max_metric_delta <= gate_policy["max_repeat_metric_delta"],
    }
    gate_result["gate_passed"] = all(gate_result.values())
    _write_json(suite_dir / "gate_result.json", gate_result)

    print(json.dumps({
        "suite_dir": str(suite_dir.resolve()),
        "all_cases_passed": summary["all_cases_passed"],
        "total_keyframe_count": dataset_summary.get("total_keyframe_count"),
        "boundary_f1_mean": metrics_payload.get("boundary_f1_mean"),
        "alpha_mae_mean": metrics_payload.get("alpha_mae_mean"),
        "trimap_error_mean": metrics_payload.get("trimap_error_mean"),
        "gate_passed": gate_result["gate_passed"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
