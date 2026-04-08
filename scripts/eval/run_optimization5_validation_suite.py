#!/usr/bin/env python
import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "docs/optimization5/benchmark/benchmark_manifest.step01.json"
DEFAULT_GATE = ROOT / "docs/optimization5/benchmark/gate_policy.step01.json"


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
    parser = argparse.ArgumentParser(description="Run optimization5 Step 01 validation suite.")
    parser.add_argument("--suite_name", required=True, type=str)
    parser.add_argument("--manifest", type=str, default=str(DEFAULT_MANIFEST))
    parser.add_argument("--gate_policy", type=str, default=str(DEFAULT_GATE))
    args = parser.parse_args()

    suite_dir = ROOT / "runs" / args.suite_name
    suite_dir.mkdir(parents=True, exist_ok=True)

    manifest = _read_json(Path(args.manifest))
    case_cfg = manifest["cases"][0]
    reviewed_dir = suite_dir / "reviewed_edge_benchmark"
    baseline_preprocess_dir = Path(case_cfg["baseline_prediction_preprocess_dir"])

    cases = []

    build_cmd = [
        "python",
        "scripts/eval/build_reviewed_edge_benchmark.py",
        "--manifest",
        args.manifest,
        "--output_dir",
        str(reviewed_dir),
    ]
    build_res = _run(build_cmd)
    build_passed = build_res["returncode"] == 0 and (reviewed_dir / "summary.json").exists()
    cases.append({"name": "reviewed_edge_dataset_build", "passed": build_passed, "returncode": build_res["returncode"]})

    contract_cmd = [
        "python",
        "scripts/eval/check_animate_contract.py",
        "--src_root_path",
        str(baseline_preprocess_dir),
    ]
    contract_res = _run(contract_cmd)
    contract_passed = contract_res["returncode"] == 0
    cases.append({"name": "baseline_preprocess_contract", "passed": contract_passed, "returncode": contract_res["returncode"]})

    metric_runs = []
    metric_paths = []
    for idx in range(3):
        out = suite_dir / "metrics" / f"edge_groundtruth_metrics_run{idx+1}.json"
        metric_cmd = [
            "python",
            "scripts/eval/compute_reviewed_edge_metrics.py",
            "--dataset_dir",
            str(reviewed_dir),
            "--prediction_preprocess_dir",
            str(baseline_preprocess_dir),
            "--output_json",
            str(out),
        ]
        metric_res = _run(metric_cmd)
        metric_runs.append(metric_res)
        metric_paths.append(out)
    metric_passed = all(run["returncode"] == 0 for run in metric_runs) and all(path.exists() for path in metric_paths)
    cases.append({"name": "reviewed_edge_groundtruth_metrics", "passed": metric_passed, "returncode": 0 if metric_passed else 1})

    metric_payloads = [_read_json(path) for path in metric_paths if path.exists()]
    keys = [
        "boundary_f1_mean",
        "alpha_mae_mean",
        "trimap_error_mean",
        "face_boundary_f1_mean",
        "hair_boundary_f1_mean",
        "hand_boundary_f1_mean",
        "cloth_boundary_f1_mean",
        "occluded_boundary_f1_mean",
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

    edge_dataset = _read_json(reviewed_dir / "summary.json") if (reviewed_dir / "summary.json").exists() else {}
    category_coverage = edge_dataset.get("category_coverage", {})
    reviewed_ratio = 1.0 if edge_dataset.get("total_keyframe_count", 0) else 0.0
    edge_dataset["reviewed_ratio"] = reviewed_ratio
    edge_metrics = metric_payloads[0] if metric_payloads else {}

    summary = {
        "suite_name": args.suite_name,
        "suite_dir": str(suite_dir.resolve()),
        "manifest": str(Path(args.manifest).resolve()),
        "gate_policy": str(Path(args.gate_policy).resolve()),
        "cases": cases,
        "all_cases_passed": all(case["passed"] for case in cases),
        "edge_dataset": edge_dataset,
        "edge_groundtruth_metrics": edge_metrics,
        "repeat_stability": {
            "repeat_count": len(metric_payloads),
            "max_metric_delta": max_metric_delta,
        },
        "artifacts": {
            "reviewed_edge_dataset_dir": str(reviewed_dir.resolve()),
            "baseline_preprocess_dir": str(baseline_preprocess_dir.resolve()),
            "metrics": [str(path.resolve()) for path in metric_paths if path.exists()],
            "review_spotcheck_contact_sheet": str((reviewed_dir / "review_spotcheck_contact_sheet.jpg").resolve()) if (reviewed_dir / "review_spotcheck_contact_sheet.jpg").exists() else None,
        },
    }
    summary_path = suite_dir / "summary.json"
    _write_json(summary_path, summary)

    summary_md = [
        f"# {args.suite_name}",
        "",
        f"- all_cases_passed: `{summary['all_cases_passed']}`",
        f"- reviewed_keyframe_count: `{edge_dataset.get('total_keyframe_count')}`",
        f"- reviewed_ratio: `{reviewed_ratio}`",
        f"- boundary_f1_mean: `{edge_metrics.get('boundary_f1_mean')}`",
        f"- alpha_mae_mean: `{edge_metrics.get('alpha_mae_mean')}`",
        f"- trimap_error_mean: `{edge_metrics.get('trimap_error_mean')}`",
        f"- face_boundary_f1_mean: `{edge_metrics.get('face_boundary_f1_mean')}`",
        f"- hair_boundary_f1_mean: `{edge_metrics.get('hair_boundary_f1_mean')}`",
        f"- hand_boundary_f1_mean: `{edge_metrics.get('hand_boundary_f1_mean')}`",
        f"- cloth_boundary_f1_mean: `{edge_metrics.get('cloth_boundary_f1_mean')}`",
        f"- occluded_boundary_f1_mean: `{edge_metrics.get('occluded_boundary_f1_mean')}`",
        f"- repeat_max_metric_delta: `{max_metric_delta}`",
        "",
        "## Category Coverage",
    ]
    for category, count in category_coverage.items():
        summary_md.append(f"- `{category}`: `{count}`")
    summary_md.extend(["", "## Cases"])
    for case in cases:
        summary_md.append(f"- `{case['name']}`: passed=`{case['passed']}` returncode=`{case['returncode']}`")
    (suite_dir / "summary.md").write_text("\n".join(summary_md) + "\n", encoding="utf-8")

    gate_cmd = [
        "python",
        "scripts/eval/summarize_optimization5_validation.py",
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
        "all_cases_passed": summary["all_cases_passed"],
        "reviewed_keyframe_count": edge_dataset.get("total_keyframe_count"),
        "category_coverage": category_coverage,
        "boundary_f1_mean": edge_metrics.get("boundary_f1_mean"),
        "alpha_mae_mean": edge_metrics.get("alpha_mae_mean"),
        "trimap_error_mean": edge_metrics.get("trimap_error_mean"),
        "gate_result": str((suite_dir / "gate_result.json").resolve()),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
