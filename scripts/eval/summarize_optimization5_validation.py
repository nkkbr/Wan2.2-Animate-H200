#!/usr/bin/env python
import argparse
import json
from pathlib import Path


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Summarize optimization5 Step 01 validation and apply gate policy.")
    parser.add_argument("--summary_json", required=True, type=str)
    parser.add_argument("--gate_policy", required=True, type=str)
    parser.add_argument("--output_json", required=True, type=str)
    args = parser.parse_args()

    summary = _read_json(Path(args.summary_json))
    gate = _read_json(Path(args.gate_policy))

    failures = []
    case_names = {case["name"] for case in summary.get("cases", [])}
    for required in gate.get("required_case_names", []):
        if required not in case_names:
            failures.append(f"missing_case:{required}")
    for required in gate.get("required_outputs", []):
        target = Path(summary["suite_dir"]) / required
        if required == Path(args.output_json).name:
            continue
        if not target.exists():
            failures.append(f"missing_output:{required}")

    edge_dataset = summary.get("edge_dataset", {})
    edge_metrics = summary.get("edge_groundtruth_metrics", {})
    category_coverage = edge_dataset.get("category_coverage", {})
    thresholds = gate.get("metric_thresholds", {})

    if edge_dataset.get("total_keyframe_count", 0) < gate.get("minimum_keyframe_count", 0):
        failures.append("keyframe_count_below_minimum")
    if edge_dataset.get("reviewed_ratio", 0.0) < gate.get("minimum_reviewed_ratio", 1.0):
        failures.append("reviewed_ratio_below_minimum")
    for category, minimum in gate.get("minimum_category_coverage", {}).items():
        if int(category_coverage.get(category, 0)) < int(minimum):
            failures.append(f"category_coverage_below_minimum:{category}")

    threshold_map = {
        "boundary_f1_mean_min": lambda v, t: v is not None and v >= t,
        "alpha_mae_mean_max": lambda v, t: v is not None and v <= t,
        "trimap_error_mean_max": lambda v, t: v is not None and v <= t,
    }
    for key, threshold in thresholds.items():
        metric_name = key.replace("_min", "").replace("_max", "")
        value = edge_metrics.get(metric_name)
        fn = threshold_map.get(key)
        if fn and not fn(value, threshold):
            failures.append(f"metric_threshold_failed:{key}")

    repeat_stability = summary.get("repeat_stability", {})
    if repeat_stability.get("max_metric_delta", 1.0) > gate.get("max_repeat_metric_delta", 0.0):
        failures.append("repeat_metric_delta_too_large")

    result = {
        "policy_name": gate.get("policy_name"),
        "suite_dir": summary["suite_dir"],
        "overall_passed": len(failures) == 0 and summary.get("all_cases_passed", False),
        "all_cases_passed": summary.get("all_cases_passed", False),
        "failures": failures,
        "reviewed_keyframe_count": edge_dataset.get("total_keyframe_count"),
        "reviewed_ratio": edge_dataset.get("reviewed_ratio"),
        "category_coverage": category_coverage,
        "boundary_f1_mean": edge_metrics.get("boundary_f1_mean"),
        "alpha_mae_mean": edge_metrics.get("alpha_mae_mean"),
        "trimap_error_mean": edge_metrics.get("trimap_error_mean"),
        "repeat_stability": repeat_stability,
    }
    _write_json(Path(args.output_json), result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
