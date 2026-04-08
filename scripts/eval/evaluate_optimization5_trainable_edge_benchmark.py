#!/usr/bin/env python
import argparse
import json
from pathlib import Path


def _safe_pct(delta, base):
    if base is None or abs(base) < 1e-8:
        return 0.0
    return float(delta / base * 100.0)


def _value_or_zero(value):
    return 0.0 if value is None else float(value)


def _nested_mean(metrics: dict | None, key: str):
    if not metrics:
        return None
    value = metrics.get(key)
    if isinstance(value, dict):
        return value.get("mean")
    return value


def main():
    parser = argparse.ArgumentParser(description="Evaluate optimization5 Step 07 trainable edge benchmark.")
    parser.add_argument("--summary_json", type=str, required=True)
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    summary = json.loads(Path(args.summary_json).read_text(encoding="utf-8"))
    baseline_reviewed = summary.get("baseline_reviewed_metrics") or {}
    candidate_reviewed = summary.get("candidate_reviewed_metrics") or {}
    baseline_alpha = summary.get("baseline_alpha_metrics") or {}
    candidate_alpha = summary.get("candidate_alpha_metrics") or {}
    boundary = summary.get("generate_boundary_metrics") or {}
    baseline_repl = summary.get("baseline_replacement_metrics") or {}
    candidate_repl = summary.get("candidate_replacement_metrics") or {}

    metrics = {
        "boundary_f1_gain_pct": _safe_pct(
            (candidate_reviewed.get("boundary_f1_mean", 0.0) - baseline_reviewed.get("boundary_f1_mean", 0.0)),
            baseline_reviewed.get("boundary_f1_mean", 0.0),
        ),
        "alpha_mae_reduction_pct": _safe_pct(
            (baseline_reviewed.get("alpha_mae_mean", 0.0) - candidate_reviewed.get("alpha_mae_mean", 0.0)),
            baseline_reviewed.get("alpha_mae_mean", 0.0),
        ),
        "trimap_error_reduction_pct": _safe_pct(
            (baseline_reviewed.get("trimap_error_mean", 0.0) - candidate_reviewed.get("trimap_error_mean", 0.0)),
            baseline_reviewed.get("trimap_error_mean", 0.0),
        ),
        "fine_boundary_iou_gain_pct": _safe_pct(
            (_value_or_zero(candidate_alpha.get("fine_boundary_iou_mean")) - _value_or_zero(baseline_alpha.get("fine_boundary_iou_mean"))),
            _value_or_zero(baseline_alpha.get("fine_boundary_iou_mean")),
        ),
        "trimap_unknown_iou_gain_pct": _safe_pct(
            (_value_or_zero(candidate_alpha.get("trimap_unknown_iou_mean")) - _value_or_zero(baseline_alpha.get("trimap_unknown_iou_mean"))),
            _value_or_zero(baseline_alpha.get("trimap_unknown_iou_mean")),
        ),
        "roi_halo_reduction_pct": _safe_pct(
            (boundary.get("halo_ratio_before", 0.0) - boundary.get("halo_ratio_after", 0.0)),
            boundary.get("halo_ratio_before", 0.0),
        ),
        "roi_gradient_gain_pct": _safe_pct(
            (boundary.get("band_gradient_after_mean", 0.0) - boundary.get("band_gradient_before_mean", 0.0)),
            boundary.get("band_gradient_before_mean", 0.0),
        ),
        "roi_edge_contrast_gain_pct": _safe_pct(
            (boundary.get("band_edge_contrast_after_mean", 0.0) - boundary.get("band_edge_contrast_before_mean", 0.0)),
            boundary.get("band_edge_contrast_before_mean", 0.0),
        ),
        "seam_degradation_pct": _safe_pct(
            (_nested_mean(candidate_repl, "seam_score") or 0.0) - (_nested_mean(baseline_repl, "seam_score") or 0.0),
            (_nested_mean(baseline_repl, "seam_score") or 0.0),
        ),
        "background_degradation_pct": _safe_pct(
            (_nested_mean(candidate_repl, "background_fluctuation") or 0.0) - (_nested_mean(baseline_repl, "background_fluctuation") or 0.0),
            (_nested_mean(baseline_repl, "background_fluctuation") or 0.0),
        ),
    }
    gates = {
        "boundary_f1_gain_ge_10pct": metrics["boundary_f1_gain_pct"] >= 10.0,
        "alpha_mae_reduction_ge_10pct": metrics["alpha_mae_reduction_pct"] >= 10.0,
        "trimap_error_reduction_ge_10pct": metrics["trimap_error_reduction_pct"] >= 10.0,
        "roi_gradient_gain_ge_12pct": metrics["roi_gradient_gain_pct"] >= 12.0,
        "roi_edge_contrast_gain_ge_12pct": metrics["roi_edge_contrast_gain_pct"] >= 12.0,
        "roi_halo_reduction_ge_10pct": metrics["roi_halo_reduction_pct"] >= 10.0,
        "seam_degradation_le_3pct": metrics["seam_degradation_pct"] <= 3.0,
        "background_degradation_le_3pct": metrics["background_degradation_pct"] <= 3.0,
    }
    result = {
        "summary_json": str(Path(args.summary_json).resolve()),
        "metrics": metrics,
        "gates": gates,
        "passed": all(gates.values()),
    }
    payload = json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True)
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
