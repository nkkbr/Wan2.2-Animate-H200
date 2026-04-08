#!/usr/bin/env python
import argparse
import json
from pathlib import Path


def _safe_pct(delta, base):
    if base is None or abs(base) < 1e-8:
        return 0.0
    return float(delta / base * 100.0)


def main():
    parser = argparse.ArgumentParser(description="Evaluate optimization5 Step 08 H200 quality tiers.")
    parser.add_argument("--summary_json", type=str, required=True)
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    summary = json.loads(Path(args.summary_json).read_text(encoding="utf-8"))
    rows = {row["tier"]: row for row in summary["rows"]}
    pair_metrics = summary.get("pair_metrics") or {}

    def pair(name: str):
        return pair_metrics.get(name) or {}

    metrics = {
        "high_vs_default_gradient_gain_pct": _safe_pct(
            (pair("default_to_high").get("band_gradient_after_mean", 0.0) - pair("default_to_high").get("band_gradient_before_mean", 0.0)),
            pair("default_to_high").get("band_gradient_before_mean", 0.0),
        ),
        "high_vs_default_contrast_gain_pct": _safe_pct(
            (pair("default_to_high").get("band_edge_contrast_after_mean", 0.0) - pair("default_to_high").get("band_edge_contrast_before_mean", 0.0)),
            pair("default_to_high").get("band_edge_contrast_before_mean", 0.0),
        ),
        "high_vs_default_halo_reduction_pct": _safe_pct(
            (pair("default_to_high").get("halo_ratio_before", 0.0) - pair("default_to_high").get("halo_ratio_after", 0.0)),
            pair("default_to_high").get("halo_ratio_before", 0.0),
        ),
        "high_vs_default_seam_degradation_pct": _safe_pct(
            (rows["high"].get("seam_score_mean", 0.0) - rows["default"].get("seam_score_mean", 0.0)),
            rows["default"].get("seam_score_mean", 0.0),
        ),
        "high_vs_default_background_degradation_pct": _safe_pct(
            (rows["high"].get("background_fluctuation_mean", 0.0) - rows["default"].get("background_fluctuation_mean", 0.0)),
            rows["default"].get("background_fluctuation_mean", 0.0),
        ),
        "high_vs_default_runtime_increase_pct": _safe_pct(
            (rows["high"].get("total_generate_sec", 0.0) - rows["default"].get("total_generate_sec", 0.0)),
            rows["default"].get("total_generate_sec", 0.0),
        ),
        "extreme_vs_high_gradient_gain_pct": _safe_pct(
            (pair("high_to_extreme").get("band_gradient_after_mean", 0.0) - pair("high_to_extreme").get("band_gradient_before_mean", 0.0)),
            pair("high_to_extreme").get("band_gradient_before_mean", 0.0),
        ),
        "extreme_vs_high_contrast_gain_pct": _safe_pct(
            (pair("high_to_extreme").get("band_edge_contrast_after_mean", 0.0) - pair("high_to_extreme").get("band_edge_contrast_before_mean", 0.0)),
            pair("high_to_extreme").get("band_edge_contrast_before_mean", 0.0),
        ),
        "extreme_vs_high_halo_reduction_pct": _safe_pct(
            (pair("high_to_extreme").get("halo_ratio_before", 0.0) - pair("high_to_extreme").get("halo_ratio_after", 0.0)),
            pair("high_to_extreme").get("halo_ratio_before", 0.0),
        ),
        "extreme_vs_high_seam_degradation_pct": _safe_pct(
            (rows["extreme"].get("seam_score_mean", 0.0) - rows["high"].get("seam_score_mean", 0.0)),
            rows["high"].get("seam_score_mean", 0.0),
        ),
        "extreme_vs_high_background_degradation_pct": _safe_pct(
            (rows["extreme"].get("background_fluctuation_mean", 0.0) - rows["high"].get("background_fluctuation_mean", 0.0)),
            rows["high"].get("background_fluctuation_mean", 0.0),
        ),
        "extreme_vs_high_runtime_increase_pct": _safe_pct(
            (rows["extreme"].get("total_generate_sec", 0.0) - rows["high"].get("total_generate_sec", 0.0)),
            rows["high"].get("total_generate_sec", 0.0),
        ),
    }
    gates = {
        "high_quality_gradient_gain_ge_3pct": metrics["high_vs_default_gradient_gain_pct"] >= 3.0,
        "high_quality_contrast_non_negative": metrics["high_vs_default_contrast_gain_pct"] >= 0.0,
        "high_quality_halo_non_worse": metrics["high_vs_default_halo_reduction_pct"] >= 0.0,
        "high_quality_seam_degradation_le_3pct": metrics["high_vs_default_seam_degradation_pct"] <= 3.0,
        "high_quality_background_degradation_le_3pct": metrics["high_vs_default_background_degradation_pct"] <= 3.0,
        "extreme_quality_gradient_gain_ge_1pct": metrics["extreme_vs_high_gradient_gain_pct"] >= 1.0,
        "extreme_quality_contrast_non_negative": metrics["extreme_vs_high_contrast_gain_pct"] >= 0.0,
        "extreme_quality_halo_non_worse": metrics["extreme_vs_high_halo_reduction_pct"] >= 0.0,
        "extreme_quality_seam_degradation_le_3pct": metrics["extreme_vs_high_seam_degradation_pct"] <= 3.0,
        "extreme_quality_background_degradation_le_3pct": metrics["extreme_vs_high_background_degradation_pct"] <= 3.0,
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
