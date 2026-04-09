#!/usr/bin/env python
import argparse
import json
from pathlib import Path


def _pct(before: float, after: float) -> float:
    if abs(before) <= 1e-9:
        return 0.0
    return (after / before - 1.0) * 100.0


def main():
    parser = argparse.ArgumentParser(description="Evaluate Optimization6 Step04 boundary ROI generative reconstruction v2 benchmark.")
    parser.add_argument("--summary_json", type=str, required=True)
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    summary = json.loads(Path(args.summary_json).read_text(encoding="utf-8"))
    rows = {row["boundary_refine_mode"]: row for row in summary["rows"]}
    metrics = summary.get("roi_metrics") or {}

    roi_gradient_before = float(metrics.get("roi_band_gradient_before_mean", 0.0) or 0.0)
    roi_gradient_after = float(metrics.get("roi_band_gradient_after_mean", 0.0) or 0.0)
    roi_contrast_before = float(metrics.get("roi_band_edge_contrast_before_mean", 0.0) or 0.0)
    roi_contrast_after = float(metrics.get("roi_band_edge_contrast_after_mean", 0.0) or 0.0)
    roi_halo_before = float(metrics.get("roi_halo_ratio_before", 0.0) or 0.0)
    roi_halo_after = float(metrics.get("roi_halo_ratio_after", 0.0) or 0.0)

    halo_reduction_pct = ((roi_halo_before - roi_halo_after) / roi_halo_before * 100.0) if roi_halo_before > 1e-9 else 0.0
    gradient_gain_pct = _pct(roi_gradient_before, roi_gradient_after)
    contrast_gain_pct = _pct(roi_contrast_before, roi_contrast_after)

    seam_before = float(rows["none"].get("seam_score_mean", 0.0) or 0.0)
    seam_after = float(rows["roi_gen_v2"].get("seam_score_mean", 0.0) or 0.0)
    bg_before = float(rows["none"].get("background_fluctuation_mean", 0.0) or 0.0)
    bg_after = float(rows["roi_gen_v2"].get("background_fluctuation_mean", 0.0) or 0.0)
    seam_degradation_pct = _pct(seam_before, seam_after)
    background_degradation_pct = _pct(bg_before, bg_after)

    result = {
        "summary_json": str(Path(args.summary_json).resolve()),
        "metrics": {
            "roi_halo_reduction_pct": halo_reduction_pct,
            "roi_gradient_gain_pct": gradient_gain_pct,
            "roi_edge_contrast_gain_pct": contrast_gain_pct,
            "seam_degradation_pct": seam_degradation_pct,
            "background_degradation_pct": background_degradation_pct,
        },
        "gates": {
            "roi_halo_reduction_ge_10pct": halo_reduction_pct >= 10.0,
            "roi_gradient_gain_ge_12pct": gradient_gain_pct >= 12.0,
            "roi_edge_contrast_gain_ge_10pct": contrast_gain_pct >= 10.0,
            "seam_degradation_le_3pct": seam_degradation_pct <= 3.0,
            "background_degradation_le_3pct": background_degradation_pct <= 3.0,
        },
    }
    result["passed"] = all(result["gates"].values())
    payload = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
