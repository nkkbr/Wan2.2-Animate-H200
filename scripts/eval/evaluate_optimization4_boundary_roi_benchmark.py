import argparse
import json
from pathlib import Path


def _safe_pct(change, base):
    if base is None or abs(base) < 1e-8:
        return 0.0
    return float(change / base * 100.0)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Step 04 boundary ROI benchmark gates.")
    parser.add_argument("--summary_json", type=str, required=True)
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    summary = json.loads(Path(args.summary_json).read_text(encoding="utf-8"))
    rows = {row["boundary_refine_mode"]: row for row in summary["rows"]}
    baseline = rows["none"]
    candidate = rows["roi_v1"]
    roi_metrics = candidate.get("roi_metrics") or {}

    halo_before = roi_metrics.get("roi_halo_ratio_before")
    halo_after = roi_metrics.get("roi_halo_ratio_after")
    gradient_before = roi_metrics.get("roi_band_gradient_before_mean")
    gradient_after = roi_metrics.get("roi_band_gradient_after_mean")
    contrast_before = roi_metrics.get("roi_band_edge_contrast_before_mean")
    contrast_after = roi_metrics.get("roi_band_edge_contrast_after_mean")
    seam_before = baseline.get("seam_score_mean")
    seam_after = candidate.get("seam_score_mean")
    bg_before = baseline.get("background_fluctuation_mean")
    bg_after = candidate.get("background_fluctuation_mean")

    metrics = {
        "roi_halo_reduction_pct": _safe_pct((halo_before - halo_after), halo_before),
        "roi_gradient_gain_pct": _safe_pct((gradient_after - gradient_before), gradient_before),
        "roi_edge_contrast_gain_pct": _safe_pct((contrast_after - contrast_before), contrast_before),
        "seam_degradation_pct": _safe_pct((seam_after - seam_before), seam_before),
        "background_degradation_pct": _safe_pct((bg_after - bg_before), bg_before),
    }
    gates = {
        "roi_halo_reduction_ge_10pct": metrics["roi_halo_reduction_pct"] >= 10.0,
        "roi_gradient_gain_ge_8pct": metrics["roi_gradient_gain_pct"] >= 8.0,
        "roi_edge_contrast_gain_ge_8pct": metrics["roi_edge_contrast_gain_pct"] >= 8.0,
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
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
