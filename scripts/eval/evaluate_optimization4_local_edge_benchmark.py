import argparse
import json
from pathlib import Path


def _safe_pct(change, base):
    if base is None or abs(base) < 1e-8:
        return 0.0
    return float(change / base * 100.0)


def main():
    parser = argparse.ArgumentParser(description="Evaluate optimization4 Step 07 local edge restoration gate.")
    parser.add_argument("--summary_json", type=str, required=True)
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    summary = json.loads(Path(args.summary_json).read_text(encoding="utf-8"))
    rows = {row["boundary_refine_mode"]: row for row in summary["rows"]}
    baseline = rows["roi_v1"]
    candidate = rows["local_edge_v1"]
    local_metrics = candidate.get("local_edge_metrics") or {}
    roi_metrics = local_metrics.get("roi_metrics") or {}
    semantic_metrics = local_metrics.get("semantic_metrics") or {}

    face_metrics = semantic_metrics.get("face_boundary", {})

    metrics = {
        "roi_halo_reduction_pct": _safe_pct(
            (roi_metrics.get("roi_halo_ratio_before") - roi_metrics.get("roi_halo_ratio_after")),
            roi_metrics.get("roi_halo_ratio_before"),
        ),
        "roi_gradient_gain_pct": _safe_pct(
            (roi_metrics.get("roi_band_gradient_after_mean") - roi_metrics.get("roi_band_gradient_before_mean")),
            roi_metrics.get("roi_band_gradient_before_mean"),
        ),
        "roi_edge_contrast_gain_pct": _safe_pct(
            (roi_metrics.get("roi_band_edge_contrast_after_mean") - roi_metrics.get("roi_band_edge_contrast_before_mean")),
            roi_metrics.get("roi_band_edge_contrast_before_mean"),
        ),
        "face_boundary_gradient_gain_pct": _safe_pct(
            (face_metrics.get("gradient_after_mean") - face_metrics.get("gradient_before_mean")),
            face_metrics.get("gradient_before_mean"),
        ),
        "face_boundary_contrast_gain_pct": _safe_pct(
            (face_metrics.get("contrast_after_mean") - face_metrics.get("contrast_before_mean")),
            face_metrics.get("contrast_before_mean"),
        ),
        "face_boundary_delta_ratio": float(
            (face_metrics.get("mad_mean") or 0.0) / max((roi_metrics.get("roi_band_mad_mean") or 1e-6), 1e-6)
        ),
        "seam_degradation_pct": _safe_pct(
            (candidate.get("seam_score_mean") - baseline.get("seam_score_mean")),
            baseline.get("seam_score_mean"),
        ),
        "background_degradation_pct": _safe_pct(
            (candidate.get("background_fluctuation_mean") - baseline.get("background_fluctuation_mean")),
            baseline.get("background_fluctuation_mean"),
        ),
    }
    gates = {
        "roi_halo_non_worse": metrics["roi_halo_reduction_pct"] >= 0.0,
        "roi_gradient_gain_ge_10pct": metrics["roi_gradient_gain_pct"] >= 10.0,
        "roi_edge_contrast_gain_ge_10pct": metrics["roi_edge_contrast_gain_pct"] >= 10.0,
        "face_boundary_gradient_non_negative": metrics["face_boundary_gradient_gain_pct"] >= 0.0,
        "face_boundary_contrast_non_negative": metrics["face_boundary_contrast_gain_pct"] >= 0.0,
        "face_boundary_delta_ratio_le_1.25": metrics["face_boundary_delta_ratio"] <= 1.25,
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
