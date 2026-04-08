#!/usr/bin/env python
import argparse
import json
from pathlib import Path


def _load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description="Evaluate Optimization5 Step03 legacy vs decoupled_v1 AB.")
    parser.add_argument("--summary_json", type=str, required=True)
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    summary = _load_summary(Path(args.summary_json))
    rows = {row["conditioning_mode"]: row for row in summary["rows"]}
    legacy = rows["legacy"]
    decoupled = rows["decoupled_v1"]
    boundary = summary.get("boundary_metrics") or {}

    hb = float(boundary.get("halo_ratio_before", 0.0) or 0.0)
    ha = float(boundary.get("halo_ratio_after", 0.0) or 0.0)
    gb = float(boundary.get("band_gradient_before_mean", 0.0) or 0.0)
    ga = float(boundary.get("band_gradient_after_mean", 0.0) or 0.0)
    cb = float(boundary.get("band_edge_contrast_before_mean", 0.0) or 0.0)
    ca = float(boundary.get("band_edge_contrast_after_mean", 0.0) or 0.0)
    seam_baseline = float(legacy.get("seam_score_mean", 0.0) or 0.0)
    seam_candidate = float(decoupled.get("seam_score_mean", 0.0) or 0.0)
    bg_baseline = float(legacy.get("background_fluctuation_mean", 0.0) or 0.0)
    bg_candidate = float(decoupled.get("background_fluctuation_mean", 0.0) or 0.0)

    halo_reduction_pct = ((hb - ha) / hb * 100.0) if hb > 1e-9 else 0.0
    gradient_gain_pct = ((ga / gb - 1.0) * 100.0) if gb > 1e-9 else 0.0
    contrast_gain_pct = ((ca / cb - 1.0) * 100.0) if cb > 1e-9 else 0.0
    seam_degradation_pct = ((seam_candidate / seam_baseline - 1.0) * 100.0) if seam_baseline > 1e-9 else 0.0
    background_degradation_pct = ((bg_candidate / bg_baseline - 1.0) * 100.0) if bg_baseline > 1e-9 else 0.0

    gates = {
        "seam_degradation_le_3pct": seam_degradation_pct <= 3.0,
        "background_degradation_le_3pct": background_degradation_pct <= 3.0,
        "halo_reduction_ge_5pct": halo_reduction_pct >= 5.0,
        "gradient_or_contrast_positive": gradient_gain_pct > 0.0 or contrast_gain_pct > 0.0,
    }
    payload = {
        "summary_json": str(Path(args.summary_json).resolve()),
        "metrics": {
            "halo_reduction_pct": halo_reduction_pct,
            "gradient_gain_pct": gradient_gain_pct,
            "contrast_gain_pct": contrast_gain_pct,
            "seam_degradation_pct": seam_degradation_pct,
            "background_degradation_pct": background_degradation_pct,
        },
        "gates": gates,
        "passed": all(gates.values()),
    }
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
