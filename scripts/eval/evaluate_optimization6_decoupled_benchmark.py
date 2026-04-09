#!/usr/bin/env python
import argparse
import json
from pathlib import Path


def _load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _pct(before: float, after: float) -> float:
    if abs(before) <= 1e-9:
        return 0.0
    return (after / before - 1.0) * 100.0


def main():
    parser = argparse.ArgumentParser(description="Evaluate Optimization6 Step03 legacy vs decoupled_v2 AB.")
    parser.add_argument("--summary_json", type=str, required=True)
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    summary = _load_summary(Path(args.summary_json))
    rows = {row["conditioning_mode"]: row for row in summary["rows"]}
    legacy = rows["legacy"]
    candidate = rows["decoupled_v2"]
    boundary = summary.get("comparisons", {}).get("decoupled_v2") or {}

    hb = float(boundary.get("halo_ratio_before", 0.0) or 0.0)
    ha = float(boundary.get("halo_ratio_after", 0.0) or 0.0)
    gb = float(boundary.get("band_gradient_before_mean", 0.0) or 0.0)
    ga = float(boundary.get("band_gradient_after_mean", 0.0) or 0.0)
    cb = float(boundary.get("band_edge_contrast_before_mean", 0.0) or 0.0)
    ca = float(boundary.get("band_edge_contrast_after_mean", 0.0) or 0.0)
    seam_before = float(legacy.get("seam_score_mean", 0.0) or 0.0)
    seam_after = float(candidate.get("seam_score_mean", 0.0) or 0.0)
    bg_before = float(legacy.get("background_fluctuation_mean", 0.0) or 0.0)
    bg_after = float(candidate.get("background_fluctuation_mean", 0.0) or 0.0)

    halo_reduction_pct = ((hb - ha) / hb * 100.0) if hb > 1e-9 else 0.0
    gradient_gain_pct = _pct(gb, ga)
    contrast_gain_pct = _pct(cb, ca)
    seam_degradation_pct = _pct(seam_before, seam_after)
    background_degradation_pct = _pct(bg_before, bg_after)

    gates = {
        "background_fluctuation_not_worse": background_degradation_pct <= 0.0,
        "seam_score_not_worse": seam_degradation_pct <= 0.0,
        "roi_gradient_gain_ge_5pct": gradient_gain_pct >= 5.0,
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
