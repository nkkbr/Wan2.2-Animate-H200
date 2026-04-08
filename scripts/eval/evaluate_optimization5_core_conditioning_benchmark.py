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
    parser = argparse.ArgumentParser(description="Evaluate Optimization5 Step04 core conditioning benchmark.")
    parser.add_argument("--summary_json", type=str, required=True)
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    summary = _load_summary(Path(args.summary_json))
    rows = {row["conditioning_mode"]: row for row in summary["rows"]}
    comparison = summary["comparisons"].get("core_rich_v1") or {}

    legacy = rows["legacy"]
    core = rows["core_rich_v1"]
    boundary = comparison or {}

    halo_before = float(boundary.get("halo_ratio_before", 0.0) or 0.0)
    halo_after = float(boundary.get("halo_ratio_after", 0.0) or 0.0)
    grad_before = float(boundary.get("band_gradient_before_mean", 0.0) or 0.0)
    grad_after = float(boundary.get("band_gradient_after_mean", 0.0) or 0.0)
    contrast_before = float(boundary.get("band_edge_contrast_before_mean", 0.0) or 0.0)
    contrast_after = float(boundary.get("band_edge_contrast_after_mean", 0.0) or 0.0)
    seam_before = float(legacy.get("seam_score_mean", 0.0) or 0.0)
    seam_after = float(core.get("seam_score_mean", 0.0) or 0.0)
    bg_before = float(legacy.get("background_fluctuation_mean", 0.0) or 0.0)
    bg_after = float(core.get("background_fluctuation_mean", 0.0) or 0.0)

    halo_reduction_pct = ((halo_before - halo_after) / halo_before * 100.0) if halo_before > 1e-9 else 0.0
    gradient_gain_pct = _pct(grad_before, grad_after)
    contrast_gain_pct = _pct(contrast_before, contrast_after)
    seam_degradation_pct = _pct(seam_before, seam_after)
    background_degradation_pct = _pct(bg_before, bg_after)

    result = {
        "summary_json": str(Path(args.summary_json).resolve()),
        "metrics": {
            "halo_reduction_pct": halo_reduction_pct,
            "gradient_gain_pct": gradient_gain_pct,
            "contrast_gain_pct": contrast_gain_pct,
            "seam_degradation_pct": seam_degradation_pct,
            "background_degradation_pct": background_degradation_pct,
        },
        "gates": {
            "halo_reduction_ge_8pct": halo_reduction_pct >= 8.0,
            "gradient_gain_ge_8pct": gradient_gain_pct >= 8.0,
            "contrast_gain_ge_8pct": contrast_gain_pct >= 8.0,
            "seam_degradation_le_3pct": seam_degradation_pct <= 3.0,
            "background_degradation_le_3pct": background_degradation_pct <= 3.0,
        },
    }
    result["passed"] = all(result["gates"].values())

    payload = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
