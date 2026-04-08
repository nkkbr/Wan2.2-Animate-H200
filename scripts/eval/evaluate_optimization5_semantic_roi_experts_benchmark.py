#!/usr/bin/env python
import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_pct(after: float, before: float) -> float:
    if before is None or abs(before) < 1e-9:
        return 0.0
    return float((after / before - 1.0) * 100.0)


def _class_gate(metrics: dict | None, *, gradient: float | None = None, contrast: float | None = None, halo: float | None = None, coverage_min: float = 0.0002) -> dict:
    metrics = metrics or {}
    coverage = float(metrics.get("outer_coverage_mean", metrics.get("coverage_mean", 0.0)) or 0.0)
    result = {"eligible": coverage >= coverage_min, "coverage": coverage}
    if gradient is not None:
        result["gradient_gain_pct"] = float(metrics.get("gradient_gain_pct", 0.0) or 0.0)
        result["gradient_pass"] = result["gradient_gain_pct"] >= gradient if result["eligible"] else False
    if contrast is not None:
        result["contrast_gain_pct"] = float(metrics.get("contrast_gain_pct", 0.0) or 0.0)
        result["contrast_pass"] = result["contrast_gain_pct"] >= contrast if result["eligible"] else False
    if halo is not None:
        result["halo_reduction_pct"] = float(metrics.get("halo_reduction_pct", 0.0) or 0.0)
        result["halo_pass"] = result["halo_reduction_pct"] >= halo if result["eligible"] else False
    result["passed"] = result["eligible"] and all(value for key, value in result.items() if key.endswith("_pass"))
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate Optimization5 Step06 semantic ROI experts benchmark.")
    parser.add_argument("--summary_json", type=str, required=True)
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    summary = _load(Path(args.summary_json))
    rows = {row["variant"]: row for row in summary["rows"]}
    comparison = (summary.get("semantic_comparisons") or {}).get("semantic_experts_v1_vs_none") or {}
    per_class = comparison.get("per_class") or {}

    face = _class_gate(per_class.get("face_boundary"), gradient=8.0, contrast=8.0, coverage_min=0.0002)
    hair = _class_gate(per_class.get("hair_boundary"), halo=10.0, coverage_min=0.0005)
    hand = _class_gate(per_class.get("hand_boundary"), gradient=10.0, coverage_min=0.0002)
    cloth = _class_gate(per_class.get("cloth_boundary"), contrast=8.0, coverage_min=0.0005)

    before = rows.get("none", {})
    after = rows.get("semantic_experts_v1", {})
    seam_degradation_pct = _safe_pct(float(after.get("seam_score_mean", 0.0) or 0.0), float(before.get("seam_score_mean", 0.0) or 0.0))
    background_degradation_pct = _safe_pct(float(after.get("background_fluctuation_mean", 0.0) or 0.0), float(before.get("background_fluctuation_mean", 0.0) or 0.0))

    any_roi_gain = face["passed"] or hair["passed"] or hand["passed"] or cloth["passed"]
    result = {
        "summary_json": str(Path(args.summary_json).resolve()),
        "class_gates": {
            "face_boundary": face,
            "hair_boundary": hair,
            "hand_boundary": hand,
            "cloth_boundary": cloth,
        },
        "metrics": {
            "seam_degradation_pct": seam_degradation_pct,
            "background_degradation_pct": background_degradation_pct,
        },
        "gates": {
            "any_roi_gain": any_roi_gain,
            "seam_degradation_le_3pct": seam_degradation_pct <= 3.0,
            "background_degradation_le_3pct": background_degradation_pct <= 3.0,
        },
        "overall_passed": any_roi_gain and seam_degradation_pct <= 3.0 and background_degradation_pct <= 3.0,
    }
    text = json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
