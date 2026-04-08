#!/usr/bin/env python
import argparse
import json
from pathlib import Path


def _load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_pct(after: float, before: float) -> float:
    if before is None or abs(before) < 1e-9:
        return 0.0
    return float((after / before - 1.0) * 100.0)


def _class_gate(metrics: dict | None, *, gradient: float | None = None, contrast: float | None = None, halo: float | None = None, coverage_min: float = 0.001) -> dict:
    metrics = metrics or {}
    coverage = float(metrics.get("outer_coverage_mean", metrics.get("coverage_mean", 0.0)) or 0.0)
    result = {
        "eligible": coverage >= coverage_min,
        "coverage": coverage,
    }
    if gradient is not None:
        result["gradient_gain_pct"] = float(metrics.get("gradient_gain_pct", 0.0) or 0.0)
        result["gradient_pass"] = result["gradient_gain_pct"] >= gradient if result["eligible"] else False
    if contrast is not None:
        result["contrast_gain_pct"] = float(metrics.get("contrast_gain_pct", 0.0) or 0.0)
        result["contrast_pass"] = result["contrast_gain_pct"] >= contrast if result["eligible"] else False
    if halo is not None:
        result["halo_reduction_pct"] = float(metrics.get("halo_reduction_pct", 0.0) or 0.0)
        result["halo_pass"] = result["halo_reduction_pct"] >= halo if result["eligible"] else False
    result["passed"] = result["eligible"] and all(
        value for key, value in result.items() if key.endswith("_pass")
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate Optimization4 Step06 semantic boundary specialization.")
    parser.add_argument("--summary_json", type=str, required=True)
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    summary = _load_summary(Path(args.summary_json))
    rows = {row["variant"]: row for row in summary["rows"]}
    comparison = (summary.get("semantic_comparisons") or {}).get("semantic_v1_vs_v2") or {}
    per_class = comparison.get("per_class") or {}

    face = _class_gate(per_class.get("face_boundary"), gradient=8.0, contrast=8.0, coverage_min=0.0002)
    hair = _class_gate(per_class.get("hair_boundary"), halo=10.0, coverage_min=0.0008)
    hand = _class_gate(per_class.get("hand_boundary"), gradient=8.0, contrast=8.0, coverage_min=0.0003)

    v2 = rows.get("v2", {})
    semantic = rows.get("semantic_v1", {})
    seam_before = float(v2.get("seam_score_mean", 0.0) or 0.0)
    seam_after = float(semantic.get("seam_score_mean", 0.0) or 0.0)
    bg_before = float(v2.get("background_fluctuation_mean", 0.0) or 0.0)
    bg_after = float(semantic.get("background_fluctuation_mean", 0.0) or 0.0)

    seam_degradation_pct = _safe_pct(seam_after, seam_before)
    background_degradation_pct = _safe_pct(bg_after, bg_before)

    hair_or_hand_pass = hair["passed"] or (not hair["eligible"] and hand["passed"])
    overall = face["passed"] and hair_or_hand_pass and seam_degradation_pct <= 3.0 and background_degradation_pct <= 3.0

    result = {
        "summary_json": str(Path(args.summary_json).resolve()),
        "class_gates": {
            "face_boundary": face,
            "hair_boundary": hair,
            "hand_boundary": hand,
        },
        "metrics": {
            "seam_degradation_pct": seam_degradation_pct,
            "background_degradation_pct": background_degradation_pct,
        },
        "gates": {
            "face_boundary_pass": face["passed"],
            "hair_or_hand_pass": hair_or_hand_pass,
            "seam_degradation_le_3pct": seam_degradation_pct <= 3.0,
            "background_degradation_le_3pct": background_degradation_pct <= 3.0,
        },
        "overall_passed": overall,
    }
    payload = json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
