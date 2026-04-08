#!/usr/bin/env python
import argparse
import json
from pathlib import Path


def _load_summary(path: Path) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected summary JSON to contain a list. Got: {type(rows)!r}")
    return rows


def _index_rows(rows: list[dict]) -> dict[tuple[str, str], dict]:
    indexed = {}
    for row in rows:
        key = (row["conditioning_mode"], row["boundary_refine_mode"])
        indexed[key] = row
    return indexed


def _compute_gate(candidate: dict, baseline: dict) -> dict:
    boundary = candidate.get("boundary_metrics") or {}
    hb = float(boundary.get("halo_ratio_before", 0.0) or 0.0)
    ha = float(boundary.get("halo_ratio_after", 0.0) or 0.0)
    gb = float(boundary.get("band_gradient_before_mean", 0.0) or 0.0)
    ga = float(boundary.get("band_gradient_after_mean", 0.0) or 0.0)
    cb = float(boundary.get("band_edge_contrast_before_mean", 0.0) or 0.0)
    ca = float(boundary.get("band_edge_contrast_after_mean", 0.0) or 0.0)

    seam_baseline = float(baseline.get("seam_score_mean", 0.0) or 0.0)
    seam_candidate = float(candidate.get("seam_score_mean", 0.0) or 0.0)
    runtime_baseline = float(baseline.get("total_generate_sec", 0.0) or 0.0)
    runtime_candidate = float(candidate.get("total_generate_sec", 0.0) or 0.0)

    halo_reduction_pct = ((hb - ha) / hb * 100.0) if hb > 1e-9 else 0.0
    gradient_gain_pct = ((ga / gb - 1.0) * 100.0) if gb > 1e-9 else 0.0
    contrast_gain_pct = ((ca / cb - 1.0) * 100.0) if cb > 1e-9 else 0.0
    seam_degradation_pct = ((seam_candidate / seam_baseline - 1.0) * 100.0) if seam_baseline > 1e-9 else 0.0
    runtime_increase_pct = ((runtime_candidate / runtime_baseline - 1.0) * 100.0) if runtime_baseline > 1e-9 else 0.0

    gates = {
        "halo_reduction_ge_25pct": halo_reduction_pct >= 25.0,
        "gradient_gain_ge_5pct": gradient_gain_pct >= 5.0,
        "contrast_gain_ge_5pct": contrast_gain_pct >= 5.0,
        "seam_degradation_le_5pct": seam_degradation_pct <= 5.0,
        "runtime_increase_le_50pct": runtime_increase_pct <= 50.0,
    }
    return {
        "candidate": {
            "conditioning_mode": candidate["conditioning_mode"],
            "boundary_refine_mode": candidate["boundary_refine_mode"],
        },
        "baseline": {
            "conditioning_mode": baseline["conditioning_mode"],
            "boundary_refine_mode": baseline["boundary_refine_mode"],
        },
        "metrics": {
            "halo_reduction_pct": halo_reduction_pct,
            "gradient_gain_pct": gradient_gain_pct,
            "contrast_gain_pct": contrast_gain_pct,
            "seam_degradation_pct": seam_degradation_pct,
            "runtime_increase_pct": runtime_increase_pct,
        },
        "gates": gates,
        "passed": all(gates.values()),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Step 07 rich generate benchmark gates.")
    parser.add_argument("--summary_json", type=str, required=True, help="Path to summary.json emitted by run_generate_rich_signal_benchmark.py")
    parser.add_argument("--output_json", type=str, default=None, help="Optional path to save gate results.")
    args = parser.parse_args()

    summary_path = Path(args.summary_json)
    rows = _load_summary(summary_path)
    indexed = _index_rows(rows)

    evaluations = []
    for conditioning_mode in ("legacy", "rich"):
        baseline = indexed[(conditioning_mode, "none")]
        candidate = indexed[(conditioning_mode, "v2")]
        deterministic = indexed[(conditioning_mode, "deterministic")]
        evaluations.append({
            "conditioning_mode": conditioning_mode,
            "v2_vs_none": _compute_gate(candidate, baseline),
            "deterministic_vs_none": _compute_gate(deterministic, baseline),
        })

    payload = {
        "summary_json": str(summary_path.resolve()),
        "evaluations": evaluations,
        "overall_v2_passed": any(item["v2_vs_none"]["passed"] for item in evaluations),
    }

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
