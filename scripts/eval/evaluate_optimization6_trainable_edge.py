#!/usr/bin/env python
import argparse
import json
from pathlib import Path


def _pct_improvement(baseline: float, candidate: float, *, lower_is_better: bool) -> float:
    if abs(baseline) <= 1e-9:
        return 0.0
    if lower_is_better:
        return (baseline - candidate) / baseline * 100.0
    return (candidate - baseline) / baseline * 100.0


def main():
    parser = argparse.ArgumentParser(description="Evaluate optimization6 Step05 trainable edge model results.")
    parser.add_argument("--baseline_metrics", required=True, type=str)
    parser.add_argument("--candidate_metrics", required=True, type=str)
    parser.add_argument("--roi_gate_json", required=True, type=str)
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    baseline = json.loads(Path(args.baseline_metrics).read_text(encoding="utf-8"))
    candidate = json.loads(Path(args.candidate_metrics).read_text(encoding="utf-8"))
    roi_gate = json.loads(Path(args.roi_gate_json).read_text(encoding="utf-8"))

    metrics = {
        "boundary_f1_gain_pct": _pct_improvement(float(baseline["boundary_f1_mean"]), float(candidate["boundary_f1_mean"]), lower_is_better=False),
        "alpha_mae_reduction_pct": _pct_improvement(float(baseline["alpha_mae_mean"]), float(candidate["alpha_mae_mean"]), lower_is_better=True),
        "trimap_error_reduction_pct": _pct_improvement(float(baseline["trimap_error_mean"]), float(candidate["trimap_error_mean"]), lower_is_better=True),
        "hair_edge_quality_gain_pct": _pct_improvement(float(baseline["hair_boundary_f1_mean"]), float(candidate["hair_boundary_f1_mean"]), lower_is_better=False),
        "roi_gradient_gain_pct": float((roi_gate.get("metrics") or {}).get("roi_gradient_gain_pct") or 0.0),
        "roi_edge_contrast_gain_pct": float((roi_gate.get("metrics") or {}).get("roi_edge_contrast_gain_pct") or 0.0),
        "roi_halo_reduction_pct": float((roi_gate.get("metrics") or {}).get("roi_halo_reduction_pct") or 0.0),
    }
    gates = {
        "boundary_f1_gain_ge_5pct": metrics["boundary_f1_gain_pct"] >= 5.0,
        "alpha_mae_reduction_ge_10pct": metrics["alpha_mae_reduction_pct"] >= 10.0,
        "trimap_error_reduction_ge_10pct": metrics["trimap_error_reduction_pct"] >= 10.0,
        "hair_edge_quality_gain_ge_8pct": metrics["hair_edge_quality_gain_pct"] >= 8.0,
        "roi_gradient_gain_ge_8pct": metrics["roi_gradient_gain_pct"] >= 8.0,
        "roi_edge_contrast_gain_ge_8pct": metrics["roi_edge_contrast_gain_pct"] >= 8.0,
        "roi_halo_reduction_ge_8pct": metrics["roi_halo_reduction_pct"] >= 8.0,
    }
    result = {
        "metrics": metrics,
        "gates": gates,
        "passed": all(gates.values()),
    }
    payload = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
