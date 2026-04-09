#!/usr/bin/env python
import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINE = ROOT / "runs" / "optimization6_step01_round1" / "summary.json"


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate_json", required=True)
    parser.add_argument("--baseline_summary", default=str(DEFAULT_BASELINE))
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    candidate = _read_json(Path(args.candidate_json))
    baseline = _read_json(Path(args.baseline_summary))["edge_groundtruth_metrics"]
    alpha_gain = (float(baseline["alpha_mae_mean"]) - float(candidate["alpha_mae_mean"])) / max(float(baseline["alpha_mae_mean"]), 1e-8) * 100.0
    trimap_gain = (float(baseline["trimap_error_mean"]) - float(candidate["trimap_error_mean"])) / max(float(baseline["trimap_error_mean"]), 1e-8) * 100.0
    hair_gain = (float(candidate["hair_boundary_f1_mean"]) - float(baseline["hair_boundary_f1_mean"])) / max(float(baseline["hair_boundary_f1_mean"]), 1e-8) * 100.0

    gate = {
        "model_id": candidate["model_id"],
        "alpha_mae_reduction_pct": alpha_gain,
        "trimap_error_reduction_pct": trimap_gain,
        "hair_edge_quality_gain_pct": hair_gain,
        "alpha_gate_passed": alpha_gain >= 15.0,
        "trimap_gate_passed": trimap_gain >= 15.0,
        "hair_gate_passed": hair_gain >= 10.0,
    }
    gate["passed_count"] = int(gate["alpha_gate_passed"]) + int(gate["trimap_gate_passed"]) + int(gate["hair_gate_passed"])
    gate["gate_passed"] = gate["passed_count"] >= 2

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(gate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(gate, ensure_ascii=False))


if __name__ == "__main__":
    main()
