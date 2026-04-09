#!/usr/bin/env python
import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINE = ROOT / "runs" / "optimization6_step01_round1" / "summary.json"


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _score(gate: dict) -> float:
    return (
        gate["passed_count"] * 1000.0
        + gate["alpha_mae_reduction_pct"] * 3.0
        + gate["trimap_error_reduction_pct"] * 2.0
        + gate["hair_edge_quality_gain_pct"] * 1.5
        + gate["boundary_f1_gain_pct"] * 1.0
        - gate["runtime_penalty_pct"] * 0.1
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite_dir", required=True)
    parser.add_argument("--baseline_summary", default=str(DEFAULT_BASELINE))
    args = parser.parse_args()

    suite_dir = Path(args.suite_dir)
    baseline = _read_json(Path(args.baseline_summary))["edge_groundtruth_metrics"]
    manifest = _read_json(suite_dir / "manifest.json")

    per_model = []
    for model_id in manifest["model_ids"]:
        metrics = _read_json(suite_dir / f"{model_id}_metrics.json")
        alpha_gain = (float(baseline["alpha_mae_mean"]) - float(metrics["alpha_mae_mean"])) / max(float(baseline["alpha_mae_mean"]), 1e-8) * 100.0
        trimap_gain = (float(baseline["trimap_error_mean"]) - float(metrics["trimap_error_mean"])) / max(float(baseline["trimap_error_mean"]), 1e-8) * 100.0
        hair_gain = (float(metrics["hair_boundary_f1_mean"]) - float(baseline["hair_boundary_f1_mean"])) / max(float(baseline["hair_boundary_f1_mean"]), 1e-8) * 100.0
        boundary_gain = (float(metrics["boundary_f1_mean"]) - float(baseline["boundary_f1_mean"])) / max(float(baseline["boundary_f1_mean"]), 1e-8) * 100.0
        runtime_penalty = float(metrics["runtime_sec_mean"]) * 100.0
        gate = {
            "model_id": model_id,
            "alpha_mae_reduction_pct": alpha_gain,
            "trimap_error_reduction_pct": trimap_gain,
            "hair_edge_quality_gain_pct": hair_gain,
            "boundary_f1_gain_pct": boundary_gain,
            "runtime_penalty_pct": runtime_penalty,
            "alpha_gate_passed": alpha_gain >= 15.0,
            "trimap_gate_passed": trimap_gain >= 15.0,
            "hair_gate_passed": hair_gain >= 10.0,
        }
        gate["passed_count"] = int(gate["alpha_gate_passed"]) + int(gate["trimap_gate_passed"]) + int(gate["hair_gate_passed"])
        gate["reviewed_gate_passed"] = gate["passed_count"] >= 2
        gate["selection_score"] = _score(gate)
        per_model.append({"metrics": metrics, "gate": gate})
        (suite_dir / f"{model_id}_gate.json").write_text(
            json.dumps(gate, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    ranked = sorted(per_model, key=lambda item: item["gate"]["selection_score"], reverse=True)
    winner = ranked[0]["gate"]["model_id"] if ranked else None
    winner_passed = bool(ranked and ranked[0]["gate"]["reviewed_gate_passed"])
    summary = {
        "suite_dir": str(suite_dir.resolve()),
        "baseline_summary": str(Path(args.baseline_summary).resolve()),
        "candidate_count": len(per_model),
        "winner_model_id": winner if winner_passed else None,
        "winner_passed": winner_passed,
        "per_model": per_model,
    }
    gate_result = {
        "winner_model_id": winner if winner_passed else None,
        "winner_passed": winner_passed,
        "all_models_failed": not winner_passed,
        "benchmarked_model_ids": [item["gate"]["model_id"] for item in ranked],
    }
    (suite_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (suite_dir / "gate_result.json").write_text(json.dumps(gate_result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(gate_result, ensure_ascii=False))


if __name__ == "__main__":
    main()
