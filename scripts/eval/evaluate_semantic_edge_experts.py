#!/usr/bin/env python
import argparse
import json
from pathlib import Path


SEMANTIC_METRIC_KEY = {
    "face": "face_boundary_f1_mean",
    "hair": "hair_boundary_f1_mean",
    "hand": "hand_boundary_f1_mean",
    "cloth": "cloth_boundary_f1_mean",
}


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _pct_improvement(baseline: float, candidate: float, lower_is_better: bool) -> float:
    if baseline == 0:
        return 0.0
    if lower_is_better:
        return float((baseline - candidate) / abs(baseline) * 100.0)
    return float((candidate - baseline) / abs(baseline) * 100.0)


def main():
    parser = argparse.ArgumentParser(description="Evaluate semantic edge experts against trainable alpha baseline.")
    parser.add_argument("--baseline_metrics", required=True)
    parser.add_argument("--candidate_metrics", required=True)
    parser.add_argument("--enabled_tags", nargs="+", required=True, choices=sorted(SEMANTIC_METRIC_KEY))
    parser.add_argument("--baseline_replacement_metrics")
    parser.add_argument("--candidate_replacement_metrics")
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    baseline = _read_json(Path(args.baseline_metrics))
    candidate = _read_json(Path(args.candidate_metrics))
    holdout_base = (baseline.get("split_metrics") or {}).get("holdout_eval") or {}
    holdout_cand = (candidate.get("split_metrics") or {}).get("holdout_eval") or {}

    semantic_gains = {}
    semantic_pass_count = 0
    for tag in args.enabled_tags:
        key = SEMANTIC_METRIC_KEY[tag]
        gain = _pct_improvement(
            float(holdout_base.get(key) or 0.0),
            float(holdout_cand.get(key) or 0.0),
            lower_is_better=False,
        )
        semantic_gains[tag] = gain
        semantic_pass_count += int(gain >= 8.0)

    overall_boundary_gain = _pct_improvement(
        float(holdout_base.get("boundary_f1_mean") or 0.0),
        float(holdout_cand.get("boundary_f1_mean") or 0.0),
        lower_is_better=False,
    )
    alpha_mae_reduction = _pct_improvement(
        float(holdout_base.get("alpha_mae_mean") or 0.0),
        float(holdout_cand.get("alpha_mae_mean") or 0.0),
        lower_is_better=True,
    )
    trimap_reduction = _pct_improvement(
        float(holdout_base.get("trimap_error_mean") or 0.0),
        float(holdout_cand.get("trimap_error_mean") or 0.0),
        lower_is_better=True,
    )

    seam_degradation = None
    background_improvement = None
    if args.baseline_replacement_metrics and args.candidate_replacement_metrics:
        baseline_repl = _read_json(Path(args.baseline_replacement_metrics))
        candidate_repl = _read_json(Path(args.candidate_replacement_metrics))
        seam_degradation = _pct_improvement(
            float((baseline_repl.get("seam_score") or {}).get("mean") or 0.0),
            float((candidate_repl.get("seam_score") or {}).get("mean") or 0.0),
            lower_is_better=True,
        )
        background_improvement = _pct_improvement(
            float((baseline_repl.get("background_fluctuation") or {}).get("mean") or 0.0),
            float((candidate_repl.get("background_fluctuation") or {}).get("mean") or 0.0),
            lower_is_better=True,
        )

    overall_safe = alpha_mae_reduction >= -2.0 and trimap_reduction >= -2.0
    gate_passed = semantic_pass_count >= 2 and overall_safe

    result = {
        "baseline_metrics": str(Path(args.baseline_metrics).resolve()),
        "candidate_metrics": str(Path(args.candidate_metrics).resolve()),
        "enabled_tags": list(args.enabled_tags),
        "semantic_gains_pct": semantic_gains,
        "semantic_pass_count": int(semantic_pass_count),
        "holdout_boundary_f1_gain_pct": overall_boundary_gain,
        "holdout_alpha_mae_reduction_pct": alpha_mae_reduction,
        "holdout_trimap_error_reduction_pct": trimap_reduction,
        "seam_degradation_pct": seam_degradation,
        "background_fluctuation_improvement_pct": background_improvement,
        "gate_passed": bool(gate_passed),
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
