#!/usr/bin/env python
import argparse
import json
from pathlib import Path


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _pct_improvement(baseline: float, candidate: float, lower_is_better: bool) -> float:
    if baseline == 0:
        return 0.0
    if lower_is_better:
        return float((baseline - candidate) / abs(baseline) * 100.0)
    return float((candidate - baseline) / abs(baseline) * 100.0)


def _nested(metrics: dict, *keys, default=None):
    cur = metrics
    for key in keys:
        if cur is None:
            return default
        cur = cur.get(key)
    return cur if cur is not None else default


def main():
    parser = argparse.ArgumentParser(description="Evaluate trainable alpha model against baseline reviewed metrics.")
    parser.add_argument("--baseline_metrics", required=True)
    parser.add_argument("--candidate_metrics", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--baseline_replacement_metrics")
    parser.add_argument("--candidate_replacement_metrics")
    args = parser.parse_args()

    baseline = _read_json(Path(args.baseline_metrics))
    candidate = _read_json(Path(args.candidate_metrics))
    baseline_repl = _read_json(Path(args.baseline_replacement_metrics)) if args.baseline_replacement_metrics else {}
    candidate_repl = _read_json(Path(args.candidate_replacement_metrics)) if args.candidate_replacement_metrics else {}

    holdout_base = _nested(baseline, "split_metrics", "holdout_eval", default={}) or {}
    holdout_cand = _nested(candidate, "split_metrics", "holdout_eval", default={}) or {}

    boundary_gain = _pct_improvement(
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
    seam_degradation = _pct_improvement(
        float(_nested(baseline_repl, "seam_score", "mean", default=0.0) or 0.0),
        float(_nested(candidate_repl, "seam_score", "mean", default=0.0) or 0.0),
        lower_is_better=True,
    ) if baseline_repl and candidate_repl else None
    background_degradation = _pct_improvement(
        float(_nested(baseline_repl, "background_fluctuation", "mean", default=0.0) or 0.0),
        float(_nested(candidate_repl, "background_fluctuation", "mean", default=0.0) or 0.0),
        lower_is_better=True,
    ) if baseline_repl and candidate_repl else None

    pass_count = sum([
        boundary_gain >= 8.0,
        alpha_mae_reduction >= 10.0,
        trimap_reduction >= 10.0,
    ])
    gate_passed = pass_count >= 2

    result = {
        "baseline_metrics": str(Path(args.baseline_metrics).resolve()),
        "candidate_metrics": str(Path(args.candidate_metrics).resolve()),
        "holdout_boundary_f1_gain_pct": boundary_gain,
        "holdout_alpha_mae_reduction_pct": alpha_mae_reduction,
        "holdout_trimap_error_reduction_pct": trimap_reduction,
        "seam_degradation_pct": seam_degradation,
        "background_fluctuation_improvement_pct": background_degradation,
        "gate_passed": gate_passed,
        "pass_count": int(pass_count),
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
