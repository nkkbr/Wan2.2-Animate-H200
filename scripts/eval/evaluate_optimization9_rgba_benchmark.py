#!/usr/bin/env python
import argparse
import json
from pathlib import Path


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _pct_improvement(baseline: float | None, candidate: float | None, *, lower_is_better: bool) -> float | None:
    if baseline in (None, 0) or candidate is None:
        return None
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Optimization9 Step04 RGBA foreground benchmark.")
    parser.add_argument("--baseline_metrics", required=True)
    parser.add_argument("--candidate_metrics", required=True)
    parser.add_argument("--baseline_replacement_metrics", required=True)
    parser.add_argument("--candidate_replacement_metrics", required=True)
    parser.add_argument("--rgba_metrics", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    baseline = _read_json(Path(args.baseline_metrics))
    candidate = _read_json(Path(args.candidate_metrics))
    baseline_repl = _read_json(Path(args.baseline_replacement_metrics))
    candidate_repl = _read_json(Path(args.candidate_replacement_metrics))
    rgba_metrics = _read_json(Path(args.rgba_metrics))

    holdout_base = _nested(baseline, "split_metrics", "holdout_eval", default={}) or {}
    holdout_cand = _nested(candidate, "split_metrics", "holdout_eval", default={}) or {}

    reviewed = {
        "hair_boundary_f1_gain_pct": _pct_improvement(
            float(holdout_base.get("hair_boundary_f1_mean") or 0.0),
            float(holdout_cand.get("hair_boundary_f1_mean") or 0.0),
            lower_is_better=False,
        ),
        "hand_boundary_f1_gain_pct": _pct_improvement(
            float(holdout_base.get("hand_boundary_f1_mean") or 0.0),
            float(holdout_cand.get("hand_boundary_f1_mean") or 0.0),
            lower_is_better=False,
        ),
        "cloth_boundary_f1_gain_pct": _pct_improvement(
            float(holdout_base.get("cloth_boundary_f1_mean") or 0.0),
            float(holdout_cand.get("cloth_boundary_f1_mean") or 0.0),
            lower_is_better=False,
        ),
        "boundary_f1_gain_pct": _pct_improvement(
            float(holdout_base.get("boundary_f1_mean") or 0.0),
            float(holdout_cand.get("boundary_f1_mean") or 0.0),
            lower_is_better=False,
        ),
        "alpha_mae_reduction_pct": _pct_improvement(
            float(holdout_base.get("alpha_mae_mean") or 0.0),
            float(holdout_cand.get("alpha_mae_mean") or 0.0),
            lower_is_better=True,
        ),
        "trimap_error_reduction_pct": _pct_improvement(
            float(holdout_base.get("trimap_error_mean") or 0.0),
            float(holdout_cand.get("trimap_error_mean") or 0.0),
            lower_is_better=True,
        ),
    }
    smoke = {
        "seam_improvement_pct": _pct_improvement(
            float(_nested(baseline_repl, "seam_score", "mean", default=0.0) or 0.0),
            float(_nested(candidate_repl, "seam_score", "mean", default=0.0) or 0.0),
            lower_is_better=True,
        ),
        "background_improvement_pct": _pct_improvement(
            float(_nested(baseline_repl, "background_fluctuation", "mean", default=0.0) or 0.0),
            float(_nested(candidate_repl, "background_fluctuation", "mean", default=0.0) or 0.0),
            lower_is_better=True,
        ),
    }
    rgba = {
        "rgba_boundary_reconstruction_improvement_pct": float(rgba_metrics.get("rgba_boundary_reconstruction_improvement_pct") or 0.0),
        "rgba_hair_reconstruction_improvement_pct": float(rgba_metrics.get("rgba_hair_reconstruction_improvement_pct") or 0.0),
        "rgba_hand_reconstruction_improvement_pct": float(rgba_metrics.get("rgba_hand_reconstruction_improvement_pct") or 0.0),
        "rgba_cloth_reconstruction_improvement_pct": float(rgba_metrics.get("rgba_cloth_reconstruction_improvement_pct") or 0.0),
    }

    failure_patterns = []
    semantic_peak = max(
        rgba["rgba_hair_reconstruction_improvement_pct"],
        rgba["rgba_hand_reconstruction_improvement_pct"],
        rgba["rgba_cloth_reconstruction_improvement_pct"],
    )
    if semantic_peak <= 0.0 and (reviewed["hair_boundary_f1_gain_pct"] or 0.0) <= 0.0 and (reviewed["cloth_boundary_f1_gain_pct"] or 0.0) <= 0.0:
        failure_patterns.append("no_semantic_rgba_breakthrough")
    if (smoke["seam_improvement_pct"] or 0.0) < -15.0 or (smoke["background_improvement_pct"] or 0.0) < -15.0:
        failure_patterns.append("smoke_regression")
    if (rgba["rgba_boundary_reconstruction_improvement_pct"] or 0.0) > 5.0 and (reviewed["alpha_mae_reduction_pct"] or 0.0) < -5.0:
        failure_patterns.append("reconstruction_only_tradeoff")

    upgrade_candidate = (
        (
            (reviewed["hair_boundary_f1_gain_pct"] or 0.0) >= 8.0
            or (reviewed["hand_boundary_f1_gain_pct"] or 0.0) >= 8.0
            or (reviewed["cloth_boundary_f1_gain_pct"] or 0.0) >= 8.0
            or (rgba["rgba_boundary_reconstruction_improvement_pct"] or 0.0) >= 15.0
        )
        and (reviewed["alpha_mae_reduction_pct"] or 0.0) >= 1.0
        and (smoke["seam_improvement_pct"] or 0.0) >= -5.0
        and (smoke["background_improvement_pct"] or 0.0) >= -8.0
    )
    interesting = (
        not upgrade_candidate
        and (
            semantic_peak >= 8.0
            or (rgba["rgba_boundary_reconstruction_improvement_pct"] or 0.0) >= 10.0
            or (reviewed["hair_boundary_f1_gain_pct"] or 0.0) >= 5.0
        )
        and (smoke["seam_improvement_pct"] or 0.0) >= -15.0
        and (smoke["background_improvement_pct"] or 0.0) >= -12.0
        and (reviewed["alpha_mae_reduction_pct"] or 0.0) >= -20.0
    )

    if upgrade_candidate:
        bucket = "upgrade_candidate"
        reason = "RGBA foreground route shows a structurally better paired foreground/composite behavior."
    elif interesting:
        bucket = "interesting_but_unproven"
        reason = "RGBA foreground route shows some paired-output value, but evidence is still insufficient."
    else:
        bucket = "reject"
        reason = "RGBA foreground route does not show a meaningful semantic or composite breakthrough."

    result = {
        "baseline_metrics": str(Path(args.baseline_metrics).resolve()),
        "candidate_metrics": str(Path(args.candidate_metrics).resolve()),
        "baseline_replacement_metrics": str(Path(args.baseline_replacement_metrics).resolve()),
        "candidate_replacement_metrics": str(Path(args.candidate_replacement_metrics).resolve()),
        "rgba_metrics": str(Path(args.rgba_metrics).resolve()),
        "bucket": bucket,
        "reason": reason,
        "failure_patterns": failure_patterns,
        "reviewed": reviewed,
        "smoke": smoke,
        "rgba": rgba,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
