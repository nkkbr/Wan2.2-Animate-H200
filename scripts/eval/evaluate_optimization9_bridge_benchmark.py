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


def _coerce_float(value, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Optimization9 Step02 matte bridge benchmark.")
    parser.add_argument("--baseline_metrics", required=True)
    parser.add_argument("--candidate_metrics", required=True)
    parser.add_argument("--baseline_replacement_metrics", required=True)
    parser.add_argument("--candidate_replacement_metrics", required=True)
    parser.add_argument("--roi_metrics", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    baseline = _read_json(Path(args.baseline_metrics))
    candidate = _read_json(Path(args.candidate_metrics))
    baseline_repl = _read_json(Path(args.baseline_replacement_metrics))
    candidate_repl = _read_json(Path(args.candidate_replacement_metrics))
    roi_metrics = _read_json(Path(args.roi_metrics))

    holdout_base = _nested(baseline, "split_metrics", "holdout_eval", default={}) or {}
    holdout_cand = _nested(candidate, "split_metrics", "holdout_eval", default={}) or {}

    reviewed = {
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
        "hair_edge_quality_gain_pct": _pct_improvement(
            float(holdout_base.get("hair_boundary_f1_mean") or 0.0),
            float(holdout_cand.get("hair_boundary_f1_mean") or 0.0),
            lower_is_better=False,
        ),
        "semi_transparent_quality_gain_pct": _pct_improvement(
            float(holdout_base.get("semi_transparent_boundary_f1_mean") or 0.0),
            float(holdout_cand.get("semi_transparent_boundary_f1_mean") or 0.0),
            lower_is_better=False,
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
    roi_gradient_gain_pct = roi_metrics.get("roi_gradient_gain_pct")
    if roi_gradient_gain_pct is None:
        roi_gradient_gain_pct = _pct_improvement(
            _coerce_float(roi_metrics.get("roi_band_gradient_before_mean")),
            _coerce_float(roi_metrics.get("roi_band_gradient_after_mean")),
            lower_is_better=False,
        )
    roi_edge_contrast_gain_pct = roi_metrics.get("roi_edge_contrast_gain_pct")
    if roi_edge_contrast_gain_pct is None:
        roi_edge_contrast_gain_pct = _pct_improvement(
            _coerce_float(roi_metrics.get("roi_band_edge_contrast_before_mean")),
            _coerce_float(roi_metrics.get("roi_band_edge_contrast_after_mean")),
            lower_is_better=False,
        )
    roi_halo_reduction_pct = roi_metrics.get("roi_halo_reduction_pct")
    if roi_halo_reduction_pct is None:
        roi_halo_reduction_pct = _pct_improvement(
            _coerce_float(roi_metrics.get("roi_halo_ratio_before")),
            _coerce_float(roi_metrics.get("roi_halo_ratio_after")),
            lower_is_better=True,
        )
    roi = {
        "roi_gradient_gain_pct": _coerce_float(roi_gradient_gain_pct),
        "roi_edge_contrast_gain_pct": _coerce_float(roi_edge_contrast_gain_pct),
        "roi_halo_reduction_pct": _coerce_float(roi_halo_reduction_pct),
    }

    failure_patterns = []
    if all((reviewed[key] or 0.0) <= 0.0 for key in ("boundary_f1_gain_pct", "alpha_mae_reduction_pct", "trimap_error_reduction_pct")):
        failure_patterns.append("reviewed_negative")
    if (reviewed["boundary_f1_gain_pct"] or 0.0) >= 8.0 and (reviewed["alpha_mae_reduction_pct"] or 0.0) < 2.0 and (reviewed["trimap_error_reduction_pct"] or 0.0) < 2.0:
        failure_patterns.append("threshold_only_gain")
    if (smoke["seam_improvement_pct"] or 0.0) < -20.0 or (smoke["background_improvement_pct"] or 0.0) < -20.0:
        failure_patterns.append("smoke_regression")
    if (reviewed["hair_edge_quality_gain_pct"] or 0.0) < 5.0 and (reviewed["semi_transparent_quality_gain_pct"] or 0.0) < 5.0:
        failure_patterns.append("no_semantic_breakthrough")
    if (roi["roi_gradient_gain_pct"] > 0.0 or roi["roi_edge_contrast_gain_pct"] > 0.0) and roi["roi_halo_reduction_pct"] < 0.0:
        failure_patterns.append("roi_tradeoff")

    upgrade_candidate = (
        (reviewed["boundary_f1_gain_pct"] or 0.0) >= 8.0
        and (reviewed["alpha_mae_reduction_pct"] or 0.0) >= 8.0
        and (reviewed["trimap_error_reduction_pct"] or 0.0) >= 8.0
        and (
            (reviewed["hair_edge_quality_gain_pct"] or 0.0) >= 8.0
            or (reviewed["semi_transparent_quality_gain_pct"] or 0.0) >= 8.0
            or (
                roi["roi_gradient_gain_pct"] >= 8.0
                and roi["roi_edge_contrast_gain_pct"] >= 8.0
                and roi["roi_halo_reduction_pct"] >= 0.0
            )
        )
        and (smoke["seam_improvement_pct"] or 0.0) >= -3.0
        and (smoke["background_improvement_pct"] or 0.0) >= -10.0
    )
    interesting = (
        not upgrade_candidate
        and (
            (reviewed["hair_edge_quality_gain_pct"] or 0.0) >= 5.0
            or (reviewed["semi_transparent_quality_gain_pct"] or 0.0) >= 5.0
            or (
                roi["roi_gradient_gain_pct"] >= 6.0
                and roi["roi_edge_contrast_gain_pct"] >= 6.0
                and (smoke["seam_improvement_pct"] or 0.0) >= -12.0
                and (smoke["background_improvement_pct"] or 0.0) >= -15.0
            )
            or (
                (reviewed["boundary_f1_gain_pct"] or 0.0) >= 8.0
                and (reviewed["alpha_mae_reduction_pct"] or 0.0) >= 2.0
                and (reviewed["trimap_error_reduction_pct"] or 0.0) >= 2.0
            )
        )
        and (smoke["seam_improvement_pct"] or 0.0) >= -20.0
        and (smoke["background_improvement_pct"] or 0.0) >= -20.0
    )

    if upgrade_candidate:
        bucket = "upgrade_candidate"
        reason = "Bridge route shows multi-metric reviewed strength and a new positive downstream mode."
    elif interesting:
        bucket = "interesting_but_unproven"
        reason = "Bridge route shows a different positive mode, but evidence is still insufficient to upgrade."
    else:
        bucket = "reject"
        reason = "Bridge route repeats old failure modes or fails to show a new useful signal."

    result = {
        "baseline_metrics": str(Path(args.baseline_metrics).resolve()),
        "candidate_metrics": str(Path(args.candidate_metrics).resolve()),
        "baseline_replacement_metrics": str(Path(args.baseline_replacement_metrics).resolve()),
        "candidate_replacement_metrics": str(Path(args.candidate_replacement_metrics).resolve()),
        "roi_metrics": str(Path(args.roi_metrics).resolve()),
        "bucket": bucket,
        "reason": reason,
        "failure_patterns": failure_patterns,
        "reviewed": reviewed,
        "smoke": smoke,
        "roi": roi,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
