import json
import math
from pathlib import Path


DEFAULT_SCORE_POLICY = {
    "groups": {
        "boundary": {
            "weight": 0.28,
            "metrics": [
                {
                    "name": "boundary_focus",
                    "path": "metrics.boundary.uncertainty_transition_focus_ratio_dilated",
                    "direction": "higher",
                    "weight": 0.45,
                },
                {
                    "name": "boundary_focus_to_interior",
                    "path": "metrics.boundary.uncertainty_transition_to_interior_ratio",
                    "direction": "higher",
                    "transform": "log1p",
                    "weight": 0.35,
                },
                {
                    "name": "boundary_uncertainty_penalty",
                    "path": "metrics.boundary.uncertainty_mean",
                    "direction": "lower",
                    "weight": 0.20,
                },
            ],
        },
        "face": {
            "weight": 0.26,
            "metrics": [
                {
                    "name": "face_center_jitter",
                    "path": "metrics.face.center_jitter_mean",
                    "direction": "lower",
                    "weight": 0.40,
                },
                {
                    "name": "face_width_jitter",
                    "path": "metrics.face.width_jitter_mean",
                    "direction": "lower",
                    "weight": 0.20,
                },
                {
                    "name": "face_valid_points",
                    "path": "metrics.face.valid_face_points_mean",
                    "direction": "higher",
                    "weight": 0.20,
                },
                {
                    "name": "face_landmark_confidence",
                    "path": "metrics.face.landmark_confidence_mean",
                    "direction": "higher",
                    "weight": 0.20,
                },
            ],
        },
        "pose": {
            "weight": 0.24,
            "metrics": [
                {
                    "name": "pose_body_conf_delta",
                    "path": "metrics.pose.body_conf_delta_mean",
                    "direction": "lower",
                    "weight": 0.20,
                },
                {
                    "name": "pose_body_jitter",
                    "path": "metrics.pose.body_jitter_mean",
                    "direction": "lower",
                    "weight": 0.25,
                },
                {
                    "name": "pose_hand_jitter",
                    "path": "metrics.pose.hand_jitter_mean",
                    "direction": "lower",
                    "weight": 0.20,
                },
                {
                    "name": "pose_velocity_spike_rate",
                    "path": "metrics.pose.velocity_spike_rate",
                    "direction": "lower",
                    "weight": 0.15,
                },
                {
                    "name": "pose_limb_continuity",
                    "path": "metrics.pose.limb_continuity_score",
                    "direction": "higher",
                    "weight": 0.20,
                },
            ],
        },
        "background": {
            "weight": 0.18,
            "metrics": [
                {
                    "name": "bg_temporal_fluctuation",
                    "path": "metrics.background.background_stats.temporal_fluctuation_mean",
                    "direction": "lower",
                    "weight": 0.35,
                },
                {
                    "name": "bg_band_stability",
                    "path": "metrics.background.background_stats.band_adjacent_background_stability",
                    "direction": "lower",
                    "weight": 0.35,
                },
                {
                    "name": "bg_unresolved_ratio",
                    "path": "metrics.background.background_stats.unresolved_ratio_mean",
                    "direction": "lower",
                    "weight": 0.20,
                },
                {
                    "name": "bg_support_conf_corr",
                    "path": "metrics.background.support_confidence_corr",
                    "direction": "higher",
                    "weight": 0.10,
                },
            ],
        },
        "runtime": {
            "weight": 0.04,
            "metrics": [
                {
                    "name": "runtime_total_seconds",
                    "path": "metrics.runtime.stage_seconds.total",
                    "direction": "lower",
                    "transform": "log1p",
                    "weight": 0.75,
                },
                {
                    "name": "runtime_peak_memory_gb",
                    "path": "metrics.runtime.peak_memory_gb",
                    "direction": "lower",
                    "weight": 0.25,
                },
            ],
        },
    }
}


def load_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_score_policy(path: str | Path | None = None):
    if path is None:
        return json.loads(json.dumps(DEFAULT_SCORE_POLICY))
    policy = load_json(path)
    return policy


def load_candidate_manifest(path: str | Path):
    manifest = load_json(path)
    candidates = manifest.get("candidates", [])
    if not candidates:
        raise ValueError(f"No candidates defined in {path}")
    default_name = manifest.get("default_candidate")
    if default_name is None:
        default_name = next((c["name"] for c in candidates if c.get("is_default")), candidates[0]["name"])
        manifest["default_candidate"] = default_name
    return manifest


def _nested_get(payload, path: str):
    current = payload
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
            continue
        return None
    return current


def _transform_value(value, transform: str | None):
    if value is None:
        return None
    value = float(value)
    if transform is None or transform == "identity":
        return value
    if transform == "log1p":
        return math.log1p(max(value, 0.0))
    if transform == "sqrt":
        return math.sqrt(max(value, 0.0))
    raise ValueError(f"Unsupported transform: {transform}")


def _normalize_scores(values, direction: str):
    finite = [value for value in values if value is not None and math.isfinite(value)]
    if not finite:
        return [0.0 for _ in values]
    v_min = min(finite)
    v_max = max(finite)
    if math.isclose(v_min, v_max, rel_tol=1e-8, abs_tol=1e-8):
        return [0.5 if value is not None and math.isfinite(value) else 0.0 for value in values]
    normalized = []
    for value in values:
        if value is None or not math.isfinite(value):
            normalized.append(0.0)
            continue
        ratio = (value - v_min) / (v_max - v_min)
        if direction == "higher":
            normalized.append(float(ratio))
        elif direction == "lower":
            normalized.append(float(1.0 - ratio))
        else:
            raise ValueError(f"Unsupported direction: {direction}")
    return normalized


def _candidate_is_valid(candidate: dict):
    return bool(candidate.get("passed", False)) and bool(candidate.get("contract_passed", False))


def score_candidates(candidates: list[dict], policy: dict | None = None):
    policy = policy or load_score_policy()
    scored = [json.loads(json.dumps(candidate)) for candidate in candidates]
    valid = [candidate for candidate in scored if _candidate_is_valid(candidate)]
    invalid_names = [candidate["name"] for candidate in scored if not _candidate_is_valid(candidate)]
    if not valid:
        return {
            "policy": policy,
            "ranking": [],
            "selected_candidate": None,
            "default_candidate": next((candidate["name"] for candidate in scored if candidate.get("is_default")), None),
            "invalid_candidates": invalid_names,
            "selection_reason": "no_valid_candidates",
            "selected_better_than_default": False,
            "score_margin_vs_default": None,
            "scored_candidates": scored,
        }

    valid_by_name = {candidate["name"]: candidate for candidate in valid}
    for candidate in valid:
        candidate["group_scores"] = {}
        candidate["metric_scores"] = {}
        candidate["total_score"] = 0.0

    for group_name, group_policy in policy.get("groups", {}).items():
        metrics = group_policy.get("metrics", [])
        if not metrics:
            continue
        group_weight = float(group_policy.get("weight", 0.0))
        for metric_spec in metrics:
            raw_values = []
            transformed_values = []
            for candidate in valid:
                raw_value = _nested_get(candidate, metric_spec["path"])
                raw_values.append(raw_value)
                transformed_values.append(_transform_value(raw_value, metric_spec.get("transform")))
            normalized_values = _normalize_scores(transformed_values, metric_spec["direction"])
            for candidate, raw_value, transformed_value, normalized_value in zip(valid, raw_values, transformed_values, normalized_values):
                metric_name = metric_spec["name"]
                candidate["metric_scores"][metric_name] = {
                    "group": group_name,
                    "path": metric_spec["path"],
                    "direction": metric_spec["direction"],
                    "raw_value": raw_value,
                    "transformed_value": transformed_value,
                    "normalized_score": normalized_value,
                    "weight": float(metric_spec.get("weight", 1.0)),
                }
        group_metric_weights = sum(float(metric.get("weight", 1.0)) for metric in metrics)
        group_metric_weights = max(group_metric_weights, 1e-6)
        for candidate in valid:
            weighted_sum = 0.0
            for metric_spec in metrics:
                metric_score = candidate["metric_scores"][metric_spec["name"]]
                weighted_sum += float(metric_spec.get("weight", 1.0)) * metric_score["normalized_score"]
            group_score = weighted_sum / group_metric_weights
            candidate["group_scores"][group_name] = {
                "weight": group_weight,
                "score": group_score,
            }
            candidate["total_score"] += group_weight * group_score

    ranking = sorted(
        [
            {
                "name": candidate["name"],
                "is_default": bool(candidate.get("is_default", False)),
                "total_score": candidate["total_score"],
                "group_scores": candidate["group_scores"],
            }
            for candidate in valid
        ],
        key=lambda item: item["total_score"],
        reverse=True,
    )
    selected_name = ranking[0]["name"]
    default_name = next((candidate["name"] for candidate in scored if candidate.get("is_default")), ranking[0]["name"])
    default_candidate = valid_by_name.get(default_name)
    selected_candidate = valid_by_name[selected_name]
    score_margin = None
    score_margin_pct = None
    selected_better = False
    top_advantages = []
    if default_candidate is not None:
        score_margin = float(selected_candidate["total_score"] - default_candidate["total_score"])
        score_margin_pct = float(score_margin / max(abs(default_candidate["total_score"]), 1e-6))
        selected_better = score_margin > 1e-9
        for metric_name, selected_metric in selected_candidate["metric_scores"].items():
            default_metric = default_candidate["metric_scores"].get(metric_name)
            if default_metric is None:
                continue
            delta = float(selected_metric["normalized_score"] - default_metric["normalized_score"])
            top_advantages.append({
                "metric": metric_name,
                "group": selected_metric["group"],
                "selected_normalized": selected_metric["normalized_score"],
                "default_normalized": default_metric["normalized_score"],
                "normalized_delta": delta,
                "selected_raw_value": selected_metric["raw_value"],
                "default_raw_value": default_metric["raw_value"],
            })
        top_advantages.sort(key=lambda item: item["normalized_delta"], reverse=True)
        top_advantages = top_advantages[:8]

    return {
        "policy": policy,
        "ranking": ranking,
        "selected_candidate": selected_name,
        "default_candidate": default_name,
        "invalid_candidates": invalid_names,
        "selection_reason": "highest_total_score",
        "selected_better_than_default": selected_better,
        "score_margin_vs_default": score_margin,
        "score_margin_pct_vs_default": score_margin_pct,
        "top_advantages_vs_default": top_advantages,
        "scored_candidates": scored,
    }
