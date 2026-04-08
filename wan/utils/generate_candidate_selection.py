import json
import math
from pathlib import Path


DEFAULT_SCORE_POLICY = {
    "groups": {
        "boundary": {
            "weight": 0.58,
            "metrics": [
                {
                    "name": "halo_reduction_pct",
                    "path": "derived_metrics.halo_reduction_pct",
                    "direction": "higher",
                    "weight": 0.34,
                },
                {
                    "name": "gradient_gain_pct",
                    "path": "derived_metrics.gradient_gain_pct",
                    "direction": "higher",
                    "weight": 0.33,
                },
                {
                    "name": "contrast_gain_pct",
                    "path": "derived_metrics.contrast_gain_pct",
                    "direction": "higher",
                    "weight": 0.33,
                },
            ],
        },
        "stability": {
            "weight": 0.27,
            "metrics": [
                {
                    "name": "seam_degradation_pct",
                    "path": "derived_metrics.seam_degradation_pct",
                    "direction": "lower",
                    "weight": 0.55,
                },
                {
                    "name": "background_degradation_pct",
                    "path": "derived_metrics.background_degradation_pct",
                    "direction": "lower",
                    "weight": 0.45,
                },
            ],
        },
        "runtime": {
            "weight": 0.15,
            "metrics": [
                {
                    "name": "runtime_increase_pct",
                    "path": "derived_metrics.runtime_increase_pct",
                    "direction": "lower",
                    "transform": "log1p_signed",
                    "weight": 1.0,
                }
            ],
        },
    }
}


def load_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_score_policy(path: str | Path | None = None):
    if path is None:
        return json.loads(json.dumps(DEFAULT_SCORE_POLICY))
    return load_json(path)


def load_candidate_manifest(path: str | Path):
    manifest = load_json(path)
    candidates = manifest.get("candidates", [])
    if not candidates:
        raise ValueError(f"No candidates defined in {path}")
    default_name = manifest.get("default_candidate")
    if default_name is None:
        default_name = next((c["name"] for c in candidates if c.get("is_default")), candidates[0]["name"])
        manifest["default_candidate"] = default_name
    candidate_names = [candidate["name"] for candidate in candidates]
    if len(set(candidate_names)) != len(candidate_names):
        raise ValueError(f"Duplicate candidate names found in {path}")
    cases = manifest.get("cases", [])
    if not cases:
        raise ValueError(f"No cases defined in {path}")
    return manifest


def _nested_get(payload, path: str):
    current = payload
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
            continue
        return None
    return current


def _safe_pct(after, before):
    if before is None:
        return None
    before = float(before)
    if abs(before) < 1e-8:
        return 0.0
    return float((float(after) / before - 1.0) * 100.0)


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
    if transform == "log1p_signed":
        if value >= 0.0:
            return math.log1p(value)
        return -math.log1p(abs(value))
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


def derive_case_candidate_metrics(case_payload: dict, default_candidate: str):
    rows = [json.loads(json.dumps(row)) for row in case_payload.get("rows", [])]
    by_name = {row["candidate_name"]: row for row in rows}
    if default_candidate not in by_name:
        raise ValueError(
            f"Default candidate '{default_candidate}' missing from case '{case_payload.get('case_name')}'."
        )
    default_row = by_name[default_candidate]
    default_boundary = default_row.get("boundary_metrics") or {}
    default_replacement = default_row.get("replacement_metrics") or {}
    default_runtime = default_row.get("runtime_stats") or {}

    default_seam = default_row.get("seam_score_mean")
    if default_seam is None:
        default_seam = ((default_replacement.get("seam_score") or {}).get("mean"))
    default_background = default_row.get("background_fluctuation_mean")
    if default_background is None:
        default_background = ((default_replacement.get("background_fluctuation") or {}).get("mean"))
    default_runtime_sec = default_row.get("total_generate_sec")
    if default_runtime_sec is None:
        default_runtime_sec = default_runtime.get("total_generate_sec")

    for row in rows:
        replacement = row.get("replacement_metrics") or {}
        boundary = row.get("boundary_metrics") or {}
        runtime = row.get("runtime_stats") or {}
        seam = row.get("seam_score_mean")
        if seam is None:
            seam = ((replacement.get("seam_score") or {}).get("mean"))
        background = row.get("background_fluctuation_mean")
        if background is None:
            background = ((replacement.get("background_fluctuation") or {}).get("mean"))
        runtime_sec = row.get("total_generate_sec")
        if runtime_sec is None:
            runtime_sec = runtime.get("total_generate_sec")

        if row["candidate_name"] == default_candidate:
            derived = {
                "halo_reduction_pct": 0.0,
                "gradient_gain_pct": 0.0,
                "contrast_gain_pct": 0.0,
                "seam_degradation_pct": 0.0,
                "background_degradation_pct": 0.0,
                "runtime_increase_pct": 0.0,
            }
        else:
            before_halo = boundary.get("halo_ratio_before")
            after_halo = boundary.get("halo_ratio_after")
            before_gradient = boundary.get("band_gradient_before_mean")
            after_gradient = boundary.get("band_gradient_after_mean")
            before_contrast = boundary.get("band_edge_contrast_before_mean")
            after_contrast = boundary.get("band_edge_contrast_after_mean")

            halo_reduction = None
            if before_halo is not None and abs(float(before_halo)) >= 1e-8 and after_halo is not None:
                halo_reduction = float((float(before_halo) - float(after_halo)) / float(before_halo) * 100.0)

            gradient_gain = None
            if before_gradient is not None and abs(float(before_gradient)) >= 1e-8 and after_gradient is not None:
                gradient_gain = float((float(after_gradient) - float(before_gradient)) / float(before_gradient) * 100.0)

            contrast_gain = None
            if before_contrast is not None and abs(float(before_contrast)) >= 1e-8 and after_contrast is not None:
                contrast_gain = float((float(after_contrast) - float(before_contrast)) / float(before_contrast) * 100.0)

            derived = {
                "halo_reduction_pct": halo_reduction,
                "gradient_gain_pct": gradient_gain,
                "contrast_gain_pct": contrast_gain,
                "seam_degradation_pct": _safe_pct(seam, default_seam) if seam is not None and default_seam is not None else None,
                "background_degradation_pct": (
                    _safe_pct(background, default_background)
                    if background is not None and default_background is not None
                    else None
                ),
                "runtime_increase_pct": (
                    _safe_pct(runtime_sec, default_runtime_sec)
                    if runtime_sec is not None and default_runtime_sec is not None
                    else None
                ),
            }
        row["derived_metrics"] = derived
        row["is_valid"] = bool(row.get("generate_returncode", 1) == 0 and row.get("output_video"))
    return rows


def score_case_candidates(case_payload: dict, default_candidate: str, policy: dict | None = None):
    policy = policy or load_score_policy()
    rows = derive_case_candidate_metrics(case_payload, default_candidate)
    valid = [row for row in rows if row.get("is_valid")]
    if not valid:
        return {
            "case_name": case_payload.get("case_name"),
            "ranking": [],
            "selected_candidate": default_candidate,
            "default_candidate": default_candidate,
            "selected_better_than_default": False,
            "score_margin_vs_default": None,
            "rows": rows,
            "selection_reason": "no_valid_candidates",
        }

    valid_by_name = {row["candidate_name"]: row for row in valid}
    for row in valid:
        row["group_scores"] = {}
        row["metric_scores"] = {}
        row["total_score"] = 0.0

    for group_name, group_policy in policy.get("groups", {}).items():
        metrics = group_policy.get("metrics", [])
        if not metrics:
            continue
        group_weight = float(group_policy.get("weight", 0.0))
        for metric_spec in metrics:
            raw_values = [_nested_get(row, metric_spec["path"]) for row in valid]
            transformed_values = [
                _transform_value(raw_value, metric_spec.get("transform"))
                for raw_value in raw_values
            ]
            normalized_values = _normalize_scores(transformed_values, metric_spec["direction"])
            for row, raw_value, transformed_value, normalized_value in zip(valid, raw_values, transformed_values, normalized_values):
                metric_name = metric_spec["name"]
                row["metric_scores"][metric_name] = {
                    "group": group_name,
                    "raw_value": raw_value,
                    "transformed_value": transformed_value,
                    "normalized_score": normalized_value,
                    "weight": float(metric_spec.get("weight", 1.0)),
                }

        for row in valid:
            metric_scores = [
                row["metric_scores"][metric_spec["name"]]
                for metric_spec in metrics
                if metric_spec["name"] in row["metric_scores"]
            ]
            if not metric_scores:
                group_score = 0.0
            else:
                denom = sum(metric_score["weight"] for metric_score in metric_scores) or 1.0
                group_score = sum(
                    metric_score["normalized_score"] * metric_score["weight"]
                    for metric_score in metric_scores
                ) / denom
            row["group_scores"][group_name] = {
                "group_score": group_score,
                "weight": group_weight,
            }
            row["total_score"] += group_score * group_weight

    ranking = sorted(
        (
            {
                "candidate_name": row["candidate_name"],
                "total_score": row["total_score"],
                "group_scores": row["group_scores"],
                "derived_metrics": row.get("derived_metrics", {}),
            }
            for row in valid
        ),
        key=lambda item: item["total_score"],
        reverse=True,
    )
    selected_candidate = ranking[0]["candidate_name"]
    default_score = valid_by_name.get(default_candidate, {}).get("total_score")
    selected_score = valid_by_name.get(selected_candidate, {}).get("total_score")
    margin = None
    if selected_score is not None and default_score is not None:
        margin = float(selected_score - default_score)

    return {
        "case_name": case_payload.get("case_name"),
        "case_config": case_payload.get("case_config", {}),
        "ranking": ranking,
        "selected_candidate": selected_candidate,
        "default_candidate": default_candidate,
        "selected_better_than_default": (
            margin is not None and selected_candidate != default_candidate and margin > 1e-8
        ),
        "score_margin_vs_default": margin,
        "rows": rows,
        "selection_reason": "ranked_by_weighted_score",
    }


def score_generate_candidates(summary: dict, policy: dict | None = None):
    policy = policy or load_score_policy()
    default_candidate = summary.get("default_candidate")
    cases = summary.get("cases", [])
    case_results = [score_case_candidates(case_payload, default_candidate, policy) for case_payload in cases]
    valid_case_results = [case_result for case_result in case_results if case_result["ranking"]]
    if not valid_case_results:
        return {
            "policy": policy,
            "default_candidate": default_candidate,
            "selected_candidate": default_candidate,
            "selected_better_than_default_ratio": 0.0,
            "selection_stable_ratio": 1.0,
            "positive_edge_triplet_ratio": 0.0,
            "overall_ranking": [],
            "case_results": case_results,
            "passed_case_count": 0,
            "total_case_count": len(case_results),
        }

    score_totals = {}
    score_counts = {}
    selection_counts = {}
    positive_triplets = 0
    better_than_default = 0
    for case_result in valid_case_results:
        selected_name = case_result["selected_candidate"]
        selection_counts[selected_name] = selection_counts.get(selected_name, 0) + 1
        if case_result["selected_better_than_default"]:
            better_than_default += 1
        selected_row = next(
            (row for row in case_result["rows"] if row["candidate_name"] == selected_name),
            None,
        )
        derived = (selected_row or {}).get("derived_metrics", {})
        if (
            (derived.get("halo_reduction_pct") or 0.0) > 0.0
            and (derived.get("gradient_gain_pct") or 0.0) > 0.0
            and (derived.get("contrast_gain_pct") or 0.0) > 0.0
        ):
            positive_triplets += 1
        for item in case_result["ranking"]:
            name = item["candidate_name"]
            score_totals[name] = score_totals.get(name, 0.0) + float(item["total_score"])
            score_counts[name] = score_counts.get(name, 0) + 1

    overall_ranking = sorted(
        (
            {
                "candidate_name": name,
                "mean_score": score_totals[name] / max(score_counts.get(name, 1), 1),
                "case_count": score_counts.get(name, 0),
                "selection_count": selection_counts.get(name, 0),
            }
            for name in score_totals
        ),
        key=lambda item: item["mean_score"],
        reverse=True,
    )
    selected_candidate = overall_ranking[0]["candidate_name"]
    stable_count = sum(1 for case_result in valid_case_results if case_result["selected_candidate"] == selected_candidate)

    return {
        "policy": policy,
        "default_candidate": default_candidate,
        "selected_candidate": selected_candidate,
        "selected_better_than_default_ratio": float(better_than_default / max(len(valid_case_results), 1)),
        "selection_stable_ratio": float(stable_count / max(len(valid_case_results), 1)),
        "positive_edge_triplet_ratio": float(positive_triplets / max(len(valid_case_results), 1)),
        "overall_ranking": overall_ranking,
        "case_results": case_results,
        "passed_case_count": len(valid_case_results),
        "total_case_count": len(case_results),
    }
