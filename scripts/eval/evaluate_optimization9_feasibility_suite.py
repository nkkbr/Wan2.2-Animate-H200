import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def reviewed_pass_count(reviewed: dict[str, Any], rules: dict[str, Any]) -> int:
    return (
        int((reviewed.get("boundary_f1_gain_pct") or -1e9) >= rules["boundary_f1_gain_pct"])
        + int((reviewed.get("alpha_mae_reduction_pct") or -1e9) >= rules["alpha_mae_reduction_pct"])
        + int((reviewed.get("trimap_error_reduction_pct") or -1e9) >= rules["trimap_error_reduction_pct"])
    )


def semantic_breakthrough_count(semantic_gains: dict[str, float], threshold: float) -> int:
    return sum(1 for value in semantic_gains.values() if value is not None and value >= threshold)


def detect_failure_patterns(route: dict[str, Any], rules: dict[str, Any]) -> list[str]:
    reviewed = route["reviewed"]
    smoke = route["smoke"]
    patterns: list[str] = []
    if (
        (reviewed.get("boundary_f1_gain_pct") or -1e9) >= rules["interesting_but_unproven"]["strong_boundary_f1_gain_pct"]
        and (reviewed.get("alpha_mae_reduction_pct") or 0.0)
        < rules["failure_patterns"]["threshold_only_gain_max_alpha_mae_reduction_pct"]
        and (reviewed.get("trimap_error_reduction_pct") or 0.0)
        < rules["failure_patterns"]["threshold_only_gain_max_trimap_reduction_pct"]
    ):
        patterns.append("threshold_only_gain")

    if (
        smoke.get("seam_improvement_pct") is not None
        and smoke["seam_improvement_pct"] <= rules["reject"]["severe_smoke_regression_pct"]
    ) or (
        smoke.get("background_improvement_pct") is not None
        and smoke["background_improvement_pct"] <= rules["reject"]["severe_smoke_regression_pct"]
    ):
        patterns.append("smoke_regression")

    if (
        (reviewed.get("boundary_f1_gain_pct") or 0.0) <= rules["failure_patterns"]["uniform_negative_threshold_pct"]
        and (reviewed.get("alpha_mae_reduction_pct") or 0.0) <= rules["failure_patterns"]["uniform_negative_threshold_pct"]
        and (reviewed.get("trimap_error_reduction_pct") or 0.0) <= rules["failure_patterns"]["uniform_negative_threshold_pct"]
    ):
        patterns.append("uniform_negative")

    if (
        abs(reviewed.get("boundary_f1_gain_pct") or 0.0) < rules["reject"]["flat_reviewed_threshold_pct"]
        and abs(reviewed.get("alpha_mae_reduction_pct") or 0.0) < rules["reject"]["flat_reviewed_threshold_pct"]
        and abs(reviewed.get("trimap_error_reduction_pct") or 0.0) < rules["reject"]["flat_reviewed_threshold_pct"]
    ):
        patterns.append("flat_reviewed")

    if semantic_breakthrough_count(
        reviewed.get("semantic_gains_pct", {}),
        rules["interesting_but_unproven"]["strong_semantic_gain_pct"],
    ) == 0:
        patterns.append("no_semantic_breakthrough")
    return patterns


def classify_route(route: dict[str, Any], rules: dict[str, Any]) -> tuple[str, str, list[str]]:
    if route["kind"] == "reference_baseline":
        return "reference", "Frozen reference baseline.", []

    reviewed = route["reviewed"]
    smoke = route["smoke"]
    reviewed_count = reviewed_pass_count(reviewed, rules["upgrade_candidate"])
    semantic_count = semantic_breakthrough_count(
        reviewed.get("semantic_gains_pct", {}),
        rules["upgrade_candidate"]["minimum_semantic_gain_pct"],
    )
    positive_smoke = int((smoke.get("seam_improvement_pct") or -1e9) > 0.0) + int(
        (smoke.get("background_improvement_pct") or -1e9) > 0.0
    )
    patterns = detect_failure_patterns(route, rules)

    if reviewed_count >= rules["upgrade_candidate"]["minimum_reviewed_pass_count"] and (
        positive_smoke >= rules["upgrade_candidate"]["minimum_positive_smoke_signals"]
        or semantic_count > 0
    ):
        return (
            "upgrade_candidate",
            "Route shows multi-metric reviewed gains and at least one meaningful downstream positive signal.",
            patterns,
        )

    if (
        (reviewed.get("boundary_f1_gain_pct") or -1e9) >= rules["interesting_but_unproven"]["strong_boundary_f1_gain_pct"]
        or semantic_breakthrough_count(
            reviewed.get("semantic_gains_pct", {}),
            rules["interesting_but_unproven"]["strong_semantic_gain_pct"],
        )
        > 0
    ) and "smoke_regression" not in patterns:
        return (
            "interesting_but_unproven",
            "Route exhibits a genuinely different positive signal, but not enough evidence to upgrade.",
            patterns,
        )

    return (
        "reject",
        "Route repeats old failure patterns or fails to show any meaningful new positive mode.",
        patterns,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--stop_rules", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    repo_root = Path.cwd()
    summary = load_json(resolve_path(repo_root, args.summary_json))
    rules = load_json(resolve_path(repo_root, args.stop_rules))

    decisions = []
    bucket_counts = {
        "reference": 0,
        "reject": 0,
        "interesting_but_unproven": 0,
        "upgrade_candidate": 0,
    }

    for route in summary["routes"]:
        bucket, reason, patterns = classify_route(route, rules)
        bucket_counts[bucket] += 1
        decisions.append(
            {
                "id": route["id"],
                "family": route["family"],
                "kind": route["kind"],
                "bucket": bucket,
                "reason": reason,
                "failure_patterns": patterns,
                "reviewed": route["reviewed"],
                "smoke": route["smoke"],
            }
        )

    result = {
        "summary_json": str(resolve_path(repo_root, args.summary_json)),
        "stop_rules": str(resolve_path(repo_root, args.stop_rules)),
        "bucket_counts": bucket_counts,
        "decisions": decisions,
        "upgrade_candidates": [item["id"] for item in decisions if item["bucket"] == "upgrade_candidate"],
        "interesting_routes": [item["id"] for item in decisions if item["bucket"] == "interesting_but_unproven"],
    }
    if result["upgrade_candidates"]:
        result["portfolio_decision"] = "upgrade_selected_routes"
    else:
        result["portfolio_decision"] = "no_upgrade_candidates_stop_after_step01_protocol"

    output_path = resolve_path(repo_root, args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
