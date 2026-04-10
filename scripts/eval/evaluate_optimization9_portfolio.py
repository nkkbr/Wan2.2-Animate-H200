#!/usr/bin/env python
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


def _bucket_base(bucket: str | None) -> float:
    if bucket == "upgrade_candidate":
        return 120.0
    if bucket == "interesting_but_unproven":
        return 60.0
    return 0.0


def _positive_sum(values: dict[str, float]) -> float:
    return float(sum(max(float(v), 0.0) for v in values.values()))


def _signal_peak(values: dict[str, float]) -> float:
    return float(max([0.0, *[float(v) for v in values.values()]]))


def _reviewed_penalty(reviewed: dict[str, float]) -> float:
    penalty = 0.0
    boundary = float(reviewed.get("boundary_f1_gain_pct") or 0.0)
    alpha = float(reviewed.get("alpha_mae_reduction_pct") or 0.0)
    trimap = float(reviewed.get("trimap_error_reduction_pct") or 0.0)
    if boundary < 0.0:
        penalty += abs(boundary) * 0.6
    if alpha < 0.0:
        penalty += abs(alpha) * 0.8
    if trimap < 0.0:
        penalty += abs(trimap) * 0.6
    return float(penalty)


def score_route(record: dict[str, Any], mode: str) -> float:
    reviewed = record.get("reviewed", {})
    smoke = record.get("smoke", {})
    signal = record.get("signal", {})
    complexity = float(record.get("engineering_complexity") or 0.0)
    structure = float(record.get("structure_delta") or 0.0)
    bucket_score = _bucket_base(record.get("route_bucket"))
    reviewed_pos = _positive_sum(reviewed)
    smoke_pos = _positive_sum(smoke)
    signal_pos = _positive_sum(signal)
    signal_peak = _signal_peak(signal)
    reviewed_penalty = _reviewed_penalty(reviewed)
    failure_penalty = 10.0 * len(record.get("failure_patterns", []))

    if mode == "round1":
        score = bucket_score + signal_peak * 1.8 + smoke_pos * 0.4 + structure * 2.5 - complexity * 1.5 - failure_penalty
    elif mode == "round2":
        score = (
            bucket_score
            + signal_peak * 1.6
            + smoke_pos * 0.7
            + reviewed_pos * 0.6
            + structure * 2.0
            - complexity * 1.5
            - reviewed_penalty
            - failure_penalty
        )
    else:
        score = (
            bucket_score
            + signal_peak * 1.5
            + smoke_pos * 0.9
            + reviewed_pos * 0.8
            + structure * 2.0
            - complexity * 1.5
            - reviewed_penalty * 1.2
            - failure_penalty
        )
    return float(score)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--mode", choices=("round1", "round2", "round3"), default="round3")
    args = parser.parse_args()

    repo_root = Path.cwd()
    summary_path = resolve_path(repo_root, args.summary_json)
    summary = load_json(summary_path)

    decisions: list[dict[str, Any]] = []
    for record in summary["routes"]:
        score = score_route(record, args.mode)
        reviewed = record.get("reviewed", {})
        smoke = record.get("smoke", {})
        signal = record.get("signal", {})

        route_decision = record.get("route_bucket") or "reject"
        promotion_ready = (
            route_decision == "interesting_but_unproven"
            and float(reviewed.get("alpha_mae_reduction_pct") or 0.0) >= 3.0
            and float(reviewed.get("trimap_error_reduction_pct") or 0.0) >= 3.0
            and float(smoke.get("seam_improvement_pct") or 0.0) >= 10.0
            and float(smoke.get("background_improvement_pct") or 0.0) >= 10.0
            and _signal_peak(signal) >= 15.0
            and float(reviewed.get("boundary_f1_gain_pct") or 0.0) >= -10.0
        )
        if promotion_ready:
            route_decision = "upgrade_candidate"

        decisions.append(
            {
                "id": record["id"],
                "route_type": record["route_type"],
                "round": record["round"],
                "route_bucket": record.get("route_bucket"),
                "final_route_decision": route_decision,
                "portfolio_score": score,
                "reviewed": reviewed,
                "smoke": smoke,
                "signal": signal,
                "failure_patterns": record.get("failure_patterns", []),
            }
        )

    decisions.sort(key=lambda item: item["portfolio_score"], reverse=True)
    upgrade_candidates = [item["id"] for item in decisions if item["final_route_decision"] == "upgrade_candidate"]
    interesting = [item["id"] for item in decisions if item["final_route_decision"] == "interesting_but_unproven"]
    top_seed = decisions[0]["id"] if decisions else None

    final_decision: dict[str, Any] = {
        "summary_json": str(summary_path),
        "mode": args.mode,
        "decisions": decisions,
        "upgrade_candidates": upgrade_candidates,
        "interesting_routes": interesting,
        "top_seed_route": top_seed,
    }

    if upgrade_candidates:
        final_decision["portfolio_decision"] = "promote_upgrade_candidates"
        final_decision["escalation_decision"] = "promote_tier3_route"
        final_decision["recommended_next_move"] = "Proceed with the highest-ranked upgrade_candidate into a dedicated next-stage implementation plan."
    elif args.mode == "round1":
        final_decision["portfolio_decision"] = "continue_screening_keep_top2"
        final_decision["escalation_decision"] = "keep_top2_for_rescoring"
        final_decision["screening_routes"] = [item["id"] for item in decisions[:2]]
    elif args.mode == "round2":
        final_decision["portfolio_decision"] = "stop_promotion_keep_top1_seed_only"
        final_decision["escalation_decision"] = "keep_top1_seed_only"
        final_decision["seed_route"] = top_seed
    else:
        final_decision["portfolio_decision"] = "stop_tier3_promotion_keep_renderable_seed_only"
        final_decision["escalation_decision"] = "no_upgrade_candidate_keep_seed_for_memory_only"
        final_decision["seed_route"] = top_seed
        final_decision["recommended_next_move"] = (
            "Do not promote any Tier3 route. Keep the strongest interesting_but_unproven route as a remembered seed only, "
            "and avoid new escalation unless a future route exceeds the current renderable foreground evidence."
        )

    output_path = resolve_path(repo_root, args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(final_decision, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
