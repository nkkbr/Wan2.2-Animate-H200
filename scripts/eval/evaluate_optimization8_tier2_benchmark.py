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


def pass_count(metrics: dict[str, float | None], boundary: float, alpha: float, trimap: float) -> int:
    return int((metrics.get("boundary_f1_gain_pct") or -1e9) >= boundary) + int(
        (metrics.get("alpha_mae_reduction_pct") or -1e9) >= alpha
    ) + int((metrics.get("trimap_error_reduction_pct") or -1e9) >= trimap)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--boundary_gain_threshold", type=float, default=8.0)
    parser.add_argument("--alpha_reduction_threshold", type=float, default=10.0)
    parser.add_argument("--trimap_reduction_threshold", type=float, default=10.0)
    args = parser.parse_args()

    repo_root = Path.cwd()
    summary_path = resolve_path(repo_root, args.summary_json)
    summary = load_json(summary_path)

    decisions: list[dict[str, Any]] = []
    promote_candidates: list[str] = []

    for record in summary["candidates"]:
        decision: dict[str, Any] = {
            "id": record["id"],
            "role": record["role"],
            "available": record.get("available", True),
            "maintenance_cost": record.get("maintenance_cost"),
            "training_required": record.get("training_required", False),
            "extra_inference_passes": record.get("extra_inference_passes", 0),
        }
        if not record.get("available", True):
            decision["decision"] = "unavailable"
            decision["reason"] = "No frozen full-stack Tier2 candidate exists."
            decisions.append(decision)
            continue

        vs_reference = record["vs_reference"]
        reviewed_passes = pass_count(
            vs_reference,
            args.boundary_gain_threshold,
            args.alpha_reduction_threshold,
            args.trimap_reduction_threshold,
        )
        smoke = record.get("smoke_vs_amplifier_baseline")
        seam_gain = None if smoke is None else smoke.get("seam_improvement_pct")
        background_gain = None if smoke is None else smoke.get("background_fluctuation_improvement_pct")
        smoke_positive_count = int((seam_gain or -1e9) > 0.0) + int((background_gain or -1e9) > 0.0)

        decision["vs_reference"] = vs_reference
        decision["vs_amplifier_baseline"] = record.get("vs_amplifier_baseline")
        decision["smoke_vs_amplifier_baseline"] = smoke
        decision["reviewed_pass_count"] = reviewed_passes
        decision["smoke_positive_count"] = smoke_positive_count

        if record["role"] == "reference_baseline":
            decision["decision"] = "reference_baseline"
            decision["reason"] = "Frozen non-train reviewed baseline."
        elif reviewed_passes >= 2 and smoke_positive_count >= 1:
            decision["decision"] = "promote_as_amplifier"
            decision["reason"] = "Stable reviewed gains plus positive smoke signal."
            promote_candidates.append(record["id"])
        elif record["id"] == summary["amplifier_baseline_id"]:
            decision["decision"] = "keep_experimental"
            decision["reason"] = "Useful Tier2 research baseline, but not strong enough to amplify Tier1."
        else:
            decision["decision"] = "reject_for_now"
            decision["reason"] = "Fails reviewed promotion gate and does not show positive smoke amplification."

        decisions.append(decision)

    final_decision = {
        "summary_json": str(summary_path),
        "promote_tier2_as_amplifier": bool(promote_candidates),
        "promote_candidates": promote_candidates,
        "decisions": decisions,
    }
    if promote_candidates:
        final_decision["amplification_decision"] = "promote_selected_amplifiers"
    else:
        final_decision["amplification_decision"] = "stop_amplification_keep_research_only"
        final_decision["recommended_next_move"] = (
            "Do not expand Tier2. Keep trainable_alpha_best as research scaffolding only and "
            "redirect effort toward stronger data or higher-risk breakthrough routes."
        )

    output_path = resolve_path(repo_root, args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(final_decision, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
