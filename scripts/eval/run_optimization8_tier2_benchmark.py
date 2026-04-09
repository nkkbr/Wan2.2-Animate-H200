import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def resolve_path(root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def extract_holdout_metrics(metrics_obj: dict[str, Any]) -> dict[str, float]:
    split_metrics = metrics_obj["split_metrics"]["holdout_eval"]
    return {
        "boundary_f1_mean": float(split_metrics["boundary_f1_mean"]),
        "alpha_mae_mean": float(split_metrics["alpha_mae_mean"]),
        "trimap_error_mean": float(split_metrics["trimap_error_mean"]),
    }


def extract_smoke_metrics(smoke_obj: dict[str, Any]) -> dict[str, float | None]:
    seam = smoke_obj.get("seam_score", {}).get("mean")
    background = smoke_obj.get("background_fluctuation", {}).get("mean")
    return {
        "seam_score_mean": None if seam is None else float(seam),
        "background_fluctuation_mean": None if background is None else float(background),
    }


def gain_higher_better(candidate: float, baseline: float) -> float | None:
    if baseline == 0:
        return None
    return 100.0 * (candidate - baseline) / baseline


def reduction_lower_better(candidate: float, baseline: float) -> float | None:
    if baseline == 0:
        return None
    return 100.0 * (baseline - candidate) / baseline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    repo_root = Path.cwd()
    manifest_path = resolve_path(repo_root, args.manifest)
    manifest = load_json(manifest_path)

    normalized: dict[str, Any] = {
        "manifest": str(manifest_path),
        "reference_baseline_id": manifest["reference_baseline_id"],
        "amplifier_baseline_id": manifest["amplifier_baseline_id"],
        "candidates": [],
    }

    candidate_index: dict[str, dict[str, Any]] = {}
    for entry in manifest["candidates"]:
        record: dict[str, Any] = {
            "id": entry["id"],
            "role": entry["role"],
            "available": bool(entry.get("available", True)),
            "maintenance_cost": entry.get("maintenance_cost"),
            "training_required": bool(entry.get("training_required", False)),
            "extra_inference_passes": int(entry.get("extra_inference_passes", 0)),
            "notes": entry.get("notes"),
        }
        if not record["available"]:
            normalized["candidates"].append(record)
            candidate_index[record["id"]] = record
            continue

        reviewed_metrics_path = resolve_path(repo_root, entry["reviewed_metrics"])
        reviewed_obj = load_json(reviewed_metrics_path)
        record["reviewed_metrics_path"] = str(reviewed_metrics_path)
        record["reviewed_holdout"] = extract_holdout_metrics(reviewed_obj)

        gate_result_path = resolve_path(repo_root, entry.get("gate_result"))
        if gate_result_path is not None:
            record["gate_result_path"] = str(gate_result_path)
            record["gate_result"] = load_json(gate_result_path)

        smoke_path = resolve_path(repo_root, entry.get("smoke_metrics"))
        if smoke_path is not None:
            record["smoke_metrics_path"] = str(smoke_path)
            record["smoke_metrics"] = extract_smoke_metrics(load_json(smoke_path))

        smoke_baseline_path = resolve_path(repo_root, entry.get("smoke_baseline_metrics"))
        if smoke_baseline_path is not None:
            record["smoke_baseline_metrics_path"] = str(smoke_baseline_path)
            record["smoke_baseline_metrics"] = extract_smoke_metrics(load_json(smoke_baseline_path))

        normalized["candidates"].append(record)
        candidate_index[record["id"]] = record

    reference = candidate_index[manifest["reference_baseline_id"]]
    amplifier = candidate_index[manifest["amplifier_baseline_id"]]
    reference_holdout = reference["reviewed_holdout"]
    amplifier_holdout = amplifier["reviewed_holdout"]
    amplifier_smoke = amplifier.get("smoke_metrics")

    for record in normalized["candidates"]:
        if not record.get("available"):
            continue

        holdout = record["reviewed_holdout"]
        record["vs_reference"] = {
            "boundary_f1_gain_pct": gain_higher_better(
                holdout["boundary_f1_mean"], reference_holdout["boundary_f1_mean"]
            ),
            "alpha_mae_reduction_pct": reduction_lower_better(
                holdout["alpha_mae_mean"], reference_holdout["alpha_mae_mean"]
            ),
            "trimap_error_reduction_pct": reduction_lower_better(
                holdout["trimap_error_mean"], reference_holdout["trimap_error_mean"]
            ),
        }
        record["vs_amplifier_baseline"] = {
            "boundary_f1_gain_pct": gain_higher_better(
                holdout["boundary_f1_mean"], amplifier_holdout["boundary_f1_mean"]
            ),
            "alpha_mae_reduction_pct": reduction_lower_better(
                holdout["alpha_mae_mean"], amplifier_holdout["alpha_mae_mean"]
            ),
            "trimap_error_reduction_pct": reduction_lower_better(
                holdout["trimap_error_mean"], amplifier_holdout["trimap_error_mean"]
            ),
        }

        candidate_smoke = record.get("smoke_metrics")
        if candidate_smoke is not None and amplifier_smoke is not None and record["id"] != manifest["amplifier_baseline_id"]:
            record["smoke_vs_amplifier_baseline"] = {
                "seam_improvement_pct": reduction_lower_better(
                    candidate_smoke["seam_score_mean"], amplifier_smoke["seam_score_mean"]
                ),
                "background_fluctuation_improvement_pct": reduction_lower_better(
                    candidate_smoke["background_fluctuation_mean"],
                    amplifier_smoke["background_fluctuation_mean"],
                ),
            }
        elif record["id"] == manifest["amplifier_baseline_id"]:
            record["smoke_vs_amplifier_baseline"] = {
                "seam_improvement_pct": 0.0,
                "background_fluctuation_improvement_pct": 0.0,
            }
        else:
            record["smoke_vs_amplifier_baseline"] = None

    output_path = resolve_path(repo_root, args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(normalized, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
