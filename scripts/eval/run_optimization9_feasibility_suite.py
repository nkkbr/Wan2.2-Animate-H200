import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def resolve_path(root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def metric_gain_higher(candidate: float, baseline: float) -> float | None:
    if baseline == 0:
        return None
    return 100.0 * (candidate - baseline) / baseline


def metric_reduction_lower(candidate: float, baseline: float) -> float | None:
    if baseline == 0:
        return None
    return 100.0 * (baseline - candidate) / baseline


def holdout_metrics(metrics_obj: dict[str, Any]) -> dict[str, float]:
    if "split_metrics" not in metrics_obj:
        raise ValueError("Reviewed metrics file missing split_metrics")
    holdout = metrics_obj["split_metrics"]["holdout_eval"]
    return {
        "boundary_f1_mean": float(holdout["boundary_f1_mean"]),
        "alpha_mae_mean": float(holdout["alpha_mae_mean"]),
        "trimap_error_mean": float(holdout["trimap_error_mean"]),
        "face_boundary_f1_mean": float(holdout["face_boundary_f1_mean"]),
        "hair_boundary_f1_mean": float(holdout["hair_boundary_f1_mean"]),
        "hand_boundary_f1_mean": float(holdout["hand_boundary_f1_mean"]),
        "cloth_boundary_f1_mean": float(holdout["cloth_boundary_f1_mean"]),
        "occluded_boundary_f1_mean": float(holdout["occluded_boundary_f1_mean"]),
        "semi_transparent_boundary_f1_mean": float(holdout["semi_transparent_boundary_f1_mean"]),
    }


def normalize_external_alpha(route: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    gate = load_json(resolve_path(repo_root, route["gate_json"]))
    metrics = load_json(resolve_path(repo_root, route["metrics_json"]))
    return {
        "id": route["id"],
        "family": route["family"],
        "kind": route["kind"],
        "available": True,
        "reviewed": {
            "boundary_f1_gain_pct": float(gate["boundary_f1_gain_pct"]),
            "alpha_mae_reduction_pct": float(gate["alpha_mae_reduction_pct"]),
            "trimap_error_reduction_pct": float(gate["trimap_error_reduction_pct"]),
            "semantic_gains_pct": {
                "hair": float(gate["hair_edge_quality_gain_pct"]),
            },
        },
        "smoke": {
            "seam_improvement_pct": None,
            "background_improvement_pct": None,
        },
        "source_paths": {
            "gate_json": str(resolve_path(repo_root, route["gate_json"])),
            "metrics_json": str(resolve_path(repo_root, route["metrics_json"])),
        },
        "notes": route.get("notes"),
        "raw_runtime_sec_mean": float(metrics["runtime_sec_mean"]),
    }


def normalize_reviewed_only(route: dict[str, Any], repo_root: Path, reference_holdout: dict[str, float]) -> dict[str, Any]:
    reviewed_obj = load_json(resolve_path(repo_root, route["reviewed_metrics"]))
    gate_obj = load_json(resolve_path(repo_root, route["gate_json"]))
    holdout = holdout_metrics(reviewed_obj)

    semantic_gains = {
        key.replace("_boundary_f1_mean", ""): metric_gain_higher(value, reference_holdout[key])
        for key, value in holdout.items()
        if key.endswith("_boundary_f1_mean") and key != "boundary_f1_mean"
    }

    smoke = {
        "seam_improvement_pct": gate_obj.get("seam_degradation_pct"),
        "background_improvement_pct": gate_obj.get("background_fluctuation_improvement_pct"),
    }
    return {
        "id": route["id"],
        "family": route["family"],
        "kind": route["kind"],
        "available": True,
        "reviewed": {
            "boundary_f1_gain_pct": metric_gain_higher(
                holdout["boundary_f1_mean"], reference_holdout["boundary_f1_mean"]
            ),
            "alpha_mae_reduction_pct": metric_reduction_lower(
                holdout["alpha_mae_mean"], reference_holdout["alpha_mae_mean"]
            ),
            "trimap_error_reduction_pct": metric_reduction_lower(
                holdout["trimap_error_mean"], reference_holdout["trimap_error_mean"]
            ),
            "semantic_gains_pct": semantic_gains,
        },
        "smoke": smoke,
        "source_paths": {
            "gate_json": str(resolve_path(repo_root, route["gate_json"])),
            "reviewed_metrics": str(resolve_path(repo_root, route["reviewed_metrics"])),
        },
        "notes": route.get("notes"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    repo_root = Path.cwd()
    manifest_path = resolve_path(repo_root, args.manifest)
    manifest = load_json(manifest_path)

    reference_route = next(
        route for route in manifest["routes"] if route["id"] == manifest["reference_route_id"]
    )
    reference_metrics = load_json(resolve_path(repo_root, reference_route["reviewed_metrics"]))
    reference_holdout = holdout_metrics(reference_metrics)

    normalized_routes: list[dict[str, Any]] = []
    normalized_routes.append(
        {
            "id": reference_route["id"],
            "family": reference_route["family"],
            "kind": reference_route["kind"],
            "available": True,
            "reviewed": {
                "boundary_f1_gain_pct": 0.0,
                "alpha_mae_reduction_pct": 0.0,
                "trimap_error_reduction_pct": 0.0,
                "semantic_gains_pct": {
                    "face": 0.0,
                    "hair": 0.0,
                    "hand": 0.0,
                    "cloth": 0.0,
                    "occluded": 0.0,
                    "semi_transparent": 0.0,
                },
            },
            "smoke": {
                "seam_improvement_pct": None,
                "background_improvement_pct": None,
            },
            "source_paths": {
                "reviewed_metrics": str(resolve_path(repo_root, reference_route["reviewed_metrics"])),
            },
            "notes": reference_route.get("notes"),
            "holdout_reference_metrics": reference_holdout,
        }
    )

    for route in manifest["routes"]:
        if route["id"] == manifest["reference_route_id"]:
            continue
        if route["family"] == "external_video_matting":
            normalized_routes.append(normalize_external_alpha(route, repo_root))
        else:
            normalized_routes.append(normalize_reviewed_only(route, repo_root, reference_holdout))

    result = {
        "manifest": str(manifest_path),
        "reference_route_id": manifest["reference_route_id"],
        "routes": normalized_routes,
    }

    output_path = resolve_path(repo_root, args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
