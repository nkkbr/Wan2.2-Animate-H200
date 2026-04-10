#!/usr/bin/env python
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


def _signal_from_gate(route_type: str, gate: dict[str, Any]) -> dict[str, float]:
    if route_type == "bridge":
        return {
            "threshold_boundary_gain_pct": float(gate["reviewed"].get("boundary_f1_gain_pct") or 0.0),
            "hair_edge_quality_gain_pct": float(gate["reviewed"].get("hair_edge_quality_gain_pct") or 0.0),
            "semi_transparent_quality_gain_pct": float(gate["reviewed"].get("semi_transparent_quality_gain_pct") or 0.0),
            "roi_gradient_gain_pct": float(gate["roi"].get("roi_gradient_gain_pct") or 0.0) if "roi" in gate else 0.0,
        }
    if route_type == "layer":
        return {
            "occlusion_temporal_improvement_pct": float(gate["layer"].get("occlusion_temporal_improvement_pct") or 0.0),
            "occlusion_gradient_gain_pct": float(gate["layer"].get("occlusion_gradient_gain_pct") or 0.0),
            "occlusion_contrast_gain_pct": float(gate["layer"].get("occlusion_contrast_gain_pct") or 0.0),
        }
    if route_type == "rgba":
        return {
            "rgba_boundary_reconstruction_improvement_pct": float(gate["rgba"].get("rgba_boundary_reconstruction_improvement_pct") or 0.0),
            "rgba_hair_reconstruction_improvement_pct": float(gate["rgba"].get("rgba_hair_reconstruction_improvement_pct") or 0.0),
            "rgba_hand_reconstruction_improvement_pct": float(gate["rgba"].get("rgba_hand_reconstruction_improvement_pct") or 0.0),
            "rgba_cloth_reconstruction_improvement_pct": float(gate["rgba"].get("rgba_cloth_reconstruction_improvement_pct") or 0.0),
        }
    if route_type == "renderable":
        return {
            "silhouette_temporal_improvement_pct": float(gate["renderable"].get("silhouette_temporal_improvement_pct") or 0.0),
            "motion_temporal_improvement_pct": float(gate["renderable"].get("motion_temporal_improvement_pct") or 0.0),
            "occlusion_temporal_improvement_pct": float(gate["renderable"].get("occlusion_temporal_improvement_pct") or 0.0),
        }
    return {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    repo_root = Path.cwd()
    manifest_path = resolve_path(repo_root, args.manifest)
    manifest = load_json(manifest_path)

    summary: dict[str, Any] = {
        "manifest": str(manifest_path),
        "protocol_reference": manifest.get("protocol_reference"),
        "routes": [],
    }
    for entry in manifest["routes"]:
        gate_path = resolve_path(repo_root, entry["gate_result"])
        findings_path = resolve_path(repo_root, entry.get("findings"))
        gate = load_json(gate_path)
        record = {
            "id": entry["id"],
            "route_type": entry["route_type"],
            "round": int(entry["round"]),
            "engineering_complexity": int(entry["engineering_complexity"]),
            "structure_delta": int(entry["structure_delta"]),
            "notes": entry.get("notes"),
            "gate_result_path": str(gate_path),
            "findings_path": None if findings_path is None else str(findings_path),
            "route_bucket": gate.get("bucket"),
            "failure_patterns": gate.get("failure_patterns", []),
            "reviewed": gate.get("reviewed", {}),
            "smoke": gate.get("smoke", {}),
            "signal": _signal_from_gate(entry["route_type"], gate),
        }
        summary["routes"].append(record)

    output_path = resolve_path(repo_root, args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
