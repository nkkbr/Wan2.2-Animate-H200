#!/usr/bin/env python
import json
import subprocess
import tempfile
from pathlib import Path


def main() -> None:
    repo_root = Path.cwd()
    manifest = repo_root / "docs/optimization9/benchmark/portfolio_manifest.step06.json"
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        summary = tmp / "summary.json"
        gate = tmp / "gate.json"
        subprocess.run(
            [
                "python",
                "scripts/eval/run_optimization9_portfolio_summary.py",
                "--manifest",
                str(manifest),
                "--output_json",
                str(summary),
            ],
            check=True,
            cwd=repo_root,
        )
        subprocess.run(
            [
                "python",
                "scripts/eval/evaluate_optimization9_portfolio.py",
                "--summary_json",
                str(summary),
                "--output_json",
                str(gate),
                "--mode",
                "round3",
            ],
            check=True,
            cwd=repo_root,
        )
        summary_obj = json.loads(summary.read_text())
        gate_obj = json.loads(gate.read_text())

    route_ids = [route["id"] for route in summary_obj["routes"]]
    assert route_ids == ["bridge_round1", "layer_round3", "rgba_round2", "renderable_round2"]
    assert gate_obj["upgrade_candidates"] == []
    assert gate_obj["top_seed_route"] == "renderable_round2"
    decisions = {item["id"]: item["final_route_decision"] for item in gate_obj["decisions"]}
    assert decisions["bridge_round1"] == "reject"
    assert decisions["layer_round3"] == "interesting_but_unproven"
    assert decisions["rgba_round2"] == "interesting_but_unproven"
    assert decisions["renderable_round2"] == "interesting_but_unproven"
    assert gate_obj["portfolio_decision"] == "stop_tier3_promotion_keep_renderable_seed_only"
    print("Optimization9 portfolio gate check: PASS")


if __name__ == "__main__":
    main()
