import json
import subprocess
import tempfile
from pathlib import Path


def main() -> None:
    repo_root = Path.cwd()
    manifest = repo_root / "docs/optimization9/benchmark/feasibility_manifest.step01.json"
    rules = repo_root / "docs/optimization9/benchmark/stop_rules.step01.json"
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        summary = tmp / "summary.json"
        gate = tmp / "gate.json"
        subprocess.run(
            [
                "python",
                "scripts/eval/run_optimization9_feasibility_suite.py",
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
                "scripts/eval/evaluate_optimization9_feasibility_suite.py",
                "--summary_json",
                str(summary),
                "--stop_rules",
                str(rules),
                "--output_json",
                str(gate),
            ],
            check=True,
            cwd=repo_root,
        )
        gate_obj = json.loads(gate.read_text())

    decisions = {item["id"]: item["bucket"] for item in gate_obj["decisions"]}
    assert decisions["optimization8_non_train_reviewed_v3"] == "reference"
    assert decisions["external_alpha_matanyone_round2"] == "reject"
    assert decisions["trainable_alpha_round1"] == "interesting_but_unproven"
    assert decisions["semantic_experts_round1"] == "reject"
    assert decisions["loss_stack_round2"] == "reject"
    assert gate_obj["upgrade_candidates"] == []
    print("Optimization9 feasibility protocol: PASS")


if __name__ == "__main__":
    main()
