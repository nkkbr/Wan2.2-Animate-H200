import json
import subprocess
import tempfile
from pathlib import Path


def main() -> None:
    repo_root = Path.cwd()
    manifest = repo_root / "docs/optimization8/benchmark/tier2_manifest.json"
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        summary = tmp / "summary.json"
        gate = tmp / "gate.json"
        subprocess.run(
            [
                "python",
                "scripts/eval/run_optimization8_tier2_benchmark.py",
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
                "scripts/eval/evaluate_optimization8_tier2_benchmark.py",
                "--summary_json",
                str(summary),
                "--output_json",
                str(gate),
            ],
            check=True,
            cwd=repo_root,
        )
        summary_obj = json.loads(summary.read_text())
        gate_obj = json.loads(gate.read_text())

    candidate_ids = [candidate["id"] for candidate in summary_obj["candidates"]]
    assert candidate_ids == [
        "non_trainable_best_from_optimization7",
        "trainable_alpha_best",
        "semantic_experts_best",
        "best_loss_stack_variant",
        "tier2_full_best",
    ]
    assert gate_obj["promote_tier2_as_amplifier"] is False
    decisions = {item["id"]: item["decision"] for item in gate_obj["decisions"]}
    assert decisions["non_trainable_best_from_optimization7"] == "reference_baseline"
    assert decisions["trainable_alpha_best"] == "keep_experimental"
    assert decisions["semantic_experts_best"] == "reject_for_now"
    assert decisions["best_loss_stack_variant"] == "reject_for_now"
    assert decisions["tier2_full_best"] == "unavailable"
    print("Optimization8 tier2 gate check: PASS")


if __name__ == "__main__":
    main()
