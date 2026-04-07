#!/usr/bin/env python
import argparse
import json
from pathlib import Path


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description="Summarize optimization3 validation and emit gate_result.json.")
    parser.add_argument("--summary_json", required=True, type=str)
    parser.add_argument("--gate_policy", required=True, type=str)
    parser.add_argument("--output_json", required=True, type=str)
    args = parser.parse_args()

    summary = _read_json(Path(args.summary_json))
    policy = _read_json(Path(args.gate_policy))

    case_map = {case["name"]: case for case in summary.get("cases", [])}
    missing_cases = [name for name in policy.get("required_case_names", []) if name not in case_map]
    failed_cases = [name for name, case in case_map.items() if not case.get("passed", False)]

    benchmark_info = summary.get("benchmark", {})
    missing_sets = [
        name for name in policy.get("required_benchmark_sets", [])
        if not benchmark_info.get(name) or len(benchmark_info.get(name, [])) == 0
    ]

    output_path = Path(args.output_json)
    required_outputs = []
    for rel in policy.get("required_outputs", []):
        candidate = output_path.parent / rel
        exists = candidate.exists() or candidate.resolve() == output_path.resolve()
        required_outputs.append({"name": rel, "exists": exists, "path": str(candidate.resolve())})

    output_missing = [item["name"] for item in required_outputs if not item["exists"] and item["name"] != output_path.name]

    gate_passed = (
        not missing_cases
        and not failed_cases
        and not missing_sets
        and not output_missing
        and summary.get("all_passed", False)
    )

    result = {
        "policy_name": policy.get("policy_name"),
        "summary_json": str(Path(args.summary_json).resolve()),
        "all_cases_passed": summary.get("all_passed", False),
        "missing_required_cases": missing_cases,
        "failed_cases": failed_cases,
        "missing_benchmark_sets": missing_sets,
        "required_outputs": required_outputs,
        "gate_passed": gate_passed,
    }
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
