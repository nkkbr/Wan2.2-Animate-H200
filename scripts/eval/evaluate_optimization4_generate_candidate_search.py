#!/usr/bin/env python
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Evaluate Optimization4 Step05 generate-side edge candidate search.")
    parser.add_argument("--selection_json", type=str, required=True)
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    selection = json.loads(Path(args.selection_json).read_text(encoding="utf-8"))
    gates = {
        "selected_better_than_default_ratio_ge_0.8": float(
            selection.get("selected_better_than_default_ratio", 0.0) or 0.0
        ) >= 0.8,
        "selection_stable_ratio_ge_0.8": float(selection.get("selection_stable_ratio", 0.0) or 0.0) >= 0.8,
        "positive_edge_triplet_ratio_gt_0": float(selection.get("positive_edge_triplet_ratio", 0.0) or 0.0) > 0.0,
        "selected_candidate_not_default": selection.get("selected_candidate") != selection.get("default_candidate"),
    }
    result = {
        "selection_json": str(Path(args.selection_json).resolve()),
        "metrics": {
            "selected_better_than_default_ratio": float(selection.get("selected_better_than_default_ratio", 0.0) or 0.0),
            "selection_stable_ratio": float(selection.get("selection_stable_ratio", 0.0) or 0.0),
            "positive_edge_triplet_ratio": float(selection.get("positive_edge_triplet_ratio", 0.0) or 0.0),
        },
        "gates": gates,
        "overall_passed": all(gates.values()),
    }
    payload = json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
