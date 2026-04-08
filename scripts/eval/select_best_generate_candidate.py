#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan.utils.generate_candidate_selection import load_score_policy, score_generate_candidates


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description="Score generate-side edge candidates and select the best one.")
    parser.add_argument("--summary_json", type=str, required=True)
    parser.add_argument("--score_policy_json", type=str, default=None)
    parser.add_argument("--output_json", type=str, required=True)
    args = parser.parse_args()

    summary = _read_json(Path(args.summary_json))
    policy = load_score_policy(args.score_policy_json)
    result = score_generate_candidates(summary, policy)
    result["source_summary_json"] = str(Path(args.summary_json).resolve())

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "selected_candidate": result["selected_candidate"],
                "default_candidate": result["default_candidate"],
                "selected_better_than_default_ratio": result["selected_better_than_default_ratio"],
                "selection_stable_ratio": result["selection_stable_ratio"],
                "positive_edge_triplet_ratio": result["positive_edge_triplet_ratio"],
                "output_json": str(output_path.resolve()),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
