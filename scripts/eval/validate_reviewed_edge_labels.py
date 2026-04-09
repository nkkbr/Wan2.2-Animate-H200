#!/usr/bin/env python
import argparse
import json
from pathlib import Path


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description="Validate reviewed edge benchmark labels against schema and file existence.")
    parser.add_argument("--dataset_dir", required=True, type=str)
    parser.add_argument("--schema", required=True, type=str)
    parser.add_argument("--output_json", required=True, type=str)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    schema = _read_json(Path(args.schema))
    summary = _read_json(dataset_dir / "summary.json")

    errors = []
    reviewed_count = 0
    total_count = 0

    for case in summary.get("cases", []):
        for record in case.get("records", []):
            total_count += 1
            label_json = _read_json(Path(record["label_json_path"]))
            for field in schema["required_top_level_fields"]:
                if field not in label_json:
                    errors.append(f"Missing top-level field '{field}' in {record['label_json_path']}")
            if label_json.get("label_status") in schema.get("allowed_label_status", []):
                reviewed_count += 1
            annotations = label_json.get("annotations", {})
            for name in schema["required_annotations"]:
                if name not in annotations:
                    errors.append(f"Missing annotation '{name}' in {record['label_json_path']}")
                    continue
                p = Path(record["label_json_path"]).parent / annotations[name]["path"]
                if not p.exists():
                    errors.append(f"Missing annotation file '{name}': {p}")
            review_metadata = label_json.get("review_metadata", {})
            for field in schema.get("review_metadata_fields", []):
                if field not in review_metadata:
                    errors.append(f"Missing review_metadata field '{field}' in {record['label_json_path']}")

    result = {
        "dataset_dir": str(dataset_dir.resolve()),
        "schema": str(Path(args.schema).resolve()),
        "total_count": total_count,
        "reviewed_count": reviewed_count,
        "reviewed_fraction": float(reviewed_count / max(total_count, 1)),
        "error_count": len(errors),
        "errors": errors,
        "schema_valid": len(errors) == 0,
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ["total_count", "reviewed_fraction", "schema_valid", "error_count"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
