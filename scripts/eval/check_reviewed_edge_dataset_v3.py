#!/usr/bin/env python
import argparse
import json
from pathlib import Path


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description="Validate reviewed edge dataset v3.")
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    schema = _read_json(Path(args.schema))
    manifest = _read_json(Path(args.manifest))
    summary = _read_json(dataset_dir / "summary.json")

    errors = []
    total_count = 0
    strong_revision_count = 0
    coverage = {key: 0 for key in ["face", "hair", "hand", "cloth", "occluded", "semi_transparent"]}

    for case in summary.get("cases", []):
        for record in case.get("records", []):
            total_count += 1
            label_json = _read_json(Path(record["label_json_path"]))
            for field in schema["required_top_level_fields"]:
                if field not in label_json:
                    errors.append(f"Missing top-level field '{field}' in {record['label_json_path']}")
            annotations = label_json.get("annotations", {})
            for name in schema["required_annotations"]:
                if name not in annotations:
                    errors.append(f"Missing annotation '{name}' in {record['label_json_path']}")
                    continue
                annotation_path = Path(record["label_json_path"]).parent / annotations[name]["path"]
                if not annotation_path.exists():
                    errors.append(f"Missing annotation file '{name}': {annotation_path}")
            review_metadata = label_json.get("review_metadata", {})
            for field in schema.get("review_metadata_fields", []):
                if field not in review_metadata:
                    errors.append(f"Missing review_metadata field '{field}' in {record['label_json_path']}")
            if review_metadata.get("revision_strength") == "strong_bootstrap_extension":
                strong_revision_count += 1
            for key in coverage:
                coverage[key] += int(label_json.get("category_presence", {}).get(key, 0))

    min_coverage = manifest["cases"][0]["minimum_category_coverage"]
    for key, minimum in min_coverage.items():
        if coverage.get(key, 0) < int(minimum):
            errors.append(f"Coverage for {key} below threshold: {coverage.get(key, 0)} < {minimum}")

    strong_revision_fraction = float(strong_revision_count / max(total_count, 1))
    if strong_revision_fraction < float(manifest["cases"][0]["minimum_strong_revision_fraction"]):
        errors.append(
            f"Strong revision fraction below threshold: {strong_revision_fraction:.4f} < "
            f"{manifest['cases'][0]['minimum_strong_revision_fraction']:.4f}"
        )

    result = {
        "dataset_dir": str(dataset_dir.resolve()),
        "schema": str(Path(args.schema).resolve()),
        "manifest": str(Path(args.manifest).resolve()),
        "total_count": total_count,
        "coverage": coverage,
        "strong_revision_count": strong_revision_count,
        "strong_revision_fraction": strong_revision_fraction,
        "schema_valid": len(errors) == 0,
        "error_count": len(errors),
        "errors": errors,
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ["total_count", "strong_revision_fraction", "schema_valid", "error_count"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
