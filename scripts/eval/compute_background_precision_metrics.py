#!/usr/bin/env python
import argparse
import json
from pathlib import Path


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description="Compute background precision proxy metrics for optimization3.")
    parser.add_argument("--src_root_path", required=True, type=str)
    parser.add_argument("--output_json", required=True, type=str)
    args = parser.parse_args()

    src_root = Path(args.src_root_path)
    runtime_stats = _read_json(src_root / "preprocess_runtime_stats.json")
    metadata = _read_json(src_root / "metadata.json")

    background_stats = runtime_stats.get("background", {})
    result = {
        "mode": "proxy",
        "background_mode": background_stats.get("mode"),
        "background_stats": background_stats.get("stats", {}),
        "metadata_background_mode": metadata.get("src_files", {}).get("background", {}).get("background_mode"),
        "background_artifact": metadata.get("src_files", {}).get("background", {}).get("path"),
        "visible_support_artifact": metadata.get("src_files", {}).get("visible_support", {}).get("path"),
        "unresolved_region_artifact": metadata.get("src_files", {}).get("unresolved_region", {}).get("path"),
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

