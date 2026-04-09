#!/usr/bin/env python
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Build deterministic train/val/holdout split for optimization6 Step05.")
    parser.add_argument("--dataset_dir", required=True, type=str)
    parser.add_argument("--output_json", required=True, type=str)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    summary = json.loads((dataset_dir / "summary.json").read_text(encoding="utf-8"))
    records = []
    for case in summary["cases"]:
        for record in case["records"]:
            records.append({
                "case_id": case["case_id"],
                "preprocess_frame_index": int(record["preprocess_frame_index"]),
                "image_path": record["image_path"],
                "label_json_path": record["label_json_path"],
            })
    records.sort(key=lambda item: (item["case_id"], item["preprocess_frame_index"]))
    total = len(records)
    holdout_count = max(4, total // 6)
    val_count = max(4, total // 6)
    train_count = max(1, total - holdout_count - val_count)

    train = records[:train_count]
    val = records[train_count:train_count + val_count]
    holdout = records[train_count + val_count:]
    payload = {
        "dataset_dir": str(dataset_dir),
        "counts": {
            "train": len(train),
            "val": len(val),
            "holdout": len(holdout),
        },
        "train": train,
        "val": val,
        "holdout": holdout,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
