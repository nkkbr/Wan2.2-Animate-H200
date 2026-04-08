#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan.utils.edge_alpha_adapter import ADAPTER_FEATURE_KEYS, load_adapter_feature_maps, stack_feature_maps


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _load_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def main():
    parser = argparse.ArgumentParser(description="Build reviewed edge adapter training dataset.")
    parser.add_argument("--reviewed_dataset_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    dataset_dir = Path(args.reviewed_dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = _read_json(dataset_dir / "summary.json")
    bundle_cache: dict[str, np.ndarray] = {}

    features = []
    alpha_targets = []
    hard_targets = []
    trimap_unknown_targets = []
    boundary_targets = []
    hair_targets = []
    hand_targets = []
    records = []

    for case in summary["cases"]:
        for record in case["records"]:
            label_json_path = Path(record["label_json_path"])
            label_json = _read_json(label_json_path)
            preprocess_dir = str(Path(label_json["label_source_preprocess_dir"]).resolve())
            if preprocess_dir not in bundle_cache:
                feature_maps, _ = load_adapter_feature_maps(preprocess_dir, ADAPTER_FEATURE_KEYS)
                bundle_cache[preprocess_dir] = stack_feature_maps(feature_maps, ADAPTER_FEATURE_KEYS).astype(np.float16)

            feature_stack = bundle_cache[preprocess_dir]
            preprocess_idx = int(label_json["preprocess_frame_index"])
            label_root = label_json_path.parent

            alpha_label = _load_gray(label_root / label_json["annotations"]["soft_alpha"]["path"]).astype(np.float32) / 255.0
            hard_label = (_load_gray(label_root / label_json["annotations"]["hard_foreground"]["path"]) > 127).astype(np.uint8)
            trimap = _load_gray(label_root / label_json["annotations"]["trimap"]["path"]).astype(np.uint8)
            boundary = (_load_gray(label_root / label_json["annotations"]["boundary_mask"]["path"]) > 127).astype(np.uint8)
            boundary_type = _load_gray(label_root / label_json["annotations"]["boundary_type_map"]["path"]).astype(np.uint8)

            features.append(feature_stack[preprocess_idx])
            alpha_targets.append(alpha_label.astype(np.float16))
            hard_targets.append(hard_label.astype(np.uint8))
            trimap_unknown_targets.append((trimap == 128).astype(np.uint8))
            boundary_targets.append(boundary.astype(np.uint8))
            hair_targets.append((boundary_type == 2).astype(np.uint8))
            hand_targets.append((boundary_type == 3).astype(np.uint8))
            records.append({
                "case_id": case["case_id"],
                "label_json_path": str(label_json_path.resolve()),
                "preprocess_dir": preprocess_dir,
                "preprocess_frame_index": preprocess_idx,
            })

    payload = {
        "features": np.stack(features).astype(np.float16),
        "alpha_target": np.stack(alpha_targets).astype(np.float16),
        "hard_target": np.stack(hard_targets).astype(np.uint8),
        "trimap_unknown_target": np.stack(trimap_unknown_targets).astype(np.uint8),
        "boundary_target": np.stack(boundary_targets).astype(np.uint8),
        "hair_target": np.stack(hair_targets).astype(np.uint8),
        "hand_target": np.stack(hand_targets).astype(np.uint8),
    }
    dataset_npz = output_dir / "edge_adapter_dataset.npz"
    np.savez_compressed(dataset_npz, **payload)

    dataset_json = output_dir / "edge_adapter_dataset.json"
    dataset_json.write_text(
        json.dumps(
            {
                "reviewed_dataset_dir": str(dataset_dir.resolve()),
                "feature_keys": ADAPTER_FEATURE_KEYS,
                "sample_count": int(len(records)),
                "records": records,
            },
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"dataset_npz": str(dataset_npz.resolve()), "sample_count": len(records)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
