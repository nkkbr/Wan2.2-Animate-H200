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

from wan.utils.media_io import load_mask_artifact


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _load_optional_volume(preprocess_dir: Path, metadata: dict, name: str):
    artifact = metadata.get("src_files", {}).get(name)
    if not artifact:
        return None
    path = preprocess_dir / artifact["path"]
    return load_mask_artifact(path, artifact.get("format"))


def _median_activity(volumes: list[np.ndarray]) -> np.ndarray:
    if not volumes:
        raise RuntimeError("No volumes provided")
    stacked = np.stack(volumes, axis=0)
    return np.median(stacked, axis=0).mean(axis=(1, 2))


def _align_volume(volume: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    target_t, target_h, target_w = target_shape
    if volume.shape[0] != target_t:
        raise ValueError(f"Frame count mismatch: {volume.shape[0]} vs {target_t}")
    if volume.shape[1:] == (target_h, target_w):
        return volume.astype(np.float32)
    resized = [
        cv2.resize(frame.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        for frame in volume
    ]
    return np.stack(resized, axis=0).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Extract optimization8 reviewed edge benchmark v3 keyframes.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    manifest = _read_json(Path(args.manifest))
    seed_dir = Path(manifest["seed_reviewed_dataset_dir"])
    seed_summary = _read_json(seed_dir / "summary.json")
    seed_case = seed_summary["cases"][0]
    seed_indices = sorted({int(record["preprocess_frame_index"]) for record in seed_case["records"]})

    output_cases = []
    for case in manifest["cases"]:
        preprocess_dirs = [Path(p) for p in case["consensus_preprocess_dirs"]]
        metas = [_read_json(p / "metadata.json") for p in preprocess_dirs]
        soft_alpha_volumes = []
        uncertainty_volumes = []
        hand_volumes = []
        hair_volumes = []
        for preprocess_dir, meta in zip(preprocess_dirs, metas):
            alpha_v2 = _load_optional_volume(preprocess_dir, meta, "alpha_v2")
            soft_alpha = _load_optional_volume(preprocess_dir, meta, "soft_alpha")
            unc = _load_optional_volume(preprocess_dir, meta, "uncertainty_map")
            hand = _load_optional_volume(preprocess_dir, meta, "hand_boundary")
            hair = _load_optional_volume(preprocess_dir, meta, "hair_boundary")
            soft_alpha_volumes.append(alpha_v2 if alpha_v2 is not None else soft_alpha)
            if unc is not None:
                uncertainty_volumes.append(unc)
            if hand is not None:
                hand_volumes.append(hand)
            if hair is not None:
                hair_volumes.append(hair)

        target_shape = soft_alpha_volumes[0].shape
        soft_alpha_volumes = [_align_volume(volume, target_shape) for volume in soft_alpha_volumes]
        uncertainty_volumes = [_align_volume(volume, target_shape) for volume in uncertainty_volumes]
        hand_volumes = [_align_volume(volume, target_shape) for volume in hand_volumes]
        hair_volumes = [_align_volume(volume, target_shape) for volume in hair_volumes]

        frame_count = int(target_shape[0])
        alpha_std = np.std(np.stack(soft_alpha_volumes, axis=0), axis=0).mean(axis=(1, 2))
        uncertainty_mean = _median_activity(uncertainty_volumes) if uncertainty_volumes else np.zeros(frame_count, dtype=np.float32)
        hand_mean = _median_activity(hand_volumes) if hand_volumes else np.zeros(frame_count, dtype=np.float32)
        hair_mean = _median_activity(hair_volumes) if hair_volumes else np.zeros(frame_count, dtype=np.float32)
        disagreement = alpha_std * 0.60 + uncertainty_mean * 0.25 + hand_mean * 0.10 + hair_mean * 0.05

        remaining = [idx for idx in range(frame_count) if idx not in seed_indices]
        ranked = sorted(remaining, key=lambda idx: float(disagreement[idx]), reverse=True)
        needed = int(case["target_keyframe_count"]) - len(seed_indices)
        expansion_indices = ranked[: max(0, needed)]
        selected = sorted(seed_indices + expansion_indices)

        output_cases.append(
            {
                "case_id": case["case_id"],
                "seed_indices": seed_indices,
                "expansion_indices": expansion_indices,
                "selected_indices": selected,
                "disagreement": [float(disagreement[idx]) for idx in selected],
                "frame_count": frame_count,
            }
        )

    output = {
        "manifest": str(Path(args.manifest).resolve()),
        "seed_reviewed_dataset_dir": str(seed_dir.resolve()),
        "cases": output_cases,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"case_count": len(output_cases), "selected_total": sum(len(case["selected_indices"]) for case in output_cases)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
