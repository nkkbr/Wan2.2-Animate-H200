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

from wan.modules.animate.preprocess.external_alpha_base import build_external_alpha_adapter
from wan.utils.external_alpha_registry import get_external_model_entry
from wan.utils.media_io import load_mask_artifact, load_person_mask_artifact, load_rgb_artifact


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _read_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _boundary_f1(pred_mask: np.ndarray, label_mask: np.ndarray) -> float:
    kernel = np.ones((3, 3), np.uint8)
    pred_edge = cv2.morphologyEx(pred_mask.astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0
    label_edge = cv2.morphologyEx(label_mask.astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0
    tol = np.ones((5, 5), np.uint8)
    pred_match = cv2.dilate(pred_edge.astype(np.uint8), tol, iterations=1) > 0
    label_match = cv2.dilate(label_edge.astype(np.uint8), tol, iterations=1) > 0
    matched_pred = np.logical_and(pred_edge, label_match).sum()
    matched_label = np.logical_and(label_edge, pred_match).sum()
    precision = float(matched_pred / max(pred_edge.sum(), 1))
    recall = float(matched_label / max(label_edge.sum(), 1))
    if precision + recall == 0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def _sad(pred_alpha: np.ndarray, label_alpha: np.ndarray) -> float:
    return float(np.abs(pred_alpha - label_alpha).sum())


def _run_candidate(model_id: str, dataset_dir: Path, background_preprocess_dir: Path, registry_path: Path) -> dict:
    summary = _read_json(dataset_dir / "summary.json")
    meta = _read_json(background_preprocess_dir / "metadata.json")
    entry = get_external_model_entry(model_id, path=registry_path)
    bg_artifact = meta["src_files"]["background"]
    bg_volume = load_rgb_artifact(background_preprocess_dir / bg_artifact["path"], bg_artifact.get("format"))
    person_mask_volume = _load_initial_mask_volume(background_preprocess_dir, meta)

    adapter = build_external_alpha_adapter(model_id=model_id, registry_path=registry_path)
    boundary_scores = []
    trimap_errors = []
    alpha_mae_values = []
    alpha_sad_values = []
    hair_scores = []
    runtime_values = []
    records = []

    for case in summary["cases"]:
        adapter.reset_sequence_state()
        ordered_records = sorted(case["records"], key=lambda item: int(item["preprocess_frame_index"]))
        sequence_outputs = None
        if entry.get("task_type") == "video_matting_with_first_frame_mask":
            target_shape = bg_volume.shape[1:3]
            frame_count = int(person_mask_volume.shape[0])
            stride_candidates = [
                int(round(int(item["source_frame_index"]) / max(int(item["preprocess_frame_index"]), 1)))
                for item in ordered_records
                if int(item["preprocess_frame_index"]) > 0
            ]
            source_stride = max(1, int(round(float(np.median(stride_candidates))))) if stride_candidates else 1
            target_indices = [index * source_stride for index in range(frame_count)]
            frames = _load_video_frames(case["video_path"], target_indices, target_shape)
            adapter.set_sequence_context(initial_mask=(person_mask_volume[0] * 255.0).astype(np.uint8))
            sequence_outputs = []
            for frame in frames:
                sequence_outputs.append(adapter.infer(frame))
        for record in ordered_records:
            label_json = _read_json(Path(record["label_json_path"]))
            label_root = Path(record["label_json_path"]).parent
            preprocess_idx = int(label_json["preprocess_frame_index"])
            if sequence_outputs is not None:
                image_rgb = sequence_outputs[preprocess_idx].get("foreground")
                if image_rgb is None:
                    image_rgb = _read_rgb(Path(record["image_path"])).astype(np.float32) / 255.0
                output = sequence_outputs[preprocess_idx]
                image_rgb = np.clip(image_rgb * 255.0, 0.0, 255.0).astype(np.uint8)
            else:
                image_rgb = _read_rgb(Path(record["image_path"]))
                background_rgb = bg_volume[preprocess_idx]
                output = adapter.infer(image_rgb, background_rgb)
            pred_alpha = np.clip(output["alpha"].astype(np.float32), 0.0, 1.0)
            pred_hard = (pred_alpha > 0.5).astype(np.uint8)

            hard_label = (_read_gray(label_root / label_json["annotations"]["hard_foreground"]["path"]) > 127).astype(np.uint8)
            alpha_label = _read_gray(label_root / label_json["annotations"]["soft_alpha"]["path"]).astype(np.float32) / 255.0
            trimap_unknown = (
                _read_gray(label_root / label_json["annotations"]["trimap_unknown"]["path"]) > 127
            ).astype(np.float32)
            boundary_label = (_read_gray(label_root / label_json["annotations"]["boundary_mask"]["path"]) > 127).astype(np.uint8)
            hair_label = (_read_gray(label_root / label_json["annotations"]["hair_edge_mask"]["path"]) > 127).astype(np.uint8)

            pred_boundary = cv2.Canny((pred_alpha * 255.0).astype(np.uint8), 32, 96) > 0
            hair_pred = np.logical_and(pred_boundary, hair_label > 0).astype(np.uint8)
            hair_gt = np.logical_and(boundary_label > 0, hair_label > 0).astype(np.uint8)

            boundary_f1 = _boundary_f1(pred_hard, hard_label)
            trimap_error = float((np.abs(pred_alpha - alpha_label) * trimap_unknown).sum() / max(trimap_unknown.sum(), 1.0))
            alpha_mae = float(np.abs(pred_alpha - alpha_label).mean())
            alpha_sad = _sad(pred_alpha, alpha_label)
            hair_boundary_f1 = _boundary_f1(hair_pred, hair_gt)

            boundary_scores.append(boundary_f1)
            trimap_errors.append(trimap_error)
            alpha_mae_values.append(alpha_mae)
            alpha_sad_values.append(alpha_sad)
            hair_scores.append(hair_boundary_f1)
            runtime_values.append(float(output["runtime_sec"]))
            records.append(
                {
                    "case_id": case["case_id"],
                    "preprocess_frame_index": preprocess_idx,
                    "boundary_f1": boundary_f1,
                    "trimap_error": trimap_error,
                    "alpha_mae": alpha_mae,
                    "alpha_sad": alpha_sad,
                    "hair_boundary_f1": hair_boundary_f1,
                    "runtime_sec": float(output["runtime_sec"]),
                }
            )

    return {
        "model_id": model_id,
        "dataset_dir": str(dataset_dir.resolve()),
        "background_preprocess_dir": str(background_preprocess_dir.resolve()),
        "registry_path": str(registry_path.resolve()),
        "label_count": len(records),
        "boundary_f1_mean": float(np.mean(boundary_scores)) if boundary_scores else None,
        "trimap_error_mean": float(np.mean(trimap_errors)) if trimap_errors else None,
        "alpha_mae_mean": float(np.mean(alpha_mae_values)) if alpha_mae_values else None,
        "alpha_sad_mean": float(np.mean(alpha_sad_values)) if alpha_sad_values else None,
        "hair_boundary_f1_mean": float(np.mean(hair_scores)) if hair_scores else None,
        "runtime_sec_mean": float(np.mean(runtime_values)) if runtime_values else None,
        "records": records,
    }


def _load_initial_mask_volume(preprocess_dir: Path, metadata: dict) -> np.ndarray:
    hard_foreground = metadata["src_files"].get("hard_foreground")
    if hard_foreground is not None:
        return load_mask_artifact(
            preprocess_dir / hard_foreground["path"],
            hard_foreground.get("format"),
        )
    person_mask_artifact = metadata["src_files"].get("person_mask")
    if person_mask_artifact is None:
        raise RuntimeError("Sequence candidate requires hard_foreground or person_mask artifact")
    return load_person_mask_artifact(
        preprocess_dir / person_mask_artifact["path"],
        person_mask_artifact.get("format"),
    )


def _load_video_frames(video_path: str, frame_indices: list[int], target_shape: tuple[int, int]) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    wanted = set(frame_indices)
    frames = {}
    current_index = 0
    while wanted:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if current_index in wanted:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_AREA)
            frames[current_index] = rgb
            wanted.remove(current_index)
        current_index += 1
    cap.release()
    missing = [index for index in frame_indices if index not in frames]
    if missing:
        raise RuntimeError(f"Missing video frames for indices: {missing[:5]}")
    return [frames[index] for index in frame_indices]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--background_preprocess_dir", required=True)
    parser.add_argument("--registry_path", required=True)
    parser.add_argument("--model_ids", nargs="+", required=True)
    parser.add_argument("--suite_dir", required=True)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    background_preprocess_dir = Path(args.background_preprocess_dir)
    registry_path = Path(args.registry_path)
    suite_dir = Path(args.suite_dir)
    suite_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for model_id in args.model_ids:
        result = _run_candidate(model_id, dataset_dir, background_preprocess_dir, registry_path)
        _write_json(suite_dir / f"{model_id}_metrics.json", result)
        results.append(
            {
                "model_id": model_id,
                "alpha_mae_mean": result["alpha_mae_mean"],
                "trimap_error_mean": result["trimap_error_mean"],
                "hair_boundary_f1_mean": result["hair_boundary_f1_mean"],
                "runtime_sec_mean": result["runtime_sec_mean"],
            }
        )

    manifest = {
        "dataset_dir": str(dataset_dir.resolve()),
        "background_preprocess_dir": str(background_preprocess_dir.resolve()),
        "registry_path": str(registry_path.resolve()),
        "model_ids": list(args.model_ids),
        "results": results,
    }
    _write_json(suite_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
