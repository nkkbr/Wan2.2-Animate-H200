#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wan.utils.media_io import load_mask_artifact, load_person_mask_artifact


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _load_gray(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _save_gray(path: Path, arr: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(np.rint(arr), 0, 255).astype(np.uint8)
    if not cv2.imwrite(str(path), arr):
        raise RuntimeError(f"Failed to write image: {path}")


def _save_rgb(path: Path, arr: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr):
        raise RuntimeError(f"Failed to write image: {path}")


def _load_rgb_frame(video_path: Path, frame_idx: int, size_wh: tuple[int, int]):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read frame {frame_idx} from {video_path}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return cv2.resize(frame, size_wh, interpolation=cv2.INTER_AREA)


def _load_optional_mask_volume(preprocess_dir: Path, metadata: dict, name: str):
    artifact = metadata["src_files"].get(name)
    if not artifact:
        return None
    loader = load_person_mask_artifact if name == "person_mask" else load_mask_artifact
    return loader(preprocess_dir / artifact["path"], artifact.get("format")).astype(np.float32)


def _align_volume(volume: np.ndarray | None, target_shape: tuple[int, int, int]) -> np.ndarray | None:
    if volume is None:
        return None
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


def _dilate(mask: np.ndarray, k: int):
    if k <= 1:
        return mask.astype(np.uint8)
    kernel = np.ones((k, k), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)


def _consensus_median(volumes: list[np.ndarray], frame_idx: int) -> np.ndarray:
    return np.median(np.stack([volume[frame_idx] for volume in volumes], axis=0), axis=0).astype(np.float32)


def _consensus_vote(volumes: list[np.ndarray], frame_idx: int, threshold: float = 0.5) -> np.ndarray:
    stacked = np.stack([(volume[frame_idx] >= threshold).astype(np.float32) for volume in volumes], axis=0)
    return (stacked.mean(axis=0) >= 0.5).astype(np.uint8)


def _boundary_from_alpha(alpha: np.ndarray) -> np.ndarray:
    img = np.clip(alpha * 255.0, 0.0, 255.0).astype(np.uint8)
    edge = cv2.Canny(img, 24, 72) > 0
    return _dilate(edge.astype(np.uint8), 5)


def _load_seed_record_map(seed_summary: dict):
    records = {}
    for case in seed_summary["cases"]:
        for record in case["records"]:
            records[int(record["preprocess_frame_index"])] = record
    return records


def _build_boundary_type(face_mask, hair_mask, hand_mask, cloth_mask, occlusion_mask, semi_mask):
    out = np.zeros_like(face_mask, dtype=np.uint8)
    occ = occlusion_mask > 0
    out[occ] = 5
    hand = np.logical_and(hand_mask > 0, out == 0)
    out[hand] = 3
    face = np.logical_and(face_mask > 0, out == 0)
    out[face] = 1
    hair = np.logical_and(hair_mask > 0, out == 0)
    out[hair] = 2
    cloth = np.logical_and(cloth_mask > 0, out == 0)
    out[cloth] = 4
    semi = np.logical_and(semi_mask > 0, out == 0)
    out[semi] = 6
    return out


def _make_overlay(image_rgb, boundary_type):
    overlay = image_rgb.copy()
    colors = {
        1: np.array([255, 255, 0], dtype=np.uint8),
        2: np.array([255, 0, 255], dtype=np.uint8),
        3: np.array([0, 255, 0], dtype=np.uint8),
        4: np.array([0, 128, 255], dtype=np.uint8),
        5: np.array([255, 0, 0], dtype=np.uint8),
        6: np.array([255, 255, 255], dtype=np.uint8),
    }
    for value, color in colors.items():
        mask = boundary_type == value
        overlay[mask] = (0.55 * overlay[mask] + 0.45 * color).astype(np.uint8)
    return overlay


def _make_contact_sheet(image_paths, out_path: Path, columns=4):
    images = []
    for path in image_paths:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is not None:
            images.append(img)
    if not images:
        return
    h = max(img.shape[0] for img in images)
    w = max(img.shape[1] for img in images)
    padded = []
    for img in images:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        canvas[: img.shape[0], : img.shape[1]] = img
        padded.append(canvas)
    rows = []
    for idx in range(0, len(padded), columns):
        chunk = padded[idx: idx + columns]
        while len(chunk) < columns:
            chunk.append(np.zeros((h, w, 3), dtype=np.uint8))
        rows.append(np.concatenate(chunk, axis=1))
    sheet = np.concatenate(rows, axis=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)


def main():
    parser = argparse.ArgumentParser(description="Build reviewed edge benchmark v3.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--selection_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--reviewer", default="codex_governed_v3")
    args = parser.parse_args()

    manifest = _read_json(Path(args.manifest))
    selection = _read_json(Path(args.selection_json))
    seed_dir = Path(manifest["seed_reviewed_dataset_dir"])
    seed_summary = _read_json(seed_dir / "summary.json")
    seed_records = _load_seed_record_map(seed_summary)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_coverage = {"face": 0, "hair": 0, "hand": 0, "cloth": 0, "occluded": 0, "semi_transparent": 0}
    split_coverage = {"seed_eval": 0, "expansion_eval": 0, "holdout_eval": 0}
    strong_revision_count = 0
    overlays = []
    all_cases = []

    for case_manifest, case_selection in zip(manifest["cases"], selection["cases"]):
        case_id = case_manifest["case_id"]
        video_path = Path(case_manifest["video_path"])
        preprocess_dirs = [Path(p) for p in case_manifest["consensus_preprocess_dirs"]]
        primary_dir = Path(case_manifest["primary_preprocess_dir"])
        baseline_dir = Path(case_manifest["baseline_prediction_preprocess_dir"])
        metas = [_read_json(path / "metadata.json") for path in preprocess_dirs]
        cap = cv2.VideoCapture(str(video_path))
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        preprocess_fps = float(_read_json(primary_dir / "metadata.json").get("fps", 5.0) or 5.0)

        alpha_volumes = []
        hard_volumes = []
        boundary_volumes = []
        occlusion_volumes = []
        uncertainty_volumes = []
        face_semantics = []
        hair_semantics = []
        hand_semantics = []
        cloth_semantics = []
        for preprocess_dir, meta in zip(preprocess_dirs, metas):
            alpha_v2 = _load_optional_mask_volume(preprocess_dir, meta, "alpha_v2")
            soft_alpha = _load_optional_mask_volume(preprocess_dir, meta, "soft_alpha")
            hard = _load_optional_mask_volume(preprocess_dir, meta, "hard_foreground")
            if hard is None:
                hard = _load_optional_mask_volume(preprocess_dir, meta, "person_mask")
            boundary = _load_optional_mask_volume(preprocess_dir, meta, "boundary_band")
            occlusion = _load_optional_mask_volume(preprocess_dir, meta, "occlusion_band")
            uncertainty = _load_optional_mask_volume(preprocess_dir, meta, "uncertainty_map")
            alpha_volumes.append(alpha_v2 if alpha_v2 is not None else soft_alpha)
            hard_volumes.append(hard)
            boundary_volumes.append(boundary)
            occlusion_volumes.append(occlusion if occlusion is not None else np.zeros_like(alpha_volumes[-1]))
            uncertainty_volumes.append(uncertainty if uncertainty is not None else np.zeros_like(alpha_volumes[-1]))
            face_semantics.append(_load_optional_mask_volume(preprocess_dir, meta, "face_boundary"))
            hair_semantics.append(_load_optional_mask_volume(preprocess_dir, meta, "hair_boundary"))
            hand_semantics.append(_load_optional_mask_volume(preprocess_dir, meta, "hand_boundary"))
            cloth_semantics.append(_load_optional_mask_volume(preprocess_dir, meta, "cloth_boundary"))

        target_shape = alpha_volumes[0].shape
        alpha_volumes = [_align_volume(volume, target_shape) for volume in alpha_volumes]
        hard_volumes = [_align_volume(volume, target_shape) for volume in hard_volumes]
        boundary_volumes = [_align_volume(volume, target_shape) for volume in boundary_volumes]
        occlusion_volumes = [_align_volume(volume, target_shape) for volume in occlusion_volumes]
        uncertainty_volumes = [_align_volume(volume, target_shape) for volume in uncertainty_volumes]
        face_semantics = [_align_volume(volume, target_shape) for volume in face_semantics]
        hair_semantics = [_align_volume(volume, target_shape) for volume in hair_semantics]
        hand_semantics = [_align_volume(volume, target_shape) for volume in hand_semantics]
        cloth_semantics = [_align_volume(volume, target_shape) for volume in cloth_semantics]

        export_h, export_w = alpha_volumes[0].shape[1:]
        case_dir = output_dir / case_id
        images_dir = case_dir / "images"
        labels_dir = case_dir / "labels"
        overlays_dir = case_dir / "review_overlays"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        overlays_dir.mkdir(parents=True, exist_ok=True)

        records = []
        seed_indices = set(case_selection["seed_indices"])
        expansion_indices = set(case_selection["expansion_indices"])
        holdout_candidates = list(sorted(expansion_indices))[-8:]
        for preprocess_idx in case_selection["selected_indices"]:
            source_idx = int(round(preprocess_idx * (source_fps / preprocess_fps)))
            frame_stem = f"frame_{preprocess_idx:06d}"
            label_root = labels_dir / frame_stem
            img_path = images_dir / f"{frame_stem}.jpg"

            if preprocess_idx in seed_records:
                seed_record = seed_records[preprocess_idx]
                seed_json = _read_json(Path(seed_record["label_json_path"]))
                seed_root = Path(seed_record["label_json_path"]).parent
                image_rgb = cv2.cvtColor(cv2.imread(seed_record["image_path"], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                image_rgb = cv2.resize(image_rgb, (export_w, export_h), interpolation=cv2.INTER_AREA)
                hard_foreground = (_load_gray(seed_root / seed_json["annotations"]["hard_foreground"]["path"]) > 127).astype(np.uint8)
                soft_alpha = _load_gray(seed_root / seed_json["annotations"]["soft_alpha"]["path"]).astype(np.float32) / 255.0
                trimap_unknown = (_load_gray(seed_root / seed_json["annotations"]["trimap_unknown"]["path"]) > 127).astype(np.uint8)
                boundary_mask = (_load_gray(seed_root / seed_json["annotations"]["boundary_mask"]["path"]) > 127).astype(np.uint8)
                occlusion_band = (_load_gray(seed_root / seed_json["annotations"]["occlusion_band"]["path"]) > 127).astype(np.uint8)
                uncertainty_map = _load_gray(seed_root / seed_json["annotations"]["uncertainty_map"]["path"]).astype(np.float32) / 255.0
                face_boundary = (_load_gray(seed_root / seed_json["annotations"]["face_boundary_mask"]["path"]) > 127).astype(np.uint8)
                hair_boundary = (_load_gray(seed_root / seed_json["annotations"]["hair_edge_mask"]["path"]) > 127).astype(np.uint8)
                hand_boundary = (_load_gray(seed_root / seed_json["annotations"]["hand_boundary_mask"]["path"]) > 127).astype(np.uint8)
                cloth_boundary = (_load_gray(seed_root / seed_json["annotations"]["cloth_boundary_mask"]["path"]) > 127).astype(np.uint8)
                semi_transparent = np.logical_and(boundary_mask > 0, trimap_unknown > 0).astype(np.uint8)
                revision_strength = "seed_migrated"
                split = "seed_eval"
                seed_source = "reviewed_v2_seed"
            else:
                image_rgb = _load_rgb_frame(video_path, source_idx, (export_w, export_h))
                soft_alpha = _consensus_median(alpha_volumes, preprocess_idx)
                hard_foreground = _consensus_vote(hard_volumes, preprocess_idx).astype(np.uint8)
                boundary_consensus = np.clip(np.mean(np.stack([vol[preprocess_idx] for vol in boundary_volumes], axis=0), axis=0), 0.0, 1.0)
                boundary_mask = np.logical_or(boundary_consensus >= 0.10, _boundary_from_alpha(soft_alpha) > 0).astype(np.uint8)
                occlusion_band = _consensus_vote(occlusion_volumes, preprocess_idx, threshold=0.10).astype(np.uint8)
                uncertainty_map = np.clip(
                    np.mean(np.stack([vol[preprocess_idx] for vol in uncertainty_volumes], axis=0), axis=0)
                    + np.std(np.stack([vol[preprocess_idx] for vol in alpha_volumes], axis=0), axis=0),
                    0.0,
                    1.0,
                ).astype(np.float32)
                trimap_unknown = np.logical_or(
                    np.logical_and(soft_alpha > 0.08, soft_alpha < 0.92),
                    uncertainty_map >= 0.10,
                ).astype(np.uint8)

                def _semantic_consensus(volumes):
                    usable = [volume for volume in volumes if volume is not None]
                    if not usable:
                        return np.zeros((export_h, export_w), dtype=np.uint8)
                    return _consensus_vote(usable, preprocess_idx, threshold=0.10).astype(np.uint8)

                face_boundary = _semantic_consensus(face_semantics)
                hair_boundary = _semantic_consensus(hair_semantics)
                hand_boundary = _semantic_consensus(hand_semantics)
                cloth_boundary = _semantic_consensus(cloth_semantics)
                if int(cloth_boundary.sum()) == 0:
                    cloth_boundary = np.logical_and(boundary_mask > 0, np.logical_and(face_boundary == 0, np.logical_and(hair_boundary == 0, hand_boundary == 0))).astype(np.uint8)
                semi_transparent = np.logical_and(boundary_mask > 0, trimap_unknown > 0).astype(np.uint8)
                revision_strength = "strong_bootstrap_extension"
                split = "holdout_eval" if preprocess_idx in holdout_candidates else "expansion_eval"
                seed_source = "multi_source_consensus_bootstrap"
                strong_revision_count += 1

            boundary_type = _build_boundary_type(face_boundary, hair_boundary, hand_boundary, cloth_boundary, occlusion_band, semi_transparent)
            overlay = _make_overlay(image_rgb, boundary_type)
            overlay_path = overlays_dir / f"{frame_stem}.png"
            _save_rgb(img_path, image_rgb)
            _save_rgb(overlay_path, overlay)
            overlays.append(overlay_path)

            _save_gray(label_root / "hard_foreground.png", hard_foreground * 255)
            _save_gray(label_root / "soft_alpha.png", soft_alpha * 255)
            _save_gray(label_root / "trimap_unknown.png", trimap_unknown * 255)
            _save_gray(label_root / "boundary_mask.png", boundary_mask * 255)
            _save_gray(label_root / "occlusion_band.png", occlusion_band * 255)
            _save_gray(label_root / "uncertainty_map.png", uncertainty_map * 255)
            _save_gray(label_root / "boundary_type_map.png", boundary_type)
            _save_gray(label_root / "face_boundary_mask.png", face_boundary * 255)
            _save_gray(label_root / "hair_edge_mask.png", hair_boundary * 255)
            _save_gray(label_root / "hand_boundary_mask.png", hand_boundary * 255)
            _save_gray(label_root / "cloth_boundary_mask.png", cloth_boundary * 255)
            _save_gray(label_root / "semi_transparent_boundary_mask.png", semi_transparent * 255)

            category_presence = {
                "face": int((face_boundary > 0).sum() > 0),
                "hair": int((hair_boundary > 0).sum() > 0),
                "hand": int((hand_boundary > 0).sum() > 0),
                "cloth": int((cloth_boundary > 0).sum() > 0),
                "occluded": int((occlusion_band > 0).sum() > 0),
                "semi_transparent": int((semi_transparent > 0).sum() > 0),
            }
            for key, value in category_presence.items():
                total_coverage[key] += int(value)
            split_coverage[split] += 1

            record_json = {
                "schema_version": "optimization8_edge_label_reviewed_v3",
                "case_id": case_id,
                "source_video_path": str(video_path.resolve()),
                "label_source_preprocess_dirs": [str(path.resolve()) for path in preprocess_dirs],
                "seed_reviewed_dataset_dir": str(seed_dir.resolve()),
                "label_status": "reviewed_consensus_v3_candidate",
                "review_metadata": {
                    "reviewer": args.reviewer,
                    "review_status": "governed_candidate",
                    "review_mode": "bootstrap_reviewed_v3_multi_source_governed",
                    "review_notes": "v3 candidate benchmark built from reviewed_v2 seeds plus multi-source consensus extension; not fully hand-redrawn GT.",
                    "review_tier": "v3_candidate",
                    "revision_strength": revision_strength,
                    "split": split,
                    "seed_source": seed_source,
                    "initial_auto_label_sources": [str(path.resolve()) for path in preprocess_dirs],
                },
                "preprocess_frame_index": int(preprocess_idx),
                "source_frame_index": int(source_idx),
                "export_shape": [int(export_h), int(export_w)],
                "difficulty_score": float(case_selection["disagreement"][case_selection["selected_indices"].index(preprocess_idx)]),
                "annotations": {
                    "hard_foreground": {"path": f"labels/{frame_stem}/hard_foreground.png"},
                    "soft_alpha": {"path": f"labels/{frame_stem}/soft_alpha.png"},
                    "trimap_unknown": {"path": f"labels/{frame_stem}/trimap_unknown.png"},
                    "boundary_mask": {"path": f"labels/{frame_stem}/boundary_mask.png"},
                    "occlusion_band": {"path": f"labels/{frame_stem}/occlusion_band.png"},
                    "uncertainty_map": {"path": f"labels/{frame_stem}/uncertainty_map.png"},
                    "boundary_type_map": {"path": f"labels/{frame_stem}/boundary_type_map.png"},
                    "face_boundary_mask": {"path": f"labels/{frame_stem}/face_boundary_mask.png"},
                    "hair_edge_mask": {"path": f"labels/{frame_stem}/hair_edge_mask.png"},
                    "hand_boundary_mask": {"path": f"labels/{frame_stem}/hand_boundary_mask.png"},
                    "cloth_boundary_mask": {"path": f"labels/{frame_stem}/cloth_boundary_mask.png"},
                    "semi_transparent_boundary_mask": {"path": f"labels/{frame_stem}/semi_transparent_boundary_mask.png"}
                },
                "category_presence": category_presence,
            }
            json_path = case_dir / f"{frame_stem}.json"
            _write_json(json_path, record_json)
            records.append(
                {
                    "case_id": case_id,
                    "preprocess_frame_index": int(preprocess_idx),
                    "source_frame_index": int(source_idx),
                    "difficulty_score": float(record_json["difficulty_score"]),
                    "image_path": str(img_path.resolve()),
                    "label_json_path": str(json_path.resolve()),
                    "category_presence": category_presence,
                }
            )

        case_summary = {
            "case_id": case_id,
            "video_path": str(video_path.resolve()),
            "label_source_preprocess_dirs": [str(path.resolve()) for path in preprocess_dirs],
            "baseline_prediction_preprocess_dir": str(baseline_dir.resolve()),
            "frame_count": len(records),
            "records": records,
        }
        all_cases.append(case_summary)

    _make_contact_sheet(overlays[:32], output_dir / "review_spotcheck_contact_sheet_v3.jpg")
    total_keyframe_count = sum(case["frame_count"] for case in all_cases)
    summary = {
        "manifest": str(Path(args.manifest).resolve()),
        "selection_json": str(Path(args.selection_json).resolve()),
        "output_dir": str(output_dir.resolve()),
        "review_mode": "bootstrap_reviewed_v3_multi_source_governed",
        "case_count": len(all_cases),
        "total_keyframe_count": total_keyframe_count,
        "category_coverage": total_coverage,
        "split_coverage": split_coverage,
        "strong_revision_count": strong_revision_count,
        "strong_revision_fraction": float(strong_revision_count / max(total_keyframe_count, 1)),
        "cases": all_cases,
    }
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps({"total_keyframe_count": total_keyframe_count, "category_coverage": total_coverage, "strong_revision_fraction": summary["strong_revision_fraction"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
