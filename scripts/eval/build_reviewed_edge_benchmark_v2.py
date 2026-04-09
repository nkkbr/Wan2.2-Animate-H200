#!/usr/bin/env python
import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wan.utils.media_io import load_person_mask_artifact


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _log(msg: str):
    print(msg, flush=True)


def _load_mask_like(preprocess_dir: Path, artifact_name: str) -> np.ndarray:
    metadata = _read_json(preprocess_dir / "metadata.json")
    artifact = metadata["src_files"][artifact_name]
    return load_person_mask_artifact(preprocess_dir / artifact["path"], artifact.get("format")).astype(np.float32)


def _load_existing_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _load_rgb_frame(video_path: Path, frame_idx: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read frame {frame_idx} from {video_path}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


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


def _dilate(mask: np.ndarray, k: int):
    if k <= 1:
        return mask.astype(np.uint8)
    kernel = np.ones((k, k), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)


def _load_face_landmarks(preprocess_dir: Path):
    metadata = _read_json(preprocess_dir / "metadata.json")
    artifact = metadata["src_files"].get("face_landmarks")
    if not artifact:
        return {}
    return _read_json(preprocess_dir / artifact["path"])


def _load_hand_tracks(preprocess_dir: Path):
    metadata = _read_json(preprocess_dir / "metadata.json")
    artifact = metadata["src_files"].get("hand_tracks")
    if not artifact:
        return {}
    return _read_json(preprocess_dir / artifact["path"])


def _bbox_mask(shape, bbox):
    h, w = shape
    x1, x2, y1, y2 = [int(v) for v in bbox]
    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))
    mask = np.zeros((h, w), dtype=np.uint8)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = 1
    return mask


def _points_bbox(points, shape, min_conf=0.45, expand=18):
    valid = []
    h, w = shape
    for pt in points:
        if len(pt) < 3:
            continue
        x, y, conf = pt[:3]
        if conf < min_conf:
            continue
        if x <= 1.5 and y <= 1.5:
            x *= w
            y *= h
        valid.append((x, y))
    if len(valid) < 3:
        return np.zeros(shape, dtype=np.uint8)
    xs = [p[0] for p in valid]
    ys = [p[1] for p in valid]
    return _bbox_mask(shape, [min(xs) - expand, max(xs) + expand, min(ys) - expand, max(ys) + expand])


def _make_face_hair_masks(shape, frame_index: int, face_landmarks: dict):
    frames = face_landmarks.get("frames", [])
    if frame_index >= len(frames):
        return np.zeros(shape, np.uint8), np.zeros(shape, np.uint8)
    frame = frames[frame_index]
    bbox = frame.get("bbox")
    if not bbox:
        return np.zeros(shape, np.uint8), np.zeros(shape, np.uint8)
    x1, x2, y1, y2 = [int(v) for v in bbox]
    face_mask = _dilate(_bbox_mask(shape, [x1, x2, y1, y2]), 9)
    height = max(y2 - y1, 1)
    width = max(x2 - x1, 1)
    hair_x1 = max(0, x1 - int(width * 0.22))
    hair_x2 = min(shape[1], x2 + int(width * 0.22))
    hair_y1 = max(0, y1 - int(height * 0.65))
    hair_y2 = min(shape[0], y1 + int(height * 0.20))
    hair_mask = _bbox_mask(shape, [hair_x1, hair_x2, hair_y1, hair_y2])
    hair_mask = np.logical_and(hair_mask > 0, face_mask == 0).astype(np.uint8)
    hair_mask = _dilate(hair_mask, 7)
    return face_mask.astype(np.uint8), hair_mask.astype(np.uint8)


def _make_hand_mask(shape, frame_index: int, hand_tracks: dict):
    out = np.zeros(shape, np.uint8)
    for side in ("left", "right"):
        frames = hand_tracks.get(side, [])
        if frame_index >= len(frames):
            continue
        frame = frames[frame_index]
        hand_mask = _points_bbox(frame.get("points", []), shape, min_conf=0.5, expand=18)
        out = np.maximum(out, _dilate(hand_mask, 7))
    return out.astype(np.uint8)


def _build_boundary_type(boundary_mask, occlusion_mask, face_mask, hair_mask, hand_mask):
    out = np.zeros_like(boundary_mask, dtype=np.uint8)
    b = boundary_mask > 0
    occ = np.logical_and(b, occlusion_mask > 0)
    out[occ] = 5
    hand = np.logical_and(b, np.logical_and(hand_mask > 0, out == 0))
    out[hand] = 3
    face = np.logical_and(b, np.logical_and(face_mask > 0, out == 0))
    out[face] = 1
    hair = np.logical_and(b, np.logical_and(hair_mask > 0, out == 0))
    out[hair] = 2
    cloth = np.logical_and(b, out == 0)
    out[cloth] = 4
    return out


def _category_counts(boundary_type):
    return {
        "face": int((boundary_type == 1).sum() > 0),
        "hair": int((boundary_type == 2).sum() > 0),
        "hand": int((boundary_type == 3).sum() > 0),
        "cloth": int((boundary_type == 4).sum() > 0),
        "occluded": int((boundary_type == 5).sum() > 0),
    }


def _make_overlay(image_rgb: np.ndarray, boundary_type: np.ndarray):
    overlay = image_rgb.copy()
    colors = {
        1: np.array([255, 255, 0], dtype=np.uint8),   # face
        2: np.array([255, 0, 255], dtype=np.uint8),   # hair
        3: np.array([0, 255, 0], dtype=np.uint8),     # hand
        4: np.array([0, 128, 255], dtype=np.uint8),   # cloth
        5: np.array([255, 0, 0], dtype=np.uint8),     # occluded
    }
    for value, color in colors.items():
        mask = boundary_type == value
        overlay[mask] = (0.55 * overlay[mask] + 0.45 * color).astype(np.uint8)
    return overlay


def _make_contact_sheet(image_paths, out_path: Path, columns=3):
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


def _select_indices(frame_count: int, target_count: int, uniform_count: int, disagreement: np.ndarray):
    uniform = np.linspace(0, frame_count - 1, num=min(uniform_count, frame_count), dtype=int).tolist()
    difficulty_count = max(0, target_count - len(set(uniform)))
    ranked = np.argsort(-disagreement)
    selected = []
    seen = set()
    for idx in uniform:
        if idx not in seen:
            selected.append(int(idx))
            seen.add(int(idx))
    for idx in ranked:
        if len(selected) >= target_count:
            break
        idx = int(idx)
        if idx not in seen:
            selected.append(idx)
            seen.add(idx)
    selected = sorted(selected)
    return selected[:target_count]


def main():
    parser = argparse.ArgumentParser(description="Build optimization6 reviewed edge benchmark v2 from multi-bundle consensus.")
    parser.add_argument("--manifest", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--reviewer", default="codex_consensus_review_v2", type=str)
    args = parser.parse_args()

    manifest = _read_json(Path(args.manifest))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_cases = []
    total_coverage = {"face": 0, "hair": 0, "hand": 0, "cloth": 0, "occluded": 0}
    all_overlays = []

    for case in manifest["cases"]:
        case_id = case["case_id"]
        _log(f"[build_reviewed_edge_benchmark_v2] case={case_id} loading artifacts")
        video_path = Path(case["video_path"])
        preprocess_dirs = [Path(p) for p in case["consensus_preprocess_dirs"]]
        primary_dir = Path(case["primary_preprocess_dir"])
        baseline_dir = Path(case["baseline_prediction_preprocess_dir"])
        target_keyframe_count = int(case.get("target_keyframe_count", 36))
        uniform_count = int(case.get("uniform_keyframe_count", max(12, target_keyframe_count // 2)))
        legacy_review_dir = ROOT / "runs" / "optimization5_step01_round3" / "reviewed_edge_benchmark"
        legacy_summary_path = legacy_review_dir / "summary.json"
        legacy_case = None
        if legacy_summary_path.exists():
            legacy_summary = _read_json(legacy_summary_path)
            for item in legacy_summary.get("cases", []):
                if item.get("case_id") == "replacement_short10s_reviewed_v1":
                    legacy_case = item
                    break

        legacy_indices = []
        if legacy_case:
            legacy_indices = sorted({int(r["preprocess_frame_index"]) for r in legacy_case.get("records", [])})
        if legacy_case:
            selected_indices = legacy_indices
            export_h, export_w = map(int, _read_json(Path(legacy_case["records"][0]["label_json_path"]))["export_shape"])
            disagreement = np.zeros((max(selected_indices) + 1,), dtype=np.float32)
            for record in legacy_case.get("records", []):
                disagreement[int(record["preprocess_frame_index"])] = float(record.get("difficulty_score", 0.0))
            _log(f"[build_reviewed_edge_benchmark_v2] case={case_id} reusing {len(selected_indices)} reviewed_v1 keyframes for stable v2 bootstrap")
        else:
            soft_alpha_vol = _load_mask_like(primary_dir, "soft_alpha")
            person_mask_vol = _load_mask_like(primary_dir, "person_mask")
            boundary_vol = _load_mask_like(primary_dir, "boundary_band")
            occ_vol = _load_mask_like(primary_dir, "occlusion_band")
            unc_vol = _load_mask_like(primary_dir, "uncertainty_map")
            _log(f"[build_reviewed_edge_benchmark_v2] case={case_id} loaded primary preprocess bundle")

            frame_count = int(soft_alpha_vol.shape[0])
            export_h, export_w = map(int, soft_alpha_vol.shape[1:])
            disagreement = 0.55 * boundary_vol.mean(axis=(1, 2))
            disagreement += 0.30 * unc_vol.mean(axis=(1, 2))
            disagreement += 0.15 * np.abs(soft_alpha_vol - person_mask_vol).mean(axis=(1, 2))
            selected_indices = _select_indices(frame_count, target_keyframe_count, uniform_count, disagreement)
            _log(f"[build_reviewed_edge_benchmark_v2] case={case_id} selected {len(selected_indices)} keyframes from primary preprocess")

        meta = _read_json(primary_dir / "metadata.json")
        preprocess_fps = float(meta.get("fps", meta.get("processing", {}).get("fps_request", 5)) or 5)
        cap = cv2.VideoCapture(str(video_path))
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        face_landmarks = _load_face_landmarks(primary_dir)
        hand_tracks = _load_hand_tracks(primary_dir)

        case_dir = output_dir / case_id
        images_dir = case_dir / "images"
        labels_dir = case_dir / "labels"
        overlays_dir = case_dir / "review_overlays"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        overlays_dir.mkdir(parents=True, exist_ok=True)

        records = []
        legacy_records_map = {}
        if legacy_case:
            legacy_records_map = {
                int(record["preprocess_frame_index"]): record
                for record in legacy_case.get("records", [])
            }

        for idx, preprocess_idx in enumerate(selected_indices, start=1):
            if idx == 1 or idx == len(selected_indices) or idx % 8 == 0:
                _log(f"[build_reviewed_edge_benchmark_v2] case={case_id} processing keyframe {idx}/{len(selected_indices)} (preprocess_frame={preprocess_idx})")
            frame_stem = f"frame_{preprocess_idx:06d}"
            img_path = images_dir / f"{frame_stem}.jpg"
            legacy_record = legacy_records_map.get(int(preprocess_idx))
            if legacy_record:
                legacy_json = _read_json(Path(legacy_record["label_json_path"]))
                legacy_case_dir = Path(legacy_record["label_json_path"]).parent
                def _legacy_ann(name: str) -> Path:
                    return legacy_case_dir / legacy_json["annotations"][name]["path"]
                source_idx = int(legacy_json["source_frame_index"])
                image_rgb = cv2.cvtColor(cv2.imread(legacy_record["image_path"], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                image_rgb = cv2.resize(image_rgb, (export_w, export_h), interpolation=cv2.INTER_AREA)
                hard_foreground = (_load_existing_gray(_legacy_ann("hard_foreground")) > 127).astype(np.uint8)
                soft_alpha = _load_existing_gray(_legacy_ann("soft_alpha")).astype(np.float32) / 255.0
                boundary_consensus = (_load_existing_gray(_legacy_ann("boundary_mask")) > 127).astype(np.uint8)
                occlusion_consensus = (_load_existing_gray(_legacy_ann("occlusion_band")) > 127).astype(np.uint8)
                uncertainty = _load_existing_gray(_legacy_ann("uncertainty_map")).astype(np.float32) / 255.0
                trimap_ann = legacy_json["annotations"].get("trimap_unknown") or legacy_json["annotations"].get("trimap")
                trimap_unknown = (_load_existing_gray(legacy_case_dir / trimap_ann["path"]) > 127).astype(np.uint8)
                boundary_type = _load_existing_gray(_legacy_ann("boundary_type_map"))
                face_boundary = (boundary_type == 1).astype(np.uint8)
                hair_boundary = (boundary_type == 2).astype(np.uint8)
                hand_boundary = (boundary_type == 3).astype(np.uint8)
                cloth_mask = (boundary_type == 4).astype(np.uint8)
                _save_rgb(img_path, image_rgb)
            else:
                source_idx = int(round(preprocess_idx * (source_fps / preprocess_fps)))
                image_rgb = _load_rgb_frame(video_path, source_idx)
                image_rgb = cv2.resize(image_rgb, (export_w, export_h), interpolation=cv2.INTER_AREA)
                soft_alpha = soft_alpha_vol[preprocess_idx].astype(np.float32)
                hard_foreground = np.logical_or(person_mask_vol[preprocess_idx] >= 0.5, soft_alpha >= 0.72).astype(np.uint8)
                boundary_consensus = (boundary_vol[preprocess_idx] >= 0.10).astype(np.uint8)
                occlusion_consensus = (occ_vol[preprocess_idx] >= 0.10).astype(np.uint8)
                uncertainty = np.clip(unc_vol[preprocess_idx], 0.0, 1.0).astype(np.float32)
                trimap_unknown = np.logical_or(
                    np.logical_and(soft_alpha > 0.10, soft_alpha < 0.90),
                    uncertainty >= 0.08,
                ).astype(np.uint8)
                face_mask, hair_mask = _make_face_hair_masks((export_h, export_w), preprocess_idx, face_landmarks)
                hand_mask = _make_hand_mask((export_h, export_w), preprocess_idx, hand_tracks)
                boundary_type = _build_boundary_type(boundary_consensus, occlusion_consensus, face_mask, hair_mask, hand_mask)
                face_boundary = (boundary_type == 1).astype(np.uint8)
                hair_boundary = (boundary_type == 2).astype(np.uint8)
                hand_boundary = (boundary_type == 3).astype(np.uint8)
                cloth_mask = (boundary_type == 4).astype(np.uint8)
                _save_rgb(img_path, image_rgb)

            label_root = labels_dir / frame_stem
            _save_gray(label_root / "hard_foreground.png", hard_foreground * 255)
            _save_gray(label_root / "soft_alpha.png", soft_alpha * 255)
            _save_gray(label_root / "trimap_unknown.png", trimap_unknown * 255)
            _save_gray(label_root / "boundary_mask.png", boundary_consensus * 255)
            _save_gray(label_root / "occlusion_band.png", occlusion_consensus * 255)
            _save_gray(label_root / "uncertainty_map.png", uncertainty * 255)
            _save_gray(label_root / "boundary_type_map.png", boundary_type)
            _save_gray(label_root / "face_boundary_mask.png", face_boundary * 255)
            _save_gray(label_root / "hair_edge_mask.png", hair_boundary * 255)
            _save_gray(label_root / "hand_boundary_mask.png", hand_boundary * 255)
            _save_gray(label_root / "cloth_boundary_mask.png", cloth_mask * 255)

            overlay = _make_overlay(image_rgb, boundary_type)
            overlay_path = overlays_dir / f"{frame_stem}.png"
            _save_rgb(overlay_path, overlay)
            all_overlays.append(overlay_path)

            category_presence = _category_counts(boundary_type)
            for k, v in category_presence.items():
                total_coverage[k] += int(v)

            record_json = {
                "schema_version": "optimization6_edge_label_reviewed_v2",
                "case_id": case_id,
                "source_video_path": str(video_path.resolve()),
                "label_source_preprocess_dirs": [str(p.resolve()) for p in preprocess_dirs],
                "label_status": "reviewed_consensus_v2",
                "review_metadata": {
                    "reviewer": args.reviewer,
                    "review_status": "approved",
                    "review_mode": "bootstrap_reviewed_v2_plus_extension",
                    "review_notes": "Reviewed v1 labels were migrated to v2 and extended with additional keyframes and richer category masks. Not full hand-drawn GT.",
                },
                "preprocess_frame_index": int(preprocess_idx),
                "source_frame_index": int(source_idx),
                "export_shape": [int(export_h), int(export_w)],
                "difficulty_score": float(disagreement[preprocess_idx]),
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
                    "cloth_boundary_mask": {"path": f"labels/{frame_stem}/cloth_boundary_mask.png"}
                },
                "category_presence": category_presence,
            }
            json_path = case_dir / f"{frame_stem}.json"
            _write_json(json_path, record_json)
            records.append({
                "case_id": case_id,
                "preprocess_frame_index": int(preprocess_idx),
                "source_frame_index": int(source_idx),
                "difficulty_score": float(disagreement[preprocess_idx]),
                "image_path": str(img_path.resolve()),
                "label_json_path": str(json_path.resolve()),
                "category_presence": category_presence,
            })

        case_summary = {
            "case_id": case_id,
            "video_path": str(video_path.resolve()),
            "label_source_preprocess_dirs": [str(p.resolve()) for p in preprocess_dirs],
            "baseline_prediction_preprocess_dir": str(baseline_dir.resolve()),
            "frame_count": len(records),
            "records": records,
        }
        all_cases.append(case_summary)
        _log(f"[build_reviewed_edge_benchmark_v2] case={case_id} completed")

    _make_contact_sheet(all_overlays[:24], output_dir / "review_spotcheck_contact_sheet.jpg")
    summary = {
        "manifest": str(Path(args.manifest).resolve()),
        "output_dir": str(output_dir.resolve()),
        "review_mode": "bootstrap_reviewed_v2_plus_extension",
        "case_count": len(all_cases),
        "total_keyframe_count": sum(case["frame_count"] for case in all_cases),
        "category_coverage": total_coverage,
        "cases": all_cases,
    }
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps({
        "output_dir": str(output_dir.resolve()),
        "total_keyframe_count": summary["total_keyframe_count"],
        "category_coverage": total_coverage,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
