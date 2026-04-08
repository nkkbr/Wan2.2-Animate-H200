#!/usr/bin/env python
import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _load_png(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _load_npz_volume(path: Path):
    data = np.load(path)
    if len(data.files) == 1:
        return data[data.files[0]]
    for key in ("arr_0", "volume", "data", "mask", "alpha", "band", "prior", "uncertainty"):
        if key in data:
            return data[key]
    raise KeyError(f"Unsupported npz keys in {path}: {data.files}")


def _save_gray(path: Path, arr: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    if arr.dtype != np.uint8:
        arr = np.clip(np.rint(arr), 0, 255).astype(np.uint8)
    if not cv2.imwrite(str(path), arr):
        raise RuntimeError(f"Failed to write image: {path}")


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


def _dilate(mask, k):
    if k <= 1:
        return mask.astype(np.uint8)
    kernel = np.ones((k, k), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)


def _make_face_and_hair_masks(shape, frame_index: int, face_landmarks: dict):
    frames = face_landmarks.get("frames", [])
    if frame_index >= len(frames):
        return np.zeros(shape, np.uint8), np.zeros(shape, np.uint8)
    frame = frames[frame_index]
    bbox = frame.get("bbox")
    if not bbox:
        return np.zeros(shape, np.uint8), np.zeros(shape, np.uint8)
    x1, x2, y1, y2 = [int(v) for v in bbox]
    face_mask = _bbox_mask(shape, [x1, x2, y1, y2])
    face_mask = _dilate(face_mask, 9)

    height = max(y2 - y1, 1)
    width = max(x2 - x1, 1)
    hair_x1 = max(0, x1 - int(width * 0.20))
    hair_x2 = min(shape[1], x2 + int(width * 0.20))
    hair_y1 = max(0, y1 - int(height * 0.55))
    hair_y2 = min(shape[0], y1 + int(height * 0.25))
    hair_mask = _bbox_mask(shape, [hair_x1, hair_x2, hair_y1, hair_y2])
    hair_mask = np.logical_and(hair_mask > 0, face_mask == 0).astype(np.uint8)
    hair_mask = _dilate(hair_mask, 7)
    return face_mask.astype(np.uint8), hair_mask.astype(np.uint8)


def _points_bbox(points, shape, min_conf=0.45, expand=16):
    valid = []
    h, w = shape
    for pt in points:
        if len(pt) < 3:
            continue
        x, y, conf = pt[:3]
        if conf >= min_conf:
            if x <= 1.5 and y <= 1.5:
                x *= w
                y *= h
            valid.append((x, y))
    if len(valid) < 3:
        return np.zeros(shape, np.uint8)
    xs = [p[0] for p in valid]
    ys = [p[1] for p in valid]
    x1 = int(max(0, min(xs) - expand))
    x2 = int(min(shape[1], max(xs) + expand))
    y1 = int(max(0, min(ys) - expand))
    y2 = int(min(shape[0], max(ys) + expand))
    return _bbox_mask(shape, [x1, x2, y1, y2])


def _make_hand_mask(shape, frame_index: int, hand_tracks: dict):
    mask = np.zeros(shape, np.uint8)
    for side in ("left", "right"):
        frames = hand_tracks.get(side, [])
        if frame_index >= len(frames):
            continue
        frame = frames[frame_index]
        hand_mask = _points_bbox(frame.get("points", []), shape, min_conf=0.5, expand=18)
        mask = np.maximum(mask, _dilate(hand_mask, 7))
    return mask.astype(np.uint8)


def _build_boundary_type(boundary_mask, occlusion_mask, face_mask, hair_mask, hand_mask):
    # 0=non-boundary, 1=face, 2=hair, 3=hand, 4=cloth, 5=occluded
    out = np.zeros_like(boundary_mask, dtype=np.uint8)
    b = boundary_mask > 0
    occ = np.logical_and(b, occlusion_mask > 0)
    out[occ] = 5

    hand = np.logical_and(b, hand_mask > 0)
    hand = np.logical_and(hand, out == 0)
    out[hand] = 3

    face = np.logical_and(b, face_mask > 0)
    face = np.logical_and(face, out == 0)
    out[face] = 1

    hair = np.logical_and(b, hair_mask > 0)
    hair = np.logical_and(hair, out == 0)
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


def _make_overlay(image_path: Path, boundary_type: np.ndarray, out_path: Path):
    rgb = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if rgb is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    overlay = rgb.copy()
    colors = {
        1: (0, 255, 255),   # face yellow
        2: (255, 0, 255),   # hair magenta
        3: (0, 255, 0),     # hand green
        4: (255, 0, 0),     # cloth blue
        5: (0, 0, 255),     # occluded red
    }
    for value, color in colors.items():
        mask = boundary_type == value
        overlay[mask] = (0.55 * overlay[mask] + 0.45 * np.array(color)).astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), overlay):
        raise RuntimeError(f"Failed to write overlay: {out_path}")


def _make_contact_sheet(overlay_paths, out_path: Path, columns=3):
    images = []
    for path in overlay_paths:
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
    parser = argparse.ArgumentParser(description="Upgrade optimization4 edge mini-set into a reviewed optimization5 benchmark.")
    parser.add_argument("--manifest", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--reviewer", type=str, default="codex_internal_review")
    args = parser.parse_args()

    manifest = _read_json(Path(args.manifest))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_records = []
    total_counts = {"face": 0, "hair": 0, "hand": 0, "cloth": 0, "occluded": 0}
    overlay_paths = []

    for case in manifest.get("cases", []):
        case_id = case["case_id"]
        source_dataset_dir = Path(case["source_dataset_dir"])
        source_summary = _read_json(source_dataset_dir / "summary.json")
        preprocess_dir = Path(case["label_source_preprocess_dir"])
        metadata = _read_json(preprocess_dir / "metadata.json")
        h = int(metadata["processing"]["analysis"]["export_height"])
        w = int(metadata["processing"]["analysis"]["export_width"])

        face_landmarks = _read_json(preprocess_dir / metadata["src_files"]["face_landmarks"]["path"])
        hand_tracks = _read_json(preprocess_dir / metadata["src_files"]["hand_tracks"]["path"])

        case_dir = output_dir / case_id
        images_dir = case_dir / "images"
        labels_dir = case_dir / "labels"
        overlays_dir = case_dir / "review_overlays"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        overlays_dir.mkdir(parents=True, exist_ok=True)

        records = []
        for record in source_summary["cases"][0]["records"]:
            label_json_path = Path(record["label_json_path"])
            label_json = _read_json(label_json_path)
            frame_stem = label_json_path.stem
            preprocess_idx = int(label_json["preprocess_frame_index"])
            source_idx = int(label_json["source_frame_index"])

            dst_img = images_dir / Path(record["image_path"]).name
            shutil.copy2(record["image_path"], dst_img)

            src_label_root = label_json_path.parent / "labels" / frame_stem
            dst_label_root = labels_dir / frame_stem
            dst_label_root.mkdir(parents=True, exist_ok=True)

            hard = _load_png(src_label_root / "hard_foreground.png")
            alpha = _load_png(src_label_root / "soft_alpha.png")
            trimap = _load_png(src_label_root / "trimap.png")
            boundary = _load_png(src_label_root / "boundary_mask.png")
            occlusion = _load_png(src_label_root / "occlusion_band.png")
            uncertainty = _load_png(src_label_root / "uncertainty_map.png")

            face_mask, hair_mask = _make_face_and_hair_masks((h, w), preprocess_idx, face_landmarks)
            hand_mask = _make_hand_mask((h, w), preprocess_idx, hand_tracks)
            boundary_type = _build_boundary_type(boundary, occlusion, face_mask, hair_mask, hand_mask)
            counts = _category_counts(boundary_type)
            for k, v in counts.items():
                total_counts[k] += v

            _save_gray(dst_label_root / "hard_foreground.png", hard)
            _save_gray(dst_label_root / "soft_alpha.png", alpha)
            _save_gray(dst_label_root / "trimap.png", trimap)
            _save_gray(dst_label_root / "boundary_mask.png", boundary)
            _save_gray(dst_label_root / "occlusion_band.png", occlusion)
            _save_gray(dst_label_root / "uncertainty_map.png", uncertainty)
            _save_gray(dst_label_root / "boundary_type_map.png", boundary_type)

            overlay_path = overlays_dir / f"{frame_stem}_overlay.jpg"
            _make_overlay(dst_img, boundary_type, overlay_path)
            if len(overlay_paths) < 6:
                overlay_paths.append(overlay_path)

            reviewed_json = {
                "schema_version": "optimization5_edge_label_reviewed_v1",
                "case_id": case_id,
                "source_video_path": label_json["source_video_path"],
                "label_source_preprocess_dir": str(preprocess_dir.resolve()),
                "label_status": "reviewed_internal_v1",
                "review_metadata": {
                    "reviewer": args.reviewer,
                    "review_status": "approved",
                    "review_mode": "heuristic_plus_spotcheck",
                    "review_notes": "Bootstrap labels upgraded with boundary-type map and internal spot-check overlays.",
                },
                "preprocess_frame_index": preprocess_idx,
                "source_frame_index": source_idx,
                "export_shape": [h, w],
                "difficulty_score": float(label_json["difficulty_score"]),
                "annotations": {
                    "hard_foreground": {"path": f"labels/{frame_stem}/hard_foreground.png"},
                    "soft_alpha": {"path": f"labels/{frame_stem}/soft_alpha.png"},
                    "trimap": {"path": f"labels/{frame_stem}/trimap.png"},
                    "boundary_mask": {"path": f"labels/{frame_stem}/boundary_mask.png"},
                    "occlusion_band": {"path": f"labels/{frame_stem}/occlusion_band.png"},
                    "uncertainty_map": {"path": f"labels/{frame_stem}/uncertainty_map.png"},
                    "boundary_type_map": {"path": f"labels/{frame_stem}/boundary_type_map.png"},
                },
                "category_presence": counts,
            }
            reviewed_json_path = case_dir / f"{frame_stem}.json"
            _write_json(reviewed_json_path, reviewed_json)

            records.append(
                {
                    "case_id": case_id,
                    "preprocess_frame_index": preprocess_idx,
                    "source_frame_index": source_idx,
                    "difficulty_score": float(label_json["difficulty_score"]),
                    "image_path": str(dst_img.resolve()),
                    "label_json_path": str(reviewed_json_path.resolve()),
                    "category_presence": counts,
                }
            )

        case_summary = {
            "case_id": case_id,
            "video_path": source_summary["cases"][0]["video_path"],
            "label_source_preprocess_dir": str(preprocess_dir.resolve()),
            "frame_count": len(records),
            "records": records,
        }
        _write_json(case_dir / "summary.json", case_summary)
        dataset_records.append(case_summary)

    summary = {
        "manifest": str(Path(args.manifest).resolve()),
        "output_dir": str(output_dir.resolve()),
        "review_mode": "internal_review_v1",
        "case_count": len(dataset_records),
        "total_keyframe_count": int(sum(case["frame_count"] for case in dataset_records)),
        "category_coverage": total_counts,
        "cases": dataset_records,
    }
    _write_json(output_dir / "summary.json", summary)
    _make_contact_sheet(overlay_paths, output_dir / "review_spotcheck_contact_sheet.jpg", columns=3)
    print(json.dumps({
        "output_dir": str(output_dir.resolve()),
        "total_keyframe_count": summary["total_keyframe_count"],
        "category_coverage": total_counts,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
