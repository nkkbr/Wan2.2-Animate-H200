#!/usr/bin/env python
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _get_frame_indices(frame_num, video_fps, clip_length, train_fps):
    start_frame = 0
    times = np.arange(0, clip_length) / train_fps
    frame_indices = start_frame + np.round(times * video_fps).astype(int)
    frame_indices = np.clip(frame_indices, 0, frame_num - 1)
    return frame_indices.tolist()


def _padding_resize(img_ori, height=512, width=512, padding_color=(0, 0, 0), interpolation=cv2.INTER_LINEAR):
    ori_height = img_ori.shape[0]
    ori_width = img_ori.shape[1]
    channel = img_ori.shape[2]

    img_pad = np.zeros((height, width, channel), dtype=np.uint8)
    if channel == 1:
        img_pad[:, :, 0] = padding_color[0]
    else:
        img_pad[:, :, 0] = padding_color[0]
        img_pad[:, :, 1] = padding_color[1]
        img_pad[:, :, 2] = padding_color[2]

    if (ori_height / ori_width) > (height / width):
        new_width = int(height / ori_height * ori_width)
        img = cv2.resize(img_ori, (new_width, height), interpolation=interpolation)
        padding = int((width - new_width) / 2)
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
        img_pad[:, padding: padding + new_width, :] = img
    else:
        new_height = int(width / ori_width * ori_height)
        img = cv2.resize(img_ori, (width, new_height), interpolation=interpolation)
        padding = int((height - new_height) / 2)
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
        img_pad[padding: padding + new_height, :, :] = img
    return np.uint8(img_pad)


def _load_npz_volume(path: Path):
    data = np.load(path)
    if len(data.files) == 1:
        return data[data.files[0]]
    for key in ("arr_0", "volume", "data", "mask", "alpha", "band", "prior", "uncertainty"):
        if key in data:
            return data[key]
    raise KeyError(f"Unsupported npz keys in {path}: {data.files}")


def _save_gray_png(path: Path, array: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(array)
    if arr.dtype != np.uint8:
        arr = np.clip(np.rint(arr), 0, 255).astype(np.uint8)
    if not cv2.imwrite(str(path), arr):
        raise RuntimeError(f"Failed to write image: {path}")


def _read_frame(video_path: Path, frame_index: int):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read frame {frame_index} from {video_path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _build_trimap(mask_uint8: np.ndarray, alpha_uint8: np.ndarray):
    fg = (alpha_uint8 >= 250).astype(np.uint8)
    bg = (alpha_uint8 <= 5).astype(np.uint8)
    unknown = ((fg == 0) & (bg == 0)).astype(np.uint8)
    trimap = np.zeros_like(mask_uint8, dtype=np.uint8)
    trimap[bg > 0] = 0
    trimap[unknown > 0] = 128
    trimap[fg > 0] = 255
    return trimap


def _select_keyframes(hard_foreground, boundary_band, uncertainty, occlusion_band, target_count: int):
    frame_count = int(hard_foreground.shape[0])
    even_count = max(target_count // 2, 1)
    difficulty_count = max(target_count - even_count, 0)

    even_indices = np.linspace(0, frame_count - 1, num=even_count, dtype=int).tolist()

    boundary_mean = boundary_band.reshape(frame_count, -1).mean(axis=1)
    uncertainty_mean = uncertainty.reshape(frame_count, -1).mean(axis=1)
    occlusion_mean = occlusion_band.reshape(frame_count, -1).mean(axis=1)
    alpha_transition = np.abs(boundary_band - hard_foreground).reshape(frame_count, -1).mean(axis=1)
    difficulty = 0.45 * boundary_mean + 0.30 * uncertainty_mean + 0.15 * occlusion_mean + 0.10 * alpha_transition
    ranked = np.argsort(-difficulty).tolist()

    selected = []
    for idx in even_indices + ranked:
        if idx not in selected:
            selected.append(int(idx))
        if len(selected) >= target_count:
            break
    return selected, difficulty.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Extract an optimization4 edge mini benchmark with bootstrap labels.")
    parser.add_argument("--manifest", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    args = parser.parse_args()

    manifest = _read_json(Path(args.manifest))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_records = []
    for case in manifest.get("cases", []):
        case_id = case["case_id"]
        case_dir = output_dir / case_id
        images_dir = case_dir / "images"
        labels_dir = case_dir / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        video_path = Path(case["video_path"])
        preprocess_dir = Path(case["label_source_preprocess_dir"])
        metadata = _read_json(preprocess_dir / "metadata.json")
        fps = float(metadata["fps"])
        frame_count = int(metadata["frame_count"])
        export_h = int(metadata["processing"]["analysis"]["export_height"])
        export_w = int(metadata["processing"]["analysis"]["export_width"])

        cap = cv2.VideoCapture(str(video_path))
        src_frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        src_fps = float(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        source_frame_indices = _get_frame_indices(src_frame_num, src_fps, frame_count, fps)

        hard_foreground = _load_npz_volume(preprocess_dir / metadata["src_files"]["person_mask"]["path"]).astype(np.float32)
        soft_alpha = _load_npz_volume(preprocess_dir / metadata["src_files"]["soft_alpha"]["path"]).astype(np.float32)
        boundary_band = _load_npz_volume(preprocess_dir / metadata["src_files"]["boundary_band"]["path"]).astype(np.float32)
        occlusion_band = _load_npz_volume(preprocess_dir / metadata["src_files"]["occlusion_band"]["path"]).astype(np.float32)
        uncertainty = _load_npz_volume(preprocess_dir / metadata["src_files"]["uncertainty_map"]["path"]).astype(np.float32)

        keyframes, difficulty = _select_keyframes(
            hard_foreground=hard_foreground,
            boundary_band=boundary_band,
            uncertainty=uncertainty,
            occlusion_band=occlusion_band,
            target_count=int(case.get("target_keyframe_count", 24)),
        )

        records = []
        for preprocess_idx in keyframes:
            source_idx = int(source_frame_indices[preprocess_idx])
            rgb = _read_frame(video_path, source_idx)
            rgb = _padding_resize(rgb, height=export_h, width=export_w, padding_color=(0, 0, 0))

            frame_stem = f"frame_{preprocess_idx:06d}"
            image_path = images_dir / f"{frame_stem}.jpg"
            image_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if not cv2.imwrite(str(image_path), image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
                raise RuntimeError(f"Failed to write image: {image_path}")

            hard = (hard_foreground[preprocess_idx] > 0.5).astype(np.uint8) * 255
            alpha = np.clip(np.rint(np.clip(soft_alpha[preprocess_idx], 0.0, 1.0) * 255.0), 0, 255).astype(np.uint8)
            boundary = (boundary_band[preprocess_idx] > 0.10).astype(np.uint8) * 255
            occlusion = (occlusion_band[preprocess_idx] > 0.10).astype(np.uint8) * 255
            uncertainty_png = np.clip(np.rint(np.clip(uncertainty[preprocess_idx], 0.0, 1.0) * 255.0), 0, 255).astype(np.uint8)
            trimap = _build_trimap(hard, alpha)

            label_frame_dir = labels_dir / frame_stem
            label_frame_dir.mkdir(parents=True, exist_ok=True)
            _save_gray_png(label_frame_dir / "hard_foreground.png", hard)
            _save_gray_png(label_frame_dir / "soft_alpha.png", alpha)
            _save_gray_png(label_frame_dir / "trimap.png", trimap)
            _save_gray_png(label_frame_dir / "boundary_mask.png", boundary)
            _save_gray_png(label_frame_dir / "occlusion_band.png", occlusion)
            _save_gray_png(label_frame_dir / "uncertainty_map.png", uncertainty_png)

            label_json = {
                "schema_version": "optimization4_edge_label_v1",
                "case_id": case_id,
                "source_video_path": str(video_path.resolve()),
                "label_source_preprocess_dir": str(preprocess_dir.resolve()),
                "label_status": case.get("label_status", "bootstrap_unreviewed"),
                "bootstrap_label_version": case.get("bootstrap_label_version", "edge_v1"),
                "preprocess_frame_index": int(preprocess_idx),
                "source_frame_index": source_idx,
                "export_shape": [export_h, export_w],
                "difficulty_score": float(difficulty[preprocess_idx]),
                "annotations": {
                    "hard_foreground": {"path": f"labels/{frame_stem}/hard_foreground.png"},
                    "soft_alpha": {"path": f"labels/{frame_stem}/soft_alpha.png"},
                    "trimap": {"path": f"labels/{frame_stem}/trimap.png"},
                    "boundary_mask": {"path": f"labels/{frame_stem}/boundary_mask.png"},
                    "occlusion_band": {"path": f"labels/{frame_stem}/occlusion_band.png"},
                    "uncertainty_map": {"path": f"labels/{frame_stem}/uncertainty_map.png"}
                }
            }
            label_json_path = case_dir / f"{frame_stem}.json"
            label_json_path.write_text(json.dumps(label_json, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

            records.append(
                {
                    "case_id": case_id,
                    "preprocess_frame_index": int(preprocess_idx),
                    "source_frame_index": source_idx,
                    "difficulty_score": float(difficulty[preprocess_idx]),
                    "image_path": str(image_path.resolve()),
                    "label_json_path": str(label_json_path.resolve()),
                }
            )

        case_summary = {
            "case_id": case_id,
            "video_path": str(video_path.resolve()),
            "label_source_preprocess_dir": str(preprocess_dir.resolve()),
            "frame_count": len(records),
            "records": records,
        }
        (case_dir / "summary.json").write_text(json.dumps(case_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        dataset_records.append(case_summary)

    dataset_summary = {
        "manifest": str(Path(args.manifest).resolve()),
        "output_dir": str(output_dir.resolve()),
        "case_count": len(dataset_records),
        "total_keyframe_count": int(sum(case["frame_count"] for case in dataset_records)),
        "cases": dataset_records,
    }
    (output_dir / "summary.json").write_text(json.dumps(dataset_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir.resolve()), "total_keyframe_count": dataset_summary["total_keyframe_count"]}))


if __name__ == "__main__":
    main()
