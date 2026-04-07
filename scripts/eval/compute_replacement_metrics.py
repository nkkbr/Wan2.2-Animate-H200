#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Compute basic replacement metrics.")
    parser.add_argument("--run_dir", type=str, default=None, help="Run directory with manifest.json.")
    parser.add_argument("--video_path", type=str, default=None, help="Generated video path.")
    parser.add_argument("--mask_path", type=str, default=None, help="Person mask path (.mp4/.npz/.npy).")
    parser.add_argument("--clip_len", type=int, default=None, help="Clip length used during generation.")
    parser.add_argument("--refert_num", type=int, default=None, help="Overlap frames used during generation.")
    parser.add_argument("--output_json", type=str, default=None, help="Output metrics json path.")
    return parser.parse_args()


def load_manifest(run_dir: Path) -> dict:
    manifest_path = run_dir / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def latest_stage(manifest: dict, stage: str) -> dict | None:
    stage_entries = manifest.get("stages", {}).get(stage, [])
    return stage_entries[-1] if stage_entries else None


def infer_paths_from_manifest(manifest: dict) -> tuple[str | None, str | None, int | None, int | None]:
    generate_entry = latest_stage(manifest, "generate")
    preprocess_entry = latest_stage(manifest, "preprocess")
    video_path = None
    mask_path = None
    clip_len = None
    refert_num = None
    if generate_entry:
        outputs = generate_entry.get("outputs", {})
        video_path = outputs.get("save_file") or outputs.get("output_video")
        args = generate_entry.get("args", {})
        clip_len = args.get("frame_num")
        refert_num = args.get("refert_num")
    if preprocess_entry:
        outputs = preprocess_entry.get("outputs", {})
        mask_path = outputs.get("src_mask") or outputs.get("mask_path")
    return video_path, mask_path, clip_len, refert_num


def load_video_frames(video_path: Path) -> tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames found in video: {video_path}")
    return np.stack(frames), float(fps)


def load_mask(mask_path: Path) -> np.ndarray:
    suffix = mask_path.suffix.lower()
    if suffix == ".npz":
        with np.load(mask_path) as payload:
            if "mask" in payload:
                mask = payload["mask"]
            else:
                keys = list(payload.keys())
                if not keys:
                    raise RuntimeError(f"No arrays found in {mask_path}")
                mask = payload[keys[0]]
    elif suffix == ".npy":
        mask = np.load(mask_path)
    else:
        cap = cv2.VideoCapture(str(mask_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open mask video: {mask_path}")
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame[:, :, 0].astype(np.float32) / 255.0)
        cap.release()
        if not frames:
            raise RuntimeError(f"No frames found in mask: {mask_path}")
        mask = np.stack(frames)
    if mask.ndim == 4:
        mask = mask[..., 0]
    return mask.astype(np.float32)


def compute_seam_positions(frame_count: int, clip_len: int | None, refert_num: int | None) -> list[int]:
    if not clip_len or refert_num is None:
        return []
    step = clip_len - refert_num
    if step <= 0:
        return []
    positions = []
    seam = step
    while seam < frame_count:
        positions.append(seam)
        seam += step
    return positions


def frame_mad(frame_a: np.ndarray, frame_b: np.ndarray, mask: np.ndarray | None = None) -> float:
    diff = np.abs(frame_a.astype(np.float32) - frame_b.astype(np.float32))
    if mask is not None:
        mask = mask.astype(np.float32)[..., None]
        denom = float(mask.sum() * diff.shape[-1])
        if denom <= 0:
            return 0.0
        return float((diff * mask).sum() / denom)
    return float(diff.mean())


def summarize(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "max": None, "min": None}
    arr = np.asarray(values, dtype=np.float32)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "max": float(arr.max()),
        "min": float(arr.min()),
    }


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve() if args.run_dir else None
    if run_dir:
        manifest = load_manifest(run_dir)
        video_from_manifest, mask_from_manifest, clip_len_from_manifest, refert_from_manifest = infer_paths_from_manifest(manifest)
    else:
        manifest = None
        video_from_manifest = None
        mask_from_manifest = None
        clip_len_from_manifest = None
        refert_from_manifest = None

    video_path = Path(args.video_path or video_from_manifest).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    mask_path = Path(args.mask_path or mask_from_manifest).resolve() if (args.mask_path or mask_from_manifest) else None
    clip_len = args.clip_len if args.clip_len is not None else clip_len_from_manifest
    refert_num = args.refert_num if args.refert_num is not None else refert_from_manifest

    frames, fps = load_video_frames(video_path)
    frame_count, height, width = frames.shape[:3]
    seam_positions = compute_seam_positions(frame_count, clip_len, refert_num)
    seam_scores = [frame_mad(frames[idx - 1], frames[idx]) for idx in seam_positions if idx > 0]

    metrics = {
        "video_path": str(video_path),
        "frame_count": int(frame_count),
        "height": int(height),
        "width": int(width),
        "fps": fps,
        "duration_sec": float(frame_count / fps) if fps > 0 else None,
        "clip_len": clip_len,
        "refert_num": refert_num,
        "seam_positions": seam_positions,
        "seam_score": summarize(seam_scores),
    }

    if mask_path:
        mask = load_mask(mask_path)
        mask = mask[:frame_count]
        metrics["mask_path"] = str(mask_path)
        mask_area_curve = mask.reshape(mask.shape[0], -1).mean(axis=1).tolist()
        metrics["mask_area"] = summarize(mask_area_curve)
        bg_scores = []
        bg_keep = 1.0 - mask
        for idx in range(1, frame_count):
            bg_scores.append(frame_mad(frames[idx - 1], frames[idx], mask=bg_keep[idx]))
        metrics["background_fluctuation"] = summarize(bg_scores)
    else:
        metrics["mask_path"] = None
        metrics["mask_area"] = summarize([])
        metrics["background_fluctuation"] = summarize([])

    if args.output_json:
        output_path = Path(args.output_json).resolve()
    elif run_dir:
        metrics_dir = run_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        output_path = metrics_dir / "basic_metrics.json"
    else:
        output_path = Path.cwd() / "basic_metrics.json"

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")

    print(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()
