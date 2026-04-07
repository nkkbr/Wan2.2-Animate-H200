import logging
import os
from pathlib import Path

import cv2
import numpy as np

from .animate_contract import load_image_rgb, read_video_rgb, validate_person_mask_frames, validate_rgb_video


INTERMEDIATE_SAVE_FORMATS = ("mp4", "png_seq", "npz")
OUTPUT_VIDEO_FORMATS = ("auto", "mp4", "png_seq", "ffv1")


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def infer_artifact_format(path: str | Path) -> str:
    path = Path(path)
    if path.is_dir():
        return "png_seq"
    if path.suffix.lower() == ".npz":
        return "npz"
    return "mp4"


def _write_rgb_png(path: Path, frame_rgb: np.ndarray) -> None:
    _ensure_parent(path)
    ok = cv2.imwrite(str(path), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError(f"Failed to write PNG frame: {path}")


def _write_gray_png(path: Path, frame_gray: np.ndarray) -> None:
    _ensure_parent(path)
    ok = cv2.imwrite(str(path), frame_gray)
    if not ok:
        raise RuntimeError(f"Failed to write grayscale PNG frame: {path}")


def write_rgb_video_mp4(frames: np.ndarray, path: str | Path, fps: float) -> None:
    path = Path(path)
    _ensure_parent(path)
    try:
        import imageio

        writer = imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8)
        try:
            for frame in frames:
                writer.append_data(frame)
        finally:
            writer.close()
        return
    except ModuleNotFoundError:
        logging.warning("imageio is not available, falling back to cv2.VideoWriter for %s", path)

    height, width = frames.shape[1:3]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open cv2.VideoWriter for {path}")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def write_rgb_png_sequence(frames: np.ndarray, directory: str | Path) -> None:
    directory = Path(directory)
    _ensure_dir(directory)
    for index, frame in enumerate(frames):
        _write_rgb_png(directory / f"{index:06d}.png", frame)


def write_rgb_npz(frames: np.ndarray, path: str | Path, fps: float) -> None:
    path = Path(path)
    _ensure_parent(path)
    np.savez_compressed(path, frames=frames, fps=np.array([fps], dtype=np.float32))


def write_mask_mp4(mask_frames: np.ndarray, path: str | Path, fps: float) -> None:
    mask_u8 = np.clip(np.rint(mask_frames * 255.0), 0, 255).astype(np.uint8)
    rgb = np.repeat(mask_u8[:, :, :, None], repeats=3, axis=3)
    write_rgb_video_mp4(rgb, path, fps)


def write_mask_png_sequence(mask_frames: np.ndarray, directory: str | Path) -> None:
    directory = Path(directory)
    _ensure_dir(directory)
    mask_u8 = np.clip(np.rint(mask_frames * 255.0), 0, 255).astype(np.uint8)
    for index, frame in enumerate(mask_u8):
        _write_gray_png(directory / f"{index:06d}.png", frame)


def write_mask_npz(mask_frames: np.ndarray, path: str | Path, fps: float) -> None:
    path = Path(path)
    _ensure_parent(path)
    np.savez_compressed(path, mask=mask_frames.astype(np.float32), fps=np.array([fps], dtype=np.float32))


def write_rgb_artifact(
    *,
    frames: np.ndarray,
    output_root: str | Path,
    stem: str,
    artifact_format: str,
    fps: float,
    artifact_type: str = "video",
) -> dict:
    validate_rgb_video(stem, frames)
    output_root = Path(output_root)
    if artifact_format not in INTERMEDIATE_SAVE_FORMATS:
        raise ValueError(f"Unsupported RGB artifact format: {artifact_format}")

    if artifact_format == "mp4":
        rel_path = f"{stem}.mp4"
        write_rgb_video_mp4(frames, output_root / rel_path, fps)
    elif artifact_format == "png_seq":
        rel_path = stem
        write_rgb_png_sequence(frames, output_root / rel_path)
    else:
        rel_path = f"{stem}.npz"
        write_rgb_npz(frames, output_root / rel_path, fps)

    return {
        "path": rel_path,
        "type": artifact_type,
        "format": artifact_format,
        "frame_count": int(frames.shape[0]),
        "height": int(frames.shape[1]),
        "width": int(frames.shape[2]),
        "channels": int(frames.shape[3]),
        "color_space": "rgb",
        "dtype": str(frames.dtype),
        "shape": list(frames.shape),
        "fps": float(fps),
    }


def write_person_mask_artifact(
    *,
    mask_frames: np.ndarray,
    output_root: str | Path,
    stem: str,
    artifact_format: str,
    fps: float,
    mask_semantics: str = "person_foreground",
) -> dict:
    validate_person_mask_frames(stem, mask_frames)
    output_root = Path(output_root)
    if artifact_format not in INTERMEDIATE_SAVE_FORMATS:
        raise ValueError(f"Unsupported mask artifact format: {artifact_format}")

    if artifact_format == "mp4":
        rel_path = f"{stem}.mp4"
        write_mask_mp4(mask_frames, output_root / rel_path, fps)
        stored_channels = 3
        stored_value_range = [0, 255]
    elif artifact_format == "png_seq":
        rel_path = stem
        write_mask_png_sequence(mask_frames, output_root / rel_path)
        stored_channels = 1
        stored_value_range = [0, 255]
    else:
        rel_path = f"{stem}.npz"
        write_mask_npz(mask_frames, output_root / rel_path, fps)
        stored_channels = 1
        stored_value_range = [0.0, 1.0]

    return {
        "path": rel_path,
        "type": "video",
        "format": artifact_format,
        "frame_count": int(mask_frames.shape[0]),
        "height": int(mask_frames.shape[1]),
        "width": int(mask_frames.shape[2]),
        "channels": 1,
        "stored_channels": stored_channels,
        "color_space": "rgb" if artifact_format == "mp4" else None,
        "dtype": str(mask_frames.dtype),
        "shape": list(mask_frames.shape),
        "fps": float(fps),
        "value_range": [0.0, 1.0],
        "stored_value_range": stored_value_range,
        "mask_semantics": mask_semantics,
    }


def load_rgb_artifact(path: str | Path, artifact_format: str | None = None) -> np.ndarray:
    path = Path(path)
    artifact_format = artifact_format or infer_artifact_format(path)
    if artifact_format == "mp4":
        return read_video_rgb(path)
    if artifact_format == "png_seq":
        if not path.is_dir():
            raise FileNotFoundError(f"PNG sequence directory does not exist: {path}")
        frames = [load_image_rgb(frame_path) for frame_path in sorted(path.glob("*.png"))]
        if not frames:
            raise FileNotFoundError(f"No PNG frames found in sequence directory: {path}")
        frames = np.stack(frames)
        validate_rgb_video(str(path), frames)
        return frames
    if artifact_format == "npz":
        data = np.load(path)
        if "frames" not in data:
            raise KeyError(f"RGB npz artifact is missing 'frames': {path}")
        frames = np.asarray(data["frames"])
        validate_rgb_video(str(path), frames)
        return frames
    raise ValueError(f"Unsupported RGB artifact format: {artifact_format}")


def load_mask_artifact(path: str | Path, artifact_format: str | None = None) -> np.ndarray:
    path = Path(path)
    artifact_format = artifact_format or infer_artifact_format(path)
    if artifact_format == "mp4":
        mask_rgb = read_video_rgb(path)
        mask = mask_rgb[:, :, :, 0].astype(np.float32) / 255.0
    elif artifact_format == "png_seq":
        if not path.is_dir():
            raise FileNotFoundError(f"PNG sequence directory does not exist: {path}")
        frames = []
        for frame_path in sorted(path.glob("*.png")):
            frame = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
            if frame is None:
                raise FileNotFoundError(f"Failed to read mask PNG frame: {frame_path}")
            frames.append(frame.astype(np.float32) / 255.0)
        if not frames:
            raise FileNotFoundError(f"No PNG frames found in mask sequence directory: {path}")
        mask = np.stack(frames)
    elif artifact_format == "npz":
        data = np.load(path)
        if "mask" not in data:
            raise KeyError(f"Mask npz artifact is missing 'mask': {path}")
        mask = np.asarray(data["mask"]).astype(np.float32)
    else:
        raise ValueError(f"Unsupported person mask artifact format: {artifact_format}")
    validate_person_mask_frames(str(path), mask)
    return mask


def load_person_mask_artifact(path: str | Path, artifact_format: str | None = None) -> np.ndarray:
    return load_mask_artifact(path, artifact_format)


def infer_output_format(save_file: str | None, output_format: str) -> str:
    if output_format != "auto":
        if output_format not in OUTPUT_VIDEO_FORMATS:
            raise ValueError(f"Unsupported output format: {output_format}")
        return output_format
    if save_file is None:
        return "mp4"
    suffix = Path(save_file).suffix.lower()
    if suffix == ".mkv":
        return "ffv1"
    if suffix == ".mp4":
        return "mp4"
    if suffix == "":
        return "png_seq"
    return "mp4"


def write_output_frames(frames: np.ndarray, save_file: str | Path, fps: float, output_format: str) -> str:
    save_path = Path(save_file)
    if output_format == "mp4":
        if save_path.suffix.lower() != ".mp4":
            save_path = save_path.with_suffix(".mp4")
        write_rgb_video_mp4(frames, save_path, fps)
        return str(save_path)
    if output_format == "png_seq":
        directory = save_path if save_path.suffix == "" else save_path.with_suffix("")
        write_rgb_png_sequence(frames, directory)
        return str(directory)
    if output_format == "ffv1":
        if save_path.suffix.lower() != ".mkv":
            save_path = save_path.with_suffix(".mkv")
        _ensure_parent(save_path)
        import imageio
        writer = imageio.get_writer(str(save_path), fps=fps, codec="ffv1")
        try:
            for frame in frames:
                writer.append_data(frame)
        finally:
            writer.close()
        return str(save_path)
    raise ValueError(f"Unsupported output format: {output_format}")


def describe_output_path(save_path: str | Path, output_format: str) -> str:
    save_path = Path(save_path)
    if output_format == "mp4":
        return str(save_path if save_path.suffix.lower() == ".mp4" else save_path.with_suffix(".mp4"))
    if output_format == "ffv1":
        return str(save_path if save_path.suffix.lower() == ".mkv" else save_path.with_suffix(".mkv"))
    if output_format == "png_seq":
        return str(save_path if save_path.suffix == "" else save_path.with_suffix(""))
    raise ValueError(f"Unsupported output format: {output_format}")
