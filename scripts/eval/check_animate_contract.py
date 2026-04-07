import argparse
import sys
import tempfile
import warnings
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan.utils.animate_contract import (
    load_image_rgb,
    read_video_rgb,
    resolve_preprocess_artifacts,
    validate_loaded_preprocess_bundle,
)


def padding_resize(img_ori, height=512, width=512, padding_color=(0, 0, 0), interpolation=cv2.INTER_LINEAR):
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
        img_pad[:, padding: padding + new_width, :] = img
    else:
        new_height = int(width / ori_width * ori_height)
        img = cv2.resize(img_ori, (width, new_height), interpolation=interpolation)
        padding = int((height - new_height) / 2)
        img_pad[padding: padding + new_height, :, :] = img

    return img_pad


def _assert_channel_dominance(region: np.ndarray, dominant: int, name: str) -> None:
    means = region.mean(axis=(0, 1))
    dominant_value = means[dominant]
    others = [means[index] for index in range(3) if index != dominant]
    if not all(dominant_value > other + 30 for other in others):
        raise AssertionError(f"{name} channel dominance check failed. means={means}")


def read_video_rgb_for_test(path: str | Path) -> tuple[np.ndarray, str]:
    try:
        return read_video_rgb(path), "decord"
    except ModuleNotFoundError as exc:
        if exc.name != "decord":
            raise

    warnings.warn(
        "decord is not installed in the current environment; falling back to cv2.VideoCapture "
        "for contract validation. This validates color ordering and shape, but not the exact "
        "production decoder path.",
        stacklevel=2,
    )
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video for fallback contract test: {path}")
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise RuntimeError(f"No frames decoded during fallback contract test: {path}")
    return np.stack(frames), "cv2-fallback"


def run_synthetic_color_contract() -> None:
    with tempfile.TemporaryDirectory(prefix="animate_contract_") as tmp_dir:
        tmp_path = Path(tmp_dir)

        rgb_image = np.zeros((48, 72, 3), dtype=np.uint8)
        rgb_image[:, :24] = [255, 0, 0]
        rgb_image[:, 24:48] = [0, 255, 0]
        rgb_image[:, 48:] = [0, 0, 255]
        image_path = tmp_path / "reference.png"
        cv2.imwrite(str(image_path), cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))

        loaded_rgb_image = load_image_rgb(image_path)
        _assert_channel_dominance(loaded_rgb_image[:, :24], dominant=0, name="reference-red")
        _assert_channel_dominance(loaded_rgb_image[:, 24:48], dominant=1, name="reference-green")
        _assert_channel_dominance(loaded_rgb_image[:, 48:], dominant=2, name="reference-blue")

        frame0 = np.zeros((64, 64, 3), dtype=np.uint8)
        frame0[:, :32] = [255, 0, 0]
        frame0[:, 32:] = [0, 0, 255]
        frame1 = np.zeros((64, 64, 3), dtype=np.uint8)
        frame1[:32, :] = [0, 255, 0]
        frame1[32:, :] = [255, 0, 0]

        video_path = tmp_path / "synthetic.mp4"
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            2.0,
            (64, 64),
        )
        if not writer.isOpened():
            raise RuntimeError("Failed to open cv2.VideoWriter for synthetic contract test.")
        for frame in (frame0, frame1):
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()

        loaded_frames, reader_name = read_video_rgb_for_test(video_path)
        if loaded_frames.shape[0] != 2:
            raise AssertionError(f"Expected 2 synthetic video frames, got {loaded_frames.shape[0]}.")
        _assert_channel_dominance(loaded_frames[0, :, :32], dominant=0, name="video-frame0-left-red")
        _assert_channel_dominance(loaded_frames[0, :, 32:], dominant=2, name="video-frame0-right-blue")
        _assert_channel_dominance(loaded_frames[1, :32, :], dominant=1, name="video-frame1-top-green")
        _assert_channel_dominance(loaded_frames[1, 32:, :], dominant=0, name="video-frame1-bottom-red")
        print(f"Synthetic video contract reader: {reader_name}")


def validate_preprocess_directory(src_root_path: str, replace_flag: bool) -> None:
    artifacts, metadata = resolve_preprocess_artifacts(src_root_path, replace_flag=replace_flag)
    cond_images, pose_reader = read_video_rgb_for_test(artifacts["pose"])
    face_images, face_reader = read_video_rgb_for_test(artifacts["face"])
    refer_image_rgb = load_image_rgb(artifacts["reference"])
    refer_image_rgb = padding_resize(refer_image_rgb, height=cond_images.shape[1], width=cond_images.shape[2])

    kwargs = {
        "cond_images": cond_images,
        "face_images": face_images,
        "refer_image_rgb": refer_image_rgb,
        "metadata": metadata,
    }
    if replace_flag:
        bg_images, bg_reader = read_video_rgb_for_test(artifacts["background"])
        person_mask_rgb, mask_reader = read_video_rgb_for_test(artifacts["person_mask"])
        person_mask_images = person_mask_rgb[:, :, :, 0].astype(np.float32) / 255.0
        kwargs["bg_images"] = bg_images
        kwargs["person_mask_images"] = person_mask_images

    validate_loaded_preprocess_bundle(**kwargs)
    print(f"Directory contract readers: pose={pose_reader}, face={face_reader}")
    if replace_flag:
        print(f"Directory contract readers: background={bg_reader}, person_mask={mask_reader}")


def main():
    parser = argparse.ArgumentParser(description="Check Wan-Animate preprocess/generate interface contracts.")
    parser.add_argument(
        "--src_root_path",
        type=str,
        default=None,
        help="Optional preprocess output directory to validate against metadata and file-level contracts.",
    )
    parser.add_argument(
        "--replace_flag",
        action="store_true",
        default=False,
        help="Validate replacement-mode artifacts in the provided --src_root_path directory.",
    )
    parser.add_argument(
        "--skip_synthetic",
        action="store_true",
        default=False,
        help="Skip the built-in synthetic RGB roundtrip checks.",
    )
    args = parser.parse_args()

    if not args.skip_synthetic:
        run_synthetic_color_contract()
        print("Synthetic RGB contract checks: PASS")

    if args.src_root_path is not None:
        validate_preprocess_directory(args.src_root_path, replace_flag=args.replace_flag)
        print(f"Preprocess directory contract checks: PASS ({args.src_root_path})")


if __name__ == "__main__":
    main()
