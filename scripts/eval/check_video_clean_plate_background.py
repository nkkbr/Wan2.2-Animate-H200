import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
PREPROCESS_DIR = REPO_ROOT / "wan" / "modules" / "animate" / "preprocess"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PREPROCESS_DIR) not in sys.path:
    sys.path.insert(0, str(PREPROCESS_DIR))

from background_clean_plate import build_clean_plate_background
from wan.utils.animate_contract import build_preprocess_metadata


def _make_background(height, width):
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    background = np.stack(
        [
            (37 * xx + 13 * yy) % 255,
            (19 * xx + 53 * yy) % 255,
            (11 * xx + 29 * yy) % 255,
        ],
        axis=-1,
    )
    return background.astype(np.uint8)


def _make_frames(frame_count, height, width):
    background = _make_background(height, width)
    frames = []
    masks = []
    for idx in range(frame_count):
        frame = background.copy()
        x1 = 12 + idx * 5
        x2 = x1 + 24
        y1, y2 = 18, 50
        frame[y1:y2, x1:x2] = np.array([245, 38, 38], dtype=np.uint8)
        mask = np.zeros((height, width), dtype=np.float32)
        mask[y1:y2, x1:x2] = 1.0
        masks.append(mask)
        frames.append(frame)
    return background, np.stack(frames), np.stack(masks)


def _masked_mae(frames, target, mask):
    mask = mask[:, :, :, None]
    diff = np.abs(frames.astype(np.float32) - target.astype(np.float32)[None])
    denom = np.maximum(mask.sum(), 1.0)
    return float((diff * mask).sum() / denom)


def _temporal_fluctuation(frames, region):
    if len(frames) <= 1:
        return 0.0
    diffs = np.abs(frames[1:].astype(np.float32) - frames[:-1].astype(np.float32)).mean(axis=-1)
    region_union = np.maximum(region[1:], region[:-1]).astype(np.float32)
    denom = np.maximum(region_union.sum(), 1.0)
    return float((diffs * region_union).sum() / denom)


def main():
    frame_count, height, width = 8, 72, 96
    background, frames, masks = _make_frames(frame_count, height, width)
    soft_band = np.zeros_like(masks, dtype=np.float32)
    for idx in range(frame_count):
        soft_band[idx] = cv2.GaussianBlur(masks[idx], (9, 9), sigmaX=0)
        soft_band[idx] = np.clip(soft_band[idx] - masks[idx], 0.0, 1.0)
    background_keep_prior = np.clip(1.0 - np.maximum(masks, soft_band), 0.0, 1.0)

    image_bg, image_debug = build_clean_plate_background(
        frames,
        masks,
        bg_inpaint_mode="image",
        soft_band=soft_band,
        background_keep_prior=background_keep_prior,
        bg_inpaint_mask_expand=6,
        bg_inpaint_radius=3.0,
        bg_inpaint_method="telea",
        bg_temporal_smooth_strength=0.0,
    )
    video_bg, video_debug = build_clean_plate_background(
        frames,
        masks,
        bg_inpaint_mode="video",
        soft_band=soft_band,
        background_keep_prior=background_keep_prior,
        bg_inpaint_mask_expand=6,
        bg_inpaint_radius=3.0,
        bg_inpaint_method="telea",
        bg_temporal_smooth_strength=0.1,
        bg_video_window_radius=2,
        bg_video_min_visible_count=1,
        bg_video_blend_strength=0.8,
    )

    image_mae = _masked_mae(image_bg, background, masks)
    video_mae = _masked_mae(video_bg, background, masks)
    image_temporal = _temporal_fluctuation(image_bg, masks)
    video_temporal = _temporal_fluctuation(video_bg, masks)
    assert video_mae < image_mae, f"clean_plate_video should reduce masked MAE ({video_mae:.3f} !< {image_mae:.3f})"
    assert video_temporal < image_temporal, (
        f"clean_plate_video should reduce temporal fluctuation ({video_temporal:.3f} !< {image_temporal:.3f})"
    )
    assert video_debug["background_mode"] == "clean_plate_video"
    assert image_debug["background_mode"] == "clean_plate_image"
    assert video_debug["support_mask"].shape == masks.shape

    with tempfile.TemporaryDirectory() as tmpdir:
        metadata = build_preprocess_metadata(
            video_path="video.mp4",
            refer_image_path="ref.png",
            output_path=tmpdir,
            replace_flag=True,
            retarget_flag=False,
            use_flux=False,
            resolution_area=[1280, 720],
            fps_request=30,
            fps_output=30.0,
            frame_count=frame_count,
            height=height,
            width=width,
            iterations=3,
            k=7,
            w_len=1,
            h_len=1,
            reference_height=1024,
            reference_width=768,
            src_files={
                "pose": {"path": "src_pose.mp4", "type": "video", "format": "mp4", "frame_count": frame_count, "height": height, "width": width, "channels": 3, "color_space": "rgb", "dtype": "uint8", "shape": [frame_count, height, width, 3], "fps": 30.0},
                "face": {"path": "src_face.mp4", "type": "video", "format": "mp4", "frame_count": frame_count, "height": 512, "width": 512, "channels": 3, "color_space": "rgb", "dtype": "uint8", "shape": [frame_count, 512, 512, 3], "fps": 30.0},
                "reference": {"path": "src_ref.png", "type": "image", "format": "png", "height": 1024, "width": 768, "channels": 3, "color_space": "rgb", "dtype": "uint8", "shape": [1024, 768, 3], "resized_height": height, "resized_width": width},
                "background": {"path": "src_bg.mp4", "type": "video", "format": "mp4", "frame_count": frame_count, "height": height, "width": width, "channels": 3, "color_space": "rgb", "dtype": "uint8", "shape": [frame_count, height, width, 3], "fps": 30.0, "background_mode": "clean_plate_video"},
                "person_mask": {"path": "src_mask.mp4", "type": "video", "format": "mp4", "frame_count": frame_count, "height": height, "width": width, "channels": 1, "stored_channels": 3, "dtype": "float32", "shape": [frame_count, height, width], "fps": 30.0, "value_range": [0.0, 1.0], "stored_value_range": [0, 255], "mask_semantics": "person_foreground"},
            },
            background_settings={
                "bg_inpaint_mode": "video",
                "bg_inpaint_method": "telea",
                "bg_inpaint_mask_expand": 6,
                "bg_inpaint_radius": 3.0,
                "bg_temporal_smooth_strength": 0.1,
                "bg_video_window_radius": 2,
                "bg_video_min_visible_count": 1,
                "bg_video_blend_strength": 0.8,
                "stats": video_debug["stats"],
            },
        )
        assert metadata["processing"]["background"]["bg_inpaint_mode"] == "video"
        assert metadata["processing"]["background"]["stats"]["support_ratio_mean"] > 0.0
        assert metadata["src_files"]["background"]["background_mode"] == "clean_plate_video"

    print("Synthetic clean-plate video consistency: PASS")
    print("Synthetic clean-plate video metadata contract: PASS")


if __name__ == "__main__":
    main()
