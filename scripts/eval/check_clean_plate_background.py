import sys
import tempfile
from pathlib import Path

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
    yy, xx = np.meshgrid(np.linspace(0, 1, height), np.linspace(0, 1, width), indexing="ij")
    background = np.stack(
        [
            xx * 255.0,
            yy * 255.0,
            (0.5 * xx + 0.5 * yy) * 255.0,
        ],
        axis=-1,
    )
    return np.clip(np.rint(background), 0, 255).astype(np.uint8)


def _make_frames(frame_count, height, width):
    background = _make_background(height, width)
    frames = []
    masks = []
    for idx in range(frame_count):
        frame = background.copy()
        x1 = 28 + idx
        x2 = 60 + idx
        y1, y2 = 20, 52
        frame[y1:y2, x1:x2] = np.array([255, 32, 32], dtype=np.uint8)
        mask = np.zeros((height, width), dtype=np.float32)
        mask[y1:y2, x1:x2] = 1.0
        frames.append(frame)
        masks.append(mask)
    return background, np.stack(frames), np.stack(masks)


def _masked_mae(frames, target, mask):
    mask = mask[:, :, :, None]
    diff = np.abs(frames.astype(np.float32) - target.astype(np.float32)[None])
    denom = np.maximum(mask.sum(), 1.0)
    return float((diff * mask).sum() / denom)


def main():
    frame_count, height, width = 6, 72, 96
    background, frames, masks = _make_frames(frame_count, height, width)

    hole_bg, hole_debug = build_clean_plate_background(
        frames,
        masks,
        bg_inpaint_mode="none",
    )
    clean_bg, clean_debug = build_clean_plate_background(
        frames,
        masks,
        bg_inpaint_mode="image",
        bg_inpaint_mask_expand=8,
        bg_inpaint_radius=3.0,
        bg_inpaint_method="telea",
        bg_temporal_smooth_strength=0.0,
    )

    hole_mae = _masked_mae(hole_bg, background, masks)
    clean_mae = _masked_mae(clean_bg, background, masks)
    assert clean_mae < hole_mae, f"clean plate should reduce masked-region error ({clean_mae:.3f} !< {hole_mae:.3f})"
    assert clean_debug["background_mode"] == "clean_plate_image"
    assert hole_debug["background_mode"] == "hole"
    assert clean_debug["inpaint_mask"].shape == masks.shape

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
                "background": {"path": "src_bg.mp4", "type": "video", "format": "mp4", "frame_count": frame_count, "height": height, "width": width, "channels": 3, "color_space": "rgb", "dtype": "uint8", "shape": [frame_count, height, width, 3], "fps": 30.0, "background_mode": "clean_plate_image"},
                "person_mask": {"path": "src_mask.mp4", "type": "video", "format": "mp4", "frame_count": frame_count, "height": height, "width": width, "channels": 1, "stored_channels": 3, "dtype": "float32", "shape": [frame_count, height, width], "fps": 30.0, "value_range": [0.0, 1.0], "stored_value_range": [0, 255], "mask_semantics": "person_foreground"},
            },
            background_settings={
                "bg_inpaint_mode": "image",
                "bg_inpaint_method": "telea",
                "bg_inpaint_mask_expand": 8,
                "bg_inpaint_radius": 3.0,
                "bg_temporal_smooth_strength": 0.0,
            },
        )
        assert metadata["processing"]["background"]["bg_inpaint_mode"] == "image"
        assert metadata["src_files"]["background"]["background_mode"] == "clean_plate_image"

    print("Synthetic clean-plate background improvement: PASS")
    print("Synthetic clean-plate metadata contract: PASS")


if __name__ == "__main__":
    main()
