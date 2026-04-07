from pathlib import Path
import sys
import tempfile

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan.utils.animate_contract import SOFT_BAND_SEMANTICS, build_preprocess_metadata
from wan.utils.media_io import load_mask_artifact, write_person_mask_artifact
from wan.utils.replacement_masks import (
    build_soft_boundary_band,
    compose_background_keep_mask,
    derive_replacement_regions,
    resize_mask_volume,
)


def main():
    frame_count, height, width = 6, 96, 128
    hard_mask = np.zeros((frame_count, height, width), dtype=np.float32)
    hard_mask[:, 24:72, 40:88] = 1.0

    soft_band = build_soft_boundary_band(hard_mask, band_width=8, blur_kernel_size=5)
    assert soft_band.shape == hard_mask.shape
    assert float(soft_band.max()) > 0.0
    assert np.allclose(soft_band[:, 40:56, 56:72], 0.0, atol=1e-4)
    assert np.any(soft_band[:, 20:24, 44:84] > 0.0)

    background_keep_hard = compose_background_keep_mask(hard_mask, mode="hard")
    background_keep_soft = compose_background_keep_mask(
        hard_mask,
        soft_band=soft_band,
        mode="soft_band",
        boundary_strength=0.5,
    )
    assert background_keep_hard.shape == background_keep_soft.shape
    assert float(background_keep_soft.min()) >= 0.0
    assert float(background_keep_soft.max()) <= 1.0
    assert np.any(background_keep_soft.numpy() < background_keep_hard.numpy())

    latent_keep = resize_mask_volume(background_keep_soft, output_size=(12, 16), mode="area")
    assert latent_keep.shape == (frame_count, 12, 16)
    assert float(latent_keep.max()) <= 1.0 and float(latent_keep.min()) >= 0.0
    regions = derive_replacement_regions(background_keep_soft, transition_low=0.1, transition_high=0.9)
    assert set(regions.keys()) == {
        "hard_background_keep",
        "transition_band",
        "free_replacement",
        "replacement_strength",
    }
    assert float(regions["transition_band"].sum()) > 0.0

    with tempfile.TemporaryDirectory() as tmpdir:
        output_root = Path(tmpdir)
        artifact = write_person_mask_artifact(
            mask_frames=soft_band,
            output_root=output_root,
            stem="src_soft_band",
            artifact_format="npz",
            fps=30.0,
            mask_semantics=SOFT_BAND_SEMANTICS,
        )
        loaded = load_mask_artifact(output_root / artifact["path"], artifact["format"])
        assert np.allclose(loaded, soft_band, atol=1e-6)

        metadata = build_preprocess_metadata(
            video_path="video.mp4",
            refer_image_path="ref.png",
            output_path=str(output_root),
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
                "background": {"path": "src_bg.mp4", "type": "video", "format": "mp4", "frame_count": frame_count, "height": height, "width": width, "channels": 3, "color_space": "rgb", "dtype": "uint8", "shape": [frame_count, height, width, 3], "fps": 30.0},
                "person_mask": {"path": "src_mask.npz", "type": "video", "format": "npz", "frame_count": frame_count, "height": height, "width": width, "channels": 1, "stored_channels": 1, "dtype": "float32", "shape": [frame_count, height, width], "fps": 30.0, "value_range": [0.0, 1.0], "stored_value_range": [0.0, 1.0], "mask_semantics": "person_foreground"},
                "soft_band": artifact,
            },
            soft_mask_settings={
                "soft_mask_mode": "soft_band",
                "soft_mask_band_width": 8,
                "soft_mask_blur_kernel": 5,
            },
        )
        assert metadata["processing"]["soft_mask"]["soft_mask_mode"] == "soft_band"
        assert metadata["src_files"]["soft_band"]["mask_semantics"] == SOFT_BAND_SEMANTICS

    print("Synthetic soft-mask preprocess/generate bridge: PASS")
    print("Synthetic soft-mask artifact roundtrip: PASS")


if __name__ == "__main__":
    main()
