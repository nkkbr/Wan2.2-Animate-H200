from pathlib import Path
import sys
import tempfile

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PREPROCESS_DIR = REPO_ROOT / "wan" / "modules" / "animate" / "preprocess"
if str(PREPROCESS_DIR) not in sys.path:
    sys.path.insert(0, str(PREPROCESS_DIR))

from boundary_fusion import fuse_boundary_signals
from matting_adapter import run_matting_adapter
from parsing_adapter import run_parsing_adapter
from wan.utils.animate_contract import (
    BACKGROUND_KEEP_PRIOR_SEMANTICS,
    BOUNDARY_BAND_SEMANTICS,
    HARD_FOREGROUND_SEMANTICS,
    SOFT_ALPHA_SEMANTICS,
    SOFT_BAND_SEMANTICS,
    build_preprocess_metadata,
    resolve_preprocess_artifacts,
    validate_loaded_preprocess_bundle,
    write_preprocess_metadata,
)
from wan.utils.media_io import load_mask_artifact, load_rgb_artifact, write_person_mask_artifact, write_rgb_artifact
from wan.utils.replacement_masks import build_soft_boundary_band


def _make_pose_meta():
    face = np.array(
        [
            [0.43, 0.34, 1.0],
            [0.46, 0.33, 1.0],
            [0.50, 0.34, 1.0],
            [0.54, 0.35, 1.0],
            [0.56, 0.39, 1.0],
            [0.52, 0.42, 1.0],
            [0.46, 0.42, 1.0],
            [0.42, 0.39, 1.0],
        ],
        dtype=np.float32,
    )
    left_hand = np.array(
        [
            [0.28, 0.60, 1.0],
            [0.26, 0.61, 1.0],
            [0.24, 0.62, 1.0],
            [0.22, 0.63, 1.0],
            [0.21, 0.61, 1.0],
            [0.23, 0.59, 1.0],
        ],
        dtype=np.float32,
    )
    right_hand = np.array(
        [
            [0.73, 0.60, 1.0],
            [0.75, 0.61, 1.0],
            [0.77, 0.62, 1.0],
            [0.79, 0.63, 1.0],
            [0.80, 0.61, 1.0],
            [0.78, 0.59, 1.0],
        ],
        dtype=np.float32,
    )
    return {
        "keypoints_face": face,
        "keypoints_left_hand": left_hand,
        "keypoints_right_hand": right_hand,
    }


def _make_synthetic_case(frame_count: int = 6, height: int = 96, width: int = 128):
    background = np.full((frame_count, height, width, 3), fill_value=np.array([46, 58, 68], dtype=np.uint8), dtype=np.uint8)
    frames = background.copy()
    hard_mask = np.zeros((frame_count, height, width), dtype=np.float32)
    hard_mask[:, 28:76, 40:88] = 1.0
    frames[:, 28:76, 40:88] = np.array([208, 180, 166], dtype=np.uint8)

    # Hair-like thin protrusions above the hard mask and hand-like thin structures on both sides.
    for index in range(frame_count):
        for x in (48, 54, 60, 66, 72, 78):
            frames[index, 20:28, x:x + 2] = np.array([214, 188, 176], dtype=np.uint8)
        frames[index, 56:60, 26:40] = np.array([210, 184, 170], dtype=np.uint8)
        frames[index, 56:60, 88:102] = np.array([210, 184, 170], dtype=np.uint8)

    soft_band = build_soft_boundary_band(hard_mask, band_width=4, blur_kernel_size=3)
    pose_metas = [_make_pose_meta() for _ in range(frame_count)]
    face_bboxes = [(50, 74, 22, 44) for _ in range(frame_count)]
    return frames, background, hard_mask, soft_band, pose_metas, face_bboxes


def main():
    frames, background, hard_mask, soft_band, pose_metas, face_bboxes = _make_synthetic_case()
    parsing = run_parsing_adapter(
        frames=frames,
        hard_mask=hard_mask,
        pose_metas=pose_metas,
        face_bboxes=face_bboxes,
        mode="heuristic",
    )
    matting = run_matting_adapter(
        frames=frames,
        hard_mask=hard_mask,
        soft_band=soft_band,
        parsing_boundary_prior=parsing["semantic_boundary_prior"],
        mode="heuristic",
    )
    fusion = fuse_boundary_signals(
        hard_mask=hard_mask,
        soft_band=soft_band,
        parsing_output=parsing,
        matting_output=matting,
        mode="heuristic",
    )

    assert fusion["hard_foreground"].shape == hard_mask.shape
    assert fusion["soft_alpha"].shape == hard_mask.shape
    assert fusion["boundary_band"].shape == hard_mask.shape
    assert fusion["background_keep_prior"].shape == hard_mask.shape
    assert float(fusion["soft_alpha"].max()) <= 1.0 and float(fusion["soft_alpha"].min()) >= 0.0

    hair_region = np.zeros_like(hard_mask, dtype=bool)
    hair_region[:, 20:28, 48:80] = True
    hand_region = np.zeros_like(hard_mask, dtype=bool)
    hand_region[:, 55:61, 24:40] = True
    hand_region[:, 55:61, 88:104] = True
    thin_region = hair_region | hand_region
    assert float(fusion["boundary_band"][thin_region].mean()) > float(soft_band[thin_region].mean())
    assert float(fusion["soft_alpha"][thin_region].mean()) > float(soft_band[thin_region].mean())
    center_region = fusion["hard_foreground"][:, 40:64, 50:78]
    assert float(center_region.mean()) > 0.99
    far_background = fusion["background_keep_prior"][:, :12, :12]
    assert float(far_background.mean()) > 0.95

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        pose_artifact = write_rgb_artifact(frames=frames, output_root=root, stem="src_pose", artifact_format="npz", fps=5.0)
        face_artifact = write_rgb_artifact(frames=background, output_root=root, stem="src_face", artifact_format="npz", fps=5.0)
        background_artifact = write_rgb_artifact(frames=background, output_root=root, stem="src_bg", artifact_format="npz", fps=5.0)
        person_mask_artifact = write_person_mask_artifact(
            mask_frames=hard_mask,
            output_root=root,
            stem="src_mask",
            artifact_format="npz",
            fps=5.0,
        )
        soft_band_artifact = write_person_mask_artifact(
            mask_frames=soft_band,
            output_root=root,
            stem="src_soft_band",
            artifact_format="npz",
            fps=5.0,
            mask_semantics=SOFT_BAND_SEMANTICS,
        )
        boundary_band_artifact = write_person_mask_artifact(
            mask_frames=fusion["boundary_band"],
            output_root=root,
            stem="src_boundary_band",
            artifact_format="npz",
            fps=5.0,
            mask_semantics=BOUNDARY_BAND_SEMANTICS,
        )
        soft_alpha_artifact = write_person_mask_artifact(
            mask_frames=fusion["soft_alpha"],
            output_root=root,
            stem="src_soft_alpha",
            artifact_format="npz",
            fps=5.0,
            mask_semantics=SOFT_ALPHA_SEMANTICS,
        )
        background_keep_prior_artifact = write_person_mask_artifact(
            mask_frames=fusion["background_keep_prior"],
            output_root=root,
            stem="src_background_keep_prior",
            artifact_format="npz",
            fps=5.0,
            mask_semantics=BACKGROUND_KEEP_PRIOR_SEMANTICS,
        )
        reference = np.full((height := frames.shape[1], width := frames.shape[2], 3), 127, dtype=np.uint8)
        ok = cv2.imwrite(str(root / "src_ref.png"), cv2.cvtColor(reference, cv2.COLOR_RGB2BGR))
        assert ok

        metadata = build_preprocess_metadata(
            video_path="video.mp4",
            refer_image_path="ref.png",
            output_path=str(root),
            replace_flag=True,
            retarget_flag=False,
            use_flux=False,
            resolution_area=[width, height],
            fps_request=5,
            fps_output=5.0,
            frame_count=frames.shape[0],
            height=height,
            width=width,
            iterations=3,
            k=7,
            w_len=1,
            h_len=1,
            reference_height=height,
            reference_width=width,
            src_files={
                "pose": pose_artifact,
                "face": face_artifact,
                "reference": {
                    "path": "src_ref.png",
                    "type": "image",
                    "format": "png",
                    "height": height,
                    "width": width,
                    "channels": 3,
                    "color_space": "rgb",
                    "dtype": "uint8",
                    "shape": [height, width, 3],
                    "resized_height": height,
                    "resized_width": width,
                },
                "background": {**background_artifact, "background_mode": "hole"},
                "person_mask": person_mask_artifact,
                "soft_band": soft_band_artifact,
                "hard_foreground": {**person_mask_artifact, "mask_semantics": HARD_FOREGROUND_SEMANTICS},
                "soft_alpha": soft_alpha_artifact,
                "boundary_band": boundary_band_artifact,
                "background_keep_prior": background_keep_prior_artifact,
            },
            soft_mask_settings={"soft_mask_mode": "soft_band", "soft_mask_band_width": 4, "soft_mask_blur_kernel": 3},
            boundary_fusion_settings={"boundary_fusion_mode": "heuristic", "parsing_mode": "heuristic", "matting_mode": "heuristic"},
        )
        write_preprocess_metadata(root, metadata)
        artifacts, loaded_metadata = resolve_preprocess_artifacts(root, replace_flag=True)
        validate_loaded_preprocess_bundle(
            cond_images=load_rgb_artifact(artifacts["pose"]["path"], artifacts["pose"]["format"]),
            face_images=load_rgb_artifact(artifacts["face"]["path"], artifacts["face"]["format"]),
            refer_image_rgb=cv2.cvtColor(cv2.imread(str(root / "src_ref.png"), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB),
            metadata=loaded_metadata,
            bg_images=load_rgb_artifact(artifacts["background"]["path"], artifacts["background"]["format"]),
            person_mask_images=load_mask_artifact(artifacts["person_mask"]["path"], artifacts["person_mask"]["format"]),
            soft_band_images=load_mask_artifact(artifacts["soft_band"]["path"], artifacts["soft_band"]["format"]),
            hard_foreground_images=load_mask_artifact(artifacts["hard_foreground"]["path"], artifacts["hard_foreground"]["format"]),
            soft_alpha_images=load_mask_artifact(artifacts["soft_alpha"]["path"], artifacts["soft_alpha"]["format"]),
            boundary_band_images=load_mask_artifact(artifacts["boundary_band"]["path"], artifacts["boundary_band"]["format"]),
            background_keep_prior_images=load_mask_artifact(
                artifacts["background_keep_prior"]["path"],
                artifacts["background_keep_prior"]["format"],
            ),
        )

    print("Synthetic parsing/matting fusion boundary gain: PASS")
    print("Synthetic fused boundary metadata contract: PASS")


if __name__ == "__main__":
    main()
