from pathlib import Path
import sys
import tempfile

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
PREPROCESS_ROOT = REPO_ROOT / "wan/modules/animate/preprocess"
if str(PREPROCESS_ROOT) not in sys.path:
    sys.path.insert(0, str(PREPROCESS_ROOT))

from reference_normalization import (  # noqa: E402
    bbox_from_pose_meta,
    estimate_driver_target_bbox,
    make_reference_normalization_preview,
    normalize_reference_image,
    project_bbox_with_letterbox,
    scale_bbox_between_shapes,
    write_reference_image,
)
from wan.utils.animate_contract import build_preprocess_metadata, validate_preprocess_metadata  # noqa: E402
from wan.utils.media_io import write_person_mask_artifact, write_rgb_artifact  # noqa: E402


def _meta_from_bbox(bbox, image_shape):
    height, width = image_shape
    x1, y1, x2, y2 = bbox
    body = np.asarray(
        [
            [x1 / width, y1 / height, 0.95],
            [x2 / width, y1 / height, 0.95],
            [x1 / width, y2 / height, 0.95],
            [x2 / width, y2 / height, 0.95],
            [((x1 + x2) / 2.0) / width, y1 / height, 0.95],
            [((x1 + x2) / 2.0) / width, y2 / height, 0.95],
        ],
        dtype=np.float32,
    )
    face = np.asarray(
        [
            [((x1 + x2) / 2.0) / width, (y1 + (y2 - y1) * 0.12) / height, 0.95],
            [((x1 + x2) / 2.0) / width, (y1 + (y2 - y1) * 0.18) / height, 0.95],
        ],
        dtype=np.float32,
    )
    empty = np.zeros((0, 3), dtype=np.float32)
    return {
        "keypoints_body": body,
        "keypoints_face": face,
        "keypoints_left_hand": empty,
        "keypoints_right_hand": empty,
    }


def _bbox_error(lhs, rhs):
    lhs = np.asarray(lhs, dtype=np.float32)
    rhs = np.asarray(rhs, dtype=np.float32)
    lhs_c = np.asarray([(lhs[0] + lhs[2]) / 2.0, (lhs[1] + lhs[3]) / 2.0], dtype=np.float32)
    rhs_c = np.asarray([(rhs[0] + rhs[2]) / 2.0, (rhs[1] + rhs[3]) / 2.0], dtype=np.float32)
    lhs_wh = np.asarray([lhs[2] - lhs[0], lhs[3] - lhs[1]], dtype=np.float32)
    rhs_wh = np.asarray([rhs[2] - rhs[0], rhs[3] - rhs[1]], dtype=np.float32)
    return float(np.linalg.norm(lhs_c - rhs_c) + np.linalg.norm(lhs_wh - rhs_wh))


def main():
    canvas_shape = (120, 160)
    driver_boxes = [
        [56, 18, 104, 110],
        [58, 19, 106, 109],
        [57, 20, 103, 111],
        [56, 18, 105, 110],
    ]
    driver_metas = [_meta_from_bbox(bbox, canvas_shape) for bbox in driver_boxes]
    target_bbox, target_stats = estimate_driver_target_bbox(
        driver_metas,
        image_shape=canvas_shape,
        source="median_first_n",
        num_frames=4,
        conf_thresh=0.35,
    )
    assert target_bbox is not None
    assert target_stats["valid_frames"] == 4

    original_reference_shape = (320, 200)
    reference_bbox_original = np.asarray([70, 20, 130, 300], dtype=np.float32)
    reference_detection_shape = (160, 100)
    reference_bbox_detection = scale_bbox_between_shapes(
        reference_bbox_original,
        from_shape=original_reference_shape,
        to_shape=reference_detection_shape,
    )
    reference_pose_meta = _meta_from_bbox(reference_bbox_detection, reference_detection_shape)
    extracted_detection_bbox = bbox_from_pose_meta(
        reference_pose_meta,
        image_shape=reference_detection_shape,
        conf_thresh=0.35,
    )
    extracted_original_bbox = scale_bbox_between_shapes(
        extracted_detection_bbox,
        from_shape=reference_detection_shape,
        to_shape=original_reference_shape,
    )
    assert extracted_original_bbox is not None
    assert np.allclose(extracted_original_bbox, reference_bbox_original, atol=2.5)

    reference_image = np.zeros((original_reference_shape[0], original_reference_shape[1], 3), dtype=np.uint8)
    x1, y1, x2, y2 = reference_bbox_original.astype(np.int32)
    reference_image[y1:y2, x1:x2] = np.array([255, 64, 64], dtype=np.uint8)

    normalized_reference, normalization_stats = normalize_reference_image(
        reference_image,
        reference_bbox=extracted_original_bbox,
        target_bbox=target_bbox,
        canvas_shape=canvas_shape,
        scale_clamp_min=0.75,
        scale_clamp_max=1.6,
    )
    assert normalized_reference is not None
    normalized_bbox = np.asarray(normalization_stats["normalized_bbox"], dtype=np.float32)
    original_letterbox_bbox = project_bbox_with_letterbox(
        extracted_original_bbox,
        image_shape=original_reference_shape,
        canvas_shape=canvas_shape,
    )
    assert _bbox_error(normalized_bbox, target_bbox) < _bbox_error(original_letterbox_bbox, target_bbox)

    preview = make_reference_normalization_preview(
        original_canvas=np.zeros((canvas_shape[0], canvas_shape[1], 3), dtype=np.uint8),
        normalized_canvas=normalized_reference,
        original_bbox=original_letterbox_bbox,
        target_bbox=target_bbox,
        normalized_bbox=normalized_bbox,
    )
    assert preview.shape[0] == canvas_shape[0]
    assert preview.shape[1] == canvas_shape[1] * 2

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        write_reference_image(root / "src_ref.png", reference_image)
        write_reference_image(root / "src_ref_normalized.png", normalized_reference)

        frame_count = 2
        height, width = canvas_shape
        dummy_rgb = np.zeros((frame_count, height, width, 3), dtype=np.uint8)
        dummy_face = np.zeros((frame_count, 512, 512, 3), dtype=np.uint8)
        dummy_mask = np.zeros((frame_count, height, width), dtype=np.float32)

        src_files = {
            "pose": write_rgb_artifact(
                frames=dummy_rgb,
                output_root=root,
                stem="src_pose",
                artifact_format="npz",
                fps=30.0,
            ),
            "face": write_rgb_artifact(
                frames=dummy_face,
                output_root=root,
                stem="src_face",
                artifact_format="npz",
                fps=30.0,
            ),
            "reference": {
                "path": "src_ref_normalized.png",
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
            "reference_original": {
                "path": "src_ref.png",
                "type": "image",
                "format": "png",
                "height": original_reference_shape[0],
                "width": original_reference_shape[1],
                "channels": 3,
                "color_space": "rgb",
                "dtype": "uint8",
                "shape": [original_reference_shape[0], original_reference_shape[1], 3],
                "resized_height": height,
                "resized_width": width,
            },
            "background": write_rgb_artifact(
                frames=dummy_rgb,
                output_root=root,
                stem="src_bg",
                artifact_format="npz",
                fps=30.0,
            ),
            "person_mask": write_person_mask_artifact(
                mask_frames=dummy_mask,
                output_root=root,
                stem="src_mask",
                artifact_format="npz",
                fps=30.0,
            ),
        }
        src_files["background"]["background_mode"] = "hole"

        metadata = build_preprocess_metadata(
            video_path="video.mp4",
            refer_image_path="reference.png",
            output_path=str(root),
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
            reference_height=height,
            reference_width=width,
            src_files=src_files,
            reference_settings={
                "reference_normalization_mode": "bbox_match",
                "reference_target_bbox_source": "median_first_n",
                "reference_target_bbox_frames": 4,
                "reference_bbox_conf_thresh": 0.35,
                "reference_scale_clamp_min": 0.75,
                "reference_scale_clamp_max": 1.6,
                "stats": normalization_stats,
            },
        )
        validate_preprocess_metadata(metadata, src_root_path=root)
        assert metadata["processing"]["reference_normalization"]["reference_normalization_mode"] == "bbox_match"
        assert metadata["src_files"]["reference"]["path"] == "src_ref_normalized.png"

    print("Synthetic reference normalization scale matching: PASS")
    print("Synthetic reference normalization metadata contract: PASS")


if __name__ == "__main__":
    main()
