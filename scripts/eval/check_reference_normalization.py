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
    estimate_driver_target_structure,
    make_reference_normalization_preview,
    normalize_reference_image,
    normalize_reference_image_structure_aware,
    project_bbox_with_letterbox,
    project_structure_with_letterbox,
    scale_bbox_between_shapes,
    scale_structure_between_shapes,
    structure_from_pose_meta,
    write_reference_image,
)
from wan.utils.animate_contract import build_preprocess_metadata, validate_preprocess_metadata  # noqa: E402
from wan.utils.media_io import write_person_mask_artifact, write_rgb_artifact  # noqa: E402


def _make_pose_meta(
    *,
    image_shape,
    bbox,
    shoulder_width_ratio,
    head_ratio,
    torso_ratio,
    leg_ratio,
):
    height, width = image_shape
    x1, y1, x2, y2 = map(float, bbox)
    cx = (x1 + x2) / 2.0
    person_h = max(y2 - y1, 1.0)
    shoulder_y = y1 + head_ratio * person_h
    hip_y = shoulder_y + torso_ratio * person_h
    foot_y = y1 + (head_ratio + torso_ratio + leg_ratio) * person_h
    foot_y = min(foot_y, y2)
    shoulder_half = 0.5 * shoulder_width_ratio * max(x2 - x1, 1.0)
    hip_half = 0.4 * shoulder_half

    body = np.zeros((14, 3), dtype=np.float32)
    body[:] = np.array([0.5, 0.5, 0.0], dtype=np.float32)
    body[0] = np.array([cx / width, (y1 + 0.45 * (shoulder_y - y1)) / height, 0.95], dtype=np.float32)  # nose
    body[1] = np.array([cx / width, shoulder_y / height, 0.95], dtype=np.float32)  # neck
    body[2] = np.array([(cx - shoulder_half) / width, shoulder_y / height, 0.95], dtype=np.float32)  # right shoulder
    body[5] = np.array([(cx + shoulder_half) / width, shoulder_y / height, 0.95], dtype=np.float32)  # left shoulder
    body[8] = np.array([(cx - hip_half) / width, hip_y / height, 0.95], dtype=np.float32)  # right hip
    body[11] = np.array([(cx + hip_half) / width, hip_y / height, 0.95], dtype=np.float32)  # left hip
    body[10] = np.array([(cx - hip_half * 0.6) / width, foot_y / height, 0.95], dtype=np.float32)  # right ankle
    body[13] = np.array([(cx + hip_half * 0.6) / width, foot_y / height, 0.95], dtype=np.float32)  # left ankle

    face = np.asarray(
        [
            [cx / width, (y1 + 0.08 * person_h) / height, 0.95],
            [cx / width, (y1 + 0.16 * person_h) / height, 0.95],
            [((cx - shoulder_half * 0.35) / width), (y1 + 0.12 * person_h) / height, 0.95],
            [((cx + shoulder_half * 0.35) / width), (y1 + 0.12 * person_h) / height, 0.95],
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


def _structure_error(lhs, rhs):
    keys = [
        ("shoulder_width", 1.0),
        ("head_height", 1.0),
        ("torso_height", 1.2),
        ("leg_height", 1.2),
        ("center_x", 0.6),
        ("foot_y", 0.8),
    ]
    err = 0.0
    for key, weight in keys:
        err += weight * abs(float(lhs[key]) - float(rhs[key]))
    return float(err)


def main():
    canvas_shape = (120, 160)
    driver_metas = [
        _make_pose_meta(
            image_shape=canvas_shape,
            bbox=[52, 12, 108, 114],
            shoulder_width_ratio=0.74,
            head_ratio=0.12,
            torso_ratio=0.27,
            leg_ratio=0.58,
        ),
        _make_pose_meta(
            image_shape=canvas_shape,
            bbox=[54, 12, 110, 115],
            shoulder_width_ratio=0.76,
            head_ratio=0.12,
            torso_ratio=0.26,
            leg_ratio=0.59,
        ),
        _make_pose_meta(
            image_shape=canvas_shape,
            bbox=[53, 13, 107, 114],
            shoulder_width_ratio=0.75,
            head_ratio=0.13,
            torso_ratio=0.26,
            leg_ratio=0.58,
        ),
    ]
    target_bbox, target_bbox_stats = estimate_driver_target_bbox(
        driver_metas,
        image_shape=canvas_shape,
        source="median_first_n",
        num_frames=3,
        conf_thresh=0.35,
    )
    target_structure, target_structure_stats = estimate_driver_target_structure(
        driver_metas,
        image_shape=canvas_shape,
        source="median_first_n",
        num_frames=3,
        conf_thresh=0.35,
    )
    assert target_bbox is not None
    assert target_structure is not None
    assert target_bbox_stats["valid_frames"] == 3
    assert target_structure_stats["valid_frames"] == 3

    original_reference_shape = (320, 200)
    reference_meta_detection = _make_pose_meta(
        image_shape=(160, 100),
        bbox=[28, 8, 72, 152],
        shoulder_width_ratio=0.46,
        head_ratio=0.18,
        torso_ratio=0.38,
        leg_ratio=0.38,
    )
    reference_bbox_detection = bbox_from_pose_meta(
        reference_meta_detection,
        image_shape=(160, 100),
        conf_thresh=0.35,
    )
    reference_structure_detection = structure_from_pose_meta(
        reference_meta_detection,
        image_shape=(160, 100),
        conf_thresh=0.35,
    )
    reference_bbox_original = scale_bbox_between_shapes(
        reference_bbox_detection,
        from_shape=(160, 100),
        to_shape=original_reference_shape,
    )
    reference_structure_original = scale_structure_between_shapes(
        reference_structure_detection,
        from_shape=(160, 100),
        to_shape=original_reference_shape,
    )

    reference_image = np.zeros((original_reference_shape[0], original_reference_shape[1], 3), dtype=np.uint8)
    bbox = np.asarray(reference_bbox_original, dtype=np.int32)
    reference_image[bbox[1]:bbox[3], bbox[0]:bbox[2]] = np.array([255, 64, 64], dtype=np.uint8)

    bbox_reference, bbox_stats = normalize_reference_image(
        reference_image,
        reference_bbox=reference_bbox_original,
        target_bbox=target_bbox,
        canvas_shape=canvas_shape,
        scale_clamp_min=0.75,
        scale_clamp_max=1.6,
    )
    structure_reference, structure_stats = normalize_reference_image_structure_aware(
        reference_image,
        reference_structure=reference_structure_original,
        target_structure=target_structure,
        canvas_shape=canvas_shape,
        scale_clamp_min=0.75,
        scale_clamp_max=1.6,
        segment_clamp_min=0.8,
        segment_clamp_max=1.25,
        width_budget_ratio=1.05,
        height_budget_ratio=1.05,
    )
    assert bbox_reference is not None
    assert structure_reference is not None

    original_letterbox_bbox = project_bbox_with_letterbox(
        reference_bbox_original,
        image_shape=original_reference_shape,
        canvas_shape=canvas_shape,
    )
    original_letterbox_structure = project_structure_with_letterbox(
        reference_structure_original,
        image_shape=original_reference_shape,
        canvas_shape=canvas_shape,
    )

    bbox_norm_bbox = np.asarray(bbox_stats["normalized_bbox"], dtype=np.float32)
    structure_norm_bbox = np.asarray(structure_stats["normalized_bbox"], dtype=np.float32)
    bbox_norm_error = _bbox_error(bbox_norm_bbox, target_bbox)
    structure_bbox_error = _bbox_error(structure_norm_bbox, target_bbox)
    assert structure_bbox_error < _bbox_error(original_letterbox_bbox, target_bbox)
    assert structure_bbox_error < bbox_norm_error

    bbox_norm_structure = {
        "center_x": float((bbox_norm_bbox[0] + bbox_norm_bbox[2]) / 2.0),
        "foot_y": float(bbox_norm_bbox[3]),
        "head_height": float(reference_structure_original["head_height"] * bbox_stats["applied_scale_factor"]),
        "torso_height": float(reference_structure_original["torso_height"] * bbox_stats["applied_scale_factor"]),
        "leg_height": float(reference_structure_original["leg_height"] * bbox_stats["applied_scale_factor"]),
        "shoulder_width": float(reference_structure_original["shoulder_width"] * bbox_stats["applied_scale_factor"]),
    }
    structure_norm_structure = structure_stats["normalized_structure"]
    assert _structure_error(structure_norm_structure, target_structure) < _structure_error(
        bbox_norm_structure,
        target_structure,
    )

    preview = make_reference_normalization_preview(
        original_canvas=np.zeros((canvas_shape[0], canvas_shape[1], 3), dtype=np.uint8),
        normalized_canvas=structure_reference,
        original_bbox=original_letterbox_bbox,
        target_bbox=target_bbox,
        normalized_bbox=structure_norm_bbox,
        original_structure=original_letterbox_structure,
        target_structure=target_structure,
        normalized_structure=structure_norm_structure,
    )
    assert preview.shape[0] == canvas_shape[0]
    assert preview.shape[1] == canvas_shape[1] * 2

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        write_reference_image(root / "src_ref.png", reference_image)
        write_reference_image(root / "src_ref_normalized.png", structure_reference)

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
                "reference_normalization_mode": "structure_match",
                "reference_target_bbox_source": "median_first_n",
                "reference_target_bbox_frames": 3,
                "reference_bbox_conf_thresh": 0.35,
                "reference_scale_clamp_min": 0.75,
                "reference_scale_clamp_max": 1.6,
                "reference_structure_segment_clamp_min": 0.8,
                "reference_structure_segment_clamp_max": 1.25,
                "reference_structure_width_budget_ratio": 1.05,
                "reference_structure_height_budget_ratio": 1.05,
                "stats": structure_stats,
            },
        )
        validate_preprocess_metadata(metadata, src_root_path=root)
        assert metadata["processing"]["reference_normalization"]["reference_normalization_mode"] == "structure_match"
        assert metadata["src_files"]["reference"]["path"] == "src_ref_normalized.png"

    print("Synthetic reference bbox normalization: PASS")
    print("Synthetic structure-aware reference normalization: PASS")
    print("Synthetic reference normalization metadata contract: PASS")


if __name__ == "__main__":
    main()
