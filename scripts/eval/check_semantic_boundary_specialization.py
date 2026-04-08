#!/usr/bin/env python
import json
import tempfile
from pathlib import Path
import sys

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PREPROCESS_DIR = REPO_ROOT / "wan" / "modules" / "animate" / "preprocess"
if str(PREPROCESS_DIR) not in sys.path:
    sys.path.insert(0, str(PREPROCESS_DIR))

from boundary_fusion import build_semantic_boundary_maps
from wan.utils.animate_contract import (
    BACKGROUND_KEEP_PRIOR_SEMANTICS,
    BOUNDARY_BAND_SEMANTICS,
    CLOTH_BOUNDARY_SEMANTICS,
    FACE_BOUNDARY_SEMANTICS,
    HAIR_BOUNDARY_SEMANTICS,
    HAND_BOUNDARY_SEMANTICS,
    OCCLUDED_BOUNDARY_SEMANTICS,
    PERSON_MASK_SEMANTICS,
    build_preprocess_metadata,
    load_preprocess_metadata,
    resolve_preprocess_artifacts,
    validate_loaded_preprocess_bundle,
    write_preprocess_metadata,
)
from wan.utils.boundary_refinement import refine_boundary_frames
from wan.utils.media_io import (
    load_mask_artifact,
    load_person_mask_artifact,
    load_rgb_artifact,
    write_rgb_artifact,
    write_person_mask_artifact,
)
from wan.utils.replacement_masks import compose_background_keep_mask


def _disk(radius_y: float, radius_x: float, center_y: float, center_x: float, height: int, width: int) -> np.ndarray:
    yy, xx = np.mgrid[:height, :width]
    return ((((yy - center_y) / max(radius_y, 1e-6)) ** 2) + (((xx - center_x) / max(radius_x, 1e-6)) ** 2) <= 1.0).astype(
        np.float32
    )


def _make_case(frame_count: int = 6, height: int = 128, width: int = 160) -> dict:
    boundary_band = np.zeros((frame_count, height, width), dtype=np.float32)
    hard_foreground = np.zeros_like(boundary_band)
    soft_alpha = np.zeros_like(boundary_band)
    background_keep_prior = np.ones_like(boundary_band, dtype=np.float32)
    uncertainty = np.zeros_like(boundary_band, dtype=np.float32)
    occlusion_band = np.zeros_like(boundary_band, dtype=np.float32)
    face_alpha = np.zeros_like(boundary_band, dtype=np.float32)
    face_uncertainty = np.zeros_like(boundary_band, dtype=np.float32)
    face_parsing = np.zeros((frame_count, height, width), dtype=np.float32)
    head_prior = np.zeros_like(boundary_band, dtype=np.float32)
    hand_prior = np.zeros_like(boundary_band, dtype=np.float32)
    part_foreground_prior = np.zeros_like(boundary_band, dtype=np.float32)
    occlusion_prior = np.zeros_like(boundary_band, dtype=np.float32)
    fine_boundary = np.zeros_like(boundary_band, dtype=np.float32)
    hair_edge = np.zeros_like(boundary_band, dtype=np.float32)
    pose_uncertainty = np.zeros_like(boundary_band, dtype=np.float32)
    generated = np.zeros((frame_count, height, width, 3), dtype=np.uint8)
    background = np.zeros_like(generated)
    background[..., 1] = 40
    background[..., 2] = 70
    generated[:] = background

    for idx in range(frame_count):
        cx = 80 + idx
        cy = 72
        body = _disk(30, 24, cy, cx, height, width)
        body_ring = np.clip(_disk(34, 28, cy, cx, height, width) - body, 0.0, 1.0)
        hair = np.zeros((height, width), dtype=np.float32)
        hair[22:38, 66:94] = 1.0
        hair[:, 74:76] = np.maximum(hair[:, 74:76], (np.arange(height)[:, None] < 30).astype(np.float32))
        hand = np.zeros((height, width), dtype=np.float32)
        hand[70:78, 40:58] = 1.0
        cloth = np.zeros((height, width), dtype=np.float32)
        cloth[90:106, 56:104] = 1.0
        occ = np.zeros((height, width), dtype=np.float32)
        occ[46:66, 70:92] = 1.0

        hard_foreground[idx] = np.clip(body + hair + hand + cloth, 0.0, 1.0)
        boundary_band[idx] = np.clip(body_ring + 0.8 * hair + 0.7 * hand + 0.6 * cloth, 0.0, 1.0)
        soft_alpha[idx] = np.clip(hard_foreground[idx] + 0.55 * boundary_band[idx], 0.0, 1.0)
        background_keep_prior[idx] = np.clip(1.0 - 0.7 * soft_alpha[idx], 0.0, 1.0)
        uncertainty[idx] = np.clip(0.2 * boundary_band[idx] + 0.4 * occ, 0.0, 1.0)
        occlusion_band[idx] = occ
        face_alpha[idx, 34:60, 64:96] = 1.0
        face_uncertainty[idx, 34:60, 64:96] = 0.1
        face_parsing[idx, 34:60, 64:96] = 1.0
        head_prior[idx, 26:64, 60:100] = 1.0
        hand_prior[idx, 66:82, 36:60] = 1.0
        part_foreground_prior[idx] = np.clip(hard_foreground[idx] - head_prior[idx] - hand_prior[idx], 0.0, 1.0)
        occlusion_prior[idx] = occ
        fine_boundary[idx] = np.clip(boundary_band[idx] + 0.5 * hair + 0.25 * hand, 0.0, 1.0)
        hair_edge[idx] = hair
        pose_uncertainty[idx] = np.clip(0.1 * body_ring + 0.25 * hand + 0.4 * occ, 0.0, 1.0)
        generated[idx, hard_foreground[idx] > 0.5] = np.array([205, 170, 148], dtype=np.uint8)
        generated[idx, hair > 0.5] = np.array([110, 78, 58], dtype=np.uint8)
        generated[idx, hand > 0.5] = np.array([214, 182, 163], dtype=np.uint8)
        generated[idx, cloth > 0.5] = np.array([130, 116, 188], dtype=np.uint8)

    parsing_output = {
        "head_prior": head_prior,
        "hand_prior": hand_prior,
        "part_foreground_prior": part_foreground_prior,
        "occlusion_prior": occlusion_prior,
        "semantic_boundary_prior": boundary_band,
    }
    matting_output = {
        "soft_alpha": soft_alpha,
        "boundary_band": boundary_band,
        "background_keep_prior": background_keep_prior,
        "occlusion_band": occlusion_band,
        "uncertainty_map": uncertainty,
        "fine_boundary_mask": fine_boundary,
        "hair_edge_mask": hair_edge,
    }
    face_analysis = {
        "face_alpha": face_alpha,
        "face_uncertainty": face_uncertainty,
        "face_parsing": face_parsing,
        "stats": {"source_counts": {"synthetic": int(frame_count)}},
    }
    pose_motion = {
        "pose_uncertainty": pose_uncertainty,
        "stats": {"mode": "synthetic"},
    }
    return {
        "generated": generated,
        "background": background,
        "hard_foreground": hard_foreground,
        "boundary_band": boundary_band,
        "soft_alpha": soft_alpha,
        "background_keep_prior": background_keep_prior,
        "uncertainty": uncertainty,
        "occlusion_band": occlusion_band,
        "parsing_output": parsing_output,
        "matting_output": matting_output,
        "face_analysis": face_analysis,
        "pose_motion": pose_motion,
    }


def main():
    case = _make_case()
    semantic = build_semantic_boundary_maps(
        boundary_band=case["boundary_band"],
        hard_foreground=case["hard_foreground"],
        parsing_output=case["parsing_output"],
        matting_output=case["matting_output"],
        face_analysis=case["face_analysis"],
        pose_motion_analysis=case["pose_motion"],
    )

    for key in ("face_boundary", "hair_boundary", "hand_boundary", "cloth_boundary", "occluded_boundary"):
        assert key in semantic, f"missing semantic boundary map: {key}"
        assert semantic[key].shape == case["boundary_band"].shape
        assert float(semantic[key].max()) <= 1.0 and float(semantic[key].min()) >= 0.0

    semantic_sum = (
        semantic["face_boundary"]
        + semantic["hair_boundary"]
        + semantic["hand_boundary"]
        + semantic["cloth_boundary"]
        + semantic["occluded_boundary"]
    )
    active = case["boundary_band"] > 1e-5
    assert float(semantic_sum[active].max()) <= float(case["boundary_band"][active].max()) + 1e-3
    active_ratio = float((semantic_sum[active] > 0.05).mean())
    assert active_ratio >= 0.65
    assert float(semantic["face_boundary"][:, 34:60, 64:96].mean()) > 0.15
    assert float(semantic["hair_boundary"][:, 22:38, 66:94].mean()) > 0.25
    assert float(semantic["hand_boundary"][:, 70:78, 40:58].mean()) > 0.20
    assert float(semantic["occluded_boundary"][:, 46:66, 70:92].max()) > 0.20

    background_keep = compose_background_keep_mask(
        person_mask=case["hard_foreground"],
        soft_band=case["boundary_band"],
        background_keep_prior=case["background_keep_prior"],
        uncertainty_map=case["uncertainty"],
        occlusion_band=case["occlusion_band"],
        face_boundary=semantic["face_boundary"],
        hair_boundary=semantic["hair_boundary"],
        hand_boundary=semantic["hand_boundary"],
        cloth_boundary=semantic["cloth_boundary"],
        occluded_boundary=semantic["occluded_boundary"],
        conditioning_mode="semantic_v1",
        boundary_strength=1.0,
    )
    assert background_keep.shape == case["hard_foreground"].shape
    assert float(background_keep[:, 22:38, 66:94].mean()) < float(background_keep[:, :12, :12].mean())

    refined, debug = refine_boundary_frames(
        generated_frames=case["generated"],
        background_frames=case["background"],
        person_mask=case["hard_foreground"],
        soft_band=case["boundary_band"],
        soft_alpha=case["soft_alpha"],
        background_confidence=np.clip(1.0 - case["uncertainty"], 0.0, 1.0),
        uncertainty_map=case["uncertainty"],
        occlusion_band=case["occlusion_band"],
        face_preserve_map=np.clip(case["face_analysis"]["face_alpha"], 0.0, 1.0),
        face_confidence_map=np.clip(1.0 - case["face_analysis"]["face_uncertainty"], 0.0, 1.0),
        detail_release_map=case["matting_output"]["fine_boundary_mask"],
        trimap_unknown_map=case["boundary_band"],
        edge_detail_map=case["matting_output"]["hair_edge_mask"],
        face_boundary_map=semantic["face_boundary"],
        hair_boundary_map=semantic["hair_boundary"],
        hand_boundary_map=semantic["hand_boundary"],
        cloth_boundary_map=semantic["cloth_boundary"],
        occluded_boundary_map=semantic["occluded_boundary"],
        structure_guard_strength=1.0,
        mode="semantic_v1",
        strength=0.42,
        sharpen=0.24,
    )
    assert refined.shape == case["generated"].shape
    assert "face_boundary_map" in debug and debug["face_boundary_map"].shape == case["boundary_band"].shape
    assert debug["metrics"]["band_gradient_after_mean"] >= debug["metrics"]["band_gradient_before_mean"] * 0.9

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        reference = case["generated"][0].copy()
        ok = cv2.imwrite(str(root / "src_ref.png"), cv2.cvtColor(reference, cv2.COLOR_RGB2BGR))
        assert ok
        pose_artifact = write_rgb_artifact(
            frames=case["generated"],
            output_root=root,
            stem="src_pose",
            artifact_format="npz",
            fps=5.0,
        )
        face_artifact = write_rgb_artifact(
            frames=case["generated"][:, 32:96, 48:112],
            output_root=root,
            stem="src_face",
            artifact_format="npz",
            fps=5.0,
        )
        background_artifact = write_rgb_artifact(
            frames=case["background"],
            output_root=root,
            stem="src_bg",
            artifact_format="npz",
            fps=5.0,
        )
        mask_artifacts = {}
        for stem, frames, semantics in (
            ("src_mask", case["hard_foreground"], PERSON_MASK_SEMANTICS),
            ("src_boundary_band", case["boundary_band"], BOUNDARY_BAND_SEMANTICS),
            ("src_background_keep_prior", case["background_keep_prior"], BACKGROUND_KEEP_PRIOR_SEMANTICS),
            ("src_face_boundary", semantic["face_boundary"], FACE_BOUNDARY_SEMANTICS),
            ("src_hair_boundary", semantic["hair_boundary"], HAIR_BOUNDARY_SEMANTICS),
            ("src_hand_boundary", semantic["hand_boundary"], HAND_BOUNDARY_SEMANTICS),
            ("src_cloth_boundary", semantic["cloth_boundary"], CLOTH_BOUNDARY_SEMANTICS),
            ("src_occluded_boundary", semantic["occluded_boundary"], OCCLUDED_BOUNDARY_SEMANTICS),
        ):
            mask_artifacts[stem] = write_person_mask_artifact(
                mask_frames=frames,
                output_root=root,
                stem=stem,
                artifact_format="npz",
                fps=5.0,
                mask_semantics=semantics,
            )

        metadata = build_preprocess_metadata(
            video_path="video.mp4",
            refer_image_path="ref.png",
            output_path=str(root),
            replace_flag=True,
            retarget_flag=False,
            use_flux=False,
            resolution_area=[case["generated"].shape[2], case["generated"].shape[1]],
            analysis_settings={"analysis_resolution_area": [case["generated"].shape[2], case["generated"].shape[1]]},
            fps_request=5,
            fps_output=5.0,
            frame_count=case["generated"].shape[0],
            height=case["generated"].shape[1],
            width=case["generated"].shape[2],
            iterations=1,
            k=5,
            w_len=8,
            h_len=16,
            reference_height=case["generated"].shape[1],
            reference_width=case["generated"].shape[2],
            src_files={
                "pose": pose_artifact,
                "face": face_artifact,
                "reference": {
                    "path": "src_ref.png",
                    "type": "image",
                    "format": "png",
                    "height": case["generated"].shape[1],
                    "width": case["generated"].shape[2],
                    "channels": 3,
                    "color_space": "rgb",
                    "dtype": "uint8",
                    "shape": [case["generated"].shape[1], case["generated"].shape[2], 3],
                    "resized_height": case["generated"].shape[1],
                    "resized_width": case["generated"].shape[2],
                },
                "background": background_artifact,
                "person_mask": mask_artifacts["src_mask"],
                "boundary_band": mask_artifacts["src_boundary_band"],
                "background_keep_prior": mask_artifacts["src_background_keep_prior"],
                "face_boundary": mask_artifacts["src_face_boundary"],
                "hair_boundary": mask_artifacts["src_hair_boundary"],
                "hand_boundary": mask_artifacts["src_hand_boundary"],
                "cloth_boundary": mask_artifacts["src_cloth_boundary"],
                "occluded_boundary": mask_artifacts["src_occluded_boundary"],
            },
            boundary_fusion_settings={"boundary_specialization_version": "semantic_v1"},
        )
        write_preprocess_metadata(root, metadata)
        loaded = load_preprocess_metadata(root)
        artifacts, _ = resolve_preprocess_artifacts(root, replace_flag=True)
        validate_loaded_preprocess_bundle(
            metadata=loaded,
            cond_images=load_rgb_artifact(artifacts["pose"]["path"], artifacts["pose"]["format"]),
            face_images=load_rgb_artifact(artifacts["face"]["path"], artifacts["face"]["format"]),
            refer_image_rgb=reference,
            bg_images=load_rgb_artifact(artifacts["background"]["path"], artifacts["background"]["format"]),
            person_mask_images=load_person_mask_artifact(root / artifacts["person_mask"]["path"], artifacts["person_mask"]["format"]),
            boundary_band_images=load_mask_artifact(root / artifacts["boundary_band"]["path"], artifacts["boundary_band"]["format"]),
            background_keep_prior_images=load_mask_artifact(
                artifacts["background_keep_prior"]["path"],
                artifacts["background_keep_prior"]["format"],
            ),
            face_boundary_images=load_mask_artifact(artifacts["face_boundary"]["path"], artifacts["face_boundary"]["format"]),
            hair_boundary_images=load_mask_artifact(artifacts["hair_boundary"]["path"], artifacts["hair_boundary"]["format"]),
            hand_boundary_images=load_mask_artifact(artifacts["hand_boundary"]["path"], artifacts["hand_boundary"]["format"]),
            cloth_boundary_images=load_mask_artifact(artifacts["cloth_boundary"]["path"], artifacts["cloth_boundary"]["format"]),
            occluded_boundary_images=load_mask_artifact(
                artifacts["occluded_boundary"]["path"],
                artifacts["occluded_boundary"]["format"],
            ),
        )
        summary = {
            "semantic_boundary_stats": semantic["stats"],
            "background_keep_mean": float(background_keep.mean()),
            "refine_metrics": debug["metrics"],
        }
        (root / "semantic_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("Synthetic semantic boundary specialization: PASS")


if __name__ == "__main__":
    main()
