# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import os
import time
import numpy as np
import torch
import cv2
from loguru import logger
from PIL import Image

try:
    from diffusers import FluxKontextPipeline
except ImportError:  # pragma: no cover - optional dependency
    FluxKontextPipeline = None

try:
    from decord import VideoReader
except ImportError:  # pragma: no cover - optional dependency
    class _CV2Batch:
        def __init__(self, frames):
            self._frames = np.asarray(frames, dtype=np.uint8)

        def asnumpy(self):
            return self._frames

    class VideoReader:  # type: ignore[override]
        def __init__(self, video_path):
            self.video_path = str(video_path)
            capture = cv2.VideoCapture(self.video_path)
            if not capture.isOpened():
                raise FileNotFoundError(f"Failed to open video via cv2 fallback: {self.video_path}")
            self._frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            self._fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            capture.release()
            if self._frame_count <= 0:
                raise ValueError(f"cv2 fallback failed to resolve a valid frame count for {self.video_path}")
            if self._fps <= 1e-6:
                self._fps = 30.0

        def __len__(self):
            return self._frame_count

        def get_avg_fps(self):
            return self._fps

        def get_frame_timestamp(self, index):
            if index < 0:
                index = self._frame_count + index
            index = int(np.clip(index, 0, self._frame_count - 1))
            timestamp = index / float(self._fps)
            return np.array([timestamp, timestamp], dtype=np.float32)

        def get_batch(self, indices):
            capture = cv2.VideoCapture(self.video_path)
            if not capture.isOpened():
                raise FileNotFoundError(f"Failed to reopen video via cv2 fallback: {self.video_path}")
            frames = []
            try:
                for index in indices:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                    ok, frame_bgr = capture.read()
                    if not ok or frame_bgr is None:
                        raise RuntimeError(
                            f"cv2 fallback failed to read frame {index} from {self.video_path}"
                        )
                    frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            finally:
                capture.release()
            return _CV2Batch(frames)
from pose2d import Pose2d
from pose2d_utils import AAPoseMeta
from utils import resize_by_area, get_frame_indices, padding_resize, get_aug_mask, get_mask_body_img
from human_visualization import draw_aapose_by_meta_new
from retarget_pose import get_retarget_pose
from signal_stabilization import (
    make_face_bbox_overlay,
    make_pose_overlay,
    stabilize_face_bboxes,
    stabilize_pose_metas,
    write_json as write_curve_json,
)
from sam_prompting import (
    build_mask_stats,
    make_mask_overlay,
    make_sam_prompts_overlay,
    plan_chunk_prompts,
    write_json as write_mask_json,
    write_prompt_keyframes,
)
from sam_runtime import (
    apply_sam_runtime_profile,
    ensure_trace_dir,
    prompt_entry_trace,
    resolve_sam_runtime_profile,
    write_chunk_trace,
)
from sam_utils import build_sam2_video_predictor
from wan.utils.animate_contract import (
    BACKGROUND_KEEP_PRIOR_SEMANTICS,
    BACKGROUND_CONFIDENCE_SEMANTICS,
    BACKGROUND_SOURCE_PROVENANCE_SEMANTICS,
    BACKGROUND_VISIBLE_SUPPORT_SEMANTICS,
    BOUNDARY_BAND_SEMANTICS,
    HARD_FOREGROUND_SEMANTICS,
    OCCLUSION_BAND_SEMANTICS,
    SOFT_ALPHA_SEMANTICS,
    ALPHA_V2_SEMANTICS,
    TRIMAP_V2_SEMANTICS,
    ALPHA_UNCERTAINTY_V2_SEMANTICS,
    FINE_BOUNDARY_MASK_SEMANTICS,
    HAIR_EDGE_MASK_SEMANTICS,
    ALPHA_CONFIDENCE_SEMANTICS,
    ALPHA_SOURCE_PROVENANCE_SEMANTICS,
    FACE_BOUNDARY_SEMANTICS,
    HAIR_BOUNDARY_SEMANTICS,
    HAND_BOUNDARY_SEMANTICS,
    CLOTH_BOUNDARY_SEMANTICS,
    OCCLUDED_BOUNDARY_SEMANTICS,
    SOFT_BAND_SEMANTICS,
    UNRESOLVED_REGION_SEMANTICS,
    UNCERTAINTY_MAP_SEMANTICS,
    load_image_rgb,
    validate_rgb_video,
)
from wan.utils.media_io import write_person_mask_artifact, write_rgb_artifact
from wan.utils.replacement_masks import build_soft_boundary_band
from boundary_fusion import (
    build_semantic_boundary_maps,
    fuse_boundary_signals,
    make_alpha_hard_compare_preview,
    make_fused_boundary_preview,
    make_uncertainty_heatmap_preview,
)
from background_clean_plate import build_clean_plate_background
from face_analysis import (
    make_face_landmark_overlay,
    make_face_parsing_preview,
    run_face_analysis,
    write_face_analysis_artifacts,
)
from pose_motion_analysis import (
    make_pose_uncertainty_preview,
    run_pose_motion_stack,
    write_pose_motion_artifacts,
)
from matting_adapter import (
    make_alpha_mask_preview,
    make_matting_alpha_preview,
    make_trimap_preview_rgb,
    run_matting_adapter,
)
from multistage_preprocess import (
    fuse_multistage_pose_metas,
    propose_face_roi_bboxes,
    propose_person_roi_bboxes,
    run_roi_pose_refinement,
)
from parsing_adapter import make_parsing_overlay, run_parsing_adapter
from reference_normalization import (
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


class ProcessPipeline():
    def __init__(
        self,
        det_checkpoint_path,
        pose2d_checkpoint_path,
        sam_checkpoint_path,
        flux_kontext_path,
        sam_apply_postprocessing=True,
        sam_runtime_profile="legacy_safe",
        sam_use_flash_attn=None,
        sam_math_kernel_on=None,
        sam_old_gpu_mode=None,
        sam_offload_video_to_cpu=None,
        sam_offload_state_to_cpu=None,
    ):
        self.pose2d = Pose2d(checkpoint=pose2d_checkpoint_path, detector_checkpoint=det_checkpoint_path)
        self.sam_runtime_config = resolve_sam_runtime_profile(
            sam_runtime_profile,
            use_flash_attn=sam_use_flash_attn,
            math_kernel_on=sam_math_kernel_on,
            old_gpu=sam_old_gpu_mode,
            offload_video_to_cpu=sam_offload_video_to_cpu,
            offload_state_to_cpu=sam_offload_state_to_cpu,
        )
        self.sam_apply_postprocessing = bool(sam_apply_postprocessing)

        model_cfg = "sam2_hiera_l.yaml"
        if sam_checkpoint_path is not None:
            apply_sam_runtime_profile(self.sam_runtime_config)
            self.predictor = build_sam2_video_predictor(
                model_cfg,
                sam_checkpoint_path,
                apply_postprocessing=self.sam_apply_postprocessing,
            )
        if flux_kontext_path is not None:
            if FluxKontextPipeline is None:
                raise ImportError(
                    "FluxKontextPipeline requires diffusers to be installed, but diffusers is not available in the current environment."
                )
            self.flux_kontext = FluxKontextPipeline.from_pretrained(flux_kontext_path, torch_dtype=torch.bfloat16).to("cuda")

    def _artifact_format(self, save_format, lossless_intermediate, stem, is_mask=False):
        if not lossless_intermediate:
            return save_format
        if is_mask:
            return "npz"
        if stem == "reference":
            return "png"
        return "png_seq"

    def __call__(
        self,
        video_path,
        refer_image_path,
        output_path,
        resolution_area=[1280, 720],
        analysis_resolution_area=None,
        analysis_min_short_side=None,
        fps=30,
        save_format="mp4",
        lossless_intermediate=False,
        face_conf_thresh=0.45,
        face_min_valid_points=15,
        face_bbox_smooth_method="ema",
        face_bbox_smooth_strength=0.7,
        face_bbox_max_scale_change=1.15,
        face_bbox_max_center_shift=0.04,
        face_bbox_hold_frames=6,
        pose_smooth_method="ema",
        pose_conf_thresh_body=0.5,
        pose_conf_thresh_hand=0.35,
        pose_conf_thresh_face=0.45,
        pose_smooth_strength_body=0.65,
        pose_smooth_strength_hand=0.35,
        pose_smooth_strength_face=0.7,
        pose_max_velocity_body=0.05,
        pose_max_velocity_hand=0.08,
        pose_max_velocity_face=0.04,
        pose_interp_max_gap=3,
        export_qa_visuals=False,
        sam_chunk_len=120,
        sam_keyframes_per_chunk=8,
        sam_prompt_body_conf_thresh=0.35,
        sam_prompt_face_conf_thresh=0.45,
        sam_prompt_hand_conf_thresh=0.35,
        sam_prompt_face_min_points=8,
        sam_prompt_hand_min_points=6,
        sam_use_negative_points=True,
        sam_negative_margin=0.08,
        sam_reprompt_interval=40,
        sam_prompt_mode="points",
        sam_debug_trace=False,
        sam_debug_trace_dir=None,
        soft_mask_mode="soft_band",
        soft_mask_band_width=24,
        soft_mask_blur_kernel=5,
        boundary_fusion_mode="v2",
        parsing_mode="heuristic",
        matting_mode="heuristic",
        parsing_head_expand=1.2,
        parsing_hand_radius_ratio=0.025,
        parsing_boundary_kernel=11,
        matting_trimap_inner_erode=3,
        matting_trimap_outer_dilate=12,
        matting_blur_kernel=5,
        alpha_v2_detail_boost=0.28,
        alpha_v2_shrink_strength=0.34,
        alpha_v2_hair_boost=0.42,
        alpha_v2_hard_threshold=0.68,
        alpha_v2_bilateral_sigma_color=0.12,
        alpha_v2_bilateral_sigma_space=5.0,
        bg_inpaint_mode="none",
        bg_inpaint_method="telea",
        bg_inpaint_mask_expand=16,
        bg_inpaint_radius=5.0,
        bg_temporal_smooth_strength=0.14,
        bg_video_window_radius=4,
        bg_video_min_visible_count=2,
        bg_video_blend_strength=0.7,
        bg_video_global_min_visible_count=3,
        bg_video_confidence_threshold=0.30,
        bg_video_global_blend_strength=0.95,
        bg_video_consistency_scale=18.0,
        reference_normalization_mode="none",
        reference_target_bbox_source="median_first_n",
        reference_target_bbox_frames=16,
        reference_bbox_conf_thresh=0.35,
        reference_scale_clamp_min=0.75,
        reference_scale_clamp_max=1.6,
        reference_structure_segment_clamp_min=0.8,
        reference_structure_segment_clamp_max=1.25,
        reference_structure_width_budget_ratio=1.05,
        reference_structure_height_budget_ratio=1.05,
        multistage_preprocess_mode="none",
        disable_person_roi_refine=False,
        disable_face_roi_refine=False,
        person_roi_stage_resolution_area=None,
        person_roi_stage_min_short_side=None,
        person_roi_expand_ratio=1.18,
        person_roi_min_size_ratio=0.20,
        person_roi_target_long_side=None,
        person_roi_body_conf_thresh=0.35,
        person_roi_hand_conf_thresh=0.25,
        person_roi_face_conf_thresh=0.35,
        person_roi_fuse_weight=0.72,
        person_roi_conf_margin=0.03,
        face_roi_stage_resolution_area=None,
        face_roi_stage_min_short_side=None,
        face_roi_expand_ratio=1.65,
        face_roi_min_size_ratio=0.12,
        face_roi_target_long_side=None,
        face_roi_conf_thresh=0.35,
        face_roi_fuse_weight=0.86,
        face_roi_conf_margin=0.02,
        multistage_pose_extra_smooth=0.10,
        multistage_face_bbox_extra_smooth=0.08,
        face_analysis_mode="heuristic",
        face_tracking_smooth_strength=0.90,
        face_tracking_max_scale_change=1.08,
        face_tracking_max_center_shift=0.016,
        face_tracking_hold_frames=10,
        face_difficulty_expand_ratio=1.20,
        face_rerun_difficulty_threshold=0.48,
        face_alpha_blur_kernel=7,
        pose_motion_stack_mode="v1",
        pose_motion_body_bidirectional_strength=0.86,
        pose_motion_hand_bidirectional_strength=0.82,
        pose_motion_face_bidirectional_strength=0.88,
        pose_motion_local_refine_strength=0.82,
        pose_motion_limb_roi_expand_ratio=1.32,
        pose_motion_hand_roi_expand_ratio=1.75,
        pose_motion_velocity_spike_quantile=0.88,
        pose_motion_uncertainty_blur_kernel=21,
        preprocess_runtime_profile="legacy_safe",
        iterations=3,
        k=7,
        w_len=1,
        h_len=1,
        retarget_flag=False,
        use_flux=False,
        replace_flag=False,
    ):
        if replace_flag:
            runtime_stage_seconds = {}
            overall_start = time.perf_counter()
            self._reset_peak_memory_stats()

            video_reader = VideoReader(video_path)
            frame_num = len(video_reader)
            print('frame_num: {}'.format(frame_num))
            
            video_fps = video_reader.get_avg_fps()
            print('video_fps: {}'.format(video_fps))
            print('fps: {}'.format(fps))

            # TODO: Maybe we can switch to PyAV later, which can get accurate frame num
            duration = video_reader.get_frame_timestamp(-1)[-1]      
            expected_frame_num = int(duration * video_fps + 0.5) 
            ratio = abs((frame_num - expected_frame_num)/frame_num)         
            if ratio > 0.1:
                print("Warning: The difference between the actual number of frames and the expected number of frames is two large")
                frame_num = expected_frame_num

            if fps == -1:
                fps = video_fps

            target_num = int(frame_num / video_fps * fps)
            print('target_num: {}'.format(target_num))
            idxs = get_frame_indices(frame_num, video_fps, target_num, fps)
            resize_start = time.perf_counter()
            raw_frames = video_reader.get_batch(idxs).asnumpy()
            validate_rgb_video("replacement input video", raw_frames)

            export_target_area = int(resolution_area[0] * resolution_area[1])
            if analysis_resolution_area is None:
                analysis_resolution_area = list(resolution_area)
            analysis_target_area = int(analysis_resolution_area[0] * analysis_resolution_area[1])

            export_frames = [
                self._resize_frame_with_analysis_policy(
                    frame,
                    target_area=export_target_area,
                    min_short_side=None,
                    divisor=16,
                )
                for frame in raw_frames
            ]
            height, width = export_frames[0].shape[:2]
            if analysis_target_area == export_target_area and analysis_min_short_side is None:
                analysis_frames = export_frames
            else:
                analysis_frames = [
                    self._resize_frame_with_analysis_policy(
                        frame,
                        target_area=analysis_target_area,
                        min_short_side=analysis_min_short_side,
                        divisor=16,
                    )
                    for frame in raw_frames
            ]
            analysis_height, analysis_width = analysis_frames[0].shape[:2]
            multistage_enabled = multistage_preprocess_mode != "none"
            person_roi_enabled = multistage_enabled and not bool(disable_person_roi_refine)
            face_roi_enabled = multistage_enabled and not bool(disable_face_roi_refine)
            person_stage_frames = None
            face_stage_frames = None
            person_stage_shape = None
            face_stage_shape = None
            if person_roi_enabled:
                person_roi_stage_resolution_area = person_roi_stage_resolution_area or analysis_resolution_area
                person_target_area = int(person_roi_stage_resolution_area[0] * person_roi_stage_resolution_area[1])
                person_stage_frames = [
                    self._resize_frame_with_analysis_policy(
                        frame,
                        target_area=person_target_area,
                        min_short_side=person_roi_stage_min_short_side,
                        divisor=16,
                    )
                    for frame in raw_frames
                ]
                person_stage_shape = person_stage_frames[0].shape[:2]
            if face_roi_enabled:
                face_roi_stage_resolution_area = face_roi_stage_resolution_area or person_roi_stage_resolution_area or analysis_resolution_area
                face_target_area = int(face_roi_stage_resolution_area[0] * face_roi_stage_resolution_area[1])
                face_stage_frames = [
                    self._resize_frame_with_analysis_policy(
                        frame,
                        target_area=face_target_area,
                        min_short_side=face_roi_stage_min_short_side,
                        divisor=16,
                    )
                    for frame in raw_frames
                ]
                face_stage_shape = face_stage_frames[0].shape[:2]
            runtime_stage_seconds["load_and_resize"] = time.perf_counter() - resize_start
            logger.info(f"Processing pose meta")

            pose_start = time.perf_counter()
            raw_pose_metas = self.pose2d(analysis_frames)
            tpl_pose_metas, pose_conf_curve = stabilize_pose_metas(
                raw_pose_metas,
                method=pose_smooth_method,
                pose_conf_thresh_body=pose_conf_thresh_body,
                pose_conf_thresh_hand=pose_conf_thresh_hand,
                pose_conf_thresh_face=pose_conf_thresh_face,
                pose_smooth_strength_body=pose_smooth_strength_body,
                pose_smooth_strength_hand=pose_smooth_strength_hand,
                pose_smooth_strength_face=pose_smooth_strength_face,
                pose_max_velocity_body=pose_max_velocity_body,
                pose_max_velocity_hand=pose_max_velocity_hand,
                pose_max_velocity_face=pose_max_velocity_face,
                pose_interp_max_gap=pose_interp_max_gap,
            )
            face_bboxes, face_bbox_curve = stabilize_face_bboxes(
                tpl_pose_metas,
                image_shape=(analysis_height, analysis_width),
                scale=1.3,
                conf_thresh=face_conf_thresh,
                min_valid_points=face_min_valid_points,
                smooth_method=face_bbox_smooth_method,
                smooth_strength=face_bbox_smooth_strength,
                max_scale_change=face_bbox_max_scale_change,
                max_center_shift=face_bbox_max_center_shift,
                hold_frames=face_bbox_hold_frames,
            )
            global_pose_seconds = time.perf_counter() - pose_start
            runtime_stage_seconds["pose_face_controls_global"] = global_pose_seconds

            multistage_stats = {
                "enabled": bool(multistage_enabled),
                "mode": multistage_preprocess_mode,
                "person_roi_enabled": bool(person_roi_enabled),
                "face_roi_enabled": bool(face_roi_enabled),
                "global_analysis": {
                    "shape": [int(analysis_height), int(analysis_width)],
                    "runtime_sec": float(global_pose_seconds),
                    "peak_memory_gb": float(self._peak_memory_gb()),
                },
                "person_roi_analysis": {
                    "enabled": bool(person_roi_enabled),
                    "shape": None if person_stage_shape is None else [int(person_stage_shape[0]), int(person_stage_shape[1])],
                    "runtime_sec": 0.0,
                    "peak_memory_gb": 0.0,
                    "proposal_stats": {},
                    "refine_stats": {},
                },
                "face_roi_analysis": {
                    "enabled": bool(face_roi_enabled),
                    "shape": None if face_stage_shape is None else [int(face_stage_shape[0]), int(face_stage_shape[1])],
                    "runtime_sec": 0.0,
                    "peak_memory_gb": 0.0,
                    "proposal_stats": {},
                    "refine_stats": {},
                },
                "fusion": {
                    "runtime_sec": 0.0,
                    "stats": {},
                },
            }

            if person_roi_enabled:
                person_stage_start = time.perf_counter()
                person_roi_bboxes, person_roi_proposal_stats = propose_person_roi_bboxes(
                    tpl_pose_metas,
                    body_conf_thresh=person_roi_body_conf_thresh,
                    hand_conf_thresh=person_roi_hand_conf_thresh,
                    face_conf_thresh=person_roi_face_conf_thresh,
                    expand_ratio=person_roi_expand_ratio,
                    min_size_ratio=person_roi_min_size_ratio,
                )
                person_roi_pose_metas, person_roi_refine_stats = run_roi_pose_refinement(
                    pose2d=self.pose2d,
                    stage_frames=person_stage_frames,
                    roi_bboxes_norm=person_roi_bboxes,
                    target_long_side=person_roi_target_long_side,
                )
                fused_pose_metas, fusion_stats = fuse_multistage_pose_metas(
                    tpl_pose_metas,
                    person_metas=person_roi_pose_metas,
                    face_metas=None,
                    person_weight=person_roi_fuse_weight,
                    conf_margin=person_roi_conf_margin,
                )
                tpl_pose_metas, pose_conf_curve = stabilize_pose_metas(
                    fused_pose_metas,
                    method=pose_smooth_method,
                    pose_conf_thresh_body=pose_conf_thresh_body,
                    pose_conf_thresh_hand=pose_conf_thresh_hand,
                    pose_conf_thresh_face=pose_conf_thresh_face,
                    pose_smooth_strength_body=min(0.98, pose_smooth_strength_body + multistage_pose_extra_smooth),
                    pose_smooth_strength_hand=min(0.95, pose_smooth_strength_hand + multistage_pose_extra_smooth * 0.6),
                    pose_smooth_strength_face=min(0.98, pose_smooth_strength_face + multistage_pose_extra_smooth),
                    pose_max_velocity_body=pose_max_velocity_body,
                    pose_max_velocity_hand=pose_max_velocity_hand,
                    pose_max_velocity_face=pose_max_velocity_face,
                    pose_interp_max_gap=pose_interp_max_gap,
                )
                face_bboxes, face_bbox_curve = stabilize_face_bboxes(
                    tpl_pose_metas,
                    image_shape=(analysis_height, analysis_width),
                    scale=1.3,
                    conf_thresh=face_conf_thresh,
                    min_valid_points=face_min_valid_points,
                    smooth_method=face_bbox_smooth_method,
                    smooth_strength=min(0.98, face_bbox_smooth_strength + multistage_face_bbox_extra_smooth),
                    max_scale_change=face_bbox_max_scale_change,
                    max_center_shift=face_bbox_max_center_shift,
                    hold_frames=face_bbox_hold_frames,
                )
                runtime_stage_seconds["pose_face_controls_person_roi"] = time.perf_counter() - person_stage_start
                multistage_stats["person_roi_analysis"].update({
                    "runtime_sec": float(runtime_stage_seconds["pose_face_controls_person_roi"]),
                    "peak_memory_gb": float(self._peak_memory_gb()),
                    "proposal_stats": person_roi_proposal_stats,
                    "refine_stats": person_roi_refine_stats,
                })
                multistage_stats["fusion"]["stats"] = fusion_stats

            if face_roi_enabled:
                face_stage_start = time.perf_counter()
                face_roi_bboxes, face_roi_proposal_stats = propose_face_roi_bboxes(
                    tpl_pose_metas,
                    face_bboxes,
                    image_shape=(analysis_height, analysis_width),
                    expand_ratio=face_roi_expand_ratio,
                    min_size_ratio=face_roi_min_size_ratio,
                    face_conf_thresh=face_roi_conf_thresh,
                )
                face_roi_pose_metas, face_roi_refine_stats = run_roi_pose_refinement(
                    pose2d=self.pose2d,
                    stage_frames=face_stage_frames,
                    roi_bboxes_norm=face_roi_bboxes,
                    target_long_side=face_roi_target_long_side,
                )
                fused_pose_metas, face_fusion_stats = fuse_multistage_pose_metas(
                    tpl_pose_metas,
                    person_metas=None,
                    face_metas=face_roi_pose_metas,
                    face_weight=face_roi_fuse_weight,
                    conf_margin=face_roi_conf_margin,
                )
                tpl_pose_metas, pose_conf_curve = stabilize_pose_metas(
                    fused_pose_metas,
                    method=pose_smooth_method,
                    pose_conf_thresh_body=pose_conf_thresh_body,
                    pose_conf_thresh_hand=pose_conf_thresh_hand,
                    pose_conf_thresh_face=pose_conf_thresh_face,
                    pose_smooth_strength_body=min(0.98, pose_smooth_strength_body + multistage_pose_extra_smooth),
                    pose_smooth_strength_hand=min(0.95, pose_smooth_strength_hand + multistage_pose_extra_smooth * 0.6),
                    pose_smooth_strength_face=min(0.99, pose_smooth_strength_face + multistage_pose_extra_smooth + 0.05),
                    pose_max_velocity_body=pose_max_velocity_body,
                    pose_max_velocity_hand=pose_max_velocity_hand,
                    pose_max_velocity_face=pose_max_velocity_face,
                    pose_interp_max_gap=pose_interp_max_gap,
                )
                face_bboxes, face_bbox_curve = stabilize_face_bboxes(
                    tpl_pose_metas,
                    image_shape=(analysis_height, analysis_width),
                    scale=1.3,
                    conf_thresh=face_conf_thresh,
                    min_valid_points=face_min_valid_points,
                    smooth_method=face_bbox_smooth_method,
                    smooth_strength=min(0.99, face_bbox_smooth_strength + multistage_face_bbox_extra_smooth + 0.05),
                    max_scale_change=face_bbox_max_scale_change,
                    max_center_shift=face_bbox_max_center_shift,
                    hold_frames=face_bbox_hold_frames,
                )
                runtime_stage_seconds["pose_face_controls_face_roi"] = time.perf_counter() - face_stage_start
                multistage_stats["face_roi_analysis"].update({
                    "runtime_sec": float(runtime_stage_seconds["pose_face_controls_face_roi"]),
                    "peak_memory_gb": float(self._peak_memory_gb()),
                    "proposal_stats": face_roi_proposal_stats,
                    "refine_stats": face_roi_refine_stats,
                })
                if multistage_stats["fusion"]["stats"]:
                    multistage_stats["fusion"]["stats"].update(face_fusion_stats)
                else:
                    multistage_stats["fusion"]["stats"] = face_fusion_stats
            multistage_stats["fusion"]["runtime_sec"] = float(
                runtime_stage_seconds.get("pose_face_controls_person_roi", 0.0)
                + runtime_stage_seconds.get("pose_face_controls_face_roi", 0.0)
            )

            face_images = []
            face_source_frames = analysis_frames
            face_source_shape = (analysis_height, analysis_width)
            if face_roi_enabled and face_stage_frames is not None:
                face_source_frames = face_stage_frames
                face_source_shape = face_stage_shape
            elif person_roi_enabled and person_stage_frames is not None:
                face_source_frames = person_stage_frames
                face_source_shape = person_stage_shape
            for idx, face_bbox_for_image in enumerate(face_bboxes):
                x1, x2, y1, y2 = face_bbox_for_image
                if face_source_shape != (analysis_height, analysis_width):
                    scale_x = face_source_shape[1] / analysis_width
                    scale_y = face_source_shape[0] / analysis_height
                    x1 = int(np.floor(x1 * scale_x))
                    x2 = int(np.ceil(x2 * scale_x))
                    y1 = int(np.floor(y1 * scale_y))
                    y2 = int(np.ceil(y2 * scale_y))
                    x1 = max(0, min(x1, face_source_shape[1] - 2))
                    x2 = max(x1 + 2, min(x2, face_source_shape[1]))
                    y1 = max(0, min(y1, face_source_shape[0] - 2))
                    y2 = max(y1 + 2, min(y2, face_source_shape[0]))
                face_image = face_source_frames[idx][y1:y2, x1:x2]
                face_image = cv2.resize(face_image, (512, 512))
                face_images.append(face_image)
            runtime_stage_seconds["pose_face_controls"] = float(
                runtime_stage_seconds["pose_face_controls_global"]
                + runtime_stage_seconds.get("pose_face_controls_person_roi", 0.0)
                + runtime_stage_seconds.get("pose_face_controls_face_roi", 0.0)
            )

            logger.info(f"Processing reference image: {refer_image_path}")
            reference_start = time.perf_counter()
            refer_img = load_image_rgb(refer_image_path)
            src_ref_path = os.path.join(output_path, 'src_ref.png')
            write_reference_image(src_ref_path, refer_img)
            reference_original_height, reference_original_width = refer_img.shape[:2]
            reference_bbox_detection = None
            reference_bbox_original = None
            reference_structure_detection = None
            reference_structure_original = None
            driver_target_bbox = None
            driver_target_structure = None
            driver_target_bbox_export = None
            driver_target_structure_export = None
            driver_target_stats = {
                "source": reference_target_bbox_source,
                "used_frames": 0,
                "valid_frames": 0,
                "frame_indices": [],
            }
            driver_target_structure_stats = {
                "source": reference_target_bbox_source,
                "used_frames": 0,
                "valid_frames": 0,
                "frame_indices": [],
            }
            normalized_reference = None
            reference_normalization = {
                "enabled": reference_normalization_mode != "none",
                "mode": reference_normalization_mode,
                "target_bbox_source": reference_target_bbox_source,
                "target_bbox_frames": int(reference_target_bbox_frames),
                "bbox_conf_thresh": float(reference_bbox_conf_thresh),
                "driver_target_bbox_stats": driver_target_stats,
                "driver_target_structure_stats": driver_target_structure_stats,
                "reference_detection_shape": None,
                "reference_bbox_detection": None,
                "reference_bbox_original": None,
                "reference_structure_detection": None,
                "reference_structure_original": None,
            }
            if reference_normalization_mode in {"bbox_match", "structure_match"}:
                reference_detection_img = self._resize_frame_with_analysis_policy(
                    refer_img,
                    target_area=analysis_target_area,
                    min_short_side=analysis_min_short_side,
                    divisor=16,
                )
                reference_pose_meta = self.pose2d([reference_detection_img])[0]
                reference_bbox_detection = bbox_from_pose_meta(
                    reference_pose_meta,
                    image_shape=reference_detection_img.shape[:2],
                    conf_thresh=reference_bbox_conf_thresh,
                )
                reference_structure_detection = structure_from_pose_meta(
                    reference_pose_meta,
                    image_shape=reference_detection_img.shape[:2],
                    conf_thresh=reference_bbox_conf_thresh,
                )
                reference_bbox_original = scale_bbox_between_shapes(
                    reference_bbox_detection,
                    from_shape=reference_detection_img.shape[:2],
                    to_shape=refer_img.shape[:2],
                )
                reference_structure_original = scale_structure_between_shapes(
                    reference_structure_detection,
                    from_shape=reference_detection_img.shape[:2],
                    to_shape=refer_img.shape[:2],
                )
                driver_target_bbox, driver_target_stats = estimate_driver_target_bbox(
                    tpl_pose_metas,
                    image_shape=(analysis_height, analysis_width),
                    source=reference_target_bbox_source,
                    num_frames=reference_target_bbox_frames,
                    conf_thresh=reference_bbox_conf_thresh,
                )
                driver_target_structure, driver_target_structure_stats = estimate_driver_target_structure(
                    tpl_pose_metas,
                    image_shape=(analysis_height, analysis_width),
                    source=reference_target_bbox_source,
                    num_frames=reference_target_bbox_frames,
                    conf_thresh=reference_bbox_conf_thresh,
                )
                reference_normalization.update({
                    "driver_target_bbox_stats": driver_target_stats,
                    "driver_target_structure_stats": driver_target_structure_stats,
                    "reference_detection_shape": [
                        int(reference_detection_img.shape[0]),
                        int(reference_detection_img.shape[1]),
                    ],
                    "reference_bbox_detection": None if reference_bbox_detection is None else [float(v) for v in reference_bbox_detection.tolist()],
                    "reference_bbox_original": None if reference_bbox_original is None else [float(v) for v in reference_bbox_original.tolist()],
                    "reference_structure_detection": reference_structure_detection,
                    "reference_structure_original": reference_structure_original,
                })
                driver_target_bbox_export = driver_target_bbox
                driver_target_structure_export = driver_target_structure
                if analysis_height != height or analysis_width != width:
                    if driver_target_bbox is not None:
                        driver_target_bbox_export = scale_bbox_between_shapes(
                            driver_target_bbox,
                            from_shape=(analysis_height, analysis_width),
                            to_shape=(height, width),
                        )
                    if driver_target_structure is not None:
                        driver_target_structure_export = scale_structure_between_shapes(
                            driver_target_structure,
                            from_shape=(analysis_height, analysis_width),
                            to_shape=(height, width),
                        )

                if reference_normalization_mode == "structure_match":
                    normalized_reference, normalization_stats = normalize_reference_image_structure_aware(
                        refer_img,
                        reference_structure=reference_structure_original,
                        target_structure=driver_target_structure_export,
                        canvas_shape=(height, width),
                        scale_clamp_min=reference_scale_clamp_min,
                        scale_clamp_max=reference_scale_clamp_max,
                        segment_clamp_min=reference_structure_segment_clamp_min,
                        segment_clamp_max=reference_structure_segment_clamp_max,
                        width_budget_ratio=reference_structure_width_budget_ratio,
                        height_budget_ratio=reference_structure_height_budget_ratio,
                    )
                else:
                    normalized_reference, normalization_stats = normalize_reference_image(
                        refer_img,
                        reference_bbox=reference_bbox_original,
                        target_bbox=driver_target_bbox_export,
                        canvas_shape=(height, width),
                        scale_clamp_min=reference_scale_clamp_min,
                        scale_clamp_max=reference_scale_clamp_max,
                    )
                reference_normalization.update(normalization_stats)
                reference_normalization.update({
                    "driver_target_bbox_export": None if driver_target_bbox_export is None else [float(v) for v in driver_target_bbox_export.tolist()],
                    "driver_target_structure_export": driver_target_structure_export,
                })
                if normalized_reference is not None:
                    normalized_path = os.path.join(output_path, "src_ref_normalized.png")
                    write_reference_image(normalized_path, normalized_reference)
                    active_reference_path = "src_ref_normalized.png"
                    active_reference_image = normalized_reference
                else:
                    active_reference_path = "src_ref.png"
                    active_reference_image = refer_img
            else:
                reference_normalization.update({
                    "applied": False,
                    "reason": "disabled",
                })
                active_reference_path = "src_ref.png"
                active_reference_image = refer_img
            runtime_stage_seconds["reference_conditioning"] = time.perf_counter() - reference_start

            reference_height, reference_width = active_reference_image.shape[:2]
            refer_canvas = active_reference_image
            if refer_canvas.shape[:2] != (height, width):
                refer_canvas = padding_resize(refer_canvas, height, width)
            logger.info(f"Processing template video: {video_path}")
            tpl_retarget_pose_metas = [AAPoseMeta.from_humanapi_meta(meta) for meta in tpl_pose_metas]
            cond_images = []

            for idx, meta in enumerate(tpl_retarget_pose_metas):
                canvas = np.zeros_like(refer_canvas)
                conditioning_image = draw_aapose_by_meta_new(canvas, meta)
                cond_images.append(conditioning_image)
            trace_dir = ensure_trace_dir(output_path=output_path, trace_dir=sam_debug_trace_dir) if sam_debug_trace else None
            if trace_dir is not None:
                write_chunk_trace(
                    trace_dir,
                    -1,
                    {
                        "stage": "session_started",
                        "video_path": video_path,
                        "frame_count": int(len(export_frames)),
                        "frame_shape": list(analysis_frames[0].shape),
                        "analysis_shape": [int(analysis_height), int(analysis_width)],
                        "export_shape": [int(height), int(width)],
                        "sam_runtime_profile": self.sam_runtime_config,
                        "sam_settings": {
                            "sam_chunk_len": int(sam_chunk_len),
                            "sam_keyframes_per_chunk": int(sam_keyframes_per_chunk),
                            "sam_prompt_body_conf_thresh": float(sam_prompt_body_conf_thresh),
                            "sam_prompt_face_conf_thresh": float(sam_prompt_face_conf_thresh),
                            "sam_prompt_hand_conf_thresh": float(sam_prompt_hand_conf_thresh),
                            "sam_prompt_face_min_points": int(sam_prompt_face_min_points),
                            "sam_prompt_hand_min_points": int(sam_prompt_hand_min_points),
                            "sam_use_negative_points": bool(sam_use_negative_points),
                            "sam_negative_margin": float(sam_negative_margin),
                            "sam_reprompt_interval": int(sam_reprompt_interval),
                            "sam_prompt_mode": sam_prompt_mode,
                            "sam_apply_postprocessing": bool(self.sam_apply_postprocessing),
                            "analysis_resolution_area": [int(analysis_resolution_area[0]), int(analysis_resolution_area[1])],
                            "analysis_min_short_side": None if analysis_min_short_side is None else int(analysis_min_short_side),
                        },
                    },
                )
            sam_start = time.perf_counter()
            masks, mask_debug = self.get_mask(
                analysis_frames,
                kp2ds_all=tpl_pose_metas,
                sam_chunk_len=sam_chunk_len,
                sam_keyframes_per_chunk=sam_keyframes_per_chunk,
                sam_prompt_body_conf_thresh=sam_prompt_body_conf_thresh,
                sam_prompt_face_conf_thresh=sam_prompt_face_conf_thresh,
                sam_prompt_hand_conf_thresh=sam_prompt_hand_conf_thresh,
                sam_prompt_face_min_points=sam_prompt_face_min_points,
                sam_prompt_hand_min_points=sam_prompt_hand_min_points,
                sam_use_negative_points=sam_use_negative_points,
                sam_negative_margin=sam_negative_margin,
                sam_reprompt_interval=sam_reprompt_interval,
                sam_prompt_mode=sam_prompt_mode,
                sam_debug_trace=sam_debug_trace,
                sam_debug_trace_dir=trace_dir,
            )
            runtime_stage_seconds["sam_mask_generation"] = time.perf_counter() - sam_start
            analysis_masks = masks
            if analysis_height != height or analysis_width != width:
                masks = self._resize_mask_frames(masks, (height, width))

            mask_post_start = time.perf_counter()
            hole_bg_images = []
            aug_masks = []

            for frame, mask in zip(export_frames, masks):
                if iterations > 0:
                    _, each_mask = get_mask_body_img(frame, mask, iterations=iterations, k=k)
                    each_aug_mask = get_aug_mask(each_mask, w_len=w_len, h_len=h_len)
                else:
                    each_aug_mask = mask

                each_bg_image = frame * (1 - each_aug_mask[:, :, None])
                hole_bg_images.append(each_bg_image)
                aug_masks.append(each_aug_mask)

            face_images = np.stack(face_images).astype(np.uint8)
            cond_images = np.stack(cond_images).astype(np.uint8)
            hole_bg_images = np.stack(hole_bg_images).astype(np.uint8)
            aug_masks = np.stack(aug_masks).astype(np.float32)
            runtime_stage_seconds["mask_postprocess"] = time.perf_counter() - mask_post_start
            soft_band_masks = None
            if soft_mask_mode != "none":
                soft_band_masks = build_soft_boundary_band(
                    aug_masks,
                    band_width=soft_mask_band_width,
                    blur_kernel_size=soft_mask_blur_kernel,
                ).astype(np.float32)
            parsing_start = time.perf_counter()
            parsing_output = run_parsing_adapter(
                frames=np.stack(export_frames).astype(np.uint8),
                hard_mask=aug_masks,
                pose_metas=tpl_pose_metas,
                face_bboxes=face_bboxes,
                mode=parsing_mode,
                face_conf_thresh=face_conf_thresh,
                hand_conf_thresh=sam_prompt_hand_conf_thresh,
                head_expand=parsing_head_expand,
                hand_radius_ratio=parsing_hand_radius_ratio,
                boundary_kernel=parsing_boundary_kernel,
            )
            runtime_stage_seconds["parsing_adapter"] = time.perf_counter() - parsing_start
            matting_start = time.perf_counter()
            matting_output = run_matting_adapter(
                frames=np.stack(export_frames).astype(np.uint8),
                hard_mask=aug_masks,
                soft_band=soft_band_masks,
                parsing_boundary_prior=parsing_output["semantic_boundary_prior"],
                parsing_head_prior=parsing_output["head_prior"],
                parsing_hand_prior=parsing_output["hand_prior"],
                parsing_occlusion_prior=parsing_output["occlusion_prior"],
                parsing_part_foreground_prior=parsing_output["part_foreground_prior"],
                mode=matting_mode,
                trimap_inner_erode=matting_trimap_inner_erode,
                trimap_outer_dilate=matting_trimap_outer_dilate,
                blur_kernel=matting_blur_kernel,
                alpha_v2_detail_boost=alpha_v2_detail_boost,
                alpha_v2_shrink_strength=alpha_v2_shrink_strength,
                alpha_v2_hair_boost=alpha_v2_hair_boost,
                alpha_v2_hard_threshold=alpha_v2_hard_threshold,
                alpha_v2_bilateral_sigma_color=alpha_v2_bilateral_sigma_color,
                alpha_v2_bilateral_sigma_space=alpha_v2_bilateral_sigma_space,
            )
            runtime_stage_seconds["matting_adapter"] = time.perf_counter() - matting_start
            fusion_start = time.perf_counter()
            boundary_fusion = fuse_boundary_signals(
                hard_mask=aug_masks,
                soft_band=soft_band_masks,
                parsing_output=parsing_output,
                matting_output=matting_output,
                mode=boundary_fusion_mode,
            )
            runtime_stage_seconds["boundary_fusion"] = time.perf_counter() - fusion_start
            hard_foreground_masks = boundary_fusion["hard_foreground"].astype(np.float32)
            soft_alpha_masks = boundary_fusion["soft_alpha"].astype(np.float32)
            boundary_band_masks = boundary_fusion["boundary_band"].astype(np.float32)
            occlusion_band_masks = boundary_fusion["occlusion_band"].astype(np.float32)
            uncertainty_map_masks = boundary_fusion["uncertainty_map"].astype(np.float32)
            background_keep_prior_masks = boundary_fusion["background_keep_prior"].astype(np.float32)
            trimap_v2_masks = (
                np.asarray(matting_output["trimap_v2"], dtype=np.float32)
                if matting_output.get("trimap_v2") is not None else None
            )
            alpha_v2_masks = (
                np.asarray(matting_output["alpha_v2"], dtype=np.float32)
                if matting_output.get("alpha_v2") is not None else None
            )
            alpha_uncertainty_v2_masks = (
                np.asarray(matting_output["alpha_uncertainty_v2"], dtype=np.float32)
                if matting_output.get("alpha_uncertainty_v2") is not None else None
            )
            fine_boundary_masks = (
                np.asarray(matting_output["fine_boundary_mask"], dtype=np.float32)
                if matting_output.get("fine_boundary_mask") is not None else None
            )
            hair_edge_masks = (
                np.asarray(matting_output["hair_edge_mask"], dtype=np.float32)
                if matting_output.get("hair_edge_mask") is not None else None
            )
            alpha_confidence_masks = (
                np.asarray(matting_output["alpha_confidence"], dtype=np.float32)
                if matting_output.get("alpha_confidence") is not None else None
            )
            alpha_source_provenance_masks = (
                np.asarray(matting_output["alpha_source_provenance"], dtype=np.float32)
                if matting_output.get("alpha_source_provenance") is not None else None
            )
            pose_motion_start = time.perf_counter()
            pose_motion_analysis = run_pose_motion_stack(
                export_frames=np.stack(export_frames).astype(np.uint8),
                pose_metas=tpl_pose_metas,
                raw_pose_metas=raw_pose_metas,
                image_shape=(height, width),
                occlusion_band=occlusion_band_masks,
                uncertainty_map=uncertainty_map_masks,
                mode=pose_motion_stack_mode,
                body_conf_thresh=pose_conf_thresh_body,
                hand_conf_thresh=pose_conf_thresh_hand,
                face_conf_thresh=pose_conf_thresh_face,
                body_bidirectional_strength=pose_motion_body_bidirectional_strength,
                hand_bidirectional_strength=pose_motion_hand_bidirectional_strength,
                face_bidirectional_strength=pose_motion_face_bidirectional_strength,
                local_refine_strength=pose_motion_local_refine_strength,
                limb_roi_expand_ratio=pose_motion_limb_roi_expand_ratio,
                hand_roi_expand_ratio=pose_motion_hand_roi_expand_ratio,
                velocity_spike_quantile=pose_motion_velocity_spike_quantile,
                uncertainty_blur_kernel=pose_motion_uncertainty_blur_kernel,
            )
            tpl_pose_metas = pose_motion_analysis["optimized_pose_metas"]
            face_bboxes, face_bbox_curve = stabilize_face_bboxes(
                tpl_pose_metas,
                image_shape=(analysis_height, analysis_width),
                scale=1.3,
                conf_thresh=face_conf_thresh,
                min_valid_points=face_min_valid_points,
                smooth_method=face_bbox_smooth_method,
                smooth_strength=min(0.99, face_bbox_smooth_strength + 0.06),
                max_scale_change=face_bbox_max_scale_change,
                max_center_shift=face_bbox_max_center_shift,
                hold_frames=face_bbox_hold_frames,
            )
            tpl_retarget_pose_metas = [AAPoseMeta.from_humanapi_meta(meta) for meta in tpl_pose_metas]
            cond_images = []
            for meta in tpl_retarget_pose_metas:
                canvas = np.zeros_like(refer_canvas)
                conditioning_image = draw_aapose_by_meta_new(canvas, meta)
                cond_images.append(conditioning_image)
            cond_images = np.stack(cond_images).astype(np.uint8)
            runtime_stage_seconds["pose_motion_stack"] = time.perf_counter() - pose_motion_start
            face_analysis = None
            if face_analysis_mode != "none":
                face_analysis_start = time.perf_counter()
                face_analysis = run_face_analysis(
                    export_frames=np.stack(export_frames).astype(np.uint8),
                    face_source_frames=np.stack(face_source_frames).astype(np.uint8),
                    pose_metas=tpl_pose_metas,
                    face_bboxes=face_bboxes,
                    export_shape=(height, width),
                    analysis_shape=(analysis_height, analysis_width),
                    face_source_shape=face_source_shape,
                    hard_foreground=hard_foreground_masks,
                    soft_alpha=soft_alpha_masks,
                    occlusion_band=occlusion_band_masks,
                    uncertainty_map=uncertainty_map_masks,
                    conf_thresh=face_conf_thresh,
                    tracking_smooth_strength=face_tracking_smooth_strength,
                    tracking_max_scale_change=face_tracking_max_scale_change,
                    tracking_max_center_shift=face_tracking_max_center_shift,
                    tracking_hold_frames=face_tracking_hold_frames,
                    difficulty_expand_ratio=face_difficulty_expand_ratio,
                    rerun_difficulty_threshold=face_rerun_difficulty_threshold,
                    alpha_blur_kernel=face_alpha_blur_kernel,
                )
                face_images = np.asarray(face_analysis["face_images"], dtype=np.uint8)
                face_bboxes = [list(map(int, bbox)) for bbox in face_analysis["tracked_bboxes_export"]]
                face_bbox_curve = face_analysis["bbox_curve"]
                runtime_stage_seconds["face_analysis"] = time.perf_counter() - face_analysis_start
            semantic_boundary_maps = build_semantic_boundary_maps(
                boundary_band=boundary_band_masks,
                hard_foreground=hard_foreground_masks,
                parsing_output=parsing_output,
                matting_output=matting_output,
                face_analysis=face_analysis,
                pose_motion_analysis=pose_motion_analysis,
            )
            background_start = time.perf_counter()
            bg_images, background_debug = build_clean_plate_background(
                np.stack(export_frames).astype(np.uint8),
                hard_foreground_masks,
                bg_inpaint_mode=bg_inpaint_mode,
                soft_band=boundary_band_masks,
                background_keep_prior=background_keep_prior_masks,
                bg_inpaint_mask_expand=bg_inpaint_mask_expand,
                bg_inpaint_radius=bg_inpaint_radius,
                bg_inpaint_method=bg_inpaint_method,
                bg_temporal_smooth_strength=bg_temporal_smooth_strength,
                bg_video_window_radius=bg_video_window_radius,
                bg_video_min_visible_count=bg_video_min_visible_count,
                bg_video_blend_strength=bg_video_blend_strength,
                bg_video_global_min_visible_count=bg_video_global_min_visible_count,
                bg_video_confidence_threshold=bg_video_confidence_threshold,
                bg_video_global_blend_strength=bg_video_global_blend_strength,
                bg_video_consistency_scale=bg_video_consistency_scale,
            )
            bg_images = np.stack(bg_images).astype(np.uint8)
            runtime_stage_seconds["background_clean_plate"] = time.perf_counter() - background_start
            diagnostics_start = time.perf_counter()
            write_curve_json(os.path.join(output_path, "face_bbox_curve.json"), face_bbox_curve)
            write_curve_json(os.path.join(output_path, "pose_conf_curve.json"), pose_conf_curve)
            write_mask_json(os.path.join(output_path, "mask_stats.json"), mask_debug["mask_stats"])
            qa_outputs = {
                "face_bbox_curve": "face_bbox_curve.json",
                "pose_conf_curve": "pose_conf_curve.json",
                "mask_stats": "mask_stats.json",
            }
            pose_motion_artifacts = write_pose_motion_artifacts(output_path, pose_motion_analysis, fps, write_json_fn=write_curve_json)
            qa_outputs.update({
                "pose_tracks": pose_motion_artifacts["pose_tracks"]["path"],
                "limb_tracks": pose_motion_artifacts["limb_tracks"]["path"],
                "hand_tracks": pose_motion_artifacts["hand_tracks"]["path"],
                "pose_visibility": pose_motion_artifacts["pose_visibility"]["path"],
                "pose_uncertainty": pose_motion_artifacts["pose_uncertainty"]["path"],
            })
            face_artifacts = {}
            if face_analysis is not None:
                face_artifacts = write_face_analysis_artifacts(output_path, face_analysis, fps, write_json_fn=write_curve_json)
                qa_outputs.update({
                    "face_landmarks": face_artifacts["face_landmarks"]["path"],
                    "face_pose": face_artifacts["face_pose"]["path"],
                    "face_expression": face_artifacts["face_expression"]["path"],
                    "face_alpha": face_artifacts["face_alpha"]["path"],
                    "face_uncertainty": face_artifacts["face_uncertainty"]["path"],
                    "face_parsing": face_artifacts["face_parsing"]["path"],
                })
            runtime_stage_seconds["diagnostic_artifacts"] = time.perf_counter() - diagnostics_start
            if export_qa_visuals:
                qa_start = time.perf_counter()
                face_bbox_overlay_frames = np.stack(export_frames).astype(np.uint8) if face_analysis is not None else np.stack(analysis_frames).astype(np.uint8)
                face_bbox_overlay = make_face_bbox_overlay(face_bbox_overlay_frames, face_bboxes, face_bbox_curve)
                pose_overlay = make_pose_overlay(np.stack(export_frames).astype(np.uint8), tpl_pose_metas)
                qa_face_overlay = write_rgb_artifact(
                    frames=face_bbox_overlay,
                    output_root=output_path,
                    stem="face_bbox_overlay",
                    artifact_format="mp4",
                    fps=fps,
                )
                qa_pose_overlay = write_rgb_artifact(
                    frames=pose_overlay,
                    output_root=output_path,
                    stem="pose_overlay",
                    artifact_format="mp4",
                    fps=fps,
                )
                mask_overlay = make_mask_overlay(np.stack(analysis_frames).astype(np.uint8), analysis_masks, mask_debug["prompt_entries"])
                prompt_overlay = make_sam_prompts_overlay(np.stack(analysis_frames).astype(np.uint8), mask_debug["prompt_entries"])
                qa_mask_overlay = write_rgb_artifact(
                    frames=mask_overlay,
                    output_root=output_path,
                    stem="mask_overlay",
                    artifact_format="mp4",
                    fps=fps,
                )
                qa_prompt_overlay = write_rgb_artifact(
                    frames=prompt_overlay,
                    output_root=output_path,
                    stem="sam_prompts_overlay",
                    artifact_format="mp4",
                    fps=fps,
                )
                prompt_keyframes = write_prompt_keyframes(output_path, np.stack(analysis_frames).astype(np.uint8), mask_debug["prompt_entries"])
                qa_outputs = {
                    **qa_outputs,
                    "face_bbox_overlay": qa_face_overlay["path"],
                    "pose_overlay": qa_pose_overlay["path"],
                    "mask_overlay": qa_mask_overlay["path"],
                    "sam_prompts_overlay": qa_prompt_overlay["path"],
                    "sam_prompt_keyframes": prompt_keyframes["path"],
                }
                qa_pose_uncertainty = write_person_mask_artifact(
                    mask_frames=pose_motion_analysis["pose_uncertainty"],
                    output_root=output_path,
                    stem="pose_uncertainty_overlay",
                    artifact_format="mp4",
                    fps=fps,
                    mask_semantics="pose_uncertainty",
                )
                qa_pose_uncertainty_heatmap = write_rgb_artifact(
                    frames=make_pose_uncertainty_preview(pose_motion_analysis["pose_uncertainty"]),
                    output_root=output_path,
                    stem="pose_uncertainty_heatmap",
                    artifact_format="mp4",
                    fps=fps,
                )
                qa_outputs.update({
                    "pose_uncertainty_overlay": qa_pose_uncertainty["path"],
                    "pose_uncertainty_heatmap": qa_pose_uncertainty_heatmap["path"],
                })
                if face_analysis is not None:
                    qa_face_landmark_overlay = write_rgb_artifact(
                        frames=make_face_landmark_overlay(np.stack(export_frames).astype(np.uint8), face_bbox_curve, face_analysis["landmarks_json"], face_analysis["head_pose_json"]),
                        output_root=output_path,
                        stem="face_landmark_overlay",
                        artifact_format="mp4",
                        fps=fps,
                    )
                    qa_face_alpha = write_person_mask_artifact(
                        mask_frames=face_analysis["face_alpha"],
                        output_root=output_path,
                        stem="face_alpha_overlay",
                        artifact_format="mp4",
                        fps=fps,
                        mask_semantics="face_alpha",
                    )
                    qa_face_uncertainty = write_person_mask_artifact(
                        mask_frames=face_analysis["face_uncertainty"],
                        output_root=output_path,
                        stem="face_uncertainty_overlay",
                        artifact_format="mp4",
                        fps=fps,
                        mask_semantics="face_uncertainty",
                    )
                    qa_face_parsing = write_rgb_artifact(
                        frames=make_face_parsing_preview(face_analysis["face_parsing"]),
                        output_root=output_path,
                        stem="face_parsing_overlay",
                        artifact_format="mp4",
                        fps=fps,
                    )
                    qa_outputs.update({
                        "face_landmark_overlay": qa_face_landmark_overlay["path"],
                        "face_alpha_overlay": qa_face_alpha["path"],
                        "face_uncertainty_overlay": qa_face_uncertainty["path"],
                        "face_parsing_overlay": qa_face_parsing["path"],
                    })
                if soft_band_masks is not None:
                    qa_soft_band_overlay = write_person_mask_artifact(
                        mask_frames=soft_band_masks,
                        output_root=output_path,
                        stem="soft_band_overlay",
                        artifact_format="mp4",
                        fps=fps,
                        mask_semantics=SOFT_BAND_SEMANTICS,
                    )
                    qa_outputs["soft_band_overlay"] = qa_soft_band_overlay["path"]
                parsing_overlay = make_parsing_overlay(np.stack(export_frames).astype(np.uint8), parsing_output)
                qa_parsing_overlay = write_rgb_artifact(
                    frames=parsing_overlay,
                    output_root=output_path,
                    stem="parsing_overlay",
                    artifact_format="mp4",
                    fps=fps,
                )
                qa_matting_alpha = write_rgb_artifact(
                    frames=make_matting_alpha_preview(soft_alpha_masks),
                    output_root=output_path,
                    stem="matting_alpha_preview",
                    artifact_format="mp4",
                    fps=fps,
                )
                qa_fused_boundary = write_rgb_artifact(
                    frames=make_fused_boundary_preview(np.stack(export_frames).astype(np.uint8), boundary_fusion),
                    output_root=output_path,
                    stem="fused_boundary_preview",
                    artifact_format="mp4",
                    fps=fps,
                )
                qa_hard_foreground = write_person_mask_artifact(
                    mask_frames=hard_foreground_masks,
                    output_root=output_path,
                    stem="hard_foreground_overlay",
                    artifact_format="mp4",
                    fps=fps,
                    mask_semantics=HARD_FOREGROUND_SEMANTICS,
                )
                qa_soft_alpha = write_person_mask_artifact(
                    mask_frames=soft_alpha_masks,
                    output_root=output_path,
                    stem="soft_alpha_overlay",
                    artifact_format="mp4",
                    fps=fps,
                    mask_semantics=SOFT_ALPHA_SEMANTICS,
                )
                qa_boundary_band = write_person_mask_artifact(
                    mask_frames=boundary_band_masks,
                    output_root=output_path,
                    stem="boundary_band_overlay",
                    artifact_format="mp4",
                    fps=fps,
                    mask_semantics=BOUNDARY_BAND_SEMANTICS,
                )
                qa_background_prior = write_person_mask_artifact(
                    mask_frames=background_keep_prior_masks,
                    output_root=output_path,
                    stem="background_keep_prior_overlay",
                    artifact_format="mp4",
                    fps=fps,
                    mask_semantics=BACKGROUND_KEEP_PRIOR_SEMANTICS,
                )
                qa_occlusion_band = write_person_mask_artifact(
                    mask_frames=occlusion_band_masks,
                    output_root=output_path,
                    stem="occlusion_band_overlay",
                    artifact_format="mp4",
                    fps=fps,
                    mask_semantics=OCCLUSION_BAND_SEMANTICS,
                )
                qa_uncertainty_map = write_person_mask_artifact(
                    mask_frames=uncertainty_map_masks,
                    output_root=output_path,
                    stem="uncertainty_map_overlay",
                    artifact_format="mp4",
                    fps=fps,
                    mask_semantics=UNCERTAINTY_MAP_SEMANTICS,
                )
                qa_uncertainty_heatmap = write_rgb_artifact(
                    frames=make_uncertainty_heatmap_preview(uncertainty_map_masks),
                    output_root=output_path,
                    stem="uncertainty_heatmap_preview",
                    artifact_format="mp4",
                    fps=fps,
                )
                qa_alpha_hard_compare = write_rgb_artifact(
                    frames=make_alpha_hard_compare_preview(hard_foreground_masks, soft_alpha_masks),
                    output_root=output_path,
                    stem="alpha_hard_compare_preview",
                    artifact_format="mp4",
                    fps=fps,
                )
                qa_outputs.update({
                    "parsing_overlay": qa_parsing_overlay["path"],
                    "matting_alpha_preview": qa_matting_alpha["path"],
                    "fused_boundary_preview": qa_fused_boundary["path"],
                    "hard_foreground_overlay": qa_hard_foreground["path"],
                    "soft_alpha_overlay": qa_soft_alpha["path"],
                    "boundary_band_overlay": qa_boundary_band["path"],
                    "background_keep_prior_overlay": qa_background_prior["path"],
                    "occlusion_band_overlay": qa_occlusion_band["path"],
                    "uncertainty_map_overlay": qa_uncertainty_map["path"],
                    "uncertainty_heatmap_preview": qa_uncertainty_heatmap["path"],
                    "alpha_hard_compare_preview": qa_alpha_hard_compare["path"],
                })
                if trimap_v2_masks is not None:
                    qa_trimap_v2 = write_rgb_artifact(
                        frames=make_trimap_preview_rgb(trimap_v2_masks),
                        output_root=output_path,
                        stem="trimap_v2_preview",
                        artifact_format="mp4",
                        fps=fps,
                    )
                    qa_outputs["trimap_v2_preview"] = qa_trimap_v2["path"]
                if alpha_v2_masks is not None:
                    qa_alpha_v2 = write_rgb_artifact(
                        frames=make_matting_alpha_preview(alpha_v2_masks),
                        output_root=output_path,
                        stem="alpha_v2_preview",
                        artifact_format="mp4",
                        fps=fps,
                    )
                    qa_outputs["alpha_v2_preview"] = qa_alpha_v2["path"]
                if alpha_uncertainty_v2_masks is not None:
                    qa_alpha_uncertainty_v2 = write_person_mask_artifact(
                        mask_frames=alpha_uncertainty_v2_masks,
                        output_root=output_path,
                        stem="alpha_uncertainty_v2_overlay",
                        artifact_format="mp4",
                        fps=fps,
                        mask_semantics=UNCERTAINTY_MAP_SEMANTICS,
                    )
                    qa_outputs["alpha_uncertainty_v2_overlay"] = qa_alpha_uncertainty_v2["path"]
                if fine_boundary_masks is not None:
                    qa_fine_boundary = write_rgb_artifact(
                        frames=make_alpha_mask_preview(fine_boundary_masks),
                        output_root=output_path,
                        stem="fine_boundary_mask_preview",
                        artifact_format="mp4",
                        fps=fps,
                    )
                    qa_outputs["fine_boundary_mask_preview"] = qa_fine_boundary["path"]
                if hair_edge_masks is not None:
                    qa_hair_edge = write_rgb_artifact(
                        frames=make_alpha_mask_preview(hair_edge_masks),
                        output_root=output_path,
                        stem="hair_edge_mask_preview",
                        artifact_format="mp4",
                        fps=fps,
                    )
                    qa_outputs["hair_edge_mask_preview"] = qa_hair_edge["path"]
                for semantic_name in (
                    "face_boundary",
                    "hair_boundary",
                    "hand_boundary",
                    "cloth_boundary",
                    "occluded_boundary",
                ):
                    semantic_frames = semantic_boundary_maps.get(semantic_name)
                    if semantic_frames is None:
                        continue
                    qa_semantic = write_rgb_artifact(
                        frames=make_alpha_mask_preview(semantic_frames),
                        output_root=output_path,
                        stem=f"{semantic_name}_preview",
                        artifact_format=self._artifact_format(save_format, lossless_intermediate, f"{semantic_name}_preview"),
                        fps=fps,
                    )
                    qa_outputs[f"{semantic_name}_preview"] = qa_semantic["path"]
                qa_background_hole = write_rgb_artifact(
                    frames=background_debug["hole_background"].astype(np.uint8),
                    output_root=output_path,
                    stem="background_hole",
                    artifact_format="mp4",
                    fps=fps,
                )
                qa_background_clean = write_rgb_artifact(
                    frames=background_debug["clean_plate_background"].astype(np.uint8),
                    output_root=output_path,
                    stem="background_clean_plate",
                    artifact_format="mp4",
                    fps=fps,
                )
                qa_background_diff = write_rgb_artifact(
                    frames=background_debug["background_diff"].astype(np.uint8),
                    output_root=output_path,
                    stem="background_diff",
                    artifact_format="mp4",
                    fps=fps,
                )
                qa_background_temporal_diff = write_rgb_artifact(
                    frames=background_debug["background_temporal_diff"].astype(np.uint8),
                    output_root=output_path,
                    stem="background_temporal_diff",
                    artifact_format="mp4",
                    fps=fps,
                )
                qa_background_mask = write_person_mask_artifact(
                    mask_frames=background_debug["inpaint_mask"].astype(np.float32),
                    output_root=output_path,
                    stem="background_inpaint_mask",
                    artifact_format="mp4",
                    fps=fps,
                    mask_semantics="background_inpaint_region",
                )
                qa_background_support = write_person_mask_artifact(
                    mask_frames=background_debug["visible_support_map"].astype(np.float32),
                    output_root=output_path,
                    stem="background_visible_support",
                    artifact_format="mp4",
                    fps=fps,
                    mask_semantics=BACKGROUND_VISIBLE_SUPPORT_SEMANTICS,
                )
                qa_background_unresolved = write_person_mask_artifact(
                    mask_frames=background_debug["unresolved_region"].astype(np.float32),
                    output_root=output_path,
                    stem="background_unresolved_region",
                    artifact_format="mp4",
                    fps=fps,
                    mask_semantics=UNRESOLVED_REGION_SEMANTICS,
                )
                qa_background_confidence = write_person_mask_artifact(
                    mask_frames=background_debug["background_confidence"].astype(np.float32),
                    output_root=output_path,
                    stem="background_confidence",
                    artifact_format="mp4",
                    fps=fps,
                    mask_semantics=BACKGROUND_CONFIDENCE_SEMANTICS,
                )
                qa_background_provenance = write_person_mask_artifact(
                    mask_frames=background_debug["background_source_provenance"].astype(np.float32),
                    output_root=output_path,
                    stem="background_source_provenance",
                    artifact_format="mp4",
                    fps=fps,
                    mask_semantics=BACKGROUND_SOURCE_PROVENANCE_SEMANTICS,
                )
                qa_outputs.update({
                    "background_hole": qa_background_hole["path"],
                    "background_clean_plate": qa_background_clean["path"],
                    "background_diff": qa_background_diff["path"],
                    "background_temporal_diff": qa_background_temporal_diff["path"],
                    "background_inpaint_mask": qa_background_mask["path"],
                    "background_visible_support": qa_background_support["path"],
                    "background_unresolved_region": qa_background_unresolved["path"],
                    "background_confidence": qa_background_confidence["path"],
                    "background_source_provenance": qa_background_provenance["path"],
                })
                if background_debug.get("clean_plate_image_background") is not None:
                    qa_background_image = write_rgb_artifact(
                        frames=background_debug["clean_plate_image_background"].astype(np.uint8),
                        output_root=output_path,
                        stem="background_clean_plate_image",
                        artifact_format="mp4",
                        fps=fps,
                    )
                    qa_outputs["background_clean_plate_image"] = qa_background_image["path"]
                if background_debug.get("clean_plate_video_background") is not None:
                    qa_background_video = write_rgb_artifact(
                        frames=background_debug["clean_plate_video_background"].astype(np.uint8),
                        output_root=output_path,
                        stem="background_clean_plate_video",
                        artifact_format="mp4",
                        fps=fps,
                    )
                    qa_outputs["background_clean_plate_video"] = qa_background_video["path"]
                if background_debug.get("clean_plate_video_v2_background") is not None:
                    qa_background_video_v2 = write_rgb_artifact(
                        frames=background_debug["clean_plate_video_v2_background"].astype(np.uint8),
                        output_root=output_path,
                        stem="background_clean_plate_video_v2",
                        artifact_format="mp4",
                        fps=fps,
                    )
                    qa_outputs["background_clean_plate_video_v2"] = qa_background_video_v2["path"]
                if reference_normalization_mode != "none":
                    original_canvas = padding_resize(refer_img, height, width)
                    original_bbox_canvas = project_bbox_with_letterbox(
                        reference_bbox_original,
                        image_shape=refer_img.shape[:2],
                        canvas_shape=(height, width),
                    )
                    original_structure_canvas = project_structure_with_letterbox(
                        reference_structure_original,
                        image_shape=refer_img.shape[:2],
                        canvas_shape=(height, width),
                    )
                    normalized_bbox = reference_normalization.get("normalized_bbox")
                    preview = make_reference_normalization_preview(
                        original_canvas=original_canvas,
                        normalized_canvas=refer_canvas,
                        original_bbox=original_bbox_canvas,
                        target_bbox=driver_target_bbox_export,
                        normalized_bbox=normalized_bbox,
                        original_structure=original_structure_canvas,
                        target_structure=driver_target_structure_export,
                        normalized_structure=reference_normalization.get("normalized_structure"),
                    )
                    preview_path = os.path.join(output_path, "reference_normalization_preview.png")
                    write_reference_image(preview_path, preview)
                    qa_outputs["reference_normalization_preview"] = "reference_normalization_preview.png"
                runtime_stage_seconds["qa_artifacts"] = time.perf_counter() - qa_start

            write_start = time.perf_counter()
            src_files = {
                "pose": write_rgb_artifact(
                    frames=cond_images,
                    output_root=output_path,
                    stem="src_pose",
                    artifact_format=self._artifact_format(save_format, lossless_intermediate, "pose"),
                    fps=fps,
                ),
                "face": write_rgb_artifact(
                    frames=face_images,
                    output_root=output_path,
                    stem="src_face",
                    artifact_format=self._artifact_format(save_format, lossless_intermediate, "face"),
                    fps=fps,
                ),
                "background": write_rgb_artifact(
                    frames=bg_images,
                    output_root=output_path,
                    stem="src_bg",
                    artifact_format=self._artifact_format(save_format, lossless_intermediate, "background"),
                    fps=fps,
                ),
                "person_mask": write_person_mask_artifact(
                    mask_frames=hard_foreground_masks,
                    output_root=output_path,
                    stem="src_mask",
                    artifact_format=self._artifact_format(save_format, lossless_intermediate, "person_mask", is_mask=True),
                    fps=fps,
                ),
                "reference": {
                    "path": active_reference_path,
                    "type": "image",
                    "format": "png",
                    "height": int(reference_height),
                    "width": int(reference_width),
                    "channels": 3,
                    "color_space": "rgb",
                    "dtype": "uint8",
                    "shape": [int(reference_height), int(reference_width), 3],
                    "resized_height": int(height),
                    "resized_width": int(width),
                },
            }
            src_files.update(pose_motion_artifacts)
            if face_artifacts:
                src_files.update(face_artifacts)
            src_files["hard_foreground"] = {
                **src_files["person_mask"],
                "mask_semantics": HARD_FOREGROUND_SEMANTICS,
            }
            if active_reference_path != "src_ref.png":
                src_files["reference_original"] = {
                    "path": "src_ref.png",
                    "type": "image",
                    "format": "png",
                    "height": int(reference_original_height),
                    "width": int(reference_original_width),
                    "channels": 3,
                    "color_space": "rgb",
                    "dtype": "uint8",
                    "shape": [int(reference_original_height), int(reference_original_width), 3],
                    "resized_height": int(height),
                    "resized_width": int(width),
                }
            if soft_band_masks is not None:
                src_files["soft_band"] = write_person_mask_artifact(
                    mask_frames=soft_band_masks,
                    output_root=output_path,
                    stem="src_soft_band",
                    artifact_format=self._artifact_format(save_format, lossless_intermediate, "soft_band", is_mask=True),
                    fps=fps,
                    mask_semantics=SOFT_BAND_SEMANTICS,
                )
            src_files["boundary_band"] = write_person_mask_artifact(
                mask_frames=boundary_band_masks,
                output_root=output_path,
                stem="src_boundary_band",
                artifact_format=self._artifact_format(save_format, lossless_intermediate, "boundary_band", is_mask=True),
                fps=fps,
                mask_semantics=BOUNDARY_BAND_SEMANTICS,
            )
            src_files["boundary_band"].update({
                "source_models": ["sam2", parsing_output["mode"], matting_output["mode"]],
                "fusion_version": boundary_fusion_mode,
                "confidence_summary": {
                    "mean": float(boundary_band_masks.mean()),
                    "p80": float(np.quantile(boundary_band_masks, 0.8)),
                    "p95": float(np.quantile(boundary_band_masks, 0.95)),
                },
            })
            src_files["occlusion_band"] = write_person_mask_artifact(
                mask_frames=occlusion_band_masks,
                output_root=output_path,
                stem="src_occlusion_band",
                artifact_format=self._artifact_format(save_format, lossless_intermediate, "occlusion_band", is_mask=True),
                fps=fps,
                mask_semantics=OCCLUSION_BAND_SEMANTICS,
            )
            src_files["occlusion_band"].update({
                "source_models": ["sam2", parsing_output["mode"], matting_output["mode"]],
                "fusion_version": boundary_fusion_mode,
                "confidence_summary": {
                    "mean": float(occlusion_band_masks.mean()),
                    "p80": float(np.quantile(occlusion_band_masks, 0.8)),
                    "p95": float(np.quantile(occlusion_band_masks, 0.95)),
                },
            })
            src_files["uncertainty_map"] = write_person_mask_artifact(
                mask_frames=uncertainty_map_masks,
                output_root=output_path,
                stem="src_uncertainty_map",
                artifact_format=self._artifact_format(save_format, lossless_intermediate, "uncertainty_map", is_mask=True),
                fps=fps,
                mask_semantics=UNCERTAINTY_MAP_SEMANTICS,
            )
            src_files["uncertainty_map"].update({
                "source_models": ["sam2", parsing_output["mode"], matting_output["mode"]],
                "fusion_version": boundary_fusion_mode,
                "confidence_summary": {
                    "mean": float(uncertainty_map_masks.mean()),
                    "p80": float(np.quantile(uncertainty_map_masks, 0.8)),
                    "p95": float(np.quantile(uncertainty_map_masks, 0.95)),
                },
            })
            src_files["soft_alpha"] = write_person_mask_artifact(
                mask_frames=soft_alpha_masks,
                output_root=output_path,
                stem="src_soft_alpha",
                artifact_format=self._artifact_format(save_format, lossless_intermediate, "soft_alpha", is_mask=True),
                fps=fps,
                mask_semantics=SOFT_ALPHA_SEMANTICS,
            )
            src_files["soft_alpha"].update({
                "source_models": ["sam2", parsing_output["mode"], matting_output["mode"]],
                "fusion_version": boundary_fusion_mode,
                "confidence_summary": {
                    "mean": float(soft_alpha_masks.mean()),
                    "p80": float(np.quantile(soft_alpha_masks, 0.8)),
                    "p95": float(np.quantile(soft_alpha_masks, 0.95)),
                },
            })
            if alpha_v2_masks is not None:
                src_files["alpha_v2"] = write_person_mask_artifact(
                    mask_frames=alpha_v2_masks,
                    output_root=output_path,
                    stem="src_alpha_v2",
                    artifact_format=self._artifact_format(save_format, lossless_intermediate, "alpha_v2", is_mask=True),
                    fps=fps,
                    mask_semantics=ALPHA_V2_SEMANTICS,
                )
            if trimap_v2_masks is not None:
                src_files["trimap_v2"] = write_person_mask_artifact(
                    mask_frames=trimap_v2_masks,
                    output_root=output_path,
                    stem="src_trimap_v2",
                    artifact_format=self._artifact_format(save_format, lossless_intermediate, "trimap_v2", is_mask=True),
                    fps=fps,
                    mask_semantics=TRIMAP_V2_SEMANTICS,
                )
            if alpha_uncertainty_v2_masks is not None:
                src_files["alpha_uncertainty_v2"] = write_person_mask_artifact(
                    mask_frames=alpha_uncertainty_v2_masks,
                    output_root=output_path,
                    stem="src_alpha_uncertainty_v2",
                    artifact_format=self._artifact_format(save_format, lossless_intermediate, "alpha_uncertainty_v2", is_mask=True),
                    fps=fps,
                    mask_semantics=ALPHA_UNCERTAINTY_V2_SEMANTICS,
                )
            if fine_boundary_masks is not None:
                src_files["fine_boundary_mask"] = write_person_mask_artifact(
                    mask_frames=fine_boundary_masks,
                    output_root=output_path,
                    stem="src_fine_boundary_mask",
                    artifact_format=self._artifact_format(save_format, lossless_intermediate, "fine_boundary_mask", is_mask=True),
                    fps=fps,
                    mask_semantics=FINE_BOUNDARY_MASK_SEMANTICS,
                )
            if hair_edge_masks is not None:
                src_files["hair_edge_mask"] = write_person_mask_artifact(
                    mask_frames=hair_edge_masks,
                    output_root=output_path,
                    stem="src_hair_edge_mask",
                    artifact_format=self._artifact_format(save_format, lossless_intermediate, "hair_edge_mask", is_mask=True),
                    fps=fps,
                    mask_semantics=HAIR_EDGE_MASK_SEMANTICS,
                )
            semantic_boundary_semantics = {
                "face_boundary": FACE_BOUNDARY_SEMANTICS,
                "hair_boundary": HAIR_BOUNDARY_SEMANTICS,
                "hand_boundary": HAND_BOUNDARY_SEMANTICS,
                "cloth_boundary": CLOTH_BOUNDARY_SEMANTICS,
                "occluded_boundary": OCCLUDED_BOUNDARY_SEMANTICS,
            }
            for semantic_name, semantics in semantic_boundary_semantics.items():
                semantic_frames = semantic_boundary_maps.get(semantic_name)
                if semantic_frames is None:
                    continue
                src_files[semantic_name] = write_person_mask_artifact(
                    mask_frames=semantic_frames,
                    output_root=output_path,
                    stem=f"src_{semantic_name}",
                    artifact_format=self._artifact_format(save_format, lossless_intermediate, semantic_name, is_mask=True),
                    fps=fps,
                    mask_semantics=semantics,
                )
                src_files[semantic_name].update({
                    "source_models": [
                        parsing_output["mode"],
                        matting_output["mode"],
                        face_analysis["stats"]["source_counts"] if face_analysis is not None else "none",
                    ],
                    "boundary_specialization_version": "semantic_v1",
                    "confidence_summary": {
                        "mean": float(np.asarray(semantic_frames, dtype=np.float32).mean()),
                        "p80": float(np.quantile(semantic_frames, 0.8)),
                        "p95": float(np.quantile(semantic_frames, 0.95)),
                    },
                })
            if alpha_confidence_masks is not None:
                src_files["alpha_confidence_v2"] = write_person_mask_artifact(
                    mask_frames=alpha_confidence_masks,
                    output_root=output_path,
                    stem="src_alpha_confidence_v2",
                    artifact_format=self._artifact_format(save_format, lossless_intermediate, "alpha_confidence_v2", is_mask=True),
                    fps=fps,
                    mask_semantics=ALPHA_CONFIDENCE_SEMANTICS,
                )
            if alpha_source_provenance_masks is not None:
                src_files["alpha_source_provenance_v2"] = write_person_mask_artifact(
                    mask_frames=alpha_source_provenance_masks,
                    output_root=output_path,
                    stem="src_alpha_source_provenance_v2",
                    artifact_format=self._artifact_format(save_format, lossless_intermediate, "alpha_source_provenance_v2", is_mask=True),
                    fps=fps,
                    mask_semantics=ALPHA_SOURCE_PROVENANCE_SEMANTICS,
                )
            src_files["background_keep_prior"] = write_person_mask_artifact(
                mask_frames=background_keep_prior_masks,
                output_root=output_path,
                stem="src_background_keep_prior",
                artifact_format=self._artifact_format(save_format, lossless_intermediate, "background_keep_prior", is_mask=True),
                fps=fps,
                mask_semantics=BACKGROUND_KEEP_PRIOR_SEMANTICS,
            )
            src_files["visible_support"] = write_person_mask_artifact(
                mask_frames=background_debug["visible_support_map"].astype(np.float32),
                output_root=output_path,
                stem="src_visible_support",
                artifact_format=self._artifact_format(save_format, lossless_intermediate, "visible_support", is_mask=True),
                fps=fps,
                mask_semantics=BACKGROUND_VISIBLE_SUPPORT_SEMANTICS,
            )
            src_files["visible_support"].update({
                "source": "background_clean_plate",
                "background_mode": background_debug["background_mode"],
                "support_summary": {
                    "mean": float(background_debug["visible_support_map"].mean()),
                    "p80": float(np.quantile(background_debug["visible_support_map"], 0.8)),
                    "p95": float(np.quantile(background_debug["visible_support_map"], 0.95)),
                },
            })
            src_files["unresolved_region"] = write_person_mask_artifact(
                mask_frames=background_debug["unresolved_region"].astype(np.float32),
                output_root=output_path,
                stem="src_unresolved_region",
                artifact_format=self._artifact_format(save_format, lossless_intermediate, "unresolved_region", is_mask=True),
                fps=fps,
                mask_semantics=UNRESOLVED_REGION_SEMANTICS,
            )
            src_files["unresolved_region"].update({
                "source": "background_clean_plate",
                "background_mode": background_debug["background_mode"],
                "ratio_mean": float(background_debug["stats"].get("unresolved_ratio_mean", 0.0)),
            })
            src_files["background_confidence"] = write_person_mask_artifact(
                mask_frames=background_debug["background_confidence"].astype(np.float32),
                output_root=output_path,
                stem="src_background_confidence",
                artifact_format=self._artifact_format(save_format, lossless_intermediate, "background_confidence", is_mask=True),
                fps=fps,
                mask_semantics=BACKGROUND_CONFIDENCE_SEMANTICS,
            )
            src_files["background_confidence"].update({
                "source": "background_clean_plate",
                "background_mode": background_debug["background_mode"],
                "confidence_summary": {
                    "mean": float(background_debug["background_confidence"].mean()),
                    "p80": float(np.quantile(background_debug["background_confidence"], 0.8)),
                    "p95": float(np.quantile(background_debug["background_confidence"], 0.95)),
                },
            })
            src_files["background_source_provenance"] = write_person_mask_artifact(
                mask_frames=background_debug["background_source_provenance"].astype(np.float32),
                output_root=output_path,
                stem="src_background_source_provenance",
                artifact_format=self._artifact_format(save_format, lossless_intermediate, "background_source_provenance", is_mask=True),
                fps=fps,
                mask_semantics=BACKGROUND_SOURCE_PROVENANCE_SEMANTICS,
            )
            src_files["background_source_provenance"].update({
                "source": "background_clean_plate",
                "background_mode": background_debug["background_mode"],
                "value_encoding": {
                    "0.0": "passthrough",
                    "0.33333334": "image_fallback",
                    "0.6666667": "global_temporal",
                    "1.0": "local_temporal",
                },
            })
            src_files["background"]["background_mode"] = background_debug["background_mode"]
            runtime_stage_seconds["write_outputs"] = time.perf_counter() - write_start
            runtime_stage_seconds["total"] = time.perf_counter() - overall_start
            runtime_stats, runtime_metrics = self._summarize_runtime_stats(
                stage_seconds=runtime_stage_seconds,
                sam_chunk_stats=mask_debug.get("sam_chunk_stats", []),
                export_shape=(height, width),
                analysis_shape=(analysis_height, analysis_width),
                frame_count=len(export_frames),
                fps=float(fps),
                preprocess_runtime_profile=preprocess_runtime_profile,
            )
            runtime_stats["background"] = {
                "mode": background_debug.get("background_mode"),
                "stats": background_debug.get("stats", {}),
            }
            runtime_stats["face_analysis"] = {
                "mode": face_analysis_mode,
                "stats": {} if face_analysis is None else face_analysis["stats"],
            }
            runtime_stats["pose_motion_stack"] = {
                "mode": pose_motion_stack_mode,
                "stats": pose_motion_analysis["stats"],
            }
            runtime_stats["multistage"] = multistage_stats
            runtime_metrics["background_mode"] = background_debug.get("background_mode")
            runtime_metrics["face_tracking_center_jitter_mean"] = None if face_analysis is None else float(face_analysis["stats"].get("center_jitter_mean", 0.0))
            runtime_metrics["face_landmark_confidence_mean"] = None if face_analysis is None else float(face_analysis["stats"].get("landmark_confidence_mean", 0.0))
            runtime_metrics["body_jitter_mean"] = float(pose_motion_analysis["stats"].get("body_jitter_mean", 0.0))
            runtime_metrics["hand_jitter_mean"] = float(pose_motion_analysis["stats"].get("hand_jitter_mean", 0.0))
            runtime_metrics["limb_continuity_score"] = float(pose_motion_analysis["stats"].get("limb_continuity_score", 0.0))
            runtime_metrics["velocity_spike_rate"] = float(pose_motion_analysis["stats"].get("velocity_spike_rate", 0.0))
            runtime_metrics["person_roi_coverage_ratio"] = float(
                multistage_stats["person_roi_analysis"].get("proposal_stats", {}).get("coverage_ratio", 0.0)
            )
            runtime_metrics["face_roi_coverage_ratio"] = float(
                multistage_stats["face_roi_analysis"].get("proposal_stats", {}).get("coverage_ratio", 0.0)
            )
            outputs = {
                "frame_count": len(export_frames),
                "fps": float(fps),
                "height": int(height),
                "width": int(width),
                "analysis_height": int(analysis_height),
                "analysis_width": int(analysis_width),
                "channels": 3,
                "reference_height": int(reference_height),
                "reference_width": int(reference_width),
                "src_files": src_files,
                "reference_normalization": reference_normalization,
                "boundary_fusion": {
                    "mode": boundary_fusion_mode,
                    "parsing_mode": parsing_output["mode"],
                    "matting_mode": matting_output["mode"],
                    "artifact_schema_version": 2,
                    "artifact_sources": ["sam2", parsing_output["mode"], matting_output["mode"]],
                    "parsing_stats": parsing_output["stats"],
                    "matting_stats": matting_output["stats"],
                    "fusion_stats": boundary_fusion["stats"],
                },
                "background": {
                    "mode": background_debug.get("background_mode"),
                    "stats": background_debug.get("stats", {}),
                },
                "face_analysis": {
                    "mode": face_analysis_mode,
                    "stats": {} if face_analysis is None else face_analysis["stats"],
                },
                "pose_motion_stack": {
                    "mode": pose_motion_stack_mode,
                    "stats": pose_motion_analysis["stats"],
                },
                "multistage": multistage_stats,
                "sam_runtime": dict(self.sam_runtime_config),
                "sam_apply_postprocessing": bool(self.sam_apply_postprocessing),
                "sam_trace_dir": str(trace_dir) if trace_dir is not None else None,
                "runtime_stats": runtime_stats,
                "runtime_metrics": runtime_metrics,
            }
            if qa_outputs:
                outputs["qa_outputs"] = qa_outputs
            return outputs
        else:
            logger.info(f"Processing reference image: {refer_image_path}")
            refer_img = load_image_rgb(refer_image_path)
            src_ref_path = os.path.join(output_path, 'src_ref.png')
            write_reference_image(src_ref_path, refer_img)
            reference_height, reference_width = refer_img.shape[:2]
            
            refer_img = resize_by_area(refer_img, resolution_area[0] * resolution_area[1], divisor=16)
            
            refer_pose_meta = self.pose2d([refer_img])[0]


            logger.info(f"Processing template video: {video_path}")
            video_reader = VideoReader(video_path)
            frame_num = len(video_reader)
            print('frame_num: {}'.format(frame_num))

            video_fps = video_reader.get_avg_fps()
            print('video_fps: {}'.format(video_fps))
            print('fps: {}'.format(fps))

            # TODO: Maybe we can switch to PyAV later, which can get accurate frame num
            duration = video_reader.get_frame_timestamp(-1)[-1]      
            expected_frame_num = int(duration * video_fps + 0.5) 
            ratio = abs((frame_num - expected_frame_num)/frame_num)         
            if ratio > 0.1:
                print("Warning: The difference between the actual number of frames and the expected number of frames is two large")
                frame_num = expected_frame_num

            if fps == -1:
                fps = video_fps
                
            target_num = int(frame_num / video_fps * fps)
            print('target_num: {}'.format(target_num))
            idxs = get_frame_indices(frame_num, video_fps, target_num, fps)
            frames = video_reader.get_batch(idxs).asnumpy()
            validate_rgb_video("animation input video", frames)

            logger.info(f"Processing pose meta")

            tpl_pose_meta0 = self.pose2d(frames[:1])[0]
            raw_tpl_pose_metas = self.pose2d(frames)
            tpl_pose_metas, pose_conf_curve = stabilize_pose_metas(
                raw_tpl_pose_metas,
                method=pose_smooth_method,
                pose_conf_thresh_body=pose_conf_thresh_body,
                pose_conf_thresh_hand=pose_conf_thresh_hand,
                pose_conf_thresh_face=pose_conf_thresh_face,
                pose_smooth_strength_body=pose_smooth_strength_body,
                pose_smooth_strength_hand=pose_smooth_strength_hand,
                pose_smooth_strength_face=pose_smooth_strength_face,
                pose_max_velocity_body=pose_max_velocity_body,
                pose_max_velocity_hand=pose_max_velocity_hand,
                pose_max_velocity_face=pose_max_velocity_face,
                pose_interp_max_gap=pose_interp_max_gap,
            )
            face_bboxes, face_bbox_curve = stabilize_face_bboxes(
                tpl_pose_metas,
                image_shape=(frames[0].shape[0], frames[0].shape[1]),
                scale=1.3,
                conf_thresh=face_conf_thresh,
                min_valid_points=face_min_valid_points,
                smooth_method=face_bbox_smooth_method,
                smooth_strength=face_bbox_smooth_strength,
                max_scale_change=face_bbox_max_scale_change,
                max_center_shift=face_bbox_max_center_shift,
                hold_frames=face_bbox_hold_frames,
            )

            face_images = []
            for idx, face_bbox_for_image in enumerate(face_bboxes):
                x1, x2, y1, y2 = face_bbox_for_image
                face_image = frames[idx][y1:y2, x1:x2]
                face_image = cv2.resize(face_image, (512, 512))
                face_images.append(face_image)

            if retarget_flag:
                if use_flux:
                    tpl_prompt, refer_prompt = self.get_editing_prompts(tpl_pose_metas, refer_pose_meta)
                    refer_input = Image.fromarray(refer_img)
                    refer_edit = self.flux_kontext(
                            image=refer_input,
                            height=refer_img.shape[0],
                            width=refer_img.shape[1],
                            prompt=refer_prompt,
                            guidance_scale=2.5,
                            num_inference_steps=28,
                        ).images[0]
                    
                    refer_edit = Image.fromarray(padding_resize(np.array(refer_edit), refer_img.shape[0], refer_img.shape[1]))
                    refer_edit_path = os.path.join(output_path, 'refer_edit.png')
                    refer_edit.save(refer_edit_path)
                    refer_edit_pose_meta = self.pose2d([np.array(refer_edit)])[0]

                    tpl_img = frames[1]
                    tpl_input = Image.fromarray(tpl_img)
                    
                    tpl_edit = self.flux_kontext(
                            image=tpl_input,
                            height=tpl_img.shape[0],
                            width=tpl_img.shape[1],
                            prompt=tpl_prompt,
                            guidance_scale=2.5,
                            num_inference_steps=28,
                        ).images[0]
                    
                    tpl_edit = Image.fromarray(padding_resize(np.array(tpl_edit), tpl_img.shape[0], tpl_img.shape[1]))
                    tpl_edit_path = os.path.join(output_path, 'tpl_edit.png')
                    tpl_edit.save(tpl_edit_path)
                    tpl_edit_pose_meta0 = self.pose2d([np.array(tpl_edit)])[0]
                    tpl_retarget_pose_metas = get_retarget_pose(tpl_pose_meta0, refer_pose_meta, tpl_pose_metas, tpl_edit_pose_meta0, refer_edit_pose_meta)
                else:
                    tpl_retarget_pose_metas = get_retarget_pose(tpl_pose_meta0, refer_pose_meta, tpl_pose_metas, None, None)
            else:
               tpl_retarget_pose_metas = [AAPoseMeta.from_humanapi_meta(meta) for meta in tpl_pose_metas]

            cond_images = []
            for idx, meta in enumerate(tpl_retarget_pose_metas):
                if retarget_flag:
                    canvas = np.zeros_like(refer_img)
                    conditioning_image = draw_aapose_by_meta_new(canvas, meta)
                else:
                    canvas = np.zeros_like(frames[0])
                    conditioning_image = draw_aapose_by_meta_new(canvas, meta)
                    conditioning_image = padding_resize(conditioning_image, refer_img.shape[0], refer_img.shape[1])

                cond_images.append(conditioning_image)

            face_images = np.stack(face_images).astype(np.uint8)
            cond_images = np.stack(cond_images).astype(np.uint8)
            qa_outputs = {}
            if export_qa_visuals:
                face_bbox_overlay = make_face_bbox_overlay(np.stack(frames).astype(np.uint8), face_bboxes, face_bbox_curve)
                pose_overlay = make_pose_overlay(np.stack(frames).astype(np.uint8), tpl_pose_metas)
                qa_face_overlay = write_rgb_artifact(
                    frames=face_bbox_overlay,
                    output_root=output_path,
                    stem="face_bbox_overlay",
                    artifact_format="mp4",
                    fps=fps,
                )
                qa_pose_overlay = write_rgb_artifact(
                    frames=pose_overlay,
                    output_root=output_path,
                    stem="pose_overlay",
                    artifact_format="mp4",
                    fps=fps,
                )
                write_curve_json(os.path.join(output_path, "face_bbox_curve.json"), face_bbox_curve)
                write_curve_json(os.path.join(output_path, "pose_conf_curve.json"), pose_conf_curve)
                qa_outputs = {
                    "face_bbox_overlay": qa_face_overlay["path"],
                    "pose_overlay": qa_pose_overlay["path"],
                    "face_bbox_curve": "face_bbox_curve.json",
                    "pose_conf_curve": "pose_conf_curve.json",
                }
            src_files = {
                "pose": write_rgb_artifact(
                    frames=cond_images,
                    output_root=output_path,
                    stem="src_pose",
                    artifact_format=self._artifact_format(save_format, lossless_intermediate, "pose"),
                    fps=fps,
                ),
                "face": write_rgb_artifact(
                    frames=face_images,
                    output_root=output_path,
                    stem="src_face",
                    artifact_format=self._artifact_format(save_format, lossless_intermediate, "face"),
                    fps=fps,
                ),
                "reference": {
                    "path": "src_ref.png",
                    "type": "image",
                    "format": "png",
                    "height": int(reference_height),
                    "width": int(reference_width),
                    "channels": 3,
                    "color_space": "rgb",
                    "dtype": "uint8",
                    "shape": [int(reference_height), int(reference_width), 3],
                    "resized_height": int(cond_images[0].shape[0]),
                    "resized_width": int(cond_images[0].shape[1]),
                },
            }
            outputs = {
                "frame_count": len(frames),
                "fps": float(fps),
                "height": int(cond_images[0].shape[0]),
                "width": int(cond_images[0].shape[1]),
                "channels": 3,
                "reference_height": int(reference_height),
                "reference_width": int(reference_width),
                "src_files": src_files,
            }
            if qa_outputs:
                outputs["qa_outputs"] = qa_outputs
            return outputs

    def get_editing_prompts(self, tpl_pose_metas, refer_pose_meta):
        arm_visible = False
        leg_visible = False
        for tpl_pose_meta in tpl_pose_metas:
            tpl_keypoints = tpl_pose_meta['keypoints_body']
            if tpl_keypoints[3].all() != 0 or tpl_keypoints[4].all() != 0 or tpl_keypoints[6].all() != 0 or tpl_keypoints[7].all() != 0:
                if (tpl_keypoints[3][0] <= 1 and tpl_keypoints[3][1] <= 1 and tpl_keypoints[3][2] >= 0.75) or (tpl_keypoints[4][0] <= 1 and tpl_keypoints[4][1] <= 1 and tpl_keypoints[4][2] >= 0.75) or \
                    (tpl_keypoints[6][0] <= 1 and tpl_keypoints[6][1] <= 1 and tpl_keypoints[6][2] >= 0.75) or (tpl_keypoints[7][0] <= 1 and tpl_keypoints[7][1] <= 1 and tpl_keypoints[7][2] >= 0.75):
                    arm_visible = True
            if tpl_keypoints[9].all() != 0 or tpl_keypoints[12].all() != 0 or tpl_keypoints[10].all() != 0 or tpl_keypoints[13].all() != 0:
                if (tpl_keypoints[9][0] <= 1 and tpl_keypoints[9][1] <= 1 and tpl_keypoints[9][2] >= 0.75) or (tpl_keypoints[12][0] <= 1 and tpl_keypoints[12][1] <= 1 and tpl_keypoints[12][2] >= 0.75) or \
                    (tpl_keypoints[10][0] <= 1 and tpl_keypoints[10][1] <= 1 and tpl_keypoints[10][2] >= 0.75) or (tpl_keypoints[13][0] <= 1 and tpl_keypoints[13][1] <= 1 and tpl_keypoints[13][2] >= 0.75):
                    leg_visible = True
            if arm_visible and leg_visible:
                break
        
        if leg_visible:
            if tpl_pose_meta['width'] > tpl_pose_meta['height']:
                tpl_prompt = "Change the person to a standard T-pose (facing forward with arms extended). The person is standing. Feet and Hands are visible in the image."
            else:
                tpl_prompt = "Change the person to a standard pose with the face oriented forward and arms extending straight down by the sides. The person is standing. Feet and Hands are visible in the image."

            if refer_pose_meta['width'] > refer_pose_meta['height']:
                refer_prompt = "Change the person to a standard T-pose (facing forward with arms extended). The person is standing. Feet and Hands are visible in the image."
            else:
                refer_prompt = "Change the person to a standard pose with the face oriented forward and arms extending straight down by the sides. The person is standing. Feet and Hands are visible in the image."
        elif arm_visible:
            if tpl_pose_meta['width'] > tpl_pose_meta['height']:
                tpl_prompt = "Change the person to a standard T-pose (facing forward with arms extended). Hands are visible in the image."
            else:
                tpl_prompt = "Change the person to a standard pose with the face oriented forward and arms extending straight down by the sides. Hands are visible in the image."

            if refer_pose_meta['width'] > refer_pose_meta['height']:
                refer_prompt = "Change the person to a standard T-pose (facing forward with arms extended). Hands are visible in the image."
            else:
                refer_prompt = "Change the person to a standard pose with the face oriented forward and arms extending straight down by the sides. Hands are visible in the image."
        else:
            tpl_prompt = "Change the person to face forward."
            refer_prompt = "Change the person to face forward."

        return tpl_prompt, refer_prompt
    

    def get_mask(
        self,
        frames,
        *,
        kp2ds_all,
        sam_chunk_len,
        sam_keyframes_per_chunk,
        sam_prompt_body_conf_thresh,
        sam_prompt_face_conf_thresh,
        sam_prompt_hand_conf_thresh,
        sam_prompt_face_min_points,
        sam_prompt_hand_min_points,
        sam_use_negative_points,
        sam_negative_margin,
        sam_reprompt_interval,
        sam_prompt_mode,
        sam_debug_trace,
        sam_debug_trace_dir,
    ):
        frame_num = len(frames)
        th_step = max(1, int(sam_chunk_len))
        num_step = max(1, (frame_num + th_step - 1) // th_step)

        all_mask = []
        chunk_plans = []
        global_prompt_entries = []
        sam_chunk_stats = []
        for index in range(num_step):
            chunk_start_time = time.perf_counter()
            self._reset_peak_memory_stats()
            start_frame = index * th_step
            end_frame = min(frame_num, (index + 1) * th_step)
            each_frames = frames[start_frame:end_frame]
            kp2ds = kp2ds_all[start_frame:end_frame]
            if len(each_frames) <= 0:
                continue
            self._validate_chunk_frames(each_frames, index)
            chunk_trace = {
                "stage": "chunk_loaded",
                "chunk_index": int(index),
                "start_frame": int(start_frame),
                "end_frame": int(end_frame - 1),
                "frame_count": int(len(each_frames)),
                "frame_shape": list(each_frames[0].shape),
                "sam_runtime_profile": dict(self.sam_runtime_config),
            }
            if sam_debug_trace and sam_debug_trace_dir is not None:
                write_chunk_trace(sam_debug_trace_dir, index, chunk_trace)

            prompt_plan = plan_chunk_prompts(
                kp2ds,
                image_shape=(each_frames[0].shape[0], each_frames[0].shape[1]),
                keyframes_per_chunk=sam_keyframes_per_chunk,
                reprompt_interval=sam_reprompt_interval,
                body_conf_thresh=sam_prompt_body_conf_thresh,
                face_conf_thresh=sam_prompt_face_conf_thresh,
                hand_conf_thresh=sam_prompt_hand_conf_thresh,
                face_min_points=sam_prompt_face_min_points,
                hand_min_points=sam_prompt_hand_min_points,
                use_negative_points=sam_use_negative_points,
                negative_margin=sam_negative_margin,
            )
            prompt_entries = prompt_plan["prompt_entries"]
            if not prompt_entries:
                fallback_x = each_frames[0].shape[1] // 2
                fallback_y = each_frames[0].shape[0] // 2
                prompt_entries = [
                    {
                        "frame_idx": 0,
                        "tags": ["fallback"],
                        "positive_points": np.asarray([[fallback_x, fallback_y]], dtype=np.int32),
                        "negative_points": np.zeros((0, 2), dtype=np.int32),
                        "points": np.asarray([[fallback_x, fallback_y]], dtype=np.int32),
                        "labels": np.asarray([1], dtype=np.int32),
                        "positive_count": 1,
                        "negative_count": 0,
                        "positive_sources": {"image_center_fallback": 1},
                        "person_bbox": None,
                    }
                ]
                prompt_plan["prompt_entries"] = prompt_entries
                prompt_plan["prompt_frames"] = [0]
                prompt_plan["reprompt_frames"] = []
            prompt_entries = [
                self._sanitize_prompt_entry(
                    prompt_entry,
                    image_shape=(each_frames[0].shape[0], each_frames[0].shape[1]),
                    chunk_index=index,
                    prompt_index=prompt_index,
                )
                for prompt_index, prompt_entry in enumerate(prompt_entries)
            ]
            chunk_trace.update(
                {
                    "stage": "prompt_plan_ready",
                    "prompt_frames": [int(x) for x in prompt_plan["prompt_frames"]],
                    "reprompt_frames": [int(x) for x in prompt_plan["reprompt_frames"]],
                    "prompt_entries": [prompt_entry_trace(prompt_entry) for prompt_entry in prompt_entries],
                }
            )
            if sam_debug_trace and sam_debug_trace_dir is not None:
                write_chunk_trace(sam_debug_trace_dir, index, chunk_trace)

            try:
                init_state_start = time.perf_counter()
                chunk_trace["stage"] = "before_init_state"
                if sam_debug_trace and sam_debug_trace_dir is not None:
                    write_chunk_trace(sam_debug_trace_dir, index, chunk_trace)
                inference_state = self.predictor.init_state_v2(
                    frames=each_frames,
                    offload_video_to_cpu=self.sam_runtime_config["offload_video_to_cpu"],
                    offload_state_to_cpu=self.sam_runtime_config["offload_state_to_cpu"],
                )
                chunk_trace["stage"] = "after_init_state"
                chunk_trace["inference_num_frames"] = int(inference_state["num_frames"])
                chunk_trace["init_state_seconds"] = float(time.perf_counter() - init_state_start)
                if sam_debug_trace and sam_debug_trace_dir is not None:
                    write_chunk_trace(sam_debug_trace_dir, index, chunk_trace)
                self.predictor.reset_state(inference_state)
                ann_obj_id = 1
                prompt_seconds = 0.0
                for prompt_offset, prompt_entry in enumerate(prompt_entries):
                    ann_frame_idx = prompt_entry["frame_idx"]
                    chunk_trace["stage"] = "before_add_new_points" if sam_prompt_mode == "points" else "before_add_new_mask"
                    chunk_trace["current_prompt_index"] = int(prompt_offset)
                    chunk_trace["current_prompt_frame_idx"] = int(ann_frame_idx)
                    if sam_debug_trace and sam_debug_trace_dir is not None:
                        write_chunk_trace(sam_debug_trace_dir, index, chunk_trace)
                    prompt_start = time.perf_counter()
                    if sam_prompt_mode == "mask_seed":
                        seed_mask = self._build_seed_mask(
                            prompt_entry,
                            image_shape=(each_frames[0].shape[0], each_frames[0].shape[1]),
                        )
                        _, out_obj_ids, out_mask_logits = self.predictor.add_new_mask(
                            inference_state=inference_state,
                            frame_idx=ann_frame_idx,
                            obj_id=ann_obj_id,
                            mask=seed_mask,
                        )
                        chunk_trace["stage"] = "after_add_new_mask"
                        chunk_trace["current_prompt_mode"] = "mask_seed"
                    else:
                        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points(
                            inference_state=inference_state,
                            frame_idx=ann_frame_idx,
                            obj_id=ann_obj_id,
                            points=prompt_entry["points"],
                            labels=prompt_entry["labels"],
                        )
                        chunk_trace["stage"] = "after_add_new_points"
                        chunk_trace["current_prompt_mode"] = "points"
                    prompt_seconds += time.perf_counter() - prompt_start
                    chunk_trace["current_prompt_obj_count"] = int(len(out_obj_ids))
                    chunk_trace["current_prompt_mask_logits_shape"] = list(out_mask_logits.shape)
                    chunk_trace["prompt_seconds_so_far"] = float(prompt_seconds)
                    if sam_debug_trace and sam_debug_trace_dir is not None:
                        write_chunk_trace(sam_debug_trace_dir, index, chunk_trace)

                propagate_start = time.perf_counter()
                chunk_trace["stage"] = "before_propagate"
                if sam_debug_trace and sam_debug_trace_dir is not None:
                    write_chunk_trace(sam_debug_trace_dir, index, chunk_trace)
                video_segments = {}
                for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
                    video_segments[out_frame_idx] = {
                        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }
                chunk_trace["stage"] = "after_propagate"
                chunk_trace["video_segments_count"] = int(len(video_segments))
                chunk_trace["video_segments_nonempty_count"] = int(sum(1 for value in video_segments.values() if value))
                chunk_trace["propagate_seconds"] = float(time.perf_counter() - propagate_start)
                chunk_trace["prompt_seconds_total"] = float(prompt_seconds)
                chunk_trace["peak_memory_gb"] = float(self._peak_memory_gb())
                chunk_trace["total_seconds"] = float(time.perf_counter() - chunk_start_time)
                if sam_debug_trace and sam_debug_trace_dir is not None:
                    write_chunk_trace(sam_debug_trace_dir, index, chunk_trace)
            except Exception as exc:
                chunk_trace["stage"] = "failed"
                chunk_trace["error"] = str(exc)
                chunk_trace["peak_memory_gb"] = float(self._peak_memory_gb())
                chunk_trace["total_seconds"] = float(time.perf_counter() - chunk_start_time)
                if sam_debug_trace and sam_debug_trace_dir is not None:
                    write_chunk_trace(sam_debug_trace_dir, index, chunk_trace)
                raise

            for out_frame_idx in range(len(each_frames)):
                segment_masks = video_segments.get(out_frame_idx, {})
                if not segment_masks:
                    all_mask.append(np.zeros(each_frames[0].shape[:2], dtype=np.uint8))
                    continue
                for out_obj_id, out_mask in segment_masks.items():
                    out_mask = out_mask[0].astype(np.uint8)
                    all_mask.append(out_mask)
                    break

            chunk_prompt_entries = []
            for prompt_entry in prompt_entries:
                global_entry = dict(prompt_entry)
                global_entry["global_frame_idx"] = int(start_frame + prompt_entry["frame_idx"])
                global_prompt_entries.append(global_entry)
                chunk_prompt_entries.append(global_entry)
            chunk_plans.append(
                {
                    "chunk_index": int(index),
                    "start_frame": int(start_frame),
                    "end_frame": int(end_frame - 1),
                    "prompt_entries": chunk_prompt_entries,
                    "reprompt_frames": prompt_plan["reprompt_frames"],
                }
            )
            sam_chunk_stats.append(
                {
                    "chunk_index": int(index),
                    "start_frame": int(start_frame),
                    "end_frame": int(end_frame - 1),
                    "frame_count": int(len(each_frames)),
                    "init_state_seconds": float(chunk_trace.get("init_state_seconds", 0.0)),
                    "prompt_seconds": float(chunk_trace.get("prompt_seconds_total", 0.0)),
                    "propagate_seconds": float(chunk_trace.get("propagate_seconds", 0.0)),
                    "total_seconds": float(chunk_trace.get("total_seconds", 0.0)),
                    "peak_memory_gb": float(chunk_trace.get("peak_memory_gb", 0.0)),
                    "prompt_count": int(len(prompt_entries)),
                }
            )
        all_mask = np.stack(all_mask).astype(np.uint8)
        mask_stats = build_mask_stats(
            masks=all_mask,
            chunk_plans=chunk_plans,
            sam_chunk_len=sam_chunk_len,
            sam_keyframes_per_chunk=sam_keyframes_per_chunk,
            sam_reprompt_interval=sam_reprompt_interval,
            sam_use_negative_points=sam_use_negative_points,
            sam_negative_margin=sam_negative_margin,
        )
        if sam_debug_trace and sam_debug_trace_dir is not None:
            write_chunk_trace(
                sam_debug_trace_dir,
                -1,
                {
                    "stage": "session_completed",
                    "frame_count": int(frame_num),
                    "chunk_count": int(num_step),
                    "mask_frame_count": int(all_mask.shape[0]),
                    "sam_runtime_profile": dict(self.sam_runtime_config),
                },
            )
        return all_mask, {
            "prompt_entries": global_prompt_entries,
            "mask_stats": mask_stats,
            "sam_trace_dir": str(sam_debug_trace_dir) if sam_debug_trace_dir is not None else None,
            "sam_runtime_profile": dict(self.sam_runtime_config),
            "sam_chunk_stats": sam_chunk_stats,
        }
    
    def _validate_chunk_frames(self, frames, chunk_index):
        if len(frames) <= 0:
            raise ValueError(f"SAM2 chunk {chunk_index} contains no frames.")
        expected_shape = frames[0].shape
        if len(expected_shape) != 3 or expected_shape[2] != 3:
            raise ValueError(f"SAM2 chunk {chunk_index} must use RGB frames shaped [H, W, 3]. Got {expected_shape}.")
        for frame_offset, frame in enumerate(frames):
            if frame.shape != expected_shape:
                raise ValueError(
                    f"SAM2 chunk {chunk_index} frame {frame_offset} shape mismatch: expected {expected_shape}, got {frame.shape}."
                )

    def _sanitize_prompt_entry(self, prompt_entry, *, image_shape, chunk_index, prompt_index):
        height, width = image_shape
        points = np.asarray(prompt_entry["points"], dtype=np.int32)
        labels = np.asarray(prompt_entry["labels"], dtype=np.int32)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(
                f"SAM2 chunk {chunk_index} prompt {prompt_index} has invalid points shape {points.shape}; expected [N, 2]."
            )
        if labels.ndim != 1 or labels.shape[0] != points.shape[0]:
            raise ValueError(
                f"SAM2 chunk {chunk_index} prompt {prompt_index} labels shape {labels.shape} does not match points {points.shape}."
            )
        if points.shape[0] <= 0:
            raise ValueError(f"SAM2 chunk {chunk_index} prompt {prompt_index} has no points.")
        if not np.all((labels == 0) | (labels == 1)):
            raise ValueError(f"SAM2 chunk {chunk_index} prompt {prompt_index} labels must be 0/1.")
        xs = points[:, 0]
        ys = points[:, 1]
        if np.any(xs < 0) or np.any(xs >= width) or np.any(ys < 0) or np.any(ys >= height):
            raise ValueError(
                f"SAM2 chunk {chunk_index} prompt {prompt_index} contains out-of-bounds points for image_shape={image_shape}. "
                f"x-range=[{int(xs.min())}, {int(xs.max())}], y-range=[{int(ys.min())}, {int(ys.max())}]."
            )
        sanitized = dict(prompt_entry)
        sanitized["points"] = points
        sanitized["labels"] = labels
        sanitized["positive_count"] = int((labels == 1).sum())
        sanitized["negative_count"] = int((labels == 0).sum())
        return sanitized

    def _build_seed_mask(self, prompt_entry, *, image_shape):
        height, width = image_shape
        seed_mask = np.zeros((height, width), dtype=np.uint8)
        person_bbox = prompt_entry.get("person_bbox")
        if person_bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in person_bbox]
            x1 = max(0, min(width - 1, x1))
            x2 = max(0, min(width, x2))
            y1 = max(0, min(height - 1, y1))
            y2 = max(0, min(height, y2))
            if x2 > x1 and y2 > y1:
                seed_mask[y1:y2, x1:x2] = 1
        if not seed_mask.any():
            positive_points = prompt_entry.get("positive_points")
            if positive_points is None:
                points = np.asarray(prompt_entry["points"], dtype=np.int32)
                labels = np.asarray(prompt_entry["labels"], dtype=np.int32)
                positive_points = points[labels == 1]
            positive_points = np.asarray(positive_points, dtype=np.int32)
            radius = max(3, min(height, width) // 64)
            for x, y in positive_points:
                cv2.circle(seed_mask, (int(x), int(y)), radius, 1, thickness=-1)
        if not seed_mask.any():
            fallback_x = width // 2
            fallback_y = height // 2
            cv2.circle(seed_mask, (fallback_x, fallback_y), max(3, min(height, width) // 64), 1, thickness=-1)
        return seed_mask.astype(bool)

    def _reset_peak_memory_stats(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def _peak_memory_gb(self):
        if not torch.cuda.is_available():
            return 0.0
        return float(torch.cuda.max_memory_allocated() / (1024 ** 3))

    def _resize_frame_with_analysis_policy(self, frame, *, target_area, min_short_side=None, divisor=16):
        resized = resize_by_area(frame, target_area, divisor=divisor)
        if min_short_side is None or min(resized.shape[:2]) >= min_short_side:
            return resized
        orig_h, orig_w = frame.shape[:2]
        if orig_h <= orig_w:
            new_h = int(min_short_side)
            new_w = int(round(orig_w * (new_h / orig_h)))
        else:
            new_w = int(min_short_side)
            new_h = int(round(orig_h * (new_w / orig_w)))
        new_h = max(divisor, int(np.ceil(new_h / divisor) * divisor))
        new_w = max(divisor, int(np.ceil(new_w / divisor) * divisor))
        interpolation = cv2.INTER_AREA if (new_w * new_h < orig_w * orig_h) else cv2.INTER_LINEAR
        return padding_resize(frame, height=new_h, width=new_w, interpolation=interpolation)

    def _resize_mask_frames(self, mask_frames, target_shape):
        target_h, target_w = target_shape
        resized_masks = []
        for mask in mask_frames:
            resized = cv2.resize(mask.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            resized_masks.append((resized >= 0.5).astype(np.uint8))
        return np.stack(resized_masks).astype(np.uint8)

    def _summarize_runtime_stats(self, *, stage_seconds, sam_chunk_stats, export_shape, analysis_shape, frame_count, fps, preprocess_runtime_profile):
        chunk_seconds = [float(chunk["total_seconds"]) for chunk in sam_chunk_stats]
        chunk_peak_memory_gb = [float(chunk["peak_memory_gb"]) for chunk in sam_chunk_stats]
        runtime_stats = {
            "preprocess_runtime_profile": preprocess_runtime_profile,
            "frame_count": int(frame_count),
            "fps": float(fps),
            "export_shape": [int(export_shape[0]), int(export_shape[1])],
            "analysis_shape": [int(analysis_shape[0]), int(analysis_shape[1])],
            "stage_seconds": {key: float(value) for key, value in stage_seconds.items()},
            "sam_chunk_stats": sam_chunk_stats,
            "sam_chunk_count": int(len(sam_chunk_stats)),
            "sam_chunk_seconds_mean": float(np.mean(chunk_seconds)) if chunk_seconds else 0.0,
            "sam_chunk_seconds_max": float(np.max(chunk_seconds)) if chunk_seconds else 0.0,
            "sam_chunk_peak_memory_gb_max": float(np.max(chunk_peak_memory_gb)) if chunk_peak_memory_gb else 0.0,
            "peak_memory_gb": float(self._peak_memory_gb()),
        }
        runtime_metrics = {
            "preprocess_total_seconds": runtime_stats["stage_seconds"].get("total", 0.0),
            "preprocess_peak_memory_gb": runtime_stats["peak_memory_gb"],
            "sam_chunk_seconds_mean": runtime_stats["sam_chunk_seconds_mean"],
            "sam_chunk_seconds_max": runtime_stats["sam_chunk_seconds_max"],
            "sam_chunk_peak_memory_gb_max": runtime_stats["sam_chunk_peak_memory_gb_max"],
            "analysis_height": int(analysis_shape[0]),
            "analysis_width": int(analysis_shape[1]),
            "export_height": int(export_shape[0]),
            "export_width": int(export_shape[1]),
        }
        return runtime_stats, runtime_metrics

    def convert_list_to_array(self, metas):
        metas_list = []
        for meta in metas:
            for key, value in meta.items():
                if type(value) is list:
                    value = np.array(value)
                meta[key] = value
            metas_list.append(meta)
        return metas_list
