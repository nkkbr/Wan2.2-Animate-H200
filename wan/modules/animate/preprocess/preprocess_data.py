# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import os
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan.utils.experiment import (
    create_run_layout,
    finalize_stage_manifest,
    should_write_manifest,
    start_stage_manifest,
)
from wan.utils.animate_contract import build_preprocess_metadata, write_preprocess_metadata
from wan.utils.media_io import INTERMEDIATE_SAVE_FORMATS
from sam_runtime import resolve_preprocess_runtime_profile


def _parse_args():
    parser = argparse.ArgumentParser(
        description="The preprocessing pipeline for Wan-animate."
    )

    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="The path to the preprocessing model's checkpoint directory. ")

    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="The path to the driving video.")
    parser.add_argument(
        "--refer_path",
        type=str,
        default=None,
        help="The path to the refererence image.")
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="The path to save the processed results.")
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Logical name for the current preprocessing run. If set, a standard run directory is created under ./runs unless --run_dir is also provided."
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Directory for the current experiment run. When set, the preprocess stage writes outputs and manifest metadata into the standardized run layout."
    )
    parser.add_argument(
        "--save_manifest",
        action="store_true",
        default=False,
        help="Write a stage manifest for this invocation. Automatically enabled when --run_name or --run_dir is provided."
    )
    
    parser.add_argument(
        "--preprocess_runtime_profile",
        type=str,
        default="legacy_safe",
        choices=["legacy_safe", "h200_safe", "h200_aggressive", "h200_extreme"],
        help="High-level preprocess runtime profile. This controls default SAM runtime, analysis resolution, chunking, prompting, and postprocessing settings unless explicitly overridden."
    )
    parser.add_argument(
        "--resolution_area",
        type=int,
        nargs=2,
        default=[1280, 720],
        help="Export/output resolution for preprocess artifacts, specified as [width, height]. Detection, SAM2, and related analysis may run at a different analysis resolution."
    )
    parser.add_argument(
        "--analysis_resolution_area",
        type=int,
        nargs=2,
        default=None,
        help="Optional analysis resolution [width, height] used for pose detection, prompt planning, SAM2, and related internal preprocessing. Defaults to the selected preprocess runtime profile."
    )
    parser.add_argument(
        "--analysis_min_short_side",
        type=int,
        default=None,
        help="Optional minimum short side for the internal analysis resolution. Useful for H200 profiles that should preserve more detail than export resolution alone would provide."
    )
    parser.add_argument(
        "--multistage_preprocess_mode",
        type=str,
        default="none",
        choices=["none", "h200_extreme"],
        help="Enable multistage preprocess refinement. 'none' keeps the current single-stage path. 'h200_extreme' adds person ROI and face ROI refinement layers."
    )
    parser.add_argument(
        "--disable_person_roi_refine",
        action="store_true",
        default=False,
        help="Disable the person ROI refinement layer while keeping the multistage framework active."
    )
    parser.add_argument(
        "--disable_face_roi_refine",
        action="store_true",
        default=False,
        help="Disable the face ROI refinement layer while keeping the multistage framework active."
    )
    parser.add_argument(
        "--person_roi_stage_resolution_area",
        type=int,
        nargs=2,
        default=None,
        help="Optional analysis resolution [width, height] for the person ROI layer."
    )
    parser.add_argument(
        "--person_roi_stage_min_short_side",
        type=int,
        default=None,
        help="Optional minimum short side for the person ROI analysis stage."
    )
    parser.add_argument(
        "--person_roi_expand_ratio",
        type=float,
        default=1.18,
        help="Expansion ratio applied to the normalized person ROI proposal."
    )
    parser.add_argument(
        "--person_roi_min_size_ratio",
        type=float,
        default=0.20,
        help="Minimum normalized size enforced for person ROI proposals."
    )
    parser.add_argument(
        "--person_roi_target_long_side",
        type=int,
        default=None,
        help="Optional cap for the long side of person ROI crops before they are fed into the ROI pose rerun."
    )
    parser.add_argument(
        "--person_roi_body_conf_thresh",
        type=float,
        default=0.35,
        help="Body keypoint confidence threshold used to build person ROI proposals."
    )
    parser.add_argument(
        "--person_roi_hand_conf_thresh",
        type=float,
        default=0.25,
        help="Hand keypoint confidence threshold used to build person ROI proposals."
    )
    parser.add_argument(
        "--person_roi_face_conf_thresh",
        type=float,
        default=0.35,
        help="Face keypoint confidence threshold used to build person ROI proposals."
    )
    parser.add_argument(
        "--person_roi_fuse_weight",
        type=float,
        default=0.72,
        help="Weight assigned to refined person ROI coordinates when fusing them back into the global pose stream."
    )
    parser.add_argument(
        "--person_roi_conf_margin",
        type=float,
        default=0.03,
        help="Minimum confidence advantage required before a person ROI point is strongly preferred over the global point."
    )
    parser.add_argument(
        "--face_roi_stage_resolution_area",
        type=int,
        nargs=2,
        default=None,
        help="Optional analysis resolution [width, height] for the face ROI layer."
    )
    parser.add_argument(
        "--face_roi_stage_min_short_side",
        type=int,
        default=None,
        help="Optional minimum short side for the face ROI analysis stage."
    )
    parser.add_argument(
        "--face_roi_expand_ratio",
        type=float,
        default=1.65,
        help="Expansion ratio applied to the normalized face ROI proposal."
    )
    parser.add_argument(
        "--face_roi_min_size_ratio",
        type=float,
        default=0.12,
        help="Minimum normalized size enforced for face ROI proposals."
    )
    parser.add_argument(
        "--face_roi_target_long_side",
        type=int,
        default=None,
        help="Optional cap for the long side of face ROI crops before they are fed into the ROI pose rerun."
    )
    parser.add_argument(
        "--face_roi_conf_thresh",
        type=float,
        default=0.35,
        help="Confidence threshold used when proposing face ROI boxes."
    )
    parser.add_argument(
        "--face_roi_fuse_weight",
        type=float,
        default=0.86,
        help="Weight assigned to refined face ROI coordinates when fusing them back into the global pose stream."
    )
    parser.add_argument(
        "--face_roi_conf_margin",
        type=float,
        default=0.02,
        help="Minimum confidence advantage required before a face ROI point is strongly preferred over the previous point."
    )
    parser.add_argument(
        "--multistage_pose_extra_smooth",
        type=float,
        default=0.10,
        help="Extra smoothing strength applied during multistage pose restabilization."
    )
    parser.add_argument(
        "--multistage_face_bbox_extra_smooth",
        type=float,
        default=0.08,
        help="Extra smoothing strength applied to face bbox stabilization after ROI reruns."
    )
    parser.add_argument(
        "--face_analysis_mode",
        type=str,
        default="heuristic",
        choices=["none", "heuristic"],
        help="Enable the structured face analysis stack. 'heuristic' writes tracked face bbox, landmarks, head pose, expression, parsing, alpha, and uncertainty artifacts."
    )
    parser.add_argument(
        "--face_tracking_smooth_strength",
        type=float,
        default=0.90,
        help="Temporal smoothing strength for the tracking-aware face bbox layer."
    )
    parser.add_argument(
        "--face_tracking_max_scale_change",
        type=float,
        default=1.08,
        help="Maximum per-frame face bbox scale change allowed in the face tracking layer."
    )
    parser.add_argument(
        "--face_tracking_max_center_shift",
        type=float,
        default=0.016,
        help="Maximum normalized per-frame center shift allowed in the face tracking layer."
    )
    parser.add_argument(
        "--face_tracking_hold_frames",
        type=int,
        default=10,
        help="Maximum number of frames for which the face tracking layer may hold/predict a bbox when detection quality drops."
    )
    parser.add_argument(
        "--face_difficulty_expand_ratio",
        type=float,
        default=1.20,
        help="Extra crop expansion ratio used for high-difficulty face frames."
    )
    parser.add_argument(
        "--face_rerun_difficulty_threshold",
        type=float,
        default=0.48,
        help="Difficulty threshold above which the face crop layer switches to a more conservative rerun crop."
    )
    parser.add_argument(
        "--face_alpha_blur_kernel",
        type=int,
        default=7,
        help="Gaussian blur kernel size used when building face alpha."
    )
    parser.add_argument(
        "--pose_motion_stack_mode",
        type=str,
        default="v1",
        choices=["none", "v1"],
        help="Structured multiscale pose/motion analysis stack. 'none' keeps legacy pose stream but still exports baseline motion artifacts; 'v1' enables bidirectional smoothing, limb-local refine, hand-local refine, visibility states, and pose uncertainty."
    )
    parser.add_argument(
        "--pose_motion_body_bidirectional_strength",
        type=float,
        default=0.86,
        help="Bidirectional smoothing strength for global body tracks."
    )
    parser.add_argument(
        "--pose_motion_hand_bidirectional_strength",
        type=float,
        default=0.82,
        help="Bidirectional smoothing strength for hand tracks."
    )
    parser.add_argument(
        "--pose_motion_face_bidirectional_strength",
        type=float,
        default=0.88,
        help="Bidirectional smoothing strength for face keypoints inside the motion stack."
    )
    parser.add_argument(
        "--pose_motion_local_refine_strength",
        type=float,
        default=0.82,
        help="Local ROI refinement smoothing strength used for limb and hand subtracks."
    )
    parser.add_argument(
        "--pose_motion_limb_roi_expand_ratio",
        type=float,
        default=1.32,
        help="Expansion ratio for limb-local ROI tracks."
    )
    parser.add_argument(
        "--pose_motion_hand_roi_expand_ratio",
        type=float,
        default=1.75,
        help="Expansion ratio for hand-local ROI tracks."
    )
    parser.add_argument(
        "--pose_motion_velocity_spike_quantile",
        type=float,
        default=0.88,
        help="Quantile used to classify motion velocity spikes inside the pose motion stack."
    )
    parser.add_argument(
        "--pose_motion_uncertainty_blur_kernel",
        type=int,
        default=21,
        help="Gaussian blur kernel size used to smooth pose motion uncertainty maps."
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="The target FPS for processing the driving video. Set to -1 to use the video's original FPS."
    )
    parser.add_argument(
        "--save_format",
        type=str,
        default="mp4",
        choices=list(INTERMEDIATE_SAVE_FORMATS),
        help="Intermediate artifact format. 'mp4' keeps legacy behavior. 'png_seq' and 'npz' enable higher-fidelity roundtrips."
    )
    parser.add_argument(
        "--lossless_intermediate",
        action="store_true",
        default=False,
        help="Enable the recommended high-fidelity intermediate layout: RGB control signals are saved as png sequences and person_mask is saved as npz."
    )
    parser.add_argument(
        "--face_conf_thresh",
        type=float,
        default=0.45,
        help="Confidence threshold for face keypoints used to build face crops."
    )
    parser.add_argument(
        "--face_min_valid_points",
        type=int,
        default=15,
        help="Minimum number of valid face keypoints required before computing a new face bbox."
    )
    parser.add_argument(
        "--face_bbox_smooth_method",
        type=str,
        default="ema",
        choices=["ema"],
        help="Face bbox smoothing method."
    )
    parser.add_argument(
        "--face_bbox_smooth_strength",
        type=float,
        default=0.7,
        help="EMA strength for face bbox smoothing. Higher values make the crop more stable."
    )
    parser.add_argument(
        "--face_bbox_max_scale_change",
        type=float,
        default=1.15,
        help="Maximum per-frame multiplicative change allowed for face bbox width/height."
    )
    parser.add_argument(
        "--face_bbox_max_center_shift",
        type=float,
        default=0.04,
        help="Maximum normalized center shift allowed for face bbox updates, expressed as a fraction of max(H, W)."
    )
    parser.add_argument(
        "--face_bbox_hold_frames",
        type=int,
        default=6,
        help="Maximum number of consecutive frames that may reuse the previous face bbox when confidence collapses."
    )
    parser.add_argument(
        "--pose_smooth_method",
        type=str,
        default="ema",
        choices=["ema"],
        help="Pose smoothing method."
    )
    parser.add_argument(
        "--pose_conf_thresh_body",
        type=float,
        default=0.5,
        help="Confidence threshold for body keypoints."
    )
    parser.add_argument(
        "--pose_conf_thresh_hand",
        type=float,
        default=0.35,
        help="Confidence threshold for hand keypoints."
    )
    parser.add_argument(
        "--pose_conf_thresh_face",
        type=float,
        default=0.45,
        help="Confidence threshold for face keypoints inside the pose stream."
    )
    parser.add_argument(
        "--pose_smooth_strength_body",
        type=float,
        default=0.65,
        help="EMA strength for body keypoint smoothing."
    )
    parser.add_argument(
        "--pose_smooth_strength_hand",
        type=float,
        default=0.35,
        help="EMA strength for hand keypoint smoothing."
    )
    parser.add_argument(
        "--pose_smooth_strength_face",
        type=float,
        default=0.7,
        help="EMA strength for face keypoint smoothing."
    )
    parser.add_argument(
        "--pose_max_velocity_body",
        type=float,
        default=0.05,
        help="Maximum normalized per-frame displacement for body keypoints after smoothing."
    )
    parser.add_argument(
        "--pose_max_velocity_hand",
        type=float,
        default=0.08,
        help="Maximum normalized per-frame displacement for hand keypoints after smoothing."
    )
    parser.add_argument(
        "--pose_max_velocity_face",
        type=float,
        default=0.04,
        help="Maximum normalized per-frame displacement for face keypoints after smoothing."
    )
    parser.add_argument(
        "--pose_interp_max_gap",
        type=int,
        default=3,
        help="Maximum number of consecutive low-confidence frames to repair via interpolation or hold."
    )
    parser.add_argument(
        "--export_qa_visuals",
        action="store_true",
        default=False,
        help="Export QA overlays and JSON files for stabilized face/pose controls and SAM2 mask generation."
    )
    parser.add_argument(
        "--sam_chunk_len",
        type=int,
        default=120,
        help="Number of frames per SAM2 tracking chunk in replacement mode."
    )
    parser.add_argument(
        "--sam_keyframes_per_chunk",
        type=int,
        default=8,
        help="Number of uniformly sampled conditioning frames per SAM2 chunk."
    )
    parser.add_argument(
        "--sam_prompt_body_conf_thresh",
        type=float,
        default=0.35,
        help="Confidence threshold for body keypoints used as SAM2 positive prompts."
    )
    parser.add_argument(
        "--sam_prompt_face_conf_thresh",
        type=float,
        default=0.45,
        help="Confidence threshold for face keypoints used to derive the SAM2 face-center prompt."
    )
    parser.add_argument(
        "--sam_prompt_hand_conf_thresh",
        type=float,
        default=0.35,
        help="Confidence threshold for hand keypoints used to derive SAM2 hand-center prompts."
    )
    parser.add_argument(
        "--sam_prompt_face_min_points",
        type=int,
        default=8,
        help="Minimum number of valid face keypoints required before adding the SAM2 face-center prompt."
    )
    parser.add_argument(
        "--sam_prompt_hand_min_points",
        type=int,
        default=6,
        help="Minimum number of valid hand keypoints required before adding a SAM2 hand-center prompt."
    )
    parser.add_argument(
        "--sam_use_negative_points",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable negative background points around the person bbox for SAM2 prompting."
    )
    parser.add_argument(
        "--sam_negative_margin",
        type=float,
        default=0.08,
        help="Margin ratio used to place negative SAM2 prompts outside the person bbox."
    )
    parser.add_argument(
        "--sam_reprompt_interval",
        type=int,
        default=40,
        help="Additional fixed-interval SAM2 re-prompt spacing within each chunk. Set to 0 to disable."
    )
    parser.add_argument(
        "--sam_prompt_mode",
        type=str,
        default="points",
        choices=["points", "mask_seed"],
        help="How SAM2 is conditioned inside each chunk. 'points' keeps the normal click-based path. 'mask_seed' is a conservative Step 01 fallback used to isolate add_new_points instability."
    )
    parser.add_argument(
        "--sam_runtime_profile",
        type=str,
        default="legacy_safe",
        choices=["legacy_safe", "h200_safe", "h200_aggressive"],
        help="Named SAM2 runtime profile. Step 01 uses this to compare stable and H200-oriented preprocess configurations."
    )
    parser.add_argument(
        "--sam_apply_postprocessing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable SAM2 postprocessing overrides such as dynamic multimask fallback and hole filling. Step 01 can disable this to isolate interaction-path crashes."
    )
    parser.add_argument(
        "--sam_use_flash_attn",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Optional explicit override for the SAM2 transformer flash-attention path."
    )
    parser.add_argument(
        "--sam_math_kernel_on",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Optional explicit override for the SAM2 transformer math-kernel path."
    )
    parser.add_argument(
        "--sam_old_gpu_mode",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Optional explicit override for the SAM2 transformer old-GPU compatibility mode."
    )
    parser.add_argument(
        "--sam_offload_video_to_cpu",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Optional explicit override for SAM2 frame offload to CPU during preprocess tracking."
    )
    parser.add_argument(
        "--sam_offload_state_to_cpu",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Optional explicit override for SAM2 inference-state offload to CPU during preprocess tracking."
    )
    parser.add_argument(
        "--sam_debug_trace",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write per-chunk SAM2 runtime traces so native failures leave a usable last-known-good state on disk."
    )
    parser.add_argument(
        "--sam_debug_trace_dir",
        type=str,
        default=None,
        help="Optional directory for SAM2 debug traces. Defaults to <save_path>/sam2_debug when --sam_debug_trace is enabled."
    )
    parser.add_argument(
        "--soft_mask_mode",
        type=str,
        default="soft_band",
        choices=["none", "soft_band"],
        help="Replacement soft-mask strategy. 'soft_band' exports a float transition band around the hard person mask."
    )
    parser.add_argument(
        "--soft_mask_band_width",
        type=int,
        default=24,
        help="Width in pixels for the exported soft boundary band."
    )
    parser.add_argument(
        "--soft_mask_blur_kernel",
        type=int,
        default=5,
        help="Optional odd kernel size used to smooth the soft boundary band. Set to 0 to disable."
    )
    parser.add_argument(
        "--boundary_fusion_mode",
        type=str,
        default="v2",
        choices=["none", "legacy", "heuristic", "v2"],
        help="Boundary fusion mode. 'legacy' preserves the optimization2 heuristic fusion, while 'v2' adds occlusion and uncertainty artifacts."
    )
    parser.add_argument(
        "--parsing_mode",
        type=str,
        default="heuristic",
        choices=["none", "heuristic"],
        help="Parsing adapter mode used to derive semantic boundary priors."
    )
    parser.add_argument(
        "--matting_mode",
        type=str,
        default="heuristic",
        choices=["none", "heuristic", "high_precision_v2", "production_v1"],
        help="Matting adapter mode used to derive soft alpha."
    )
    parser.add_argument(
        "--parsing_head_expand",
        type=float,
        default=1.2,
        help="Expansion factor for the parsing adapter's head prior derived from the face region."
    )
    parser.add_argument(
        "--parsing_hand_radius_ratio",
        type=float,
        default=0.025,
        help="Normalized radius used to rasterize hand priors in the parsing adapter."
    )
    parser.add_argument(
        "--parsing_boundary_kernel",
        type=int,
        default=11,
        help="Kernel size used to define the heuristic parsing boundary ring."
    )
    parser.add_argument(
        "--matting_trimap_inner_erode",
        type=int,
        default=3,
        help="Inner erosion radius for the heuristic matting trimap."
    )
    parser.add_argument(
        "--matting_trimap_outer_dilate",
        type=int,
        default=12,
        help="Outer dilation radius for the heuristic matting trimap."
    )
    parser.add_argument(
        "--matting_blur_kernel",
        type=int,
        default=5,
        help="Blur kernel applied to the heuristic soft alpha."
    )
    parser.add_argument(
        "--alpha_v2_detail_boost",
        type=float,
        default=0.28,
        help="Detail preservation boost used by matting_mode=high_precision_v2."
    )
    parser.add_argument(
        "--alpha_v2_shrink_strength",
        type=float,
        default=0.34,
        help="Boundary contraction strength used by matting_mode=high_precision_v2."
    )
    parser.add_argument(
        "--alpha_v2_hair_boost",
        type=float,
        default=0.42,
        help="Hair-edge preservation boost used by matting_mode=high_precision_v2."
    )
    parser.add_argument(
        "--alpha_v2_hard_threshold",
        type=float,
        default=0.68,
        help="Alpha threshold used to derive refined_hard_foreground in matting_mode=high_precision_v2."
    )
    parser.add_argument(
        "--alpha_v2_bilateral_sigma_color",
        type=float,
        default=0.12,
        help="Bilateral sigmaColor used by matting_mode=high_precision_v2."
    )
    parser.add_argument(
        "--alpha_v2_bilateral_sigma_space",
        type=float,
        default=5.0,
        help="Bilateral sigmaSpace used by matting_mode=high_precision_v2."
    )
    parser.add_argument(
        "--alpha_v3_detail_boost",
        type=float,
        default=0.24,
        help="Detail boost used by matting_mode=production_v1."
    )
    parser.add_argument(
        "--alpha_v3_color_mix",
        type=float,
        default=0.42,
        help="Color-based alpha mixing weight used by matting_mode=production_v1."
    )
    parser.add_argument(
        "--alpha_v3_active_blend",
        type=float,
        default=0.55,
        help="Blend strength between legacy alpha and refined alpha used by matting_mode=production_v1."
    )
    parser.add_argument(
        "--alpha_v3_delta_clip",
        type=float,
        default=0.10,
        help="Maximum per-pixel alpha correction magnitude used by matting_mode=production_v1."
    )
    parser.add_argument(
        "--bg_inpaint_mode",
        type=str,
        default="none",
        choices=["none", "image", "video", "video_v2"],
        help="Background construction mode for replacement. 'video' keeps the legacy clean_plate_video v1 path, while 'video_v2' enables the visibility-aware background 2.0 path."
    )
    parser.add_argument(
        "--bg_inpaint_method",
        type=str,
        default="telea",
        choices=["telea", "ns"],
        help="OpenCV inpaint method used when --bg_inpaint_mode=image."
    )
    parser.add_argument(
        "--bg_inpaint_mask_expand",
        type=int,
        default=16,
        help="Additional mask expansion in pixels before background inpainting."
    )
    parser.add_argument(
        "--bg_inpaint_radius",
        type=float,
        default=5.0,
        help="Inpaint radius used for clean plate background construction."
    )
    parser.add_argument(
        "--bg_temporal_smooth_strength",
        type=float,
        default=0.14,
        help="EMA-like temporal smoothing strength applied only inside inpainted regions of the clean plate."
    )
    parser.add_argument(
        "--bg_video_window_radius",
        type=int,
        default=4,
        help="Half window radius used by clean_plate_video when aggregating visible background from neighboring frames."
    )
    parser.add_argument(
        "--bg_video_min_visible_count",
        type=int,
        default=2,
        help="Minimum number of visible background observations required before clean_plate_video trusts a temporal prefill."
    )
    parser.add_argument(
        "--bg_video_blend_strength",
        type=float,
        default=0.7,
        help="Blend strength between temporal background aggregation and the image clean-plate fallback in clean_plate_video mode."
    )
    parser.add_argument(
        "--bg_video_global_min_visible_count",
        type=int,
        default=3,
        help="Minimum full-video visible observation count required before visibility-aware clean_plate_video_v2 trusts a global temporal background estimate."
    )
    parser.add_argument(
        "--bg_video_confidence_threshold",
        type=float,
        default=0.30,
        help="Confidence threshold below which clean_plate_video_v2 marks an inpainted pixel as unresolved."
    )
    parser.add_argument(
        "--bg_video_global_blend_strength",
        type=float,
        default=0.95,
        help="Blend strength between the full-video global temporal estimate and the image clean-plate fallback in clean_plate_video_v2."
    )
    parser.add_argument(
        "--bg_video_consistency_scale",
        type=float,
        default=18.0,
        help="Scale used to convert temporal observation deviation into confidence for clean_plate_video_v2."
    )
    parser.add_argument(
        "--reference_normalization_mode",
        type=str,
        default="none",
        choices=["none", "bbox_match", "structure_match"],
        help="Optional replacement reference normalization strategy. 'bbox_match' scales and re-centers the reference subject to match the driver subject occupancy. 'structure_match' additionally aligns head/torso/leg proportions and shoulder width using a structure-aware warp."
    )
    parser.add_argument(
        "--reference_target_bbox_source",
        type=str,
        default="median_first_n",
        choices=["first_frame", "median_first_n"],
        help="How to estimate the target subject bbox from the driver video."
    )
    parser.add_argument(
        "--reference_target_bbox_frames",
        type=int,
        default=16,
        help="Number of initial driver frames used when reference_target_bbox_source=median_first_n."
    )
    parser.add_argument(
        "--reference_bbox_conf_thresh",
        type=float,
        default=0.35,
        help="Confidence threshold for pose points used to derive both driver and reference person bboxes."
    )
    parser.add_argument(
        "--reference_scale_clamp_min",
        type=float,
        default=0.75,
        help="Lower clamp for the reference normalization scale factor."
    )
    parser.add_argument(
        "--reference_scale_clamp_max",
        type=float,
        default=1.6,
        help="Upper clamp for the reference normalization scale factor."
    )
    parser.add_argument(
        "--reference_structure_segment_clamp_min",
        type=float,
        default=0.8,
        help="Lower clamp for per-segment structure scaling in structure_match mode."
    )
    parser.add_argument(
        "--reference_structure_segment_clamp_max",
        type=float,
        default=1.25,
        help="Upper clamp for per-segment structure scaling in structure_match mode."
    )
    parser.add_argument(
        "--reference_structure_width_budget_ratio",
        type=float,
        default=1.05,
        help="Maximum allowed normalized reference width relative to the driver replacement-region width budget."
    )
    parser.add_argument(
        "--reference_structure_height_budget_ratio",
        type=float,
        default=1.05,
        help="Maximum allowed normalized reference height relative to the driver replacement-region height budget."
    )

    parser.add_argument(
        "--replace_flag",
        action="store_true",
        default=False,
        help="Whether to use replacement mode.")
    parser.add_argument(
        "--retarget_flag",
        action="store_true",
        default=False,
        help="Whether to use pose retargeting. Currently only supported in animation mode")
    parser.add_argument(
        "--use_flux",
        action="store_true",
        default=False,
        help="Whether to use image editing in pose retargeting. Recommended if the character in the reference image or the first frame of the driving video is not in a standard, front-facing pose")
    
    # Parameters for the mask strategy in replacement mode. These control the mask's size and shape. Refer to https://arxiv.org/pdf/2502.06145
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of iterations for mask dilation."
    )
    parser.add_argument(
        "--k",
        type=int,
        default=7,
        help="Number of kernel size for mask dilation."
    )
    parser.add_argument(
        "--w_len",
        type=int,
        default=1,
        help="The number of subdivisions for the grid along the 'w' dimension. A higher value results in a more detailed contour. A value of 1 means no subdivision is performed."
    )
    parser.add_argument(
        "--h_len",
        type=int,
        default=1,
        help="The number of subdivisions for the grid along the 'h' dimension. A higher value results in a more detailed contour. A value of 1 means no subdivision is performed."
    )
    args = parser.parse_args()

    return args, parser


def _apply_preprocess_runtime_profile_defaults(args, parser):
    profile_defaults = resolve_preprocess_runtime_profile(args.preprocess_runtime_profile)
    for field, value in profile_defaults.items():
        if field == "preprocess_runtime_profile":
            continue
        if not hasattr(args, field):
            continue
        current_value = getattr(args, field)
        default_value = parser.get_default(field)
        if current_value == default_value or (current_value is None and default_value is None):
            setattr(args, field, value)
    return profile_defaults


if __name__ == '__main__':
    args, parser = _parse_args()
    preprocess_runtime_profile = _apply_preprocess_runtime_profile_defaults(args, parser)
    args_dict = vars(args)
    print(args_dict)

    assert len(args.resolution_area) == 2, "resolution_area should be a list of two integers [width, height]"
    if args.analysis_resolution_area is not None:
        assert len(args.analysis_resolution_area) == 2, "analysis_resolution_area should be a list of two integers [width, height]"
    if args.person_roi_stage_resolution_area is not None:
        assert len(args.person_roi_stage_resolution_area) == 2, "person_roi_stage_resolution_area should be [width, height]"
    if args.face_roi_stage_resolution_area is not None:
        assert len(args.face_roi_stage_resolution_area) == 2, "face_roi_stage_resolution_area should be [width, height]"
    if args.analysis_min_short_side is not None and args.analysis_min_short_side <= 0:
        raise ValueError("analysis_min_short_side must be > 0 when provided.")
    if args.person_roi_stage_min_short_side is not None and args.person_roi_stage_min_short_side <= 0:
        raise ValueError("person_roi_stage_min_short_side must be > 0 when provided.")
    if args.face_roi_stage_min_short_side is not None and args.face_roi_stage_min_short_side <= 0:
        raise ValueError("face_roi_stage_min_short_side must be > 0 when provided.")
    if args.person_roi_target_long_side is not None and args.person_roi_target_long_side <= 0:
        raise ValueError("person_roi_target_long_side must be > 0 when provided.")
    if args.face_roi_target_long_side is not None and args.face_roi_target_long_side <= 0:
        raise ValueError("face_roi_target_long_side must be > 0 when provided.")
    assert not args.use_flux or args.retarget_flag, "Image editing with FLUX can only be used when pose retargeting is enabled."
    assert args.ckpt_path is not None, "Please provide --ckpt_path."
    assert Path(args.ckpt_path).exists(), f"Checkpoint path does not exist: {args.ckpt_path}"
    assert args.video_path is not None, "Please provide --video_path."
    assert Path(args.video_path).exists(), f"Video path does not exist: {args.video_path}"
    assert args.refer_path is not None, "Please provide --refer_path."
    assert Path(args.refer_path).exists(), f"Reference image path does not exist: {args.refer_path}"
    if args.reference_scale_clamp_min <= 0 or args.reference_scale_clamp_max <= 0:
        raise ValueError("reference_scale_clamp_min and reference_scale_clamp_max must be > 0.")
    if args.reference_scale_clamp_min > args.reference_scale_clamp_max:
        raise ValueError("reference_scale_clamp_min must be <= reference_scale_clamp_max.")
    if args.reference_structure_segment_clamp_min <= 0 or args.reference_structure_segment_clamp_max <= 0:
        raise ValueError("reference_structure_segment_clamp_min and reference_structure_segment_clamp_max must be > 0.")
    if args.reference_structure_segment_clamp_min > args.reference_structure_segment_clamp_max:
        raise ValueError("reference_structure_segment_clamp_min must be <= reference_structure_segment_clamp_max.")
    if args.reference_structure_width_budget_ratio <= 0 or args.reference_structure_height_budget_ratio <= 0:
        raise ValueError("reference_structure_width_budget_ratio and reference_structure_height_budget_ratio must be > 0.")
    if args.person_roi_expand_ratio <= 0 or args.face_roi_expand_ratio <= 0:
        raise ValueError("person_roi_expand_ratio and face_roi_expand_ratio must be > 0.")
    if args.person_roi_min_size_ratio <= 0 or args.face_roi_min_size_ratio <= 0:
        raise ValueError("person_roi_min_size_ratio and face_roi_min_size_ratio must be > 0.")
    if not 0.0 <= float(args.person_roi_fuse_weight) <= 1.0:
        raise ValueError("person_roi_fuse_weight must be in [0, 1].")
    if not 0.0 <= float(args.face_roi_fuse_weight) <= 1.0:
        raise ValueError("face_roi_fuse_weight must be in [0, 1].")
    if args.multistage_preprocess_mode == "none" and (args.disable_person_roi_refine or args.disable_face_roi_refine):
        raise ValueError("disable_person_roi_refine/disable_face_roi_refine are only meaningful when multistage_preprocess_mode is enabled.")
    if args.bg_video_window_radius < 1:
        raise ValueError("bg_video_window_radius must be >= 1.")
    if args.bg_video_min_visible_count < 1:
        raise ValueError("bg_video_min_visible_count must be >= 1.")
    if not 0.0 <= float(args.bg_video_blend_strength) <= 1.0:
        raise ValueError("bg_video_blend_strength must be in [0, 1].")
    if not 0.0 <= float(args.face_tracking_smooth_strength) < 1.0:
        raise ValueError("face_tracking_smooth_strength must be in [0, 1).")
    if args.face_tracking_max_scale_change <= 1.0:
        raise ValueError("face_tracking_max_scale_change must be > 1.0.")
    if args.face_tracking_hold_frames < 0:
        raise ValueError("face_tracking_hold_frames must be >= 0.")
    if args.face_difficulty_expand_ratio < 1.0:
        raise ValueError("face_difficulty_expand_ratio must be >= 1.0.")
    if args.face_alpha_blur_kernel <= 0 or args.face_alpha_blur_kernel % 2 == 0:
        raise ValueError("face_alpha_blur_kernel must be a positive odd integer.")
    if not 0.0 <= float(args.pose_motion_body_bidirectional_strength) < 1.0:
        raise ValueError("pose_motion_body_bidirectional_strength must be in [0, 1).")
    if not 0.0 <= float(args.pose_motion_hand_bidirectional_strength) < 1.0:
        raise ValueError("pose_motion_hand_bidirectional_strength must be in [0, 1).")
    if not 0.0 <= float(args.pose_motion_face_bidirectional_strength) < 1.0:
        raise ValueError("pose_motion_face_bidirectional_strength must be in [0, 1).")
    if not 0.0 <= float(args.pose_motion_local_refine_strength) < 1.0:
        raise ValueError("pose_motion_local_refine_strength must be in [0, 1).")
    if args.pose_motion_limb_roi_expand_ratio <= 1.0:
        raise ValueError("pose_motion_limb_roi_expand_ratio must be > 1.0.")
    if args.pose_motion_hand_roi_expand_ratio <= 1.0:
        raise ValueError("pose_motion_hand_roi_expand_ratio must be > 1.0.")
    if not 0.5 <= float(args.pose_motion_velocity_spike_quantile) < 1.0:
        raise ValueError("pose_motion_velocity_spike_quantile must be in [0.5, 1.0).")
    if args.pose_motion_uncertainty_blur_kernel <= 0 or args.pose_motion_uncertainty_blur_kernel % 2 == 0:
        raise ValueError("pose_motion_uncertainty_blur_kernel must be a positive odd integer.")
    run_layout = None
    manifest_token = None
    if should_write_manifest(args):
        if args.run_name is None and args.run_dir is None:
            base = Path(args.save_path).name if args.save_path is not None else "preprocess"
            args.run_name = f"{base}_{Path(args.video_path).stem}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        run_layout = create_run_layout(run_name=args.run_name, run_dir=args.run_dir, repo_root=REPO_ROOT)
        if args.save_path is None:
            args.save_path = str(run_layout["preprocess_dir"].resolve())
    assert args.save_path is not None, "Please provide --save_path or enable run tracking with --run_name/--run_dir."

    pose2d_checkpoint_path = os.path.join(args.ckpt_path, 'pose2d/vitpose_h_wholebody.onnx')
    det_checkpoint_path = os.path.join(args.ckpt_path, 'det/yolov10m.onnx')

    sam2_checkpoint_path = os.path.join(args.ckpt_path, 'sam2/sam2_hiera_large.pt') if args.replace_flag else None
    flux_kontext_path = os.path.join(args.ckpt_path, 'FLUX.1-Kontext-dev') if args.use_flux else None
    from process_pipepline import ProcessPipeline
    process_pipeline = ProcessPipeline(
        det_checkpoint_path=det_checkpoint_path,
        pose2d_checkpoint_path=pose2d_checkpoint_path,
        sam_checkpoint_path=sam2_checkpoint_path,
        flux_kontext_path=flux_kontext_path,
        sam_apply_postprocessing=args.sam_apply_postprocessing,
        sam_runtime_profile=args.sam_runtime_profile,
        sam_use_flash_attn=args.sam_use_flash_attn,
        sam_math_kernel_on=args.sam_math_kernel_on,
        sam_old_gpu_mode=args.sam_old_gpu_mode,
        sam_offload_video_to_cpu=args.sam_offload_video_to_cpu,
        sam_offload_state_to_cpu=args.sam_offload_state_to_cpu,
    )
    os.makedirs(args.save_path, exist_ok=True)
    if run_layout is not None:
        manifest_token = start_stage_manifest(
            run_layout,
            "preprocess",
            args,
            inputs={
                "video_path": args.video_path,
                "refer_path": args.refer_path,
                "checkpoint_path": args.ckpt_path,
            },
        )
    try:
        pipeline_outputs = process_pipeline(video_path=args.video_path,
                                            refer_image_path=args.refer_path,
                                            output_path=args.save_path,
                                            resolution_area=args.resolution_area,
                                            analysis_resolution_area=args.analysis_resolution_area,
                                            analysis_min_short_side=args.analysis_min_short_side,
                                            fps=args.fps,
                                            save_format=args.save_format,
                                            lossless_intermediate=args.lossless_intermediate,
                                            face_conf_thresh=args.face_conf_thresh,
                                            face_min_valid_points=args.face_min_valid_points,
                                            face_bbox_smooth_method=args.face_bbox_smooth_method,
                                            face_bbox_smooth_strength=args.face_bbox_smooth_strength,
                                            face_bbox_max_scale_change=args.face_bbox_max_scale_change,
                                            face_bbox_max_center_shift=args.face_bbox_max_center_shift,
                                            face_bbox_hold_frames=args.face_bbox_hold_frames,
                                            pose_smooth_method=args.pose_smooth_method,
                                            pose_conf_thresh_body=args.pose_conf_thresh_body,
                                            pose_conf_thresh_hand=args.pose_conf_thresh_hand,
                                            pose_conf_thresh_face=args.pose_conf_thresh_face,
                                            pose_smooth_strength_body=args.pose_smooth_strength_body,
                                            pose_smooth_strength_hand=args.pose_smooth_strength_hand,
                                            pose_smooth_strength_face=args.pose_smooth_strength_face,
                                            pose_max_velocity_body=args.pose_max_velocity_body,
                                            pose_max_velocity_hand=args.pose_max_velocity_hand,
                                            pose_max_velocity_face=args.pose_max_velocity_face,
                                            pose_interp_max_gap=args.pose_interp_max_gap,
                                            export_qa_visuals=args.export_qa_visuals,
                                            sam_chunk_len=args.sam_chunk_len,
                                            sam_keyframes_per_chunk=args.sam_keyframes_per_chunk,
                                            sam_prompt_body_conf_thresh=args.sam_prompt_body_conf_thresh,
                                            sam_prompt_face_conf_thresh=args.sam_prompt_face_conf_thresh,
                                            sam_prompt_hand_conf_thresh=args.sam_prompt_hand_conf_thresh,
                                            sam_prompt_face_min_points=args.sam_prompt_face_min_points,
                                            sam_prompt_hand_min_points=args.sam_prompt_hand_min_points,
                                            sam_use_negative_points=args.sam_use_negative_points,
                                            sam_negative_margin=args.sam_negative_margin,
                                            sam_reprompt_interval=args.sam_reprompt_interval,
                                            sam_prompt_mode=args.sam_prompt_mode,
                                            sam_debug_trace=args.sam_debug_trace,
                                            sam_debug_trace_dir=args.sam_debug_trace_dir,
                                            soft_mask_mode=args.soft_mask_mode,
                                            soft_mask_band_width=args.soft_mask_band_width,
                                            soft_mask_blur_kernel=args.soft_mask_blur_kernel,
                                            boundary_fusion_mode=args.boundary_fusion_mode,
                                            parsing_mode=args.parsing_mode,
                                            matting_mode=args.matting_mode,
                                            parsing_head_expand=args.parsing_head_expand,
                                            parsing_hand_radius_ratio=args.parsing_hand_radius_ratio,
                                            parsing_boundary_kernel=args.parsing_boundary_kernel,
                                            matting_trimap_inner_erode=args.matting_trimap_inner_erode,
                                            matting_trimap_outer_dilate=args.matting_trimap_outer_dilate,
                                            matting_blur_kernel=args.matting_blur_kernel,
                                            alpha_v2_detail_boost=args.alpha_v2_detail_boost,
                                            alpha_v2_shrink_strength=args.alpha_v2_shrink_strength,
                                            alpha_v2_hair_boost=args.alpha_v2_hair_boost,
                                            alpha_v2_hard_threshold=args.alpha_v2_hard_threshold,
                                            alpha_v2_bilateral_sigma_color=args.alpha_v2_bilateral_sigma_color,
                                            alpha_v2_bilateral_sigma_space=args.alpha_v2_bilateral_sigma_space,
                                            alpha_v3_detail_boost=args.alpha_v3_detail_boost,
                                            alpha_v3_color_mix=args.alpha_v3_color_mix,
                                            alpha_v3_active_blend=args.alpha_v3_active_blend,
                                            alpha_v3_delta_clip=args.alpha_v3_delta_clip,
                                            bg_inpaint_mode=args.bg_inpaint_mode,
                                            bg_inpaint_method=args.bg_inpaint_method,
                                            bg_inpaint_mask_expand=args.bg_inpaint_mask_expand,
                                            bg_inpaint_radius=args.bg_inpaint_radius,
                                            bg_temporal_smooth_strength=args.bg_temporal_smooth_strength,
                                            bg_video_window_radius=args.bg_video_window_radius,
                                            bg_video_min_visible_count=args.bg_video_min_visible_count,
                                            bg_video_blend_strength=args.bg_video_blend_strength,
                                            bg_video_global_min_visible_count=args.bg_video_global_min_visible_count,
                                            bg_video_confidence_threshold=args.bg_video_confidence_threshold,
                                            bg_video_global_blend_strength=args.bg_video_global_blend_strength,
                                            bg_video_consistency_scale=args.bg_video_consistency_scale,
                                            reference_normalization_mode=args.reference_normalization_mode,
                                            reference_target_bbox_source=args.reference_target_bbox_source,
                                            reference_target_bbox_frames=args.reference_target_bbox_frames,
                                            reference_bbox_conf_thresh=args.reference_bbox_conf_thresh,
                                            reference_scale_clamp_min=args.reference_scale_clamp_min,
                                            reference_scale_clamp_max=args.reference_scale_clamp_max,
                                            reference_structure_segment_clamp_min=args.reference_structure_segment_clamp_min,
                                            reference_structure_segment_clamp_max=args.reference_structure_segment_clamp_max,
                                            reference_structure_width_budget_ratio=args.reference_structure_width_budget_ratio,
                                            reference_structure_height_budget_ratio=args.reference_structure_height_budget_ratio,
                                            multistage_preprocess_mode=args.multistage_preprocess_mode,
                                            disable_person_roi_refine=args.disable_person_roi_refine,
                                            disable_face_roi_refine=args.disable_face_roi_refine,
                                            person_roi_stage_resolution_area=args.person_roi_stage_resolution_area,
                                            person_roi_stage_min_short_side=args.person_roi_stage_min_short_side,
                                            person_roi_expand_ratio=args.person_roi_expand_ratio,
                                            person_roi_min_size_ratio=args.person_roi_min_size_ratio,
                                            person_roi_target_long_side=args.person_roi_target_long_side,
                                            person_roi_body_conf_thresh=args.person_roi_body_conf_thresh,
                                            person_roi_hand_conf_thresh=args.person_roi_hand_conf_thresh,
                                            person_roi_face_conf_thresh=args.person_roi_face_conf_thresh,
                                            person_roi_fuse_weight=args.person_roi_fuse_weight,
                                            person_roi_conf_margin=args.person_roi_conf_margin,
                                            face_roi_stage_resolution_area=args.face_roi_stage_resolution_area,
                                            face_roi_stage_min_short_side=args.face_roi_stage_min_short_side,
                                            face_roi_expand_ratio=args.face_roi_expand_ratio,
                                            face_roi_min_size_ratio=args.face_roi_min_size_ratio,
                                            face_roi_target_long_side=args.face_roi_target_long_side,
                                            face_roi_conf_thresh=args.face_roi_conf_thresh,
                                            face_roi_fuse_weight=args.face_roi_fuse_weight,
                                            face_roi_conf_margin=args.face_roi_conf_margin,
                                            multistage_pose_extra_smooth=args.multistage_pose_extra_smooth,
                                            multistage_face_bbox_extra_smooth=args.multistage_face_bbox_extra_smooth,
                                            face_analysis_mode=args.face_analysis_mode,
                                            face_tracking_smooth_strength=args.face_tracking_smooth_strength,
                                            face_tracking_max_scale_change=args.face_tracking_max_scale_change,
                                            face_tracking_max_center_shift=args.face_tracking_max_center_shift,
                                            face_tracking_hold_frames=args.face_tracking_hold_frames,
                                            face_difficulty_expand_ratio=args.face_difficulty_expand_ratio,
                                            face_rerun_difficulty_threshold=args.face_rerun_difficulty_threshold,
                                            face_alpha_blur_kernel=args.face_alpha_blur_kernel,
                                            pose_motion_stack_mode=args.pose_motion_stack_mode,
                                            pose_motion_body_bidirectional_strength=args.pose_motion_body_bidirectional_strength,
                                            pose_motion_hand_bidirectional_strength=args.pose_motion_hand_bidirectional_strength,
                                            pose_motion_face_bidirectional_strength=args.pose_motion_face_bidirectional_strength,
                                            pose_motion_local_refine_strength=args.pose_motion_local_refine_strength,
                                            pose_motion_limb_roi_expand_ratio=args.pose_motion_limb_roi_expand_ratio,
                                            pose_motion_hand_roi_expand_ratio=args.pose_motion_hand_roi_expand_ratio,
                                            pose_motion_velocity_spike_quantile=args.pose_motion_velocity_spike_quantile,
                                            pose_motion_uncertainty_blur_kernel=args.pose_motion_uncertainty_blur_kernel,
                                            preprocess_runtime_profile=args.preprocess_runtime_profile,
                                            iterations=args.iterations,
                                            k=args.k,
                                            w_len=args.w_len,
                                            h_len=args.h_len,
                                            retarget_flag=args.retarget_flag,
                                            use_flux=args.use_flux,
                                            replace_flag=args.replace_flag)
        metadata = build_preprocess_metadata(
            video_path=args.video_path,
            refer_image_path=args.refer_path,
            output_path=args.save_path,
            replace_flag=args.replace_flag,
            retarget_flag=args.retarget_flag,
            use_flux=args.use_flux,
            resolution_area=args.resolution_area,
            analysis_settings={
                "preprocess_runtime_profile": args.preprocess_runtime_profile,
                "resolved_profile": preprocess_runtime_profile,
                "analysis_resolution_area": args.analysis_resolution_area,
                "analysis_min_short_side": args.analysis_min_short_side,
                "analysis_height": pipeline_outputs.get("analysis_height"),
                "analysis_width": pipeline_outputs.get("analysis_width"),
                "export_height": pipeline_outputs["height"],
                "export_width": pipeline_outputs["width"],
            },
            fps_request=args.fps,
            fps_output=pipeline_outputs["fps"],
            frame_count=pipeline_outputs["frame_count"],
            height=pipeline_outputs["height"],
            width=pipeline_outputs["width"],
            iterations=args.iterations,
            k=args.k,
            w_len=args.w_len,
            h_len=args.h_len,
            reference_height=pipeline_outputs["reference_height"],
            reference_width=pipeline_outputs["reference_width"],
            src_files=pipeline_outputs["src_files"],
            intermediate_save_format=args.save_format,
            lossless_intermediate=args.lossless_intermediate,
            control_stabilization={
                "face_conf_thresh": args.face_conf_thresh,
                "face_min_valid_points": args.face_min_valid_points,
                "face_bbox_smooth_method": args.face_bbox_smooth_method,
                "face_bbox_smooth_strength": args.face_bbox_smooth_strength,
                "face_bbox_max_scale_change": args.face_bbox_max_scale_change,
                "face_bbox_max_center_shift": args.face_bbox_max_center_shift,
                "face_bbox_hold_frames": args.face_bbox_hold_frames,
                "pose_smooth_method": args.pose_smooth_method,
                "pose_conf_thresh_body": args.pose_conf_thresh_body,
                "pose_conf_thresh_hand": args.pose_conf_thresh_hand,
                "pose_conf_thresh_face": args.pose_conf_thresh_face,
                "pose_smooth_strength_body": args.pose_smooth_strength_body,
                "pose_smooth_strength_hand": args.pose_smooth_strength_hand,
                "pose_smooth_strength_face": args.pose_smooth_strength_face,
                "pose_max_velocity_body": args.pose_max_velocity_body,
                "pose_max_velocity_hand": args.pose_max_velocity_hand,
                "pose_max_velocity_face": args.pose_max_velocity_face,
                "pose_interp_max_gap": args.pose_interp_max_gap,
                "export_qa_visuals": args.export_qa_visuals,
            },
            mask_generation={
                "sam_chunk_len": args.sam_chunk_len,
                "sam_keyframes_per_chunk": args.sam_keyframes_per_chunk,
                "sam_prompt_body_conf_thresh": args.sam_prompt_body_conf_thresh,
                "sam_prompt_face_conf_thresh": args.sam_prompt_face_conf_thresh,
                "sam_prompt_hand_conf_thresh": args.sam_prompt_hand_conf_thresh,
                "sam_prompt_face_min_points": args.sam_prompt_face_min_points,
                "sam_prompt_hand_min_points": args.sam_prompt_hand_min_points,
                "sam_use_negative_points": args.sam_use_negative_points,
                "sam_negative_margin": args.sam_negative_margin,
                "sam_reprompt_interval": args.sam_reprompt_interval,
                "sam_prompt_mode": args.sam_prompt_mode,
                "preprocess_runtime_profile": args.preprocess_runtime_profile,
                "sam_runtime_profile": args.sam_runtime_profile,
                "sam_apply_postprocessing": args.sam_apply_postprocessing,
                "sam_use_flash_attn": args.sam_use_flash_attn,
                "sam_math_kernel_on": args.sam_math_kernel_on,
                "sam_old_gpu_mode": args.sam_old_gpu_mode,
                "sam_offload_video_to_cpu": args.sam_offload_video_to_cpu,
                "sam_offload_state_to_cpu": args.sam_offload_state_to_cpu,
                "sam_debug_trace": args.sam_debug_trace,
                "sam_debug_trace_dir": pipeline_outputs.get("sam_trace_dir"),
                "sam_runtime_resolved": pipeline_outputs.get("sam_runtime", {}),
                "sam_apply_postprocessing_resolved": pipeline_outputs.get("sam_apply_postprocessing"),
            },
            soft_mask_settings={
                "soft_mask_mode": args.soft_mask_mode,
                "soft_mask_band_width": args.soft_mask_band_width,
                "soft_mask_blur_kernel": args.soft_mask_blur_kernel,
            },
            background_settings={
                "bg_inpaint_mode": args.bg_inpaint_mode,
                "bg_inpaint_method": args.bg_inpaint_method,
                "bg_inpaint_mask_expand": args.bg_inpaint_mask_expand,
                "bg_inpaint_radius": args.bg_inpaint_radius,
                "bg_temporal_smooth_strength": args.bg_temporal_smooth_strength,
                "bg_video_window_radius": args.bg_video_window_radius,
                "bg_video_min_visible_count": args.bg_video_min_visible_count,
                "bg_video_blend_strength": args.bg_video_blend_strength,
                "bg_video_global_min_visible_count": args.bg_video_global_min_visible_count,
                "bg_video_confidence_threshold": args.bg_video_confidence_threshold,
                "bg_video_global_blend_strength": args.bg_video_global_blend_strength,
                "bg_video_consistency_scale": args.bg_video_consistency_scale,
                "stats": pipeline_outputs.get("background", {}),
            },
            boundary_fusion_settings={
                "boundary_fusion_mode": args.boundary_fusion_mode,
                "parsing_mode": args.parsing_mode,
                "matting_mode": args.matting_mode,
                "parsing_head_expand": args.parsing_head_expand,
                "parsing_hand_radius_ratio": args.parsing_hand_radius_ratio,
                "parsing_boundary_kernel": args.parsing_boundary_kernel,
                "matting_trimap_inner_erode": args.matting_trimap_inner_erode,
                "matting_trimap_outer_dilate": args.matting_trimap_outer_dilate,
                "matting_blur_kernel": args.matting_blur_kernel,
                "alpha_v2_detail_boost": args.alpha_v2_detail_boost,
                "alpha_v2_shrink_strength": args.alpha_v2_shrink_strength,
                "alpha_v2_hair_boost": args.alpha_v2_hair_boost,
                "alpha_v2_hard_threshold": args.alpha_v2_hard_threshold,
                "alpha_v2_bilateral_sigma_color": args.alpha_v2_bilateral_sigma_color,
                "alpha_v2_bilateral_sigma_space": args.alpha_v2_bilateral_sigma_space,
                "alpha_v3_detail_boost": args.alpha_v3_detail_boost,
                "alpha_v3_color_mix": args.alpha_v3_color_mix,
                "alpha_v3_active_blend": args.alpha_v3_active_blend,
                "alpha_v3_delta_clip": args.alpha_v3_delta_clip,
                "stats": pipeline_outputs.get("boundary_fusion", {}),
            },
            reference_settings={
                "reference_normalization_mode": args.reference_normalization_mode,
                "reference_target_bbox_source": args.reference_target_bbox_source,
                "reference_target_bbox_frames": args.reference_target_bbox_frames,
                "reference_bbox_conf_thresh": args.reference_bbox_conf_thresh,
                "reference_scale_clamp_min": args.reference_scale_clamp_min,
                "reference_scale_clamp_max": args.reference_scale_clamp_max,
                "reference_structure_segment_clamp_min": args.reference_structure_segment_clamp_min,
                "reference_structure_segment_clamp_max": args.reference_structure_segment_clamp_max,
                "reference_structure_width_budget_ratio": args.reference_structure_width_budget_ratio,
                "reference_structure_height_budget_ratio": args.reference_structure_height_budget_ratio,
                "stats": pipeline_outputs.get("reference_normalization", {}),
            },
            runtime_stats=pipeline_outputs.get("runtime_stats", {}),
            qa_outputs=pipeline_outputs.get("qa_outputs", {}),
        )
        metadata["processing"]["multistage"] = {
            "multistage_preprocess_mode": args.multistage_preprocess_mode,
            "disable_person_roi_refine": args.disable_person_roi_refine,
            "disable_face_roi_refine": args.disable_face_roi_refine,
            "person_roi_stage_resolution_area": args.person_roi_stage_resolution_area,
            "person_roi_stage_min_short_side": args.person_roi_stage_min_short_side,
            "person_roi_expand_ratio": args.person_roi_expand_ratio,
            "person_roi_min_size_ratio": args.person_roi_min_size_ratio,
            "person_roi_target_long_side": args.person_roi_target_long_side,
            "person_roi_body_conf_thresh": args.person_roi_body_conf_thresh,
            "person_roi_hand_conf_thresh": args.person_roi_hand_conf_thresh,
            "person_roi_face_conf_thresh": args.person_roi_face_conf_thresh,
            "person_roi_fuse_weight": args.person_roi_fuse_weight,
            "person_roi_conf_margin": args.person_roi_conf_margin,
            "face_roi_stage_resolution_area": args.face_roi_stage_resolution_area,
            "face_roi_stage_min_short_side": args.face_roi_stage_min_short_side,
            "face_roi_expand_ratio": args.face_roi_expand_ratio,
            "face_roi_min_size_ratio": args.face_roi_min_size_ratio,
            "face_roi_target_long_side": args.face_roi_target_long_side,
            "face_roi_conf_thresh": args.face_roi_conf_thresh,
            "face_roi_fuse_weight": args.face_roi_fuse_weight,
            "face_roi_conf_margin": args.face_roi_conf_margin,
            "multistage_pose_extra_smooth": args.multistage_pose_extra_smooth,
            "multistage_face_bbox_extra_smooth": args.multistage_face_bbox_extra_smooth,
            "stats": pipeline_outputs.get("multistage", {}),
        }
        metadata["processing"]["face_analysis"] = {
            "face_analysis_mode": args.face_analysis_mode,
            "face_tracking_smooth_strength": args.face_tracking_smooth_strength,
            "face_tracking_max_scale_change": args.face_tracking_max_scale_change,
            "face_tracking_max_center_shift": args.face_tracking_max_center_shift,
            "face_tracking_hold_frames": args.face_tracking_hold_frames,
            "face_difficulty_expand_ratio": args.face_difficulty_expand_ratio,
            "face_rerun_difficulty_threshold": args.face_rerun_difficulty_threshold,
            "face_alpha_blur_kernel": args.face_alpha_blur_kernel,
            "stats": pipeline_outputs.get("face_analysis", {}),
        }
        metadata["processing"]["pose_motion_stack"] = {
            "pose_motion_stack_mode": args.pose_motion_stack_mode,
            "pose_motion_body_bidirectional_strength": args.pose_motion_body_bidirectional_strength,
            "pose_motion_hand_bidirectional_strength": args.pose_motion_hand_bidirectional_strength,
            "pose_motion_face_bidirectional_strength": args.pose_motion_face_bidirectional_strength,
            "pose_motion_local_refine_strength": args.pose_motion_local_refine_strength,
            "pose_motion_limb_roi_expand_ratio": args.pose_motion_limb_roi_expand_ratio,
            "pose_motion_hand_roi_expand_ratio": args.pose_motion_hand_roi_expand_ratio,
            "pose_motion_velocity_spike_quantile": args.pose_motion_velocity_spike_quantile,
            "pose_motion_uncertainty_blur_kernel": args.pose_motion_uncertainty_blur_kernel,
            "stats": pipeline_outputs.get("pose_motion_stack", {}),
        }
        metadata_path = write_preprocess_metadata(args.save_path, metadata)
        runtime_stats_path = None
        if pipeline_outputs.get("runtime_stats"):
            runtime_stats_path = Path(args.save_path) / "preprocess_runtime_stats.json"
            runtime_stats_path.write_text(
                json.dumps(pipeline_outputs["runtime_stats"], indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        if manifest_token is not None:
            output_dir = Path(args.save_path).resolve()
            stage_outputs = {
                "save_path": str(output_dir),
                "generated_files": sorted(str(path.resolve()) for path in output_dir.iterdir()),
                "metadata_path": str(metadata_path.resolve()),
            }
            if runtime_stats_path is not None and runtime_stats_path.exists():
                stage_outputs["runtime_stats"] = str(runtime_stats_path.resolve())
            for artifact_name, artifact in pipeline_outputs["src_files"].items():
                path = output_dir / artifact["path"]
                if path.exists():
                    stage_outputs[artifact_name] = str(path.resolve())
            metadata_file = output_dir / "metadata.json"
            if metadata_file.exists():
                stage_outputs["metadata"] = str(metadata_file.resolve())
            if "qa_outputs" in pipeline_outputs:
                for qa_name, qa_path in pipeline_outputs["qa_outputs"].items():
                    stage_outputs[qa_name] = str((output_dir / qa_path).resolve())
            if pipeline_outputs.get("sam_trace_dir") is not None:
                stage_outputs["sam_trace_dir"] = str(Path(pipeline_outputs["sam_trace_dir"]).resolve())
            finalize_stage_manifest(
                run_layout,
                manifest_token,
                status="completed",
                outputs=stage_outputs,
                metrics=pipeline_outputs.get("runtime_metrics", {}),
            )
    except Exception as exc:
        if manifest_token is not None:
            failure_outputs = {"save_path": str(Path(args.save_path).resolve())}
            trace_dir = Path(args.sam_debug_trace_dir) if args.sam_debug_trace_dir is not None else Path(args.save_path) / "sam2_debug"
            if trace_dir.exists():
                failure_outputs["sam_trace_dir"] = str(trace_dir.resolve())
            finalize_stage_manifest(
                run_layout,
                manifest_token,
                status="failed",
                outputs=failure_outputs,
                error=str(exc),
            )
        raise
