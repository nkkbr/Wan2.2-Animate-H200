# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import os
import numpy as np
import torch
from diffusers import FluxKontextPipeline
import cv2
from loguru import logger
from PIL import Image

from decord import VideoReader
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
from wan.utils.animate_contract import SOFT_BAND_SEMANTICS, load_image_rgb, validate_rgb_video
from wan.utils.media_io import write_person_mask_artifact, write_rgb_artifact
from wan.utils.replacement_masks import build_soft_boundary_band
from background_clean_plate import build_clean_plate_background
from reference_normalization import (
    bbox_from_pose_meta,
    estimate_driver_target_bbox,
    make_reference_normalization_preview,
    normalize_reference_image,
    project_bbox_with_letterbox,
    scale_bbox_between_shapes,
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
        bg_inpaint_mode="none",
        bg_inpaint_method="telea",
        bg_inpaint_mask_expand=16,
        bg_inpaint_radius=5.0,
        bg_temporal_smooth_strength=0.0,
        reference_normalization_mode="none",
        reference_target_bbox_source="median_first_n",
        reference_target_bbox_frames=16,
        reference_bbox_conf_thresh=0.35,
        reference_scale_clamp_min=0.75,
        reference_scale_clamp_max=1.6,
        iterations=3,
        k=7,
        w_len=1,
        h_len=1,
        retarget_flag=False,
        use_flux=False,
        replace_flag=False,
    ):
        if replace_flag:

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
            validate_rgb_video("replacement input video", frames)

            frames = [resize_by_area(frame, resolution_area[0] * resolution_area[1], divisor=16) for frame in frames]
            height, width = frames[0].shape[:2]
            logger.info(f"Processing pose meta")

            raw_pose_metas = self.pose2d(frames)
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
                image_shape=(height, width),
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

            logger.info(f"Processing reference image: {refer_image_path}")
            refer_img = load_image_rgb(refer_image_path)
            src_ref_path = os.path.join(output_path, 'src_ref.png')
            write_reference_image(src_ref_path, refer_img)
            reference_original_height, reference_original_width = refer_img.shape[:2]
            reference_bbox_detection = None
            reference_bbox_original = None
            driver_target_bbox = None
            driver_target_stats = {
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
                "reference_detection_shape": None,
                "reference_bbox_detection": None,
                "reference_bbox_original": None,
            }
            if reference_normalization_mode == "bbox_match":
                reference_detection_img = resize_by_area(
                    refer_img,
                    resolution_area[0] * resolution_area[1],
                    divisor=16,
                )
                reference_pose_meta = self.pose2d([reference_detection_img])[0]
                reference_bbox_detection = bbox_from_pose_meta(
                    reference_pose_meta,
                    image_shape=reference_detection_img.shape[:2],
                    conf_thresh=reference_bbox_conf_thresh,
                )
                reference_bbox_original = scale_bbox_between_shapes(
                    reference_bbox_detection,
                    from_shape=reference_detection_img.shape[:2],
                    to_shape=refer_img.shape[:2],
                )
                driver_target_bbox, driver_target_stats = estimate_driver_target_bbox(
                    tpl_pose_metas,
                    image_shape=(height, width),
                    source=reference_target_bbox_source,
                    num_frames=reference_target_bbox_frames,
                    conf_thresh=reference_bbox_conf_thresh,
                )
                reference_normalization.update({
                    "driver_target_bbox_stats": driver_target_stats,
                    "reference_detection_shape": [
                        int(reference_detection_img.shape[0]),
                        int(reference_detection_img.shape[1]),
                    ],
                    "reference_bbox_detection": None if reference_bbox_detection is None else [float(v) for v in reference_bbox_detection.tolist()],
                    "reference_bbox_original": None if reference_bbox_original is None else [float(v) for v in reference_bbox_original.tolist()],
                })
                normalized_reference, normalization_stats = normalize_reference_image(
                    refer_img,
                    reference_bbox=reference_bbox_original,
                    target_bbox=driver_target_bbox,
                    canvas_shape=(height, width),
                    scale_clamp_min=reference_scale_clamp_min,
                    scale_clamp_max=reference_scale_clamp_max,
                )
                reference_normalization.update(normalization_stats)
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
                        "frame_count": int(len(frames)),
                        "frame_shape": list(frames[0].shape),
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
                    },
                },
            )
            masks, mask_debug = self.get_mask(
                frames,
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

            hole_bg_images = []
            aug_masks = []

            for frame, mask in zip(frames, masks):
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
            soft_band_masks = None
            if soft_mask_mode != "none":
                soft_band_masks = build_soft_boundary_band(
                    aug_masks,
                    band_width=soft_mask_band_width,
                    blur_kernel_size=soft_mask_blur_kernel,
                ).astype(np.float32)
            bg_images, background_debug = build_clean_plate_background(
                np.stack(frames).astype(np.uint8),
                aug_masks,
                bg_inpaint_mode=bg_inpaint_mode,
                soft_band=soft_band_masks,
                bg_inpaint_mask_expand=bg_inpaint_mask_expand,
                bg_inpaint_radius=bg_inpaint_radius,
                bg_inpaint_method=bg_inpaint_method,
                bg_temporal_smooth_strength=bg_temporal_smooth_strength,
            )
            bg_images = np.stack(bg_images).astype(np.uint8)
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
                mask_overlay = make_mask_overlay(np.stack(frames).astype(np.uint8), masks, mask_debug["prompt_entries"])
                prompt_overlay = make_sam_prompts_overlay(np.stack(frames).astype(np.uint8), mask_debug["prompt_entries"])
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
                prompt_keyframes = write_prompt_keyframes(output_path, np.stack(frames).astype(np.uint8), mask_debug["prompt_entries"])
                write_curve_json(os.path.join(output_path, "face_bbox_curve.json"), face_bbox_curve)
                write_curve_json(os.path.join(output_path, "pose_conf_curve.json"), pose_conf_curve)
                write_mask_json(os.path.join(output_path, "mask_stats.json"), mask_debug["mask_stats"])
                qa_outputs = {
                    "face_bbox_overlay": qa_face_overlay["path"],
                    "pose_overlay": qa_pose_overlay["path"],
                    "face_bbox_curve": "face_bbox_curve.json",
                    "pose_conf_curve": "pose_conf_curve.json",
                    "mask_overlay": qa_mask_overlay["path"],
                    "sam_prompts_overlay": qa_prompt_overlay["path"],
                    "sam_prompt_keyframes": prompt_keyframes["path"],
                    "mask_stats": "mask_stats.json",
                }
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
                qa_background_mask = write_person_mask_artifact(
                    mask_frames=background_debug["inpaint_mask"].astype(np.float32),
                    output_root=output_path,
                    stem="background_inpaint_mask",
                    artifact_format="mp4",
                    fps=fps,
                    mask_semantics="background_inpaint_region",
                )
                qa_outputs.update({
                    "background_hole": qa_background_hole["path"],
                    "background_clean_plate": qa_background_clean["path"],
                    "background_diff": qa_background_diff["path"],
                    "background_inpaint_mask": qa_background_mask["path"],
                })
                if reference_normalization_mode != "none":
                    original_canvas = padding_resize(refer_img, height, width)
                    original_bbox_canvas = project_bbox_with_letterbox(
                        reference_bbox_original,
                        image_shape=refer_img.shape[:2],
                        canvas_shape=(height, width),
                    )
                    normalized_bbox = reference_normalization.get("normalized_bbox")
                    preview = make_reference_normalization_preview(
                        original_canvas=original_canvas,
                        normalized_canvas=refer_canvas,
                        original_bbox=original_bbox_canvas,
                        target_bbox=driver_target_bbox,
                        normalized_bbox=normalized_bbox,
                    )
                    preview_path = os.path.join(output_path, "reference_normalization_preview.png")
                    write_reference_image(preview_path, preview)
                    qa_outputs["reference_normalization_preview"] = "reference_normalization_preview.png"

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
                    mask_frames=aug_masks,
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
            src_files["background"]["background_mode"] = background_debug["background_mode"]
            outputs = {
                "frame_count": len(frames),
                "fps": float(fps),
                "height": int(height),
                "width": int(width),
                "channels": 3,
                "reference_height": int(reference_height),
                "reference_width": int(reference_width),
                "src_files": src_files,
                "reference_normalization": reference_normalization,
                "sam_runtime": dict(self.sam_runtime_config),
                "sam_apply_postprocessing": bool(self.sam_apply_postprocessing),
                "sam_trace_dir": str(trace_dir) if trace_dir is not None else None,
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
        for index in range(num_step):
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
                if sam_debug_trace and sam_debug_trace_dir is not None:
                    write_chunk_trace(sam_debug_trace_dir, index, chunk_trace)
                self.predictor.reset_state(inference_state)
                ann_obj_id = 1
                for prompt_offset, prompt_entry in enumerate(prompt_entries):
                    ann_frame_idx = prompt_entry["frame_idx"]
                    chunk_trace["stage"] = "before_add_new_points" if sam_prompt_mode == "points" else "before_add_new_mask"
                    chunk_trace["current_prompt_index"] = int(prompt_offset)
                    chunk_trace["current_prompt_frame_idx"] = int(ann_frame_idx)
                    if sam_debug_trace and sam_debug_trace_dir is not None:
                        write_chunk_trace(sam_debug_trace_dir, index, chunk_trace)
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
                    chunk_trace["current_prompt_obj_count"] = int(len(out_obj_ids))
                    chunk_trace["current_prompt_mask_logits_shape"] = list(out_mask_logits.shape)
                    if sam_debug_trace and sam_debug_trace_dir is not None:
                        write_chunk_trace(sam_debug_trace_dir, index, chunk_trace)

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
                if sam_debug_trace and sam_debug_trace_dir is not None:
                    write_chunk_trace(sam_debug_trace_dir, index, chunk_trace)
            except Exception as exc:
                chunk_trace["stage"] = "failed"
                chunk_trace["error"] = str(exc)
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

    def convert_list_to_array(self, metas):
        metas_list = []
        for meta in metas:
            for key, value in meta.items():
                if type(value) is list:
                    value = np.array(value)
                meta[key] = value
            metas_list.append(meta)
        return metas_list
