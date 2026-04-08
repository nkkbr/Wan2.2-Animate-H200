# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import json
import logging
import math
import os
import time
import cv2
import types
from copy import deepcopy
from functools import partial
from pathlib import Path
from einops import rearrange
import numpy as np
import torch

import torch.distributed as dist
from peft import set_peft_model_state_dict
from tqdm import tqdm
from .distributed.fsdp import shard_model
from .distributed.sequence_parallel import sp_attn_forward, sp_dit_forward
from .distributed.util import get_world_size

from .modules.animate import WanAnimateModel
from .modules.animate import CLIPModel
from .modules.t5 import T5EncoderModel
from .modules.vae2_1 import Wan2_1_VAE
from .modules.animate.animate_utils import TensorList, get_loraconfig
from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .utils.animate_contract import (
    BACKGROUND_KEEP_MASK_SEMANTICS,
    BACKGROUND_KEEP_PRIOR_SEMANTICS,
    BOUNDARY_BAND_SEMANTICS,
    HARD_FOREGROUND_SEMANTICS,
    OCCLUSION_BAND_SEMANTICS,
    PERSON_MASK_SEMANTICS,
    SOFT_ALPHA_SEMANTICS,
    UNCERTAINTY_MAP_SEMANTICS,
    load_image_rgb,
    read_video_rgb,
    resolve_preprocess_artifacts,
    validate_loaded_preprocess_bundle,
    validate_refert_num,
)
from .utils.boundary_refinement import (
    refine_boundary_frames,
    rgb_frames_to_tensor_video,
    tensor_video_to_rgb_frames,
    write_boundary_refinement_debug_artifacts,
)
from .utils.clip_blending import (
    blend_clip_overlap,
    mean_abs_difference,
    summarize_scalar_series,
    write_seam_debug_artifacts,
    write_seam_summary,
)
from .utils.guidance import combine_animate_guidance_predictions
from .utils.media_io import load_mask_artifact, load_person_mask_artifact, load_rgb_artifact, write_person_mask_artifact
from .utils.replacement_masks import compose_background_keep_mask, derive_replacement_regions, resize_mask_volume
from .utils.temporal_handoff import (
    compose_temporal_handoff_latents,
    pack_overlap_tensor_to_latent_slots,
    write_temporal_handoff_debug,
)



class WanAnimate:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        init_on_cpu=True,
        convert_model_dtype=False,
        use_relighting_lora=False
    ):
        r"""
        Initializes the generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_sp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of sequence parallel.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            init_on_cpu (`bool`, *optional*, defaults to True):
                Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
            convert_model_dtype (`bool`, *optional*, defaults to False):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.
            use_relighting_lora (`bool`, *optional*, defaults to False):
               Whether to use relighting lora for character replacement. 
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.init_on_cpu = init_on_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        if t5_fsdp or dit_fsdp or use_sp:
            self.init_on_cpu = False

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None,
        )

        self.clip = CLIPModel(
            dtype=torch.float16,
            device=self.device,
            checkpoint_path=os.path.join(checkpoint_dir,
                                         config.clip_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.clip_tokenizer))

        self.vae = Wan2_1_VAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        logging.info(f"Creating WanAnimate from {checkpoint_dir}")

        if not dit_fsdp:
            self.noise_model = WanAnimateModel.from_pretrained(
                checkpoint_dir,
                torch_dtype=self.param_dtype,
                device_map=self.device)
        else:
            self.noise_model = WanAnimateModel.from_pretrained(
                checkpoint_dir, torch_dtype=self.param_dtype)

        self.noise_model = self._configure_model(
            model=self.noise_model,
            use_sp=use_sp,
            dit_fsdp=dit_fsdp,
            shard_fn=shard_fn,
            convert_model_dtype=convert_model_dtype,
            use_lora=use_relighting_lora,
            checkpoint_dir=checkpoint_dir,
            config=config
            )

        if use_sp:
            self.sp_size = get_world_size()
        else:
            self.sp_size = 1

        self.sample_neg_prompt = config.sample_neg_prompt
        self.sample_prompt = config.prompt
        self.last_runtime_stats = None


    def _configure_model(self, model, use_sp, dit_fsdp, shard_fn,
                         convert_model_dtype, use_lora, checkpoint_dir, config):
        """
        Configures a model object. This includes setting evaluation modes,
        applying distributed parallel strategy, and handling device placement.

        Args:
            model (torch.nn.Module):
                The model instance to configure.
            use_sp (`bool`):
                Enable distribution strategy of sequence parallel.
            dit_fsdp (`bool`):
                Enable FSDP sharding for DiT model.
            shard_fn (callable):
                The function to apply FSDP sharding.
            convert_model_dtype (`bool`):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.

        Returns:
            torch.nn.Module:
                The configured model.
        """
        model.eval().requires_grad_(False)

        if use_sp:
            for block in model.blocks:
                block.self_attn.forward = types.MethodType(
                    sp_attn_forward, block.self_attn)

            model.use_context_parallel = True

        if dist.is_initialized():
            dist.barrier()

        if use_lora:
            logging.info("Loading Relighting Lora. ")
            lora_config = get_loraconfig(
                transformer=model,
                rank=128,
                alpha=128
            )
            model.add_adapter(lora_config)
            lora_path = os.path.join(checkpoint_dir, config.lora_checkpoint)
            peft_state_dict = torch.load(lora_path)["state_dict"]
            set_peft_model_state_dict(model, peft_state_dict)

        if dit_fsdp:
            model = shard_fn(model, use_lora=use_lora)
        else:
            if convert_model_dtype:
                model.to(self.param_dtype)
            if not self.init_on_cpu:
                model.to(self.device)

        return model

    def inputs_padding(self, array, target_len):
        idx = 0
        flip = False
        target_array = []
        while len(target_array) < target_len:
            target_array.append(deepcopy(array[idx]))
            if flip:
                idx -= 1
            else:
                idx += 1
            if idx == 0 or idx == len(array) - 1:
                flip = not flip
        return target_array[:target_len]

    def get_valid_len(self, real_len, clip_len=81, overlap=1):
        real_clip_len = clip_len - overlap
        last_clip_num = (real_len - overlap) % real_clip_len
        if last_clip_num == 0:
            extra = 0
        else:
            extra = real_clip_len - last_clip_num
        target_len = real_len + extra
        return target_len


    def get_i2v_mask(self, lat_t, lat_h, lat_w, mask_len=1, mask_pixel_values=None, device="cuda"):
        if mask_pixel_values is None:
            msk = torch.zeros(1, (lat_t-1) * 4 + 1, lat_h, lat_w, device=device)
        else:
            msk = mask_pixel_values.clone()
        msk[:, :mask_len] = 1
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]
        return msk

    def padding_resize(self, img_ori, height=512, width=512, padding_color=(0, 0, 0), interpolation=cv2.INTER_LINEAR):
        ori_height = img_ori.shape[0]
        ori_width = img_ori.shape[1]
        channel = img_ori.shape[2]

        img_pad = np.zeros((height, width, channel))
        if channel == 1:
            img_pad[:, :, 0] = padding_color[0]
        else:
            img_pad[:, :, 0] = padding_color[0]
            img_pad[:, :, 1] = padding_color[1]
            img_pad[:, :, 2] = padding_color[2]

        if (ori_height / ori_width) > (height / width):
            new_width = int(height / ori_height * ori_width)
            img = cv2.resize(img_ori, (new_width, height), interpolation=interpolation)
            padding = int((width - new_width) / 2)
            if len(img.shape) == 2:
                img = img[:, :, np.newaxis]  
            img_pad[:, padding: padding + new_width, :] = img
        else:
            new_height = int(width / ori_width * ori_height)
            img = cv2.resize(img_ori, (width, new_height), interpolation=interpolation)
            padding = int((height - new_height) / 2)
            if len(img.shape) == 2:
                img = img[:, :, np.newaxis]  
            img_pad[padding: padding + new_height, :, :] = img

        img_pad = np.uint8(img_pad)

        return img_pad

    def prepare_source(self, src_pose_artifact, src_face_artifact, src_ref_artifact):
        cond_images = load_rgb_artifact(src_pose_artifact["path"], src_pose_artifact.get("format"))
        face_images = load_rgb_artifact(src_face_artifact["path"], src_face_artifact.get("format"))
        height, width = cond_images[0].shape[:2]
        refer_images = load_image_rgb(src_ref_artifact["path"])
        if refer_images.shape[:2] != (height, width):
            refer_images = self.padding_resize(refer_images, height=height, width=width)
        return cond_images, face_images, refer_images
    
    def prepare_source_for_replace(self, src_bg_artifact, src_mask_artifact):
        bg_images = load_rgb_artifact(src_bg_artifact["path"], src_bg_artifact.get("format"))
        person_mask_images = load_person_mask_artifact(src_mask_artifact["path"], src_mask_artifact.get("format"))
        return bg_images, person_mask_images

    def _load_aligned_driver_video_frames(self, preprocess_metadata, *, frame_count, height, width):
        if preprocess_metadata is None:
            return None
        video_path = preprocess_metadata.get("source_inputs", {}).get("video_path")
        if not video_path:
            return None
        video_path = Path(video_path)
        if not video_path.exists():
            logging.warning("Boundary refinement requested original video, but source video does not exist: %s", video_path)
            return None

        try:
            from decord import VideoReader

            reader = VideoReader(str(video_path))
            video_fps = float(reader.get_avg_fps())
            target_fps = float(preprocess_metadata.get("fps", video_fps))
            indices = self._get_frame_indices(len(reader), video_fps, frame_count, target_fps)
            frames = reader.get_batch(indices).asnumpy()
        except Exception as exc:
            logging.warning("Falling back to full RGB video read for boundary refinement source video due to: %s", exc)
            try:
                frames = read_video_rgb(video_path)
            except Exception as read_exc:
                logging.warning("Failed to read source video for boundary refinement: %s", read_exc)
                return None
            if len(frames) < frame_count:
                frames = np.concatenate([frames, np.repeat(frames[-1:], repeats=frame_count - len(frames), axis=0)], axis=0)
            frames = frames[:frame_count]

        if frames.shape[1:3] != (height, width):
            frames = np.stack([self.padding_resize(frame, height=height, width=width) for frame in frames], axis=0)
        return np.asarray(frames, dtype=np.uint8)

    @staticmethod
    def _get_frame_indices(frame_num, video_fps, clip_length, target_fps):
        times = np.arange(0, clip_length, dtype=np.float32) / float(target_fps)
        frame_indices = np.round(times * float(video_fps)).astype(int)
        frame_indices = np.clip(frame_indices, 0, int(frame_num) - 1)
        return frame_indices.tolist()

    def _apply_boundary_refinement(
        self,
        *,
        videos,
        preprocess_metadata,
        bg_images,
        replacement_masks,
        real_frame_len,
        fps,
        save_debug_dir,
        boundary_refine_mode,
        boundary_refine_strength,
        boundary_refine_sharpen,
        boundary_refine_use_clean_plate,
    ):
        stats = {
            "mode_requested": boundary_refine_mode,
            "applied": False,
            "background_source": None,
        }
        if boundary_refine_mode == "none":
            return videos, stats

        generated_video = videos[0]
        generated_frames = tensor_video_to_rgb_frames(generated_video)
        if generated_frames.shape[0] != real_frame_len:
            raise ValueError(
                f"Boundary refinement expects {real_frame_len} frames, got generated_frames={generated_frames.shape[0]}."
            )

        background_mode = (
            preprocess_metadata.get("src_files", {}).get("background", {}).get(
                "background_mode",
                preprocess_metadata.get("processing", {}).get("background", {}).get("bg_inpaint_mode"),
            )
            if preprocess_metadata is not None else None
        )
        background_frames = None
        background_source = None
        if boundary_refine_use_clean_plate and bg_images is not None and background_mode not in {None, "none", "hole"}:
            background_frames = np.asarray(bg_images[:real_frame_len], dtype=np.uint8)
            background_source = "clean_plate"
        else:
            background_frames = self._load_aligned_driver_video_frames(
                preprocess_metadata,
                frame_count=real_frame_len,
                height=generated_frames.shape[1],
                width=generated_frames.shape[2],
            )
            if background_frames is not None:
                background_source = "source_video"
            elif bg_images is not None:
                background_frames = np.asarray(bg_images[:real_frame_len], dtype=np.uint8)
                background_source = "background_artifact_fallback"

        if background_frames is None:
            logging.warning("Boundary refinement skipped because no usable background source was available.")
            stats["skipped_reason"] = "missing_background_source"
            return videos, stats

        refined_frames, debug_data = refine_boundary_frames(
            generated_frames=generated_frames,
            background_frames=background_frames,
            person_mask=replacement_masks["person_mask"][:real_frame_len].cpu().numpy(),
            soft_band=(
                replacement_masks["boundary_band"][:real_frame_len].cpu().numpy()
                if replacement_masks["boundary_band"] is not None else None
            ),
            soft_alpha=(
                replacement_masks["soft_alpha"][:real_frame_len].cpu().numpy()
                if replacement_masks["soft_alpha"] is not None else None
            ),
            strength=boundary_refine_strength,
            sharpen=boundary_refine_sharpen,
        )
        refined_video = rgb_frames_to_tensor_video(refined_frames).unsqueeze(0).to(dtype=videos.dtype)
        stats.update({
            "applied": True,
            "background_source": background_source,
            "background_mode": background_mode,
            "strength": float(boundary_refine_strength),
            "sharpen": float(boundary_refine_sharpen),
            **debug_data["metrics"],
        })
        if save_debug_dir is not None:
            stats["debug_paths"] = write_boundary_refinement_debug_artifacts(
                save_debug_dir=save_debug_dir,
                fps=fps,
                generated_frames=generated_frames,
                refined_frames=refined_frames,
                background_frames=background_frames,
                person_mask=replacement_masks["person_mask"][:real_frame_len].cpu().numpy(),
                debug_data=debug_data,
            )
        return refined_video, stats

    def _prepare_replacement_masks(
        self,
        *,
        person_mask_images,
        soft_band_images,
        hard_foreground_images,
        soft_alpha_images,
        boundary_band_images,
        background_keep_prior_images,
        visible_support_images,
        unresolved_region_images,
        background_confidence_images,
        background_source_provenance_images,
        occlusion_band_images,
        uncertainty_map_images,
        lat_h,
        lat_w,
        mask_mode,
        boundary_strength,
        downsample_mode,
        transition_low,
        transition_high,
    ):
        person_mask = torch.as_tensor(np.asarray(person_mask_images), dtype=torch.float32)
        hard_foreground = (
            torch.as_tensor(np.asarray(hard_foreground_images), dtype=torch.float32)
            if hard_foreground_images is not None else person_mask
        )
        boundary_band = (
            torch.as_tensor(np.asarray(boundary_band_images), dtype=torch.float32)
            if boundary_band_images is not None else None
        )
        if boundary_band is None and soft_band_images is not None:
            boundary_band = torch.as_tensor(np.asarray(soft_band_images), dtype=torch.float32)
        soft_alpha = (
            torch.as_tensor(np.asarray(soft_alpha_images), dtype=torch.float32)
            if soft_alpha_images is not None else None
        )
        background_keep_prior = (
            torch.as_tensor(np.asarray(background_keep_prior_images), dtype=torch.float32)
            if background_keep_prior_images is not None else None
        )
        visible_support = (
            torch.as_tensor(np.asarray(visible_support_images), dtype=torch.float32)
            if visible_support_images is not None else None
        )
        unresolved_region = (
            torch.as_tensor(np.asarray(unresolved_region_images), dtype=torch.float32)
            if unresolved_region_images is not None else None
        )
        background_confidence = (
            torch.as_tensor(np.asarray(background_confidence_images), dtype=torch.float32)
            if background_confidence_images is not None else None
        )
        background_source_provenance = (
            torch.as_tensor(np.asarray(background_source_provenance_images), dtype=torch.float32)
            if background_source_provenance_images is not None else None
        )
        occlusion_band = (
            torch.as_tensor(np.asarray(occlusion_band_images), dtype=torch.float32)
            if occlusion_band_images is not None else None
        )
        uncertainty_map = (
            torch.as_tensor(np.asarray(uncertainty_map_images), dtype=torch.float32)
            if uncertainty_map_images is not None else None
        )
        background_keep = compose_background_keep_mask(
            hard_foreground,
            soft_band=boundary_band,
            soft_alpha=soft_alpha,
            background_keep_prior=background_keep_prior,
            visible_support=visible_support,
            unresolved_region=unresolved_region,
            background_confidence=background_confidence,
            mode=mask_mode,
            boundary_strength=boundary_strength,
        )
        background_keep_latent = resize_mask_volume(
            background_keep,
            output_size=(lat_h, lat_w),
            mode=downsample_mode,
        )
        soft_band_latent = None
        if boundary_band is not None:
            soft_band_latent = resize_mask_volume(
                boundary_band,
                output_size=(lat_h, lat_w),
                mode=downsample_mode,
            )
        soft_alpha_latent = None
        if soft_alpha is not None:
            soft_alpha_latent = resize_mask_volume(
                soft_alpha,
                output_size=(lat_h, lat_w),
                mode=downsample_mode,
            )
        pixel_regions = derive_replacement_regions(
            background_keep,
            transition_low=transition_low,
            transition_high=transition_high,
        )
        latent_regions = derive_replacement_regions(
            background_keep_latent,
            transition_low=transition_low,
            transition_high=transition_high,
        )
        return {
            "person_mask": hard_foreground,
            "hard_foreground": hard_foreground,
            "soft_band": boundary_band,
            "boundary_band": boundary_band,
            "soft_alpha": soft_alpha,
            "background_keep_prior": background_keep_prior,
            "visible_support": visible_support,
            "unresolved_region": unresolved_region,
            "background_confidence": background_confidence,
            "background_source_provenance": background_source_provenance,
            "occlusion_band": occlusion_band,
            "uncertainty_map": uncertainty_map,
            "background_keep": background_keep,
            "background_keep_latent": background_keep_latent,
            "soft_band_latent": soft_band_latent,
            "soft_alpha_latent": soft_alpha_latent,
            "pixel_regions": pixel_regions,
            "latent_regions": latent_regions,
        }

    def _write_replacement_mask_debug(
        self,
        *,
        save_debug_dir,
        fps,
        replacement_masks,
        real_frame_len,
    ):
        if save_debug_dir is None:
            return {}
        debug_dir = Path(save_debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        artifacts = {}
        mask_artifacts = {
            "person_mask_hard": replacement_masks["person_mask"][:real_frame_len].cpu().numpy(),
            "hard_foreground": replacement_masks["hard_foreground"][:real_frame_len].cpu().numpy(),
            "background_keep_mask": replacement_masks["background_keep"][:real_frame_len].cpu().numpy(),
            "transition_band": replacement_masks["pixel_regions"]["transition_band"][:real_frame_len].cpu().numpy(),
            "free_replacement_region": replacement_masks["pixel_regions"]["free_replacement"][:real_frame_len].cpu().numpy(),
            "replacement_strength": replacement_masks["pixel_regions"]["replacement_strength"][:real_frame_len].cpu().numpy(),
            "latent_background_keep_mask": replacement_masks["background_keep_latent"][:real_frame_len].cpu().numpy(),
            "latent_transition_band": replacement_masks["latent_regions"]["transition_band"][:real_frame_len].cpu().numpy(),
        }
        if replacement_masks["soft_band"] is not None:
            mask_artifacts["soft_band"] = replacement_masks["soft_band"][:real_frame_len].cpu().numpy()
            mask_artifacts["boundary_band"] = replacement_masks["soft_band"][:real_frame_len].cpu().numpy()
        if replacement_masks["soft_alpha"] is not None:
            mask_artifacts["soft_alpha"] = replacement_masks["soft_alpha"][:real_frame_len].cpu().numpy()
        if replacement_masks["background_keep_prior"] is not None:
            mask_artifacts["background_keep_prior"] = replacement_masks["background_keep_prior"][:real_frame_len].cpu().numpy()
        if replacement_masks["visible_support"] is not None:
            mask_artifacts["visible_support"] = replacement_masks["visible_support"][:real_frame_len].cpu().numpy()
        if replacement_masks["unresolved_region"] is not None:
            mask_artifacts["unresolved_region"] = replacement_masks["unresolved_region"][:real_frame_len].cpu().numpy()
        if replacement_masks["background_confidence"] is not None:
            mask_artifacts["background_confidence"] = replacement_masks["background_confidence"][:real_frame_len].cpu().numpy()
        if replacement_masks["background_source_provenance"] is not None:
            mask_artifacts["background_source_provenance"] = replacement_masks["background_source_provenance"][:real_frame_len].cpu().numpy()
        if replacement_masks["occlusion_band"] is not None:
            mask_artifacts["occlusion_band"] = replacement_masks["occlusion_band"][:real_frame_len].cpu().numpy()
        if replacement_masks["uncertainty_map"] is not None:
            mask_artifacts["uncertainty_map"] = replacement_masks["uncertainty_map"][:real_frame_len].cpu().numpy()
        if replacement_masks["soft_band_latent"] is not None:
            mask_artifacts["latent_soft_band"] = replacement_masks["soft_band_latent"][:real_frame_len].cpu().numpy()
        if replacement_masks["soft_alpha_latent"] is not None:
            mask_artifacts["latent_soft_alpha"] = replacement_masks["soft_alpha_latent"][:real_frame_len].cpu().numpy()

        for stem, mask_frames in mask_artifacts.items():
            artifact = write_person_mask_artifact(
                mask_frames=mask_frames,
                output_root=debug_dir,
                stem=stem,
                artifact_format="mp4",
                fps=fps,
                mask_semantics=stem,
            )
            artifacts[stem] = str((debug_dir / artifact["path"]).resolve())
        return artifacts

    def _encode_text_conditions(self, input_prompt, n_prompt, offload_model, need_negative_text):
        start_time = time.perf_counter()
        context_null = None
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            if need_negative_text:
                context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            if need_negative_text:
                context_null = self.text_encoder([n_prompt], torch.device('cpu'))
                context_null = [t.to(self.device) for t in context_null]
        return context, context_null, time.perf_counter() - start_time

    def _build_static_reference_conditions(self, refer_images, lat_h, lat_w):
        start_time = time.perf_counter()
        with (
            torch.autocast(device_type=str(self.device), dtype=torch.bfloat16, enabled=True),
            torch.no_grad()
        ):
            refer_pixel_values = rearrange(
                torch.tensor(refer_images / 127.5 - 1), "h w c -> 1 c h w"
            ).to(device=self.device, dtype=torch.bfloat16)
            reference_video = rearrange(refer_pixel_values, "b c h w -> b c 1 h w")
            ref_latents = self.vae.encode(reference_video)
            ref_latents = torch.stack(ref_latents)
            mask_ref = self.get_i2v_mask(1, lat_h, lat_w, 1, device=self.device)
            y_ref = torch.concat([mask_ref, ref_latents[0]]).to(dtype=torch.bfloat16, device=self.device)
            img = reference_video[0, :, 0]
            clip_context = self.clip.visual([img[:, None, :, :]]).to(dtype=torch.bfloat16, device=self.device)
        return {
            "y_ref": y_ref,
            "clip_context": clip_context,
        }, time.perf_counter() - start_time

    def _write_runtime_stats(self, save_debug_dir, runtime_stats):
        if save_debug_dir is None:
            return None
        debug_dir = Path(save_debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        stats_path = debug_dir / "wan_animate_runtime_stats.json"
        with stats_path.open("w", encoding="utf-8") as handle:
            json.dump(runtime_stats, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
        return str(stats_path.resolve())

    def _summarize_seams(self, seam_stats):
        if not seam_stats:
            return {}
        return {
            "boundary_before": summarize_scalar_series([item["boundary_score_before"] for item in seam_stats]),
            "boundary_after": summarize_scalar_series([item["boundary_score_after"] for item in seam_stats]),
            "overlap_before": summarize_scalar_series([item["overlap_mad_before"] for item in seam_stats]),
            "overlap_after_prev": summarize_scalar_series([item["overlap_mad_after_prev"] for item in seam_stats]),
            "overlap_after_curr": summarize_scalar_series([item["overlap_mad_after_curr"] for item in seam_stats]),
        }

    def _summarize_temporal_handoffs(self, handoff_stats):
        if not handoff_stats:
            return {}
        numeric_keys = [
            "latent_slots",
            "blend_strength_mean",
            "base_to_memory_mad",
            "composed_to_base_mad",
            "composed_to_memory_mad",
        ]
        summary = {}
        for key in numeric_keys:
            values = [item[key] for item in handoff_stats if item.get(key) is not None]
            summary[key] = summarize_scalar_series(values)
        summary["applied_count"] = int(sum(1 for item in handoff_stats if item.get("applied")))
        return summary

    def _resolve_guidance_config(self, guidance_uncond_mode, face_guide_scale, text_guide_scale):
        if guidance_uncond_mode not in {"legacy_both", "face_only", "text_only", "decoupled"}:
            raise ValueError(f"Unsupported guidance_uncond_mode: {guidance_uncond_mode}")
        face_guide_scale = float(face_guide_scale)
        text_guide_scale = float(text_guide_scale)
        if face_guide_scale < 1.0 or text_guide_scale < 1.0:
            raise ValueError(
                f"face_guide_scale and text_guide_scale must be >= 1.0. Got {face_guide_scale}, {text_guide_scale}."
            )
        if guidance_uncond_mode == "legacy_both":
            if abs(face_guide_scale - text_guide_scale) > 1e-6:
                raise ValueError(
                    "legacy_both guidance requires matching face/text guide scales."
                )
            return {
                "mode": guidance_uncond_mode,
                "need_negative_text": face_guide_scale > 1.0,
                "legacy_scale": face_guide_scale,
                "face_scale": face_guide_scale,
                "text_scale": text_guide_scale,
                "use_face_branch": False,
                "use_text_branch": False,
                "forward_passes_per_step": 2 if face_guide_scale > 1.0 else 1,
            }
        use_face_branch = guidance_uncond_mode in {"face_only", "decoupled"} and face_guide_scale > 1.0
        use_text_branch = guidance_uncond_mode in {"text_only", "decoupled"} and text_guide_scale > 1.0
        return {
            "mode": guidance_uncond_mode,
            "need_negative_text": use_text_branch,
            "legacy_scale": None,
            "face_scale": face_guide_scale,
            "text_scale": text_guide_scale,
            "use_face_branch": use_face_branch,
            "use_text_branch": use_text_branch,
            "forward_passes_per_step": 1 + int(use_face_branch) + int(use_text_branch),
        }

    def generate(
        self,
        src_root_path,
        replace_flag=False,
        clip_len=77,
        refert_num=5,
        overlap_blend_mode="mask_aware",
        overlap_background_current_strength=0.35,
        seam_debug_max_points=6,
        temporal_handoff_mode="pixel",
        temporal_handoff_strength=1.0,
        temporal_handoff_debug_max_points=4,
        guidance_uncond_mode="legacy_both",
        face_guide_scale=1.0,
        text_guide_scale=1.0,
        replacement_mask_mode="soft_band",
        replacement_mask_downsample_mode="area",
        replacement_boundary_strength=0.5,
        replacement_transition_low=0.1,
        replacement_transition_high=0.9,
        boundary_refine_mode="none",
        boundary_refine_strength=0.35,
        boundary_refine_sharpen=0.15,
        boundary_refine_use_clean_plate=True,
        shift=5.0,
        sample_solver='dpm++',
        sampling_steps=20,
        guide_scale=1,
        input_prompt="",
        n_prompt="",
        seed=-1,
        offload_model=True,
        save_debug_dir=None,
        log_runtime_stats=False,
        quality_preset="none",
    ):
        r"""
        Generates video frames from input image using diffusion process.

        Args:
            src_root_path ('str'):
                Process output path
            replace_flag (`bool`, *optional*, defaults to False):
                Whether to use character replace.
            clip_len (`int`, *optional*, defaults to 77):
                How many frames to generate per clips. The number should be 4n+1
            refert_num (`int`, *optional*, defaults to 5):
                How many frames are reused for temporal guidance. Must satisfy 0 < refert_num < clip_len.
            overlap_blend_mode (`str`, *optional*, defaults to "mask_aware"):
                How decoded overlap frames from adjacent clips are merged before final concatenation.
            overlap_background_current_strength (`float`, *optional*, defaults to 0.35):
                Current-clip alpha multiplier used in hard background-keep regions during mask-aware blending.
            seam_debug_max_points (`int`, *optional*, defaults to 6):
                How many seam comparison bundles are exported under save_debug_dir.
            temporal_handoff_mode (`str`, *optional*, defaults to "pixel"):
                Temporal handoff prototype. `pixel` preserves the baseline decode->re-encode path, `latent` reuses previous clip latents,
                and `hybrid` keeps background-like regions from the pixel path while injecting latent memory into replacement regions.
            temporal_handoff_strength (`float`, *optional*, defaults to 1.0):
                Blend strength for `latent` and `hybrid` temporal handoff prototypes.
            temporal_handoff_debug_max_points (`int`, *optional*, defaults to 4):
                Number of temporal handoff latent bundles exported under save_debug_dir.
            guidance_uncond_mode (`str`, *optional*, defaults to "legacy_both"):
                Guidance branch strategy. `legacy_both` preserves the original coupled text+face CFG behavior.
            face_guide_scale (`float`, *optional*, defaults to 1.0):
                Face expression guidance scale used by `face_only` and `decoupled` modes.
            text_guide_scale (`float`, *optional*, defaults to 1.0):
                Text guidance scale used by `text_only` and `decoupled` modes.
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. 
            sample_solver (`str`, *optional*, defaults to 'dpm++'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 20):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float` or tuple[`float`], *optional*, defaults 1.0):
                Legacy compatibility guidance scale. It is kept for backward compatibility and should be aligned with
                face/text guide scales when using `legacy_both`.
            input_prompt (`str`):
                Text prompt for content generation. We don't recommend custom prompts (although they work)
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM
            boundary_refine_mode (`str`, *optional*, defaults to "none"):
                Optional pixel-domain post refinement mode. `deterministic` applies a boundary-band composite and local sharpening.
            boundary_refine_strength (`float`, *optional*, defaults to 0.35):
                Strength of the outer-band background composite in pixel-domain refinement.
            boundary_refine_sharpen (`float`, *optional*, defaults to 0.15):
                Strength of the inner-boundary unsharp mask during pixel-domain refinement.
            boundary_refine_use_clean_plate (`bool`, *optional*, defaults to True):
                Whether pixel-domain refinement should prefer the preprocess clean-plate background artifact when available.
            save_debug_dir (`str`, *optional*, defaults to None):
                Optional directory for runtime statistics emitted by the animate pipeline.
            log_runtime_stats (`bool`, *optional*, defaults to False):
                Whether to log clip-level runtime and memory statistics.
            quality_preset (`str`, *optional*, defaults to "none"):
                Optional runtime preset label used for logging and experiment tracking.

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N, H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames
                - H: Frame height 
                - W: Frame width 
        """
        refert_num = validate_refert_num(refert_num, clip_len=clip_len)
        if overlap_blend_mode not in {"none", "linear", "mask_aware"}:
            raise ValueError(f"Unsupported overlap_blend_mode: {overlap_blend_mode}")
        if not 0.0 <= float(overlap_background_current_strength) <= 1.0:
            raise ValueError(
                f"overlap_background_current_strength must be in [0, 1]. Got {overlap_background_current_strength}."
            )
        if seam_debug_max_points < 0:
            raise ValueError(f"seam_debug_max_points must be >= 0. Got {seam_debug_max_points}.")
        if temporal_handoff_mode not in {"pixel", "latent", "hybrid"}:
            raise ValueError(f"Unsupported temporal_handoff_mode: {temporal_handoff_mode}")
        if not 0.0 <= float(temporal_handoff_strength) <= 1.0:
            raise ValueError(
                f"temporal_handoff_strength must be in [0, 1]. Got {temporal_handoff_strength}."
            )
        if boundary_refine_mode not in {"none", "deterministic"}:
            raise ValueError(f"Unsupported boundary_refine_mode: {boundary_refine_mode}")
        if not 0.0 <= float(boundary_refine_strength) <= 1.0:
            raise ValueError(f"boundary_refine_strength must be in [0, 1]. Got {boundary_refine_strength}.")
        if not 0.0 <= float(boundary_refine_sharpen) <= 1.0:
            raise ValueError(f"boundary_refine_sharpen must be in [0, 1]. Got {boundary_refine_sharpen}.")
        if temporal_handoff_debug_max_points < 0:
            raise ValueError(
                f"temporal_handoff_debug_max_points must be >= 0. Got {temporal_handoff_debug_max_points}."
            )
        guidance_config = self._resolve_guidance_config(
            guidance_uncond_mode=guidance_uncond_mode,
            face_guide_scale=face_guide_scale,
            text_guide_scale=text_guide_scale,
        )
        self.last_runtime_stats = None

        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        if input_prompt == "":
            input_prompt = self.sample_prompt

        artifacts, preprocess_metadata = resolve_preprocess_artifacts(
            src_root_path,
            replace_flag=replace_flag,
            logger=logging.getLogger(__name__),
        )
        if preprocess_metadata is not None:
            logging.info(
                "Loaded preprocess metadata: storage_format=%s color_space=%s replace_flag=%s",
                preprocess_metadata["storage_format"],
                preprocess_metadata["color_space"],
                preprocess_metadata["replace_flag"],
            )
            if replace_flag:
                background_mode = preprocess_metadata.get("src_files", {}).get("background", {}).get(
                    "background_mode",
                    preprocess_metadata.get("processing", {}).get("background", {}).get("bg_inpaint_mode", "unknown"),
                )
                reference_mode = preprocess_metadata.get("processing", {}).get("reference_normalization", {}).get(
                    "reference_normalization_mode",
                    "none",
                )
                logging.info(
                    "Mask contract: src_mask=%s, generate uses %s. background_mode=%s reference_mode=%s",
                    preprocess_metadata.get("mask_semantics", PERSON_MASK_SEMANTICS),
                    BACKGROUND_KEEP_MASK_SEMANTICS,
                    background_mode,
                    reference_mode,
                )

        cond_images, face_images, refer_images = self.prepare_source(
            src_pose_artifact=artifacts["pose"],
            src_face_artifact=artifacts["face"],
            src_ref_artifact=artifacts["reference"],
        )
        
        total_start_time = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

        context, context_null, text_condition_encode_sec = self._encode_text_conditions(
            input_prompt=input_prompt,
            n_prompt=n_prompt,
            offload_model=offload_model,
            need_negative_text=guidance_config["need_negative_text"],
        )

        real_frame_len = len(cond_images)
        target_len = self.get_valid_len(real_frame_len, clip_len, overlap=refert_num)
        logging.info('real frames: {} target frames: {}'.format(real_frame_len, target_len))
        cond_images = self.inputs_padding(cond_images, target_len)
        face_images = self.inputs_padding(face_images, target_len)
        
        soft_band_images = None
        hard_foreground_images = None
        soft_alpha_images = None
        boundary_band_images = None
        background_keep_prior_images = None
        visible_support_images = None
        unresolved_region_images = None
        background_confidence_images = None
        background_source_provenance_images = None
        occlusion_band_images = None
        uncertainty_map_images = None
        bg_images = None
        if replace_flag:
            bg_images, person_mask_images = self.prepare_source_for_replace(
                src_bg_artifact=artifacts["background"],
                src_mask_artifact=artifacts["person_mask"],
            )
            if "hard_foreground" in artifacts:
                hard_foreground_images = load_mask_artifact(artifacts["hard_foreground"]["path"], artifacts["hard_foreground"].get("format"))
            if "soft_band" in artifacts:
                soft_band_images = load_mask_artifact(artifacts["soft_band"]["path"], artifacts["soft_band"].get("format"))
            if "boundary_band" in artifacts:
                boundary_band_images = load_mask_artifact(artifacts["boundary_band"]["path"], artifacts["boundary_band"].get("format"))
            if "soft_alpha" in artifacts:
                soft_alpha_images = load_mask_artifact(artifacts["soft_alpha"]["path"], artifacts["soft_alpha"].get("format"))
            if "background_keep_prior" in artifacts:
                background_keep_prior_images = load_mask_artifact(
                    artifacts["background_keep_prior"]["path"],
                    artifacts["background_keep_prior"].get("format"),
                )
            if "visible_support" in artifacts:
                visible_support_images = load_mask_artifact(
                    artifacts["visible_support"]["path"],
                    artifacts["visible_support"].get("format"),
                )
            if "unresolved_region" in artifacts:
                unresolved_region_images = load_mask_artifact(
                    artifacts["unresolved_region"]["path"],
                    artifacts["unresolved_region"].get("format"),
                )
            if "background_confidence" in artifacts:
                background_confidence_images = load_mask_artifact(
                    artifacts["background_confidence"]["path"],
                    artifacts["background_confidence"].get("format"),
                )
            if "background_source_provenance" in artifacts:
                background_source_provenance_images = load_mask_artifact(
                    artifacts["background_source_provenance"]["path"],
                    artifacts["background_source_provenance"].get("format"),
                )
            if "occlusion_band" in artifacts:
                occlusion_band_images = load_mask_artifact(
                    artifacts["occlusion_band"]["path"],
                    artifacts["occlusion_band"].get("format"),
                )
            if "uncertainty_map" in artifacts:
                uncertainty_map_images = load_mask_artifact(
                    artifacts["uncertainty_map"]["path"],
                    artifacts["uncertainty_map"].get("format"),
                )
            validate_loaded_preprocess_bundle(
                cond_images=np.asarray(cond_images[:real_frame_len]),
                face_images=np.asarray(face_images[:real_frame_len]),
                refer_image_rgb=refer_images,
                metadata=preprocess_metadata,
                bg_images=bg_images,
                person_mask_images=person_mask_images,
                soft_band_images=soft_band_images,
                hard_foreground_images=hard_foreground_images,
                soft_alpha_images=soft_alpha_images,
                boundary_band_images=boundary_band_images,
                background_keep_prior_images=background_keep_prior_images,
                visible_support_images=visible_support_images,
                unresolved_region_images=unresolved_region_images,
                background_confidence_images=background_confidence_images,
                background_source_provenance_images=background_source_provenance_images,
                occlusion_band_images=occlusion_band_images,
                uncertainty_map_images=uncertainty_map_images,
            )
            bg_images = self.inputs_padding(bg_images, target_len)
            person_mask_images = self.inputs_padding(person_mask_images, target_len)
            if hard_foreground_images is not None:
                hard_foreground_images = self.inputs_padding(hard_foreground_images, target_len)
            if soft_band_images is not None:
                soft_band_images = self.inputs_padding(soft_band_images, target_len)
            if boundary_band_images is not None:
                boundary_band_images = self.inputs_padding(boundary_band_images, target_len)
            if soft_alpha_images is not None:
                soft_alpha_images = self.inputs_padding(soft_alpha_images, target_len)
            if background_keep_prior_images is not None:
                background_keep_prior_images = self.inputs_padding(background_keep_prior_images, target_len)
            if visible_support_images is not None:
                visible_support_images = self.inputs_padding(visible_support_images, target_len)
            if unresolved_region_images is not None:
                unresolved_region_images = self.inputs_padding(unresolved_region_images, target_len)
            if background_confidence_images is not None:
                background_confidence_images = self.inputs_padding(background_confidence_images, target_len)
            if background_source_provenance_images is not None:
                background_source_provenance_images = self.inputs_padding(background_source_provenance_images, target_len)
            if occlusion_band_images is not None:
                occlusion_band_images = self.inputs_padding(occlusion_band_images, target_len)
            if uncertainty_map_images is not None:
                uncertainty_map_images = self.inputs_padding(uncertainty_map_images, target_len)
        else:
            validate_loaded_preprocess_bundle(
                cond_images=np.asarray(cond_images[:real_frame_len]),
                face_images=np.asarray(face_images[:real_frame_len]),
                refer_image_rgb=refer_images,
                metadata=preprocess_metadata,
            )

        height, width = refer_images.shape[:2]
        lat_h = height // 8
        lat_w = width // 8
        static_reference_conditions, reference_condition_encode_sec = self._build_static_reference_conditions(
            refer_images=refer_images,
            lat_h=lat_h,
            lat_w=lat_w,
        )
        replacement_masks = None
        replacement_mask_debug = {}
        if replace_flag:
            if soft_band_images is None and boundary_band_images is None and replacement_mask_mode != "hard":
                logging.warning(
                    "replacement_mask_mode=%s requested, but preprocess bundle has no soft boundary artifact. Falling back to hard mask.",
                    replacement_mask_mode,
                )
                replacement_mask_mode = "hard"
            replacement_masks = self._prepare_replacement_masks(
                person_mask_images=np.asarray(person_mask_images),
                soft_band_images=np.asarray(soft_band_images) if soft_band_images is not None else None,
                hard_foreground_images=np.asarray(hard_foreground_images) if hard_foreground_images is not None else None,
                soft_alpha_images=np.asarray(soft_alpha_images) if soft_alpha_images is not None else None,
                boundary_band_images=np.asarray(boundary_band_images) if boundary_band_images is not None else None,
                background_keep_prior_images=(
                    np.asarray(background_keep_prior_images) if background_keep_prior_images is not None else None
                ),
                visible_support_images=np.asarray(visible_support_images) if visible_support_images is not None else None,
                unresolved_region_images=np.asarray(unresolved_region_images) if unresolved_region_images is not None else None,
                background_confidence_images=(
                    np.asarray(background_confidence_images) if background_confidence_images is not None else None
                ),
                background_source_provenance_images=(
                    np.asarray(background_source_provenance_images) if background_source_provenance_images is not None else None
                ),
                occlusion_band_images=np.asarray(occlusion_band_images) if occlusion_band_images is not None else None,
                uncertainty_map_images=np.asarray(uncertainty_map_images) if uncertainty_map_images is not None else None,
                lat_h=lat_h,
                lat_w=lat_w,
                mask_mode=replacement_mask_mode,
                boundary_strength=replacement_boundary_strength,
                downsample_mode=replacement_mask_downsample_mode,
                transition_low=replacement_transition_low,
                transition_high=replacement_transition_high,
            )
            replacement_mask_debug = self._write_replacement_mask_debug(
                save_debug_dir=save_debug_dir,
                fps=preprocess_metadata["fps"] if preprocess_metadata is not None else 30,
                replacement_masks=replacement_masks,
                real_frame_len=real_frame_len,
            )
        static_condition_encode_sec = text_condition_encode_sec + reference_condition_encode_sec

        runtime_stats = {
            "quality_preset": quality_preset,
            "offload_model": bool(offload_model),
            "clip_len": clip_len,
            "refert_num": refert_num,
            "overlap_blend_mode": overlap_blend_mode,
            "overlap_background_current_strength": float(overlap_background_current_strength),
            "seam_debug_max_points": int(seam_debug_max_points),
            "temporal_handoff_mode": temporal_handoff_mode,
            "temporal_handoff_strength": float(temporal_handoff_strength),
            "temporal_handoff_debug_max_points": int(temporal_handoff_debug_max_points),
            "sampling_steps": sampling_steps,
            "guide_scale": float(guide_scale),
            "guidance_uncond_mode": guidance_config["mode"],
            "face_guide_scale": float(guidance_config["face_scale"]),
            "text_guide_scale": float(guidance_config["text_scale"]),
            "legacy_guide_scale": (
                float(guidance_config["legacy_scale"])
                if guidance_config["legacy_scale"] is not None else None
            ),
            "guidance_forward_passes_per_step": int(guidance_config["forward_passes_per_step"]),
            "real_frame_len": int(real_frame_len),
            "target_len": int(target_len),
            "replacement_mask_mode": replacement_mask_mode if replace_flag else None,
            "replacement_mask_downsample_mode": replacement_mask_downsample_mode if replace_flag else None,
            "replacement_boundary_strength": float(replacement_boundary_strength) if replace_flag else None,
            "replacement_transition_low": float(replacement_transition_low) if replace_flag else None,
            "replacement_transition_high": float(replacement_transition_high) if replace_flag else None,
            "boundary_refine_mode": boundary_refine_mode if replace_flag else "none",
            "boundary_refine_strength": float(boundary_refine_strength) if replace_flag else None,
            "boundary_refine_sharpen": float(boundary_refine_sharpen) if replace_flag else None,
            "boundary_refine_use_clean_plate": bool(boundary_refine_use_clean_plate) if replace_flag else None,
            "background_mode": (
                preprocess_metadata.get("src_files", {}).get("background", {}).get(
                    "background_mode",
                    preprocess_metadata.get("processing", {}).get("background", {}).get("bg_inpaint_mode", "unknown"),
                )
                if replace_flag and preprocess_metadata is not None else None
            ),
            "reference_normalization_mode": (
                preprocess_metadata.get("processing", {}).get("reference_normalization", {}).get(
                    "reference_normalization_mode",
                    "none",
                )
                if preprocess_metadata is not None else None
            ),
            "reference_artifact_path": artifacts["reference"]["path"],
            "face_landmarks_available": bool(preprocess_metadata and "face_landmarks" in preprocess_metadata.get("src_files", {})),
            "face_pose_available": bool(preprocess_metadata and "face_pose" in preprocess_metadata.get("src_files", {})),
            "face_expression_available": bool(preprocess_metadata and "face_expression" in preprocess_metadata.get("src_files", {})),
            "face_alpha_available": bool(preprocess_metadata and "face_alpha" in preprocess_metadata.get("src_files", {})),
            "face_parsing_available": bool(preprocess_metadata and "face_parsing" in preprocess_metadata.get("src_files", {})),
            "face_uncertainty_available": bool(preprocess_metadata and "face_uncertainty" in preprocess_metadata.get("src_files", {})),
            "pose_tracks_available": bool(preprocess_metadata and "pose_tracks" in preprocess_metadata.get("src_files", {})),
            "limb_tracks_available": bool(preprocess_metadata and "limb_tracks" in preprocess_metadata.get("src_files", {})),
            "hand_tracks_available": bool(preprocess_metadata and "hand_tracks" in preprocess_metadata.get("src_files", {})),
            "pose_visibility_available": bool(preprocess_metadata and "pose_visibility" in preprocess_metadata.get("src_files", {})),
            "pose_uncertainty_available": bool(preprocess_metadata and "pose_uncertainty" in preprocess_metadata.get("src_files", {})),
            "soft_band_available": bool(soft_band_images is not None) if replace_flag else False,
            "hard_foreground_available": bool(hard_foreground_images is not None) if replace_flag else False,
            "soft_alpha_available": bool(soft_alpha_images is not None) if replace_flag else False,
            "boundary_band_available": bool(boundary_band_images is not None) if replace_flag else False,
            "background_keep_prior_available": bool(background_keep_prior_images is not None) if replace_flag else False,
            "visible_support_available": bool(visible_support_images is not None) if replace_flag else False,
            "unresolved_region_available": bool(unresolved_region_images is not None) if replace_flag else False,
            "background_confidence_available": bool(background_confidence_images is not None) if replace_flag else False,
            "background_source_provenance_available": bool(background_source_provenance_images is not None) if replace_flag else False,
            "occlusion_band_available": bool(occlusion_band_images is not None) if replace_flag else False,
            "uncertainty_map_available": bool(uncertainty_map_images is not None) if replace_flag else False,
            "text_condition_encode_sec": text_condition_encode_sec,
            "reference_condition_encode_sec": reference_condition_encode_sec,
            "static_condition_encode_sec": static_condition_encode_sec,
            "clip_stats": [],
            "seam_stats": [],
            "temporal_handoff_stats": [],
            "boundary_refinement": {
                "mode_requested": boundary_refine_mode if replace_flag else "none",
                "applied": False,
            },
        }
        if replacement_mask_debug:
            runtime_stats["replacement_mask_debug"] = replacement_mask_debug
        if log_runtime_stats:
            logging.info(
                "Static animate conditions encoded once: text=%.3fs reference=%.3fs total=%.3fs guidance_mode=%s face_scale=%.2f text_scale=%.2f",
                text_condition_encode_sec,
                reference_condition_encode_sec,
                static_condition_encode_sec,
                guidance_config["mode"],
                guidance_config["face_scale"],
                guidance_config["text_scale"],
            )

        start = 0
        end = clip_len
        all_out_frames = []
        previous_output_latent = None
        clip_index = 0
        while True:
            if start + refert_num >= len(cond_images):
                break

            clip_index += 1
            clip_start_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(self.device)
            if start == 0:
                mask_reft_len = 0
            else:
                mask_reft_len = refert_num

            batch = {
                        "conditioning_pixel_values": torch.zeros(1, 3, clip_len, height, width),
                        "bg_pixel_values": torch.zeros(1, 3, clip_len, height, width),
                        "face_pixel_values": torch.zeros(1, 3, clip_len, 512, 512),
                        "refer_pixel_values": torch.zeros(1, 3, height, width),
                        "refer_t_pixel_values": torch.zeros(refert_num, 3, height, width)
                    }   

            batch["conditioning_pixel_values"] = rearrange(
                torch.tensor(np.stack(cond_images[start:end]) / 127.5 - 1),
                "t h w c -> 1 c t h w",
            )
            batch["face_pixel_values"] = rearrange(
                torch.tensor(np.stack(face_images[start:end]) / 127.5 - 1),
                "t h w c -> 1 c t h w",
            )

            batch["refer_pixel_values"] = rearrange(
                torch.tensor(refer_images / 127.5 - 1), "h w c -> 1 c h w"
            )

            if start > 0:
                batch["refer_t_pixel_values"] = rearrange(
                    out_frames[0, :, -refert_num:].clone().detach(),
                    "c t h w -> t c h w",
                )

            batch["refer_t_pixel_values"] = rearrange(batch["refer_t_pixel_values"],
                                            "t c h w -> 1 c t h w",
                                            )

            if replace_flag:
                batch["bg_pixel_values"] = rearrange(
                    torch.tensor(np.stack(bg_images[start:end]) / 127.5 - 1),
                    "t h w c -> 1 c t h w",
                )

            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device=self.device, dtype=torch.bfloat16)

            refer_t_pixel_values = batch["refer_t_pixel_values"]
            conditioning_pixel_values = batch["conditioning_pixel_values"]
            face_pixel_values = batch["face_pixel_values"]

            B, _, H, W = batch["refer_pixel_values"].shape
            T = clip_len
            lat_t = T // 4 + 1
            target_shape = [lat_t + 1, lat_h, lat_w]
            clip_background_keep_latent = None
            clip_transition_band_latent = None
            clip_replacement_strength_latent = None
            if replace_flag:
                clip_background_keep_latent = replacement_masks["background_keep_latent"][start:end].to(device=self.device, dtype=torch.float32)
                clip_transition_band_latent = replacement_masks["latent_regions"]["transition_band"][start:end].to(device=self.device, dtype=torch.float32)
                clip_replacement_strength_latent = replacement_masks["latent_regions"]["replacement_strength"][start:end].to(
                    device=self.device,
                    dtype=torch.float32,
                )
            noise = [
                torch.randn(
                    16,
                    target_shape[0],
                    target_shape[1],
                    target_shape[2],
                    dtype=torch.float32,
                    device=self.device,
                    generator=seed_g,
                )
            ]
        
            max_seq_len = int(math.ceil(np.prod(target_shape) // 4 / self.sp_size)) * self.sp_size
            if max_seq_len % self.sp_size != 0:
                raise ValueError(f"max_seq_len {max_seq_len} is not divisible by sp_size {self.sp_size}")

            with (
                torch.autocast(device_type=str(self.device), dtype=torch.bfloat16, enabled=True),
                torch.no_grad()
            ):
                if sample_solver == 'unipc':
                    sample_scheduler = FlowUniPCMultistepScheduler(
                        num_train_timesteps=self.num_train_timesteps,
                        shift=1,
                        use_dynamic_shifting=False)
                    sample_scheduler.set_timesteps(
                        sampling_steps, device=self.device, shift=shift)
                    timesteps = sample_scheduler.timesteps
                elif sample_solver == 'dpm++':
                    sample_scheduler = FlowDPMSolverMultistepScheduler(
                        num_train_timesteps=self.num_train_timesteps,
                        shift=1,
                        use_dynamic_shifting=False)
                    sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                    timesteps, _ = retrieve_timesteps(
                        sample_scheduler,
                        device=self.device,
                        sigmas=sampling_sigmas)
                else:
                    raise NotImplementedError("Unsupported solver.")

                latents = noise

                vae_encode_start_time = time.perf_counter()
                pose_latents_no_ref =  self.vae.encode(conditioning_pixel_values.to(torch.bfloat16))
                pose_latents_no_ref = torch.stack(pose_latents_no_ref)
                pose_latents = torch.cat([pose_latents_no_ref], dim=2)
                temporal_handoff_stats = None
                temporal_handoff_debug_paths = None
                temporal_handoff_blend_mask = None
                temporal_handoff_memory_latents = None
                temporal_handoff_base_latents = None

                if replace_flag:
                    bg_pixel_values = batch["bg_pixel_values"]
                    if mask_reft_len > 0:
                        pixel_handoff_input = torch.concat(
                            [refer_t_pixel_values[0, :, :mask_reft_len], bg_pixel_values[0, :, mask_reft_len:]],
                            dim=1,
                        ).to(self.device)
                    else:
                        pixel_handoff_input = torch.concat([bg_pixel_values[0]], dim=1).to(self.device)

                    bg_only_input = torch.concat([bg_pixel_values[0]], dim=1).to(self.device)
                    if temporal_handoff_mode == "pixel" or mask_reft_len == 0 or previous_output_latent is None:
                        y_reft = self.vae.encode([pixel_handoff_input])[0]
                        temporal_handoff_stats = {
                            "mode": temporal_handoff_mode,
                            "overlap_frames": int(mask_reft_len),
                            "latent_slots": 0,
                            "applied": False,
                            "blend_strength_mean": 0.0,
                            "base_to_memory_mad": None,
                            "composed_to_base_mad": 0.0,
                            "composed_to_memory_mad": None,
                        }
                    else:
                        if temporal_handoff_mode == "latent":
                            y_reft_base = self.vae.encode([bg_only_input])[0]
                            replacement_strength_slots = None
                        else:
                            y_reft_base = self.vae.encode([pixel_handoff_input])[0]
                            replacement_strength_slots = pack_overlap_tensor_to_latent_slots(
                                clip_replacement_strength_latent[:mask_reft_len],
                                reduction="mean",
                            )
                            temporal_handoff_blend_mask = replacement_strength_slots[: y_reft_base.shape[1]]
                        temporal_handoff_base_latents = y_reft_base
                        y_reft, temporal_handoff_stats = compose_temporal_handoff_latents(
                            base_latents=y_reft_base,
                            previous_output_latents=previous_output_latent,
                            overlap_frames=mask_reft_len,
                            mode=temporal_handoff_mode,
                            strength=temporal_handoff_strength,
                            replacement_strength_slots=replacement_strength_slots,
                        )
                        temporal_handoff_memory_latents = previous_output_latent[:, -temporal_handoff_stats["latent_slots"]:].to(
                            dtype=torch.float32,
                            device=y_reft.device,
                        ) if temporal_handoff_stats["latent_slots"] > 0 else None

                    msk_reft = self.get_i2v_mask(
                        lat_t,
                        lat_h,
                        lat_w,
                        mask_reft_len,
                        mask_pixel_values=clip_background_keep_latent[None],
                        device=self.device,
                    )
                else:
                    if mask_reft_len > 0:
                        pixel_handoff_input = torch.concat(
                            [
                                torch.nn.functional.interpolate(
                                    refer_t_pixel_values[0, :, :mask_reft_len].cpu(),
                                    size=(H, W),
                                    mode="bicubic",
                                ),
                                torch.zeros(3, T - mask_reft_len, H, W),
                            ],
                            dim=1,
                        ).to(self.device)
                        zero_input = torch.concat([torch.zeros(3, T - mask_reft_len, H, W)], dim=1).to(self.device)
                    else:
                        pixel_handoff_input = torch.concat([torch.zeros(3, T - mask_reft_len, H, W)], dim=1).to(self.device)
                        zero_input = pixel_handoff_input

                    if temporal_handoff_mode == "pixel" or mask_reft_len == 0 or previous_output_latent is None:
                        y_reft = self.vae.encode([pixel_handoff_input])[0]
                        temporal_handoff_stats = {
                            "mode": temporal_handoff_mode,
                            "overlap_frames": int(mask_reft_len),
                            "latent_slots": 0,
                            "applied": False,
                            "blend_strength_mean": 0.0,
                            "base_to_memory_mad": None,
                            "composed_to_base_mad": 0.0,
                            "composed_to_memory_mad": None,
                        }
                    else:
                        y_reft_base = self.vae.encode([zero_input if temporal_handoff_mode == "latent" else pixel_handoff_input])[0]
                        temporal_handoff_base_latents = y_reft_base
                        y_reft, temporal_handoff_stats = compose_temporal_handoff_latents(
                            base_latents=y_reft_base,
                            previous_output_latents=previous_output_latent,
                            overlap_frames=mask_reft_len,
                            mode="latent" if temporal_handoff_mode == "hybrid" else temporal_handoff_mode,
                            strength=temporal_handoff_strength,
                            replacement_strength_slots=None,
                        )
                        temporal_handoff_memory_latents = previous_output_latent[:, -temporal_handoff_stats["latent_slots"]:].to(
                            dtype=torch.float32,
                            device=y_reft.device,
                        ) if temporal_handoff_stats["latent_slots"] > 0 else None
                    msk_reft = self.get_i2v_mask(lat_t, lat_h, lat_w, mask_reft_len, device=self.device)

                clip_vae_encode_sec = time.perf_counter() - vae_encode_start_time
                temporal_handoff_stats["clip_index"] = clip_index
                temporal_handoff_stats["start_frame"] = int(start)
                temporal_handoff_stats["mode_requested"] = temporal_handoff_mode
                if (
                    save_debug_dir is not None
                    and temporal_handoff_stats.get("applied")
                    and len(runtime_stats["temporal_handoff_stats"]) < temporal_handoff_debug_max_points
                    and temporal_handoff_base_latents is not None
                ):
                    temporal_handoff_debug_paths = write_temporal_handoff_debug(
                        save_debug_dir=save_debug_dir,
                        handoff_index=len(runtime_stats["temporal_handoff_stats"]) + 1,
                        stats=temporal_handoff_stats,
                        base_latents=temporal_handoff_base_latents[:, :temporal_handoff_stats["latent_slots"]],
                        memory_latents=temporal_handoff_memory_latents,
                        composed_latents=y_reft[:, :temporal_handoff_stats["latent_slots"]],
                        blend_mask=temporal_handoff_blend_mask[:temporal_handoff_stats["latent_slots"]]
                        if temporal_handoff_blend_mask is not None and temporal_handoff_stats["latent_slots"] > 0 else None,
                    )
                    temporal_handoff_stats["debug_paths"] = temporal_handoff_debug_paths
                runtime_stats["temporal_handoff_stats"].append(temporal_handoff_stats)
                y_reft = torch.concat([msk_reft, y_reft]).to(dtype=torch.bfloat16, device=self.device)
                y = torch.concat([static_reference_conditions["y_ref"], y_reft], dim=1)

                arg_c = {
                    "context": context, 
                    "seq_len": max_seq_len,
                    "clip_fea": static_reference_conditions["clip_context"],
                    "y": [y],
                    "pose_latents": pose_latents,
                    "face_pixel_values": face_pixel_values,
                }

                arg_face_null = None
                arg_text_null = None
                arg_legacy_null = None
                face_pixel_values_uncond = None
                if guidance_config["use_face_branch"] or (
                    guidance_config["mode"] == "legacy_both" and guidance_config["legacy_scale"] > 1.0
                ):
                    face_pixel_values_uncond = face_pixel_values * 0 - 1
                if guidance_config["mode"] == "legacy_both" and guidance_config["legacy_scale"] > 1.0:
                    arg_legacy_null = {
                        "context": context_null,
                        "seq_len": max_seq_len,
                        "clip_fea": static_reference_conditions["clip_context"],
                        "y": [y],
                        "pose_latents": pose_latents,
                        "face_pixel_values": face_pixel_values_uncond,
                    }
                else:
                    if guidance_config["use_face_branch"]:
                        arg_face_null = {
                            "context": context,
                            "seq_len": max_seq_len,
                            "clip_fea": static_reference_conditions["clip_context"],
                            "y": [y],
                            "pose_latents": pose_latents,
                            "face_pixel_values": face_pixel_values_uncond,
                        }
                    if guidance_config["use_text_branch"]:
                        arg_text_null = {
                            "context": context_null,
                            "seq_len": max_seq_len,
                            "clip_fea": static_reference_conditions["clip_context"],
                            "y": [y],
                            "pose_latents": pose_latents,
                            "face_pixel_values": face_pixel_values,
                        }

                sampling_start_time = time.perf_counter()
                for i, t in enumerate(tqdm(timesteps)):
                    latent_model_input = latents
                    timestep = [t]

                    timestep = torch.stack(timestep)

                    noise_pred_cond = TensorList(
                         self.noise_model(TensorList(latent_model_input), t=timestep, **arg_c)
                    )

                    noise_pred_legacy_null = None
                    noise_pred_face_null = None
                    noise_pred_text_null = None
                    if guidance_config["mode"] == "legacy_both" and guidance_config["legacy_scale"] > 1.0:
                        noise_pred_legacy_null = TensorList(
                            self.noise_model(
                                TensorList(latent_model_input), t=timestep, **arg_legacy_null
                            )
                        )
                    else:
                        if guidance_config["use_face_branch"]:
                            noise_pred_face_null = TensorList(
                                self.noise_model(
                                    TensorList(latent_model_input), t=timestep, **arg_face_null
                                )
                            )
                        if guidance_config["use_text_branch"]:
                            noise_pred_text_null = TensorList(
                                self.noise_model(
                                    TensorList(latent_model_input), t=timestep, **arg_text_null
                                )
                            )
                    noise_pred = combine_animate_guidance_predictions(
                        cond_pred=noise_pred_cond,
                        guidance_mode=guidance_config["mode"],
                        legacy_scale=guidance_config["legacy_scale"],
                        face_scale=guidance_config["face_scale"],
                        text_scale=guidance_config["text_scale"],
                        legacy_null_pred=noise_pred_legacy_null,
                        face_null_pred=noise_pred_face_null,
                        text_null_pred=noise_pred_text_null,
                    )

                    temp_x0 = sample_scheduler.step(
                        noise_pred[0].unsqueeze(0),
                        t,
                        latents[0].unsqueeze(0),
                        return_dict=False,
                        generator=seed_g,
                    )[0]
                    latents[0] = temp_x0.squeeze(0)

                    x0 = latents

                clip_sampling_sec = time.perf_counter() - sampling_start_time
                vae_decode_start_time = time.perf_counter()
                x0 = [x.to(dtype=torch.float32) for x in x0]
                previous_output_latent = x0[0][:, 1:].detach().to(dtype=torch.bfloat16).cpu()
                out_frames = torch.stack(self.vae.decode([x0[0][:, 1:]])).to(dtype=torch.float32).cpu()
                clip_vae_decode_sec = time.perf_counter() - vae_decode_start_time
                seam_debug_paths = None
                if start != 0:
                    prev_overlap = all_out_frames[-1][:, :, -refert_num:].clone()
                    curr_overlap = out_frames[:, :, :refert_num].clone()
                    overlap_pixel_regions = None
                    if replace_flag and overlap_blend_mode == "mask_aware":
                        overlap_pixel_regions = {
                            key: replacement_masks["pixel_regions"][key][start:start + refert_num]
                            for key in ("hard_background_keep", "transition_band", "free_replacement")
                        }
                    blend_result = blend_clip_overlap(
                        prev_overlap,
                        curr_overlap,
                        mode=overlap_blend_mode,
                        pixel_regions=overlap_pixel_regions,
                        background_current_strength=overlap_background_current_strength,
                    )
                    blended_overlap = blend_result["blended"].cpu()
                    all_out_frames[-1][:, :, -refert_num:] = blended_overlap
                    out_frames_tail = out_frames[:, :, refert_num:]
                    all_out_frames.append(out_frames_tail)

                    next_frame = out_frames[:, :, refert_num:refert_num + 1]
                    boundary_before = mean_abs_difference(prev_overlap[:, :, -1:], next_frame)
                    boundary_after = mean_abs_difference(blended_overlap[:, :, -1:], next_frame)
                    seam_stat = {
                        "seam_index": len(runtime_stats["seam_stats"]) + 1,
                        "start_frame": int(start),
                        "overlap_len": int(refert_num),
                        "blend_mode": overlap_blend_mode,
                        "boundary_score_before": boundary_before,
                        "boundary_score_after": boundary_after,
                    }
                    seam_stat.update(blend_result["stats"])
                    if replace_flag and overlap_pixel_regions is not None:
                        seam_stat.update({
                            "background_ratio": float(overlap_pixel_regions["hard_background_keep"].float().mean().item()),
                            "transition_ratio": float(overlap_pixel_regions["transition_band"].float().mean().item()),
                            "free_replacement_ratio": float(overlap_pixel_regions["free_replacement"].float().mean().item()),
                        })
                    if save_debug_dir is not None and seam_stat["seam_index"] <= seam_debug_max_points:
                        seam_debug_paths = write_seam_debug_artifacts(
                            save_debug_dir=save_debug_dir,
                            seam_index=seam_stat["seam_index"],
                            fps=preprocess_metadata["fps"] if preprocess_metadata is not None else 30,
                            prev_overlap=prev_overlap,
                            curr_overlap=curr_overlap,
                            blended_overlap=blended_overlap,
                            alpha_map=blend_result["alpha_map"],
                        )
                        seam_stat["debug_paths"] = seam_debug_paths
                    runtime_stats["seam_stats"].append(seam_stat)
                else:
                    all_out_frames.append(out_frames)

                clip_peak_memory_bytes = None
                if torch.cuda.is_available():
                    clip_peak_memory_bytes = int(torch.cuda.max_memory_allocated(self.device))
                clip_total_sec = time.perf_counter() - clip_start_time
                clip_stats = {
                    "clip_index": clip_index,
                    "start_frame": int(start),
                    "end_frame": int(end),
                    "mask_reft_len": int(mask_reft_len),
                    "temporal_handoff_mode": temporal_handoff_mode,
                    "temporal_handoff_applied": bool(temporal_handoff_stats.get("applied")),
                    "temporal_handoff_latent_slots": int(temporal_handoff_stats.get("latent_slots", 0)),
                    "guidance_mode": guidance_config["mode"],
                    "guidance_forward_passes_per_step": int(guidance_config["forward_passes_per_step"]),
                    "vae_encode_sec": clip_vae_encode_sec,
                    "sampling_sec": clip_sampling_sec,
                    "vae_decode_sec": clip_vae_decode_sec,
                    "total_sec": clip_total_sec,
                    "peak_memory_bytes": clip_peak_memory_bytes,
                    "peak_memory_gb": round(clip_peak_memory_bytes / (1024 ** 3), 3) if clip_peak_memory_bytes is not None else None,
                }
                if replace_flag:
                    clip_stats.update({
                        "background_keep_mean": float(clip_background_keep_latent.mean().item()),
                        "background_keep_min": float(clip_background_keep_latent.min().item()),
                        "background_keep_max": float(clip_background_keep_latent.max().item()),
                        "transition_ratio": float(clip_transition_band_latent.mean().item()),
                    })
                if seam_debug_paths is not None:
                    clip_stats["seam_debug_paths"] = seam_debug_paths
                if temporal_handoff_debug_paths is not None:
                    clip_stats["temporal_handoff_debug_paths"] = temporal_handoff_debug_paths
                runtime_stats["clip_stats"].append(clip_stats)
                if log_runtime_stats:
                    logging.info(
                        "Clip %d frames[%d,%d) total=%.3fs encode=%.3fs sample=%.3fs decode=%.3fs peak_mem=%.3fGB",
                        clip_index,
                        start,
                        end,
                        clip_total_sec,
                        clip_vae_encode_sec,
                        clip_sampling_sec,
                        clip_vae_decode_sec,
                        clip_stats["peak_memory_gb"] or 0.0,
                    )

                start += clip_len - refert_num
                end += clip_len - refert_num

        videos = torch.cat(all_out_frames, dim=2)[:, :, :real_frame_len]
        if replace_flag and boundary_refine_mode != "none":
            boundary_refine_start = time.perf_counter()
            videos, boundary_refinement_stats = self._apply_boundary_refinement(
                videos=videos,
                preprocess_metadata=preprocess_metadata,
                bg_images=np.asarray(bg_images[:real_frame_len]) if bg_images is not None else None,
                replacement_masks=replacement_masks,
                real_frame_len=real_frame_len,
                fps=preprocess_metadata["fps"] if preprocess_metadata is not None else 30.0,
                save_debug_dir=save_debug_dir,
                boundary_refine_mode=boundary_refine_mode,
                boundary_refine_strength=boundary_refine_strength,
                boundary_refine_sharpen=boundary_refine_sharpen,
                boundary_refine_use_clean_plate=boundary_refine_use_clean_plate,
            )
            boundary_refinement_stats["runtime_sec"] = time.perf_counter() - boundary_refine_start
            runtime_stats["boundary_refinement"] = boundary_refinement_stats
        total_generate_sec = time.perf_counter() - total_start_time
        peak_memory_bytes = None
        if torch.cuda.is_available():
            peak_memory_bytes = int(torch.cuda.max_memory_allocated(self.device))
        clip_count = len(runtime_stats["clip_stats"])
        runtime_stats.update({
            "clip_count": clip_count,
            "total_generate_sec": total_generate_sec,
            "avg_clip_sec": (sum(item["total_sec"] for item in runtime_stats["clip_stats"]) / clip_count) if clip_count > 0 else 0.0,
            "peak_memory_bytes": peak_memory_bytes,
            "peak_memory_gb": round(peak_memory_bytes / (1024 ** 3), 3) if peak_memory_bytes is not None else None,
        })
        runtime_stats["seam_summary"] = self._summarize_seams(runtime_stats["seam_stats"])
        runtime_stats["temporal_handoff_summary"] = self._summarize_temporal_handoffs(runtime_stats["temporal_handoff_stats"])
        if save_debug_dir is not None and runtime_stats["seam_stats"]:
            runtime_stats["seam_debug_path"] = write_seam_summary(save_debug_dir, runtime_stats["seam_stats"])
        runtime_stats["stats_path"] = self._write_runtime_stats(save_debug_dir, runtime_stats)
        self.last_runtime_stats = runtime_stats
        if log_runtime_stats:
            logging.info(
                "Wan-Animate runtime summary: total=%.3fs clips=%d static=%.3fs peak_mem=%.3fGB",
                total_generate_sec,
                clip_count,
                static_condition_encode_sec,
                runtime_stats["peak_memory_gb"] or 0.0,
            )
            if runtime_stats["seam_summary"]:
                logging.info(
                    "Seam summary: boundary_before=%.4f boundary_after=%.4f overlap_before=%.4f overlap_after_prev=%.4f",
                    runtime_stats["seam_summary"]["boundary_before"]["mean"] or 0.0,
                    runtime_stats["seam_summary"]["boundary_after"]["mean"] or 0.0,
                    runtime_stats["seam_summary"]["overlap_before"]["mean"] or 0.0,
                    runtime_stats["seam_summary"]["overlap_after_prev"]["mean"] or 0.0,
                )
            if runtime_stats["temporal_handoff_summary"]:
                logging.info(
                    "Temporal handoff summary: mode=%s applied=%d slots=%.2f base_to_memory=%.4f composed_to_base=%.4f",
                    temporal_handoff_mode,
                    runtime_stats["temporal_handoff_summary"]["applied_count"],
                    runtime_stats["temporal_handoff_summary"]["latent_slots"]["mean"] or 0.0,
                    runtime_stats["temporal_handoff_summary"]["base_to_memory_mad"]["mean"] or 0.0,
                    runtime_stats["temporal_handoff_summary"]["composed_to_base_mad"]["mean"] or 0.0,
                )
            if runtime_stats.get("boundary_refinement", {}).get("applied"):
                logging.info(
                    "Boundary refinement summary: source=%s runtime=%.3fs gradient_before=%.4f gradient_after=%.4f halo_before=%.4f halo_after=%.4f",
                    runtime_stats["boundary_refinement"].get("background_source"),
                    runtime_stats["boundary_refinement"].get("runtime_sec") or 0.0,
                    runtime_stats["boundary_refinement"].get("band_gradient_before_mean") or 0.0,
                    runtime_stats["boundary_refinement"].get("band_gradient_after_mean") or 0.0,
                    runtime_stats["boundary_refinement"].get("halo_ratio_before") or 0.0,
                    runtime_stats["boundary_refinement"].get("halo_ratio_after") or 0.0,
                )
        return videos[0] if self.rank == 0 else None
