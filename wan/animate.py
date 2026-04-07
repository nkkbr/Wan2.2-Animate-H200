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
import torch.nn.functional as F
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
    PERSON_MASK_SEMANTICS,
    load_image_rgb,
    resolve_preprocess_artifacts,
    validate_loaded_preprocess_bundle,
    validate_refert_num,
)
from .utils.media_io import load_person_mask_artifact, load_rgb_artifact



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
        refer_images = self.padding_resize(refer_images, height=height, width=width)
        return cond_images, face_images, refer_images
    
    def prepare_source_for_replace(self, src_bg_artifact, src_mask_artifact):
        bg_images = load_rgb_artifact(src_bg_artifact["path"], src_bg_artifact.get("format"))
        person_mask_images = load_person_mask_artifact(src_mask_artifact["path"], src_mask_artifact.get("format"))
        return bg_images, person_mask_images

    def _encode_text_conditions(self, input_prompt, n_prompt, offload_model, guide_scale):
        start_time = time.perf_counter()
        context_null = None
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            if guide_scale > 1:
                context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            if guide_scale > 1:
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

    def generate(
        self,
        src_root_path,
        replace_flag=False,
        clip_len=77,
        refert_num=5,
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
                How many frames are reused for temporal guidance. Supported values are 1 and 5.
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. 
            sample_solver (`str`, *optional*, defaults to 'dpm++'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 20):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float` or tuple[`float`], *optional*, defaults 1.0):
                Classifier-free guidance scale. We only use it for expression control. 
                In most cases, it's not necessary and faster generation can be achieved without it. 
                When expression adjustments are needed, you may consider using this feature.
            input_prompt (`str`):
                Text prompt for content generation. We don't recommend custom prompts (although they work)
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM
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
        refert_num = validate_refert_num(refert_num)
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
                logging.info(
                    "Mask contract: src_mask=%s, generate uses %s.",
                    preprocess_metadata.get("mask_semantics", PERSON_MASK_SEMANTICS),
                    BACKGROUND_KEEP_MASK_SEMANTICS,
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
            guide_scale=guide_scale,
        )

        real_frame_len = len(cond_images)
        target_len = self.get_valid_len(real_frame_len, clip_len, overlap=refert_num)
        logging.info('real frames: {} target frames: {}'.format(real_frame_len, target_len))
        cond_images = self.inputs_padding(cond_images, target_len)
        face_images = self.inputs_padding(face_images, target_len)
        
        if replace_flag:
            bg_images, person_mask_images = self.prepare_source_for_replace(
                src_bg_artifact=artifacts["background"],
                src_mask_artifact=artifacts["person_mask"],
            )
            validate_loaded_preprocess_bundle(
                cond_images=np.asarray(cond_images[:real_frame_len]),
                face_images=np.asarray(face_images[:real_frame_len]),
                refer_image_rgb=refer_images,
                metadata=preprocess_metadata,
                bg_images=bg_images,
                person_mask_images=person_mask_images,
            )
            bg_images = self.inputs_padding(bg_images, target_len)
            person_mask_images = self.inputs_padding(person_mask_images, target_len)
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
        static_condition_encode_sec = text_condition_encode_sec + reference_condition_encode_sec

        runtime_stats = {
            "quality_preset": quality_preset,
            "offload_model": bool(offload_model),
            "clip_len": clip_len,
            "refert_num": refert_num,
            "sampling_steps": sampling_steps,
            "guide_scale": float(guide_scale),
            "real_frame_len": int(real_frame_len),
            "target_len": int(target_len),
            "text_condition_encode_sec": text_condition_encode_sec,
            "reference_condition_encode_sec": reference_condition_encode_sec,
            "static_condition_encode_sec": static_condition_encode_sec,
            "clip_stats": [],
        }
        if log_runtime_stats:
            logging.info(
                "Static animate conditions encoded once: text=%.3fs reference=%.3fs total=%.3fs",
                text_condition_encode_sec,
                reference_condition_encode_sec,
                static_condition_encode_sec,
            )

        start = 0
        end = clip_len
        all_out_frames = []
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
                        "person_mask_pixel_values": torch.zeros(1, 1, clip_len, height, width),
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

                batch["person_mask_pixel_values"] = rearrange(
                    torch.tensor(np.stack(person_mask_images[start:end])[:, :, :, None]),
                    "t h w c -> 1 t c h w",
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

                if mask_reft_len > 0:
                    if replace_flag:
                        bg_pixel_values = batch["bg_pixel_values"]
                        y_reft = self.vae.encode(
                            [
                                torch.concat([refer_t_pixel_values[0, :, :mask_reft_len], bg_pixel_values[0, :, mask_reft_len:]], dim=1).to(self.device)
                            ]
                        )[0]
                        background_keep_mask_pixel_values = 1 - batch["person_mask_pixel_values"]
                        background_keep_mask_pixel_values = rearrange(background_keep_mask_pixel_values, "b t c h w -> (b t) c h w")
                        background_keep_mask_pixel_values = F.interpolate(background_keep_mask_pixel_values, size=(H//8, W//8), mode='nearest')
                        background_keep_mask_pixel_values = rearrange(background_keep_mask_pixel_values, "(b t) c h w -> b t c h w", b=1)[:,:,0]
                        msk_reft = self.get_i2v_mask(lat_t, lat_h, lat_w, mask_reft_len, mask_pixel_values=background_keep_mask_pixel_values, device=self.device)
                    else:
                        y_reft = self.vae.encode(
                            [
                                torch.concat(
                                    [
                                        torch.nn.functional.interpolate(refer_t_pixel_values[0, :, :mask_reft_len].cpu(),
                                                                        size=(H, W), mode="bicubic"),
                                        torch.zeros(3, T - mask_reft_len, H, W),
                                    ],
                                    dim=1,
                                ).to(self.device)
                            ]
                        )[0]
                        msk_reft = self.get_i2v_mask(lat_t, lat_h, lat_w, mask_reft_len, device=self.device)
                else:
                    if replace_flag:
                        bg_pixel_values = batch["bg_pixel_values"]
                        background_keep_mask_pixel_values = 1 - batch["person_mask_pixel_values"]
                        background_keep_mask_pixel_values = rearrange(background_keep_mask_pixel_values, "b t c h w -> (b t) c h w")
                        background_keep_mask_pixel_values = F.interpolate(background_keep_mask_pixel_values, size=(H//8, W//8), mode='nearest')
                        background_keep_mask_pixel_values = rearrange(background_keep_mask_pixel_values, "(b t) c h w -> b t c h w", b=1)[:,:,0]
                        y_reft = self.vae.encode(
                            [
                                torch.concat(
                                    [
                                        bg_pixel_values[0],
                                    ],
                                    dim=1,
                                ).to(self.device)
                            ]
                        )[0]
                        msk_reft = self.get_i2v_mask(lat_t, lat_h, lat_w, mask_reft_len, mask_pixel_values=background_keep_mask_pixel_values, device=self.device)
                    else:
                        y_reft = self.vae.encode(
                            [
                                torch.concat(
                                    [
                                        torch.zeros(3, T - mask_reft_len, H, W),
                                    ],
                                    dim=1,
                                ).to(self.device)
                            ]
                        )[0]
                        msk_reft = self.get_i2v_mask(lat_t, lat_h, lat_w, mask_reft_len, device=self.device)

                clip_vae_encode_sec = time.perf_counter() - vae_encode_start_time
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

                if guide_scale > 1:
                    face_pixel_values_uncond = face_pixel_values * 0 - 1
                    arg_null = {
                        "context": context_null,
                        "seq_len": max_seq_len,
                        "clip_fea": static_reference_conditions["clip_context"],
                        "y": [y],
                        "pose_latents": pose_latents,
                        "face_pixel_values": face_pixel_values_uncond,
                    }

                sampling_start_time = time.perf_counter()
                for i, t in enumerate(tqdm(timesteps)):
                    latent_model_input = latents
                    timestep = [t]

                    timestep = torch.stack(timestep)

                    noise_pred_cond = TensorList(
                         self.noise_model(TensorList(latent_model_input), t=timestep, **arg_c)
                    )

                    if guide_scale > 1:
                        noise_pred_uncond = TensorList(
                             self.noise_model(
                                TensorList(latent_model_input), t=timestep, **arg_null
                            )
                        )
                        noise_pred = noise_pred_uncond + guide_scale * (
                            noise_pred_cond - noise_pred_uncond
                        )
                    else:
                        noise_pred = noise_pred_cond

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
                out_frames = torch.stack(self.vae.decode([x0[0][:, 1:]]))
                clip_vae_decode_sec = time.perf_counter() - vae_decode_start_time
                
                if start != 0:
                    out_frames = out_frames[:, :, refert_num:]

                all_out_frames.append(out_frames.cpu())

                clip_peak_memory_bytes = None
                if torch.cuda.is_available():
                    clip_peak_memory_bytes = int(torch.cuda.max_memory_allocated(self.device))
                clip_total_sec = time.perf_counter() - clip_start_time
                clip_stats = {
                    "clip_index": clip_index,
                    "start_frame": int(start),
                    "end_frame": int(end),
                    "mask_reft_len": int(mask_reft_len),
                    "vae_encode_sec": clip_vae_encode_sec,
                    "sampling_sec": clip_sampling_sec,
                    "vae_decode_sec": clip_vae_decode_sec,
                    "total_sec": clip_total_sec,
                    "peak_memory_bytes": clip_peak_memory_bytes,
                    "peak_memory_gb": round(clip_peak_memory_bytes / (1024 ** 3), 3) if clip_peak_memory_bytes is not None else None,
                }
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
        return videos[0] if self.rank == 0 else None
