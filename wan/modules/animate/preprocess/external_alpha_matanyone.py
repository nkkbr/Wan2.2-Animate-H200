from __future__ import annotations

import numpy as np
import torch
from torchvision.transforms.functional import to_tensor

try:
    from .external_alpha_base import ExternalAlphaAdapterBase, ExternalAlphaConfig
except ImportError:
    from external_alpha_base import ExternalAlphaAdapterBase, ExternalAlphaConfig


class MatAnyoneAdapter(ExternalAlphaAdapterBase):
    requires_background = False

    def __init__(self, config: ExternalAlphaConfig):
        self.processor = None
        self._initialized = False
        super().__init__(config)

    def load_model(self):
        try:
            from matanyone import InferenceCore  # noqa: F401
            from matanyone.utils.get_default_model import get_matanyone_model
        except ImportError as exc:
            raise ImportError(
                "MatAnyone is not installed. Install it before using "
                f"{self.config.model_id}."
            ) from exc

        model = get_matanyone_model(str(self.weight_path), device=self.device)
        return model.eval().to(self.device)

    def reset_sequence_state(self) -> None:
        super().reset_sequence_state()
        from matanyone import InferenceCore

        self.processor = InferenceCore(self.model, cfg=self.model.cfg, device=self.device)
        self._initialized = False

    def set_sequence_context(self, **kwargs) -> None:
        super().set_sequence_context(**kwargs)
        initial_mask = kwargs.get("initial_mask")
        if initial_mask is None:
            raise ValueError("MatAnyoneAdapter requires initial_mask sequence context")
        mask = np.asarray(initial_mask)
        if mask.ndim != 2:
            raise ValueError(f"Expected initial_mask [H,W], got {mask.shape}")
        if mask.dtype != np.uint8:
            if mask.max() <= 1.0:
                mask = np.clip(mask * 255.0, 0.0, 255.0).astype(np.uint8)
            else:
                mask = np.clip(mask, 0.0, 255.0).astype(np.uint8)
        self.sequence_context["initial_mask"] = mask

    def _infer_impl(self, src_rgb: np.ndarray, background_rgb: np.ndarray | None = None) -> dict:
        if self.processor is None:
            self.reset_sequence_state()
        initial_mask = self.sequence_context.get("initial_mask")
        if initial_mask is None:
            raise ValueError("MatAnyoneAdapter requires set_sequence_context(initial_mask=...)")
        if initial_mask.shape != src_rgb.shape[:2]:
            raise ValueError(
                f"Initial mask/frame shape mismatch: {initial_mask.shape} vs {src_rgb.shape[:2]}"
            )

        image = to_tensor(src_rgb).float().to(self.device)
        if not self._initialized:
            mask = torch.from_numpy(initial_mask).to(self.device)
            output_prob = self.processor.step(image, mask, objects=[1])
            warmup_frames = int(self.adapter_kwargs.get("warmup_frames", 10))
            for _ in range(warmup_frames):
                output_prob = self.processor.step(image, first_frame_pred=True)
            output_prob = self.processor.step(image)
            self._initialized = True
        else:
            output_prob = self.processor.step(image)

        alpha = self.processor.output_prob_to_mask(output_prob).detach().float().cpu().numpy()
        alpha = np.clip(alpha.astype(np.float32), 0.0, 1.0)
        trimap_unknown = np.logical_and(alpha > 0.04, alpha < 0.96).astype(np.float32)
        alpha_confidence = np.clip(
            1.0 - np.maximum(np.abs(alpha - 0.5) * 1.7, trimap_unknown * 0.25),
            0.0,
            1.0,
        ).astype(np.float32)
        hair_alpha = np.clip(
            alpha * np.maximum(trimap_unknown, (alpha > 0.75).astype(np.float32)),
            0.0,
            1.0,
        ).astype(np.float32)
        foreground = (src_rgb.astype(np.float32) / 255.0) * alpha[:, :, None]
        return {
            "alpha": alpha,
            "foreground": foreground.astype(np.float32),
            "trimap_unknown": trimap_unknown.astype(np.float32),
            "alpha_confidence": alpha_confidence.astype(np.float32),
            "hair_alpha": hair_alpha.astype(np.float32),
            "alpha_source_provenance": np.ones_like(alpha, dtype=np.float32),
            "warmup_frames": int(self.adapter_kwargs.get("warmup_frames", 10)),
        }
