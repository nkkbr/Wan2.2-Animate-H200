from __future__ import annotations

import numpy as np
import torch

try:
    from .external_alpha_base import ExternalAlphaAdapterBase, ExternalAlphaConfig
except ImportError:
    from external_alpha_base import ExternalAlphaAdapterBase, ExternalAlphaConfig


BackgroundMattingV2Config = ExternalAlphaConfig


class BackgroundMattingV2Adapter(ExternalAlphaAdapterBase):
    requires_background = True

    def load_model(self):
        model = torch.jit.load(str(self.weight_path), map_location=self.device)
        return model.eval().to(self.device)

    def _infer_impl(self, src_rgb: np.ndarray, bgr_rgb: np.ndarray | None = None) -> dict:
        if bgr_rgb is None:
            raise ValueError("BackgroundMattingV2Adapter requires background input")
        src = (
            torch.from_numpy(src_rgb.astype(np.float32) / 255.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device, self.dtype)
        )
        bgr = (
            torch.from_numpy(bgr_rgb.astype(np.float32) / 255.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device, self.dtype)
        )
        with torch.inference_mode():
            pha, fgr, *_ = self.model(src, bgr)
        alpha = pha[0, 0].detach().float().cpu().numpy()
        foreground = fgr[0].detach().float().cpu().permute(1, 2, 0).numpy()
        alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)
        confidence = np.clip(1.0 - np.abs(alpha - 0.5) * 2.0, 0.0, 1.0).astype(np.float32)
        hair_alpha = np.clip(alpha * np.maximum(confidence > 0.35, alpha > 0.7), 0.0, 1.0)
        return {
            "alpha": alpha,
            "foreground": np.clip(foreground, 0.0, 1.0).astype(np.float32),
            "hair_alpha": hair_alpha.astype(np.float32),
            "trimap_unknown": np.logical_and(alpha > 0.05, alpha < 0.95).astype(np.float32),
            "alpha_confidence": confidence.astype(np.float32),
            "alpha_source_provenance": np.ones_like(alpha, dtype=np.float32),
        }
