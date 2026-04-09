from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

from wan.utils.external_alpha_registry import ensure_external_model_weight, get_external_model_entry


@dataclass
class BackgroundMattingV2Config:
    model_id: str
    device: str = "cuda"
    precision: str = "fp32"


class BackgroundMattingV2Adapter:
    def __init__(self, config: BackgroundMattingV2Config):
        self.config = config
        self.entry = get_external_model_entry(config.model_id)
        self.weight_path = ensure_external_model_weight(config.model_id)
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if config.precision == "fp16" and self.device.type == "cuda" else torch.float32
        self.model = torch.jit.load(str(self.weight_path), map_location=self.device)
        self.model = self.model.eval().to(self.device)

    def infer(self, src_rgb: np.ndarray, bgr_rgb: np.ndarray) -> dict:
        if src_rgb.shape != bgr_rgb.shape:
            raise ValueError(f"Source/background shape mismatch: {src_rgb.shape} vs {bgr_rgb.shape}")
        if src_rgb.ndim != 3 or src_rgb.shape[-1] != 3:
            raise ValueError(f"Expected [H,W,3] RGB arrays. Got {src_rgb.shape}")

        src = torch.from_numpy(src_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(self.device, self.dtype)
        bgr = torch.from_numpy(bgr_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(self.device, self.dtype)

        start = time.perf_counter()
        with torch.inference_mode():
            pha, fgr, *_ = self.model(src, bgr)
        runtime_sec = time.perf_counter() - start

        alpha = pha[0, 0].detach().float().cpu().numpy()
        foreground = fgr[0].detach().float().cpu().permute(1, 2, 0).numpy()
        foreground = np.clip(foreground, 0.0, 1.0)
        alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)
        confidence = np.clip(1.0 - np.abs(alpha - 0.5) * 2.0, 0.0, 1.0).astype(np.float32)
        hair_alpha = np.clip(alpha * (confidence > 0.35).astype(np.float32), 0.0, 1.0)

        return {
            "alpha": alpha,
            "foreground": foreground.astype(np.float32),
            "hair_alpha": hair_alpha.astype(np.float32),
            "trimap_unknown": np.logical_and(alpha > 0.05, alpha < 0.95).astype(np.float32),
            "alpha_confidence": confidence.astype(np.float32),
            "alpha_source_provenance": np.ones_like(alpha, dtype=np.float32),
            "runtime_sec": runtime_sec,
            "weight_path": str(self.weight_path),
            "model_id": self.config.model_id,
            "source_repo": self.entry.get("source_repo"),
            "release_or_commit": self.entry.get("release_or_commit"),
            "weight_url": self.entry.get("weight_url"),
            "sha256": self.entry.get("sha256"),
            "license": self.entry.get("license"),
        }
