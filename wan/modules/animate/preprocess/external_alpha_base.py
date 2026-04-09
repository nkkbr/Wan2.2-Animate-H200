from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from wan.utils.external_alpha_registry import (
    DEFAULT_REGISTRY,
    ensure_external_model_weight,
    get_external_model_entry,
)


@dataclass
class ExternalAlphaConfig:
    model_id: str
    registry_path: str | Path = DEFAULT_REGISTRY
    cache_root: str | Path | None = None
    device: str = "cuda"
    precision: str = "fp32"


class ExternalAlphaAdapterBase(ABC):
    requires_background = False

    def __init__(self, config: ExternalAlphaConfig):
        self.config = config
        self.entry = get_external_model_entry(config.model_id, path=config.registry_path)
        self.adapter_kwargs = dict(self.entry.get("adapter_kwargs", {}))
        self.weight_path = ensure_external_model_weight(
            config.model_id,
            cache_root=config.cache_root,
            registry_path=config.registry_path,
        )
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.dtype = (
            torch.float16
            if config.precision == "fp16" and self.device.type == "cuda"
            else torch.float32
        )
        self.sequence_context = {}
        self.model = self.load_model()
        self.reset_sequence_state()

    @abstractmethod
    def load_model(self):
        raise NotImplementedError

    @abstractmethod
    def _infer_impl(self, src_rgb: np.ndarray, background_rgb: np.ndarray | None = None) -> dict:
        raise NotImplementedError

    def reset_sequence_state(self) -> None:
        self.sequence_state = None
        self.sequence_context = {}

    def set_sequence_context(self, **kwargs) -> None:
        self.sequence_context = dict(kwargs)

    def validate_inputs(
        self,
        src_rgb: np.ndarray,
        background_rgb: np.ndarray | None = None,
    ) -> None:
        if src_rgb.ndim != 3 or src_rgb.shape[-1] != 3:
            raise ValueError(f"Expected src_rgb [H,W,3], got {src_rgb.shape}")
        if background_rgb is not None and background_rgb.shape != src_rgb.shape:
            raise ValueError(
                f"Source/background shape mismatch: {src_rgb.shape} vs {background_rgb.shape}"
            )
        if self.requires_background and background_rgb is None:
            raise ValueError(f"{self.config.model_id} requires background_rgb")

    @staticmethod
    def _default_unknown_from_alpha(alpha: np.ndarray, low: float = 0.05, high: float = 0.95) -> np.ndarray:
        return np.logical_and(alpha > low, alpha < high).astype(np.float32)

    @staticmethod
    def _default_confidence_from_alpha(alpha: np.ndarray) -> np.ndarray:
        return np.clip(1.0 - np.abs(alpha - 0.5) * 2.0, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _default_hair_alpha(alpha: np.ndarray, trimap_unknown: np.ndarray) -> np.ndarray:
        return np.clip(alpha * np.maximum(trimap_unknown, (alpha > 0.7).astype(np.float32)), 0.0, 1.0)

    def _emit_provenance(self) -> dict:
        return {
            "model_id": self.config.model_id,
            "source_repo": self.entry.get("source_repo"),
            "release_or_commit": self.entry.get("release_or_commit"),
            "weight_url": self.entry.get("weight_url"),
            "sha256": self.entry.get("sha256"),
            "license": self.entry.get("license"),
            "weight_path": str(self.weight_path),
        }

    def _finalize_output(self, raw: dict, runtime_sec: float) -> dict:
        alpha = np.clip(np.asarray(raw["alpha"], dtype=np.float32), 0.0, 1.0)
        trimap_unknown = raw.get("trimap_unknown")
        if trimap_unknown is None:
            trimap_unknown = self._default_unknown_from_alpha(alpha)
        else:
            trimap_unknown = np.clip(np.asarray(trimap_unknown, dtype=np.float32), 0.0, 1.0)
        alpha_confidence = raw.get("alpha_confidence")
        if alpha_confidence is None:
            alpha_confidence = self._default_confidence_from_alpha(alpha)
        else:
            alpha_confidence = np.clip(np.asarray(alpha_confidence, dtype=np.float32), 0.0, 1.0)
        hair_alpha = raw.get("hair_alpha")
        if hair_alpha is None:
            hair_alpha = self._default_hair_alpha(alpha, trimap_unknown)
        else:
            hair_alpha = np.clip(np.asarray(hair_alpha, dtype=np.float32), 0.0, 1.0)
        alpha_source_provenance = raw.get("alpha_source_provenance")
        if alpha_source_provenance is None:
            alpha_source_provenance = np.ones_like(alpha, dtype=np.float32)
        else:
            alpha_source_provenance = np.clip(
                np.asarray(alpha_source_provenance, dtype=np.float32),
                0.0,
                1.0,
            )

        output = {
            "alpha": alpha,
            "trimap_unknown": trimap_unknown,
            "hair_alpha": hair_alpha,
            "alpha_confidence": alpha_confidence,
            "alpha_source_provenance": alpha_source_provenance,
            "soft_alpha_ext": alpha,
            "trimap_unknown_ext": trimap_unknown,
            "hair_alpha_ext": hair_alpha,
            "alpha_confidence_ext": alpha_confidence,
            "alpha_source_provenance_ext": alpha_source_provenance,
            "runtime_sec": float(runtime_sec),
            **self._emit_provenance(),
        }
        if raw.get("foreground") is not None:
            output["foreground"] = np.clip(
                np.asarray(raw["foreground"], dtype=np.float32),
                0.0,
                1.0,
            )
        for key, value in raw.items():
            if key not in output:
                output[key] = value
        return output

    def infer(self, src_rgb: np.ndarray, background_rgb: np.ndarray | None = None) -> dict:
        self.validate_inputs(src_rgb, background_rgb)
        start = time.perf_counter()
        raw = self._infer_impl(src_rgb, background_rgb)
        runtime_sec = time.perf_counter() - start
        return self._finalize_output(raw, runtime_sec=runtime_sec)


def build_external_alpha_adapter(
    *,
    model_id: str,
    registry_path: str | Path = DEFAULT_REGISTRY,
    cache_root: str | Path | None = None,
    device: str = "cuda",
    precision: str = "fp32",
) -> ExternalAlphaAdapterBase:
    entry = get_external_model_entry(model_id, path=registry_path)
    task_type = entry.get("task_type", "")
    config = ExternalAlphaConfig(
        model_id=model_id,
        registry_path=registry_path,
        cache_root=cache_root,
        device=device,
        precision=precision,
    )
    if task_type == "background_matting_with_clean_plate":
        try:
            from .external_alpha_backgroundmattingv2 import BackgroundMattingV2Adapter
        except ImportError:
            from external_alpha_backgroundmattingv2 import BackgroundMattingV2Adapter

        return BackgroundMattingV2Adapter(config)
    if task_type == "video_matting":
        try:
            from .external_alpha_rvm import RobustVideoMattingAdapter
        except ImportError:
            from external_alpha_rvm import RobustVideoMattingAdapter

        return RobustVideoMattingAdapter(config)
    if task_type == "video_matting_with_first_frame_mask":
        try:
            from .external_alpha_matanyone import MatAnyoneAdapter
        except ImportError:
            from external_alpha_matanyone import MatAnyoneAdapter

        return MatAnyoneAdapter(config)
    raise ValueError(f"Unsupported external alpha task type for {model_id}: {task_type}")
