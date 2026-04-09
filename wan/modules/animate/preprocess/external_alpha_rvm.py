from __future__ import annotations

import numpy as np
import torch

try:
    from .external_alpha_base import ExternalAlphaAdapterBase, ExternalAlphaConfig
except ImportError:
    from external_alpha_base import ExternalAlphaAdapterBase, ExternalAlphaConfig


class RobustVideoMattingAdapter(ExternalAlphaAdapterBase):
    requires_background = False

    def __init__(self, config: ExternalAlphaConfig):
        self._recurrent_state = (None, None, None, None)
        super().__init__(config)

    def load_model(self):
        model = torch.jit.load(str(self.weight_path), map_location=self.device)
        return model.eval().to(self.device)

    def reset_sequence_state(self) -> None:
        self._recurrent_state = (None, None, None, None)

    def _infer_impl(self, src_rgb: np.ndarray, background_rgb: np.ndarray | None = None) -> dict:
        src = (
            torch.from_numpy(src_rgb.astype(np.float32) / 255.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device, self.dtype)
        )
        with torch.inference_mode():
            fgr, pha, r1, r2, r3, r4 = self.model(
                src,
                self._recurrent_state[0],
                self._recurrent_state[1],
                self._recurrent_state[2],
                self._recurrent_state[3],
                1.0,
                False,
            )
        self._recurrent_state = (r1, r2, r3, r4)
        alpha = pha[0, 0].detach().float().cpu().numpy()
        foreground = fgr[0].detach().float().cpu().permute(1, 2, 0).numpy()
        trimap_unknown = np.logical_and(alpha > 0.04, alpha < 0.96).astype(np.float32)
        alpha_confidence = np.clip(
            1.0 - np.maximum(np.abs(alpha - 0.5) * 1.8, trimap_unknown * 0.35),
            0.0,
            1.0,
        ).astype(np.float32)
        hair_alpha = np.clip(alpha * np.maximum(trimap_unknown, (alpha > 0.75).astype(np.float32)), 0.0, 1.0)
        return {
            "alpha": alpha.astype(np.float32),
            "foreground": np.clip(foreground, 0.0, 1.0).astype(np.float32),
            "trimap_unknown": trimap_unknown.astype(np.float32),
            "alpha_confidence": alpha_confidence.astype(np.float32),
            "hair_alpha": hair_alpha.astype(np.float32),
            "alpha_source_provenance": np.ones_like(alpha, dtype=np.float32),
            "recurrent_state_shapes": [
                None if state is None else list(state.shape) for state in self._recurrent_state
            ],
        }
