from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class _Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=max(1, out_channels // 8), num_channels=out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=max(1, out_channels // 8), num_channels=out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_bridge_focus_map(
    *,
    input_trimap_unknown: np.ndarray,
    input_boundary_roi_mask: np.ndarray,
    input_boundary_band: np.ndarray,
    input_uncertainty: np.ndarray,
) -> np.ndarray:
    trimap = np.asarray(input_trimap_unknown, dtype=np.float32)
    roi = np.asarray(input_boundary_roi_mask, dtype=np.float32)
    band = np.asarray(input_boundary_band, dtype=np.float32)
    uncertainty = np.asarray(input_uncertainty, dtype=np.float32)
    focus = np.maximum.reduce([
        trimap,
        roi,
        band,
        np.clip(1.5 * uncertainty, 0.0, 1.0),
    ])
    return np.clip(focus, 0.0, 1.0).astype(np.float32)


def stack_bridge_input_channels(
    *,
    foreground_patch: np.ndarray,
    background_patch: np.ndarray,
    input_soft_alpha: np.ndarray,
    input_trimap_unknown: np.ndarray,
    input_boundary_roi_mask: np.ndarray,
    input_person_mask: np.ndarray,
    input_boundary_band: np.ndarray,
    input_uncertainty: np.ndarray,
) -> np.ndarray:
    fg = np.asarray(foreground_patch, dtype=np.float32) / 255.0
    bg = np.asarray(background_patch, dtype=np.float32) / 255.0
    alpha = np.asarray(input_soft_alpha, dtype=np.float32)[..., None]
    trimap = np.asarray(input_trimap_unknown, dtype=np.float32)[..., None]
    roi = np.asarray(input_boundary_roi_mask, dtype=np.float32)[..., None]
    person = np.asarray(input_person_mask, dtype=np.float32)[..., None]
    band = np.asarray(input_boundary_band, dtype=np.float32)[..., None]
    uncertainty = np.asarray(input_uncertainty, dtype=np.float32)[..., None]
    rgb_delta = np.abs(fg - bg).mean(axis=2, keepdims=True).astype(np.float32)
    focus = build_bridge_focus_map(
        input_trimap_unknown=input_trimap_unknown,
        input_boundary_roi_mask=input_boundary_roi_mask,
        input_boundary_band=input_boundary_band,
        input_uncertainty=input_uncertainty,
    )[..., None]
    return np.concatenate(
        [fg, bg, alpha, trimap, roi, person, band, uncertainty, rgb_delta, focus],
        axis=2,
    ).astype(np.float32)


class MatteBridgeModel(nn.Module):
    def __init__(self, in_channels: int, width: int = 32):
        super().__init__()
        self.enc1 = _Block(in_channels, width)
        self.pool1 = nn.AvgPool2d(2, 2)
        self.enc2 = _Block(width, width * 2)
        self.pool2 = nn.AvgPool2d(2, 2)
        self.mid = _Block(width * 2, width * 4)
        self.up2 = nn.ConvTranspose2d(width * 4, width * 2, kernel_size=2, stride=2)
        self.dec2 = _Block(width * 4, width * 2)
        self.up1 = nn.ConvTranspose2d(width * 2, width, kernel_size=2, stride=2)
        self.dec1 = _Block(width * 2, width)
        self.out = nn.Conv2d(width, 2, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        base_alpha: torch.Tensor,
        focus_map: torch.Tensor,
        *,
        gate_strength: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        xm = self.mid(self.pool2(x2))
        x = self.up2(xm)
        x = self.dec2(torch.cat([x, x2], dim=1))
        x = self.up1(x)
        x = self.dec1(torch.cat([x, x1], dim=1))
        raw = self.out(x)
        pred_matte = torch.sigmoid(raw[:, :1])
        pred_gate = torch.sigmoid(raw[:, 1:2])
        bridge_gate = torch.clamp(pred_gate * focus_map * float(gate_strength), 0.0, 1.0)
        final_alpha = torch.clamp(base_alpha * (1.0 - bridge_gate) + pred_matte * bridge_gate, 0.0, 1.0)
        return pred_matte, bridge_gate, final_alpha


def build_model_from_checkpoint(checkpoint: dict) -> MatteBridgeModel:
    model = MatteBridgeModel(
        in_channels=int(checkpoint["in_channels"]),
        width=int(checkpoint.get("width", 32)),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def predict_bridge_patches(
    *,
    model: MatteBridgeModel,
    inputs: np.ndarray,
    base_alpha: np.ndarray,
    focus_map: np.ndarray,
    device: str | torch.device,
    batch_size: int = 8,
    gate_strength: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    device = torch.device(device)
    pred_mattes = []
    pred_gates = []
    final_alphas = []
    with torch.no_grad():
        for start in range(0, inputs.shape[0], batch_size):
            batch_inputs = torch.from_numpy(inputs[start:start + batch_size]).to(device=device, dtype=torch.float32).permute(0, 3, 1, 2)
            batch_alpha = torch.from_numpy(base_alpha[start:start + batch_size]).to(device=device, dtype=torch.float32)[:, None]
            batch_focus = torch.from_numpy(focus_map[start:start + batch_size]).to(device=device, dtype=torch.float32)[:, None]
            pred_matte, pred_gate, final_alpha = model(
                batch_inputs,
                batch_alpha,
                batch_focus,
                gate_strength=gate_strength,
            )
            pred_mattes.append(pred_matte.detach().cpu().numpy()[:, 0].astype(np.float32))
            pred_gates.append(pred_gate.detach().cpu().numpy()[:, 0].astype(np.float32))
            final_alphas.append(final_alpha.detach().cpu().numpy()[:, 0].astype(np.float32))
    return (
        np.concatenate(pred_mattes, axis=0).astype(np.float32),
        np.concatenate(pred_gates, axis=0).astype(np.float32),
        np.concatenate(final_alphas, axis=0).astype(np.float32),
    )
