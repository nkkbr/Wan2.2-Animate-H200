from __future__ import annotations

from typing import Sequence

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


class SemanticEdgeExpertsModel(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        semantic_tags: Sequence[str],
        width: int = 32,
        residual_scale: float = 0.20,
        semantic_emb_dim: int = 8,
    ):
        super().__init__()
        self.semantic_tags = tuple(str(tag) for tag in semantic_tags)
        self.tag_to_index = {tag: idx for idx, tag in enumerate(self.semantic_tags)}
        self.residual_scale = float(residual_scale)
        self.semantic_embedding = nn.Embedding(len(self.semantic_tags), semantic_emb_dim)
        trunk_in = in_channels + semantic_emb_dim
        self.enc1 = _Block(trunk_in, width)
        self.pool1 = nn.AvgPool2d(2, 2)
        self.enc2 = _Block(width, width * 2)
        self.pool2 = nn.AvgPool2d(2, 2)
        self.mid = _Block(width * 2, width * 2)
        self.up2 = nn.ConvTranspose2d(width * 2, width * 2, kernel_size=2, stride=2)
        self.dec2 = _Block(width * 4, width)
        self.up1 = nn.ConvTranspose2d(width, width, kernel_size=2, stride=2)
        self.dec1 = _Block(width * 2, width)
        self.heads = nn.ModuleDict({
            tag: nn.Conv2d(width, 1, kernel_size=1)
            for tag in self.semantic_tags
        })

    def forward(self, x: torch.Tensor, base_alpha: torch.Tensor, semantic_ids: torch.Tensor) -> torch.Tensor:
        emb = self.semantic_embedding(semantic_ids.long())
        emb = emb[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])
        x = torch.cat([x, emb], dim=1)
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        xm = self.mid(self.pool2(x2))
        x = self.up2(xm)
        x = self.dec2(torch.cat([x, x2], dim=1))
        x = self.up1(x)
        feat = self.dec1(torch.cat([x, x1], dim=1))
        residual = []
        for idx in range(feat.shape[0]):
            tag = self.semantic_tags[int(semantic_ids[idx].item())]
            residual.append(self.heads[tag](feat[idx:idx + 1]))
        residual = torch.cat(residual, dim=0)
        residual = torch.tanh(residual) * self.residual_scale
        return torch.clamp(base_alpha + residual, 0.0, 1.0)


def build_model_from_checkpoint(checkpoint: dict) -> SemanticEdgeExpertsModel:
    model = SemanticEdgeExpertsModel(
        in_channels=int(checkpoint["in_channels"]),
        semantic_tags=tuple(checkpoint["semantic_tags"]),
        width=int(checkpoint.get("width", 32)),
        residual_scale=float(checkpoint.get("residual_scale", 0.20)),
        semantic_emb_dim=int(checkpoint.get("semantic_emb_dim", 8)),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def predict_patches(
    *,
    model: SemanticEdgeExpertsModel,
    inputs: np.ndarray,
    base_alpha: np.ndarray,
    semantic_ids: np.ndarray,
    device: str | torch.device,
    batch_size: int = 8,
) -> np.ndarray:
    model.eval()
    device = torch.device(device)
    outputs = []
    with torch.no_grad():
        for start in range(0, inputs.shape[0], batch_size):
            batch_inputs = torch.from_numpy(inputs[start:start + batch_size]).to(device=device, dtype=torch.float32).permute(0, 3, 1, 2)
            batch_alpha = torch.from_numpy(base_alpha[start:start + batch_size]).to(device=device, dtype=torch.float32)[:, None]
            batch_semantic = torch.from_numpy(semantic_ids[start:start + batch_size]).to(device=device, dtype=torch.long)
            pred = model(batch_inputs, batch_alpha, batch_semantic).detach().cpu().numpy()[:, 0]
            outputs.append(pred.astype(np.float32))
    return np.concatenate(outputs, axis=0).astype(np.float32)
