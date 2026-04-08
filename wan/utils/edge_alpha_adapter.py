import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .animate_contract import (
    ALPHA_CONFIDENCE_SEMANTICS,
    ALPHA_SOURCE_PROVENANCE_SEMANTICS,
    PERSON_MASK_SEMANTICS,
    SOFT_ALPHA_SEMANTICS,
    TRIMAP_UNKNOWN_SEMANTICS,
    load_preprocess_metadata,
)
from .media_io import load_mask_artifact, load_person_mask_artifact, write_person_mask_artifact


ADAPTER_FEATURE_KEYS = [
    "person_mask",
    "soft_alpha",
    "boundary_band",
    "occlusion_band",
    "uncertainty_map",
    "visible_support",
    "unresolved_region",
    "soft_band",
    "face_alpha",
    "face_uncertainty",
]


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EdgeAlphaAdapter(nn.Module):
    def __init__(self, in_channels: int, width: int = 24, residual_scale: float = 0.16):
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.down1 = _ConvBlock(in_channels, width)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.down2 = _ConvBlock(width, width * 2)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.mid = _ConvBlock(width * 2, width * 2)
        self.up2 = nn.ConvTranspose2d(width * 2, width * 2, kernel_size=2, stride=2)
        self.dec2 = _ConvBlock(width * 4, width)
        self.up1 = nn.ConvTranspose2d(width, width, kernel_size=2, stride=2)
        self.dec1 = _ConvBlock(width * 2, width)
        self.out = nn.Conv2d(width, 1, kernel_size=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        base_alpha = features[:, 1:2]
        x1 = self.down1(features)
        x2 = self.down2(self.pool1(x1))
        xm = self.mid(self.pool2(x2))
        x = self.up2(xm)
        x = self.dec2(torch.cat([x, x2], dim=1))
        x = self.up1(x)
        x = self.dec1(torch.cat([x, x1], dim=1))
        residual = torch.tanh(self.out(x)) * self.residual_scale
        return torch.clamp(base_alpha + residual, 0.0, 1.0)


def create_edge_alpha_adapter(checkpoint: dict) -> EdgeAlphaAdapter:
    feature_keys = checkpoint.get("feature_keys", ADAPTER_FEATURE_KEYS)
    width = int(checkpoint.get("width", 24))
    residual_scale = float(checkpoint.get("residual_scale", 0.16))
    model = EdgeAlphaAdapter(len(feature_keys), width=width, residual_scale=residual_scale)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def _load_optional_mask(preprocess_dir: Path, metadata: dict, key: str, fallback: np.ndarray) -> np.ndarray:
    if key not in metadata["src_files"]:
        return np.zeros_like(fallback, dtype=np.float32)
    artifact = metadata["src_files"][key]
    loader = load_person_mask_artifact if artifact.get("type") == "video" else load_mask_artifact
    return loader(preprocess_dir / artifact["path"], artifact.get("format")).astype(np.float32)


def load_adapter_feature_maps(preprocess_dir: str | Path, feature_keys: list[str] | None = None) -> tuple[dict, dict]:
    preprocess_dir = Path(preprocess_dir)
    metadata = load_preprocess_metadata(preprocess_dir)
    if metadata is None:
        raise FileNotFoundError(f"Preprocess metadata not found under {preprocess_dir}")
    artifacts = metadata["src_files"]
    base_artifact = artifacts["person_mask"]
    base_mask = load_person_mask_artifact(preprocess_dir / base_artifact["path"], base_artifact.get("format")).astype(np.float32)
    feature_keys = feature_keys or ADAPTER_FEATURE_KEYS
    feature_maps = {}
    for key in feature_keys:
        if key == "person_mask":
            feature_maps[key] = base_mask
            continue
        feature_maps[key] = _load_optional_mask(preprocess_dir, metadata, key, base_mask)
    return feature_maps, metadata


def stack_feature_maps(feature_maps: dict, feature_keys: list[str] | None = None) -> np.ndarray:
    feature_keys = feature_keys or ADAPTER_FEATURE_KEYS
    stacked = [np.asarray(feature_maps[key], dtype=np.float32) for key in feature_keys]
    return np.stack(stacked, axis=1).astype(np.float32)


def predict_edge_alpha(
    *,
    model: EdgeAlphaAdapter,
    features: np.ndarray,
    device: str | torch.device,
    batch_size: int = 4,
) -> np.ndarray:
    device = torch.device(device)
    preds = []
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            batch = torch.from_numpy(features[start:start + batch_size]).to(device=device, dtype=torch.float32)
            pred = model(batch).detach().cpu().numpy()[:, 0]
            preds.append(pred.astype(np.float32))
    return np.concatenate(preds, axis=0).astype(np.float32)


def derive_adapter_artifacts(
    *,
    pred_alpha: np.ndarray,
    feature_maps: dict,
    threshold: float,
) -> dict:
    pred_alpha = np.asarray(pred_alpha, dtype=np.float32)
    base_alpha = np.asarray(feature_maps["soft_alpha"], dtype=np.float32)
    person_mask = np.asarray(feature_maps["person_mask"], dtype=np.float32)
    boundary_band = np.asarray(feature_maps["boundary_band"], dtype=np.float32)
    uncertainty_map = np.asarray(feature_maps["uncertainty_map"], dtype=np.float32)
    soft_band = np.asarray(feature_maps["soft_band"], dtype=np.float32)
    face_alpha = np.asarray(feature_maps.get("face_alpha", np.zeros_like(pred_alpha)), dtype=np.float32)

    support = np.clip(np.maximum.reduce([person_mask, boundary_band, 0.65 * soft_band]), 0.0, 1.0)
    adapted_alpha = np.clip(np.minimum(pred_alpha, np.maximum(support, 0.92 * pred_alpha)), 0.0, 1.0).astype(np.float32)
    adapted_hard = (adapted_alpha >= float(threshold)).astype(np.float32)
    adapted_hard = np.maximum(adapted_hard, (person_mask > 0.75).astype(np.float32) * 0.25)
    adapted_hard = (adapted_hard >= 0.5).astype(np.float32)

    trimap_unknown = np.clip(
        ((adapted_alpha > 0.08) & (adapted_alpha < 0.92)).astype(np.float32)
        * np.maximum(boundary_band, 0.45 * soft_band),
        0.0,
        1.0,
    ).astype(np.float32)
    alpha_confidence = np.clip(
        1.0 - (
            0.55 * np.abs(adapted_alpha - base_alpha)
            + 0.30 * uncertainty_map
            + 0.15 * boundary_band
        ),
        0.0,
        1.0,
    ).astype(np.float32)
    hair_alpha = np.clip(
        adapted_alpha * np.maximum(boundary_band, 0.55 * soft_band) * np.clip(1.0 - 0.65 * face_alpha, 0.0, 1.0),
        0.0,
        1.0,
    ).astype(np.float32)
    alpha_source_provenance = np.clip(
        np.where(np.abs(adapted_alpha - base_alpha) > 0.01, 1.0, 0.25),
        0.0,
        1.0,
    ).astype(np.float32)

    return {
        "soft_alpha": adapted_alpha,
        "person_mask": adapted_hard,
        "trimap_unknown": trimap_unknown,
        "alpha_confidence": alpha_confidence,
        "hair_alpha": hair_alpha,
        "alpha_source_provenance": alpha_source_provenance,
    }


def write_adapter_bundle(
    *,
    source_preprocess_dir: str | Path,
    output_preprocess_dir: str | Path,
    artifacts: dict,
    checkpoint_info: dict,
) -> Path:
    import shutil

    source_preprocess_dir = Path(source_preprocess_dir)
    output_preprocess_dir = Path(output_preprocess_dir)
    if output_preprocess_dir.exists():
        shutil.rmtree(output_preprocess_dir)
    shutil.copytree(source_preprocess_dir, output_preprocess_dir)

    metadata_path = output_preprocess_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    fps = float(metadata["fps"])

    person_info = write_person_mask_artifact(
        mask_frames=artifacts["person_mask"],
        output_root=output_preprocess_dir,
        stem="src_mask",
        artifact_format="npz",
        fps=fps,
        mask_semantics=PERSON_MASK_SEMANTICS,
    )
    soft_alpha_info = write_person_mask_artifact(
        mask_frames=artifacts["soft_alpha"],
        output_root=output_preprocess_dir,
        stem="src_soft_alpha",
        artifact_format="npz",
        fps=fps,
        mask_semantics=SOFT_ALPHA_SEMANTICS,
    )
    trimap_info = write_person_mask_artifact(
        mask_frames=artifacts["trimap_unknown"],
        output_root=output_preprocess_dir,
        stem="src_trimap_unknown",
        artifact_format="npz",
        fps=fps,
        mask_semantics=TRIMAP_UNKNOWN_SEMANTICS,
    )
    conf_info = write_person_mask_artifact(
        mask_frames=artifacts["alpha_confidence"],
        output_root=output_preprocess_dir,
        stem="src_alpha_confidence",
        artifact_format="npz",
        fps=fps,
        mask_semantics=ALPHA_CONFIDENCE_SEMANTICS,
    )
    hair_info = write_person_mask_artifact(
        mask_frames=artifacts["hair_alpha"],
        output_root=output_preprocess_dir,
        stem="src_hair_alpha",
        artifact_format="npz",
        fps=fps,
        mask_semantics="hair_alpha",
    )
    prov_info = write_person_mask_artifact(
        mask_frames=artifacts["alpha_source_provenance"],
        output_root=output_preprocess_dir,
        stem="src_alpha_source_provenance",
        artifact_format="npz",
        fps=fps,
        mask_semantics=ALPHA_SOURCE_PROVENANCE_SEMANTICS,
    )

    metadata["src_files"]["person_mask"] = person_info
    metadata["src_files"]["soft_alpha"] = soft_alpha_info
    metadata["src_files"]["trimap_unknown"] = trimap_info
    metadata["src_files"]["alpha_confidence"] = conf_info
    metadata["src_files"]["hair_alpha"] = hair_info
    metadata["src_files"]["alpha_source_provenance"] = prov_info
    metadata.setdefault("processing", {}).setdefault("matting", {}).update({
        "mode": "edge_adapter_v1",
        "adapter_checkpoint": checkpoint_info.get("checkpoint_path"),
        "adapter_threshold": float(checkpoint_info.get("threshold", 0.5)),
        "adapter_feature_keys": list(checkpoint_info.get("feature_keys", ADAPTER_FEATURE_KEYS)),
    })
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output_preprocess_dir
