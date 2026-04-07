import json
import math
from pathlib import Path

import numpy as np
import torch


def overlap_frames_to_latent_slots(overlap_frames: int) -> int:
    if overlap_frames <= 0:
        return 0
    return 1 + ((int(overlap_frames) + 2) // 4)


def _reduce_chunk(chunk: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "mean":
        return chunk.mean(dim=0)
    if reduction == "max":
        return chunk.max(dim=0).values
    raise ValueError(f"Unsupported temporal reduction: {reduction}")


def pack_overlap_tensor_to_latent_slots(
    frames: torch.Tensor | np.ndarray,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    tensor = torch.as_tensor(frames, dtype=torch.float32)
    if tensor.ndim not in {3, 4}:
        raise ValueError(f"Expected [T,H,W] or [T,C,H,W]. Got {tensor.shape}.")
    if tensor.shape[0] == 0:
        spatial_shape = tensor.shape[1:]
        return torch.zeros((0, *spatial_shape), dtype=torch.float32)

    slots = [_reduce_chunk(tensor[0:1], reduction)]
    for start in range(1, tensor.shape[0], 4):
        slots.append(_reduce_chunk(tensor[start:start + 4], reduction))
    return torch.stack(slots, dim=0).to(dtype=torch.float32)


def _mean_abs_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.to(dtype=torch.float32) - right.to(dtype=torch.float32)).abs().mean().item())


def compose_temporal_handoff_latents(
    *,
    base_latents: torch.Tensor,
    previous_output_latents: torch.Tensor | None,
    overlap_frames: int,
    mode: str,
    strength: float,
    replacement_strength_slots: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    base_latents = base_latents.to(dtype=torch.float32)
    if mode not in {"pixel", "latent", "hybrid"}:
        raise ValueError(f"Unsupported temporal handoff mode: {mode}")
    strength = float(max(0.0, min(strength, 1.0)))
    stats = {
        "mode": mode,
        "overlap_frames": int(overlap_frames),
        "latent_slots": 0,
        "applied": False,
        "blend_strength_mean": 0.0,
        "base_to_memory_mad": None,
        "composed_to_base_mad": 0.0,
        "composed_to_memory_mad": None,
    }
    if mode == "pixel" or previous_output_latents is None or overlap_frames <= 0 or strength <= 0.0:
        return base_latents, stats

    previous_output_latents = previous_output_latents.to(device=base_latents.device, dtype=base_latents.dtype)
    latent_slots = min(
        overlap_frames_to_latent_slots(overlap_frames),
        base_latents.shape[1],
        previous_output_latents.shape[1],
    )
    if latent_slots <= 0:
        return base_latents, stats

    memory = previous_output_latents[:, -latent_slots:]
    composed = base_latents.clone()
    if mode == "latent":
        blend = torch.full(
            (1, latent_slots, base_latents.shape[-2], base_latents.shape[-1]),
            strength,
            dtype=base_latents.dtype,
            device=base_latents.device,
        )
    else:
        if replacement_strength_slots is None:
            raise ValueError("replacement_strength_slots is required for hybrid temporal handoff.")
        replacement_strength_slots = torch.as_tensor(
            replacement_strength_slots,
            dtype=base_latents.dtype,
            device=base_latents.device,
        )
        if replacement_strength_slots.ndim == 3:
            replacement_strength_slots = replacement_strength_slots.unsqueeze(0)
        blend = torch.clamp(replacement_strength_slots[:, :latent_slots], 0.0, 1.0) * strength

    composed[:, :latent_slots] = composed[:, :latent_slots] * (1.0 - blend) + memory * blend
    stats.update({
        "latent_slots": int(latent_slots),
        "applied": True,
        "blend_strength_mean": float(blend.mean().item()),
        "base_to_memory_mad": _mean_abs_difference(base_latents[:, :latent_slots], memory),
        "composed_to_base_mad": _mean_abs_difference(composed[:, :latent_slots], base_latents[:, :latent_slots]),
        "composed_to_memory_mad": _mean_abs_difference(composed[:, :latent_slots], memory),
    })
    return composed, stats


def write_temporal_handoff_debug(
    *,
    save_debug_dir: str | Path,
    handoff_index: int,
    stats: dict,
    base_latents: torch.Tensor,
    memory_latents: torch.Tensor | None,
    composed_latents: torch.Tensor,
    blend_mask: torch.Tensor | None = None,
) -> dict[str, str]:
    handoff_dir = Path(save_debug_dir) / "temporal_handoffs"
    handoff_dir.mkdir(parents=True, exist_ok=True)
    stem = f"handoff_{handoff_index:03d}"

    stats_path = handoff_dir / f"{stem}_stats.json"
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")

    payload = {
        "base": base_latents.detach().cpu().numpy().astype(np.float32),
        "composed": composed_latents.detach().cpu().numpy().astype(np.float32),
    }
    if memory_latents is not None:
        payload["memory"] = memory_latents.detach().cpu().numpy().astype(np.float32)
    if blend_mask is not None:
        payload["blend_mask"] = blend_mask.detach().cpu().numpy().astype(np.float32)
    latents_path = handoff_dir / f"{stem}_latents.npz"
    np.savez_compressed(latents_path, **payload)
    return {
        "stats": str(stats_path.resolve()),
        "latents": str(latents_path.resolve()),
    }
