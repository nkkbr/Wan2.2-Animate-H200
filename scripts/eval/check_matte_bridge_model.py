#!/usr/bin/env python
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan.utils.matte_bridge_model import MatteBridgeModel, build_bridge_focus_map, stack_bridge_input_channels


def main() -> None:
    torch.manual_seed(1234)
    np.random.seed(1234)

    fg = np.random.randint(0, 255, size=(64, 64, 3), dtype=np.uint8)
    bg = np.random.randint(0, 255, size=(64, 64, 3), dtype=np.uint8)
    alpha = np.random.rand(64, 64).astype(np.float32)
    trimap = (np.random.rand(64, 64) > 0.7).astype(np.uint8)
    roi = (np.random.rand(64, 64) > 0.8).astype(np.uint8)
    person = (np.random.rand(64, 64) > 0.4).astype(np.uint8)
    band = np.random.rand(64, 64).astype(np.float32) * 0.8
    uncertainty = np.random.rand(64, 64).astype(np.float32) * 0.5
    focus = build_bridge_focus_map(
        input_trimap_unknown=trimap,
        input_boundary_roi_mask=roi,
        input_boundary_band=band,
        input_uncertainty=uncertainty,
    )
    stacked = stack_bridge_input_channels(
        foreground_patch=fg,
        background_patch=bg,
        input_soft_alpha=alpha,
        input_trimap_unknown=trimap,
        input_boundary_roi_mask=roi,
        input_person_mask=person,
        input_boundary_band=band,
        input_uncertainty=uncertainty,
    )
    model = MatteBridgeModel(in_channels=stacked.shape[-1], width=16)
    x = torch.from_numpy(stacked[None]).permute(0, 3, 1, 2)
    base_alpha = torch.from_numpy(alpha[None, None])
    focus_tensor = torch.from_numpy(focus[None, None])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    target = torch.clamp(base_alpha + 0.25 * torch.sin(x[:, :1]), 0.0, 1.0)

    with torch.no_grad():
        _, _, pred0 = model(x.float(), base_alpha.float(), focus_tensor.float(), gate_strength=1.0)
        initial = torch.mean(torch.abs(pred0 - target)).item()

    for _ in range(20):
        _, _, pred = model(x.float(), base_alpha.float(), focus_tensor.float(), gate_strength=1.0)
        loss = torch.mean(torch.abs(pred - target))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        pred_matte, pred_gate, pred1 = model(x.float(), base_alpha.float(), focus_tensor.float(), gate_strength=1.0)
        final = torch.mean(torch.abs(pred1 - target)).item()

    result = {
        "initial_l1": float(initial),
        "final_l1": float(final),
        "loss_decreased": bool(final < initial),
        "pred_matte_range": [float(pred_matte.min().item()), float(pred_matte.max().item())],
        "pred_gate_range": [float(pred_gate.min().item()), float(pred_gate.max().item())],
        "final_alpha_range": [float(pred1.min().item()), float(pred1.max().item())],
    }
    print(json.dumps(result, ensure_ascii=False))
    if not result["loss_decreased"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
