#!/usr/bin/env python
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan.utils.trainable_alpha_model import TrainableAlphaModel, stack_input_channels


def main():
    rng = np.random.default_rng(1234)
    fg = rng.integers(0, 255, size=(8, 96, 96, 3), dtype=np.uint8)
    bg = rng.integers(0, 255, size=(8, 96, 96, 3), dtype=np.uint8)
    alpha = rng.random((8, 96, 96), dtype=np.float32)
    trimap = ((alpha > 0.15) & (alpha < 0.85)).astype(np.float32)
    boundary = trimap.copy()
    person = (alpha > 0.5).astype(np.float32)
    uncertainty = np.clip(np.abs(alpha - 0.5) * 1.5, 0.0, 1.0).astype(np.float32)
    gt = np.clip(alpha * 0.92 + 0.04, 0.0, 1.0).astype(np.float32)

    stacked = np.stack([
        stack_input_channels(
            foreground_patch=fg[i],
            background_patch=bg[i],
            input_soft_alpha=alpha[i],
            input_trimap_unknown=trimap[i],
            input_boundary_roi_mask=boundary[i],
            input_person_mask=person[i],
            input_boundary_band=boundary[i],
            input_uncertainty=uncertainty[i],
        )
        for i in range(len(alpha))
    ])
    x = torch.from_numpy(stacked).permute(0, 3, 1, 2).float()
    base_alpha = torch.from_numpy(alpha).float()[:, None]
    target = torch.from_numpy(gt).float()

    model = TrainableAlphaModel(in_channels=x.shape[1], width=16, residual_scale=0.20)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    with torch.no_grad():
        pred0 = model(x, base_alpha)[:, 0]
        loss0 = torch.abs(pred0 - target).mean().item()

    for _ in range(20):
        pred = model(x, base_alpha)[:, 0]
        loss = torch.abs(pred - target).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    with torch.no_grad():
        pred1 = model(x, base_alpha)[:, 0]
        loss1 = torch.abs(pred1 - target).mean().item()

    result = {
        "input_shape": list(x.shape),
        "initial_l1": float(loss0),
        "final_l1": float(loss1),
        "loss_decreased": bool(loss1 < loss0),
        "pred_range": [float(pred1.min().item()), float(pred1.max().item())],
    }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
