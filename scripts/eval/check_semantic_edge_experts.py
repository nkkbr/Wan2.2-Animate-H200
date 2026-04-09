#!/usr/bin/env python
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan.utils.semantic_edge_experts import SemanticEdgeExpertsModel


def main():
    torch.manual_seed(1234)
    np.random.seed(1234)

    model = SemanticEdgeExpertsModel(in_channels=12, semantic_tags=("hair", "face"), width=16, residual_scale=0.20)
    x = torch.rand(6, 12, 64, 64)
    base_alpha = torch.rand(6, 1, 64, 64)
    semantic_ids = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long)
    target = torch.clamp(base_alpha[:, 0] + 0.15 * torch.sin(x[:, 0] * 2.0), 0.0, 1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    with torch.no_grad():
        pred0 = model(x, base_alpha, semantic_ids)[:, 0]
        initial = torch.mean(torch.abs(pred0 - target)).item()

    for _ in range(20):
        pred = model(x, base_alpha, semantic_ids)[:, 0]
        loss = torch.mean(torch.abs(pred - target))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        pred1 = model(x, base_alpha, semantic_ids)[:, 0]
        final = torch.mean(torch.abs(pred1 - target)).item()

    result = {
        "initial_l1": float(initial),
        "final_l1": float(final),
        "loss_decreased": bool(final < initial),
        "pred_range": [float(pred1.min().item()), float(pred1.max().item())],
        "semantic_tags": list(model.semantic_tags),
    }
    print(json.dumps(result, ensure_ascii=False))
    if not result["loss_decreased"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
