#!/usr/bin/env python
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan.utils.edge_alpha_adapter import EdgeAlphaAdapter


def main():
    torch.manual_seed(123)
    np.random.seed(123)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EdgeAlphaAdapter(in_channels=10, width=12, residual_scale=0.18).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

    height, width = 96, 128
    features = np.zeros((8, 10, height, width), dtype=np.float32)
    targets = np.zeros((8, height, width), dtype=np.float32)
    yy, xx = np.mgrid[0:height, 0:width]
    for idx in range(8):
        cx = 36 + idx * 4
        cy = 48
        radius = 20 + (idx % 3)
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        alpha = np.clip(1.0 - (dist - radius) / 6.0, 0.0, 1.0).astype(np.float32)
        base_alpha = np.clip(alpha + 0.10 * np.sin(xx / 9.0) - 0.08 * np.cos(yy / 11.0), 0.0, 1.0).astype(np.float32)
        hard = (base_alpha > 0.5).astype(np.float32)
        band = np.clip(np.abs(base_alpha - hard) * 3.0, 0.0, 1.0)
        features[idx, 0] = hard
        features[idx, 1] = base_alpha
        features[idx, 2] = band
        features[idx, 3] = 0.0
        features[idx, 4] = 0.2 * band
        features[idx, 5] = np.clip(hard + 0.5 * band, 0.0, 1.0)
        features[idx, 6] = 1.0 - features[idx, 5]
        features[idx, 7] = band
        features[idx, 8] = 0.0
        features[idx, 9] = 0.0
        targets[idx] = alpha

    x = torch.from_numpy(features).to(device=device, dtype=torch.float32)
    y = torch.from_numpy(targets).to(device=device, dtype=torch.float32)
    initial = None
    final = None
    for step in range(40):
        pred = model(x)[:, 0]
        loss = F.l1_loss(pred, y) + 0.2 * F.mse_loss(pred, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 0:
            initial = float(loss.detach().cpu())
        final = float(loss.detach().cpu())

    payload = {
        "status": "PASS" if final < initial * 0.65 else "FAIL",
        "initial_loss": initial,
        "final_loss": final,
        "loss_ratio": final / max(initial, 1e-6),
    }
    print(json.dumps(payload, ensure_ascii=False))
    raise SystemExit(0 if payload["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
