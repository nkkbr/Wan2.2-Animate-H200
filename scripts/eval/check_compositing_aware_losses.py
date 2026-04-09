#!/usr/bin/env python
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan.utils.edge_losses import (
    compositing_reconstruction_loss,
    contrast_preservation_loss,
    gradient_preservation_loss,
    resolve_loss_stack,
)


def main():
    torch.manual_seed(0)

    h = w = 32
    yy, xx = torch.meshgrid(torch.linspace(-1.0, 1.0, h), torch.linspace(-1.0, 1.0, w), indexing="ij")
    target = torch.clamp(1.0 - torch.sqrt(xx * xx + yy * yy), min=0.0, max=1.0)[None]
    good = torch.clamp(target + 0.02 * torch.sin(xx * 9.0)[None], 0.0, 1.0)
    bad = torch.full_like(target, 0.5)
    fg = torch.stack([0.9 * torch.ones_like(target[0]), 0.7 * torch.ones_like(target[0]), 0.6 * torch.ones_like(target[0])])[None]
    bg = torch.stack([0.1 + 0.1 * xx, 0.15 + 0.1 * yy, 0.2 + 0.05 * xx])[None]
    focus = ((target > 0.05) & (target < 0.95)).float()

    losses = {
        "good_composite": float(compositing_reconstruction_loss(good, target, fg, bg, focus).item()),
        "bad_composite": float(compositing_reconstruction_loss(bad, target, fg, bg, focus).item()),
        "good_gradient": float(gradient_preservation_loss(good, target, focus).item()),
        "bad_gradient": float(gradient_preservation_loss(bad, target, focus).item()),
        "good_contrast": float(contrast_preservation_loss(good, target, fg, bg, focus).item()),
        "bad_contrast": float(contrast_preservation_loss(bad, target, fg, bg, focus).item()),
        "stack": resolve_loss_stack("composite_grad_contrast_v1"),
    }

    assert losses["good_composite"] < losses["bad_composite"]
    assert losses["good_gradient"] < losses["bad_gradient"]
    assert losses["good_contrast"] < losses["bad_contrast"]

    print(json.dumps(losses, ensure_ascii=False))


if __name__ == "__main__":
    main()
