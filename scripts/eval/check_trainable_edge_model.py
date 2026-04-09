#!/usr/bin/env python
import json
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan.utils.trainable_edge_model import EdgeRefinementNet, split_outputs


def main():
    model = EdgeRefinementNet()
    x = torch.randn(2, 5, 64, 64)
    logits = model(x)
    outputs = split_outputs(logits)
    assert outputs.alpha.shape == (2, 1, 64, 64)
    assert outputs.boundary.shape == (2, 1, 64, 64)
    print(json.dumps({"status": "PASS", "alpha_mean": float(outputs.alpha.mean()), "boundary_mean": float(outputs.boundary.mean())}))


if __name__ == "__main__":
    main()
