#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan.utils.edge_alpha_adapter import (
    create_edge_alpha_adapter,
    derive_adapter_artifacts,
    load_adapter_feature_maps,
    predict_edge_alpha,
    stack_feature_maps,
    write_adapter_bundle,
)


def main():
    parser = argparse.ArgumentParser(description="Apply a trained edge alpha adapter to a preprocess bundle.")
    parser.add_argument("--source_preprocess_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_preprocess_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    feature_keys = checkpoint.get("feature_keys")
    feature_maps, _ = load_adapter_feature_maps(args.source_preprocess_dir, feature_keys)
    features = stack_feature_maps(feature_maps, feature_keys)

    model = create_edge_alpha_adapter(checkpoint)
    device = args.device if torch.cuda.is_available() else "cpu"
    model.to(device)
    pred_alpha = predict_edge_alpha(model=model, features=features, device=device, batch_size=args.batch_size)
    artifacts = derive_adapter_artifacts(
        pred_alpha=pred_alpha,
        feature_maps=feature_maps,
        threshold=float(checkpoint.get("threshold", 0.5)),
    )
    output_dir = write_adapter_bundle(
        source_preprocess_dir=args.source_preprocess_dir,
        output_preprocess_dir=args.output_preprocess_dir,
        artifacts=artifacts,
        checkpoint_info={
            "checkpoint_path": str(checkpoint_path.resolve()),
            "threshold": float(checkpoint.get("threshold", 0.5)),
            "feature_keys": list(feature_keys),
        },
    )
    print(json.dumps({
        "output_preprocess_dir": str(Path(output_dir).resolve()),
        "threshold": float(checkpoint.get("threshold", 0.5)),
        "frame_count": int(pred_alpha.shape[0]),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
