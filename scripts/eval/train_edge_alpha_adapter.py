#!/usr/bin/env python
import argparse
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


def _boundary_f1(pred_mask: np.ndarray, label_mask: np.ndarray) -> float:
    import cv2

    kernel = np.ones((3, 3), np.uint8)
    pred_edge = cv2.morphologyEx(pred_mask.astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0
    label_edge = cv2.morphologyEx(label_mask.astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0
    tolerance_kernel = np.ones((5, 5), np.uint8)
    pred_match_region = cv2.dilate(pred_edge.astype(np.uint8), tolerance_kernel, iterations=1) > 0
    label_match_region = cv2.dilate(label_edge.astype(np.uint8), tolerance_kernel, iterations=1) > 0
    matched_pred = np.logical_and(pred_edge, label_match_region).sum()
    matched_label = np.logical_and(label_edge, pred_match_region).sum()
    precision = float(matched_pred / max(pred_edge.sum(), 1))
    recall = float(matched_label / max(label_edge.sum(), 1))
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def _evaluate_predictions(
    pred_alpha: np.ndarray,
    target_alpha: np.ndarray,
    target_hard: np.ndarray,
    target_trimap_unknown: np.ndarray,
) -> dict:
    thresholds = np.linspace(0.35, 0.75, 17)
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in thresholds:
        f1_values = [
            _boundary_f1((pred_alpha[idx] >= threshold).astype(np.uint8), target_hard[idx].astype(np.uint8))
            for idx in range(len(pred_alpha))
        ]
        mean_f1 = float(np.mean(f1_values))
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_threshold = float(threshold)

    pred_hard = (pred_alpha >= best_threshold).astype(np.uint8)
    trimap_unknown = target_trimap_unknown.astype(np.float32)
    alpha_mae = float(np.abs(pred_alpha - target_alpha).mean())
    trimap_error = float((np.abs(pred_alpha - target_alpha) * trimap_unknown).sum() / max(float(trimap_unknown.sum()), 1.0))
    return {
        "boundary_f1_mean": float(np.mean([
            _boundary_f1(pred_hard[idx], target_hard[idx].astype(np.uint8)) for idx in range(len(pred_hard))
        ])),
        "alpha_mae_mean": alpha_mae,
        "trimap_error_mean": trimap_error,
        "best_threshold": best_threshold,
    }


def _score(metrics: dict) -> float:
    return float(metrics["boundary_f1_mean"] - 3.5 * metrics["alpha_mae_mean"] - 0.5 * metrics["trimap_error_mean"])


def main():
    parser = argparse.ArgumentParser(description="Train a lightweight edge alpha adapter.")
    parser.add_argument("--dataset_npz", type=str, required=True)
    parser.add_argument("--dataset_json", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--residual_scale", type=float, default=0.16)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--boundary_weight", type=float, default=4.0)
    parser.add_argument("--hair_weight", type=float, default=2.0)
    parser.add_argument("--hand_weight", type=float, default=1.5)
    parser.add_argument("--bce_weight", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset = np.load(args.dataset_npz)
    dataset_info = json.loads(Path(args.dataset_json).read_text(encoding="utf-8"))
    feature_keys = dataset_info["feature_keys"]

    features = dataset["features"].astype(np.float32)
    alpha_target = dataset["alpha_target"].astype(np.float32)
    hard_target = dataset["hard_target"].astype(np.float32)
    trimap_unknown = dataset["trimap_unknown_target"].astype(np.float32)
    boundary_target = dataset["boundary_target"].astype(np.float32)
    hair_target = dataset["hair_target"].astype(np.float32)
    hand_target = dataset["hand_target"].astype(np.float32)

    sample_count = features.shape[0]
    indices = np.arange(sample_count)
    val_mask = (indices % 4) == 0
    train_idx = indices[~val_mask]
    val_idx = indices[val_mask]
    if len(val_idx) == 0:
        val_idx = indices[-4:]
        train_idx = indices[:-4]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = EdgeAlphaAdapter(features.shape[1], width=args.width, residual_scale=args.residual_scale).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_features = torch.from_numpy(features[train_idx]).to(device=device, dtype=torch.float32)
    train_alpha = torch.from_numpy(alpha_target[train_idx]).to(device=device, dtype=torch.float32)
    train_hard = torch.from_numpy(hard_target[train_idx]).to(device=device, dtype=torch.float32)
    train_boundary = torch.from_numpy(boundary_target[train_idx]).to(device=device, dtype=torch.float32)
    train_trimap = torch.from_numpy(trimap_unknown[train_idx]).to(device=device, dtype=torch.float32)
    train_hair = torch.from_numpy(hair_target[train_idx]).to(device=device, dtype=torch.float32)
    train_hand = torch.from_numpy(hand_target[train_idx]).to(device=device, dtype=torch.float32)

    best = {
        "score": None,
        "state": None,
        "metrics": None,
        "threshold": 0.5,
    }
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(train_features.shape[0], device=device)
        epoch_loss = 0.0
        for start in range(0, len(perm), args.batch_size):
            batch_idx = perm[start:start + args.batch_size]
            x = train_features[batch_idx]
            y_alpha = train_alpha[batch_idx]
            y_hard = train_hard[batch_idx]
            boundary = train_boundary[batch_idx]
            trimap = train_trimap[batch_idx]
            hair = train_hair[batch_idx]
            hand = train_hand[batch_idx]

            pred = model(x)[:, 0]
            weight = 1.0 + args.boundary_weight * boundary + 1.5 * trimap + args.hair_weight * hair + args.hand_weight * hand
            loss_alpha = ((pred - y_alpha).abs() * weight).mean()
            loss_bce = F.binary_cross_entropy(pred, y_hard, weight=1.0 + 1.5 * boundary)
            loss = loss_alpha + float(args.bce_weight) * loss_bce
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                val_pred = model(torch.from_numpy(features[val_idx]).to(device=device, dtype=torch.float32))[:, 0].detach().cpu().numpy()
            metrics = _evaluate_predictions(
                val_pred,
                alpha_target[val_idx],
                hard_target[val_idx],
                trimap_unknown[val_idx],
            )
            score = _score(metrics)
            history.append({
                "epoch": epoch,
                "train_loss": epoch_loss / max(int(np.ceil(len(perm) / args.batch_size)), 1),
                "val_score": score,
                **metrics,
            })
            if best["score"] is None or score > best["score"]:
                best["score"] = score
                best["state"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                best["metrics"] = metrics
                best["threshold"] = metrics["best_threshold"]

    checkpoint = {
        "model_state": best["state"],
        "feature_keys": feature_keys,
        "width": int(args.width),
        "residual_scale": float(args.residual_scale),
        "threshold": float(best["threshold"]),
        "best_val_metrics": best["metrics"],
    }
    ckpt_path = output_dir / "edge_alpha_adapter.pt"
    torch.save(checkpoint, ckpt_path)

    model.load_state_dict(best["state"])
    model.eval()
    with torch.no_grad():
        full_pred = model(torch.from_numpy(features).to(device=device, dtype=torch.float32))[:, 0].detach().cpu().numpy()
    full_metrics = _evaluate_predictions(full_pred, alpha_target, hard_target, trimap_unknown)

    summary = {
        "dataset_npz": str(Path(args.dataset_npz).resolve()),
        "dataset_json": str(Path(args.dataset_json).resolve()),
        "checkpoint_path": str(ckpt_path.resolve()),
        "feature_keys": feature_keys,
        "width": int(args.width),
        "residual_scale": float(args.residual_scale),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "best_val_score": float(best["score"]),
        "best_val_metrics": best["metrics"],
        "full_dataset_metrics": full_metrics,
        "history": history,
    }
    summary_path = output_dir / "train_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "checkpoint_path": str(ckpt_path.resolve()),
        "full_dataset_metrics": full_metrics,
        "best_val_metrics": best["metrics"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
