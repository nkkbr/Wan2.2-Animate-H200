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

from wan.utils.trainable_alpha_model import TrainableAlphaModel, stack_input_channels
from wan.utils.edge_losses import (
    compositing_reconstruction_loss,
    contrast_preservation_loss,
    gradient_preservation_loss,
    resolve_loss_stack,
)


def _boundary_f1(pred_mask: np.ndarray, label_mask: np.ndarray) -> float:
    import cv2

    kernel = np.ones((3, 3), np.uint8)
    pred_edge = cv2.morphologyEx(pred_mask.astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0
    label_edge = cv2.morphologyEx(label_mask.astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0
    tol = np.ones((5, 5), np.uint8)
    pred_match = cv2.dilate(pred_edge.astype(np.uint8), tol, iterations=1) > 0
    label_match = cv2.dilate(label_edge.astype(np.uint8), tol, iterations=1) > 0
    matched_pred = np.logical_and(pred_edge, label_match).sum()
    matched_label = np.logical_and(label_edge, pred_match).sum()
    precision = float(matched_pred / max(pred_edge.sum(), 1))
    recall = float(matched_label / max(label_edge.sum(), 1))
    if precision + recall == 0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def _evaluate(pred_alpha: np.ndarray, gt_alpha: np.ndarray, gt_boundary: np.ndarray, gt_trimap: np.ndarray):
    thresholds = np.linspace(0.35, 0.75, 17)
    best_threshold = 0.5
    best_boundary = -1.0
    for threshold in thresholds:
        boundary_scores = [
            _boundary_f1((pred_alpha[idx] >= threshold).astype(np.uint8), (gt_alpha[idx] >= 0.5).astype(np.uint8))
            for idx in range(len(pred_alpha))
        ]
        score = float(np.mean(boundary_scores))
        if score > best_boundary:
            best_boundary = score
            best_threshold = float(threshold)
    pred_mask = (pred_alpha >= best_threshold).astype(np.uint8)
    boundary_f1 = float(np.mean([
        _boundary_f1(pred_mask[idx], (gt_alpha[idx] >= 0.5).astype(np.uint8))
        for idx in range(len(pred_alpha))
    ]))
    alpha_mae = float(np.abs(pred_alpha - gt_alpha).mean())
    trimap_error = float((np.abs(pred_alpha - gt_alpha) * gt_trimap).sum() / max(float(gt_trimap.sum()), 1.0))
    alpha_sad = float(np.abs(pred_alpha - gt_alpha).sum())
    boundary_roi_f1 = float(np.mean([
        _boundary_f1(
            np.logical_and(pred_mask[idx] > 0, gt_boundary[idx] > 0).astype(np.uint8),
            np.logical_and((gt_alpha[idx] >= 0.5), gt_boundary[idx] > 0).astype(np.uint8),
        )
        for idx in range(len(pred_alpha))
    ]))
    return {
        "boundary_f1_mean": boundary_f1,
        "boundary_roi_f1_mean": boundary_roi_f1,
        "alpha_mae_mean": alpha_mae,
        "trimap_error_mean": trimap_error,
        "alpha_sad_mean": alpha_sad,
        "best_threshold": best_threshold,
    }


def _score(metrics: dict) -> float:
    return float(
        1.00 * metrics["boundary_f1_mean"]
        + 0.50 * metrics["boundary_roi_f1_mean"]
        - 6.00 * metrics["alpha_mae_mean"]
        - 2.50 * metrics["trimap_error_mean"]
    )


def _load_dataset(dataset_npz: Path, dataset_json: Path):
    arrays = np.load(dataset_npz)
    info = json.loads(dataset_json.read_text(encoding="utf-8"))
    records = info["records"]
    return arrays, records, info


def _build_inputs(arrays: np.lib.npyio.NpzFile) -> np.ndarray:
    sample_count = arrays["foreground_patch"].shape[0]
    return np.stack([
        stack_input_channels(
            foreground_patch=arrays["foreground_patch"][i],
            background_patch=arrays["background_patch"][i],
            input_soft_alpha=arrays["input_soft_alpha"][i].astype(np.float32),
            input_trimap_unknown=arrays["input_trimap_unknown"][i].astype(np.float32),
            input_boundary_roi_mask=arrays["input_boundary_roi_mask"][i].astype(np.float32),
            input_person_mask=arrays["input_person_mask"][i].astype(np.float32),
            input_boundary_band=arrays["input_boundary_band"][i].astype(np.float32),
            input_uncertainty=arrays["input_uncertainty"][i].astype(np.float32),
        )
        for i in range(sample_count)
    ]).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Train trainable alpha model on roi_dataset_v2.")
    parser.add_argument("--dataset_npz", required=True)
    parser.add_argument("--dataset_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--residual_scale", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--boundary_weight", type=float, default=4.0)
    parser.add_argument("--trimap_weight", type=float, default=2.0)
    parser.add_argument("--semi_weight", type=float, default=2.0)
    parser.add_argument("--hard_negative_weight", type=float, default=1.5)
    parser.add_argument("--bce_weight", type=float, default=0.10)
    parser.add_argument("--unknown_weight", type=float, default=0.75)
    parser.add_argument("--preserve_weight", type=float, default=0.50)
    parser.add_argument("--loss_stack", choices=("pixel_v1", "composite_v1", "composite_grad_v1", "composite_grad_contrast_v1"), default="pixel_v1")
    parser.add_argument("--composite_weight", type=float, default=-1.0)
    parser.add_argument("--gradient_weight", type=float, default=-1.0)
    parser.add_argument("--contrast_weight", type=float, default=-1.0)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    arrays, records, info = _load_dataset(Path(args.dataset_npz), Path(args.dataset_json))
    inputs = _build_inputs(arrays)
    gt_alpha = arrays["gt_soft_alpha"].astype(np.float32)
    gt_boundary = arrays["gt_boundary_mask"].astype(np.float32)
    gt_trimap = arrays["gt_trimap_unknown"].astype(np.float32)
    gt_semi = arrays["gt_semi_transparent"].astype(np.float32)
    hard_mask = (gt_alpha >= 0.5).astype(np.float32)
    is_hard_negative = np.asarray([float(record["is_hard_negative"]) for record in records], dtype=np.float32)
    dataset_split = np.asarray([record["dataset_split"] for record in records])
    difficulty = arrays["difficulty_score"].astype(np.float32)

    train_idx = np.where(dataset_split == "train")[0]
    val_idx = np.where(dataset_split == "val")[0]
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise RuntimeError("Dataset must contain train and val splits.")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = TrainableAlphaModel(in_channels=inputs.shape[-1], width=args.width, residual_scale=args.residual_scale).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_x = torch.from_numpy(inputs[train_idx]).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)
    train_alpha = torch.from_numpy(gt_alpha[train_idx]).to(device=device, dtype=torch.float32)
    train_boundary = torch.from_numpy(gt_boundary[train_idx]).to(device=device, dtype=torch.float32)
    train_trimap = torch.from_numpy(gt_trimap[train_idx]).to(device=device, dtype=torch.float32)
    train_semi = torch.from_numpy(gt_semi[train_idx]).to(device=device, dtype=torch.float32)
    train_hard = torch.from_numpy(hard_mask[train_idx]).to(device=device, dtype=torch.float32)
    train_is_hn = torch.from_numpy(is_hard_negative[train_idx]).to(device=device, dtype=torch.float32)
    train_base_alpha = torch.from_numpy(inputs[train_idx, :, :, 6]).to(device=device, dtype=torch.float32)
    train_difficulty = torch.from_numpy(difficulty[train_idx]).to(device=device, dtype=torch.float32)
    train_fg = torch.from_numpy(arrays["foreground_patch"][train_idx].astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)
    train_bg = torch.from_numpy(arrays["background_patch"][train_idx].astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)

    val_x = torch.from_numpy(inputs[val_idx]).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)
    val_base_alpha = torch.from_numpy(inputs[val_idx, :, :, 6]).to(device=device, dtype=torch.float32)

    loss_stack = resolve_loss_stack(
        args.loss_stack,
        composite_weight=None if args.composite_weight < 0 else args.composite_weight,
        gradient_weight=None if args.gradient_weight < 0 else args.gradient_weight,
        contrast_weight=None if args.contrast_weight < 0 else args.contrast_weight,
    )

    best = {"score": None, "state": None, "metrics": None, "epoch": None}
    history = []
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(train_idx), device=device)
        batch_losses = []
        for start in range(0, len(perm), args.batch_size):
            batch_idx = perm[start:start + args.batch_size]
            x = train_x[batch_idx]
            y_alpha = train_alpha[batch_idx]
            y_boundary = train_boundary[batch_idx]
            y_trimap = train_trimap[batch_idx]
            y_semi = train_semi[batch_idx]
            y_hard = train_hard[batch_idx]
            y_hn = train_is_hn[batch_idx]
            y_base_alpha = train_base_alpha[batch_idx]
            y_difficulty = train_difficulty[batch_idx]
            y_fg = train_fg[batch_idx]
            y_bg = train_bg[batch_idx]

            pred = model(x, y_base_alpha[:, None])[:, 0]
            weight = (
                1.0
                + float(args.boundary_weight) * y_boundary
                + float(args.trimap_weight) * y_trimap
                + float(args.semi_weight) * y_semi
                + float(args.hard_negative_weight) * y_hn[:, None, None]
                + 0.35 * y_difficulty[:, None, None]
            )
            loss_alpha = ((pred - y_alpha).abs() * weight).mean()
            unknown_focus = torch.clamp(y_trimap + 0.5 * y_semi + 0.5 * y_boundary, min=0.0, max=1.0)
            loss_unknown = ((pred - y_alpha).abs() * (1.0 + 3.0 * unknown_focus)).mean()
            preserve_mask = 1.0 - torch.clamp(y_trimap + y_boundary + y_semi, min=0.0, max=1.0)
            loss_preserve = ((pred - y_base_alpha).abs() * preserve_mask).mean()
            loss_bce = F.binary_cross_entropy(pred, y_hard, weight=1.0 + 1.5 * y_boundary + 0.5 * y_trimap)
            comp_focus = torch.clamp(y_boundary + y_trimap + y_semi, min=0.0, max=1.0)
            loss_composite = compositing_reconstruction_loss(pred, y_alpha, y_fg, y_bg, comp_focus)
            loss_gradient = gradient_preservation_loss(pred, y_alpha, comp_focus)
            loss_contrast = contrast_preservation_loss(pred, y_alpha, y_fg, y_bg, comp_focus)
            loss = (
                loss_alpha
                + float(args.unknown_weight) * loss_unknown
                + float(args.preserve_weight) * loss_preserve
                + float(args.bce_weight) * loss_bce
                + float(loss_stack["composite_weight"]) * loss_composite
                + float(loss_stack["gradient_weight"]) * loss_gradient
                + float(loss_stack["contrast_weight"]) * loss_contrast
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            val_pred = model(val_x, val_base_alpha[:, None])[:, 0].detach().cpu().numpy()
        val_metrics = _evaluate(val_pred, gt_alpha[val_idx], gt_boundary[val_idx], gt_trimap[val_idx])
        score = _score(val_metrics)
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(batch_losses)) if batch_losses else None,
            "val_score": score,
            **val_metrics,
        })
        if best["score"] is None or score > best["score"]:
            best = {
                "score": score,
                "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "metrics": val_metrics,
                "epoch": epoch,
            }
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= int(args.patience):
                break

    if best["state"] is None:
        raise RuntimeError("Training did not produce a best checkpoint.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": best["state"],
        "in_channels": int(inputs.shape[-1]),
        "width": int(args.width),
        "residual_scale": float(args.residual_scale),
        "best_val_metrics": best["metrics"],
        "best_epoch": int(best["epoch"]),
        "feature_spec": [
            "foreground_rgb",
            "background_rgb",
            "input_soft_alpha",
            "input_trimap_unknown",
            "input_boundary_roi_mask",
            "input_person_mask",
            "input_boundary_band",
            "input_uncertainty",
        ],
    }
    ckpt_path = output_dir / "trainable_alpha_model.pt"
    torch.save(checkpoint, ckpt_path)

    summary = {
        "dataset_npz": str(Path(args.dataset_npz).resolve()),
        "dataset_json": str(Path(args.dataset_json).resolve()),
        "checkpoint_path": str(ckpt_path.resolve()),
        "best_epoch": int(best["epoch"]),
        "best_val_score": float(best["score"]),
        "best_val_metrics": best["metrics"],
        "history": history,
        "config": {
            "width": int(args.width),
            "residual_scale": float(args.residual_scale),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "boundary_weight": float(args.boundary_weight),
            "trimap_weight": float(args.trimap_weight),
            "semi_weight": float(args.semi_weight),
            "hard_negative_weight": float(args.hard_negative_weight),
            "bce_weight": float(args.bce_weight),
            "unknown_weight": float(args.unknown_weight),
            "preserve_weight": float(args.preserve_weight),
            "loss_stack": str(args.loss_stack),
            "composite_weight": float(loss_stack["composite_weight"]),
            "gradient_weight": float(loss_stack["gradient_weight"]),
            "contrast_weight": float(loss_stack["contrast_weight"]),
            "patience": int(args.patience),
        },
    }
    (output_dir / "train_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "checkpoint_path": str(ckpt_path.resolve()),
        "best_val_metrics": best["metrics"],
        "best_epoch": best["epoch"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
