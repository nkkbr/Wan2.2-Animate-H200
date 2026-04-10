#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan.utils.edge_losses import (
    compositing_reconstruction_loss,
    contrast_preservation_loss,
    gradient_preservation_loss,
)
from wan.utils.matte_bridge_model import (
    MatteBridgeModel,
    build_bridge_focus_map,
    stack_bridge_input_channels,
)


def _boundary_f1(pred_mask: np.ndarray, label_mask: np.ndarray) -> float:
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


def _masked_boundary_f1(pred_mask: np.ndarray, gt_mask: np.ndarray, category_mask: np.ndarray) -> float | None:
    category = category_mask > 0
    if int(category.sum()) == 0:
        return None
    pred = np.logical_and(pred_mask > 0, category).astype(np.uint8)
    gt = np.logical_and(gt_mask > 0, category).astype(np.uint8)
    return _boundary_f1(pred, gt)


def _evaluate(
    pred_alpha: np.ndarray,
    gt_alpha: np.ndarray,
    gt_boundary: np.ndarray,
    gt_trimap: np.ndarray,
    gt_hair: np.ndarray,
    gt_semi: np.ndarray,
) -> dict[str, float]:
    thresholds = np.linspace(0.35, 0.75, 17)
    best_threshold = 0.5
    best_boundary = -1.0
    gt_mask = (gt_alpha >= 0.5).astype(np.uint8)
    for threshold in thresholds:
        scores = [
            _boundary_f1((pred_alpha[idx] >= threshold).astype(np.uint8), gt_mask[idx])
            for idx in range(len(pred_alpha))
        ]
        score = float(np.mean(scores))
        if score > best_boundary:
            best_boundary = score
            best_threshold = float(threshold)
    pred_mask = (pred_alpha >= best_threshold).astype(np.uint8)
    boundary_f1 = float(np.mean([_boundary_f1(pred_mask[idx], gt_mask[idx]) for idx in range(len(pred_alpha))]))
    alpha_mae = float(np.abs(pred_alpha - gt_alpha).mean())
    trimap_error = float((np.abs(pred_alpha - gt_alpha) * gt_trimap).sum() / max(float(gt_trimap.sum()), 1.0))
    hair_scores = [
        _masked_boundary_f1(pred_mask[idx], gt_mask[idx], gt_hair[idx])
        for idx in range(len(pred_alpha))
    ]
    hair_scores = [score for score in hair_scores if score is not None]
    semi_scores = [
        _masked_boundary_f1(pred_mask[idx], gt_mask[idx], gt_semi[idx])
        for idx in range(len(pred_alpha))
    ]
    semi_scores = [score for score in semi_scores if score is not None]
    return {
        "boundary_f1_mean": boundary_f1,
        "alpha_mae_mean": alpha_mae,
        "trimap_error_mean": trimap_error,
        "hair_boundary_f1_mean": float(np.mean(hair_scores)) if hair_scores else 0.0,
        "semi_transparent_boundary_f1_mean": float(np.mean(semi_scores)) if semi_scores else 0.0,
        "best_threshold": best_threshold,
    }


def _score(metrics: dict[str, float]) -> float:
    return float(
        1.00 * metrics["boundary_f1_mean"]
        + 0.60 * metrics["hair_boundary_f1_mean"]
        + 0.40 * metrics["semi_transparent_boundary_f1_mean"]
        - 6.00 * metrics["alpha_mae_mean"]
        - 2.50 * metrics["trimap_error_mean"]
    )


def _build_inputs(arrays: np.lib.npyio.NpzFile, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    inputs = []
    focus_maps = []
    for idx in indices.tolist():
        inputs.append(
            stack_bridge_input_channels(
                foreground_patch=arrays["foreground_patch"][idx],
                background_patch=arrays["background_patch"][idx],
                input_soft_alpha=arrays["input_soft_alpha"][idx].astype(np.float32),
                input_trimap_unknown=arrays["input_trimap_unknown"][idx].astype(np.float32),
                input_boundary_roi_mask=arrays["input_boundary_roi_mask"][idx].astype(np.float32),
                input_person_mask=arrays["input_person_mask"][idx].astype(np.float32),
                input_boundary_band=arrays["input_boundary_band"][idx].astype(np.float32),
                input_uncertainty=arrays["input_uncertainty"][idx].astype(np.float32),
            )
        )
        focus_maps.append(
            build_bridge_focus_map(
                input_trimap_unknown=arrays["input_trimap_unknown"][idx].astype(np.float32),
                input_boundary_roi_mask=arrays["input_boundary_roi_mask"][idx].astype(np.float32),
                input_boundary_band=arrays["input_boundary_band"][idx].astype(np.float32),
                input_uncertainty=arrays["input_uncertainty"][idx].astype(np.float32),
            )
        )
    return np.stack(inputs).astype(np.float32), np.stack(focus_maps).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train matte bridge model on roi_dataset_v2.")
    parser.add_argument("--dataset_npz", required=True)
    parser.add_argument("--dataset_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--gate_strength", type=float, default=1.0)
    parser.add_argument("--bridge_error_threshold", type=float, default=0.04)
    parser.add_argument("--final_weight", type=float, default=1.0)
    parser.add_argument("--pred_weight", type=float, default=0.35)
    parser.add_argument("--gate_weight", type=float, default=0.20)
    parser.add_argument("--composite_weight", type=float, default=0.0)
    parser.add_argument("--gradient_weight", type=float, default=0.0)
    parser.add_argument("--contrast_weight", type=float, default=0.0)
    parser.add_argument("--boundary_weight", type=float, default=4.0)
    parser.add_argument("--trimap_weight", type=float, default=2.0)
    parser.add_argument("--semi_weight", type=float, default=2.0)
    parser.add_argument("--hair_weight", type=float, default=2.5)
    parser.add_argument("--hard_negative_weight", type=float, default=1.5)
    parser.add_argument("--gate_target_mode", choices=("binary", "continuous"), default="binary")
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_val_samples", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset_npz = Path(args.dataset_npz)
    dataset_json = Path(args.dataset_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays = np.load(dataset_npz)
    info = json.loads(dataset_json.read_text(encoding="utf-8"))
    records = info["records"]
    gt_alpha = arrays["gt_soft_alpha"].astype(np.float32)
    gt_boundary = arrays["gt_boundary_mask"].astype(np.float32)
    gt_trimap = arrays["gt_trimap_unknown"].astype(np.float32)
    gt_semi = arrays["gt_semi_transparent"].astype(np.float32)
    gt_hair = arrays["gt_hair_edge"].astype(np.float32)
    hard_mask = (gt_alpha >= 0.5).astype(np.float32)
    base_alpha = arrays["input_soft_alpha"].astype(np.float32)
    split = np.asarray([record["dataset_split"] for record in records])
    is_hard_negative = np.asarray([float(record["is_hard_negative"]) for record in records], dtype=np.float32)
    difficulty = arrays["difficulty_score"].astype(np.float32)

    train_idx = np.where(split == "train")[0]
    val_idx = np.where(split == "val")[0]
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise RuntimeError("Dataset must contain train and val splits.")
    if int(args.max_train_samples) > 0:
        train_idx = train_idx[: int(args.max_train_samples)]
    if int(args.max_val_samples) > 0:
        val_idx = val_idx[: int(args.max_val_samples)]

    train_inputs, train_focus_maps = _build_inputs(arrays, train_idx)
    val_inputs, val_focus_maps = _build_inputs(arrays, val_idx)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = MatteBridgeModel(in_channels=train_inputs.shape[-1], width=args.width).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_x = torch.from_numpy(train_inputs).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)
    train_focus = torch.from_numpy(train_focus_maps).to(device=device, dtype=torch.float32)
    train_base_alpha = torch.from_numpy(base_alpha[train_idx]).to(device=device, dtype=torch.float32)
    train_gt_alpha = torch.from_numpy(gt_alpha[train_idx]).to(device=device, dtype=torch.float32)
    train_gt_boundary = torch.from_numpy(gt_boundary[train_idx]).to(device=device, dtype=torch.float32)
    train_gt_trimap = torch.from_numpy(gt_trimap[train_idx]).to(device=device, dtype=torch.float32)
    train_gt_semi = torch.from_numpy(gt_semi[train_idx]).to(device=device, dtype=torch.float32)
    train_gt_hair = torch.from_numpy(gt_hair[train_idx]).to(device=device, dtype=torch.float32)
    train_hard_mask = torch.from_numpy(hard_mask[train_idx]).to(device=device, dtype=torch.float32)
    train_hn = torch.from_numpy(is_hard_negative[train_idx]).to(device=device, dtype=torch.float32)
    train_difficulty = torch.from_numpy(difficulty[train_idx]).to(device=device, dtype=torch.float32)
    train_fg = torch.from_numpy(arrays["foreground_patch"][train_idx].astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)
    train_bg = torch.from_numpy(arrays["background_patch"][train_idx].astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)

    val_x = torch.from_numpy(val_inputs).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)
    val_focus = torch.from_numpy(val_focus_maps).to(device=device, dtype=torch.float32)
    val_base_alpha = torch.from_numpy(base_alpha[val_idx]).to(device=device, dtype=torch.float32)

    best: dict[str, object] = {"score": None, "state": None, "metrics": None, "epoch": None}
    history = []
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(train_idx), device=device)
        batch_losses = []
        for start in range(0, len(perm), args.batch_size):
            batch_idx = perm[start:start + args.batch_size]
            x = train_x[batch_idx]
            focus = train_focus[batch_idx]
            y_base_alpha = train_base_alpha[batch_idx]
            y_alpha = train_gt_alpha[batch_idx]
            y_boundary = train_gt_boundary[batch_idx]
            y_trimap = train_gt_trimap[batch_idx]
            y_semi = train_gt_semi[batch_idx]
            y_hair = train_gt_hair[batch_idx]
            y_hard = train_hard_mask[batch_idx]
            y_hn = train_hn[batch_idx]
            y_difficulty = train_difficulty[batch_idx]
            y_fg = train_fg[batch_idx]
            y_bg = train_bg[batch_idx]

            pred_matte, pred_gate, final_alpha = model(
                x,
                y_base_alpha[:, None],
                focus[:, None],
                gate_strength=args.gate_strength,
            )
            pred_matte = pred_matte[:, 0]
            pred_gate = pred_gate[:, 0]
            final_alpha = final_alpha[:, 0]
            bridge_error = torch.abs(y_base_alpha - y_alpha)
            if args.gate_target_mode == "continuous":
                error_signal = torch.clamp(bridge_error / max(float(args.bridge_error_threshold), 1e-4), min=0.0, max=1.0)
            else:
                error_signal = (bridge_error > float(args.bridge_error_threshold)).to(torch.float32)
            target_gate = error_signal * torch.clamp(
                focus + 0.5 * y_boundary + 0.5 * y_semi + 0.75 * y_hair,
                min=0.0,
                max=1.0,
            )

            weight = (
                1.0
                + float(args.boundary_weight) * y_boundary
                + float(args.trimap_weight) * y_trimap
                + float(args.semi_weight) * y_semi
                + float(args.hair_weight) * y_hair
                + float(args.hard_negative_weight) * y_hn[:, None, None]
                + 0.35 * y_difficulty[:, None, None]
            )
            loss_final = ((final_alpha - y_alpha).abs() * weight).sum() / torch.clamp(weight.sum(), min=1.0)
            matte_weight = 1.0 + 2.0 * focus + 1.0 * y_boundary + 0.5 * y_semi + 1.25 * y_hair
            loss_pred = ((pred_matte - y_alpha).abs() * matte_weight).sum() / torch.clamp(matte_weight.sum(), min=1.0)
            gate_weight = 1.0 + 2.0 * focus + 1.5 * y_hair
            loss_gate = F.binary_cross_entropy(pred_gate, target_gate, weight=gate_weight)
            loss_composite = compositing_reconstruction_loss(final_alpha, y_alpha, y_fg, y_bg, focus)
            loss_gradient = gradient_preservation_loss(final_alpha, y_alpha, focus)
            loss_contrast = contrast_preservation_loss(final_alpha, y_alpha, y_fg, y_bg, focus)
            loss = (
                float(args.final_weight) * loss_final
                + float(args.pred_weight) * loss_pred
                + float(args.gate_weight) * loss_gate
                + float(args.composite_weight) * loss_composite
                + float(args.gradient_weight) * loss_gradient
                + float(args.contrast_weight) * loss_contrast
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            _, _, val_pred = model(
                val_x,
                val_base_alpha[:, None],
                val_focus[:, None],
                gate_strength=args.gate_strength,
            )
        val_metrics = _evaluate(
            val_pred[:, 0].detach().cpu().numpy(),
            gt_alpha[val_idx],
            gt_boundary[val_idx],
            gt_trimap[val_idx],
            gt_hair[val_idx],
            gt_semi[val_idx],
        )
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
                "state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "metrics": val_metrics,
                "epoch": epoch,
            }
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= int(args.patience):
                break

    checkpoint = {
        "in_channels": int(train_inputs.shape[-1]),
        "width": int(args.width),
        "gate_strength": float(args.gate_strength),
        "bridge_error_threshold": float(args.bridge_error_threshold),
        "model_state": best["state"],
        "best_epoch": best["epoch"],
        "best_val_metrics": best["metrics"],
        "history": history,
    }
    ckpt_path = output_dir / "matte_bridge_model.pt"
    torch.save(checkpoint, ckpt_path)

    summary = {
        "dataset_npz": str(dataset_npz.resolve()),
        "dataset_json": str(dataset_json.resolve()),
        "output_dir": str(output_dir.resolve()),
        "checkpoint": str(ckpt_path.resolve()),
        "best_epoch": int(best["epoch"]) if best["epoch"] is not None else None,
        "best_val_score": float(best["score"]) if best["score"] is not None else None,
        "best_val_metrics": best["metrics"],
        "history": history,
        "config": {
            "width": int(args.width),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "gate_strength": float(args.gate_strength),
            "bridge_error_threshold": float(args.bridge_error_threshold),
            "final_weight": float(args.final_weight),
            "pred_weight": float(args.pred_weight),
            "gate_weight": float(args.gate_weight),
            "composite_weight": float(args.composite_weight),
            "gradient_weight": float(args.gradient_weight),
            "contrast_weight": float(args.contrast_weight),
            "boundary_weight": float(args.boundary_weight),
            "trimap_weight": float(args.trimap_weight),
            "semi_weight": float(args.semi_weight),
            "hair_weight": float(args.hair_weight),
            "hard_negative_weight": float(args.hard_negative_weight),
            "gate_target_mode": str(args.gate_target_mode),
            "max_train_samples": int(args.max_train_samples),
            "max_val_samples": int(args.max_val_samples),
        },
    }
    (output_dir / "train_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
