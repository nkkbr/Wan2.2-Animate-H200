#!/usr/bin/env python
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan.utils.semantic_edge_experts import SemanticEdgeExpertsModel
from wan.utils.trainable_alpha_model import stack_input_channels


SEMANTIC_TARGET_KEY = {
    "face": "gt_face_boundary",
    "hair": "gt_hair_edge",
    "hand": "gt_hand_boundary",
    "cloth": "gt_cloth_boundary",
}


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


def _load_dataset(dataset_npz: Path, dataset_json: Path):
    arrays = np.load(dataset_npz)
    info = json.loads(dataset_json.read_text(encoding="utf-8"))
    return arrays, info["records"], info


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


def _evaluate(pred_alpha, gt_alpha, semantic_boundary, gt_trimap, semantic_names):
    thresholds = np.linspace(0.35, 0.75, 17)
    best_threshold = 0.5
    best_score = -1.0
    pred_cache = {}
    gt_mask = (gt_alpha >= 0.5).astype(np.uint8)
    for threshold in thresholds:
        pred_mask = (pred_alpha >= threshold).astype(np.uint8)
        pred_cache[threshold] = pred_mask
        score = float(np.mean([
            _boundary_f1(pred_mask[i], gt_mask[i])
            for i in range(len(pred_alpha))
        ]))
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    pred_mask = pred_cache[best_threshold]
    overall_boundary_f1 = float(np.mean([
        _boundary_f1(pred_mask[i], gt_mask[i])
        for i in range(len(pred_alpha))
    ]))
    alpha_mae = float(np.abs(pred_alpha - gt_alpha).mean())
    trimap_error = float((np.abs(pred_alpha - gt_alpha) * gt_trimap).sum() / max(float(gt_trimap.sum()), 1.0))
    per_semantic_scores = defaultdict(list)
    for i, name in enumerate(semantic_names):
        per_semantic_scores[name].append(_boundary_f1(pred_mask[i], semantic_boundary[i]))
    return {
        "boundary_f1_mean": overall_boundary_f1,
        "alpha_mae_mean": alpha_mae,
        "trimap_error_mean": trimap_error,
        "best_threshold": best_threshold,
        "semantic_boundary_f1": {name: float(np.mean(vals)) for name, vals in per_semantic_scores.items()},
    }


def _score(metrics, enabled_tags):
    score = float(metrics["boundary_f1_mean"] - 6.0 * metrics["alpha_mae_mean"] - 2.5 * metrics["trimap_error_mean"])
    for tag in enabled_tags:
        score += 0.6 * float(metrics["semantic_boundary_f1"].get(tag, 0.0))
    return score


def main():
    parser = argparse.ArgumentParser(description="Train semantic edge experts on roi_dataset_v2.")
    parser.add_argument("--dataset_npz", required=True)
    parser.add_argument("--dataset_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--semantic_tags", nargs="+", required=True, choices=sorted(SEMANTIC_TARGET_KEY))
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--residual_scale", type=float, default=0.20)
    parser.add_argument("--semantic_emb_dim", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--semantic_weight", type=float, default=5.0)
    parser.add_argument("--boundary_weight", type=float, default=3.0)
    parser.add_argument("--preserve_weight", type=float, default=0.6)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    arrays, records, info = _load_dataset(Path(args.dataset_npz), Path(args.dataset_json))
    inputs = _build_inputs(arrays)
    gt_alpha = arrays["gt_soft_alpha"].astype(np.float32)
    gt_trimap = arrays["gt_trimap_unknown"].astype(np.float32)
    gt_boundary = arrays["gt_boundary_mask"].astype(np.float32)
    difficulty = arrays["difficulty_score"].astype(np.float32)
    base_alpha = inputs[:, :, :, 6].astype(np.float32)

    enabled_tags = tuple(args.semantic_tags)
    semantic_records = []
    semantic_indices = []
    semantic_target = []
    semantic_ids = []
    for idx, record in enumerate(records):
        if record["task_type"] != "semantic_boundary_expert":
            continue
        tag = record["semantic_boundary_tag"]
        if tag not in enabled_tags:
            continue
        semantic_indices.append(idx)
        semantic_records.append(record)
        semantic_target.append(arrays[SEMANTIC_TARGET_KEY[tag]][idx].astype(np.float32))
        semantic_ids.append(enabled_tags.index(tag))

    if not semantic_indices:
        raise RuntimeError("No semantic expert samples found for requested tags.")

    semantic_indices = np.asarray(semantic_indices, dtype=np.int64)
    semantic_target = np.asarray(semantic_target, dtype=np.float32)
    semantic_ids = np.asarray(semantic_ids, dtype=np.int64)
    semantic_inputs = inputs[semantic_indices]
    semantic_gt_alpha = gt_alpha[semantic_indices]
    semantic_gt_trimap = gt_trimap[semantic_indices]
    semantic_gt_boundary = gt_boundary[semantic_indices]
    semantic_base_alpha = base_alpha[semantic_indices]
    semantic_difficulty = difficulty[semantic_indices]
    dataset_split = np.asarray([record["dataset_split"] for record in semantic_records])
    semantic_names = np.asarray([record["semantic_boundary_tag"] for record in semantic_records])

    train_idx = np.where(dataset_split == "train")[0]
    val_idx = np.where(dataset_split == "val")[0]
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise RuntimeError("Semantic expert dataset requires train and val samples.")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = SemanticEdgeExpertsModel(
        in_channels=semantic_inputs.shape[-1],
        semantic_tags=enabled_tags,
        width=args.width,
        residual_scale=args.residual_scale,
        semantic_emb_dim=args.semantic_emb_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_x = torch.from_numpy(semantic_inputs[train_idx]).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)
    train_base_alpha = torch.from_numpy(semantic_base_alpha[train_idx]).to(device=device, dtype=torch.float32)
    train_gt_alpha = torch.from_numpy(semantic_gt_alpha[train_idx]).to(device=device, dtype=torch.float32)
    train_semantic_target = torch.from_numpy(semantic_target[train_idx]).to(device=device, dtype=torch.float32)
    train_gt_boundary = torch.from_numpy(semantic_gt_boundary[train_idx]).to(device=device, dtype=torch.float32)
    train_gt_trimap = torch.from_numpy(semantic_gt_trimap[train_idx]).to(device=device, dtype=torch.float32)
    train_difficulty = torch.from_numpy(semantic_difficulty[train_idx]).to(device=device, dtype=torch.float32)
    train_semantic_ids = torch.from_numpy(semantic_ids[train_idx]).to(device=device, dtype=torch.long)

    val_x = torch.from_numpy(semantic_inputs[val_idx]).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)
    val_base_alpha = torch.from_numpy(semantic_base_alpha[val_idx]).to(device=device, dtype=torch.float32)
    val_semantic_ids = torch.from_numpy(semantic_ids[val_idx]).to(device=device, dtype=torch.long)

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
            y_base_alpha = train_base_alpha[batch_idx]
            y_gt_alpha = train_gt_alpha[batch_idx]
            y_semantic_target = train_semantic_target[batch_idx]
            y_gt_boundary = train_gt_boundary[batch_idx]
            y_gt_trimap = train_gt_trimap[batch_idx]
            y_difficulty = train_difficulty[batch_idx]
            y_semantic_ids = train_semantic_ids[batch_idx]

            pred = model(x, y_base_alpha[:, None], y_semantic_ids)[:, 0]
            focus = torch.clamp(y_semantic_target + 0.75 * y_gt_boundary + 0.75 * y_gt_trimap, 0.0, 1.0)
            preserve = 1.0 - torch.clamp(y_semantic_target + y_gt_trimap, 0.0, 1.0)
            weight = 1.0 + float(args.semantic_weight) * y_semantic_target + float(args.boundary_weight) * y_gt_boundary + 0.35 * y_difficulty[:, None, None]
            loss_alpha = ((pred - y_gt_alpha).abs() * weight).mean()
            loss_focus = ((pred - y_gt_alpha).abs() * (1.0 + 4.0 * focus)).mean()
            loss_preserve = ((pred - y_base_alpha).abs() * preserve).mean()
            semantic_mask = (pred >= 0.35).float()
            loss_semantic = F.binary_cross_entropy(semantic_mask.clamp(1e-4, 1.0 - 1e-4), y_semantic_target.clamp(0.0, 1.0))
            loss = loss_alpha + 0.8 * loss_focus + float(args.preserve_weight) * loss_preserve + 0.2 * loss_semantic
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            val_pred = model(val_x, val_base_alpha[:, None], val_semantic_ids)[:, 0].detach().cpu().numpy()
        val_metrics = _evaluate(
            val_pred,
            semantic_gt_alpha[val_idx],
            semantic_target[val_idx],
            semantic_gt_trimap[val_idx],
            semantic_names[val_idx],
        )
        score = _score(val_metrics, enabled_tags)
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
        raise RuntimeError("Semantic experts training did not produce a best checkpoint.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / "semantic_edge_experts.pt"
    torch.save({
        "model_state": best["state"],
        "in_channels": int(semantic_inputs.shape[-1]),
        "semantic_tags": list(enabled_tags),
        "width": int(args.width),
        "residual_scale": float(args.residual_scale),
        "semantic_emb_dim": int(args.semantic_emb_dim),
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
    }, ckpt_path)

    summary = {
        "dataset_npz": str(Path(args.dataset_npz).resolve()),
        "dataset_json": str(Path(args.dataset_json).resolve()),
        "checkpoint_path": str(ckpt_path.resolve()),
        "enabled_tags": list(enabled_tags),
        "best_epoch": int(best["epoch"]),
        "best_val_score": float(best["score"]),
        "best_val_metrics": best["metrics"],
        "sample_counts": {
            "total": int(len(semantic_indices)),
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(np.sum(dataset_split == "test")),
            "semantic_distribution": dict(Counter(semantic_names.tolist())),
        },
        "history": history,
        "config": {
            "width": int(args.width),
            "residual_scale": float(args.residual_scale),
            "semantic_emb_dim": int(args.semantic_emb_dim),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "semantic_weight": float(args.semantic_weight),
            "boundary_weight": float(args.boundary_weight),
            "preserve_weight": float(args.preserve_weight),
            "patience": int(args.patience),
        },
    }
    (output_dir / "train_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "checkpoint_path": str(ckpt_path.resolve()),
        "best_val_metrics": best["metrics"],
        "best_epoch": int(best["epoch"]),
        "enabled_tags": list(enabled_tags),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
