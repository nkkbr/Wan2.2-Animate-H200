#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan.utils.animate_contract import load_preprocess_metadata
from wan.utils.media_io import load_mask_artifact
from wan.utils.trainable_edge_model import EdgeRefinementNet, split_outputs


def _load_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.float32) / 255.0


def _load_rgb(path: Path, size: tuple[int, int]) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
    return img.astype(np.float32) / 255.0


class EdgeDataset(Dataset):
    def __init__(self, records: list[dict], baseline_preprocess_dir: Path, size: tuple[int, int]):
        self.samples = []
        metadata = load_preprocess_metadata(baseline_preprocess_dir)
        artifacts = metadata["src_files"]
        self.soft_alpha = load_mask_artifact(baseline_preprocess_dir / artifacts["soft_alpha"]["path"], artifacts["soft_alpha"].get("format"))
        self.boundary_band = load_mask_artifact(baseline_preprocess_dir / artifacts["boundary_band"]["path"], artifacts["boundary_band"].get("format"))
        for record in records:
            frame_idx = int(record["preprocess_frame_index"])
            image = _load_rgb(Path(record["image_path"]), size)
            label_json = json.loads(Path(record["label_json_path"]).read_text(encoding="utf-8"))
            label_root = Path(record["label_json_path"]).parent
            alpha = _load_gray(label_root / label_json["annotations"]["soft_alpha"]["path"])
            boundary = _load_gray(label_root / label_json["annotations"]["boundary_mask"]["path"])
            hair = _load_gray(label_root / label_json["annotations"]["hair_edge_mask"]["path"])
            alpha = cv2.resize(alpha, size, interpolation=cv2.INTER_LINEAR)
            boundary = cv2.resize(boundary, size, interpolation=cv2.INTER_LINEAR)
            hair = cv2.resize(hair, size, interpolation=cv2.INTER_LINEAR)
            soft_alpha = cv2.resize(self.soft_alpha[frame_idx], size, interpolation=cv2.INTER_LINEAR)
            boundary_band = cv2.resize(self.boundary_band[frame_idx], size, interpolation=cv2.INTER_LINEAR)

            inputs = np.concatenate([
                image,
                soft_alpha[..., None],
                boundary_band[..., None],
            ], axis=-1)
            target = np.stack([alpha, boundary], axis=0)
            self.samples.append({
                "inputs": torch.from_numpy(inputs).permute(2, 0, 1).contiguous().float(),
                "target": torch.from_numpy(target).contiguous().float(),
                "hair": torch.from_numpy(hair[None]).contiguous().float(),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        return self.samples[index]


def _epoch(model, loader, optimizer, device):
    training = optimizer is not None
    model.train(training)
    alpha_loss_fn = nn.L1Loss()
    bce = nn.BCELoss(reduction="none")
    total_loss = 0.0
    total_items = 0
    for batch in loader:
        inputs = batch["inputs"].to(device)
        target = batch["target"].to(device)
        hair = batch["hair"].to(device)
        with torch.set_grad_enabled(training):
            outputs = split_outputs(model(inputs))
            alpha_loss = alpha_loss_fn(outputs.alpha, target[:, 0:1])
            boundary_loss = bce(outputs.boundary, target[:, 1:2])
            weight = 1.0 + 1.5 * hair
            boundary_loss = (boundary_loss * weight).mean()
            loss = alpha_loss + 0.8 * boundary_loss
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += float(loss.item()) * inputs.shape[0]
        total_items += inputs.shape[0]
    return total_loss / max(total_items, 1)


def main():
    parser = argparse.ArgumentParser(description="Train a tiny reviewed-data edge model for optimization6 Step05.")
    parser.add_argument("--split_json", required=True, type=str)
    parser.add_argument("--baseline_preprocess_dir", required=True, type=str)
    parser.add_argument("--run_dir", required=True, type=str)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=160)
    args = parser.parse_args()

    split = json.loads(Path(args.split_json).read_text(encoding="utf-8"))
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    size = (int(args.width), int(args.height))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = EdgeDataset(split["train"], Path(args.baseline_preprocess_dir), size)
    val_ds = EdgeDataset(split["val"], Path(args.baseline_preprocess_dir), size)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

    model = EdgeRefinementNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    best = {"val_loss": None, "epoch": None}
    history = []
    ckpt_path = run_dir / "best.pt"
    for epoch in range(1, int(args.epochs) + 1):
        train_loss = _epoch(model, train_loader, optimizer, device)
        with torch.no_grad():
            val_loss = _epoch(model, val_loader, None, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if best["val_loss"] is None or val_loss < best["val_loss"]:
            best = {"val_loss": val_loss, "epoch": epoch}
            torch.save({"model": model.state_dict(), "size": size}, ckpt_path)

    summary = {
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt_path),
        "best": best,
        "history": history,
        "device": str(device),
    }
    (run_dir / "training_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"checkpoint": str(ckpt_path), "best_val_loss": best["val_loss"], "best_epoch": best["epoch"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
