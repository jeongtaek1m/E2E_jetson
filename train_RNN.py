#!/usr/bin/env python3
"""
train.py
========
Training script for the End-to-End Autopilot model described in *autopilot_training.ipynb*.
…

Author : ChatGPT (o3)
Created: 2025-06-02
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from autopilot_dataset import AutopilotDataset
# 아래쪽에서 import하는 이름을 E2E_CRNN으로 바꾸든지, 
# AutopilotModel로 바꾸든 하나로 맞춰야 합니다.
from autopilot_model import E2E_CRNN  

WANDB_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
#  Utility helpers
# ─────────────────────────────────────────────────────────────────────────────
def get_data_loaders(
    train_dir: Path,
    val_dir: Path,
    batch_size: int,
    num_workers: int = 10,
    img_size: int = 224,
) -> tuple[DataLoader, DataLoader]:
    """Create `torch.utils.data.DataLoader` objects for training and validation."""

    train_ds = AutopilotDataset(
        train_dir,
        img_size,
        random_horizontal_flip=False,
        random_noise=True,
        random_blur=True,
        random_color_jitter=True,
        keep_images_in_ram=False,
        use_rnn=True,   # RNN 모드 활성화
    )
    val_ds = AutopilotDataset(
        val_dir,
        img_size,
        random_horizontal_flip=False,
        random_noise=False,
        random_blur=False,
        random_color_jitter=False,
        keep_images_in_ram=False,
        use_rnn=True,   # 검증에서도 RNN 모드
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run **one** training epoch and return mean loss."""
    model.train()
    running_loss = 0.0

    for imgs, speed_seqs, targets in loader:
        imgs       = imgs.to(device,      non_blocking=True)
        speed_seqs = speed_seqs.to(device, non_blocking=True)
        targets    = targets.to(device)

        optimizer.zero_grad()
        outputs = model(imgs, speed_seqs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: E2E_CRNN,          # 또는 AutopilotModel
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    log_features: bool = False,
) -> float:
    """Validate the model and return mean loss."""
    model.eval()
    val_loss = 0.0

    for imgs, speed_seqs, targets in loader:
        imgs       = imgs.to(device,      non_blocking=True)
        speed_seqs = speed_seqs.to(device, non_blocking=True)
        targets    = targets.to(device)

        outputs = model(imgs, speed_seqs)
        val_loss += criterion(outputs, targets).item() * imgs.size(0)

        # if log_features and WANDB_AVAILABLE:
        #     feats = model.extract_intermediate_features(imgs)
        #     wandb.log({name: wandb.Histogram(t.cpu().flatten()) for name, t in feats.items()})
        #     log_features = False

    return val_loss / len(loader.dataset)


# ─────────────────────────────────────────────────────────────────────────────
#  Main training routine
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="AutopilotModel trainer")
    root_dir = Path("./datasets")
    # Data
    parser.add_argument("--train_dir", type=Path, default=root_dir / "train")
    parser.add_argument("--val_dir",   type=Path, default=root_dir / "valid")

    # Training params
    parser.add_argument("--epochs",       type=int,   default=100)
    parser.add_argument("--batch_size",   type=int,   default=64)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    # Model & logging
    parser.add_argument("--output_size", type=int,   default=2,
                        help="Size of the final output vector (e.g., 2-DOF)")
    parser.add_argument("--dropout",     type=float, default=0.3)
    parser.add_argument("--no_pretrained", action="store_true",
                        help="Disable ImageNet weights")

    parser.add_argument("--wandb_project", type=str, default="autopilot-e2e",
                        help="W&B project name")

    args = parser.parse_args()

    # Environment – limit visible GPUs (optional)
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training on {device}")

    # W&B
    if WANDB_AVAILABLE:
        wandb.init(project=args.wandb_project, config=vars(args), save_code=False)

    # Data
    train_loader, val_loader = get_data_loaders(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
    )

    # Model, criterion, optimizer
    model = E2E_CRNN(
        output_size=args.output_size,
        dropout_prob=args.dropout,
        pretrained=not args.no_pretrained,
    ).to(device)
    print(f"[INFO] Model: {model.__class__.__name__}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    best_val_loss = float("inf")

    # ───────────────────────────────────────
    #  Training loop
    # ───────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss   = evaluate(model, val_loader, criterion, device, log_features=True)
        scheduler.step(val_loss)

        if WANDB_AVAILABLE:
            wandb.log({
                "epoch":      epoch,
                "train/loss": train_loss,
                "val/loss":   val_loss,
                "lr":         optimizer.param_groups[0]["lr"],
            })

        print(f"[Epoch {epoch:03d}/{args.epochs}]  Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        # Checkpoint 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = Path("best_model_rnn.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"[INFO] New best – model saved to {ckpt_path}.")

    ckpt_path = Path("last_model_rnn.pt")
    torch.save(model.state_dict(), ckpt_path)

    if WANDB_AVAILABLE:
        wandb.finish()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
