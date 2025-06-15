
#!/usr/bin/env python3
"""
train.py
========
Training script for the End‑to‑End Autopilot model described in *autopilot_training.ipynb*.

Basic usage
-----------
$ python train.py --train-dir /path/to/train --val-dir /path/to/val --epochs 50

The script supports Weights & Biases (wandb) logging and can optionally export
intermediate ResNet features (layer1, layer2) **only during evaluation**.

Adjust the hyper‑parameters with command‑line flags or by editing the defaults
at the bottom of this file.

Author : ChatGPT (o3)
Created: 2025‑06‑02
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor
import wandb
from autopilot_dataset import AutopilotDataset
from autopilot_model import E2E_CNN

WANDB_AVAILABLE = True


def get_data_loaders(
    train_dir: Path,
    val_dir: Path,
    batch_size: int,
    num_workers: int = 10,
    img_size: int = 224,
) -> tuple[DataLoader, DataLoader]:
    """Create `torch.utils.data.DataLoader` objects for training and validation."""

    # common_tfms = [
    #     transforms.ToTensor(),
    #     transforms.Resize((img_size, img_size), antialias=True),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ]

    # train_tfms = transforms.Compose(
    #     [
    #         transforms.RandomApply(
    #             [transforms.ColorJitter(brightness=0.3)],
    #             p=0.3,              
    #         ),
    #         *common_tfms,
    #     ]
    # )
    # val_tfms = transforms.Compose(common_tfms)

    train_ds = AutopilotDataset(train_dir,
                                img_size,
                                random_horizontal_flip=False,
                                random_noise=True,
                                random_blur=True,
                                random_color_jitter=True,
                                keep_images_in_ram=True)   
    val_ds  =  AutopilotDataset(val_dir,
                                img_size,
                                random_horizontal_flip=False,
                                random_noise=False,
                                random_blur=False,
                                random_color_jitter=False,
                                keep_images_in_ram=True)

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

    for imgs, targets in loader:
        imgs, targets = imgs.to(device, non_blocking=True), targets.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: E2E_CNN,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    log_features: bool = False,
) -> float:
    """Validate the model and return mean loss."""
    model.eval()
    val_loss = 0.0

    for imgs, targets in loader:
        imgs, targets = imgs.to(device, non_blocking=True), targets.to(device)
        outputs = model(imgs)
        val_loss += criterion(outputs, targets).item() * imgs.size(0)

        # if log_features and WANDB_AVAILABLE:
        #     feats = model.extract_intermediate_features(imgs)
        #     # Log layer statistics as histograms (one batch per epoch for brevity)
        #     wandb.log({name: wandb.Histogram(t.cpu().flatten()) for name, t in feats.items()})
        #     log_features = False  # only log once per epoch to reduce overhead

    return val_loss / len(loader.dataset)


# ─────────────────────────────────────────────────────────────────────────────
#  Main training routine
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:  # noqa: C901
    parser = argparse.ArgumentParser(description="AutopilotModel trainer")
    root_dir = Path("./datasets")
    # Data
    parser.add_argument("--train_dir", type=Path, default=root_dir / "train")
    parser.add_argument("--val_dir", type=Path, default=root_dir / "valid")

    # Training params
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    # Model & logging
    parser.add_argument("--output_size", type=int, default=2, help="Size of the final output vector (e.g., 2‑DOF)")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--no_pretrained", action="store_true", help="Disable ImageNet weights")

    parser.add_argument("--wandb_project", type=str, default="autopilot-e2e", help="W&B project name")
    parser.add_argument("--freeze_front", action="store_true", help="Freeze the backbone layers")
    parser.add_argument("--resume_checkpoint", type=Path, default="./checkpoint/last_model_cnn.pt",
                        help="불러올 체크포인트(.pt) 경로")


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
    model = E2E_CNN(
        output_size=args.output_size,
        dropout_prob=args.dropout,
        pretrained=not args.no_pretrained,
        freeze_front=args.freeze_front,
    ).to(device)
    print(f"[INFO] Model: {model.__class__.__name__}")

    if args.freeze_front and args.resume_checkpoint is not None and args.resume_checkpoint.exists():
        print(f"[INFO] Loading pretrained weights from {args.resume_checkpoint}")
        state = torch.load(args.resume_checkpoint, weights_only = True)
        model.load_state_dict(state, strict=False)


    criterion = nn.MSELoss()
    if args.freeze_front:
        optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
        )
    else:
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
        val_loss = evaluate(model, val_loader, criterion, device, log_features=True)
        scheduler.step(val_loss)

        if WANDB_AVAILABLE:
            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )

        print(
            f"[Epoch {epoch:03d}/{args.epochs}]  Train: {train_loss:.7f} | Val: {val_loss:.7f}"
        )
        ckpt_path = Path("epoch_model_cnn.pt")
        torch.save(model.state_dict(),ckpt_path)

    if WANDB_AVAILABLE:
        wandb.finish()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":  # pragma: no cover
    main()
