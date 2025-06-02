
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

WANDB_AVAILABLE = True

# ─────────────────────────────────────────────────────────────────────────────
#  Autopilot model
# ─────────────────────────────────────────────────────────────────────────────
class AutopilotModel(nn.Module):
    """ResNet‑18 backbone with a small MLP head.

    Parameters
    ----------
    output_size : int
        Dimension of the final regression output (e.g. 2 for *steer*, *throttle*).
    dropout_prob : float, optional
        Dropout probability applied between the fully‑connected layers.
    pretrained : bool, optional
        If *True*, load ImageNet‑1K weights for the backbone.
    """

    def __init__(
        self,
        output_size: int = 2,
        dropout_prob: float = 0.3,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        self.backbone = torchvision.models.resnet18(
            weights="IMAGENET1K_V1" if pretrained else None
        )

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(64, output_size),
        )

        # Feature‑extractor is **created lazily** the first time it is needed.
        self._feature_extractor: nn.Module | None = None

    # --------------------------------------------------------------------- #
    #  Forward                                                                #
    # --------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Inference – maps an RGB image tensor to actuator commands."""
        return self.backbone(x)

    # --------------------------------------------------------------------- #
    #  Lazy feature extractor                                               #
    # --------------------------------------------------------------------- #
    def extract_intermediate_features(
        self, x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Return layer1 and layer2 feature tensors (eval‑only).

        This method **never** runs during `.train()` – call it from a validation
        loop with `torch.no_grad()` instead.
        """
        if self.training:
            raise RuntimeError("`extract_intermediate_features` called in train mode")

        if self._feature_extractor is None:
            self._feature_extractor = create_feature_extractor(
                self.backbone, return_nodes={"layer1": "feat1", "layer2": "feat2"}
            )
            self._feature_extractor.eval()

        return self._feature_extractor(x)


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
    model: AutopilotModel,
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

        if log_features and WANDB_AVAILABLE:
            feats = model.extract_intermediate_features(imgs)
            # Log layer statistics as histograms (one batch per epoch for brevity)
            wandb.log({name: wandb.Histogram(t.cpu().flatten()) for name, t in feats.items()})
            log_features = False  # only log once per epoch to reduce overhead

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
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    # Model & logging
    parser.add_argument("--output_size", type=int, default=2, help="Size of the final output vector (e.g., 2‑DOF)")

    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--no_pretrained", action="store_true", help="Disable ImageNet weights")

    parser.add_argument("--wandb_project", type=str, default="autopilot-e2e", help="W&B project name")

    args = parser.parse_args()

    # Environment – limit visible GPUs (optional)
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training on {device} …")

    # W&B
    if WANDB_AVAILABLE:
        wandb.init(project=args.wandb_project, config=vars(args), save_code=True)

    # Data
    train_loader, val_loader = get_data_loaders(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
    )

    # Model, criterion, optimizer
    model = AutopilotModel(
        output_size=args.output_size,
        dropout_prob=args.dropout,
        pretrained=not args.no_pretrained,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    PATIENCE = 10

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
            f"[Epoch {epoch:03d}/{args.epochs}]  Train: {train_loss:.4f} | Val: {val_loss:.4f}"
        )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            ckpt_path = Path("best_model.pt")
            torch.save(model.state_dict(), ckpt_path)
            # if WANDB_AVAILABLE:
                # wandb.save(str(ckpt_path))
            print(f"[INFO] New best – model saved to {ckpt_path}.")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print("[INFO] Early stopping: no improvement.")
                break

    if WANDB_AVAILABLE:
        wandb.finish()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":  # pragma: no cover
    main()
