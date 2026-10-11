"""MedScan Assist - Model Training Pipeline.

Orchestrates two-phase fine-tuning of EfficientNet-B0 on Chest X-Rays:
- Phase 1: Frozen backbone, training custom classification head.
- Phase 2: Unfreezing the last 2 backbone blocks with reduced learning rate.
Features weighted CrossEntropyLoss for class imbalance, AdamW optimizer,
ReduceLROnPlateau scheduler, early stopping, checkpointing, tqdm progress,
metrics plotting, and graceful KeyboardInterrupt handling.
"""

import argparse
import json
import logging
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import Config
from models.model_architecture import (
    build_efficientnet_b0,
    freeze_backbone,
    unfreeze_backbone,
)
from training.prepare_dataset import create_dataloaders

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train")


def set_seed(seed: int = Config.SEED) -> None:
    """Set random seeds across random, numpy, and torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info("Deterministic seed set to %d", seed)


class EarlyStopping:
    """Early stopping monitor based on validation loss."""

    def __init__(self, patience: int = Config.EARLY_STOPPING_PATIENCE, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < (self.best_loss - self.min_delta):
            self.best_loss = val_loss
            self.counter = 0
            return True  # Improved
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False  # Did not improve


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> Tuple[float, float]:
    """Execute one training epoch with tqdm progress reporting."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch:02d} [Train]", leave=False)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{(correct / total):.4f}"})

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> Tuple[float, float]:
    """Evaluate model on validation split."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch:02d} [Val]  ", leave=False)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

        pbar.set_postfix({"val_loss": f"{loss.item():.4f}", "val_acc": f"{(correct / total):.4f}"})

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return epoch_loss, epoch_acc


def save_complete_checkpoint(
    file_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, Any],
    history: Dict[str, List[float]],
    config_dict: Dict[str, Any],
) -> None:
    """Save training checkpoint including model state, optimizer, history, and metadata."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "architecture": Config.MODEL_NAME,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "class_names": Config.CLASS_NAMES,
        "class_to_idx": Config.CLASS_TO_IDX,
        "idx_to_class": Config.IDX_TO_CLASS,
        "epoch": epoch,
        "metrics": metrics,
        "training_history": history,
        "config": config_dict,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "disclaimer": Config.RESEARCH_DISCLAIMER,
    }
    torch.save(payload, str(file_path))
    logger.info("Saved checkpoint to: %s", file_path)


def plot_training_history(history: Dict[str, List[float]], output_path: Path) -> None:
    """Plot training and validation loss and accuracy curves."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    ax1.plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#1f77b4")
    ax1.plot(epochs, history["val_loss"], "s-", label="Val Loss", color="#d62728")
    ax1.set_title("Cross-Entropy Loss vs Epoch", fontsize=12)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    # Accuracy curve
    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], "o-", label="Train Acc (%)", color="#2ca02c")
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], "s-", label="Val Acc (%)", color="#ff7f0e")
    ax2.set_title("Accuracy vs Epoch", fontsize=12)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    plt.suptitle("MedScan Assist - Training & Validation Curves (Research Use Only)", fontsize=13)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Training curve plot saved to: %s", output_path)


def train_model(
    data_dir: Path,
    checkpoint_dir: Path,
    reports_dir: Path,
    epochs: int = Config.EPOCHS,
    phase1_epochs: int = 5,
    batch_size: int = Config.BATCH_SIZE,
    learning_rate: float = Config.LEARNING_RATE,
    weight_decay: float = Config.WEIGHT_DECAY,
    patience: int = Config.EARLY_STOPPING_PATIENCE,
    resume_checkpoint_path: Optional[Path] = None,
    seed: int = Config.SEED,
) -> Dict[str, Any]:
    """Execute full two-phase model training pipeline."""
    set_seed(seed)
    device = Config.DEVICE
    logger.info("Using compute device: %s", device)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Prepare DataLoaders
    dataloaders, datasets, class_weights = create_dataloaders(
        processed_dir=data_dir,
        batch_size=batch_size,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    logger.info("Initialized CrossEntropyLoss with weights: %s", class_weights.tolist())

    # 2. Build Model
    model = build_efficientnet_b0(
        num_classes=Config.NUM_CLASSES,
        pretrained=True,
        freeze_backbone_layers=True,
        dropout_rate=Config.DROPOUT_RATE,
    )
    model.to(device)

    # 3. Setup Optimizer and Scheduler for Phase 1
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6)
    early_stopper = EarlyStopping(patience=patience)

    start_epoch = 1
    best_val_loss = float("inf")
    best_epoch = 0
    best_metrics: Dict[str, float] = {}

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "learning_rate": [],
    }

    # Resume if requested
    if resume_checkpoint_path and resume_checkpoint_path.exists():
        logger.info("Resuming training from checkpoint: %s", resume_checkpoint_path)
        checkpoint = torch.load(resume_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception as e:
                logger.warning("Could not restore optimizer state: %s", e)
        start_epoch = checkpoint.get("epoch", 0) + 1
        history = checkpoint.get("training_history", history)
        best_val_loss = checkpoint.get("metrics", {}).get("val_loss", float("inf"))
        logger.info("Resumed from epoch %d with best_val_loss: %.4f", start_epoch, best_val_loss)

    config_dict = {
        "model_name": Config.MODEL_NAME,
        "epochs": epochs,
        "phase1_epochs": phase1_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "patience": patience,
        "seed": seed,
        "class_weights": class_weights.cpu().tolist(),
        "device": str(device),
    }

    best_checkpoint_path = checkpoint_dir / "best_model.pth"
    latest_checkpoint_path = checkpoint_dir / "latest_checkpoint.pth"
    interrupted_checkpoint_path = checkpoint_dir / "interrupted_checkpoint.pth"
    history_json_path = reports_dir / "training_history.json"
    curves_plot_path = reports_dir / "training_curves.png"

    logger.info("Starting training loop (Total epochs: %d, Phase 1: %d)", epochs, phase1_epochs)

    try:
        for epoch in range(start_epoch, epochs + 1):
            # Phase Transition: Unfreeze last 2 blocks after phase1_epochs
            if epoch == phase1_epochs + 1:
                logger.info(">>> Transitioning to Phase 2: Unfreezing top 2 feature blocks with reduced LR.")
                unfreeze_backbone(model, num_blocks_to_unfreeze=2)
                phase2_lr = learning_rate * 0.1
                optimizer = AdamW(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=phase2_lr,
                    weight_decay=weight_decay,
                )
                scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6)

            current_lr = optimizer.param_groups[0]["lr"]
            history["learning_rate"].append(current_lr)

            # Training step
            train_loss, train_acc = train_one_epoch(
                model=model,
                dataloader=dataloaders["train"],
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
            )

            # Validation step
            val_loss, val_acc = evaluate_epoch(
                model=model,
                dataloader=dataloaders["val"],
                criterion=criterion,
                device=device,
                epoch=epoch,
            )

            # Step learning rate scheduler
            scheduler.step(val_loss)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            phase_label = "Phase 1 (Head)" if epoch <= phase1_epochs else "Phase 2 (Fine-tuning)"
            logger.info(
                "[%s] Epoch %02d/%02d | LR: %.2e | Train Loss: %.4f | Train Acc: %.2f%% | Val Loss: %.4f | Val Acc: %.2f%%",
                phase_label, epoch, epochs, current_lr, train_loss, train_acc * 100, val_loss, val_acc * 100
            )

            # Save latest checkpoint
            save_complete_checkpoint(
                file_path=latest_checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics={"val_loss": val_loss, "val_acc": val_acc, "train_loss": train_loss, "train_acc": train_acc},
                history=history,
                config_dict=config_dict,
            )

            # Check if this is the best epoch
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_metrics = {
                    "best_epoch": best_epoch,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                }
                save_complete_checkpoint(
                    file_path=best_checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics=best_metrics,
                    history=history,
                    config_dict=config_dict,
                )
                logger.info("--> New best model recorded at epoch %d with val_loss: %.4f", epoch, val_loss)

            # Early stopping check
            improved = early_stopper(val_loss)
            if early_stopper.early_stop:
                logger.warning(
                    "Early stopping triggered at epoch %d. No improvement for %d epochs.",
                    epoch, early_stopper.patience
                )
                break

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user. Saving current checkpoint state before exit.")
        save_complete_checkpoint(
            file_path=interrupted_checkpoint_path,
            model=model,
            optimizer=optimizer,
            epoch=len(history["train_loss"]),
            metrics={"val_loss": history["val_loss"][-1] if history["val_loss"] else None},
            history=history,
            config_dict=config_dict,
        )

    # Save training history JSON
    with open(history_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "disclaimer": Config.RESEARCH_DISCLAIMER,
                "best_epoch": best_epoch,
                "best_metrics": best_metrics,
                "history": history,
                "config": config_dict,
            },
            f,
            indent=2,
        )

    # Plot training curves
    if len(history["train_loss"]) > 0:
        plot_training_history(history, curves_plot_path)

    _print_training_summary(best_epoch, best_metrics, best_checkpoint_path)

    return {
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "best_checkpoint_path": str(best_checkpoint_path),
        "history_json_path": str(history_json_path),
        "curves_plot_path": str(curves_plot_path),
    }


def _print_training_summary(
    best_epoch: int,
    best_metrics: Dict[str, float],
    best_checkpoint_path: Path,
) -> None:
    """Print terminal summary of model training results."""
    print("\n" + "=" * 70)
    print("MEDSCAN ASSIST - TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best Epoch:           {best_epoch}")
    print(f"Best Validation Loss: {best_metrics.get('val_loss', 'N/A'):.4f}")
    val_acc = best_metrics.get("val_acc")
    print(f"Best Validation Acc:  {val_acc * 100:.2f}%" if val_acc is not None else "N/A")
    print(f"Checkpoint Saved To:  {best_checkpoint_path}")
    print("=" * 70 + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser for train.py."""
    parser = argparse.ArgumentParser(description="Train MedScan Assist Chest X-ray classification model.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.path.join(Config.BASE_DIR, "data", "processed"),
        help="Path to directory containing processed train/val/test splits (default: data/processed)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=Config.CHECKPOINT_DIR,
        help=f"Directory to store checkpoints (default: {Config.CHECKPOINT_DIR})",
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default=os.path.join(Config.REPORTS_DIR, "evaluation"),
        help="Directory to store evaluation plots and history (default: reports/evaluation)",
    )
    parser.add_argument("--epochs", type=int, default=Config.EPOCHS, help="Total training epochs.")
    parser.add_argument("--phase1-epochs", type=int, default=5, help="Number of head-only epochs in phase 1.")
    parser.add_argument("--batch-size", type=int, default=Config.BATCH_SIZE, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=Config.LEARNING_RATE, help="Initial learning rate.")
    parser.add_argument("--weight-decay", type=float, default=Config.WEIGHT_DECAY, help="Weight decay.")
    parser.add_argument("--patience", type=int, default=Config.EARLY_STOPPING_PATIENCE, help="Early stopping patience.")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume training from.")
    parser.add_argument("--seed", type=int, default=Config.SEED, help="Random seed.")
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()

    # If processed data dir does not exist, check default DATA_DIR
    data_dir_p = Path(args.data_dir)
    if not data_dir_p.exists() and Path(Config.DATA_DIR).exists():
        logger.warning("Processed dir %s not found. Falling back to %s", data_dir_p, Config.DATA_DIR)
        data_dir_p = Path(Config.DATA_DIR)

    resume_p = Path(args.resume) if args.resume else None

    try:
        train_model(
            data_dir=data_dir_p,
            checkpoint_dir=Path(args.checkpoint_dir),
            reports_dir=Path(args.reports_dir),
            epochs=args.epochs,
            phase1_epochs=args.phase1_epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            resume_checkpoint_path=resume_p,
            seed=args.seed,
        )
        sys.exit(0)
    except Exception as exc:
        logger.exception("Training failed: %s", exc)
        sys.exit(1)
