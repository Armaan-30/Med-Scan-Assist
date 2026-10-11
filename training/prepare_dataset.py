"""MedScan Assist - Dataset Preparation Module.

Handles stratified dataset re-splitting (addressing small validation sets),
directory structuring, PyTorch Dataset construction with safe RGB loading
and error handling, torchvision transform pipelines (with clinical safeguards:
no vertical flips), class weight computation, and DataLoader creation.
"""

import argparse
import json
import logging
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import Config

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("prepare_dataset")


# ==============================================================================
# Transform Definitions
# ==============================================================================

def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """Return clinical training augmentation pipeline.

    Rules:
    - Resize(256), CenterCrop(224) preserves standard anatomical proportions
    - RandomHorizontalFlip(p=0.5): Valid for chest symmetry exploration
    - RandomRotation(10): Slight patient positioning variance
    - ColorJitter(brightness=0.1, contrast=0.1): Slight exposure variance
    - IMPORTANT: NO vertical flip or aggressive warping to avoid anatomical distortions.
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.IMAGENET_MEAN, std=Config.IMAGENET_STD),
    ])


def get_eval_transforms(image_size: int = 224) -> transforms.Compose:
    """Return deterministic validation/test preprocessing pipeline."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.IMAGENET_MEAN, std=Config.IMAGENET_STD),
    ])


# ==============================================================================
# PyTorch Dataset
# ==============================================================================

class ChestXrayDataset(Dataset):
    """Robust PyTorch Dataset for Chest X-Ray images.

    Features:
    - Standardizes all images to 3-channel RGB.
    - Gracefully handles and logs unreadable or corrupted files by substituting
      with adjacent valid samples rather than halting training batches.
    - Exposes targets, classes, and class_to_idx for analysis and class weighting.
    """

    def __init__(
        self,
        samples: Optional[List[Tuple[str, int]]] = None,
        data_dir: Optional[Union[str, Path]] = None,
        transform: Optional[Callable] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
    ):
        """Initialize dataset from explicit samples list or from a split directory.

        Args:
            samples: List of (image_file_path, class_idx) tuples.
            data_dir: Root directory of split containing subdirectories for each class.
            transform: PyTorch torchvision transform pipeline.
            class_to_idx: Class name to integer mapping (defaults to Config.CLASS_TO_IDX).
        """
        self.transform = transform
        self.class_to_idx = class_to_idx or Config.CLASS_TO_IDX
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.classes = sorted(list(self.class_to_idx.keys()), key=lambda k: self.class_to_idx[k])
        self.samples: List[Tuple[str, int]] = []

        if samples is not None:
            self.samples = [(str(p), int(t)) for p, t in samples]
        elif data_dir is not None:
            self._scan_directory(Path(data_dir))
        else:
            raise ValueError("Either `samples` or `data_dir` must be provided to ChestXrayDataset.")

        self.targets = [s[1] for s in self.samples]

    def _scan_directory(self, split_dir: Path) -> None:
        """Scan directory structure for class folders and image files."""
        valid_exts = {ext.lower() for ext in Config.ALLOWED_EXTENSIONS}
        for cls_name, cls_idx in self.class_to_idx.items():
            cls_path = split_dir / cls_name
            if not cls_path.is_dir():
                logger.warning("Class subdirectory '%s' does not exist in %s", cls_name, split_dir)
                continue

            for root, _, files in os.walk(cls_path):
                for f in sorted(files):
                    file_p = Path(root) / f
                    if file_p.suffix.lower() in valid_exts:
                        self.samples.append((str(file_p), cls_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        max_retries = 3
        curr_idx = idx

        for attempt in range(max_retries):
            file_path, target = self.samples[curr_idx]
            try:
                with Image.open(file_path) as img:
                    img_rgb = img.convert("RGB")
                    if self.transform:
                        img_tensor = self.transform(img_rgb)
                    else:
                        img_tensor = transforms.ToTensor()(img_rgb)
                    return img_tensor, target
            except (UnidentifiedImageError, OSError, Exception) as exc:
                logger.warning(
                    "Error reading image %s: %s (attempt %d/%d). Skipping to next sample.",
                    file_path, exc, attempt + 1, max_retries
                )
                curr_idx = (curr_idx + 1) % len(self.samples)

        # Fallback in the rare case multiple consecutive images fail
        logger.error("Failed to load sample after %d attempts. Generating blank image fallback.", max_retries)
        blank = Image.new("RGB", (Config.IMAGE_SIZE, Config.IMAGE_SIZE), color=(0, 0, 0))
        tensor = self.transform(blank) if self.transform else transforms.ToTensor()(blank)
        return tensor, target


# ==============================================================================
# Class Weights Computation
# ==============================================================================

def compute_dataset_class_weights(
    targets: List[int],
    num_classes: int = Config.NUM_CLASSES,
) -> torch.Tensor:
    """Compute balanced class weights for loss re-weighting.

    Uses standard balanced formulation:
        weight_c = total_samples / (num_classes * count_c)
    """
    targets_np = np.array(targets)
    unique_classes = np.arange(num_classes)
    
    if len(targets_np) == 0:
        return torch.ones(num_classes, dtype=torch.float32)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=targets_np,
    )
    return torch.tensor(weights, dtype=torch.float32)


# ==============================================================================
# DataLoader Factory
# ==============================================================================

def create_dataloaders(
    processed_dir: Union[str, Path],
    batch_size: int = Config.BATCH_SIZE,
    num_workers: int = Config.NUM_WORKERS,
    pin_memory: bool = torch.cuda.is_available(),
    image_size: int = Config.IMAGE_SIZE,
) -> Tuple[Dict[str, DataLoader], Dict[str, ChestXrayDataset], torch.Tensor]:
    """Create train, val, and test DataLoaders from processed dataset directory.

    Returns:
        (dataloaders_dict, datasets_dict, train_class_weights)
    """
    processed_dir = Path(processed_dir)
    train_dir = processed_dir / "train"
    val_dir = processed_dir / "val"
    test_dir = processed_dir / "test"

    for d in [train_dir, val_dir, test_dir]:
        if not d.is_dir():
            raise FileNotFoundError(f"Required split directory not found: {d}")

    train_dataset = ChestXrayDataset(data_dir=train_dir, transform=get_train_transforms(image_size))
    val_dataset = ChestXrayDataset(data_dir=val_dir, transform=get_eval_transforms(image_size))
    test_dataset = ChestXrayDataset(data_dir=test_dir, transform=get_eval_transforms(image_size))

    class_weights = compute_dataset_class_weights(train_dataset.targets, Config.NUM_CLASSES)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=len(train_dataset) > batch_size,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    dataloaders = {"train": train_loader, "val": val_loader, "test": test_loader}
    datasets = {"train": train_dataset, "val": val_dataset, "test": test_dataset}

    logger.info(
        "DataLoaders built. Sizes -> Train: %d, Val: %d, Test: %d. Class weights: %s",
        len(train_dataset), len(val_dataset), len(test_dataset), class_weights.tolist()
    )

    return dataloaders, datasets, class_weights


# ==============================================================================
# Dataset Preparation and Stratified Splitting
# ==============================================================================

def prepare_dataset(
    raw_data_dir: Union[str, Path],
    output_dir: Union[str, Path],
    val_ratio: float = 0.15,
    min_val_threshold: int = 100,
    seed: int = Config.SEED,
    force_resplit: bool = False,
    symlink: bool = False,
) -> Dict[str, Any]:
    """Prepare and structure the dataset splits into output_dir.

    If the raw validation split has fewer than min_val_threshold images
    (e.g. only 16 images in raw Kaggle chest_xray) or force_resplit is True:
    - Combines raw train + raw val
    - Performs stratified 85/15 train/val split with fixed seed (42)
    - Strictly preserves the original test set untouched
    - Copies (or symlinks) images to output_dir
    - Saves a comprehensive split summary JSON
    """
    raw_data_dir = Path(raw_data_dir)
    output_dir = Path(output_dir)
    logger.info("Preparing dataset from %s into %s (val_ratio=%.2f, seed=%d)", raw_data_dir, output_dir, val_ratio, seed)

    if not raw_data_dir.is_dir():
        raise FileNotFoundError(f"Raw data directory does not exist: {raw_data_dir}")

    valid_exts = {ext.lower() for ext in Config.ALLOWED_EXTENSIONS}
    classes = Config.CLASS_NAMES

    # Scan raw splits
    def collect_split_files(split_name: str) -> List[Tuple[Path, str]]:
        collected: List[Tuple[Path, str]] = []
        s_dir = raw_data_dir / split_name
        if not s_dir.is_dir():
            return collected
        for cls_name in classes:
            c_dir = s_dir / cls_name
            if not c_dir.is_dir():
                continue
            for root, _, files in os.walk(c_dir):
                for f in sorted(files):
                    f_path = Path(root) / f
                    if f_path.suffix.lower() in valid_exts:
                        collected.append((f_path, cls_name))
        return collected

    raw_train_files = collect_split_files("train")
    raw_val_files = collect_split_files("val")
    raw_test_files = collect_split_files("test")

    logger.info(
        "Raw counts -> Train: %d, Val: %d, Test: %d",
        len(raw_train_files), len(raw_val_files), len(raw_test_files)
    )

    should_resplit = force_resplit or (len(raw_val_files) < min_val_threshold)
    logger.info("Re-splitting validation set from train pool: %s", should_resplit)

    if should_resplit:
        # Pool raw train and raw val
        pool = raw_train_files + raw_val_files
        pool_paths = [item[0] for item in pool]
        pool_labels = [item[1] for item in pool]

        train_paths, val_paths, train_labels, val_labels = train_test_split(
            pool_paths,
            pool_labels,
            test_size=val_ratio,
            stratify=pool_labels,
            random_state=seed,
            shuffle=True,
        )
        final_train_files = list(zip(train_paths, train_labels))
        final_val_files = list(zip(val_paths, val_labels))
    else:
        final_train_files = raw_train_files
        final_val_files = raw_val_files

    final_test_files = raw_test_files

    # Prepare destination directories
    splits_to_build = {
        "train": final_train_files,
        "val": final_val_files,
        "test": final_test_files,
    }

    summary_counts: Dict[str, Dict[str, int]] = {}

    for split_name, files in splits_to_build.items():
        summary_counts[split_name] = {cls_name: 0 for cls_name in classes}
        for src_path, cls_name in files:
            summary_counts[split_name][cls_name] += 1
            dst_folder = output_dir / split_name / cls_name
            dst_folder.mkdir(parents=True, exist_ok=True)
            dst_file = dst_folder / src_path.name

            # Avoid re-copying if identical destination already exists
            if not dst_file.exists():
                if symlink:
                    try:
                        dst_file.symlink_to(src_path.resolve())
                        continue
                    except (OSError, NotImplementedError):
                        # Symlink may fail on Windows without privileges; fallback to copy
                        pass
                shutil.copy2(src_path, dst_file)

    # Calculate class weights for the new train split
    train_labels_int = [Config.CLASS_TO_IDX[label] for _, label in final_train_files]
    class_weights_train = compute_dataset_class_weights(train_labels_int, Config.NUM_CLASSES)

    summary: Dict[str, Any] = {
        "disclaimer": Config.RESEARCH_DISCLAIMER,
        "raw_source_dir": str(raw_data_dir.resolve()),
        "processed_dir": str(output_dir.resolve()),
        "seed": seed,
        "resplit_performed": should_resplit,
        "val_ratio_used": val_ratio if should_resplit else None,
        "counts": summary_counts,
        "total_images": {s: sum(summary_counts[s].values()) for s in summary_counts},
        "computed_class_weights": {
            cls_name: round(float(class_weights_train[idx]), 4)
            for cls_name, idx in Config.CLASS_TO_IDX.items()
        },
    }

    # Save summary JSON
    summary_path = output_dir / "split_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Also save to reports/evaluation if available
    reports_eval = Path(Config.REPORTS_DIR) / "evaluation"
    reports_eval.mkdir(parents=True, exist_ok=True)
    with open(reports_eval / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Dataset prepared successfully. Summary saved to %s", summary_path)
    _print_split_summary(summary)

    return summary


def _print_split_summary(summary: Dict[str, Any]) -> None:
    """Print console summary for dataset preparation."""
    print("\n" + "=" * 70)
    print("MEDSCAN ASSIST - DATASET PREPARATION SUMMARY")
    print("=" * 70)
    print(f"Processed Directory: {summary['processed_dir']}")
    print(f"Stratified Resplit:  {summary['resplit_performed']} (Seed: {summary['seed']})")
    print("-" * 70)
    print(f"{'Split':<10} | {'NORMAL':<10} | {'PNEUMONIA':<10} | {'Total':<10}")
    print("-" * 70)
    for split, counts in summary["counts"].items():
        n = counts.get("NORMAL", 0)
        p = counts.get("PNEUMONIA", 0)
        t = summary["total_images"][split]
        print(f"{split:<10} | {n:<10} | {p:<10} | {t:<10}")
    print("-" * 70)
    print(f"Computed Class Weights: {summary['computed_class_weights']}")
    print("=" * 70 + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser for prepare_dataset."""
    parser = argparse.ArgumentParser(description="Prepare and balance MedScan Assist dataset splits.")
    parser.add_argument(
        "--raw-data-dir",
        type=str,
        default=Config.DATA_DIR,
        help=f"Path to raw chest X-ray dataset directory (default: {Config.DATA_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(Config.BASE_DIR, "data", "processed"),
        help="Destination directory for processed splits (default: data/processed)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Proportion of training data to re-split into validation (default: 0.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=Config.SEED,
        help=f"Random seed for reproducibility (default: {Config.SEED})",
    )
    parser.add_argument(
        "--force-resplit",
        action="store_true",
        help="Force 15% stratified re-split even if raw val set has >= 100 images.",
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Use symlinks instead of copying images if supported by OS.",
    )
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        prepare_dataset(
            raw_data_dir=args.raw_data_dir,
            output_dir=args.output_dir,
            val_ratio=args.val_ratio,
            seed=args.seed,
            force_resplit=args.force_resplit,
            symlink=args.symlink,
        )
        sys.exit(0)
    except Exception as exc:
        logger.exception("Failed to prepare dataset: %s", exc)
        sys.exit(1)
