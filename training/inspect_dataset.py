"""MedScan Assist - Dataset Inspection Script.

Scans the chest X-ray dataset directory structure, validates splits, counts
images per class, detects corrupted/unreadable files, computes class imbalance,
identifies potential duplicates via MD5 hashing, visualizes sample images,
and outputs a comprehensive inspection report with medical research disclaimers.
"""

import argparse
import hashlib
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError

# Ensure project root is in sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("inspect_dataset")


def compute_file_md5(file_path: Path, block_size: int = 65536) -> str:
    """Compute MD5 checksum for a file to detect duplicates."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def check_image_readability(file_path: Path) -> Tuple[bool, Optional[str], Optional[Tuple[int, int]], Optional[str]]:
    """Check if an image file can be read and verified with PIL.

    Returns:
        (is_valid, error_message, image_size, image_mode)
    """
    try:
        with Image.open(file_path) as img:
            img.verify()
        # verify() closes the file in some PIL versions; reopen to check size/mode
        with Image.open(file_path) as img:
            size = img.size
            mode = img.mode
        return True, None, size, mode
    except (UnidentifiedImageError, OSError, Exception) as exc:
        return False, str(exc), None, None


def inspect_dataset(
    data_dir: Path,
    output_dir: Path,
    min_val_threshold: int = 100,
    save_samples_count: int = 4,
) -> Dict[str, Any]:
    """Scan and analyze dataset structure and integrity.

    Args:
        data_dir: Root dataset directory containing train, val, test splits.
        output_dir: Directory to save inspection reports and sample plots.
        min_val_threshold: Minimum sample count recommended for validation split.
        save_samples_count: Number of sample images per class/split to plot.

    Returns:
        Dictionary containing comprehensive inspection results.
    """
    logger.info("Starting dataset inspection on: %s", data_dir)
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists() or not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset root directory does not exist or is not a directory: {data_dir}")

    expected_splits = ["train", "val", "test"]
    expected_classes = Config.CLASS_NAMES  # ["NORMAL", "PNEUMONIA"]

    found_splits = [s for s in expected_splits if (data_dir / s).is_dir()]
    missing_splits = [s for s in expected_splits if s not in found_splits]

    split_counts: Dict[str, Dict[str, int]] = {s: defaultdict(int) for s in expected_splits}
    corrupted_files: List[Dict[str, str]] = []
    hash_to_paths: Dict[str, List[str]] = defaultdict(list)
    image_specs: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    sample_images: Dict[str, Dict[str, List[Path]]] = {
        s: {c: [] for c in expected_classes} for s in expected_splits
    }

    total_valid_images = 0
    total_scanned_images = 0

    valid_extensions = {ext.lower() for ext in Config.ALLOWED_EXTENSIONS}

    # Iterate over splits
    for split in expected_splits:
        split_path = data_dir / split
        if not split_path.is_dir():
            logger.warning("Split directory missing: %s", split_path)
            continue

        for cls_name in expected_classes:
            class_path = split_path / cls_name
            if not class_path.is_dir():
                logger.warning("Class directory missing: %s", class_path)
                continue

            for root, _, files in os.walk(class_path):
                for file_name in sorted(files):
                    file_path = Path(root) / file_name
                    if file_path.suffix.lower() not in valid_extensions:
                        continue

                    total_scanned_images += 1
                    is_readable, err_msg, img_size, img_mode = check_image_readability(file_path)

                    if not is_readable:
                        logger.error("Corrupted image found: %s (%s)", file_path, err_msg)
                        corrupted_files.append({
                            "path": str(file_path),
                            "split": split,
                            "class": cls_name,
                            "error": err_msg or "Unknown error",
                        })
                        continue

                    total_valid_images += 1
                    split_counts[split][cls_name] += 1
                    image_specs[f"{split}/{cls_name}"].append(img_size)

                    # Compute MD5 for duplicate detection
                    file_hash = compute_file_md5(file_path)
                    hash_to_paths[file_hash].append(str(file_path))

                    # Collect samples
                    if len(sample_images[split][cls_name]) < save_samples_count:
                        sample_images[split][cls_name].append(file_path)

    # Detect duplicates
    duplicates = {h: paths for h, paths in hash_to_paths.items() if len(paths) > 1}
    num_duplicate_groups = len(duplicates)
    num_duplicate_images = sum(len(paths) - 1 for paths in duplicates.values())

    # Calculate ratios and totals per split
    split_summaries: Dict[str, Any] = {}
    recommendations: List[str] = []

    for split in expected_splits:
        counts = dict(split_counts[split])
        split_total = sum(counts.values())
        normal_cnt = counts.get("NORMAL", 0)
        pneu_cnt = counts.get("PNEUMONIA", 0)

        imbalance_ratio = None
        if normal_cnt > 0 and pneu_cnt > 0:
            imbalance_ratio = round(pneu_cnt / normal_cnt, 3)

        split_summaries[split] = {
            "total_images": split_total,
            "counts_by_class": counts,
            "imbalance_ratio_pneumonia_to_normal": imbalance_ratio,
        }

        # Check validation threshold
        if split == "val":
            if split_total < min_val_threshold:
                rec = (
                    f"Validation split has only {split_total} images (< {min_val_threshold}). "
                    f"Recommend re-splitting training data (e.g. 15% stratified) using prepare_dataset.py."
                )
                logger.warning(rec)
                recommendations.append(rec)
            split_summaries[split]["too_few_samples_warning"] = split_total < min_val_threshold

    if missing_splits:
        recommendations.append(f"Missing expected splits: {missing_splits}. Verify dataset extraction.")
    if corrupted_files:
        recommendations.append(f"Found {len(corrupted_files)} corrupted files that should be cleaned or excluded.")
    if num_duplicate_groups > 0:
        recommendations.append(
            f"Found {num_duplicate_images} duplicate image instances across {num_duplicate_groups} distinct files."
        )

    # Generate sample visualization figure
    samples_plot_path = output_dir / "dataset_samples.png"
    _generate_sample_grid(sample_images, samples_plot_path)

    # Prepare JSON report
    report: Dict[str, Any] = {
        "disclaimer": Config.RESEARCH_DISCLAIMER,
        "dataset_root": str(data_dir.resolve()),
        "scanned_images_total": total_scanned_images,
        "valid_images_total": total_valid_images,
        "corrupted_images_count": len(corrupted_files),
        "corrupted_files": corrupted_files,
        "splits_summary": split_summaries,
        "duplicate_groups_count": num_duplicate_groups,
        "duplicate_extra_images_count": num_duplicate_images,
        "sample_duplicates": {k: duplicates[k] for k in list(duplicates.keys())[:10]},
        "recommendations": recommendations,
        "samples_plot_saved_to": str(samples_plot_path),
    }

    report_path = output_dir / "dataset_inspection.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Dataset inspection report saved to: %s", report_path)
    _print_console_summary(report)

    return report


def _generate_sample_grid(sample_images: Dict[str, Dict[str, List[Path]]], save_path: Path) -> None:
    """Save a grid of sample chest X-rays across splits and classes."""
    splits = ["train", "val", "test"]
    classes = ["NORMAL", "PNEUMONIA"]

    all_samples: List[Tuple[str, str, Path]] = []
    for split in splits:
        for cls_name in classes:
            paths = sample_images.get(split, {}).get(cls_name, [])
            for p in paths[:2]:  # 2 samples per combination = up to 12 images
                all_samples.append((split, cls_name, p))

    if not all_samples:
        logger.warning("No sample images available to plot.")
        return

    n_samples = len(all_samples)
    cols = 4
    rows = (n_samples + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    for idx, ax in enumerate(axes):
        if idx < n_samples:
            split, cls_name, img_path = all_samples[idx]
            try:
                img = Image.open(img_path).convert("L")  # grayscale display
                ax.imshow(img, cmap="gray")
                ax.set_title(f"[{split.upper()}] {cls_name}\n{img_path.name}", fontsize=9)
            except Exception as e:
                ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center")
            ax.axis("off")
        else:
            ax.axis("off")

    plt.suptitle("MedScan Assist - Dataset Sample Images (Research Use Only)", fontsize=14, y=0.98)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Sample images saved to: %s", save_path)


def _print_console_summary(report: Dict[str, Any]) -> None:
    """Print an aesthetic, formatted console summary of the dataset inspection."""
    print("\n" + "=" * 70)
    print("MEDSCAN ASSIST - DATASET INSPECTION SUMMARY")
    print("=" * 70)
    print(f"Dataset Root:    {report['dataset_root']}")
    print(f"Total Scanned:   {report['scanned_images_total']}")
    print(f"Total Valid:     {report['valid_images_total']}")
    print(f"Corrupted Files: {report['corrupted_images_count']}")
    print(f"Duplicate Files: {report['duplicate_extra_images_count']} extra files across {report['duplicate_groups_count']} groups")
    print("-" * 70)
    print(f"{'Split':<10} | {'NORMAL':<10} | {'PNEUMONIA':<10} | {'Total':<10} | {'Ratio (P:N)':<12}")
    print("-" * 70)

    for split, data in report["splits_summary"].items():
        counts = data["counts_by_class"]
        normal = counts.get("NORMAL", 0)
        pneu = counts.get("PNEUMONIA", 0)
        tot = data["total_images"]
        ratio = data["imbalance_ratio_pneumonia_to_normal"]
        ratio_str = f"{ratio:.2f} : 1" if ratio is not None else "N/A"
        print(f"{split:<10} | {normal:<10} | {pneu:<10} | {tot:<10} | {ratio_str:<12}")

    print("-" * 70)
    if report["recommendations"]:
        print("RECOMMENDATIONS:")
        for idx, rec in enumerate(report["recommendations"], 1):
            print(f"  {idx}. {rec}")
    else:
        print("Status: Dataset splits and files are well-balanced and valid.")
    print("=" * 70 + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct argument parser for CLI execution."""
    parser = argparse.ArgumentParser(
        description="Inspect and validate Chest X-ray dataset for MedScan Assist."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=Config.DATA_DIR,
        help=f"Path to dataset directory containing train/val/test splits (default: {Config.DATA_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(Config.REPORTS_DIR, "evaluation"),
        help="Path to output directory for inspection JSON and sample plots.",
    )
    parser.add_argument(
        "--min-val",
        type=int,
        default=100,
        help="Minimum sample count threshold for validation split (default: 100).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=4,
        help="Number of sample images per class to capture (default: 4).",
    )
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()

    target_data_dir = Path(args.data_dir)
    target_output_dir = Path(args.output_dir)

    try:
        report = inspect_dataset(
            data_dir=target_data_dir,
            output_dir=target_output_dir,
            min_val_threshold=args.min_val,
            save_samples_count=args.samples,
        )
        if report.get("corrupted_images_count", 0) > 0:
            logger.warning("Dataset inspection completed with warnings regarding corrupted images.")
        sys.exit(0)
    except FileNotFoundError as fnf:
        logger.error("Dataset inspection aborted: %s", fnf)
        sys.exit(1)
    except Exception as exc:
        logger.exception("Unexpected error during dataset inspection: %s", exc)
        sys.exit(2)
