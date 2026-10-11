"""MedScan Assist - Test Set Model Evaluation Script.

Strictly evaluates the trained model on the held-out test split.
Calculates Accuracy, Precision, Recall, Specificity, F1-Score, and ROC-AUC.
Generates publication-quality Confusion Matrix, ROC curve, and Precision-Recall
curve plots, along with a detailed clinical test metrics report containing
research limitations and class distributions.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import Config
from models.model_architecture import load_checkpoint_weights
from training.prepare_dataset import ChestXrayDataset, get_eval_transforms

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("evaluate")


@torch.no_grad()
def run_evaluation(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run model across evaluation dataloader and collect predictions.

    Returns:
        (y_true, y_pred, y_prob) where y_prob is positive class (PNEUMONIA) probability.
    """
    model.eval()
    all_targets: List[int] = []
    all_preds: List[int] = []
    all_probs: List[float] = []

    for images, targets in tqdm(dataloader, desc="Evaluating on Test Set"):
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)

        preds = torch.argmax(probs, dim=1)
        # Class 1 is PNEUMONIA
        pneumonia_probs = probs[:, 1]

        all_targets.extend(targets.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_probs.extend(pneumonia_probs.cpu().numpy().tolist())

    return np.array(all_targets), np.array(all_preds), np.array(all_probs)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: Path,
) -> None:
    """Plot and save annotated confusion matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
        ax=ax,
        annot_kws={"size": 14, "weight": "bold"},
    )
    ax.set_title("Test Confusion Matrix (Research Use Only)", fontsize=12, pad=12)
    ax.set_xlabel("Predicted Class", fontsize=11)
    ax.set_ylabel("True Class", fontsize=11)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix saved to: %s", output_path)


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    roc_auc: float,
    output_path: Path,
) -> None:
    """Plot and save Receiver Operating Characteristic (ROC) curve."""
    fig, ax = plt.subplots(figsize=(6, 5))
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ax.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    except Exception as e:
        logger.warning("Could not compute full ROC curve: %s", e)
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--")

    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    ax.set_title("ROC Curve - Pneumonia Detection", fontsize=12)
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("ROC curve plot saved to: %s", output_path)


def plot_pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    pr_auc: float,
    output_path: Path,
) -> None:
    """Plot and save Precision-Recall curve."""
    fig, ax = plt.subplots(figsize=(6, 5))
    try:
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ax.plot(rec, prec, color="#2ca02c", lw=2, label=f"PR (AP = {pr_auc:.4f})")
    except Exception as e:
        logger.warning("Could not compute PR curve: %s", e)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall (Sensitivity)", fontsize=11)
    ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=11)
    ax.set_title("Precision-Recall Curve", fontsize=12)
    ax.legend(loc="lower left")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Precision-Recall curve plot saved to: %s", output_path)


def evaluate_test_set(
    checkpoint_path: Path,
    test_dir: Path,
    output_dir: Path,
    batch_size: int = Config.BATCH_SIZE,
) -> Dict[str, Any]:
    """Execute complete held-out evaluation on the test set."""
    output_dir.mkdir(parents=True, exist_ok=True)
    device = Config.DEVICE
    logger.info("Loading checkpoint for evaluation: %s (Device: %s)", checkpoint_path, device)

    # 1. Load model checkpoint
    model, checkpoint_meta = load_checkpoint_weights(str(checkpoint_path), device=device)
    model.eval()

    # 2. Setup held-out test dataset
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Test split directory not found: {test_dir}")

    test_dataset = ChestXrayDataset(data_dir=test_dir, transform=get_eval_transforms(Config.IMAGE_SIZE))
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    total_test_images = len(test_dataset)
    class_distribution = {
        cls_name: int(np.sum(np.array(test_dataset.targets) == idx))
        for cls_name, idx in Config.CLASS_TO_IDX.items()
    }
    logger.info("Test set size: %d images. Distribution: %s", total_test_images, class_distribution)

    # 3. Generate Predictions
    y_true, y_pred, y_prob = run_evaluation(model, test_loader, device)

    # 4. Compute Statistical Metrics
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    # Confusion Matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    else:
        tn = fp = fn = tp = 0
        specificity = 0.0

    # ROC-AUC and PR-AUC
    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = 0.0

    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except Exception:
        pr_auc = 0.0

    # Per-class classification report
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=Config.CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    # 5. Generate Plots
    cm_plot_path = output_dir / "confusion_matrix.png"
    roc_plot_path = output_dir / "roc_curve.png"
    pr_plot_path = output_dir / "precision_recall_curve.png"

    plot_confusion_matrix(cm, Config.CLASS_NAMES, cm_plot_path)
    plot_roc_curve(y_true, y_prob, roc_auc, roc_plot_path)
    plot_pr_curve(y_true, y_prob, pr_auc, pr_plot_path)

    # 6. Assemble Report
    metrics_report: Dict[str, Any] = {
        "disclaimer": Config.RESEARCH_DISCLAIMER,
        "checkpoint_evaluated": str(checkpoint_path.resolve()),
        "test_dataset_path": str(test_dir.resolve()),
        "test_set_size": total_test_images,
        "class_distribution": class_distribution,
        "overall_metrics": {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall_sensitivity": round(rec, 4),
            "specificity": round(specificity, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
        },
        "confusion_matrix": {
            "true_negatives_normal": int(tn),
            "false_positives_pneumonia": int(fp),
            "false_negatives_normal": int(fn),
            "true_positives_pneumonia": int(tp),
            "raw_matrix": cm.tolist(),
        },
        "classification_report": report_dict,
        "limitations": [
            "Evaluated strictly on held-out frontal chest radiographs; test set was NEVER used for tuning or early stopping.",
            "Sensitivity and specificity are subject to clinical variations across pediatric vs adult patient cohorts.",
            "Performance may degrade on portable (AP) bedside X-rays, motion artifacts, or severe underlying comorbidities.",
            "This model operates as an assistive triaging tool and is NOT a standalone clinical diagnostic device.",
        ],
        "plots": {
            "confusion_matrix": str(cm_plot_path),
            "roc_curve": str(roc_plot_path),
            "pr_curve": str(pr_plot_path),
        },
    }

    metrics_json_path = output_dir / "test_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_report, f, indent=2)

    logger.info("Evaluation report saved to: %s", metrics_json_path)
    _print_evaluation_results(metrics_report)

    return metrics_report


def _print_evaluation_results(report: Dict[str, Any]) -> None:
    """Print clean terminal results of model evaluation."""
    m = report["overall_metrics"]
    cm = report["confusion_matrix"]

    print("\n" + "=" * 70)
    print("MEDSCAN ASSIST - HELD-OUT TEST EVALUATION RESULTS")
    print("=" * 70)
    print(f"Test Set Size:       {report['test_set_size']} images")
    print(f"Class Distribution:  {report['class_distribution']}")
    print("-" * 70)
    print(f"Accuracy:            {m['accuracy'] * 100:.2f}%")
    print(f"Sensitivity/Recall:  {m['recall_sensitivity'] * 100:.2f}%")
    print(f"Specificity:         {m['specificity'] * 100:.2f}%")
    print(f"Precision:           {m['precision'] * 100:.2f}%")
    print(f"F1-Score:            {m['f1_score']:.4f}")
    print(f"ROC-AUC:             {m['roc_auc']:.4f}")
    print(f"PR-AUC:              {m['pr_auc']:.4f}")
    print("-" * 70)
    print("Confusion Matrix Breakdown:")
    print(f"  True Negatives (Normal correctly identified):     {cm['true_negatives_normal']}")
    print(f"  False Positives (Normal miscalled Pneumonia):     {cm['false_positives_pneumonia']}")
    print(f"  False Negatives (Pneumonia missed):               {cm['false_negatives_normal']}")
    print(f"  True Positives (Pneumonia correctly identified):  {cm['true_positives_pneumonia']}")
    print("=" * 70 + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct argument parser for evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate MedScan Assist model on held-out test split.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=Config.BEST_MODEL_PATH,
        help=f"Path to model checkpoint .pth (default: {Config.BEST_MODEL_PATH})",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.path.join(Config.BASE_DIR, "data", "processed"),
        help="Path to data directory containing test/ split (default: data/processed)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(Config.REPORTS_DIR, "evaluation"),
        help="Directory to save evaluation metrics and plots (default: reports/evaluation)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=Config.BATCH_SIZE,
        help=f"Evaluation batch size (default: {Config.BATCH_SIZE})",
    )
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()

    checkpoint_file = Path(args.checkpoint)
    data_dir_p = Path(args.data_dir)
    test_split_p = data_dir_p / "test"

    # Fallback to raw dataset test split if processed/test is not found
    if not test_split_p.exists():
        raw_test = Path(Config.TEST_DIR)
        if raw_test.exists():
            logger.warning("Test split not found in %s. Using Config.TEST_DIR (%s)", test_split_p, raw_test)
            test_split_p = raw_test

    try:
        evaluate_test_set(
            checkpoint_path=checkpoint_file,
            test_dir=test_split_p,
            output_dir=Path(args.output_dir),
            batch_size=args.batch_size,
        )
        sys.exit(0)
    except Exception as exc:
        logger.exception("Evaluation failed: %s", exc)
        sys.exit(1)
