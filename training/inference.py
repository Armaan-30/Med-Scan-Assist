"""MedScan Assist - Inference Utilities Module.

Provides single-image and batch inference capabilities for Chest X-Ray
classification using the trained EfficientNet-B0 model. Implements clinical
confidence thresholding, abstention/needs_review triage flagging, consistent
eval preprocessing, and structured diagnostic result output.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import Config
from models.model_architecture import load_checkpoint_weights
from training.prepare_dataset import get_eval_transforms

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("inference")

# Cached model instance for efficiency across inference calls
_GLOBAL_MODEL: Optional[torch.nn.Module] = None
_GLOBAL_CHECKPOINT_PATH: Optional[str] = None


def get_inference_model(
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: Optional[torch.device] = None,
) -> torch.nn.Module:
    """Retrieve or lazily initialize the cached inference model."""
    global _GLOBAL_MODEL, _GLOBAL_CHECKPOINT_PATH

    target_checkpoint = str(checkpoint_path or Config.BEST_MODEL_PATH)
    target_device = device or Config.DEVICE

    if _GLOBAL_MODEL is not None and _GLOBAL_CHECKPOINT_PATH == target_checkpoint:
        return _GLOBAL_MODEL

    logger.info("Initializing inference model from: %s on device: %s", target_checkpoint, target_device)
    model, _ = load_checkpoint_weights(target_checkpoint, device=target_device)
    model.eval()

    _GLOBAL_MODEL = model
    _GLOBAL_CHECKPOINT_PATH = target_checkpoint
    return _GLOBAL_MODEL


def preprocess_image(
    image_input: Union[str, Path, Image.Image],
    image_size: int = Config.IMAGE_SIZE,
) -> Tuple[torch.Tensor, Optional[str]]:
    """Load and preprocess a single image for model ingestion.

    Returns:
        (preprocessed_tensor, source_path_str)
    """
    eval_transform = get_eval_transforms(image_size)
    source_path: Optional[str] = None

    if isinstance(image_input, (str, Path)):
        source_path = str(Path(image_input).resolve())
        with Image.open(source_path) as img:
            img_rgb = img.convert("RGB")
            tensor = eval_transform(img_rgb)
    elif isinstance(image_input, Image.Image):
        img_rgb = image_input.convert("RGB")
        tensor = eval_transform(img_rgb)
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}. Expected str, Path, or PIL.Image.")

    return tensor, source_path


def predict_image(
    image_input: Union[str, Path, Image.Image],
    model: Optional[torch.nn.Module] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: Optional[torch.device] = None,
    threshold: float = Config.CONFIDENCE_THRESHOLD,
) -> Dict[str, Any]:
    """Execute prediction on a single Chest X-Ray image.

    Args:
        image_input: File path or PIL Image object.
        model: Optional instantiated model. If None, loaded from checkpoint.
        checkpoint_path: Path to .pth checkpoint if model is None.
        device: Target compute device.
        threshold: Confidence threshold for flag abstention/clinical review.

    Returns:
        Structured result dict containing predicted class, confidence, scores,
        needs_review flag, raw logits, and research disclaimer.
    """
    target_device = device or Config.DEVICE
    if model is None:
        model = get_inference_model(checkpoint_path=checkpoint_path, device=target_device)

    model.eval()
    tensor, source_path = preprocess_image(image_input)
    tensor = tensor.unsqueeze(0).to(target_device)  # Add batch dim [1, 3, H, W]

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)

    probs_np = probs.cpu().numpy()
    logits_np = logits.squeeze(0).cpu().numpy()

    top_idx = int(np.argmax(probs_np))
    predicted_class = Config.CLASS_NAMES[top_idx]
    confidence = float(probs_np[top_idx])

    scores = {
        cls_name: round(float(probs_np[idx]), 4)
        for idx, cls_name in enumerate(Config.CLASS_NAMES)
    }

    # Abstention / Review logic: if maximum confidence falls below threshold
    needs_review = bool(confidence < threshold)

    return {
        "disclaimer": Config.RESEARCH_DISCLAIMER,
        "image_path": source_path,
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
        "scores": scores,
        "needs_review": needs_review,
        "threshold_applied": threshold,
        "raw_outputs": [round(float(x), 4) for x in logits_np.tolist()],
    }


def predict_batch(
    image_inputs: List[Union[str, Path, Image.Image]],
    model: Optional[torch.nn.Module] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: Optional[torch.device] = None,
    threshold: float = Config.CONFIDENCE_THRESHOLD,
    batch_size: int = 32,
) -> List[Dict[str, Any]]:
    """Execute efficient batched prediction over a list of image inputs.

    Args:
        image_inputs: List of file paths or PIL Image instances.
        model: Preloaded model instance or None.
        checkpoint_path: Path to checkpoint.
        device: Compute device.
        threshold: Confidence threshold.
        batch_size: Mini-batch chunk size.

    Returns:
        List of structured result dictionaries.
    """
    target_device = device or Config.DEVICE
    if model is None:
        model = get_inference_model(checkpoint_path=checkpoint_path, device=target_device)

    model.eval()
    results: List[Dict[str, Any]] = []

    for i in range(0, len(image_inputs), batch_size):
        chunk = image_inputs[i : i + batch_size]
        tensors = []
        paths = []

        for item in chunk:
            t, p = preprocess_image(item)
            tensors.append(t)
            paths.append(p)

        batch_tensor = torch.stack(tensors).to(target_device)

        with torch.no_grad():
            batch_logits = model(batch_tensor)
            batch_probs = F.softmax(batch_logits, dim=1)

        for j in range(len(chunk)):
            probs_np = batch_probs[j].cpu().numpy()
            logits_np = batch_logits[j].cpu().numpy()
            top_idx = int(np.argmax(probs_np))
            pred_cls = Config.CLASS_NAMES[top_idx]
            conf = float(probs_np[top_idx])

            scores = {
                cls_name: round(float(probs_np[idx]), 4)
                for idx, cls_name in enumerate(Config.CLASS_NAMES)
            }

            results.append({
                "disclaimer": Config.RESEARCH_DISCLAIMER,
                "image_path": paths[j],
                "predicted_class": pred_cls,
                "confidence": round(conf, 4),
                "scores": scores,
                "needs_review": bool(conf < threshold),
                "threshold_applied": threshold,
                "raw_outputs": [round(float(x), 4) for x in logits_np.tolist()],
            })

    return results


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser for inference."""
    parser = argparse.ArgumentParser(description="MedScan Assist - Chest X-Ray Model Inference.")
    parser.add_argument("--image", type=str, default="", help="Path to single image file for prediction.")
    parser.add_argument("--dir", type=str, default="", help="Path to directory of images for batch prediction.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=Config.BEST_MODEL_PATH,
        help=f"Path to model checkpoint (default: {Config.BEST_MODEL_PATH})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=Config.CONFIDENCE_THRESHOLD,
        help=f"Confidence threshold for triage review flag (default: {Config.CONFIDENCE_THRESHOLD})",
    )
    parser.add_argument("--output", type=str, default="", help="Optional path to save JSON inference output.")
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()

    if not args.image and not args.dir:
        parser.print_help()
        sys.exit(1)

    try:
        if args.image:
            res = predict_image(
                image_input=args.image,
                checkpoint_path=args.checkpoint,
                threshold=args.threshold,
            )
            print("\n" + "=" * 60)
            print("MEDSCAN ASSIST - INFERENCE RESULT")
            print("=" * 60)
            print(f"Image:            {res['image_path']}")
            print(f"Prediction:       {res['predicted_class']}")
            print(f"Confidence:       {res['confidence'] * 100:.2f}%")
            print(f"Needs Review:     {res['needs_review']} (threshold: {args.threshold})")
            print(f"Class Scores:     {res['scores']}")
            print("=" * 60 + "\n")
            output_payload = res

        elif args.dir:
            dir_p = Path(args.dir)
            valid_exts = {e.lower() for e in Config.ALLOWED_EXTENSIONS}
            image_files = [
                f for f in sorted(dir_p.iterdir())
                if f.is_file() and f.suffix.lower() in valid_exts
            ]
            logger.info("Found %d images in directory %s", len(image_files), dir_p)
            batch_res = predict_batch(
                image_inputs=image_files,
                checkpoint_path=args.checkpoint,
                threshold=args.threshold,
            )
            print("\n" + "=" * 70)
            print(f"MEDSCAN ASSIST - BATCH INFERENCE ({len(batch_res)} images)")
            print("=" * 70)
            for item in batch_res:
                flag = " [!] NEEDS REVIEW" if item["needs_review"] else ""
                print(f"{Path(item['image_path']).name:<30} -> {item['predicted_class']:<10} ({item['confidence']*100:.1f}%){flag}")
            print("=" * 70 + "\n")
            output_payload = batch_res

        if args.output:
            out_p = Path(args.output)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(output_payload, f, indent=2)
            logger.info("Inference output saved to %s", out_p)

        sys.exit(0)
    except Exception as exc:
        logger.exception("Inference failed: %s", exc)
        sys.exit(1)
