"""Image Preprocessing Module for MedScan Assist.

Provides data augmentations for model training and deterministic preprocessing
transforms for evaluation and real-time clinical inference.
"""

import logging
from typing import Optional, Union
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from config.config import Config

logger = logging.getLogger("medscan.preprocessing")


class ConvertToRGB:
    """Custom transform to convert any PIL image (e.g. grayscale 'L' or 'RGBA') to 3-channel RGB."""

    def __call__(self, img: Image.Image) -> Image.Image:
        if img.mode != "RGB":
            return img.convert("RGB")
        return img


def get_training_transforms(image_size: Optional[int] = None) -> transforms.Compose:
    """Return PyTorch transform pipeline for training with data augmentation.

    Augmentations are clinically conservative: slight rotation, horizontal flip,
    slight affine shear/scale, and brightness/contrast jitter.

    Args:
        image_size: Target image square dimension. Defaults to Config.IMAGE_SIZE (224).

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    target_size = image_size or getattr(Config, "IMAGE_SIZE", 224)
    mean = getattr(Config, "IMAGENET_MEAN", [0.485, 0.456, 0.406])
    std = getattr(Config, "IMAGENET_STD", [0.229, 0.224, 0.225])

    return transforms.Compose([
        ConvertToRGB(),
        transforms.Resize((int(target_size * 1.14), int(target_size * 1.14))),  # ~256 for 224
        transforms.RandomResizedCrop(target_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=(-10, 10)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_eval_transforms(image_size: Optional[int] = None) -> transforms.Compose:
    """Return deterministic PyTorch transform pipeline for validation and inference.

    Standard ImageNet protocol:
    1. Convert to RGB
    2. Resize short side to 256
    3. Center crop to 224
    4. Convert to tensor [0.0, 1.0]
    5. Normalize with ImageNet mean and std

    Args:
        image_size: Target square crop size. Defaults to Config.IMAGE_SIZE (224).

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    target_size = image_size or getattr(Config, "IMAGE_SIZE", 224)
    resize_dim = 256
    mean = getattr(Config, "IMAGENET_MEAN", [0.485, 0.456, 0.406])
    std = getattr(Config, "IMAGENET_STD", [0.229, 0.224, 0.225])

    return transforms.Compose([
        ConvertToRGB(),
        transforms.Resize(resize_dim),
        transforms.CenterCrop(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def validate_preprocessed_tensor(tensor: torch.Tensor, expected_size: int = 224) -> None:
    """Validate that the preprocessed tensor has correct shape and valid numerical ranges.

    Args:
        tensor: Tensor to check (expected shape [1, 3, H, W] or [3, H, W]).
        expected_size: Expected spatial dimension (height and width).

    Raises:
        ValueError: If shape is invalid, contains NaNs or Infs, or has unexpected dimensions.
    """
    if torch.isnan(tensor).any():
        raise ValueError("Preprocessed tensor contains NaN values.")
    if torch.isinf(tensor).any():
        raise ValueError("Preprocessed tensor contains infinite values.")

    # Check shape
    if tensor.dim() == 4:
        b, c, h, w = tensor.shape
        if b != 1 or c != 3 or h != expected_size or w != expected_size:
            raise ValueError(
                f"Invalid batch tensor shape: {tuple(tensor.shape)}. Expected (1, 3, {expected_size}, {expected_size})."
            )
    elif tensor.dim() == 3:
        c, h, w = tensor.shape
        if c != 3 or h != expected_size or w != expected_size:
            raise ValueError(
                f"Invalid 3D tensor shape: {tuple(tensor.shape)}. Expected (3, {expected_size}, {expected_size})."
            )
    else:
        raise ValueError(f"Unexpected tensor dimensionality: {tensor.dim()}D (shape: {tuple(tensor.shape)}).")


def preprocess_pil_image(
    image: Image.Image,
    device: Optional[Union[torch.device, str]] = None
) -> torch.Tensor:
    """Preprocess a PIL Image into a model-ready batch tensor [1, 3, 224, 224].

    Args:
        image: PIL Image object.
        device: PyTorch device (CPU or CUDA). Defaults to Config.DEVICE.

    Returns:
        torch.Tensor of shape [1, 3, 224, 224] on the specified device.
    """
    if device is None:
        device = getattr(Config, "DEVICE", torch.device("cpu"))

    target_size = getattr(Config, "IMAGE_SIZE", 224)
    eval_transforms = get_eval_transforms(image_size=target_size)

    tensor = eval_transforms(image)  # Shape: [3, 224, 224]
    batch_tensor = tensor.unsqueeze(0)  # Shape: [1, 3, 224, 224]

    validate_preprocessed_tensor(batch_tensor, expected_size=target_size)
    return batch_tensor.to(device)


def preprocess_for_inference(
    image_path: str,
    device: Optional[Union[torch.device, str]] = None
) -> torch.Tensor:
    """Load an image from disk and preprocess into a model-ready tensor [1, 3, 224, 224].

    Args:
        image_path: File path to the image.
        device: PyTorch device. Defaults to Config.DEVICE.

    Returns:
        torch.Tensor of shape [1, 3, 224, 224] on the target device.
    """
    try:
        with Image.open(image_path) as img:
            img.load()
            return preprocess_pil_image(img, device=device)
    except Exception as exc:
        logger.error("Failed to load and preprocess image at %s: %s", image_path, exc)
        raise
