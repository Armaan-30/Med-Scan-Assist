"""MedScan Assist - Model Architecture Definition.

Defines the EfficientNet-B0 convolutional neural network architecture tailored
for Chest X-Ray classification (Normal vs Pneumonia), including custom
classification head, feature extractor freezing/unfreezing, Grad-CAM layer hooks,
and checkpoint serialization utilities.
"""

from datetime import datetime, timezone
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from config.config import Config


def build_efficientnet_b0(
    num_classes: int = Config.NUM_CLASSES,
    pretrained: bool = True,
    freeze_backbone_layers: bool = True,
    dropout_rate: float = Config.DROPOUT_RATE,
) -> nn.Module:
    """Build and configure the EfficientNet-B0 model for Chest X-Ray classification.

    Replaces the default 1000-class ImageNet head with a custom Dropout and Linear
    layer for binary classification.

    Args:
        num_classes: Number of target output classes (default: 2).
        pretrained: Whether to load ImageNet pre-trained weights.
        freeze_backbone_layers: If True, freezes the feature extractor parameters.
        dropout_rate: Dropout probability applied before the final linear layer.

    Returns:
        nn.Module: Configured PyTorch EfficientNet-B0 model.
    """
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b0(weights=weights)

    if freeze_backbone_layers:
        freeze_backbone(model)

    # In EfficientNet-B0, model.features outputs 1280 channels
    in_features: int = model.classifier[1].in_features  # 1280

    # Custom classifier head: Dropout(p) -> Linear(1280, num_classes)
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate, inplace=True),
        nn.Linear(in_features=in_features, out_features=num_classes),
    )

    return model


def freeze_backbone(model: nn.Module) -> None:
    """Freeze all feature extraction layers in the model backbone.

    Args:
        model: EfficientNet model instance.
    """
    if hasattr(model, "features"):
        for param in model.features.parameters():
            param.requires_grad = False


def unfreeze_backbone(model: nn.Module, num_blocks_to_unfreeze: Optional[int] = None) -> None:
    """Unfreeze layers in the model backbone for fine-tuning.

    Args:
        model: EfficientNet model instance.
        num_blocks_to_unfreeze: Number of top feature blocks to unfreeze from the end.
            If None, unfreezes the entire backbone.
    """
    if not hasattr(model, "features"):
        return

    if num_blocks_to_unfreeze is None:
        for param in model.features.parameters():
            param.requires_grad = True
    else:
        # Freeze everything first
        for param in model.features.parameters():
            param.requires_grad = False

        # Unfreeze the last N children in features
        feature_blocks = list(model.features.children())
        blocks_to_activate = feature_blocks[-num_blocks_to_unfreeze:]
        for block in blocks_to_activate:
            for param in block.parameters():
                param.requires_grad = True


def get_target_gradcam_layer(model: nn.Module) -> nn.Module:
    """Retrieve the target convolutional layer for Grad-CAM explainability.

    For EfficientNet-B0, this corresponds to the final convolutional block
    in `model.features[-1]`.

    Args:
        model: EfficientNet model instance.

    Returns:
        nn.Module: The target layer to compute activation maps and gradients.

    Raises:
        AttributeError: If `features` attribute is missing on the model.
    """
    if hasattr(model, "features"):
        return model.features[-1]
    raise AttributeError("Model does not have a 'features' attribute for Grad-CAM targeting.")


def save_checkpoint(
    model: nn.Module,
    file_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    metrics: Optional[Dict[str, float]] = None,
    class_names: Optional[List[str]] = None,
    class_to_idx: Optional[Dict[str, int]] = None,
    config_dict: Optional[Dict[str, Any]] = None,
) -> str:
    """Serialize and save model weights, architecture metadata, and training state.

    Args:
        model: PyTorch model to save.
        file_path: Destination path for checkpoint file (.pth).
        optimizer: Optional optimizer state to preserve.
        epoch: Current epoch number.
        metrics: Optional validation/training metrics dictionary.
        class_names: List of class labels (e.g. ['NORMAL', 'PNEUMONIA']).
        class_to_idx: Mapping from class names to index.
        config_dict: Dictionary of configuration parameters.

    Returns:
        str: Absolute path of saved checkpoint.
    """
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

    checkpoint_payload: Dict[str, Any] = {
        "architecture": Config.MODEL_NAME,
        "model_state_dict": model.state_dict(),
        "class_names": class_names or Config.CLASS_NAMES,
        "class_to_idx": class_to_idx or Config.CLASS_TO_IDX,
        "epoch": epoch,
        "metrics": metrics or {},
        "config": config_dict or {
            "image_size": Config.IMAGE_SIZE,
            "dropout_rate": Config.DROPOUT_RATE,
            "confidence_threshold": Config.CONFIDENCE_THRESHOLD,
        },
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "disclaimer": Config.RESEARCH_DISCLAIMER,
    }

    if optimizer is not None:
        checkpoint_payload["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(checkpoint_payload, file_path)
    return os.path.abspath(file_path)


def load_checkpoint_weights(
    file_path: str,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Load and instantiate a model from a saved checkpoint file.

    Args:
        file_path: Path to checkpoint file.
        device: Device to map model parameters onto. Defaults to Config.DEVICE.

    Returns:
        Tuple[nn.Module, Dict[str, Any]]: (Instantiated model in eval mode, checkpoint metadata dict)

    Raises:
        FileNotFoundError: If checkpoint file does not exist.
        KeyError: If required keys are missing in the checkpoint.
    """
    target_device = device or Config.DEVICE

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Checkpoint file not found: {file_path}")

    checkpoint = torch.load(file_path, map_location=target_device)

    required_keys = {"model_state_dict", "class_names"}
    missing_keys = required_keys - set(checkpoint.keys())
    if missing_keys:
        raise KeyError(f"Invalid checkpoint. Missing required keys: {missing_keys}")

    class_names = checkpoint.get("class_names", Config.CLASS_NAMES)
    num_classes = len(class_names)

    dropout_rate = checkpoint.get("config", {}).get("dropout_rate", Config.DROPOUT_RATE)
    model = build_efficientnet_b0(
        num_classes=num_classes,
        pretrained=False,
        freeze_backbone_layers=False,
        dropout_rate=dropout_rate,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(target_device)
    model.eval()

    return model, checkpoint
