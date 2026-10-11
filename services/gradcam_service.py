"""Grad-CAM Explainability Service for MedScan Assist.

Generates gradient-weighted class activation mapping (Grad-CAM) visualizations
for chest X-ray predictions, highlighting feature regions influencing the model's
classification decision.

DISCLAIMER: Grad-CAM visualizes regions that influenced the model output.
It does NOT prove disease presence, etiology, or histological pathology.
"""

import logging
import os
import uuid
from typing import Any, List, Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    _CV2_AVAILABLE = False

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    _GRADCAM_LIB_AVAILABLE = True
except ImportError:
    GradCAM = None
    ClassifierOutputTarget = None
    _GRADCAM_LIB_AVAILABLE = False

from config.config import Config

logger = logging.getLogger("medscan.gradcam")

GRADCAM_DISCLAIMER = (
    "Grad-CAM shows regions that influenced the model output. It does not prove disease presence."
)


def get_default_target_layer(model: nn.Module) -> List[nn.Module]:
    """Identify the target convolutional layer for Grad-CAM.

    Prioritizes:
    1. model.features[-1] (standard EfficientNet / ConvNeXt / VGG)
    2. model.backbone.features[-1] (if wrapped)
    3. Last Conv2d layer found across model modules
    """
    if hasattr(model, "features") and isinstance(model.features, (nn.Sequential, nn.ModuleList)) and len(model.features) > 0:
        return [model.features[-1]]

    if hasattr(model, "backbone"):
        backbone = getattr(model, "backbone")
        if hasattr(backbone, "features") and len(backbone.features) > 0:
            return [backbone.features[-1]]

    # Search for the final Conv2d module in reverse order
    conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    if conv_layers:
        return [conv_layers[-1]]

    raise ValueError("Unable to determine target convolutional layer for Grad-CAM on model.")


def _apply_jet_colormap(grayscale_cam: np.ndarray) -> np.ndarray:
    """Map a 2D float array in [0, 1] to an RGB heatmap using JET colormap."""
    if _CV2_AVAILABLE and cv2 is not None:
        cam_uint8 = np.uint8(255 * grayscale_cam)
        heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        return heatmap.astype(np.float32) / 255.0

    # Fallback Jet colormap implementation using pure numpy
    x = np.clip(grayscale_cam, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def overlay_cam_on_image(
    original_rgb_norm: np.ndarray,
    cam_mask: np.ndarray,
    alpha: float = 0.5
) -> np.ndarray:
    """Superimpose 2D heatmap mask on a [0, 1] RGB image.

    Args:
        original_rgb_norm: RGB float array in [0, 1] of shape (H, W, 3).
        cam_mask: Float array in [0, 1] of shape (H, W).
        alpha: Blending weight for heatmap.

    Returns:
        Blended RGB uint8 image array of shape (H, W, 3).
    """
    heatmap = _apply_jet_colormap(cam_mask)
    overlay = alpha * heatmap + (1.0 - alpha) * original_rgb_norm
    overlay = np.clip(overlay, 0.0, 1.0)
    return np.uint8(255 * overlay)


def create_side_by_side_image(
    original_img: Image.Image,
    overlay_img: Image.Image,
    predicted_label: Optional[str] = None,
    confidence: Optional[float] = None
) -> Image.Image:
    """Create a polished side-by-side comparison image with banners and clinical disclaimer."""
    target_dim = 400
    orig_resized = original_img.convert("RGB").resize((target_dim, target_dim), Image.Resampling.BILINEAR)
    over_resized = overlay_img.convert("RGB").resize((target_dim, target_dim), Image.Resampling.BILINEAR)

    header_height = 50
    footer_height = 40
    padding = 10
    total_width = target_dim * 2 + padding * 3
    total_height = header_height + target_dim + footer_height + padding * 2

    canvas = Image.new("RGB", (total_width, total_height), color=(24, 28, 36))
    draw = ImageDraw.Draw(canvas)

    # Paste images
    y_img = header_height + padding
    canvas.paste(orig_resized, (padding, y_img))
    canvas.paste(over_resized, (target_dim + padding * 2, y_img))

    # Header labels
    header_orig = "Original Chest Radiograph"
    header_overlay = "Grad-CAM Activation Saliency"
    if predicted_label:
        conf_str = f" ({confidence*100:.1f}%)" if confidence is not None else ""
        header_overlay += f" [{predicted_label}{conf_str}]"

    draw.text((padding + 10, 18), header_orig, fill=(220, 225, 235))
    draw.text((target_dim + padding * 2 + 10, 18), header_overlay, fill=(245, 166, 35))

    # Footer disclaimer
    disclaimer_text = f"Notice: {GRADCAM_DISCLAIMER}"
    y_footer = total_height - footer_height + 5
    draw.text((padding + 10, y_footer), disclaimer_text, fill=(180, 185, 195))

    return canvas


def get_gradcam_visualization(
    model: nn.Module,
    input_tensor: torch.Tensor,
    original_image: Union[str, Image.Image],
    target_class: Optional[int] = None,
    target_layers: Optional[List[nn.Module]] = None,
    save_path: Optional[str] = None
) -> Tuple[Image.Image, np.ndarray, Image.Image, str]:
    """Generate Grad-CAM visualization for a model prediction.

    Args:
        model: PyTorch model (eval mode recommended).
        input_tensor: Preprocessed tensor of shape [1, 3, 224, 224].
        original_image: File path or PIL Image object.
        target_class: Target class index. If None, highest scoring class is used.
        target_layers: Model layers to compute gradients against. Defaults to last conv layer.
        save_path: Optional file path to save the generated visualization.

    Returns:
        Tuple containing:
            - heatmap_overlay: PIL.Image of original blended with heatmap
            - raw_heatmap: np.ndarray 2D float in [0, 1]
            - combined_visualization: PIL.Image side-by-side comparison with disclaimers
            - saved_path: Path where visualization was stored on disk
    """
    model.eval()

    # Load original image as PIL
    if isinstance(original_image, str):
        pil_orig = Image.open(original_image).convert("RGB")
    else:
        pil_orig = original_image.convert("RGB")

    # Select target layer
    if target_layers is None:
        target_layers = get_default_target_layer(model)

    device = next(model.parameters()).device
    tensor_dev = input_tensor.to(device)

    # 1. Compute Grad-CAM mask
    raw_cam = None
    try:
        if _GRADCAM_LIB_AVAILABLE and GradCAM is not None:
            targets = [ClassifierOutputTarget(target_class)] if target_class is not None else None
            with GradCAM(model=model, target_layers=target_layers) as cam:
                grayscale_cam = cam(input_tensor=tensor_dev, targets=targets)
                raw_cam = grayscale_cam[0, :]  # Shape: [H, W]
    except Exception as exc:
        logger.warning("pytorch_grad_cam failed, attempting fallback manual hook: %s", exc)
        raw_cam = None

    # Fallback if library failed or was unavailable
    if raw_cam is None:
        raw_cam = _manual_gradcam_fallback(model, target_layers[0], tensor_dev, target_class)

    # Ensure normalized in [0, 1]
    cam_min, cam_max = float(raw_cam.min()), float(raw_cam.max())
    if cam_max > cam_min:
        norm_cam = (raw_cam - cam_min) / (cam_max - cam_min)
    else:
        norm_cam = np.zeros_like(raw_cam)

    # Resize CAM to match original image dimensions for accurate overlay
    orig_w, orig_h = pil_orig.size
    cam_pil = Image.fromarray(np.uint8(norm_cam * 255)).resize((orig_w, orig_h), Image.Resampling.BILINEAR)
    cam_resized = np.asarray(cam_pil, dtype=np.float32) / 255.0

    orig_rgb_norm = np.asarray(pil_orig, dtype=np.float32) / 255.0
    overlay_arr = overlay_cam_on_image(orig_rgb_norm, cam_resized, alpha=0.5)
    heatmap_overlay = Image.fromarray(overlay_arr)

    # Side-by-side composite
    combined_img = create_side_by_side_image(pil_orig, heatmap_overlay)

    # Determine save path
    if not save_path:
        out_dir = os.path.join(getattr(Config, "REPORTS_DIR", "reports"), "gradcam")
        os.makedirs(out_dir, exist_ok=True)
        filename = f"gradcam_{uuid.uuid4().hex[:8]}.png"
        save_path = os.path.join(out_dir, filename)
    else:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    combined_img.save(save_path, format="PNG")
    logger.info("Saved Grad-CAM visualization to %s", save_path)

    return heatmap_overlay, norm_cam, combined_img, save_path


def _manual_gradcam_fallback(
    model: nn.Module,
    target_layer: nn.Module,
    input_tensor: torch.Tensor,
    target_class: Optional[int]
) -> np.ndarray:
    """Pure PyTorch Grad-CAM fallback if pytorch-grad-cam encounters execution issues."""
    activations: List[torch.Tensor] = []
    gradients: List[torch.Tensor] = []

    def forward_hook(module, inp, out):
        activations.append(out)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    try:
        input_tensor.requires_grad_(True)
        outputs = model(input_tensor)
        if target_class is None:
            target_class = int(outputs.argmax(dim=-1).item())

        score = outputs[0, target_class]
        model.zero_grad()
        score.backward(retain_graph=False)

        act = activations[0].detach()  # [1, C, H, W]
        grad = gradients[0].detach()   # [1, C, H, W]

        # Global average pool of gradients
        weights = torch.mean(grad, dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        cam = torch.sum(weights * act, dim=1, keepdim=True)    # [1, 1, H, W]
        cam = torch.relu(cam)

        cam_np = cam.squeeze().cpu().numpy()
        return cam_np
    finally:
        h1.remove()
        h2.remove()
