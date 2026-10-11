"""Image validation module for MedScan Assist.

Provides technical image validation checks including file extension,
MIME type magic bytes verification, file size, readability, dimensions,
brightness, blur detection, and color mode verification.

DISCLAIMER: These checks evaluate basic technical image quality and file integrity only.
They DO NOT assess clinical adequacy, patient positioning, or diagnostic quality.
"""

import io
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    _CV2_AVAILABLE = False

from config.config import Config

logger = logging.getLogger("medscan.image_validator")

# Magic bytes mapping for allowed formats
MAGIC_BYTES_MAP = {
    b"\xff\xd8\xff": "image/jpeg",
    b"\x89PNG\r\n\x1a\n": "image/png",
    b"BM": "image/bmp",
    b"II*\x00": "image/tiff",  # Little-endian TIFF
    b"MM\x00*": "image/tiff",  # Big-endian TIFF
}

TECHNICAL_DISCLAIMER = (
    "Notice: Image validation performs basic technical and file integrity checks only. "
    "It does not evaluate clinical diagnostic adequacy, chest anatomy inclusion, or radiological positioning."
)


def validate_file_extension(filename: str, allowed_extensions: Optional[set] = None) -> Tuple[bool, Optional[str]]:
    """Validate that the file extension is supported.

    Args:
        filename: Name or path of the image file.
        allowed_extensions: Optional set of allowed extensions. Defaults to Config.ALLOWED_EXTENSIONS.

    Returns:
        Tuple of (is_valid, error_message).
    """
    if allowed_extensions is None:
        allowed_extensions = getattr(Config, "ALLOWED_EXTENSIONS", {".jpg", ".jpeg", ".png", ".bmp", ".tiff"})

    ext = os.path.splitext(filename)[1].lower()
    if not ext:
        return False, "File has no extension. Please provide an image file with a valid extension (e.g., .png, .jpg)."
    if ext not in allowed_extensions:
        allowed_str = ", ".join(sorted(allowed_extensions))
        return False, f"Unsupported file extension '{ext}'. Allowed extensions are: {allowed_str}."
    return True, None


def validate_mime_type(file_bytes: bytes) -> Tuple[bool, Optional[str], Optional[str]]:
    """Validate image MIME type by inspecting leading header magic bytes.

    Args:
        file_bytes: Raw bytes of the image file.

    Returns:
        Tuple of (is_valid, detected_mime_type, error_message).
    """
    if not file_bytes:
        return False, None, "File content is empty (0 bytes)."

    for magic, mime in MAGIC_BYTES_MAP.items():
        if file_bytes.startswith(magic):
            return True, mime, None

    # Check DICOM prefix at offset 128 (preamble is 128 bytes followed by 'DICM')
    if len(file_bytes) >= 132 and file_bytes[128:132] == b"DICM":
        return True, "application/dicom", None

    return False, None, "File signature does not match any accepted image format (JPEG, PNG, BMP, TIFF)."


def validate_file_size(
    file_bytes_or_path: Union[bytes, str],
    max_size_mb: Optional[float] = None
) -> Tuple[bool, float, Optional[str]]:
    """Validate that the file size is within limits.

    Args:
        file_bytes_or_path: Raw bytes or string file path.
        max_size_mb: Max allowed size in megabytes. Defaults to Config.MAX_FILE_SIZE_BYTES or 10 MB.

    Returns:
        Tuple of (is_valid, actual_size_mb, error_message).
    """
    if max_size_mb is None:
        max_bytes = getattr(Config, "MAX_FILE_SIZE_BYTES", 10 * 1024 * 1024)
        max_size_mb = max_bytes / (1024 * 1024)

    if isinstance(file_bytes_or_path, (bytes, bytearray)):
        size_bytes = len(file_bytes_or_path)
    else:
        if not os.path.isfile(file_bytes_or_path):
            return False, 0.0, f"File does not exist: {file_bytes_or_path}"
        size_bytes = os.path.getsize(file_bytes_or_path)

    size_mb = size_bytes / (1024 * 1024)
    if size_bytes == 0:
        return False, 0.0, "File size is 0 bytes."
    if size_mb > max_size_mb:
        return False, size_mb, f"File size ({size_mb:.2f} MB) exceeds maximum allowed size ({max_size_mb:.2f} MB)."
    return True, size_mb, None


def validate_image_readable(image_input: Union[str, bytes]) -> Tuple[bool, Optional[Image.Image], Optional[str]]:
    """Attempt to open and verify the image with PIL.

    Args:
        image_input: File path (str) or raw image bytes.

    Returns:
        Tuple of (is_valid, PIL_Image_or_None, error_message).
    """
    try:
        if isinstance(image_input, (bytes, bytearray)):
            stream = io.BytesIO(image_input)
            img = Image.open(stream)
        else:
            if not os.path.isfile(image_input):
                return False, None, f"Image file not found: {image_input}"
            img = Image.open(image_input)

        img.load()  # Force load pixel data
        return True, img, None
    except Exception as exc:
        logger.warning("Failed to open image: %s", exc)
        return False, None, f"Corrupted or unreadable image file: {exc}"


def validate_dimensions(
    image: Image.Image,
    min_dim: Optional[int] = None,
    max_dim: Optional[int] = None
) -> Tuple[bool, Tuple[int, int], Optional[str]]:
    """Validate image dimensions against minimum and maximum boundaries.

    Args:
        image: PIL Image object.
        min_dim: Minimum width/height in pixels (default: 32 or Config.MIN_IMAGE_DIMENSION).
        max_dim: Maximum width/height in pixels (default: 10000 or Config.MAX_IMAGE_DIMENSION).

    Returns:
        Tuple of (is_valid, (width, height), error_message).
    """
    if min_dim is None:
        min_dim = getattr(Config, "MIN_IMAGE_DIMENSION", 32)
    if max_dim is None:
        max_dim = getattr(Config, "MAX_IMAGE_DIMENSION", 10000)

    width, height = image.size
    if width < min_dim or height < min_dim:
        return False, (width, height), (
            f"Image resolution ({width}x{height}) is too small. "
            f"Minimum required dimension is {min_dim}x{min_dim} pixels."
        )
    if width > max_dim or height > max_dim:
        return False, (width, height), (
            f"Image resolution ({width}x{height}) exceeds allowable maximum dimension of {max_dim}x{max_dim} pixels."
        )
    return True, (width, height), None


def validate_image_mode(image: Image.Image) -> Tuple[bool, str, Optional[str]]:
    """Validate that the image mode is supported (L, RGB, RGBA).

    Args:
        image: PIL Image object.

    Returns:
        Tuple of (is_valid, mode, error_message).
    """
    supported_modes = {"L", "RGB", "RGBA", "P"}
    mode = image.mode
    if mode not in supported_modes:
        return False, mode, f"Unsupported color mode '{mode}'. Supported modes are: {', '.join(sorted(supported_modes))}."
    return True, mode, None


def check_brightness(
    image: Image.Image,
    min_threshold: Optional[float] = None,
    max_threshold: Optional[float] = None
) -> Tuple[bool, float, Optional[str]]:
    """Check if the image is excessively dark or washed out/bright.

    Args:
        image: PIL Image object.
        min_threshold: Mean intensity lower bound [0-255]. Default: 30.0 or Config.MIN_BRIGHTNESS_THRESHOLD.
        max_threshold: Mean intensity upper bound [0-255]. Default: 240.0 or Config.MAX_BRIGHTNESS_THRESHOLD.

    Returns:
        Tuple of (is_normal, mean_brightness, warning_or_error_message).
    """
    if min_threshold is None:
        min_threshold = getattr(Config, "MIN_BRIGHTNESS_THRESHOLD", 30.0)
    if max_threshold is None:
        max_threshold = getattr(Config, "MAX_BRIGHTNESS_THRESHOLD", 240.0)

    # Convert to grayscale to evaluate mean luminance
    gray = image.convert("L")
    arr = np.asarray(gray, dtype=np.float32)
    mean_val = float(np.mean(arr))

    if mean_val < min_threshold:
        return False, mean_val, (
            f"Image is unusually dark (mean intensity: {mean_val:.1f} < {min_threshold:.1f}). "
            "Under-exposed scans may obscure anatomical landmarks."
        )
    if mean_val > max_threshold:
        return False, mean_val, (
            f"Image is excessively bright/overexposed (mean intensity: {mean_val:.1f} > {max_threshold:.1f}). "
            "Over-exposed scans may saturate fine pulmonary details."
        )
    return True, mean_val, None


def check_blur(
    image: Image.Image,
    blur_threshold: Optional[float] = None
) -> Tuple[bool, float, Optional[str]]:
    """Detect excessive blur using the variance of the Laplacian.

    Args:
        image: PIL Image object.
        blur_threshold: Threshold below which an image is considered blurry (default: ~100.0).

    Returns:
        Tuple of (is_sharp, laplacian_variance, warning_message).
    """
    if blur_threshold is None:
        blur_threshold = getattr(Config, "MIN_LAPLACIAN_VAR_THRESHOLD", 100.0)

    gray = image.convert("L")
    arr = np.asarray(gray, dtype=np.uint8)

    if _CV2_AVAILABLE and cv2 is not None:
        laplacian = cv2.Laplacian(arr, cv2.CV_64F)
        variance = float(laplacian.var())
    else:
        # Fallback 3x3 Laplacian filter via numpy
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
        # Apply simple 2D convolution
        h, w = arr.shape
        if h > 2 and w > 2:
            sub = (
                arr[:-2, 1:-1] * kernel[0, 1] +
                arr[1:-1, :-2] * kernel[1, 0] +
                arr[1:-1, 1:-1] * kernel[1, 1] +
                arr[1:-1, 2:] * kernel[1, 2] +
                arr[2:, 1:-1] * kernel[2, 1]
            )
            variance = float(np.var(sub))
        else:
            variance = 0.0

    if variance < blur_threshold:
        return False, variance, (
            f"Image appears blurry or low-contrast (Laplacian variance: {variance:.1f} < {blur_threshold:.1f}). "
            "Diagnostic accuracy may be reduced."
        )
    return True, variance, None


def run_all_checks(
    file_path: Optional[str] = None,
    file_bytes: Optional[bytes] = None
) -> Dict[str, Any]:
    """Run all image validation checks on a file path or raw bytes.

    Args:
        file_path: Path to the image file (optional if file_bytes provided).
        file_bytes: Raw bytes of the image file (optional if file_path provided).

    Returns:
        Dict containing:
            - is_valid: bool (True if all critical checks pass)
            - warnings: List[str] (non-fatal quality notices like minor blur/brightness)
            - errors: List[str] (fatal errors like unreadable, invalid extension)
            - quality_info: Dict[str, Any] with detailed metrics
            - disclaimer: str
    """
    warnings: List[str] = []
    errors: List[str] = []
    quality_info: Dict[str, Any] = {
        "file_path": file_path,
        "extension_valid": False,
        "mime_type": None,
        "size_mb": 0.0,
        "dimensions": None,
        "mode": None,
        "brightness_mean": None,
        "blur_laplacian_var": None,
    }

    # 1. Read bytes if needed
    if file_bytes is None and file_path is not None:
        if not os.path.exists(file_path):
            errors.append(f"Image file does not exist: {file_path}")
            return {
                "is_valid": False,
                "warnings": warnings,
                "errors": errors,
                "quality_info": quality_info,
                "disclaimer": TECHNICAL_DISCLAIMER,
            }
        try:
            with open(file_path, "rb") as f:
                file_bytes = f.read()
        except Exception as e:
            errors.append(f"Cannot read file from disk: {e}")
            return {
                "is_valid": False,
                "warnings": warnings,
                "errors": errors,
                "quality_info": quality_info,
                "disclaimer": TECHNICAL_DISCLAIMER,
            }

    if file_bytes is None:
        errors.append("Neither file_path nor file_bytes was provided.")
        return {
            "is_valid": False,
            "warnings": warnings,
            "errors": errors,
            "quality_info": quality_info,
            "disclaimer": TECHNICAL_DISCLAIMER,
        }

    # 2. File extension check (if file_path provided)
    if file_path:
        ext_valid, ext_err = validate_file_extension(file_path)
        quality_info["extension_valid"] = ext_valid
        if not ext_valid and ext_err:
            errors.append(ext_err)
    else:
        quality_info["extension_valid"] = True

    # 3. File size check
    size_valid, size_mb, size_err = validate_file_size(file_bytes)
    quality_info["size_mb"] = round(size_mb, 2)
    if not size_valid and size_err:
        errors.append(size_err)

    # 4. MIME type check
    mime_valid, mime_type, mime_err = validate_mime_type(file_bytes)
    quality_info["mime_type"] = mime_type
    if not mime_valid and mime_err:
        errors.append(mime_err)

    # If basic file format checks failed, return early
    if errors:
        return {
            "is_valid": False,
            "warnings": warnings,
            "errors": errors,
            "quality_info": quality_info,
            "disclaimer": TECHNICAL_DISCLAIMER,
        }

    # 5. Image readable check
    read_valid, pil_image, read_err = validate_image_readable(file_bytes)
    if not read_valid or pil_image is None:
        errors.append(read_err or "Image could not be decoded.")
        return {
            "is_valid": False,
            "warnings": warnings,
            "errors": errors,
            "quality_info": quality_info,
            "disclaimer": TECHNICAL_DISCLAIMER,
        }

    # 6. Dimensions check
    dim_valid, dims, dim_err = validate_dimensions(pil_image)
    quality_info["dimensions"] = {"width": dims[0], "height": dims[1]}
    if not dim_valid and dim_err:
        errors.append(dim_err)

    # 7. Mode check
    mode_valid, mode_str, mode_err = validate_image_mode(pil_image)
    quality_info["mode"] = mode_str
    if not mode_valid and mode_err:
        errors.append(mode_err)

    # 8. Brightness check (treated as quality warning rather than hard blocking error)
    bright_valid, mean_bright, bright_msg = check_brightness(pil_image)
    quality_info["brightness_mean"] = round(mean_bright, 1)
    if not bright_valid and bright_msg:
        warnings.append(bright_msg)

    # 9. Blur check (treated as quality warning)
    blur_valid, blur_var, blur_msg = check_blur(pil_image)
    quality_info["blur_laplacian_var"] = round(blur_var, 1)
    if not blur_valid and blur_msg:
        warnings.append(blur_msg)

    is_valid = len(errors) == 0

    return {
        "is_valid": is_valid,
        "warnings": warnings,
        "errors": errors,
        "quality_info": quality_info,
        "disclaimer": TECHNICAL_DISCLAIMER,
    }


class ImageValidator:
    """Class wrapper for image validation routines."""

    def __init__(self) -> None:
        pass

    def run_all_checks(
        self,
        file_path: Optional[str] = None,
        file_bytes: Optional[bytes] = None
    ) -> Dict[str, Any]:
        return run_all_checks(file_path=file_path, file_bytes=file_bytes)

    def validate_file_extension(self, filename: str) -> Tuple[bool, Optional[str]]:
        return validate_file_extension(filename)

    def validate_mime_type(self, file_bytes: bytes) -> Tuple[bool, Optional[str], Optional[str]]:
        return validate_mime_type(file_bytes)

    def validate_file_size(self, file_bytes_or_path: Union[bytes, str]) -> Tuple[bool, float, Optional[str]]:
        return validate_file_size(file_bytes_or_path)

    def validate_image_readable(self, file_bytes_or_path: Union[bytes, str]) -> Tuple[bool, Optional[Image.Image], Optional[str]]:
        return validate_image_readable(file_bytes_or_path)

