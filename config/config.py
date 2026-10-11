"""MedScan Assist - Configuration Module.

Central configuration management for MedScan Assist, loading environment
variables with robust defaults for paths, training hyperparameters,
image validation, quality thresholds, and inference settings.
"""

import os
from pathlib import Path
from typing import List, Tuple
from dotenv import load_dotenv
import torch

# Base Directory Resolution
PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERCEL_RUNTIME: bool = os.getenv("VERCEL", "").lower() == "1"
RUNTIME_DATA_DIR: str = (
    os.path.join(os.getenv("TMPDIR", "/tmp"), "medscan-assist")
    if VERCEL_RUNTIME else PROJECT_ROOT
)

# Load environment variables from .env file at project root if present
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))


class Config:
    """Central configuration class for MedScan Assist."""

    # Project Information
    PROJECT_NAME: str = "MedScan Assist"
    VERSION: str = "1.0.0"
    RESEARCH_DISCLAIMER: str = (
        "RESEARCH USE ONLY: MedScan Assist is an artificial intelligence research "
        "prototype intended solely to assist clinical decision-making. It is not "
        "cleared or approved as a standalone diagnostic medical device. All findings "
        "must be reviewed and verified by a licensed healthcare professional."
    )

    # Random Seed
    SEED: int = int(os.getenv("SEED", "42"))

    # Device Detection & Configuration
    DEVICE_OVERRIDE: str = os.getenv("DEVICE", "").strip()
    if DEVICE_OVERRIDE:
        DEVICE: torch.device = torch.device(DEVICE_OVERRIDE)
    else:
        DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Directory Paths (Relative to PROJECT_ROOT)
    BASE_DIR: str = PROJECT_ROOT
    DATA_DIR: str = os.getenv("DATASET_DIR", os.path.join(PROJECT_ROOT, "data", "chest_xray"))
    TRAIN_DIR: str = os.path.join(DATA_DIR, "train")
    VAL_DIR: str = os.path.join(DATA_DIR, "val")
    TEST_DIR: str = os.path.join(DATA_DIR, "test")

    CHECKPOINT_DIR: str = os.path.join(PROJECT_ROOT, "models", "checkpoints")
    BEST_MODEL_PATH: str = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    LATEST_MODEL_PATH: str = os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pth")

    REPORTS_DIR: str = os.path.join(RUNTIME_DATA_DIR, "reports")
    UPLOADS_DIR: str = os.path.join(RUNTIME_DATA_DIR, "uploads")
    LOGS_DIR: str = os.path.join(RUNTIME_DATA_DIR, "logs")

    # Model Architecture Settings
    MODEL_NAME: str = "efficientnet_b0"
    NUM_CLASSES: int = 2
    CLASS_NAMES: List[str] = ["NORMAL", "PNEUMONIA"]
    CLASS_TO_IDX: dict = {cls: idx for idx, cls in enumerate(CLASS_NAMES)}
    IDX_TO_CLASS: dict = {idx: cls for idx, cls in enumerate(CLASS_NAMES)}

    # Model Hyperparameters
    IMAGE_SIZE: int = int(os.getenv("IMAGE_SIZE", "224"))
    INPUT_SHAPE: Tuple[int, int, int] = (3, IMAGE_SIZE, IMAGE_SIZE)
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "32"))
    LEARNING_RATE: float = float(os.getenv("LEARNING_RATE", "0.001"))
    EPOCHS: int = int(os.getenv("EPOCHS", "25"))
    EARLY_STOPPING_PATIENCE: int = int(os.getenv("EARLY_STOPPING_PATIENCE", "5"))
    WEIGHT_DECAY: float = float(os.getenv("WEIGHT_DECAY", "1e-4"))
    DROPOUT_RATE: float = float(os.getenv("DROPOUT_RATE", "0.3"))
    NUM_WORKERS: int = int(os.getenv("NUM_WORKERS", "4" if os.name != "nt" else "0"))

    # ImageNet Normalization Constants
    IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
    IMAGENET_STD: List[float] = [0.229, 0.224, 0.225]

    # Image Validation Settings
    MAX_FILE_SIZE_BYTES: int = int(os.getenv("MAX_FILE_SIZE_BYTES", str(10 * 1024 * 1024)))  # 10 MB
    ALLOWED_EXTENSIONS: set = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".dcm"}
    MIN_IMAGE_DIMENSION: int = 64
    MAX_IMAGE_DIMENSION: int = 4096

    # Quality Check Thresholds
    # Brightness bounds for 8-bit grayscale intensity mean [0, 255]
    MIN_BRIGHTNESS_THRESHOLD: float = float(os.getenv("MIN_BRIGHTNESS_THRESHOLD", "20.0"))
    MAX_BRIGHTNESS_THRESHOLD: float = float(os.getenv("MAX_BRIGHTNESS_THRESHOLD", "235.0"))
    # Blur detection threshold based on variance of the Laplacian (higher means less blurry)
    MIN_LAPLACIAN_VAR_THRESHOLD: float = float(os.getenv("MIN_LAPLACIAN_VAR_THRESHOLD", "80.0"))

    # Confidence and Abstention Threshold
    # If maximum softmax confidence is below this threshold, the model abstains or flags uncertainty
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.70"))

    @classmethod
    def ensure_directories(cls) -> None:
        """Create necessary directories if they do not already exist."""
        paths = [cls.REPORTS_DIR, cls.UPLOADS_DIR, cls.LOGS_DIR]
        if not VERCEL_RUNTIME:
            paths.append(cls.CHECKPOINT_DIR)
        for path in paths:
            os.makedirs(path, exist_ok=True)


# Automatically ensure required directories are initialized on import
Config.ensure_directories()

