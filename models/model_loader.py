"""MedScan Assist - Thread-Safe Model Loading Service.

Provides singleton-pattern caching, validation, and metadata inspection for the
trained Chest X-Ray EfficientNet-B0 model. Ensures thread safety during concurrent
inferences in multi-threaded web API environments.
"""

import logging
import os
import threading
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn

from config.config import Config
from models.model_architecture import build_efficientnet_b0

logger = logging.getLogger(__name__)


class ModelLoader:
    """Thread-safe singleton model loader and cache manager."""

    _instance: Optional["ModelLoader"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        """Initialize ModelLoader instance attributes."""
        self._model: Optional[nn.Module] = None
        self._metadata: Dict[str, Any] = {}
        self._checkpoint_path: Optional[str] = None
        self._device: torch.device = Config.DEVICE
        self._is_loaded: bool = False
        self._inner_lock: threading.Lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "ModelLoader":
        """Thread-safe singleton accessor."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (mainly used for testing)."""
        with cls._lock:
            cls._instance = None

    def validate_checkpoint(self, checkpoint_path: str) -> Tuple[bool, str]:
        """Validate integrity and required keys of a checkpoint file on disk.

        Args:
            checkpoint_path: Path to checkpoint file.

        Returns:
            Tuple[bool, str]: (is_valid, status_message)
        """
        if not os.path.exists(checkpoint_path):
            return False, f"Checkpoint file does not exist at '{checkpoint_path}'."

        if not os.path.isfile(checkpoint_path):
            return False, f"Path '{checkpoint_path}' is not a regular file."

        try:
            # Inspect checkpoint dictionary on CPU
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        except Exception as exc:
            return False, f"Failed to deserialize checkpoint file: {exc}"

        if not isinstance(checkpoint, dict):
            return False, "Checkpoint file does not contain a valid dictionary payload."

        required_keys = {"model_state_dict"}
        missing_keys = required_keys - set(checkpoint.keys())
        if missing_keys:
            return False, f"Checkpoint missing required keys: {missing_keys}"

        return True, "Checkpoint passed integrity verification."

    def load_model(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        force_reload: bool = False,
    ) -> Tuple[Optional[nn.Module], Dict[str, Any]]:
        """Load and cache the trained model from disk in a thread-safe manner.

        Args:
            checkpoint_path: Path to model checkpoint (.pth). If None, uses Config.BEST_MODEL_PATH.
            device: PyTorch device to host the model. Defaults to Config.DEVICE.
            force_reload: If True, forces reloading from disk even if cached.

        Returns:
            Tuple[Optional[nn.Module], Dict[str, Any]]:
                (model, status_dict) where model is None if checkpoint is missing or invalid.
        """
        target_path = checkpoint_path or Config.BEST_MODEL_PATH
        target_device = device or Config.DEVICE

        with self._inner_lock:
            if (
                self._is_loaded
                and not force_reload
                and self._checkpoint_path == target_path
                and self._device == target_device
                and self._model is not None
            ):
                return self._model, {
                    "success": True,
                    "message": "Model retrieved from memory cache.",
                    "metadata": self._metadata,
                }

            is_valid, msg = self.validate_checkpoint(target_path)
            if not is_valid:
                logger.warning("Checkpoint loading skipped: %s", msg)
                return None, {
                    "success": False,
                    "message": msg,
                    "metadata": {},
                }

            try:
                checkpoint = torch.load(target_path, map_location=target_device)
                class_names = checkpoint.get("class_names", Config.CLASS_NAMES)
                dropout_rate = checkpoint.get("config", {}).get("dropout_rate", Config.DROPOUT_RATE)

                model = build_efficientnet_b0(
                    num_classes=len(class_names),
                    pretrained=False,
                    freeze_backbone_layers=False,
                    dropout_rate=dropout_rate,
                )
                model.load_state_dict(checkpoint["model_state_dict"])
                model.to(target_device)
                model.eval()

                self._model = model
                self._checkpoint_path = target_path
                self._device = target_device
                self._metadata = {
                    "architecture": checkpoint.get("architecture", Config.MODEL_NAME),
                    "epoch": checkpoint.get("epoch"),
                    "metrics": checkpoint.get("metrics", {}),
                    "class_names": class_names,
                    "class_to_idx": checkpoint.get("class_to_idx", Config.CLASS_TO_IDX),
                    "training_config": checkpoint.get("config", {}),
                    "saved_at": checkpoint.get("saved_at"),
                    "checkpoint_path": target_path,
                    "device": str(target_device),
                    "disclaimer": Config.RESEARCH_DISCLAIMER,
                }
                self._is_loaded = True

                logger.info("Successfully loaded MedScan Assist model from %s to %s", target_path, target_device)
                return self._model, {
                    "success": True,
                    "message": "Model loaded successfully into memory.",
                    "metadata": self._metadata,
                }

            except Exception as exc:
                logger.error("Error occurred while loading model: %s", str(exc), exc_info=True)
                self._is_loaded = False
                self._model = None
                return None, {
                    "success": False,
                    "message": f"Failed to load checkpoint: {str(exc)}",
                    "metadata": {},
                }

    def get_model(self) -> Optional[nn.Module]:
        """Return the currently loaded model instance or None."""
        return self._model

    def get_model_info(self) -> Dict[str, Any]:
        """Retrieve runtime metadata and architecture information of the cached model.

        Returns:
            Dict[str, Any]: Metadata dictionary.
        """
        with self._inner_lock:
            return {
                "is_loaded": self._is_loaded,
                "device": str(self._device),
                "checkpoint_path": self._checkpoint_path,
                "model_architecture": Config.MODEL_NAME,
                "input_shape": list(Config.INPUT_SHAPE),
                "class_names": self._metadata.get("class_names", Config.CLASS_NAMES),
                "metadata": self._metadata,
                "disclaimer": Config.RESEARCH_DISCLAIMER,
            }

    def is_ready(self) -> bool:
        """Check if a model is currently loaded and available for inference."""
        return self._is_loaded and self._model is not None

    def clear_cache(self) -> None:
        """Clear cached model and reset state."""
        with self._inner_lock:
            self._model = None
            self._metadata = {}
            self._checkpoint_path = None
            self._is_loaded = False
