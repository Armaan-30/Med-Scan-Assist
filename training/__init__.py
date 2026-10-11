"""Training package for MedScan Assist.

Provides end-to-end dataset inspection, stratified preparation, PyTorch dataloaders,
two-phase model training, held-out test evaluation, and inference utilities.
"""

from training.inspect_dataset import inspect_dataset
from training.prepare_dataset import (
    ChestXrayDataset,
    create_dataloaders,
    get_eval_transforms,
    get_train_transforms,
    prepare_dataset,
)
from training.train import train_model
from training.evaluate import evaluate_test_set
from training.inference import predict_batch, predict_image

__all__ = [
    "inspect_dataset",
    "ChestXrayDataset",
    "create_dataloaders",
    "get_train_transforms",
    "get_eval_transforms",
    "prepare_dataset",
    "train_model",
    "evaluate_test_set",
    "predict_image",
    "predict_batch",
]
