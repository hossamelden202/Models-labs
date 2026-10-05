
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset


class TrainingAdapterError(RuntimeError):
    """Raised when ModelLab objects cannot be adapted for training."""


class _ClassificationTrainingDataset(Dataset):
    """
    Thin adapter around ModelLab's ImageClassificationDataset.

    It intentionally returns only:

        (image_tensor, class_index)

    so the generic training engine does not need to know about
    ModelLab evaluation metadata such as sample IDs and paths.
    """

    def __init__(
        self,
        dataset: Any,
    ) -> None:
        self.dataset = dataset

        try:
            self.length = len(dataset)
        except Exception as exc:
            raise TrainingAdapterError(
                "ModelLab dataset does not implement __len__()."
            ) from exc

    def __len__(self) -> int:
        return self.length

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, int]:
        try:
            sample = self.dataset.get_sample(index)
        except Exception as exc:
            raise TrainingAdapterError(
                f"Could not read dataset sample at index {index}."
            ) from exc

        # ModelLab sample objects may expose input/label directly.
        if hasattr(sample, "input") and hasattr(sample, "label"):
            image = sample.input
            label = sample.label

        # Some implementations may return a dictionary.
        elif isinstance(sample, dict):
            image = (
                sample.get("input")
                or sample.get("image")
                or sample.get("inputs")
            )

            label = (
                sample.get("label")
                if "label" in sample
                else sample.get("target")
            )

        # Fallback for tuple/list samples.
        elif isinstance(sample, (tuple, list)) and len(sample) >= 2:
            image = sample[0]
            label = sample[1]

        else:
            raise TrainingAdapterError(
                "Unsupported ModelLab dataset sample format: "
                f"{type(sample)!r}"
            )

        if not isinstance(image, torch.Tensor):
            image = torch.as_tensor(image)

        if not isinstance(label, (int, float)):
            if isinstance(label, torch.Tensor):
                label = int(label.item())
            else:
                try:
                    label = int(label)
                except Exception as exc:
                    raise TrainingAdapterError(
                        f"Could not convert label to integer: {label!r}"
                    ) from exc

        return image, int(label)


@dataclass(frozen=True)
class TrainingDataLoaders:
    train: DataLoader
    validation: DataLoader


def build_training_dataloaders(
    train_dataset: Any,
    validation_dataset: Any,
    *,
    batch_size: int,
    num_workers: int = 0,
    shuffle: bool = True,
    pin_memory: bool | None = None,
) -> TrainingDataLoaders:
    """
    Adapt ModelLab classification datasets into training DataLoaders.
    """

    if batch_size <= 0:
        raise TrainingAdapterError(
            f"batch_size must be > 0, got {batch_size}"
        )

    if num_workers < 0:
        raise TrainingAdapterError(
            f"num_workers must be >= 0, got {num_workers}"
        )

    train = _ClassificationTrainingDataset(
        train_dataset
    )

    validation = _ClassificationTrainingDataset(
        validation_dataset
    )

    if len(train) == 0:
        raise TrainingAdapterError(
            "Training dataset contains zero samples."
        )

    if len(validation) == 0:
        raise TrainingAdapterError(
            "Validation dataset contains zero samples."
        )

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=(
            bool(num_workers > 0)
        ),
    )

    validation_loader = DataLoader(
        validation,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=(
            bool(num_workers > 0)
        ),
    )

    return TrainingDataLoaders(
        train=train_loader,
        validation=validation_loader,
    )


def load_model_for_training(
    model_spec: Any,
    *,
    device: str = "auto",
) -> torch.nn.Module:
    """
    Load an existing ModelLab Torch model and expose its underlying
    nn.Module for the training engine.

    No second model-loading implementation is introduced here.
    """

    try:
        from modellab.evaluation.torch_model import (
            TorchImageClassifier,
        )
    except Exception as exc:
        raise TrainingAdapterError(
            "Could not import ModelLab TorchImageClassifier."
        ) from exc

    try:
        classifier = TorchImageClassifier(
            model_spec,
            device=device,
        )
    except Exception as exc:
        raise TrainingAdapterError(
            "ModelLab could not load the supplied model specification."
        ) from exc

    model = getattr(classifier, "model", None)

    if not isinstance(model, torch.nn.Module):
        raise TrainingAdapterError(
            "TorchImageClassifier did not expose a torch.nn.Module "
            "through its .model attribute."
        )

    return model


def count_trainable_parameters(
    model: torch.nn.Module,
) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def count_parameters(
    model: torch.nn.Module,
) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


__all__ = [
    "TrainingAdapterError",
    "TrainingDataLoaders",
    "build_training_dataloaders",
    "count_parameters",
    "count_trainable_parameters",
    "load_model_for_training",
]
