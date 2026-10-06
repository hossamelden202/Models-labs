from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from modellab.evaluation.image_dataset import ImageClassificationDataset


class TrainingAdapterError(RuntimeError):
    """Raised when ModelLab data/model objects cannot be adapted for training."""


class _ClassificationTrainingDataset(Dataset):
    """
    Adapt a ModelLab ImageClassificationDataset to the training engine.

    Output:
        (image_tensor, integer_label)

    Image tensors are always float32 CHW tensors.
    """

    def __init__(self, dataset: Any):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    @staticmethod
    def _image_to_tensor(image: Any) -> torch.Tensor:
        """
        Convert common image representations into float32 CHW tensors.

        Supported:
          - torch.Tensor
          - PIL.Image
          - numpy.ndarray
        """

        if isinstance(image, torch.Tensor):
            tensor = image

        elif isinstance(image, np.ndarray):
            tensor = torch.from_numpy(
                np.ascontiguousarray(image)
            )

        else:
            # PIL.Image and other array-like image objects.
            try:
                array = np.asarray(image)
            except Exception as exc:
                raise TrainingAdapterError(
                    f"Could not convert image of type "
                    f"{type(image).__name__} to a tensor."
                ) from exc

            if array.ndim == 0:
                raise TrainingAdapterError(
                    f"Image conversion produced a scalar for "
                    f"type {type(image).__name__}."
                )

            tensor = torch.from_numpy(
                np.ascontiguousarray(array)
            )

        if not isinstance(tensor, torch.Tensor):
            raise TrainingAdapterError(
                "Image conversion did not produce a torch.Tensor."
            )

        # Images should be floating point for neural-network training.
        if not torch.is_floating_point(tensor):
            tensor = tensor.float()
        else:
            tensor = tensor.to(dtype=torch.float32)

        # Convert uint8-style [0, 255] images to [0, 1].
        if tensor.numel() > 0:
            max_value = float(tensor.max().detach().cpu())

            if max_value > 1.0:
                tensor = tensor / 255.0

        # HWC -> CHW for image arrays.
        if tensor.ndim == 3:
            # Existing CHW tensor.
            if tensor.shape[0] in (1, 3, 4):
                pass

            # HWC tensor.
            elif tensor.shape[-1] in (1, 3, 4):
                tensor = tensor.permute(2, 0, 1).contiguous()

            else:
                raise TrainingAdapterError(
                    "3D image tensor has an unsupported shape: "
                    f"{tuple(tensor.shape)}"
                )

        elif tensor.ndim == 2:
            # Grayscale HxW -> 1xHxW.
            tensor = tensor.unsqueeze(0)

        else:
            raise TrainingAdapterError(
                "Expected a 2D or 3D image tensor, got shape "
                f"{tuple(tensor.shape)}"
            )

        return tensor

    def __getitem__(self, index: int):
        try:
            sample = self.dataset.get_sample(index)
        except Exception as exc:
            raise TrainingAdapterError(
                f"Could not read ModelLab dataset sample {index}."
            ) from exc

        if sample is None:
            raise TrainingAdapterError(
                f"Dataset sample {index} is None."
            )

        # ModelLab DatasetSample:
        #   sample.input -> actual image
        #   sample.label -> integer class label
        image = getattr(sample, "input", None)
        label = getattr(sample, "label", None)

        if image is None:
            raise TrainingAdapterError(
                f"Dataset sample {index} did not contain an input image."
            )

        if label is None:
            raise TrainingAdapterError(
                f"Dataset sample {index} did not contain a label."
            )

        image = self._image_to_tensor(image)

        if isinstance(label, torch.Tensor):
            if label.numel() != 1:
                raise TrainingAdapterError(
                    f"Dataset sample {index} label tensor must contain "
                    f"exactly one value; got shape {tuple(label.shape)}."
                )
            label = int(label.item())

        elif isinstance(label, (int, float, np.integer, np.floating)):
            label = int(label)

        else:
            raise TrainingAdapterError(
                f"Dataset sample {index} has unsupported label type "
                f"{type(label).__name__}."
            )

        return image, label

# ---------------------------------------------------------------------------
# Generic classification-training compatibility layer
# ---------------------------------------------------------------------------
#
# These helpers adapt existing ModelLab model/dataset objects to the generic
# training engine. They intentionally do not introduce a second model or
# dataset registry.
# ---------------------------------------------------------------------------


@dataclass
class TrainingDataLoaders:
    """Training and validation DataLoaders used by training_runner."""

    train: DataLoader
    validation: DataLoader


def load_model_for_training(
    model_spec: Any,
    *,
    device: str = "auto",
) -> torch.nn.Module:
    """
    Load an existing ModelLab TorchModelSpec as a trainable PyTorch module.

    Unlike TorchImageClassifier, the returned module remains trainable:
    parameters are not frozen and the module is put into training mode.

    This function is intentionally restricted to classification models because
    the generic training_engine.train_classifier() is a classification engine.
    Detection training uses the dedicated detection_training adapter.
    """

    if model_spec is None:
        raise TrainingAdapterError(
            "model_spec is required for generic classification training."
        )

    task = getattr(model_spec, "task", "classification")

    if task != "classification":
        raise TrainingAdapterError(
            "Generic training adapter only supports classification models; "
            f"received task={task!r}. Detection training must use the "
            "dedicated detection training adapter."
        )

    try:
        from modellab.evaluation.torch_model import (
            TorchImageClassifier,
        )
    except Exception as exc:
        raise TrainingAdapterError(
            "Could not import ModelLab TorchImageClassifier."
        ) from exc

    try:
        adapter = TorchImageClassifier(
            model_spec,
            device=device,
        )
    except Exception as exc:
        raise TrainingAdapterError(
            f"Could not load model '{getattr(model_spec, 'model_id', '<unknown>')}' "
            "for generic training."
        ) from exc

    model = adapter.model

    if not isinstance(model, torch.nn.Module):
        raise TrainingAdapterError(
            "Loaded ModelLab model is not a torch.nn.Module."
        )

    # TorchImageClassifier freezes parameters because it is an inference
    # adapter. Undo that specifically for the generic training path.
    for parameter in model.parameters():
        parameter.requires_grad_(True)

    model.train()

    return model


def count_parameters(model: torch.nn.Module) -> int:
    """Return the total number of model parameters."""

    if not isinstance(model, torch.nn.Module):
        raise TrainingAdapterError(
            f"Expected torch.nn.Module, got {type(model).__name__}."
        )

    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


def count_trainable_parameters(model: torch.nn.Module) -> int:
    """Return the number of parameters currently requiring gradients."""

    if not isinstance(model, torch.nn.Module):
        raise TrainingAdapterError(
            f"Expected torch.nn.Module, got {type(model).__name__}."
        )

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def build_training_dataloaders(
    train_dataset: Any,
    validation_dataset: Any,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
) -> TrainingDataLoaders:
    """
    Adapt existing ModelLab classification datasets to PyTorch DataLoaders.

    The underlying ModelLab datasets remain unchanged. This wrapper only
    converts DatasetSample objects into (image_tensor, integer_label) pairs.
    """

    if train_dataset is None:
        raise TrainingAdapterError(
            "train_dataset is required."
        )

    if validation_dataset is None:
        raise TrainingAdapterError(
            "validation_dataset is required."
        )

    try:
        batch_size = int(batch_size)
        num_workers = int(num_workers)
    except (TypeError, ValueError) as exc:
        raise TrainingAdapterError(
            "batch_size and num_workers must be integers."
        ) from exc

    if batch_size <= 0:
        raise TrainingAdapterError(
            f"batch_size must be > 0, got {batch_size}."
        )

    if num_workers < 0:
        raise TrainingAdapterError(
            f"num_workers must be >= 0, got {num_workers}."
        )

    try:
        train_adapter = _ClassificationTrainingDataset(
            train_dataset
        )
        validation_adapter = _ClassificationTrainingDataset(
            validation_dataset
        )
    except Exception as exc:
        raise TrainingAdapterError(
            "Could not adapt ModelLab classification datasets."
        ) from exc

    if len(train_adapter) == 0:
        raise TrainingAdapterError(
            "Training dataset is empty."
        )

    if len(validation_adapter) == 0:
        raise TrainingAdapterError(
            "Validation dataset is empty."
        )

    # A generator gives deterministic DataLoader shuffling when the caller
    # has already seeded PyTorch through the training engine.
    generator = torch.Generator()

    train_loader = DataLoader(
        train_adapter,
        batch_size=batch_size,
        shuffle=bool(shuffle),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        generator=generator if shuffle else None,
    )

    validation_loader = DataLoader(
        validation_adapter,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return TrainingDataLoaders(
        train=train_loader,
        validation=validation_loader,
    )
