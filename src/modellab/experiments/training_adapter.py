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

