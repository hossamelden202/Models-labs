from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TrainingConfigError(ValueError):
    """Raised when a training experiment configuration is invalid."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OptimizerConfig(_Strict):
    name: Literal[
        "sgd",
        "adam",
        "adamw",
        "rmsprop",
    ] = "adamw"

    learning_rate: float = Field(gt=0)
    weight_decay: float = Field(0.0, ge=0)
    momentum: float | None = Field(None, ge=0, lt=1)


class SchedulerConfig(_Strict):
    name: Literal[
        "none",
        "step",
        "cosine",
        "cosine_warm_restarts",
        "plateau",
        "one_cycle",
    ] = "none"

    warmup_epochs: int = Field(0, ge=0)
    step_size: int | None = Field(None, gt=0)
    gamma: float | None = Field(None, gt=0, le=1)
    min_lr: float | None = Field(None, ge=0)


class AugmentationConfig(_Strict):
    enabled: bool = True

    probability: float = Field(
        0.5,
        ge=0,
        le=1,
    )

    crop_fraction: float | None = Field(
        None,
        gt=0,
        le=1,
    )

    horizontal_flip: bool = False

    brightness_factor: float | None = Field(
        None,
        gt=0,
    )

    contrast_factor: float | None = Field(
        None,
        gt=0,
    )

    saturation_factor: float | None = Field(
        None,
        ge=0,
    )

    hue_shift: float | None = Field(
        None,
        ge=-0.5,
        le=0.5,
    )

    rotation_degrees: float | None = Field(
        None,
        gt=0,
        le=360,
    )

    blur_radius: float | None = Field(
        None,
        gt=0,
        le=50,
    )

    noise_std: float | None = Field(
        None,
        gt=0,
        le=255,
    )

    random_occlusion_fraction: float | None = Field(
        None,
        gt=0,
        lt=1,
    )


class LossConfig(_Strict):
    name: Literal[
        "cross_entropy",
        "weighted_cross_entropy",
        "focal",
    ] = "cross_entropy"

    label_smoothing: float = Field(
        0.0,
        ge=0,
        lt=1,
    )

    focal_gamma: float | None = Field(
        None,
        gt=0,
    )

    class_weights: dict[str, float] | None = None

    @model_validator(mode="after")
    def validate_loss(self):
        if self.name == "focal" and self.focal_gamma is None:
            raise ValueError(
                "focal loss requires focal_gamma"
            )

        if self.name != "focal" and self.focal_gamma is not None:
            raise ValueError(
                "focal_gamma only applies to focal loss"
            )

        if self.name == "weighted_cross_entropy":
            if not self.class_weights:
                raise ValueError(
                    "weighted_cross_entropy requires class_weights"
                )

        return self


class DataConfig(_Strict):
    batch_size: int = Field(
        32,
        gt=0,
    )

    num_workers: int = Field(
        2,
        ge=0,
    )

    input_width: int = Field(
        224,
        gt=0,
    )

    input_height: int = Field(
        224,
        gt=0,
    )

    sampler: Literal[
        "default",
        "weighted",
        "balanced",
    ] = "default"

    augmentation: AugmentationConfig = Field(
        default_factory=AugmentationConfig
    )


class FreezeConfig(_Strict):
    mode: Literal[
        "none",
        "backbone",
        "all_but_head",
        "last_n_blocks",
    ] = "none"

    last_n_blocks: int | None = Field(
        None,
        gt=0,
    )

    @model_validator(mode="after")
    def validate_freeze(self):
        if self.mode == "last_n_blocks":
            if self.last_n_blocks is None:
                raise ValueError(
                    "last_n_blocks mode requires last_n_blocks"
                )
        elif self.last_n_blocks is not None:
            raise ValueError(
                "last_n_blocks only applies to last_n_blocks mode"
            )

        return self


class TrainingConfig(_Strict):
    epochs: int = Field(
        10,
        gt=0,
    )

    seed: int = Field(
        42,
        ge=0,
    )

    optimizer: OptimizerConfig = Field(
        default_factory=lambda: OptimizerConfig(
            learning_rate=1e-3
        )
    )

    scheduler: SchedulerConfig = Field(
        default_factory=SchedulerConfig
    )

    loss: LossConfig = Field(
        default_factory=LossConfig
    )

    data: DataConfig = Field(
        default_factory=DataConfig
    )

    freeze: FreezeConfig = Field(
        default_factory=FreezeConfig
    )

    gradient_accumulation_steps: int = Field(
        1,
        gt=0,
    )

    mixed_precision: bool = True

    max_grad_norm: float | None = Field(
        None,
        gt=0,
    )

    early_stopping_patience: int | None = Field(
        None,
        gt=0,
    )

    monitor: Literal[
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "loss",
    ] = "macro_f1"


class TrainingChange(_Strict):
    """
    A single controlled change relative to the baseline training config.

    Examples:

        {
            "path": "optimizer.learning_rate",
            "value": 0.0001
        }

        {
            "path": "data.augmentation.crop_fraction",
            "value": 0.8
        }

        {
            "path": "data.batch_size",
            "value": 64
        }
    """

    path: str = Field(
        min_length=1
    )

    value: Any


class TrainingIntervention(_Strict):
    kind: Literal["training"] = "training"

    changes: list[TrainingChange] = Field(
        min_length=1
    )

    @model_validator(mode="after")
    def validate_unique_paths(self):
        paths = [change.path for change in self.changes]

        if len(paths) != len(set(paths)):
            raise ValueError(
                "training intervention contains duplicate paths"
            )

        return self


@dataclass(frozen=True)
class TrainingRunRequest:
    experiment_id: str
    seed: int
    config: TrainingConfig
    changes: tuple[TrainingChange, ...] = ()
    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class TrainingRunResult:
    experiment_id: str
    status: str
    checkpoint_path: str | None = None
    evaluation_id: str | None = None
    metrics: dict[str, float] = field(
        default_factory=dict
    )
    history: list[dict[str, Any]] = field(
        default_factory=list
    )
    artifacts: dict[str, str] = field(
        default_factory=dict
    )
    error: dict[str, Any] | None = None


def set_nested_value(
    obj: dict[str, Any],
    path: str,
    value: Any,
) -> None:
    """
    Set a dotted configuration path.

    Example:
        set_nested_value(
            config,
            "optimizer.learning_rate",
            1e-4,
        )
    """

    parts = path.split(".")

    if not all(parts):
        raise TrainingConfigError(
            f"invalid configuration path: {path!r}"
        )

    current = obj

    for part in parts[:-1]:
        if part not in current:
            raise TrainingConfigError(
                f"unknown configuration path: {path!r}"
            )

        if not isinstance(current[part], dict):
            raise TrainingConfigError(
                f"configuration path is not an object: {path!r}"
            )

        current = current[part]

    leaf = parts[-1]

    if leaf not in current:
        raise TrainingConfigError(
            f"unknown configuration field: {path!r}"
        )

    current[leaf] = value


def apply_training_changes(
    config: TrainingConfig,
    changes: list[TrainingChange],
) -> TrainingConfig:
    """
    Apply controlled changes to a baseline TrainingConfig.

    The baseline object is never mutated.
    """

    payload = config.model_dump(
        mode="python"
    )

    for change in changes:
        set_nested_value(
            payload,
            change.path,
            change.value,
        )

    try:
        return TrainingConfig.model_validate(
            payload
        )
    except Exception as exc:
        raise TrainingConfigError(
            f"invalid training configuration after "
            f"applying changes: {exc}"
        ) from exc
