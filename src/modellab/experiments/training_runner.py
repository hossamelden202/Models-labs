
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from modellab.experiments.training_adapter import (
    build_training_dataloaders,
    count_parameters,
    count_trainable_parameters,
    load_model_for_training,
)
from modellab.experiments.training_config import (
    TrainingConfig,
    TrainingIntervention,
    apply_training_changes,
)
from modellab.experiments.training_engine import (
    EpochResult,
    TrainingResult,
    train_classifier,
)


class TrainingRunnerError(RuntimeError):
    """Raised when a ModelLab training experiment cannot run."""


@dataclass
class TrainingExperimentResult:
    """
    Result of one actual training experiment.

    The checkpoint paths are deliberately explicit so the existing
    experiment runner can subsequently evaluate the trained model.
    """

    best_checkpoint: str
    final_checkpoint: str
    best_epoch: int
    best_metric: float
    total_epochs: int
    stopped_early: bool
    total_parameters: int
    trainable_parameters: int
    history: list[dict[str, Any]]
    config: dict[str, Any]


def run_training_experiment(
    *,
    model_spec: Any,
    train_dataset: Any,
    validation_dataset: Any,
    intervention: TrainingIntervention | None = None,
    base_config: TrainingConfig | None = None,
    output_dir: str | Path,
    device: str = "auto",
    on_epoch: Callable[[EpochResult], None] | None = None,
) -> TrainingExperimentResult:
    """
    Execute one real ModelLab training experiment.

    Inputs are existing ModelLab objects. This function does not
    implement a second model registry or dataset registry.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if base_config is None:
        config = TrainingConfig()
    else:
        config = base_config.model_copy(deep=True)

    # Apply the experiment's requested changes.
    if intervention is not None:
        config = apply_training_changes(
            config,
            intervention.changes,
        )

    # ---------------------------------------------------------
    # Load existing ModelLab model
    # ---------------------------------------------------------

    model = load_model_for_training(
        model_spec,
        device=device,
    )

    total_parameters = count_parameters(model)

    trainable_parameters = count_trainable_parameters(model)

    # ---------------------------------------------------------
    # Build existing ModelLab datasets into training loaders
    # ---------------------------------------------------------

    loaders = build_training_dataloaders(
        train_dataset,
        validation_dataset,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        shuffle=True,
    )

    # ---------------------------------------------------------
    # Train
    # ---------------------------------------------------------

    result: TrainingResult = train_classifier(
        model=model,
        train_loader=loaders.train,
        val_loader=loaders.validation,
        config=config,
        num_classes=_infer_num_classes(
            model_spec,
            train_dataset,
        ),
        output_dir=output_dir,
        device=device,
        on_epoch=on_epoch,
    )

    # ---------------------------------------------------------
    # Persist a compact experiment manifest
    # ---------------------------------------------------------

    history = [
        asdict(epoch)
        for epoch in result.history
    ]

    return TrainingExperimentResult(
        best_checkpoint=result.best_checkpoint,
        final_checkpoint=result.final_checkpoint,
        best_epoch=result.best_epoch,
        best_metric=result.best_metric,
        total_epochs=result.total_epochs,
        stopped_early=result.stopped_early,
        total_parameters=total_parameters,
        trainable_parameters=trainable_parameters,
        history=history,
        config=config.model_dump(),
    )


def _infer_num_classes(
    model_spec: Any,
    dataset: Any,
) -> int:
    """
    Prefer ModelLab model metadata, then dataset metadata.

    This keeps the training runner compatible with the existing
    ModelLab model/dataset objects without introducing a new schema.
    """

    candidates: list[Any] = []

    for attribute in (
        "num_classes",
        "class_count",
    ):
        if hasattr(model_spec, attribute):
            candidates.append(
                getattr(model_spec, attribute)
            )

    if hasattr(dataset, "num_classes"):
        candidates.append(
            getattr(dataset, "num_classes")
        )

    if hasattr(dataset, "class_names"):
        class_names = getattr(
            dataset,
            "class_names",
        )

        if class_names is not None:
            try:
                candidates.append(
                    len(class_names)
                )
            except TypeError:
                pass

    for value in candidates:
        try:
            value = int(value)
        except (TypeError, ValueError):
            continue

        if value > 0:
            return value

    # Last-resort inference from observed labels.
    labels: set[int] = set()

    try:
        length = len(dataset)
    except Exception as exc:
        raise TrainingRunnerError(
            "Could not determine dataset length while "
            "inferring the number of classes."
        ) from exc

    for index in range(length):
        sample = dataset.get_sample(index)

        if hasattr(sample, "label"):
            label = sample.label
        elif isinstance(sample, dict):
            label = sample.get(
                "label",
                sample.get("target"),
            )
        elif isinstance(sample, (tuple, list)):
            label = sample[-1]
        else:
            continue

        try:
            labels.add(int(label))
        except (TypeError, ValueError):
            continue

    if not labels:
        raise TrainingRunnerError(
            "Could not infer num_classes from the model or dataset."
        )

    return max(labels) + 1


__all__ = [
    "TrainingExperimentResult",
    "TrainingRunnerError",
    "run_training_experiment",
]
