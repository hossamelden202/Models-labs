
from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from modellab.experiments.training_config import TrainingConfig


class TrainingEngineError(RuntimeError):
    """Raised when a training run cannot be completed."""


@dataclass
class EpochResult:
    epoch: int
    train_loss: float
    train_accuracy: float
    train_macro_f1: float
    train_balanced_accuracy: float
    val_loss: float
    val_accuracy: float
    val_macro_f1: float
    val_balanced_accuracy: float
    learning_rate: float


@dataclass
class TrainingResult:
    best_epoch: int
    best_metric: float
    best_checkpoint: str
    final_checkpoint: str
    history: list[EpochResult]
    stopped_early: bool
    total_epochs: int


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Deterministic behavior is preferred for experiment reproducibility.
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def _unwrap_output(output: Any) -> torch.Tensor:
    """
    Normalize common model outputs into a logits tensor.

    Supports:
      Tensor
      tuple/list whose first item is Tensor
      dict containing logits/output
    """
    if isinstance(output, torch.Tensor):
        return output

    if isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, torch.Tensor):
                return item

    if isinstance(output, dict):
        for key in ("logits", "output", "outputs"):
            value = output.get(key)
            if isinstance(value, torch.Tensor):
                return value

    raise TrainingEngineError(
        f"Model output could not be converted to logits: {type(output)!r}"
    )


def _extract_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract inputs and targets from common ModelLab/training batches.

    Supported forms:

      (inputs, targets)

      (ids, paths, labels, inputs)
      (ids, paths, labels, inputs, ...)
      dictionaries containing inputs/images/x and labels/targets/y

    The important ModelLab evaluation format is:

      (ids, paths, labels, batch)

    where `labels` is the target tensor and `batch` is the image tensor.
    """
    if isinstance(batch, dict):
        inputs = None
        targets = None

        for key in ("inputs", "images", "image", "x"):
            value = batch.get(key)
            if isinstance(value, torch.Tensor):
                inputs = value
                break

        for key in ("targets", "labels", "label", "y"):
            value = batch.get(key)
            if isinstance(value, torch.Tensor):
                targets = value
                break

        if inputs is not None and targets is not None:
            return inputs, targets

        raise TrainingEngineError(
            "Dictionary batch must contain tensor inputs/images/x "
            "and tensor targets/labels/y."
        )

    if not isinstance(batch, (tuple, list)):
        raise TrainingEngineError(
            f"Unsupported batch type: {type(batch)!r}"
        )

    tensors = [
        item for item in batch
        if isinstance(item, torch.Tensor)
    ]

    if len(batch) == 2:
        a, b = batch

        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            # Images normally have >= 3 dimensions; labels normally have <= 2.
            if a.ndim >= 3 and b.ndim <= 2:
                return a, b.long()

            if b.ndim >= 3 and a.ndim <= 2:
                return b, a.long()

            # Fall back to conventional (inputs, targets).
            return a, b.long()

    # ModelLab evaluation stream:
    # (ids, paths, labels, inputs)
    if len(batch) >= 4:
        candidates = [
            item for item in batch
            if isinstance(item, torch.Tensor)
        ]

        image_candidates = [
            item for item in candidates
            if item.ndim >= 3
        ]

        label_candidates = [
            item for item in candidates
            if item.ndim <= 2
        ]

        if image_candidates and label_candidates:
            return image_candidates[-1], label_candidates[0].long()

    # Generic fallback: identify by tensor dimensionality.
    image_candidates = [
        item for item in tensors
        if item.ndim >= 3
    ]

    label_candidates = [
        item for item in tensors
        if item.ndim <= 2
    ]

    if image_candidates and label_candidates:
        return image_candidates[-1], label_candidates[0].long()

    raise TrainingEngineError(
        "Could not identify image inputs and labels in training batch."
    )


def _classification_metrics(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    num_classes: int,
) -> tuple[float, float, float]:
    targets = targets.detach().cpu().numpy().astype(np.int64)
    predictions = predictions.detach().cpu().numpy().astype(np.int64)

    accuracy = float(np.mean(targets == predictions))

    f1_values: list[float] = []
    recall_values: list[float] = []

    for cls in range(num_classes):
        tp = int(np.sum((targets == cls) & (predictions == cls)))
        fp = int(np.sum((targets != cls) & (predictions == cls)))
        fn = int(np.sum((targets == cls) & (predictions != cls)))

        precision = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0.0
        )

        if precision + recall > 0:
            f1 = (
                2.0 * precision * recall
                / (precision + recall)
            )
        else:
            f1 = 0.0

        f1_values.append(f1)
        recall_values.append(recall)

    macro_f1 = float(np.mean(f1_values)) if f1_values else 0.0
    balanced_accuracy = (
        float(np.mean(recall_values))
        if recall_values
        else 0.0
    )

    return accuracy, macro_f1, balanced_accuracy


def _build_loss(
    config: TrainingConfig,
    num_classes: int,
    device: torch.device,
) -> nn.Module:
    loss_cfg = config.loss

    class_weights = None

    if loss_cfg.class_weights is not None:
        if len(loss_cfg.class_weights) != num_classes:
            raise TrainingEngineError(
                "loss.class_weights length must equal num_classes: "
                f"{len(loss_cfg.class_weights)} != {num_classes}"
            )

        class_weights = torch.tensor(
            loss_cfg.class_weights,
            dtype=torch.float32,
            device=device,
        )

    if loss_cfg.name in (
        "cross_entropy",
        "weighted_cross_entropy",
    ):
        return nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=float(loss_cfg.label_smoothing),
        )

    if loss_cfg.name == "focal":

        gamma = float(loss_cfg.focal_gamma)

        class FocalLoss(nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(
                self,
                logits: torch.Tensor,
                targets: torch.Tensor,
            ) -> torch.Tensor:
                ce = nn.functional.cross_entropy(
                    logits,
                    targets,
                    weight=class_weights,
                    label_smoothing=float(
                        loss_cfg.label_smoothing
                    ),
                    reduction="none",
                )

                pt = torch.exp(-ce)
                loss = ((1.0 - pt) ** gamma) * ce

                return loss.mean()

        return FocalLoss()

    raise TrainingEngineError(
        f"Unsupported loss: {loss_cfg.name!r}"
    )


def _build_optimizer(
    model: nn.Module,
    config: TrainingConfig,
) -> torch.optim.Optimizer:
    cfg = config.optimizer

    parameters = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    if not parameters:
        raise TrainingEngineError(
            "No trainable parameters remain after freezing."
        )

    name = cfg.name.lower()

    if name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=float(cfg.learning_rate),
            weight_decay=float(cfg.weight_decay),
            momentum=float(cfg.momentum),
        )

    if name == "adam":
        return torch.optim.Adam(
            parameters,
            lr=float(cfg.learning_rate),
            weight_decay=float(cfg.weight_decay),
        )

    if name == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=float(cfg.learning_rate),
            weight_decay=float(cfg.weight_decay),
        )

    if name == "rmsprop":
        return torch.optim.RMSprop(
            parameters,
            lr=float(cfg.learning_rate),
            weight_decay=float(cfg.weight_decay),
            momentum=float(cfg.momentum),
        )

    raise TrainingEngineError(
        f"Unsupported optimizer: {cfg.name!r}"
    )


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    *,
    steps_per_epoch: int,
) -> tuple[Any | None, bool]:
    """
    Returns:
        scheduler
        step_per_batch

    OneCycleLR must be stepped after every optimizer update.
    All other currently supported schedulers are epoch-based.
    """
    cfg = config.scheduler
    name = cfg.name.lower()

    if name == "none":
        return None, False

    if name == "step":
        return (
            torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=int(cfg.step_size),
                gamma=float(cfg.gamma),
            ),
            False,
        )

    if name == "cosine":
        return (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, int(config.epochs)),
                eta_min=float(cfg.min_lr),
            ),
            False,
        )

    if name == "cosine_warm_restarts":
        return (
            torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=max(1, int(cfg.step_size)),
                T_mult=1,
                eta_min=float(cfg.min_lr),
            ),
            False,
        )

    if name == "plateau":
        return (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=float(cfg.gamma),
                patience=max(0, int(cfg.step_size)),
                min_lr=float(cfg.min_lr),
            ),
            False,
        )

    if name == "one_cycle":
        total_steps = max(
            1,
            int(config.epochs)
            * max(1, int(steps_per_epoch)),
        )

        return (
            torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=float(cfg.min_lr)
                if float(cfg.min_lr) > float(config.optimizer.learning_rate)
                else float(config.optimizer.learning_rate),
                total_steps=total_steps,
            ),
            True,
        )

    raise TrainingEngineError(
        f"Unsupported scheduler: {cfg.name!r}"
    )


def _apply_freezing(
    model: nn.Module,
    config: TrainingConfig,
) -> None:
    mode = config.freeze.mode

    if mode == "none":
        for parameter in model.parameters():
            parameter.requires_grad = True
        return

    if mode == "all_but_head":
        for parameter in model.parameters():
            parameter.requires_grad = False

        found_head = False

        for name, parameter in model.named_parameters():
            lower = name.lower()

            if any(
                token in lower
                for token in (
                    "classifier",
                    "fc",
                    "head",
                    "linear",
                )
            ):
                parameter.requires_grad = True
                found_head = True

        if not found_head:
            raise TrainingEngineError(
                "freeze.mode='all_but_head' could not identify "
                "a classifier/head parameter."
            )

        return

    if mode == "backbone":
        for parameter in model.parameters():
            parameter.requires_grad = False

        found_head = False

        for name, parameter in model.named_parameters():
            lower = name.lower()

            if any(
                token in lower
                for token in (
                    "classifier",
                    "fc",
                    "head",
                    "linear",
                )
            ):
                parameter.requires_grad = True
                found_head = True

        if not found_head:
            raise TrainingEngineError(
                "freeze.mode='backbone' could not identify "
                "a classifier/head parameter."
            )

        return

    if mode == "all":
        for parameter in model.parameters():
            parameter.requires_grad = False
        return

    if mode == "last_n_blocks":
        # Conservative generic implementation:
        # freeze all parameters, then unfreeze parameters belonging
        # to the last N top-level parameter groups plus classifier/head.
        n = int(config.freeze.last_n_blocks)

        named = list(model.named_parameters())

        for _, parameter in named:
            parameter.requires_grad = False

        if n <= 0:
            raise TrainingEngineError(
                "freeze.last_n_blocks must be > 0."
            )

        prefixes: list[str] = []

        for name, _ in named:
            prefix = name.split(".")[0]

            if prefix not in prefixes:
                prefixes.append(prefix)

        selected = set(prefixes[-n:])

        for name, parameter in named:
            prefix = name.split(".")[0]

            lower = name.lower()

            if (
                prefix in selected
                or any(
                    token in lower
                    for token in (
                        "classifier",
                        "fc",
                        "head",
                        "linear",
                    )
                )
            ):
                parameter.requires_grad = True

        return

    raise TrainingEngineError(
        f"Unsupported freeze mode: {mode!r}"
    )


def _monitor_value(
    monitor: str,
    *,
    loss: float,
    accuracy: float,
    macro_f1: float,
    balanced_accuracy: float,
) -> float:
    values = {
        "loss": float(loss),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "balanced_accuracy": float(balanced_accuracy),
    }

    if monitor not in values:
        raise TrainingEngineError(
            f"Unsupported monitor: {monitor!r}"
        )

    return values[monitor]


def _is_better(
    value: float,
    best: float | None,
    monitor: str,
) -> bool:
    if best is None:
        return True

    if monitor == "loss":
        return value < best

    return value > best


def _save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    metric: float,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "epoch": int(epoch),
        "metric": float(metric),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(payload, path)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            default=str,
        )
    )


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    *,
    accumulation_steps: int,
    mixed_precision: bool,
    max_grad_norm: float | None,
    scheduler: Any,
    scheduler_per_batch: bool,
) -> tuple[float, float, float, float]:
    model.train()

    running_loss = 0.0
    total = 0

    all_targets: list[torch.Tensor] = []
    all_predictions: list[torch.Tensor] = []

    optimizer.zero_grad(set_to_none=True)

    for batch_index, batch in enumerate(loader):
        inputs, targets = _extract_batch(batch)

        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).long()

        if targets.ndim > 1:
            targets = targets.view(-1)

        autocast_device = device.type

        use_amp = bool(
            mixed_precision
            and device.type in ("cuda", "cpu")
        )

        with torch.autocast(
            device_type=autocast_device,
            enabled=use_amp,
            dtype=(
                torch.float16
                if device.type == "cuda"
                else torch.bfloat16
            ),
        ):
            output = model(inputs)
            logits = _unwrap_output(output)

            loss = criterion(logits, targets)
            scaled_loss = loss / accumulation_steps

        scaler.scale(scaled_loss).backward()

        should_step = (
            (batch_index + 1) % accumulation_steps == 0
            or (batch_index + 1) == len(loader)
        )

        if should_step:
            if max_grad_norm is not None:
                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(max_grad_norm),
                )

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            if scheduler is not None and scheduler_per_batch:
                scheduler.step()

        batch_size = int(targets.shape[0])

        running_loss += (
            float(loss.detach().cpu())
            * batch_size
        )

        total += batch_size

        predictions = torch.argmax(
            logits.detach(),
            dim=1,
        )

        all_targets.append(targets.detach().cpu())
        all_predictions.append(predictions.detach().cpu())

    if total == 0:
        raise TrainingEngineError(
            "Training loader produced zero samples."
        )

    targets_all = torch.cat(all_targets)
    predictions_all = torch.cat(all_predictions)

    accuracy, macro_f1, balanced_accuracy = (
        _classification_metrics(
            targets_all,
            predictions_all,
            int(
                max(
                    int(targets_all.max().item()) + 1,
                    int(predictions_all.max().item()) + 1,
                )
            ),
        )
    )

    return (
        running_loss / total,
        accuracy,
        macro_f1,
        balanced_accuracy,
    )


@torch.no_grad()
def _validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    mixed_precision: bool,
    num_classes: int,
) -> tuple[float, float, float, float]:
    model.eval()

    running_loss = 0.0
    total = 0

    all_targets: list[torch.Tensor] = []
    all_predictions: list[torch.Tensor] = []

    for batch in loader:
        inputs, targets = _extract_batch(batch)

        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).long()

        if targets.ndim > 1:
            targets = targets.view(-1)

        use_amp = bool(
            mixed_precision
            and device.type in ("cuda", "cpu")
        )

        with torch.autocast(
            device_type=device.type,
            enabled=use_amp,
            dtype=(
                torch.float16
                if device.type == "cuda"
                else torch.bfloat16
            ),
        ):
            output = model(inputs)
            logits = _unwrap_output(output)
            loss = criterion(logits, targets)

        batch_size = int(targets.shape[0])

        running_loss += (
            float(loss.detach().cpu())
            * batch_size
        )

        total += batch_size

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        all_targets.append(targets.detach().cpu())
        all_predictions.append(predictions.detach().cpu())

    if total == 0:
        raise TrainingEngineError(
            "Validation loader produced zero samples."
        )

    targets_all = torch.cat(all_targets)
    predictions_all = torch.cat(all_predictions)

    accuracy, macro_f1, balanced_accuracy = (
        _classification_metrics(
            targets_all,
            predictions_all,
            num_classes,
        )
    )

    return (
        running_loss / total,
        accuracy,
        macro_f1,
        balanced_accuracy,
    )


def _resolve_device(
    device: str,
) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")

        if torch.backends.mps.is_available():
            return torch.device("mps")

        return torch.device("cpu")

    return torch.device(device)


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
    num_classes: int,
    output_dir: str | Path,
    *,
    device: str = "auto",
    on_epoch: Callable[[EpochResult], None] | None = None,
) -> TrainingResult:
    """
    Train a classification model using a ModelLab TrainingConfig.

    This is intentionally model-agnostic. ModelLab is responsible for
    constructing the model and datasets; this engine owns optimization,
    losses, schedulers, freezing, metrics, checkpoints and history.
    """
    _set_seed(int(config.seed))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch_device = _resolve_device(device)

    model = model.to(torch_device)

    _apply_freezing(model, config)

    criterion = _build_loss(
        config,
        num_classes,
        torch_device,
    )

    optimizer = _build_optimizer(
        model,
        config,
    )

    accumulation_steps = max(
        1,
        int(config.gradient_accumulation_steps),
    )

    scheduler, scheduler_per_batch = _build_scheduler(
        optimizer,
        config,
        steps_per_epoch=max(
            1,
            math.ceil(
                len(train_loader)
                / accumulation_steps
            ),
        ),
    )

    use_amp = bool(
        config.mixed_precision
        and torch_device.type == "cuda"
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    history: list[EpochResult] = []

    best_metric: float | None = None
    best_epoch = 0

    best_checkpoint = (
        output_dir / "best.pt"
    )

    final_checkpoint = (
        output_dir / "last.pt"
    )

    patience = config.early_stopping_patience

    bad_epochs = 0
    stopped_early = False

    for epoch in range(1, int(config.epochs) + 1):
        train_loss, train_accuracy, train_macro_f1, train_balanced = (
            _train_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                torch_device,
                scaler,
                accumulation_steps=accumulation_steps,
                mixed_precision=use_amp,
                max_grad_norm=(
                    float(config.max_grad_norm)
                    if config.max_grad_norm is not None
                    else None
                ),
                scheduler=scheduler,
                scheduler_per_batch=scheduler_per_batch,
            )
        )

        val_loss, val_accuracy, val_macro_f1, val_balanced = (
            _validate_epoch(
                model,
                val_loader,
                criterion,
                torch_device,
                mixed_precision=use_amp,
                num_classes=num_classes,
            )
        )

        if scheduler is not None and not scheduler_per_batch:
            if config.scheduler.name == "plateau":
                scheduler.step(val_macro_f1)
            else:
                scheduler.step()

        lr = float(
            optimizer.param_groups[0]["lr"]
        )

        epoch_result = EpochResult(
            epoch=epoch,
            train_loss=train_loss,
            train_accuracy=train_accuracy,
            train_macro_f1=train_macro_f1,
            train_balanced_accuracy=train_balanced,
            val_loss=val_loss,
            val_accuracy=val_accuracy,
            val_macro_f1=val_macro_f1,
            val_balanced_accuracy=val_balanced,
            learning_rate=lr,
        )

        history.append(epoch_result)

        monitor = config.monitor

        current_metric = _monitor_value(
            monitor,
            loss=val_loss,
            accuracy=val_accuracy,
            macro_f1=val_macro_f1,
            balanced_accuracy=val_balanced,
        )

        improved = _is_better(
            current_metric,
            best_metric,
            monitor,
        )

        if improved:
            best_metric = current_metric
            best_epoch = epoch
            bad_epochs = 0

            _save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                current_metric,
                best_checkpoint,
            )
        else:
            bad_epochs += 1

        _save_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch,
            current_metric,
            final_checkpoint,
        )

        print(
            "[TRAIN] "
            f"epoch={epoch}/{config.epochs} "
            f"train_loss={train_loss:.5f} "
            f"val_loss={val_loss:.5f} "
            f"val_acc={val_accuracy:.4f} "
            f"val_macro_f1={val_macro_f1:.4f} "
            f"val_balanced={val_balanced:.4f} "
            f"lr={lr:.3e} "
            f"{'BEST' if improved else ''}",
            flush=True,
        )

        if on_epoch is not None:
            on_epoch(epoch_result)

        if (
            patience is not None
            and patience > 0
            and bad_epochs >= int(patience)
        ):
            stopped_early = True

            print(
                "[TRAIN] "
                f"early stopping at epoch {epoch}",
                flush=True,
            )

            break

    if best_metric is None:
        raise TrainingEngineError(
            "Training completed without producing a valid best metric."
        )

    result = TrainingResult(
        best_epoch=best_epoch,
        best_metric=float(best_metric),
        best_checkpoint=str(best_checkpoint),
        final_checkpoint=str(final_checkpoint),
        history=history,
        stopped_early=stopped_early,
        total_epochs=len(history),
    )

    _write_json(
        output_dir / "history.json",
        [asdict(item) for item in history],
    )

    _write_json(
        output_dir / "result.json",
        asdict(result),
    )

    _write_json(
        output_dir / "training_config.json",
        config.model_dump(),
    )

    return result


__all__ = [
    "EpochResult",
    "TrainingEngineError",
    "TrainingResult",
    "train_classifier",
]
