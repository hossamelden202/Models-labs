
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

from modellab.experiments.runner import ExperimentOutcome, FamilyRun
from modellab.experiments.spec import ExperimentSpec, experiment_id
from modellab.experiments.detection_training import train_detection


def _load_baseline(
    store,
    family_id: str,
) -> dict[str, Any] | None:
    path = (
        store.root
        / "experiments"
        / family_id
        / "baseline.json"
    )

    if not path.exists():
        return None

    payload = json.loads(path.read_text())

    if not isinstance(payload, dict):
        return None

    metrics = payload.get("metrics")

    if not isinstance(metrics, dict):
        return None

    if metrics.get("task") != "detection":
        return None

    return payload


def _jsonable(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, Path):
        return str(value)

    try:
        from dataclasses import asdict, is_dataclass

        if is_dataclass(value) and not isinstance(value, type):
            return _jsonable(asdict(value))
    except Exception:
        pass

    if isinstance(value, dict):
        return {
            str(k): _jsonable(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]

    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass

    return value


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None

        result = float(value)

        if not math.isfinite(result):
            return None

        return result

    except (TypeError, ValueError):
        return None


def _comparison(
    baseline: Any,
    trained_metrics: dict[str, Any],
) -> dict[str, Any]:
    if baseline is None:
        return {
            "available": False,
            "reason": "no compatible detection baseline was supplied",
        }

    # baseline.json is loaded as a JSON dictionary.
    # Its detection metrics live under the "metrics" key.
    if isinstance(baseline, dict):
        baseline_metrics = baseline.get("metrics")
    else:
        baseline_metrics = getattr(baseline, "metrics", None)

    if not isinstance(baseline_metrics, dict):
        return {
            "available": False,
            "reason": "no compatible detection baseline was supplied",
        }

    if baseline_metrics.get("task") != "detection":
        return {
            "available": False,
            "reason": "no compatible detection baseline was supplied",
        }

    metric_names = (
        "precision",
        "recall",
        "f1",
        "mean_iou",
    )

    baseline_values = {}
    trained_values = {}
    delta = {}

    for name in metric_names:
        baseline_value = _safe_float(baseline_metrics.get(name))
        trained_value = _safe_float(trained_metrics.get(name))

        baseline_values[name] = baseline_value
        trained_values[name] = trained_value

        if baseline_value is not None and trained_value is not None:
            delta[name] = trained_value - baseline_value
        else:
            delta[name] = None

    return {
        "available": True,
        "baseline": baseline_values,
        "trained": trained_values,
        "delta": delta,
    }



def _train_one(
    store,
    family_id: str,
    spec: ExperimentSpec,
    *,
    model,
    dataset,
    force: bool,
) -> ExperimentOutcome:
    exp_id = experiment_id(spec)

    exp_dir = (
        store.root
        / "experiments"
        / family_id
        / exp_id
    )

    result_path = exp_dir / "result.json"

    if result_path.exists() and not force:
        result = json.loads(result_path.read_text())

        return ExperimentOutcome(
            experiment_id=exp_id,
            status=result.get("status", "completed"),
            directory=exp_dir,
            result=result,
        )

    exp_dir.mkdir(parents=True, exist_ok=True)

    training_dir = exp_dir / "training"
    training_dir.mkdir(parents=True, exist_ok=True)

    train_images = (
        store.root
        / "detection_training_data"
        / "train"
        / "images"
    )

    val_images = (
        store.root
        / "detection_training_data"
        / "val"
        / "images"
    )

    if not train_images.exists():
        raise FileNotFoundError(
            f"Detection training images not found: {train_images}"
        )

    if not val_images.exists():
        raise FileNotFoundError(
            f"Detection validation images not found: {val_images}"
        )

    class_names = list(
        getattr(model.spec, "class_names", None)
        or []
    )

    if not class_names:
        raise ValueError(
            "Detection model spec has no class_names"
        )

    model_path = Path(model.spec.path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Detection model checkpoint not found: {model_path}"
        )

    try:
        training_result = train_detection(
            model_path=model_path,
            train_images=train_images,
            val_images=val_images,
            class_names=class_names,
            config=spec.intervention,
            output_dir=training_dir,
            device="auto",
        )

        training_payload = _jsonable(
            getattr(training_result, "payload", training_result)
        )

        if not isinstance(training_payload, dict):
            training_payload = {
                "result": training_payload,
            }

        checkpoint = training_payload.get("best_checkpoint")

        if not checkpoint:
            checkpoint = training_payload.get("best")

        # Ultralytics writes the checkpoint to:
        # <output_dir>/weights/best.pt
        #
        # Some train_detection() result objects do not expose that
        # path in their payload even though training completed
        # successfully. Fall back to the actual persisted checkpoint.
        if not checkpoint:
            checkpoint = training_dir / "weights" / "best.pt"

        checkpoint = Path(checkpoint)

        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Best detection checkpoint not found: {checkpoint}"
            )

        from ultralytics import YOLO

        from modellab.evaluation.detection_engine import (
            DetectionEvalConfig,
            run_detection_evaluation,
        )
        from modellab.evaluation.torch_model import TorchImageClassifier

        trained_yolo = YOLO(str(checkpoint))
        trained_module = trained_yolo.model

        trained_spec = model.spec.model_copy(deep=True)
        trained_spec.path = checkpoint
        trained_spec.source = "module"
        trained_spec.task = "detection"
        trained_spec.num_classes = len(class_names)
        trained_spec.class_names = class_names

        trained_model = TorchImageClassifier(
            trained_spec,
            device="auto",
            module=trained_module,
        )

        evaluation_id = f"{exp_id}-trained"

        # force=True means this experiment is being rerun. The evaluation
        # store does not allow creating the same evaluation ID twice, so
        # remove the previous evaluation artifact before recomputing it.
        evaluation_dir = store.root / "evaluations" / evaluation_id

        if force and evaluation_dir.exists():
            shutil.rmtree(evaluation_dir)

        from modellab.evaluation.detection_dataset import YOLODetectionDataset
        from modellab.evaluation.preprocessing import PreprocessConfig

        intervention_values = {
            change.path: change.value
            for change in spec.intervention.changes
        }

        eval_width = int(
            intervention_values.get("data.input_width", 224)
        )
        eval_height = int(
            intervention_values.get("data.input_height", eval_width)
        )

        evaluation_dataset = YOLODetectionDataset(
            root=dataset.root,
            class_names=list(model.spec.class_names),
            dataset_id=f"{dataset.dataset_id}-evaluation-{exp_id}",
            preprocess=PreprocessConfig(
                resize=(eval_height, eval_width),
            ),
        )

        evaluation = run_detection_evaluation(
            trained_model,
            evaluation_dataset,
            store,
            DetectionEvalConfig(
                batch_size=8,
                num_workers=0,
                seed=spec.seed,
                confidence_threshold=0.25,
                iou_threshold=0.50,
            ),
            evaluation_id=evaluation_id,
        )

        trained_metrics = dict(evaluation.metrics)

        baseline = _load_baseline(
            store,
            family_id,
        )

        comparison = _comparison(
            baseline,
            trained_metrics,
        )

        result = {
            "status": "completed",
            "task": "detection",
            "cache_key": exp_id,
            "checkpoint": str(checkpoint),
            "training": training_payload,
            "evaluation": _jsonable(
                evaluation.to_dict()
                if hasattr(evaluation, "to_dict")
                else {
                    "evaluation_id": getattr(
                        evaluation,
                        "evaluation_id",
                        f"{exp_id}-trained",
                    ),
                    "metrics": trained_metrics,
                }
            ),
            "comparison": comparison,
        }

    except Exception as exc:
        result = {
            "status": "failed",
            "task": "detection",
            "cache_key": exp_id,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }

        result_path.write_text(
            json.dumps(
                _jsonable(result),
                indent=2,
                sort_keys=True,
            )
        )

        return ExperimentOutcome(
            experiment_id=exp_id,
            status="failed",
            directory=exp_dir,
            result=result,
        )

    result_path.write_text(
        json.dumps(
            _jsonable(result),
            indent=2,
            sort_keys=True,
        )
    )

    return ExperimentOutcome(
        experiment_id=exp_id,
        status="completed",
        directory=exp_dir,
        result=result,
    )


def run_detection_family(
    store,
    family_id: str,
    specs: list[ExperimentSpec],
    *,
    model=None,
    dataset=None,
    analysis=None,
    force: bool = False,
    stop_on_failure: bool = False,
    alpha: float = 0.05,
) -> FamilyRun:
    if model is None:
        raise ValueError("model is required for detection training")

    if dataset is None:
        raise ValueError("dataset is required for detection training")

    outcomes: list[ExperimentOutcome] = []

    for spec in specs:
        try:
            outcome = _train_one(
                store,
                family_id,
                spec,
                model=model,
                dataset=dataset,
                force=force,
            )

        except Exception as exc:
            exp_id = experiment_id(spec)
            exp_dir = (
                store.root
                / "experiments"
                / family_id
                / exp_id
            )
            exp_dir.mkdir(parents=True, exist_ok=True)

            result = {
                "status": "failed",
                "task": "detection",
                "cache_key": exp_id,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            }

            (exp_dir / "result.json").write_text(
                json.dumps(
                    result,
                    indent=2,
                    sort_keys=True,
                )
            )

            outcome = ExperimentOutcome(
                experiment_id=exp_id,
                status="failed",
                directory=exp_dir,
                result=result,
            )

        outcomes.append(outcome)

        if stop_on_failure and outcome.status != "completed":
            break

    family_dir = (
        store.root
        / "experiments"
        / family_id
    )

    report = {
        "family_id": family_id,
        "task": "detection",
        "num_specs": len(specs),
        "num_outcomes": len(outcomes),
        "outcomes": [
            {
                "experiment_id": outcome.experiment_id,
                "status": outcome.status,
                "directory": str(outcome.directory),
            }
            for outcome in outcomes
        ],
    }

    report_path = family_dir / "detection_family_report.json"
    report_path.write_text(
        json.dumps(
            _jsonable(report),
            indent=2,
            sort_keys=True,
        )
    )

    return FamilyRun(
        outcomes=outcomes,
        report=report,
        directory=family_dir,
    )


__all__ = [
    "run_detection_family",
]
