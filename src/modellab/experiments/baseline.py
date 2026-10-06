
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from modellab.core.errors import EvaluationError
from modellab.core.results import ArtifactStore


class BaselineError(EvaluationError):
    """Raised when an experiment baseline is invalid."""


def _evaluation_dir(
    store: ArtifactStore,
    evaluation_id: str,
) -> Path:
    if not evaluation_id:
        raise BaselineError(
            "evaluation_id must not be empty"
        )

    if (
        Path(evaluation_id).name != evaluation_id
        or evaluation_id in {".", ".."}
    ):
        raise BaselineError(
            f"invalid evaluation id: {evaluation_id!r}"
        )

    return (
        store.root
        / "evaluations"
        / evaluation_id
    )


def _load_json(
    path: Path,
    description: str,
) -> Any:
    if not path.exists():
        raise BaselineError(
            f"{description} does not exist: {path}"
        )

    try:
        return json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except Exception as exc:
        raise BaselineError(
            f"could not read {description}: {path}"
        ) from exc


def _detect_task(
    metadata: dict[str, Any],
    metrics: dict[str, Any],
) -> str:
    task = metadata.get("task")

    if task in {
        "classification",
        "detection",
    }:
        return str(task)

    task = metrics.get("task")

    if task in {
        "classification",
        "detection",
    }:
        return str(task)

    raise BaselineError(
        "evaluation does not specify a supported task"
    )


def _load_evaluation(
    store: ArtifactStore,
    evaluation_id: str,
) -> dict[str, Any]:

    directory = _evaluation_dir(
        store,
        evaluation_id,
    )

    if not directory.exists():
        raise BaselineError(
            f"evaluation {evaluation_id!r} "
            f"does not exist under {store.root}"
        )

    metadata = _load_json(
        directory / "metadata.json",
        "evaluation metadata",
    )

    metrics = _load_json(
        directory / "metrics.json",
        "evaluation metrics",
    )

    if not isinstance(metadata, dict):
        raise BaselineError(
            "evaluation metadata must be an object"
        )

    if not isinstance(metrics, dict):
        raise BaselineError(
            "evaluation metrics must be an object"
        )

    task = _detect_task(
        metadata,
        metrics,
    )

    prediction_path = None

    if task == "detection":
        candidate = (
            directory
            / "predictions.json"
        )

        if not candidate.exists():
            raise BaselineError(
                "detection evaluation requires "
                "predictions.json"
            )

        prediction_path = candidate

    else:
        candidate = (
            directory
            / "predictions.parquet"
        )

        if not candidate.exists():
            raise BaselineError(
                "classification evaluation requires "
                "predictions.parquet"
            )

        prediction_path = candidate

    predictions = _load_json(
        prediction_path,
        "detection predictions",
    ) if task == "detection" else None

    return {
        "evaluation_id": evaluation_id,
        "task": task,
        "metadata": metadata,
        "metrics": metrics,
        "prediction_artifact": (
            prediction_path
            .relative_to(store.root)
            .as_posix()
        ),
        "predictions": predictions,
    }


def create_baseline(
    store: ArtifactStore,
    family_id: str,
    evaluation_id: str,
) -> dict[str, Any]:

    if not family_id:
        raise BaselineError(
            "family_id must not be empty"
        )

    evaluation = _load_evaluation(
        store,
        evaluation_id,
    )

    relative = (
        Path("experiments")
        / family_id
        / "baseline.json"
    )

    existing = store.resolve(
        relative
    )

    if existing.exists():
        raise BaselineError(
            f"baseline already exists for "
            f"family {family_id!r}"
        )

    payload = {
        "family_id": family_id,
        "evaluation_id": evaluation_id,
        "task": evaluation["task"],
        "metrics": evaluation["metrics"],
        "metadata": evaluation["metadata"],
        "prediction_artifact": (
            evaluation["prediction_artifact"]
        ),
    }

    store.write_json(
        relative,
        payload,
    )

    return payload


def load_baseline(
    store: ArtifactStore,
    family_id: str,
) -> dict[str, Any]:

    if not family_id:
        raise BaselineError(
            "family_id must not be empty"
        )

    relative = (
        Path("experiments")
        / family_id
        / "baseline.json"
    )

    path = store.resolve(
        relative
    )

    if not path.exists():
        raise BaselineError(
            f"baseline for family "
            f"{family_id!r} does not exist"
        )

    payload = _load_json(
        path,
        "baseline",
    )

    if not isinstance(
        payload,
        dict,
    ):
        raise BaselineError(
            "baseline must be a JSON object"
        )

    task = payload.get("task")

    if task not in {
        "classification",
        "detection",
    }:
        raise BaselineError(
            f"unsupported baseline task: {task!r}"
        )

    return payload


def compare_baseline(
    baseline: dict[str, Any],
    candidate_metrics: dict[str, Any],
) -> dict[str, Any]:

    if not isinstance(
        baseline,
        dict,
    ):
        raise BaselineError(
            "baseline must be an object"
        )

    if not isinstance(
        candidate_metrics,
        dict,
    ):
        raise BaselineError(
            "candidate metrics must be an object"
        )

    baseline_task = baseline.get(
        "task"
    )

    candidate_task = candidate_metrics.get(
        "task"
    )

    if candidate_task is None:
        candidate_task = baseline_task

    if baseline_task != candidate_task:
        raise BaselineError(
            "cannot compare different tasks: "
            f"baseline={baseline_task!r}, "
            f"candidate={candidate_task!r}"
        )

    if baseline_task == "detection":

        metric_names = (
            "precision",
            "recall",
            "f1",
            "mean_iou",
        )

        baseline_metrics = (
            baseline.get("metrics")
            or {}
        )

        candidate = {}

        deltas = {}

        for name in metric_names:

            if name not in baseline_metrics:
                continue

            if name not in candidate_metrics:
                continue

            base_value = float(
                baseline_metrics[name]
            )

            candidate_value = float(
                candidate_metrics[name]
            )

            candidate[name] = (
                candidate_value
            )

            deltas[name] = (
                candidate_value
                - base_value
            )

        # F1 is the primary aggregate detection metric.
        base_f1 = baseline_metrics.get(
            "f1"
        )

        candidate_f1 = candidate_metrics.get(
            "f1"
        )

        if (
            base_f1 is not None
            and candidate_f1 is not None
        ):
            base_f1 = float(base_f1)
            candidate_f1 = float(candidate_f1)

            if candidate_f1 > base_f1:
                verdict = "improved"
            elif candidate_f1 < base_f1:
                verdict = "degraded"
            else:
                verdict = "no_change"
        else:
            verdict = "unknown"

        return {
            "task": "detection",
            "verdict": verdict,
            "primary_metric": "f1",
            "baseline": {
                name: baseline_metrics.get(name)
                for name in metric_names
            },
            "candidate": candidate,
            "delta": deltas,
        }

    # --------------------------------------------------------
    # Classification comparison.
    # --------------------------------------------------------

    metric_names = (
        "accuracy",
        "macro_f1",
        "macro_precision",
        "macro_recall",
        "weighted_f1",
        "balanced_accuracy",
    )

    baseline_metrics = (
        baseline.get("metrics")
        or {}
    )

    deltas = {}

    for name in metric_names:

        if name not in baseline_metrics:
            continue

        if name not in candidate_metrics:
            continue

        deltas[name] = (
            float(candidate_metrics[name])
            - float(baseline_metrics[name])
        )

    base_f1 = baseline_metrics.get(
        "macro_f1"
    )

    candidate_f1 = candidate_metrics.get(
        "macro_f1"
    )

    if (
        base_f1 is not None
        and candidate_f1 is not None
    ):
        base_f1 = float(base_f1)
        candidate_f1 = float(candidate_f1)

        if candidate_f1 > base_f1:
            verdict = "improved"
        elif candidate_f1 < base_f1:
            verdict = "degraded"
        else:
            verdict = "no_change"
    else:
        verdict = "unknown"

    return {
        "task": "classification",
        "verdict": verdict,
        "primary_metric": "macro_f1",
        "baseline": {
            name: baseline_metrics.get(name)
            for name in metric_names
        },
        "candidate": {
            name: candidate_metrics.get(name)
            for name in metric_names
            if name in candidate_metrics
        },
        "delta": deltas,
    }
