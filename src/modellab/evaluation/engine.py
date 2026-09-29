import contextlib
import os
import platform
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from pydantic import BaseModel, ConfigDict, Field
from torch.utils.data import DataLoader, Dataset

import modellab
from modellab.core.errors import EvaluationError
from modellab.core.models import ModelMetadata
from modellab.core.results import ArtifactStore, EvaluationResult
from modellab.evaluation.image_dataset import ImageClassificationDataset
from modellab.evaluation.metrics import compute_metrics
from modellab.evaluation.torch_model import TorchImageClassifier
from modellab.utils import get_logger, set_seed

log = get_logger("evaluation")


class EvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_size: int = Field(32, ge=1)
    num_workers: int = Field(0, ge=0)
    seed: int = Field(42, ge=0, lt=2**32)


class _ImageStream(Dataset):
    def __init__(self, dataset: ImageClassificationDataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        sample = self.dataset.get_sample(index)
        if not isinstance(sample.input, torch.Tensor):
            raise EvaluationError(
                "dataset samples must be tensors, build the dataset with a preprocess config"
            )
        if sample.label is None:
            raise EvaluationError(f"sample {sample.sample_id} has no label")
        return sample.sample_id, sample.metadata.get("path", ""), sample.label, sample.input


def _collate(items):
    ids, paths, labels, tensors = zip(*items, strict=True)
    try:
        batch = torch.stack(tensors)
    except RuntimeError as exc:
        raise EvaluationError(
            "images in one batch have different shapes, set resize in the preprocess config"
        ) from exc
    return list(ids), list(paths), torch.tensor(labels), batch


def _schema(k: int) -> pa.Schema:
    return pa.schema(
        [
            ("sample_id", pa.string()),
            ("path", pa.string()),
            ("true_label", pa.int32()),
            ("predicted_label", pa.int32()),
            ("confidence", pa.float32()),
            ("correct", pa.bool_()),
            ("class_probabilities", pa.list_(pa.float32(), k)),
        ]
    )


def _table(ids, paths, labels, pred, probs, schema) -> pa.Table:
    n, k = probs.shape
    confidence = probs[np.arange(n), pred]
    return pa.table(
        {
            "sample_id": pa.array(ids, pa.string()),
            "path": pa.array(paths, pa.string()),
            "true_label": pa.array(labels.astype(np.int32), pa.int32()),
            "predicted_label": pa.array(pred.astype(np.int32), pa.int32()),
            "confidence": pa.array(confidence, pa.float32()),
            "correct": pa.array(pred == labels, pa.bool_()),
            "class_probabilities": pa.FixedSizeListArray.from_arrays(
                pa.array(probs.ravel(), pa.float32()), k
            ),
        },
        schema=schema,
    )


def _check_compatible(meta: ModelMetadata, names: list[str]) -> None:
    if len(names) != meta.num_classes:
        raise EvaluationError(
            f"dataset has {len(names)} classes {names} but the model has {meta.num_classes}, "
            "set class_names in the model spec if the dataset is missing some classes"
        )
    if meta.class_names is not None and list(meta.class_names) != list(names):
        raise EvaluationError(
            f"model class names {meta.class_names} do not match dataset class order {names}"
        )


def _new_id() -> str:
    return f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:6]}"


def run_evaluation(
    model: TorchImageClassifier,
    dataset: ImageClassificationDataset,
    store: ArtifactStore,
    config: EvalConfig | None = None,
    evaluation_id: str | None = None,
) -> EvaluationResult:
    config = config or EvalConfig()
    meta = model.metadata()
    names = dataset.class_names
    _check_compatible(meta, names)
    k = len(names)

    eval_id = evaluation_id or _new_id()
    if Path(eval_id).name != eval_id or eval_id in (".", ".."):
        raise EvaluationError(f"invalid evaluation id: {eval_id!r}")
    relative = Path("evaluations") / eval_id
    out_dir = store.resolve(relative)
    if out_dir.exists():
        raise EvaluationError(f"evaluation '{eval_id}' already exists in {store.root}")

    set_seed(config.seed)
    started = datetime.now(timezone.utc)
    schema = _schema(k)
    tmp_path = out_dir / "predictions.parquet.tmp"
    final_path = out_dir / "predictions.parquet"

    loader = DataLoader(
        _ImageStream(dataset),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=_collate,
        pin_memory=model.device.type == "cuda",
    )
    total = len(loader)
    every = max(1, total // 10)
    log.info(
        "evaluating %d images, batch size %d, device %s", len(dataset), config.batch_size, model.device
    )

    out_dir.mkdir(parents=True)
    writer = None
    truths, predictions, probabilities = [], [], []
    try:
        writer = pq.ParquetWriter(tmp_path, schema)
        for step, (ids, paths, labels, batch) in enumerate(loader, start=1):
            probs = model.predict_probabilities(batch)
            labels = labels.numpy()
            pred = probs.argmax(axis=1)
            writer.write_table(_table(ids, paths, labels, pred, probs, schema))
            truths.append(labels)
            predictions.append(pred)
            probabilities.append(probs)
            if step % every == 0 or step == total:
                log.info("evaluated %d/%d batches", step, total)
        writer.close()
        writer = None

        y_true = np.concatenate(truths)
        y_pred = np.concatenate(predictions)
        metrics, confusion = compute_metrics(y_true, y_pred, np.concatenate(probabilities), names)

        metadata: dict[str, Any] = {
            "evaluation_id": eval_id,
            "model_id": meta.model_id,
            "dataset_id": dataset.dataset_id,
            "timestamp": started.isoformat(),
            "seed": config.seed,
            "device": str(model.device),
            "batch_size": config.batch_size,
            "num_workers": config.num_workers,
            "preprocessing": (
                dataset.preprocess.model_dump(mode="json") if dataset.preprocess else None
            ),
            "num_samples": int(len(y_true)),
            "num_classes": k,
            "class_names": names,
            "model": model.spec.model_dump(mode="json"),
            "versions": {
                "modellab": modellab.__version__,
                "torch": torch.__version__,
                "python": platform.python_version(),
            },
        }

        store.write_json(relative / "metrics.json", metrics)
        store.write_json(relative / "confusion_matrix.json", confusion)
        store.write_json(relative / "metadata.json", metadata)
        os.replace(tmp_path, final_path)
    except BaseException:
        if writer is not None:
            with contextlib.suppress(Exception):
                writer.close()
        shutil.rmtree(out_dir, ignore_errors=True)
        raise

    return EvaluationResult(
        model_id=meta.model_id,
        dataset_id=dataset.dataset_id,
        metrics=metrics,
        prediction_artifact=(relative / "predictions.parquet").as_posix(),
        created_at=started,
        configuration=metadata,
    )
