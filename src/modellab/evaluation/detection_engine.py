from __future__ import annotations

import contextlib
import json
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
from torchvision.ops import batched_nms
from pydantic import BaseModel, ConfigDict, Field
from torch.utils.data import DataLoader, Dataset

import modellab
from modellab.core.errors import EvaluationError
from modellab.core.results import ArtifactStore, EvaluationResult
from modellab.evaluation.detection_dataset import YOLODetectionDataset
from modellab.evaluation.torch_model import (
    TorchImageClassifier,
    _extract_detection_output,
)
from modellab.utils import set_seed


class DetectionEvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_size: int = Field(8, ge=1)
    num_workers: int = Field(0, ge=0)
    seed: int = Field(42, ge=0, lt=2**32)
    confidence_threshold: float = Field(0.25, ge=0.0, le=1.0)
    iou_threshold: float = Field(0.50, gt=0.0, le=1.0)


class _DetectionStream(Dataset):
    def __init__(self, dataset: YOLODetectionDataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset.get_sample(index)

        if not isinstance(sample.input, torch.Tensor):
            raise EvaluationError(
                f"sample {sample.sample_id}: input must be a torch.Tensor"
            )

        return (
            sample.sample_id,
            sample.metadata.get("path", ""),
            sample.metadata.get("boxes", []),
            sample.input,
        )


def _collate(items):
    ids, paths, targets, tensors = zip(*items, strict=True)

    try:
        batch = torch.stack(tensors)
    except RuntimeError as exc:
        raise EvaluationError(
            "images in one detection batch have different shapes; "
            "set resize in the preprocess config"
        ) from exc

    return list(ids), list(paths), list(targets), batch


def _xywh_to_xyxy(box):
    x, y, w, h = box

    return np.array(
        [
            x - w / 2,
            y - h / 2,
            x + w / 2,
            y + h / 2,
        ],
        dtype=np.float32,
    )


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    intersection = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return float(intersection / union)


def _decode_yolo(
    output: torch.Tensor,
    num_classes: int,
    confidence_threshold: float,
    iou_threshold: float,
    input_height: int,
    input_width: int,
):
    """
    Post-process an Ultralytics YOLO inference tensor.

    DetectionModel.forward() has already performed the DFL/anchor
    decoding. This function performs confidence filtering, class-aware
    NMS, and conversion to normalized xyxy coordinates.

    Input:
        (batch, 4 + num_classes, candidates)

    Output:
        One list of detections per batch item, with normalized boxes.
    """

    if output.ndim != 3:
        raise EvaluationError(
            "expected detection output of shape "
            f"(batch, 4 + classes, candidates), got {tuple(output.shape)}"
        )

    expected_channels = 4 + num_classes

    if output.shape[1] != expected_channels:
        raise EvaluationError(
            "detection model output has "
            f"{output.shape[1]} channels, expected "
            f"4 + {num_classes} = {expected_channels}"
        )

    if input_height <= 0 or input_width <= 0:
        raise EvaluationError(
            "detection input dimensions must be positive, got "
            f"{input_height}x{input_width}"
        )

    output = output.detach().float()

    results = []

    for row in output:
        boxes_xywh = row[:4].T
        class_scores = row[4:].T

        class_ids = class_scores.argmax(dim=1)
        confidences = class_scores.max(dim=1).values

        keep = confidences >= float(confidence_threshold)

        if not bool(keep.any()):
            results.append([])
            continue

        boxes_xywh = boxes_xywh[keep]
        class_ids = class_ids[keep]
        confidences = confidences[keep]

        # xywh -> xyxy.
        x = boxes_xywh[:, 0]
        y = boxes_xywh[:, 1]
        w = boxes_xywh[:, 2]
        h = boxes_xywh[:, 3]

        boxes_xyxy = torch.stack(
            (
                x - w / 2.0,
                y - h / 2.0,
                x + w / 2.0,
                y + h / 2.0,
            ),
            dim=1,
        )

        # Remove NaN/Inf values.
        finite = torch.isfinite(boxes_xyxy).all(dim=1)
        finite &= torch.isfinite(confidences)

        if not bool(finite.any()):
            results.append([])
            continue

        boxes_xyxy = boxes_xyxy[finite]
        class_ids = class_ids[finite]
        confidences = confidences[finite]

        # Remove zero/negative-area boxes.
        widths = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
        heights = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]

        valid_size = (widths > 0) & (heights > 0)

        if not bool(valid_size.any()):
            results.append([])
            continue

        boxes_xyxy = boxes_xyxy[valid_size]
        class_ids = class_ids[valid_size]
        confidences = confidences[valid_size]

        # Class-aware NMS.
        keep_indices = batched_nms(
            boxes_xyxy,
            confidences,
            class_ids,
            float(iou_threshold),
        )

        # Keep at most 300 detections per image.
        keep_indices = keep_indices[:300]

        boxes_xyxy = boxes_xyxy[keep_indices]
        class_ids = class_ids[keep_indices]
        confidences = confidences[keep_indices]

        # Predictions are in input-image pixels.
        # ModelLab YOLO targets are normalized [0, 1].
        scale = boxes_xyxy.new_tensor(
            [
                float(input_width),
                float(input_height),
                float(input_width),
                float(input_height),
            ]
        )

        boxes_normalized = boxes_xyxy / scale
        boxes_normalized = boxes_normalized.clamp(0.0, 1.0)

        detections = []

        for box, cls, confidence in zip(
            boxes_normalized,
            class_ids,
            confidences,
            strict=True,
        ):
            detections.append(
                {
                    "class_id": int(cls.item()),
                    "confidence": float(confidence.item()),
                    "box": [
                        float(value)
                        for value in box.tolist()
                    ],
                }
            )

        results.append(detections)

    return results



def _greedy_match(predictions, targets, iou_threshold):
    pairs = []

    used_predictions = set()
    used_targets = set()

    candidates = []

    for pi, pred in enumerate(predictions):
        for ti, target in enumerate(targets):
            if pred["class_id"] != target["class_id"]:
                continue

            iou = _iou(pred["box"], target["box"])

            if iou >= iou_threshold:
                candidates.append((iou, pi, ti))

    candidates.sort(reverse=True)

    for iou, pi, ti in candidates:
        if pi in used_predictions or ti in used_targets:
            continue

        used_predictions.add(pi)
        used_targets.add(ti)
        pairs.append((pi, ti, iou))

    return pairs


def _new_id():
    return (
        f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_"
        f"{uuid.uuid4().hex[:6]}"
    )


def run_detection_evaluation(
    model: TorchImageClassifier,
    dataset: YOLODetectionDataset,
    store: ArtifactStore,
    config: DetectionEvalConfig | None = None,
    evaluation_id: str | None = None,
) -> EvaluationResult:

    config = config or DetectionEvalConfig()

    if model.spec.task != "detection":
        raise EvaluationError(
            f"detection evaluator requires task='detection', "
            f"got {model.spec.task!r}"
        )

    if not model.spec.class_names:
        raise EvaluationError(
            "detection evaluation requires model class_names"
        )

    names = list(model.spec.class_names)
    k = len(names)

    eval_id = evaluation_id or _new_id()

    if Path(eval_id).name != eval_id or eval_id in (".", ".."):
        raise EvaluationError(f"invalid evaluation id: {eval_id!r}")

    relative = Path("evaluations") / eval_id
    out_dir = store.resolve(relative)

    if out_dir.exists():
        raise EvaluationError(
            f"evaluation '{eval_id}' already exists in {store.root}"
        )

    set_seed(config.seed)

    started = datetime.now(timezone.utc)

    loader = DataLoader(
        _DetectionStream(dataset),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=_collate,
        pin_memory=model.device.type == "cuda",
    )

    total_tp = 0
    total_fp = 0
    total_fn = 0
    matched_ious = []

    per_class = {
        i: {"tp": 0, "fp": 0, "fn": 0, "ious": []}
        for i in range(k)
    }

    prediction_rows = []

    out_dir.mkdir(parents=True)

    try:
        for ids, paths, targets, batch in loader:
            batch = model._prepare_model_input(batch)

            try:
                with torch.no_grad():
                    if model._amp:
                        with torch.autocast(
                            device_type="cuda",
                            dtype=torch.float16,
                        ):
                            raw = model.model(batch)
                    else:
                        raw = model.model(batch)
            except Exception as exc:
                raise EvaluationError(
                    f"detection model forward failed on input "
                    f"{tuple(batch.shape)}: {exc}"
                ) from exc

            raw = _extract_detection_output(
                raw,
                model.spec.num_classes,
            )

            if model.spec.input_size is None:
                input_height = int(batch.shape[-2])
                input_width = int(batch.shape[-1])
            else:
                input_height = int(model.spec.input_size[0])
                input_width = int(model.spec.input_size[1])

            predictions = _decode_yolo(
                raw,
                model.spec.num_classes,
                config.confidence_threshold,
                config.iou_threshold,
                input_height,
                input_width,
            )

            for sample_id, path, target, preds in zip(
                ids,
                paths,
                targets,
                predictions,
                strict=True,
            ):
                target_boxes = [
                    {
                        "class_id": int(t["class_id"]),
                        "box": _xywh_to_xyxy(
                            (
                                t["x_center"],
                                t["y_center"],
                                t["width"],
                                t["height"],
                            )
                        ).tolist(),
                    }
                    for t in target
                ]

                pairs = _greedy_match(
                    preds,
                    target_boxes,
                    config.iou_threshold,
                )

                matched_pred = {pi for pi, _, _ in pairs}
                matched_target = {ti for _, ti, _ in pairs}

                total_tp += len(pairs)
                total_fp += len(preds) - len(pairs)
                total_fn += len(target_boxes) - len(pairs)

                for pi, ti, iou in pairs:
                    cls = preds[pi]["class_id"]

                    matched_ious.append(iou)

                    if cls in per_class:
                        per_class[cls]["tp"] += 1
                        per_class[cls]["ious"].append(iou)

                for pi, pred in enumerate(preds):
                    if pi not in matched_pred:
                        cls = pred["class_id"]
                        if cls in per_class:
                            per_class[cls]["fp"] += 1

                for ti, gt in enumerate(target_boxes):
                    if ti not in matched_target:
                        cls = gt["class_id"]
                        if cls in per_class:
                            per_class[cls]["fn"] += 1

                prediction_rows.append(
                    {
                        "sample_id": sample_id,
                        "path": path,
                        "true_count": len(target_boxes),
                        "predicted_count": len(preds),
                        "true_boxes": target_boxes,
                        "predictions": preds,
                    }
                )

        precision = (
            total_tp / (total_tp + total_fp)
            if total_tp + total_fp
            else 0.0
        )

        recall = (
            total_tp / (total_tp + total_fn)
            if total_tp + total_fn
            else 0.0
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

        per_class_metrics = {}

        for cls, stats in per_class.items():
            tp = stats["tp"]
            fp = stats["fp"]
            fn = stats["fn"]

            p = tp / (tp + fp) if tp + fp else 0.0
            r = tp / (tp + fn) if tp + fn else 0.0
            f = 2 * p * r / (p + r) if p + r else 0.0

            per_class_metrics[names[cls]] = {
                "class_id": cls,
                "precision": p,
                "recall": r,
                "f1": f,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "mean_iou": (
                    float(np.mean(stats["ious"]))
                    if stats["ious"]
                    else 0.0
                ),
            }

        metrics = {
            "task": "detection",
            "num_samples": len(dataset),
            "num_classes": k,
            "class_names": names,
            "true_positives": total_tp,
            "false_positives": total_fp,
            "false_negatives": total_fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_iou": (
                float(np.mean(matched_ious))
                if matched_ious
                else 0.0
            ),
            "confidence_threshold": config.confidence_threshold,
            "iou_threshold": config.iou_threshold,
            "per_class": per_class_metrics,
        }

        metadata = {
            "evaluation_id": eval_id,
            "model_id": model.spec.model_id,
            "dataset_id": dataset.dataset_id,
            "task": "detection",
            "timestamp": started.isoformat(),
            "seed": config.seed,
            "device": str(model.device),
            "batch_size": config.batch_size,
            "num_workers": config.num_workers,
            "num_samples": len(dataset),
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
        store.write_json(relative / "metadata.json", metadata)
        store.write_json(
            relative / "predictions.json",
            prediction_rows,
        )

    except BaseException:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise

    return EvaluationResult(
        model_id=model.spec.model_id,
        dataset_id=dataset.dataset_id,
        metrics=metrics,
        prediction_artifact=(
            relative / "predictions.json"
        ).as_posix(),
        created_at=started,
        configuration=metadata,
    )
