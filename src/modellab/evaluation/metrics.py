import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from modellab.core.errors import EvaluationError


def _ok(value: float, average: str) -> dict:
    return {"status": "ok", "value": float(value), "average": average}


def _not_available(reason: str) -> dict:
    return {"status": "not_available", "reason": reason}


def _missing_classes(y_true: np.ndarray, k: int) -> list[int]:
    return sorted(set(range(k)) - set(np.unique(y_true).tolist()))


def _auroc(y_true, probs, names) -> dict:
    k = len(names)
    missing = _missing_classes(y_true, k)
    if missing:
        return _not_available(f"no samples for classes {missing}")
    try:
        if k == 2:
            value = roc_auc_score(y_true, probs[:, 1])
            return _ok(value, f"binary, positive class '{names[1]}'")
        scores = probs / probs.sum(axis=1, keepdims=True)
        value = roc_auc_score(
            y_true, scores, multi_class="ovr", average="macro", labels=list(range(k))
        )
        return _ok(value, "macro one-vs-rest")
    except ValueError as exc:
        return _not_available(str(exc))


def _auprc(y_true, probs, names) -> dict:
    k = len(names)
    missing = _missing_classes(y_true, k)
    if missing:
        return _not_available(f"no samples for classes {missing}")
    try:
        if k == 2:
            value = average_precision_score(y_true, probs[:, 1])
            return _ok(value, f"binary, positive class '{names[1]}'")
        per_class = [average_precision_score(y_true == c, probs[:, c]) for c in range(k)]
        return _ok(np.mean(per_class), "macro one-vs-rest")
    except ValueError as exc:
        return _not_available(str(exc))


def compute_metrics(y_true, y_pred, probs, class_names) -> tuple[dict, dict]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    names = list(class_names)
    k = len(names)
    n = len(y_true)

    if k < 2:
        raise EvaluationError("need at least two classes to compute metrics")
    if n == 0:
        raise EvaluationError("cannot compute metrics without samples")
    if y_true.shape != (n,) or y_pred.shape != (n,) or probs.shape != (n, k):
        raise EvaluationError(
            f"inconsistent shapes: y_true {y_true.shape}, y_pred {y_pred.shape}, "
            f"probs {probs.shape}, classes {k}"
        )
    for name, values in (("y_true", y_true), ("y_pred", y_pred)):
        if values.min() < 0 or values.max() >= k:
            raise EvaluationError(f"{name} contains labels outside [0, {k})")

    labels = np.arange(k)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )

    metrics = {
        "num_samples": int(n),
        "accuracy": float((y_true == y_pred).mean()),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class": [
            {
                "class_index": i,
                "class_name": names[i],
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(k)
        ],
        "auroc": _auroc(y_true, probs, names),
        "auprc": _auprc(y_true, probs, names),
    }
    confusion = {
        "labels": names,
        "rows": "true",
        "columns": "predicted",
        "matrix": matrix.tolist(),
    }
    return metrics, confusion
