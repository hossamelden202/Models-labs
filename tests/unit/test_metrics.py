import json

import numpy as np
import pytest

pytest.importorskip("sklearn")

from modellab.core.errors import EvaluationError  # noqa: E402
from modellab.evaluation.metrics import compute_metrics  # noqa: E402


def one_hot(pred, k):
    probs = np.zeros((len(pred), k))
    probs[np.arange(len(pred)), pred] = 1.0
    return probs


def test_binary_metrics():
    y_true = [0, 0, 1, 1, 1]
    y_pred = [0, 1, 1, 1, 0]
    metrics, confusion = compute_metrics(y_true, y_pred, one_hot(y_pred, 2), ["neg", "pos"])
    assert confusion["matrix"] == [[1, 1], [1, 2]]
    assert confusion["labels"] == ["neg", "pos"]
    assert metrics["num_samples"] == 5
    assert metrics["accuracy"] == pytest.approx(0.6)
    assert metrics["macro_precision"] == pytest.approx((0.5 + 2 / 3) / 2)
    assert metrics["macro_recall"] == pytest.approx((0.5 + 2 / 3) / 2)
    assert metrics["macro_f1"] == pytest.approx((0.5 + 2 / 3) / 2)
    assert metrics["weighted_f1"] == pytest.approx(0.6)
    neg, pos = metrics["per_class"]
    assert (neg["class_name"], neg["support"]) == ("neg", 2)
    assert (neg["precision"], neg["recall"], neg["f1"]) == pytest.approx((0.5, 0.5, 0.5))
    assert (pos["class_name"], pos["support"]) == ("pos", 3)
    assert (pos["precision"], pos["recall"], pos["f1"]) == pytest.approx((2 / 3,) * 3)


def test_multiclass_metrics():
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 1, 1, 1, 2, 0]
    metrics, confusion = compute_metrics(y_true, y_pred, one_hot(y_pred, 3), ["a", "b", "c"])
    assert confusion["matrix"] == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    assert metrics["accuracy"] == pytest.approx(4 / 6)
    assert metrics["macro_precision"] == pytest.approx((0.5 + 2 / 3 + 1.0) / 3)
    assert metrics["macro_recall"] == pytest.approx((0.5 + 1.0 + 0.5) / 3)
    assert metrics["macro_f1"] == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)
    assert metrics["weighted_f1"] == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)
    a, b, c = metrics["per_class"]
    assert (a["precision"], a["recall"], a["f1"]) == pytest.approx((0.5, 0.5, 0.5))
    assert (b["precision"], b["recall"], b["f1"]) == pytest.approx((2 / 3, 1.0, 0.8))
    assert (c["precision"], c["recall"], c["f1"]) == pytest.approx((1.0, 0.5, 2 / 3))
    assert [x["support"] for x in metrics["per_class"]] == [2, 2, 2]


def test_perfect_scores():
    y = [0, 1, 2, 0, 1, 2]
    metrics, _ = compute_metrics(y, y, one_hot(y, 3), ["a", "b", "c"])
    assert metrics["accuracy"] == 1.0
    assert metrics["auroc"]["status"] == "ok"
    assert metrics["auroc"]["value"] == pytest.approx(1.0)
    assert metrics["auprc"]["value"] == pytest.approx(1.0)
    assert metrics["auroc"]["average"] == "macro one-vs-rest"


def test_binary_auroc_and_auprc_known_values():
    probs = np.array([[0.9, 0.1], [0.6, 0.4], [0.65, 0.35], [0.2, 0.8]])
    y_true = [0, 0, 1, 1]
    y_pred = probs.argmax(axis=1)
    metrics, _ = compute_metrics(y_true, y_pred, probs, ["neg", "pos"])
    assert metrics["auroc"]["value"] == pytest.approx(0.75)
    assert metrics["auprc"]["value"] == pytest.approx(5 / 6)
    assert "pos" in metrics["auroc"]["average"]


def test_missing_class_is_reported_not_faked():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    metrics, confusion = compute_metrics(y_true, y_pred, one_hot(y_pred, 3), ["a", "b", "c"])
    assert metrics["per_class"][2]["support"] == 0
    assert metrics["per_class"][2]["recall"] == 0.0
    assert confusion["matrix"] == [[1, 1, 0], [0, 2, 0], [0, 0, 0]]
    for key in ("auroc", "auprc"):
        assert metrics[key]["status"] == "not_available"
        assert "[2]" in metrics[key]["reason"]


def test_binary_with_a_single_class_present():
    y_true = [0, 0, 0]
    y_pred = [0, 0, 1]
    metrics, _ = compute_metrics(y_true, y_pred, one_hot(y_pred, 2), ["a", "b"])
    assert metrics["auroc"]["status"] == "not_available"
    assert metrics["auprc"]["status"] == "not_available"


def test_metrics_are_json_safe():
    y_true = [0, 0, 1]
    y_pred = [0, 1, 1]
    metrics, confusion = compute_metrics(y_true, y_pred, one_hot(y_pred, 3), ["a", "b", "c"])
    json.dumps(metrics, allow_nan=False)
    json.dumps(confusion, allow_nan=False)


@pytest.mark.parametrize(
    "y_true, y_pred, probs, names",
    [
        ([], [], np.zeros((0, 2)), ["a", "b"]),
        ([0, 1], [0, 1], np.eye(2), ["a"]),
        ([0, 2], [0, 1], np.eye(2), ["a", "b"]),
        ([0, 1], [0, 2], np.eye(2), ["a", "b"]),
        ([-1, 1], [0, 1], np.eye(2), ["a", "b"]),
        ([0, 1], [0, 1], np.eye(3), ["a", "b"]),
        ([0, 1, 1], [0, 1], np.eye(2), ["a", "b"]),
    ],
)
def test_invalid_inputs(y_true, y_pred, probs, names):
    with pytest.raises(EvaluationError):
        compute_metrics(y_true, y_pred, probs, names)
