import pytest
from pydantic import ValidationError

from modellab.core.results import EvaluationResult, Prediction


def test_prediction_roundtrip():
    p = Prediction(
        sample_id="a",
        predicted_class=1,
        confidence=0.75,
        class_probabilities=[0.25, 0.75],
    )
    assert Prediction.model_validate_json(p.model_dump_json()) == p
    assert p.model_dump(mode="json")["predicted_class"] == 1


def test_prediction_probabilities_optional():
    p = Prediction(sample_id="a", predicted_class=0, confidence=0.5)
    assert p.class_probabilities is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"confidence": 1.5},
        {"confidence": -0.1},
        {"predicted_class": -1},
        {"predicted_class": 2, "class_probabilities": [0.5, 0.5]},
    ],
)
def test_prediction_validation(kwargs):
    base = {"sample_id": "a", "predicted_class": 0, "confidence": 0.5}
    with pytest.raises(ValidationError):
        Prediction(**{**base, **kwargs})


def test_evaluation_result_roundtrip():
    r = EvaluationResult(
        model_id="m",
        dataset_id="d",
        metrics={"accuracy": 0.9},
        prediction_artifact="evaluations/e1/predictions.parquet",
        configuration={"batch_size": 8},
    )
    assert r.created_at.tzinfo is not None
    assert EvaluationResult.model_validate_json(r.model_dump_json()) == r
