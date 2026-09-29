import pytest
from pydantic import ValidationError

from modellab.core.experiments import Experiment, ExperimentStatus


def test_defaults_and_unique_ids():
    a = Experiment(name="a", seed=1)
    b = Experiment(name="a", seed=1)
    assert a.experiment_id != b.experiment_id
    assert a.status is ExperimentStatus.CREATED
    assert a.created_at.tzinfo is not None


def test_roundtrip():
    e = Experiment(name="n", description="d", seed=3, parameters={"lr": 0.1, "tags": ["x"]})
    assert Experiment.model_validate_json(e.model_dump_json()) == e
    assert e.model_dump(mode="json")["status"] == "created"


def test_validation():
    with pytest.raises(ValidationError):
        Experiment(name="n", seed=-1)
    with pytest.raises(ValidationError):
        Experiment(name="n", seed=1, status="paused")
