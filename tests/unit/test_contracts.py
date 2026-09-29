import pytest
from pydantic import ValidationError

from modellab.core.datasets import DatasetAdapter, DatasetSample
from modellab.core.models import ModelAdapter, ModelMetadata
from modellab.core.results import Prediction


class DummyDataset(DatasetAdapter):
    def __init__(self, n=4):
        self._n = n

    @property
    def dataset_id(self):
        return "dummy"

    def __len__(self):
        return self._n

    def get_sample(self, index):
        if not 0 <= index < self._n:
            raise IndexError(index)
        return DatasetSample(sample_id=f"s{index}", input=index, label=index % 2)

    def get_label(self, index):
        return index % 2


class DummyModel(ModelAdapter):
    def metadata(self):
        return ModelMetadata(model_id="dummy", framework="none", num_classes=2)

    def predict(self, sample):
        cls = int(sample.input) % 2
        probs = [0.0, 0.0]
        probs[cls] = 1.0
        return Prediction(
            sample_id=sample.sample_id,
            predicted_class=cls,
            confidence=1.0,
            class_probabilities=probs,
        )


def test_dataset_contract():
    ds = DummyDataset()
    assert len(ds) == 4
    assert ds.get_label(3) == 1
    assert ds.get_metadata(0) == {}
    assert [s.sample_id for s in ds.iterate()] == ["s0", "s1", "s2", "s3"]
    with pytest.raises(IndexError):
        ds.get_sample(4)


def test_incomplete_subclasses_cannot_be_built():
    class NoPredict(ModelAdapter):
        def metadata(self):
            return None

    class NoLen(DatasetAdapter):
        pass

    with pytest.raises(TypeError):
        NoPredict()
    with pytest.raises(TypeError):
        NoLen()


def test_model_contract():
    model = DummyModel()
    assert model.metadata().num_classes == 2
    samples = list(DummyDataset().iterate())
    singles = [model.predict(s) for s in samples]
    assert model.predict_batch(samples) == singles
    assert [p.predicted_class for p in singles] == [0, 1, 0, 1]


def test_sample_metadata_is_optional_and_not_shared():
    a = DatasetSample(sample_id="a", input=1)
    b = DatasetSample(sample_id="b", input=2)
    assert a.label is None
    assert a.metadata == {} and a.metadata is not b.metadata


def test_metadata_validation():
    with pytest.raises(ValidationError):
        ModelMetadata(model_id="m", framework="f", num_classes=0)
    with pytest.raises(ValidationError):
        ModelMetadata(model_id="m", framework="f", num_classes=2, class_names=["a"])
    meta = ModelMetadata(
        model_id="m", framework="f", num_classes=2, class_names=["a", "b"], input_size=(32, 32)
    )
    assert ModelMetadata.model_validate_json(meta.model_dump_json()) == meta
