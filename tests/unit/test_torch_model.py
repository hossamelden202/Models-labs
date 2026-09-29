import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import numpy as np  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from modellab.core.datasets import DatasetSample  # noqa: E402
from modellab.core.errors import ConfigurationError, EvaluationError, ModelLoadError  # noqa: E402
from modellab.evaluation.torch_model import (  # noqa: E402
    TorchImageClassifier,
    TorchModelSpec,
    resolve_device,
)

DUMMY = Path(__file__).resolve().parents[1] / "dummy_models.py"
COLORS = {"red": (1.0, 0.0, 0.0), "green": (0.0, 1.0, 0.0), "blue": (0.0, 0.0, 1.0)}
HAS_CUDA = torch.cuda.is_available()

_module_spec = importlib.util.spec_from_file_location("dummy_models_for_tests", DUMMY)
dummy = importlib.util.module_from_spec(_module_spec)
_module_spec.loader.exec_module(dummy)


def ref(name):
    return f"{DUMMY}:{name}"


def make_spec(factory="identity_classifier", **kwargs):
    fields = {"model_id": "m", "source": "factory", "num_classes": 3, "factory": ref(factory)}
    fields.update(kwargs)
    return TorchModelSpec(**fields)


def cpu_adapter(*args, **kwargs):
    return TorchImageClassifier(make_spec(*args, **kwargs), device="cpu")


def color_sample(name, size=4):
    tensor = torch.tensor(COLORS[name]).view(3, 1, 1).expand(3, size, size).clone()
    return DatasetSample(sample_id=name, input=tensor)


def rgb_samples():
    return [color_sample(c) for c in ("red", "green", "blue")]


def rand_input(n=4, size=5):
    gen = torch.Generator().manual_seed(123)
    return torch.rand(n, 3, size, size, generator=gen)


def expected_probs(model, x):
    with torch.no_grad():
        return torch.softmax(model(x), dim=1).numpy()


def save(tmp_path, payload, name="ckpt.pt"):
    path = tmp_path / name
    torch.save(payload, path)
    return path


def state_spec(path, **kwargs):
    return make_spec(
        "random_classifier", source="state_dict", path=path, factory_kwargs={"seed": 0}, **kwargs
    )


def test_metadata():
    adapter = cpu_adapter(class_names=["r", "g", "b"], input_size=(4, 4))
    meta = adapter.metadata()
    assert meta.model_id == "m"
    assert meta.framework == "pytorch"
    assert meta.num_classes == 3
    assert meta.class_names == ["r", "g", "b"]
    assert meta.device == "cpu"
    assert meta.input_size == (4, 4)
    assert meta.details == {"source": "factory", "mixed_precision": False}


def test_model_is_frozen_and_in_eval_mode():
    adapter = cpu_adapter()
    assert not adapter.model.training
    assert all(not p.requires_grad for p in adapter.model.parameters())


def test_predictions_are_correct():
    preds = cpu_adapter().predict_batch(rgb_samples())
    assert [p.sample_id for p in preds] == ["red", "green", "blue"]
    assert [p.predicted_class for p in preds] == [0, 1, 2]
    expected = np.exp(10) / (np.exp(10) + 2)
    for p in preds:
        assert len(p.class_probabilities) == 3
        assert sum(p.class_probabilities) == pytest.approx(1.0, abs=1e-6)
        assert p.confidence == max(p.class_probabilities)
        assert p.confidence == pytest.approx(expected, abs=1e-6)


def test_probability_array_shape_and_dtype():
    probs = cpu_adapter().predict_probabilities(rand_input(6))
    assert probs.shape == (6, 3)
    assert probs.dtype == np.float32


def test_batch_matches_single_predictions():
    adapter = cpu_adapter("random_classifier")
    x = rand_input(5)
    samples = [DatasetSample(sample_id=f"s{i}", input=x[i]) for i in range(5)]
    batch = adapter.predict_batch(samples)
    singles = [adapter.predict(s) for s in samples]
    for b, s in zip(batch, singles, strict=True):
        assert b.predicted_class == s.predicted_class
        assert b.class_probabilities == pytest.approx(s.class_probabilities, abs=1e-6)


def test_deterministic_across_adapters():
    x = rand_input()
    a = cpu_adapter("random_classifier").predict_probabilities(x)
    b = cpu_adapter("random_classifier").predict_probabilities(x)
    assert np.array_equal(a, b)


def test_empty_batch():
    assert cpu_adapter().predict_batch([]) == []


@pytest.mark.parametrize(
    "mode, extra",
    [
        ("dict", {}),
        ("attr", {}),
        ("tuple", {"output_index": 1}),
        ("custom_key", {"output_key": "scores"}),
        ("attr", {"output_key": "logits"}),
    ],
)
def test_output_formats(mode, extra):
    adapter = cpu_adapter("wrapped", factory_kwargs={"mode": mode}, **extra)
    assert [p.predicted_class for p in adapter.predict_batch(rgb_samples())] == [0, 1, 2]


@pytest.mark.parametrize(
    "mode, extra, message",
    [
        ("tuple", {}, "shape"),
        ("tuple", {"output_index": 5}, "out of range"),
        ("dict", {"output_key": "nope"}, "no 'nope' key"),
        ("attr", {"output_key": "nope"}, "no attribute"),
        ("bad_shape", {}, "shape"),
        ("nan", {}, "non-finite"),
    ],
)
def test_bad_model_outputs(mode, extra, message):
    adapter = cpu_adapter("wrapped", factory_kwargs={"mode": mode}, **extra)
    with pytest.raises(EvaluationError, match=message):
        adapter.predict_batch(rgb_samples())


def test_class_count_mismatch():
    with pytest.raises(EvaluationError, match="classes"):
        cpu_adapter(num_classes=4).predict_batch(rgb_samples())


def test_wrong_channel_count():
    sample = DatasetSample(sample_id="g", input=torch.zeros(1, 4, 4))
    with pytest.raises(EvaluationError, match="forward failed"):
        cpu_adapter().predict_batch([sample])


def test_non_tensor_input():
    with pytest.raises(EvaluationError, match="tensor"):
        cpu_adapter().predict_batch([DatasetSample(sample_id="x", input="not a tensor")])


def test_wrong_batch_rank():
    with pytest.raises(EvaluationError, match="batch must have shape"):
        cpu_adapter().predict_probabilities(torch.zeros(3, 4, 4))


def test_mixed_shapes_in_one_batch():
    samples = [color_sample("red", 4), color_sample("green", 5)]
    with pytest.raises(EvaluationError, match="identical"):
        cpu_adapter().predict_batch(samples)


def test_single_logit_binary():
    adapter = cpu_adapter("binary_single_logit", num_classes=2, single_logit_binary=True)
    preds = adapter.predict_batch([color_sample("red"), color_sample("green")])
    assert [p.predicted_class for p in preds] == [1, 0]
    assert preds[0].class_probabilities[1] == pytest.approx(1 / (1 + np.exp(-5)), abs=1e-6)
    assert sum(preds[1].class_probabilities) == pytest.approx(1.0, abs=1e-6)


def test_single_logit_output_needs_the_flag():
    adapter = cpu_adapter("binary_single_logit", num_classes=2)
    with pytest.raises(EvaluationError, match="classes"):
        adapter.predict_batch([color_sample("red")])


def test_torchscript_source(tmp_path):
    traced = torch.jit.trace(dummy.identity_classifier(), torch.zeros(1, 3, 4, 4))
    path = tmp_path / "traced.pt"
    traced.save(str(path))
    spec = TorchModelSpec(model_id="ts", source="torchscript", num_classes=3, path=path)
    adapter = TorchImageClassifier(spec, device="cpu")
    assert [p.predicted_class for p in adapter.predict_batch(rgb_samples())] == [0, 1, 2]


def test_full_module_source(tmp_path):
    path = save(tmp_path, dummy.identity_classifier(), "full.pt")
    spec = TorchModelSpec(model_id="pk", source="module", num_classes=3, path=path)
    adapter = TorchImageClassifier(spec, device="cpu")
    assert [p.predicted_class for p in adapter.predict_batch(rgb_samples())] == [0, 1, 2]


def test_module_source_rejects_non_modules(tmp_path):
    path = save(tmp_path, {"a": 1}, "dict.pt")
    spec = TorchModelSpec(model_id="pk", source="module", num_classes=3, path=path)
    with pytest.raises(ModelLoadError, match="nn.Module"):
        TorchImageClassifier(spec, device="cpu")


def test_state_dict_weights_are_loaded(tmp_path):
    source = dummy.random_classifier(seed=1)
    path = save(tmp_path, source.state_dict())
    x = rand_input()
    loaded = TorchImageClassifier(state_spec(path), device="cpu")
    assert np.allclose(loaded.predict_probabilities(x), expected_probs(source, x), atol=1e-6)
    untouched = cpu_adapter("random_classifier", factory_kwargs={"seed": 0})
    assert not np.allclose(untouched.predict_probabilities(x), expected_probs(source, x), atol=1e-3)


def test_state_dict_nested_key_and_prefix(tmp_path):
    source = dummy.random_classifier(seed=1)
    prefixed = {f"module.{k}": v for k, v in source.state_dict().items()}
    path = save(tmp_path, {"model_state_dict": prefixed, "epoch": 3})
    spec = state_spec(path, state_dict_key="model_state_dict", strip_prefix="module.")
    x = rand_input()
    loaded = TorchImageClassifier(spec, device="cpu")
    assert np.allclose(loaded.predict_probabilities(x), expected_probs(source, x), atol=1e-6)


def test_state_dict_failures(tmp_path):
    good = save(tmp_path, dummy.random_classifier(seed=1).state_dict(), "good.pt")
    with pytest.raises(ModelLoadError, match="nope"):
        TorchImageClassifier(state_spec(good, state_dict_key="nope"), device="cpu")

    not_a_dict = save(tmp_path, torch.zeros(2), "tensor.pt")
    with pytest.raises(ModelLoadError, match="state dict"):
        TorchImageClassifier(state_spec(not_a_dict), device="cpu")

    wrong_shape = save(tmp_path, dummy.random_classifier(num_classes=4).state_dict(), "wrong.pt")
    with pytest.raises(ModelLoadError, match="could not load"):
        TorchImageClassifier(state_spec(wrong_shape), device="cpu")

    with pytest.raises(ModelLoadError, match="not found"):
        TorchImageClassifier(state_spec(tmp_path / "missing.pt"), device="cpu")


def test_library_architecture_through_import_path():
    spec = TorchModelSpec(
        model_id="r18",
        source="factory",
        num_classes=3,
        factory="torchvision.models:resnet18",
        factory_kwargs={"weights": None, "num_classes": 3},
    )
    probs = TorchImageClassifier(spec, device="cpu").predict_probabilities(rand_input(2, 32))
    assert probs.shape == (2, 3)
    assert probs.sum(axis=1) == pytest.approx([1.0, 1.0], abs=1e-5)


@pytest.mark.parametrize(
    "factory, message",
    [
        ("nocolon", "module:callable"),
        ("no_such_module_xyz:fn", "cannot import"),
        (f"{DUMMY.parent / 'missing.py'}:fn", "not found"),
        (ref("no_such_function"), "no attribute"),
        (ref("raises_error"), "boom"),
        (ref("not_a_model"), "nn.Module"),
    ],
)
def test_factory_failures(factory, message):
    spec = TorchModelSpec(model_id="m", source="factory", num_classes=3, factory=factory)
    with pytest.raises(ModelLoadError, match=message):
        TorchImageClassifier(spec, device="cpu")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"source": "torchscript"},
        {"source": "torchscript", "path": "a.pt", "factory": "m:f"},
        {"source": "module", "path": "a.pt", "factory_kwargs": {"a": 1}},
        {"source": "state_dict", "path": "a.pt"},
        {"source": "state_dict", "factory": "m:f"},
        {"source": "factory"},
        {"source": "factory", "factory": "m:f", "path": "a.pt"},
        {"source": "factory", "factory": "m:f", "state_dict_key": "k"},
        {"source": "factory", "factory": "m:f", "strip_prefix": "module."},
        {"source": "factory", "factory": "m:f", "class_names": ["a"]},
        {"source": "factory", "factory": "m:f", "single_logit_binary": True},
        {"source": "factory", "factory": "m:f", "num_classes": 0},
        {"source": "factory", "factory": "m:f", "unknown": 1},
        {"source": "bogus"},
    ],
)
def test_invalid_specs(kwargs):
    with pytest.raises(ValidationError):
        TorchModelSpec(**{"model_id": "m", "num_classes": 3, **kwargs})


def test_spec_json_roundtrip(tmp_path):
    spec = state_spec(tmp_path / "w.pt", state_dict_key="sd")
    assert TorchModelSpec.model_validate_json(spec.model_dump_json()) == spec


def test_mixed_precision_is_ignored_on_cpu():
    adapter = cpu_adapter(mixed_precision=True)
    assert adapter.metadata().details["mixed_precision"] is False
    assert [p.predicted_class for p in adapter.predict_batch(rgb_samples())] == [0, 1, 2]


def test_resolve_device():
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type == ("cuda" if HAS_CUDA else "cpu")
    with pytest.raises(ConfigurationError):
        resolve_device("tpu")


@pytest.mark.skipif(HAS_CUDA, reason="cuda is available")
def test_cuda_requested_but_unavailable():
    with pytest.raises(ConfigurationError, match="CUDA"):
        resolve_device("cuda")


@pytest.mark.skipif(not HAS_CUDA, reason="cuda not available")
def test_cuda_matches_cpu():
    x = rand_input()
    gpu = TorchImageClassifier(make_spec("random_classifier"), device="cuda")
    assert gpu.metadata().device.startswith("cuda")
    cpu = cpu_adapter("random_classifier")
    assert np.allclose(gpu.predict_probabilities(x), cpu.predict_probabilities(x), atol=1e-5)


@pytest.mark.skipif(not HAS_CUDA, reason="cuda not available")
def test_cuda_mixed_precision():
    gpu = TorchImageClassifier(make_spec(mixed_precision=True), device="cuda")
    assert gpu.metadata().details["mixed_precision"] is True
    preds = gpu.predict_batch(rgb_samples())
    assert [p.predicted_class for p in preds] == [0, 1, 2]
    assert preds[0].confidence == pytest.approx(np.exp(10) / (np.exp(10) + 2), abs=1e-2)
