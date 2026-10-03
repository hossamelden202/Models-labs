import textwrap

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from PIL import Image  # noqa: E402
from torch import nn  # noqa: E402

from modellab.core.errors import ModelLoadError  # noqa: E402
from modellab.core.results import ArtifactStore  # noqa: E402
from modellab.evaluation.engine import EvalConfig, run_evaluation  # noqa: E402
from modellab.evaluation.image_dataset import ImageClassificationDataset  # noqa: E402
from modellab.evaluation.preprocessing import PreprocessConfig  # noqa: E402
from modellab.evaluation.torch_model import TorchImageClassifier, TorchModelSpec  # noqa: E402
from modellab.loading.architectures import (  # noqa: E402
    ArchitectureDef,
    build_architecture,
    describe_architectures,
    register_architecture,
    unregister_architecture,
)
from modellab.loading.artifact import inspect_artifact  # noqa: E402
from modellab.loading.diagnostics import CheckpointMismatchError  # noqa: E402
from modellab.loading.resolver import ModelRequest, ResolveSettings, resolve_model  # noqa: E402

NAMES = ["RED", "GREEN", "BLUE"]


def layers(width=4):
    return [
        {"type": "Conv2d", "args": {"in_channels": 3, "out_channels": width, "kernel_size": 3, "padding": 1}},
        {"type": "BatchNorm2d", "args": {"num_features": width}},
        {"type": "ReLU", "args": {}},
        {"type": "AdaptiveAvgPool2d", "args": {"output_size": 1}},
        {"type": "Flatten", "args": {}},
        {"type": "Linear", "args": {"in_features": width, "out_features": 3}},
    ]


CONFIG = {"layers": layers()}
torch.manual_seed(5)
X = torch.randn(4, 3, 8, 8)


def reference(config=CONFIG, seed=0):
    torch.manual_seed(seed)
    if isinstance(config, list):
        config = {"layers": config}
    module = build_architecture("sequential", config, 3)
    for m in module.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.running_mean.uniform_(-1, 1)
            m.running_var.uniform_(0.5, 1.5)
    return module.eval()


def wrap(sd):
    return {
        "model_state_dict": sd, "class_names": NAMES, "image_size": 8, "image_mean": [0.5] * 3,
        "image_std": [0.25] * 3, "epoch": 3, "optimizer": {"state": {}, "param_groups": []},
    }


def save(tmp_path, name, obj):
    path = tmp_path / name
    torch.save(obj, path)
    return path


def request(path, **kwargs):
    return ModelRequest(artifact=str(path), **kwargs)


def outputs(spec_dict):
    clf = TorchImageClassifier(TorchModelSpec.model_validate(spec_dict), device="cpu")
    return clf.predict_probabilities(X)


def expected(module):
    with torch.no_grad():
        return torch.softmax(module(X), dim=1).numpy()


def test_inspection_identifies_formats_and_reads_hints(tmp_path):
    ref = reference()
    sd = ref.state_dict()
    plain = inspect_artifact(save(tmp_path, "plain.pt", sd))
    assert plain.format == "torch_archive" and plain.self_contained is False
    summary = plain.state_dicts[0]
    assert summary.key_path is None and summary.num_tensors == len(sd)
    assert summary.num_parameters == sum(t.numel() for t in sd.values())
    assert summary.top_level_prefixes == ["0", "1", "5"] and summary.last_matrix == {"key": "5.weight", "shape": [3, 4]}

    wrapped = inspect_artifact(save(tmp_path, "wrapped.pt", wrap(sd)))
    assert [s.key_path for s in wrapped.state_dicts] == ["model_state_dict"] and wrapped.metadata["epoch"] == 3
    hints = {h["field"]: h["value"] for h in wrapped.hints}
    assert hints["class_names"] == NAMES and hints["input_size"] == [8, 8]
    assert hints["preprocess.mean"] == [0.5] * 3 and hints["preprocess.std"] == [0.25] * 3

    prefixed = inspect_artifact(save(tmp_path, "dp.pt", {f"module.{k}": v for k, v in sd.items()}))
    assert prefixed.state_dicts[0].common_prefix == "module."

    scripted = tmp_path / "ts.pt"
    torch.jit.trace(ref, torch.zeros(1, 3, 8, 8)).save(str(scripted))
    ts = inspect_artifact(scripted)
    assert ts.format == "torchscript" and ts.self_contained is True
    pickled = inspect_artifact(save(tmp_path, "full.pt", ref))
    assert pickled.format == "pickle_object" and pickled.warnings
    (tmp_path / "m.onnx").write_bytes(b"\x08\x07onnx")
    (tmp_path / "junk.bin").write_bytes(b"random bytes that are no model")
    assert inspect_artifact(tmp_path / "m.onnx").format == "onnx"
    assert inspect_artifact(tmp_path / "junk.bin").format == "unknown"
    assert inspect_artifact(tmp_path / "nope.pt").format == "missing"
    assert str(tmp_path) not in plain.model_dump_json() + wrapped.model_dump_json()
    assert {a["name"] for a in describe_architectures()} >= {"sequential", "torchvision", "timm"}


def test_safetensors_artifact(tmp_path):
    safetensors = pytest.importorskip("safetensors.torch")
    ref = reference()
    path = tmp_path / "w.safetensors"
    safetensors.save_file({k: v.contiguous() for k, v in ref.state_dict().items()}, str(path))
    info = inspect_artifact(path)
    assert info.format == "safetensors" and info.state_dicts[0].num_tensors == len(ref.state_dict())
    res = resolve_model(request(path, class_names=NAMES, input_size=(8, 8), architecture="sequential", architecture_config=CONFIG))
    assert res.status == "ready" and np.allclose(outputs(res.spec), expected(ref), atol=1e-5)


def test_self_contained_models_and_pickle_gating(tmp_path):
    ref = reference()
    scripted = tmp_path / "ts.pt"
    torch.jit.trace(ref, torch.zeros(1, 3, 8, 8)).save(str(scripted))
    res = resolve_model(request(scripted, class_names=NAMES, input_size=(8, 8)))
    assert res.status == "ready" and res.route == "self_contained" and res.spec["source"] == "torchscript"
    assert [c["check"] for c in res.validation["checks"]] == ["model_loaded", "forward_pass"]
    assert np.allclose(outputs(res.spec), expected(ref), atol=1e-5)

    needs = resolve_model(request(scripted))
    assert needs.status == "needs_input" and [m.field for m in needs.missing] == ["num_classes"]
    measured = resolve_model(request(scripted, input_size=(8, 8)))
    assert measured.status == "ready" and measured.spec["num_classes"] == 3
    assert any(i.field == "num_classes" and "forward pass" in i.source for i in measured.inferred)
    wrong = resolve_model(request(scripted, class_names=NAMES, input_size=(8, 8), preprocess={"color_mode": "L"}))
    assert wrong.status == "invalid" and any("forward pass failed" in e for e in wrong.errors)
    both = resolve_model(request(scripted, class_names=NAMES, architecture="sequential"))
    assert both.status == "invalid" and "self-contained" in both.errors[0]

    full = save(tmp_path, "full.pt", ref)
    untrusted = resolve_model(request(full, class_names=NAMES, input_size=(8, 8)))
    assert untrusted.status == "needs_input" and [m.field for m in untrusted.missing] == ["trust_executable"]
    trusted = resolve_model(request(full, class_names=NAMES, input_size=(8, 8), trust_executable=True))
    assert trusted.status == "ready" and trusted.spec["source"] == "module"
    assert np.allclose(outputs(trusted.spec), expected(ref), atol=1e-5)
    off = resolve_model(request(full, class_names=NAMES, trust_executable=True), ResolveSettings(allow_pickle=False))
    assert off.status == "invalid" and "disabled" in off.errors[0]
    mixed = resolve_model(request(full, class_names=NAMES, trust_executable=True, architecture="sequential"))
    assert mixed.status == "invalid" and "re-save" in mixed.errors[0]


def test_registered_architecture_configuration_and_missing_information(tmp_path):
    ref = reference()
    sd = ref.state_dict()
    wrapped = save(tmp_path, "wrapped.pt", wrap(sd))
    res = resolve_model(request(wrapped, architecture="sequential", architecture_config=CONFIG))
    assert res.status == "ready" and res.route == "registered_architecture"
    assert res.spec["architecture"] == "sequential" and res.spec["state_dict_key"] == "model_state_dict"
    assert res.spec["class_names"] == NAMES and res.spec["input_size"] == [8, 8]
    assert res.preprocess["mean"] == [0.5] * 3 and res.preprocess["std"] == [0.25] * 3
    applied = {i.field for i in res.inferred if i.applied}
    assert {"state_dict_key", "class_names", "num_classes", "input_size", "preprocess.mean"} <= applied
    assert res.diagnostics["compatible"] and res.diagnostics["num_matched"] == len(sd)
    assert np.allclose(outputs(res.spec), expected(ref), atol=1e-5)
    clf = TorchImageClassifier(TorchModelSpec.model_validate(res.spec), device="cpu")
    assert clf.metadata().details["architecture"] == "sequential"
    assert TorchModelSpec.model_validate(res.spec).model_dump(mode="json") == res.spec

    wide = reference(layers(6), seed=1)
    wide_path = save(tmp_path, "wide.pt", wide.state_dict())
    res = resolve_model(request(wide_path, class_names=NAMES, input_size=(8, 8), architecture="sequential", architecture_config={"layers": layers(6)}))
    assert res.status == "ready" and np.allclose(outputs(res.spec), expected(wide), atol=1e-5)

    plain = save(tmp_path, "plain.pt", sd)
    prefixed = save(tmp_path, "dp.pt", {f"module.{k}": v for k, v in sd.items()})
    res = resolve_model(request(prefixed, class_names=NAMES, input_size=(8, 8), architecture="sequential", architecture_config=CONFIG))
    assert res.status == "ready" and res.spec["strip_prefix"] == "module."
    assert any(i.field == "strip_prefix" and i.applied for i in res.inferred)
    assert np.allclose(outputs(res.spec), expected(ref), atol=1e-5)

    res = resolve_model(request(plain, class_names=NAMES, probe_models=["resnet18"]))
    assert res.status == "needs_input" and res.missing[0].field == "architecture"
    assert "sequential" in res.missing[0].options and "auto" in res.missing[0].options and res.candidates == []
    res = resolve_model(request(plain, class_names=NAMES, architecture="sequential"))
    assert res.status == "needs_input" and res.missing[0].field == "architecture_config.layers"
    assert res.missing[0].json_schema["required"] == ["layers"]
    res = resolve_model(request(plain, architecture="sequential", architecture_config=CONFIG))
    assert res.status == "needs_input" and res.missing[0].field == "num_classes"
    assert any(i.field == "num_classes" and i.value == 3 and not i.applied for i in res.inferred)
    for bad, text in [
        ({"layers": [{"type": "Bogus", "args": {}}]}, "unknown layer type"),
        ({"layers": [{"type": "Conv2d", "args": {"in_channels": 3, "nope": 1}}]}, "invalid arguments"),
    ]:
        res = resolve_model(request(plain, class_names=NAMES, architecture="sequential", architecture_config=bad))
        assert res.status == "invalid" and text in res.errors[0]
    res = resolve_model(request(plain, class_names=NAMES + ["EXTRA"], architecture="sequential", architecture_config=CONFIG))
    assert res.status == "invalid" and "outputs" in " ".join(res.errors)
    res = resolve_model(request(plain, class_names=NAMES, architecture="nope"))
    assert res.status == "invalid" and "registered" in res.errors[0]
    res = resolve_model(request(plain, architecture_config=CONFIG))
    assert res.status == "invalid" and "needs architecture" in res.errors[0]


def test_checkpoint_diagnostics_never_hide_incompatibilities(tmp_path):
    ref = reference()
    sd = dict(ref.state_dict())
    asked = dict(class_names=NAMES, input_size=(8, 8), architecture="sequential", architecture_config=CONFIG)

    short = {k: v for k, v in sd.items() if k != "5.bias"}
    res = resolve_model(request(save(tmp_path, "short.pt", short), **asked))
    assert res.status == "invalid" and res.diagnostics["missing_keys"] == ["5.bias"] and "missing keys" in res.errors[0]
    res = resolve_model(request(save(tmp_path, "extra.pt", {**sd, "extra.weight": torch.zeros(2)}), **asked))
    assert res.status == "invalid" and res.diagnostics["unexpected_keys"] == ["extra.weight"]
    wrong_config = {"layers": layers()[:-1] + [{"type": "Linear", "args": {"in_features": 4, "out_features": 4}}]}
    wrong = save(tmp_path, "four.pt", build_architecture("sequential", wrong_config, 4).state_dict())
    res = resolve_model(request(wrong, **asked))
    assert res.status == "invalid" and [m["key"] for m in res.diagnostics["shape_mismatches"]] == ["5.bias", "5.weight"]
    assert res.diagnostics["shape_mismatches"][0] == {"key": "5.bias", "expected": [3], "found": [4]}

    spec = TorchModelSpec(model_id="m", source="state_dict", path=tmp_path / "short.pt", architecture="sequential", architecture_config=CONFIG, num_classes=3)
    with pytest.raises(CheckpointMismatchError) as caught:
        TorchImageClassifier(spec, device="cpu")
    assert isinstance(caught.value, ModelLoadError) and caught.value.diagnostics["compatible"] is False
    assert str(caught.value).startswith("could not load model 'm'")

    tolerant = {k: v for k, v in sd.items() if not k.endswith("num_batches_tracked")}
    res = resolve_model(request(save(tmp_path, "tol.pt", tolerant), **asked))
    assert res.status == "ready" and res.diagnostics["tolerated_missing"] == ["1.num_batches_tracked"]
    assert any("num_batches_tracked" in w for w in res.warnings) and np.allclose(outputs(res.spec), expected(ref), atol=1e-5)

    res = resolve_model(request(save(tmp_path, "none.pt", {"epoch": 1, "name": "x"}), **asked))
    assert res.status == "invalid" and "no state dict" in res.errors[0]
    two = save(tmp_path, "two.pt", {"model": sd, "ema": sd})
    res = resolve_model(request(two, **asked))
    assert res.status == "needs_input" and res.missing[0].field == "state_dict_key" and sorted(res.missing[0].options) == ["ema", "model"]
    assert resolve_model(request(two, state_dict_key="ema", **asked)).status == "ready"
    res = resolve_model(request(two, state_dict_key="nope", **asked))
    assert res.status == "invalid" and "nope" in res.errors[0]


FACTORY = textwrap.dedent(
    """
    from torch import nn

    def build(width=4):
        return nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(width, 3),
        )
    """
)


def test_custom_factory_route_security_and_legacy_compatibility(tmp_path):
    factory = tmp_path / "custom_factory.py"
    factory.write_text(FACTORY)
    ref = reference()
    ckpt = save(tmp_path, "ck.pt", ref.state_dict())
    asked = dict(class_names=NAMES, input_size=(8, 8), factory="build")

    res = resolve_model(request(ckpt, **asked), factory_file=factory)
    assert res.status == "needs_input" and [m.field for m in res.missing] == ["trust_executable"]
    res = resolve_model(request(ckpt, trust_executable=True, **asked), factory_file=factory)
    assert res.status == "ready" and res.route == "custom_factory" and res.spec["source"] == "state_dict"
    assert res.spec["factory"] == f"{factory}:build" and "architecture" not in res.spec
    assert np.allclose(outputs(res.spec), expected(ref), atol=1e-5)
    off = resolve_model(request(ckpt, trust_executable=True, **asked), ResolveSettings(allow_custom_code=False), factory_file=factory)
    assert off.status == "invalid" and "disabled" in off.errors[0]
    colon = resolve_model(request(ckpt, trust_executable=True, **{**asked, "factory": "a:b"}), factory_file=factory)
    assert colon.status == "invalid" and "callable name only" in colon.errors[0]
    wide = resolve_model(request(ckpt, trust_executable=True, factory_kwargs={"width": 8}, **asked), factory_file=factory)
    assert wide.status == "invalid" and wide.diagnostics["num_shape_mismatches"] > 0
    both = resolve_model(request(ckpt, trust_executable=True, architecture="sequential", **asked), factory_file=factory)
    assert both.status == "invalid" and "not both" in both.errors[0]

    alone = resolve_model(ModelRequest(factory=f"{factory}:build", num_classes=3, input_size=(8, 8)))
    assert alone.status == "needs_input" and alone.missing[0].field == "trust_executable"
    alone = resolve_model(ModelRequest(factory=f"{factory}:build", num_classes=3, input_size=(8, 8), trust_executable=True))
    assert alone.status == "ready" and alone.spec["source"] == "factory"
    assert resolve_model(ModelRequest()).missing[0].field == "artifact"

    legacy = TorchModelSpec(model_id="x", source="factory", num_classes=3, factory=f"{factory}:build")
    assert "architecture" not in legacy.model_dump(mode="json") and "architecture_config" not in legacy.model_dump_json()
    clf = TorchImageClassifier(legacy, device="cpu")
    assert clf.metadata().details == {"source": "factory", "mixed_precision": False}
    old = TorchModelSpec(model_id="y", source="state_dict", path=ckpt, factory=f"{factory}:build", num_classes=3)
    assert np.allclose(TorchImageClassifier(old, device="cpu").predict_probabilities(X), expected(ref), atol=1e-5)
    bad = save(tmp_path, "bad.pt", build_architecture("sequential", {"layers": layers()[:-1] + [{"type": "Linear", "args": {"in_features": 4, "out_features": 4}}]}, 4).state_dict())
    with pytest.raises(ModelLoadError, match="could not load"):
        TorchImageClassifier(
            TorchModelSpec(
                model_id="z",
                source="state_dict",
                path=bad,
                factory=f"{factory}:build",
                num_classes=3,
            )
        )
    with pytest.raises(ValueError, match="either architecture or factory"):
        TorchModelSpec(model_id="w", source="state_dict", path=ckpt, factory="a:b", architecture="sequential", num_classes=3)
    with pytest.raises(ValueError, match="either architecture or factory"):
        TorchModelSpec(model_id="w", source="factory", factory="a:b", architecture="sequential", num_classes=3)


def test_auto_architecture_candidates_and_ambiguity(tmp_path):
    tv = pytest.importorskip("torchvision")
    torch.manual_seed(0)
    net = tv.models.resnet18(weights=None, num_classes=5).eval()
    path = save(tmp_path, "r18.pt", net.state_dict())
    asked = dict(num_classes=5, input_size=(32, 32), probe_models=["resnet18", "resnet34", "mobilenet_v3_small"])

    res = resolve_model(request(path, **asked))
    assert res.status == "needs_input" and res.missing[0].field == "architecture"
    assert [(c.architecture, c.config["name"]) for c in res.candidates] == [("torchvision", "resnet18")]
    res = resolve_model(request(path, architecture="auto", **asked))
    assert res.status == "ready" and res.spec["architecture"] == "torchvision"
    assert res.spec["architecture_config"]["name"] == "resnet18"
    assert any(i.field == "architecture" and i.applied for i in res.inferred)
    x = torch.randn(2, 3, 32, 32)
    clf = TorchImageClassifier(TorchModelSpec.model_validate(res.spec), device="cpu")
    with torch.no_grad():
        assert np.allclose(clf.predict_probabilities(x), torch.softmax(net(x), dim=1).numpy(), atol=1e-4)
    explicit = resolve_model(request(path, architecture="torchvision", architecture_config={"name": "resnet34"}, **asked))
    assert explicit.status == "invalid" and explicit.diagnostics["num_missing_keys"] > 0
    unknown = resolve_model(request(path, architecture="torchvision", architecture_config={"name": "nope"}, **asked))
    assert unknown.status == "invalid" and "unknown torchvision model" in " ".join(unknown.errors)
    assert resolve_model(request(path, architecture="auto", probe_architectures=False, num_classes=5)).status == "needs_input"
    assert resolve_model(request(path, architecture="auto", input_size=(32, 32))).missing[0].field == "num_classes"

    from modellab.loading.architectures import ARCHITECTURES

    register_architecture(ArchitectureDef(name="resnet_alias", description="alias", config_model=ARCHITECTURES["torchvision"].config_model,
                                          build=ARCHITECTURES["torchvision"].build, probe=ARCHITECTURES["torchvision"].probe))
    try:
        res = resolve_model(request(path, architecture="auto", **asked))
        assert res.status == "needs_input" and len(res.candidates) == 2
        assert {c.architecture for c in res.candidates} == {"torchvision", "resnet_alias"}
    finally:
        unregister_architecture("resnet_alias")


def test_resolved_models_work_with_the_phase_2_evaluator(tmp_path):
    module = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 3))
    with torch.no_grad():
        module[2].weight.copy_(torch.eye(3) * 10)
        module[2].bias.zero_()
    ckpt = save(tmp_path, "identity.pt", {"model_state_dict": module.state_dict(), "class_names": NAMES, "image_size": 8})
    config = {"layers": [
        {"type": "AdaptiveAvgPool2d", "args": {"output_size": 1}}, {"type": "Flatten", "args": {}},
        {"type": "Linear", "args": {"in_features": 3, "out_features": 3}},
    ]}
    res = resolve_model(request(ckpt, architecture="sequential", architecture_config=config))
    assert res.status == "ready"
    root = tmp_path / "imgs"
    for cls, color in zip(NAMES, [(200, 50, 40), (40, 200, 50), (40, 50, 200)], strict=True):
        for i in range(4):
            path = root / cls / f"{i}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 8), color).save(path)
    model = TorchImageClassifier(TorchModelSpec.model_validate(res.spec), device="cpu")
    dataset = ImageClassificationDataset.from_folder(root, class_names=NAMES, preprocess=PreprocessConfig.model_validate(res.preprocess))
    store = ArtifactStore(tmp_path / "art")
    result = run_evaluation(model, dataset, store, EvalConfig(batch_size=4), evaluation_id="e1")
    assert result.metrics["accuracy"] == 1.0 and result.metrics["num_samples"] == 12
    meta = result.configuration["model"]
    assert meta["architecture"] == "sequential" and meta["architecture_config"] == res.spec["architecture_config"]
