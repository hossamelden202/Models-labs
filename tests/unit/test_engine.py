import importlib.util  # noqa: F401
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")
pytest.importorskip("sklearn")

from PIL import Image  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from modellab.core.errors import DatasetError, EvaluationError  # noqa: E402
from modellab.core.results import ArtifactStore  # noqa: E402
from modellab.evaluation.engine import EvalConfig, run_evaluation  # noqa: E402
from modellab.evaluation.image_dataset import ImageClassificationDataset  # noqa: E402
from modellab.evaluation.preprocessing import PreprocessConfig  # noqa: E402
from modellab.evaluation.torch_model import TorchImageClassifier, TorchModelSpec  # noqa: E402

DUMMY = Path(__file__).resolve().parents[1] / "dummy_models.py"
COLORS = {"red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255)}
MIXED = {
    "blood": ["red", "red", "green"],
    "gore": ["green", "green"],
    "neutral": ["blue", "blue"],
}
HAS_CUDA = torch.cuda.is_available()
COLUMNS = [
    "sample_id",
    "path",
    "true_label",
    "predicted_label",
    "confidence",
    "correct",
    "class_probabilities",
]


def build_folder(root, layout, size=8):
    for cls, colors in layout.items():
        for i, color in enumerate(colors):
            path = root / cls / f"{i}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (size, size), COLORS[color]).save(path)
    return root


def make_dataset(root, layout=None, class_names=None, preprocessed=True):
    build_folder(root, layout or MIXED)
    pre = PreprocessConfig() if preprocessed else None

    # This evaluation fixture intentionally uses blood=0, gore=1, neutral=2.
    if class_names is None and layout is None:
        class_names = ["blood", "gore", "neutral"]

    return ImageClassificationDataset.from_folder(
        root,
        class_names=class_names,
        preprocess=pre,
    )


def make_model(factory="identity_classifier", num_classes=3, class_names=None, device="cpu"):
    spec = TorchModelSpec(
        model_id="m",
        source="factory",
        num_classes=num_classes,
        factory=f"{DUMMY}:{factory}",
        class_names=class_names,
    )
    return TorchImageClassifier(spec, device=device)


def run(tmp_path, model, dataset, eval_id=None, **cfg):
    store = ArtifactStore(tmp_path / "art")
    result = run_evaluation(model, dataset, store, EvalConfig(**cfg), evaluation_id=eval_id)
    return result, store.root / "evaluations" / result.configuration["evaluation_id"]


def label_metrics(metrics):
    keys = ("num_samples", "accuracy", "macro_precision", "macro_recall", "macro_f1")
    return {k: metrics[k] for k in (*keys, "weighted_f1", "per_class")}


def test_multiclass_metrics_and_confusion(tmp_path):
    result, out = run(tmp_path, make_model(), make_dataset(tmp_path / "ds"), batch_size=4)
    m = result.metrics
    assert m["num_samples"] == 7
    assert m["accuracy"] == pytest.approx(6 / 7)
    assert m["macro_precision"] == pytest.approx(8 / 9)
    assert m["macro_recall"] == pytest.approx(8 / 9)
    assert m["macro_f1"] == pytest.approx((0.8 + 0.8 + 1.0) / 3)
    assert m["weighted_f1"] == pytest.approx(6 / 7)
    per_class = {c["class_name"]: c for c in m["per_class"]}
    assert per_class["blood"]["recall"] == pytest.approx(2 / 3)
    assert per_class["blood"]["precision"] == pytest.approx(1.0)
    assert per_class["gore"]["precision"] == pytest.approx(2 / 3)
    assert per_class["gore"]["recall"] == pytest.approx(1.0)
    assert [c["support"] for c in m["per_class"]] == [3, 2, 2]
    assert m["auroc"]["status"] == "ok"

    confusion = json.loads((out / "confusion_matrix.json").read_text())
    assert confusion["matrix"] == [
        [2, 1, 0],
        [0, 2, 0],
        [0, 0, 2],
    ]
    assert confusion["labels"] == ["blood", "gore", "neutral"]
    assert json.loads((out / "metrics.json").read_text()) == m


def test_binary_evaluation(tmp_path):
    layout = {"red": ["red", "red"], "green": ["green", "green", "red"]}
    names = ["red", "green"]
    ds = make_dataset(tmp_path / "ds", layout, class_names=names)
    model = make_model("identity_two_class", 2, class_names=names)
    result, out = run(tmp_path, model, ds, batch_size=2)
    m = result.metrics
    assert m["accuracy"] == pytest.approx(0.8)
    confusion = json.loads((out / "confusion_matrix.json").read_text())
    assert confusion["matrix"] == [[2, 0], [1, 2]]
    assert [c["support"] for c in m["per_class"]] == [2, 3]
    assert m["per_class"][0]["precision"] == pytest.approx(2 / 3)
    assert m["per_class"][1]["recall"] == pytest.approx(2 / 3)
    assert m["auroc"]["status"] == "ok"
    assert 0.5 <= m["auroc"]["value"] <= 1.0
    assert m["auprc"]["status"] == "ok"


def test_predictions_file(tmp_path):
    _, out = run(tmp_path, make_model(), make_dataset(tmp_path / "ds"), batch_size=3)
    df = pd.read_parquet(out / "predictions.parquet")
    assert list(df.columns) == COLUMNS
    assert df.sample_id.tolist() == [
        "blood/0.png",
        "blood/1.png",
        "blood/2.png",
        "gore/0.png",
        "gore/1.png",
        "neutral/0.png",
        "neutral/1.png",
    ]
    assert df.true_label.tolist() == [0, 0, 0, 1, 1, 2, 2]
    assert df.predicted_label.tolist() == [0, 0, 1, 1, 1, 2, 2]
    assert df.correct.tolist() == [True, True, False, True, True, True, True]
    assert df.path[0] == str(tmp_path / "ds" / "blood" / "0.png")
    probs = np.stack(df.class_probabilities.to_list())
    assert probs.shape == (7, 3)
    assert probs.sum(axis=1) == pytest.approx(np.ones(7), abs=1e-5)
    assert df.confidence.to_numpy() == pytest.approx(probs.max(axis=1), abs=1e-6)
    assert (probs.argmax(axis=1) == df.predicted_label.to_numpy()).all()


def test_artifact_files_and_metadata(tmp_path):
    result, out = run(
        tmp_path, make_model(), make_dataset(tmp_path / "ds"), eval_id="run1", batch_size=4, seed=7
    )
    names = sorted(p.name for p in out.iterdir())
    assert names == [
        "confusion_matrix.json",
        "metadata.json",
        "metrics.json",
        "predictions.parquet",
    ]
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["evaluation_id"] == "run1"
    assert meta["model_id"] == "m"
    assert meta["dataset_id"] == "ds"
    assert meta["seed"] == 7
    assert meta["device"] == "cpu"
    assert meta["batch_size"] == 4
    assert meta["num_samples"] == 7
    assert meta["num_classes"] == 3
    assert meta["class_names"] ==["blood", "gore", "neutral"]
    assert meta["preprocessing"] == PreprocessConfig().model_dump(mode="json")
    assert meta["model"]["source"] == "factory"
    assert "timestamp" in meta and "torch" in meta["versions"]
    assert result.prediction_artifact == "evaluations/run1/predictions.parquet"
    assert result.model_id == "m" and result.dataset_id == "ds"


@pytest.mark.parametrize("batch_size", [1, 3, 100])
def test_batch_size_does_not_change_results(tmp_path, batch_size):
    ds = make_dataset(tmp_path / "ds")
    ref, ref_out = run(tmp_path, make_model(), ds, eval_id="ref", batch_size=2)
    got, got_out = run(tmp_path, make_model(), ds, eval_id="got", batch_size=batch_size)
    assert label_metrics(got.metrics) == label_metrics(ref.metrics)
    a = pd.read_parquet(ref_out / "predictions.parquet")
    b = pd.read_parquet(got_out / "predictions.parquet")
    assert a.predicted_label.tolist() == b.predicted_label.tolist()
    pa_, pb_ = (np.stack(f.class_probabilities.to_list()) for f in (a, b))
    assert np.allclose(pa_, pb_, atol=1e-6)


def test_reproducible_results(tmp_path):
    ds = make_dataset(tmp_path / "ds")
    first, first_out = run(tmp_path, make_model("random_classifier"), ds, "one", batch_size=3)
    second, second_out = run(tmp_path, make_model("random_classifier"), ds, "two", batch_size=3)
    assert first.metrics == second.metrics
    a = pd.read_parquet(first_out / "predictions.parquet")
    b = pd.read_parquet(second_out / "predictions.parquet")
    assert a.drop(columns="class_probabilities").equals(b.drop(columns="class_probabilities"))
    pa_, pb_ = (np.stack(f.class_probabilities.to_list()) for f in (a, b))
    assert np.array_equal(pa_, pb_)


def test_worker_processes_give_same_predictions(tmp_path):
    ds = make_dataset(tmp_path / "ds")
    _, base = run(tmp_path, make_model(), ds, "base", batch_size=2)
    _, workers = run(tmp_path, make_model(), ds, "workers", batch_size=2, num_workers=2)
    a = pd.read_parquet(base / "predictions.parquet")
    b = pd.read_parquet(workers / "predictions.parquet")
    assert a.sample_id.tolist() == b.sample_id.tolist()
    assert a.predicted_label.tolist() == b.predicted_label.tolist()


def test_class_count_mismatch(tmp_path):
    ds = make_dataset(tmp_path / "ds", {"blood": ["red"], "gore": ["green"]})
    with pytest.raises(EvaluationError, match="2 classes"):
        run(tmp_path, make_model(), ds)


def test_class_order_mismatch(tmp_path):
    ds = make_dataset(tmp_path / "ds")
    model = make_model(class_names=["neutral", "gore", "blood"])
    with pytest.raises(EvaluationError, match="order"):
        run(tmp_path, model, ds)


def test_corrupt_image_is_reported_and_leaves_no_partial_output(tmp_path):
    root = build_folder(tmp_path / "ds", MIXED)
    (root / "blood" / "bad.png").write_bytes(b"not an image")
    ds = ImageClassificationDataset.from_folder(root, preprocess=PreprocessConfig())
    with pytest.raises(DatasetError, match="bad.png"):
        run(tmp_path, make_model(), ds, batch_size=2)
    evaluations = tmp_path / "art" / "evaluations"
    assert not evaluations.exists() or not any(evaluations.iterdir())


def test_evaluation_id_rules(tmp_path):
    ds = make_dataset(tmp_path / "ds")
    run(tmp_path, make_model(), ds, eval_id="dup")
    with pytest.raises(EvaluationError, match="already exists"):
        run(tmp_path, make_model(), ds, eval_id="dup")
    for bad in ("a/b", "../x", ".."):
        with pytest.raises(EvaluationError, match="invalid evaluation id"):
            run(tmp_path, make_model(), ds, eval_id=bad)


def test_dataset_without_preprocessing_is_rejected(tmp_path):
    ds = make_dataset(tmp_path / "ds", preprocessed=False)
    with pytest.raises(EvaluationError, match="tensors"):
        run(tmp_path, make_model(), ds)


def test_images_of_different_sizes_are_rejected(tmp_path):
    root = build_folder(tmp_path / "ds", {"blood": ["red"], "gore": ["green"]})
    Image.new("RGB", (6, 6), COLORS["green"]).save(root / "gore" / "0.png")
    ds = ImageClassificationDataset.from_folder(root, preprocess=PreprocessConfig())
    model = make_model("identity_two_class", 2)
    with pytest.raises(EvaluationError, match="different shapes"):
        run(tmp_path, model, ds, batch_size=2)


def test_resize_makes_mixed_sizes_work(tmp_path):
    root = build_folder(tmp_path / "ds", {"blood": ["red"], "gore": ["green"]})
    Image.new("RGB", (6, 6), COLORS["green"]).save(root / "gore" / "0.png")
    pre = PreprocessConfig(resize=(4, 4))
    ds = ImageClassificationDataset.from_folder(root, preprocess=pre)
    result, _ = run(tmp_path, make_model("identity_two_class", 2), ds, batch_size=2)
    assert result.metrics["accuracy"] == 1.0


def test_eval_config_validation():
    with pytest.raises(ValidationError):
        EvalConfig(batch_size=0)
    with pytest.raises(ValidationError):
        EvalConfig(num_workers=-1)
    with pytest.raises(ValidationError):
        EvalConfig(unknown=1)


@pytest.mark.skipif(not HAS_CUDA, reason="cuda not available")
def test_cuda_evaluation_matches_cpu(tmp_path):
    ds = make_dataset(tmp_path / "ds")
    cpu, _ = run(tmp_path, make_model(), ds, eval_id="cpu", batch_size=4)
    gpu, gpu_out = run(
        tmp_path, make_model(device="cuda"), ds, eval_id="gpu", batch_size=4
    )
    assert gpu.configuration["device"].startswith("cuda")
    assert label_metrics(gpu.metrics) == label_metrics(cpu.metrics)
