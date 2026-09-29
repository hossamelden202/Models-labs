import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("pandas")
pytest.importorskip("pyarrow")
pytest.importorskip("sklearn")

from PIL import Image  # noqa: E402

from modellab.cli.main import main  # noqa: E402

DUMMY = Path(__file__).resolve().parents[1] / "dummy_models.py"
COLORS = {"red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255)}
MIXED = {
    "blood": ["red", "red", "green"],
    "gore": ["green", "green"],
    "neutral": ["blue", "blue"],
}
FILES = ["confusion_matrix.json", "metadata.json", "metrics.json", "predictions.parquet"]


def build_folder(root):
    for cls, colors in MIXED.items():
        for i, color in enumerate(colors):
            path = root / cls / f"{i}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 8), COLORS[color]).save(path)
    return root


def write_model_file(path, num_classes=3):
    path.write_text(
        "model:\n"
        "  model_id: cli-model\n"
        "  source: factory\n"
        f"  num_classes: {num_classes}\n"
        f'  factory: "{DUMMY}:identity_classifier"\n'
        "  class_names: [blood, gore, neutral]\n"
        "preprocess:\n"
        "  resize: [8, 8]\n"
    )
    return path


def evaluate_args(model_file, dataset, out, *extra):
    return [
        "evaluate",
        "--model",
        str(model_file),
        "--dataset",
        str(dataset),
        "--output",
        str(out),
        "--device",
        "cpu",
        "--batch-size",
        "4",
        *extra,
    ]


def single_run_dir(out):
    dirs = list((out / "evaluations").iterdir())
    assert len(dirs) == 1
    return dirs[0]


def test_evaluate_folder(tmp_path, capsys):
    ds = build_folder(tmp_path / "ds")
    model_file = write_model_file(tmp_path / "model.yaml")
    out = tmp_path / "out"
    assert main(evaluate_args(model_file, ds, out)) == 0
    assert "accuracy" in capsys.readouterr().out

    run_dir = single_run_dir(out)
    assert sorted(p.name for p in run_dir.iterdir()) == FILES
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["num_samples"] == 7
    assert metrics["accuracy"] == pytest.approx(6 / 7)
    meta = json.loads((run_dir / "metadata.json").read_text())
    assert meta["model_id"] == "cli-model"
    assert meta["preprocessing"]["resize"] == [8, 8]
    assert meta["batch_size"] == 4


def test_evaluate_csv(tmp_path):
    build_folder(tmp_path / "ds")
    rows = ["path,label"]
    for cls, colors in MIXED.items():
        rows.extend(f"ds/{cls}/{i}.png,{cls}" for i in range(len(colors)))
    csv_file = tmp_path / "labels.csv"
    csv_file.write_text("\n".join(rows) + "\n")
    model_file = write_model_file(tmp_path / "model.yaml")
    out = tmp_path / "out"
    assert main(evaluate_args(model_file, csv_file, out)) == 0
    run_dir = single_run_dir(out)
    assert sorted(p.name for p in run_dir.iterdir()) == FILES
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["accuracy"] == pytest.approx(6 / 7)
    assert json.loads((run_dir / "metadata.json").read_text())["dataset_id"] == "labels"


def test_evaluation_id_option(tmp_path):
    ds = build_folder(tmp_path / "ds")
    model_file = write_model_file(tmp_path / "model.yaml")
    out = tmp_path / "out"
    assert main(evaluate_args(model_file, ds, out, "--evaluation-id", "fixed")) == 0
    assert (out / "evaluations" / "fixed" / "metrics.json").is_file()
    assert main(evaluate_args(model_file, ds, out, "--evaluation-id", "fixed")) == 1


def test_missing_model_file(tmp_path, capsys):
    ds = build_folder(tmp_path / "ds")
    code = main(evaluate_args(tmp_path / "nope.yaml", ds, tmp_path / "out"))
    assert code == 1
    assert "error:" in capsys.readouterr().err


def test_bad_dataset_path(tmp_path, capsys):
    model_file = write_model_file(tmp_path / "model.yaml")
    code = main(evaluate_args(model_file, tmp_path / "missing", tmp_path / "out"))
    assert code == 1
    assert "dataset" in capsys.readouterr().err


def test_invalid_model_file(tmp_path, capsys):
    ds = build_folder(tmp_path / "ds")
    bad = tmp_path / "bad.yaml"
    bad.write_text("model:\n  model_id: x\n  source: factory\n  num_classes: 3\n")
    assert main(evaluate_args(bad, ds, tmp_path / "out")) == 1
    assert "invalid model file" in capsys.readouterr().err


def test_invalid_batch_size(tmp_path):
    ds = build_folder(tmp_path / "ds")
    model_file = write_model_file(tmp_path / "model.yaml")
    args = evaluate_args(model_file, ds, tmp_path / "out")
    args[args.index("--batch-size") + 1] = "0"
    assert main(args) == 1


def test_invalid_device_is_rejected_by_the_parser(tmp_path):
    ds = build_folder(tmp_path / "ds")
    model_file = write_model_file(tmp_path / "model.yaml")
    args = evaluate_args(model_file, ds, tmp_path / "out")
    args[args.index("--device") + 1] = "tpu"
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2
