import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("scipy")
pytest.importorskip("sklearn")
pytest.importorskip("pyarrow")

import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from modellab.core.results import ArtifactStore  # noqa: E402
from modellab.evaluation.engine import EvalConfig, run_evaluation  # noqa: E402
from modellab.evaluation.image_dataset import ImageClassificationDataset  # noqa: E402
from modellab.evaluation.preprocessing import PreprocessConfig  # noqa: E402
from modellab.evaluation.torch_model import TorchImageClassifier, TorchModelSpec  # noqa: E402
from modellab.experiments import (  # noqa: E402
    ExperimentConfigError,
    ExperimentError,
    create_baseline,
    parse_spec,
    run_experiment,
    run_family,
)
from modellab.experiments import transforms as T  # noqa: E402

DUMMY = Path(__file__).resolve().parents[1] / "dummy_models.py"
COLORS = {"RED": (200, 50, 40), "GREEN": (40, 200, 50), "BLUE": (40, 50, 200)}
NAMES = list(COLORS)


def build(tmp_path, store_name="art"):
    root = tmp_path / "imgs"
    if not root.exists():
        for cls, color in COLORS.items():
            for i in range(12):
                path = root / cls / f"{i:03d}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (8, 8), color).save(path)
    model_spec = TorchModelSpec(
        model_id="toy", source="factory", num_classes=3,
        factory=f"{DUMMY}:identity_classifier", class_names=NAMES,
    )
    model = TorchImageClassifier(model_spec, device="cpu")
    dataset = ImageClassificationDataset.from_folder(root, class_names=NAMES, preprocess=PreprocessConfig())
    store = ArtifactStore(tmp_path / store_name)
    run_evaluation(model, dataset, store, EvalConfig(batch_size=4), evaluation_id="base")
    create_baseline(store, "fam", "base")
    return store, model, dataset, root


def spec(name, intervention, expected=None, **kwargs):
    return parse_spec(
        {"name": name, "hypothesis": {"claim": name, "expected_direction": expected},
         "intervention": intervention, **kwargs}
    )


def snapshot(root):
    return {
        p.relative_to(root).as_posix(): (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest())
        for p in sorted(root.rglob("*")) if p.is_file()
    }


def state(model):
    return {k: v.clone() for k, v in model.model.state_dict().items()}


def test_every_registered_op_keeps_size_and_mode_and_noise_is_seeded():
    arr = (np.add.outer(np.arange(16), np.arange(16)) * 8).clip(0, 255).astype(np.uint8)
    img = Image.merge(
        "RGB",
        [
            Image.fromarray(arr),
            Image.fromarray(arr[::-1]),
            Image.fromarray(np.fliplr(arr)),
        ],
    )
    params = {
        "brightness": {"factor": 1.5}, "contrast": {"factor": 0.5}, "saturation": {"factor": 0.0},
        "hue": {"shift": 0.25}, "blur": {"radius": 1.5}, "noise": {"std": 20},
        "jpeg": {"quality": 30}, "rotation": {"angle": 15}, "occlusion": {"fraction": 0.25},
        "crop": {"fraction": 0.5}, "hflip": {"semantically_valid": True}, "grayscale": {},
        "channel_swap": {"order": [2, 1, 0]}, "channel_drop": {"channel": 1},
    }
    assert set(params) == set(T.REGISTRY)
    for name, values in params.items():
        op = T.REGISTRY[name]
        out = op.apply(img, op.params(**values), T.sample_rng(1, 0, "a"))
        assert out.size == img.size and out.mode == img.mode, name
        assert not np.array_equal(np.asarray(out), np.asarray(img)), name

    noise = T.REGISTRY["noise"]
    p = noise.params(std=20)
    a = noise.apply(img, p, T.sample_rng(1, 0, "x"))
    b = noise.apply(img, p, T.sample_rng(1, 0, "x"))
    other = noise.apply(img, p, T.sample_rng(1, 0, "y"))
    later = noise.apply(img, p, T.sample_rng(1, 1, "x"))
    assert np.array_equal(np.asarray(a), np.asarray(b))
    assert not np.array_equal(np.asarray(a), np.asarray(other))
    assert not np.array_equal(np.asarray(a), np.asarray(later))
    occ = T.REGISTRY["occlusion"]
    centred = occ.params(fraction=0.25)
    assert not occ.stochastic(centred) and occ.stochastic(occ.params(fraction=0.25, position="random"))


def test_image_experiments_with_a_toy_model(tmp_path, monkeypatch):
    store, model, dataset, root = build(tmp_path)
    data_before = snapshot(root)
    weights_before = state(model)
    base_dir = store.root / "evaluations" / "base"
    base_files = snapshot(base_dir)

    drop = spec("drop_red", {"kind": "image_transform", "op": "channel_drop", "params": {"channel": 0}}, "degrade")
    red_only = spec(
        "drop_red_on_four", {"kind": "image_transform", "op": "channel_drop", "params": {"channel": 0}},
        population={"sample_ids": [f"RED/{i:03d}.png" for i in range(4)]},
    )
    flip = spec("flip", {"kind": "image_transform", "op": "hflip", "params": {"semantically_valid": True}}, "no_change")
    resize = spec("resize", {"kind": "preprocessing", "changes": {"resize": [4, 4]}}, "no_change")
    noise = spec("noise", {"kind": "image_transform", "op": "noise", "params": {"std": 120}},
                 repetitions=3, seed=7, n_bootstrap=200, n_permutations=500)
    run = run_family(store, "fam", [drop, red_only, flip, resize, noise], model=model, dataset=dataset)
    assert [o.status for o in run.outcomes] == ["completed"] * 5
    res = {o.result["spec"]["name"]: o.result for o in run.outcomes}

    d = res["drop_red"]
    st = d["statistics"]["affected"]
    assert (st["discordant_worsened"], st["discordant_improved"]) == (12, 0)
    assert st["mcnemar_p"] == pytest.approx(2 * 0.5**12, rel=1e-6)
    acc = d["comparison"]["global"]["differences"]["accuracy"]
    assert (acc["baseline"], acc["intervention"]) == (pytest.approx(1.0), pytest.approx(24 / 36))
    assert d["verdict"]["label"] == "degraded" and d["verdict"]["matches_expected"] is True
    assert d["evaluations"] == [f"{d['experiment_id']}_r0"]
    raw = pd.read_parquet(store.root / "evaluations" / d["evaluations"][0] / "predictions.parquet")
    assert len(raw) == 36 and (raw.predicted_label[raw.sample_id.str.startswith("RED")] == 1).all()

    part = res["drop_red_on_four"]
    assert part["population"]["num_affected"] == 4
    assert part["comparison"]["affected"]["differences"]["accuracy"]["difference"] == pytest.approx(-1.0)
    assert part["comparison"]["global"]["differences"]["accuracy"]["difference"] == pytest.approx(-4 / 36)
    assert len(pd.read_parquet(store.root / "evaluations" / part["evaluations"][0] / "predictions.parquet")) == 4

    for name in ("flip", "resize"):
        r = res[name]
        assert r["statistics"]["affected"]["mcnemar_p"] == 1.0
        assert r["comparison"]["global"]["differences"]["accuracy"]["difference"] == 0.0
        assert r["verdict"]["label"] == "no_practical_difference"
    assert res["resize"]["spec"]["intervention"]["changes"] == {"resize": [4, 4]}

    n = res["noise"]
    assert [r["seed"] for r in n["repetitions"]] == [7, 8, 9] and len(n["evaluations"]) == 3
    assert n["statistics"]["affected"]["primary_test"] == "sign_flip_permutation"
    assert n["statistics"]["affected"]["mcnemar_p"] is None

    assert snapshot(root) == data_before and snapshot(base_dir) == base_files
    after = state(model)
    assert all(torch.equal(weights_before[k], after[k]) for k in after)
    assert not model.model.training and d["model_state_sha256"] == res["flip"]["model_state_sha256"]

    other, model2, dataset2, _ = build(tmp_path, "art_b")
    run_family(other, "fam", [drop, red_only, flip, resize, noise], model=model2, dataset=dataset2)
    for o in run.outcomes:
        a = (store.root / "experiments/fam" / o.experiment_id / "result.json").read_bytes()
        b = (other.root / "experiments/fam" / o.experiment_id / "result.json").read_bytes()
        assert a == b, o.result["spec"]["name"]

    again = run_family(store, "fam", [drop, flip], model=model, dataset=dataset)
    assert [o.status for o in again.outcomes] == ["cached", "cached"]

    class Boom(T._P):
        pass

    def explode(img, p, rng):
        raise RuntimeError("boom")

    monkeypatch.setitem(T.REGISTRY, "boom", T.Op(Boom, explode))
    boom = spec("boom", {"kind": "image_transform", "op": "boom", "params": {}})
    mixed = run_family(store, "fam", [drop, boom], model=model, dataset=dataset)
    assert [o.status for o in mixed.outcomes] == ["cached", "failed"]
    failed = mixed.outcomes[1].result
    assert failed["status"] == "failed" and "boom" in failed["error"]["message"]
    assert mixed.report["num_failed"] == 1 and mixed.report["num_completed"] == 1
    assert mixed.report["failed"][0]["name"] == "boom"
    assert str(store.root) not in json.dumps(failed)
    retry = run_family(store, "fam", [drop, boom], model=model, dataset=dataset)
    assert [o.status for o in retry.outcomes] == ["cached", "failed"]
    with pytest.raises(RuntimeError, match="boom"):
        run_experiment(store, "fam", boom, model=model, dataset=dataset, raise_on_failure=True)


def test_incompatible_model_dataset_and_noop_preprocessing(tmp_path):
    store, model, dataset, root = build(tmp_path)
    drop = spec("drop", {"kind": "image_transform", "op": "channel_drop", "params": {"channel": 0}})

    other_spec = TorchModelSpec(
        model_id="toy", source="factory", num_classes=3,
        factory=f"{DUMMY}:identity_classifier", factory_kwargs={"scale": 5.0}, class_names=NAMES,
    )
    with pytest.raises(ExperimentError, match="model spec"):
        run_experiment(store, "fam", drop, model=TorchImageClassifier(other_spec, device="cpu"), dataset=dataset)

    resized = ImageClassificationDataset.from_folder(
        root, class_names=NAMES, preprocess=PreprocessConfig(resize=(4, 4))
    )
    with pytest.raises(ExperimentError, match="preprocessing differs"):
        run_experiment(store, "fam", drop, model=model, dataset=resized)

    reordered = ImageClassificationDataset.from_folder(
        root, class_names=["GREEN", "RED", "BLUE"], preprocess=PreprocessConfig()
    )
    with pytest.raises(ExperimentError, match="class order"):
        run_experiment(store, "fam", drop, model=model, dataset=reordered)

    noop = spec("noop", {"kind": "preprocessing", "changes": {"color_mode": "RGB"}})
    with pytest.raises(ExperimentConfigError, match="nothing is intervened"):
        run_experiment(store, "fam", noop, model=model, dataset=dataset)
    unknown = spec("unknown", {"kind": "preprocessing", "changes": {"zoom": 2}})
    with pytest.raises(ExperimentConfigError, match="unknown preprocess"):
        run_experiment(store, "fam", unknown, model=model, dataset=dataset)
    gone = spec("gone", {"kind": "image_transform", "op": "channel_drop", "params": {"channel": 0}},
                population={"sample_ids": ["RED/999.png"]})
    with pytest.raises(ExperimentConfigError, match="not in the baseline"):
        run_experiment(store, "fam", gone, model=model, dataset=dataset)
