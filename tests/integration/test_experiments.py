import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("sklearn")
pytest.importorskip("pyarrow")

from modellab.analysis import FailureConfig, analyze_failures  # noqa: E402
from modellab.analysis.stats import benjamini_hochberg, holm  # noqa: E402
from modellab.core.results import ArtifactStore  # noqa: E402
from modellab.experiments import (  # noqa: E402
    ExperimentConfigError,
    ExperimentError,
    create_baseline,
    expand_matrix,
    experiment_id,
    load_baseline,
    parse_spec,
    run_experiment,
    run_family,
)
from modellab.experiments.stats import (  # noqa: E402
    bootstrap_mean_ci,
    mcnemar_exact,
    paired_summary,
    sign_flip_p,
)

FIXTURES = Path(__file__).resolve().parents[1] / "analysis_fixtures.py"
_spec = importlib.util.spec_from_file_location("analysis_fixtures_for_experiments", FIXTURES)
fx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fx)


def ref_bh(p):
    m = len(p)
    order = sorted(range(m), key=lambda i: p[i])
    q, running = [0.0] * m, 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        running = min(running, p[i] * m / rank)
        q[i] = running
    return q


def make_store(tmp_path, name):
    store = ArtifactStore(tmp_path / name)
    fx.write_evaluation(store.root, "base", fx.make_predictions(), fx.NAMES)
    return store


def hashes(directory):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.iterdir())}


def spec(name, intervention, expected=None, **kwargs):
    return parse_spec(
        {
            "name": name,
            "hypothesis": {"claim": f"{name} changes accuracy", "expected_direction": expected},
            "intervention": intervention,
            **kwargs,
        }
    )


WEIGHTS = {"kind": "inference", "class_weights": {"NEUTRAL": 3.0}}
TEMP = {"kind": "inference", "temperature": 2.0}
THRESHOLD = {"kind": "inference", "confidence_threshold": 0.6, "fallback_class": "NEUTRAL"}
MODE_L_IDS = [f"NEUTRAL/{k:03d}.png" for k in range(40)]


def test_statistics_helpers_match_known_values():
    assert mcnemar_exact(0, 10) == pytest.approx(2 * 0.5**10)
    assert mcnemar_exact(5, 5) == pytest.approx(1.0)
    assert mcnemar_exact(0, 0) == 1.0
    assert benjamini_hochberg([0.01, 0.04, 0.03, 0.005]).tolist() == pytest.approx([0.02, 0.04, 0.04, 0.02])
    assert holm([0.01, 0.04, 0.03, 0.005]).tolist() == pytest.approx([0.03, 0.06, 0.06, 0.02])

    d = np.array([1.0] * 10)
    assert sign_flip_p(d, 5000, 1) < 0.01
    assert sign_flip_p(np.zeros(20), 500, 1) == 1.0
    low, high = bootstrap_mean_ci(np.array([1.0] * 30 + [0.0] * 210), 1000, 3)
    assert low < 0.125 < high and (low, high) == bootstrap_mean_ci(np.array([1.0] * 30 + [0.0] * 210), 1000, 3)
    assert bootstrap_mean_ci([], 100, 1) == (None, None)

    base = np.array([True] * 12 + [False] * 8)
    better = np.array([True] * 12 + [True] * 6 + [False] * 2)
    out = paired_summary(base, [better], 500, 2000, 5)
    assert (out["discordant_improved"], out["discordant_worsened"]) == (6, 0)
    diff = better.astype(float) - base.astype(float)
    assert out["accuracy_difference"] == pytest.approx(0.3)
    assert out["cohens_dz"] == pytest.approx(diff.mean() / diff.std(ddof=1))
    assert out["mcnemar_p"] == pytest.approx(2 * 0.5**6)
    assert out["odds_ratio"] == pytest.approx(6.5 / 0.5)
    assert out["ci_low"] <= 0.3 <= out["ci_high"] and out["primary_test"] == "mcnemar_exact"
    reps = paired_summary(base, [better, base], 500, 2000, 5)
    assert reps["primary_test"] == "sign_flip_permutation" and reps["mcnemar_p"] is None


def test_inference_family_known_outcomes_resume_and_reproducibility(tmp_path):
    store = make_store(tmp_path, "art")
    baseline = create_baseline(store, "fam", "base")
    base_dir = store.root / "evaluations" / "base"
    before = hashes(base_dir)

    weights = spec("neutral_weight", WEIGHTS, "improve")
    temp = spec("temperature", TEMP, "no_change")
    thr = spec("threshold", THRESHOLD, "degrade")
    run = run_family(store, "fam", [weights, temp, thr])
    assert [o.status for o in run.outcomes] == ["completed"] * 3
    res = {o.result["spec"]["name"]: o.result for o in run.outcomes}
    assert hashes(base_dir) == before

    w = res["neutral_weight"]
    st = w["statistics"]["affected"]
    assert (st["discordant_improved"], st["discordant_worsened"]) == (30, 0)
    assert st["mcnemar_p"] == pytest.approx(2 * 0.5**30, rel=1e-6)
    assert st["permutation_p"] < 0.01 and st["accuracy_difference"] == pytest.approx(0.125)
    d = np.array([1.0] * 30 + [0.0] * 210)
    assert st["cohens_dz"] == pytest.approx(d.mean() / d.std(ddof=1))
    assert st["ci_low"] > 0 and st["ci_low"] < 0.125 < st["ci_high"]
    acc = w["comparison"]["global"]["differences"]["accuracy"]
    assert (acc["baseline"], acc["intervention"]) == (pytest.approx(0.8), pytest.approx(0.925))
    assert w["comparison"]["global"]["baseline"]["metrics"]["balanced_accuracy"] == pytest.approx(0.8)
    assert w["verdict"]["label"] == "improved" and w["verdict"]["matches_expected"] is True
    assert w["affected_samples"]["num_improved"] == 30 and w["affected_samples"]["num_worsened"] == 0
    assert set(w["affected_samples"]["improved_ids"]) <= {f"NEUTRAL/{k:03d}.png" for k in range(30)}
    assert w["baseline"]["fingerprint"] == baseline.fingerprint
    assert w["control"]["seed"] == 42 and w["environment"]["modellab"]

    t = res["temperature"]
    assert t["statistics"]["affected"]["mcnemar_p"] == 1.0
    assert t["comparison"]["global"]["differences"]["accuracy"]["difference"] == 0.0
    assert t["comparison"]["global"]["differences"]["mean_confidence"]["difference"] < 0
    assert t["comparison"]["global"]["differences"]["ece"]["difference"] != 0
    assert t["verdict"]["label"] == "no_practical_difference" and t["verdict"]["matches_expected"] is True

    c = res["threshold"]
    cst = c["statistics"]["affected"]
    assert (cst["discordant_worsened"], cst["discordant_improved"]) == (20, 0)
    assert cst["mcnemar_p"] == pytest.approx(2 * 0.5**20, rel=1e-6)
    assert c["comparison"]["global"]["differences"]["accuracy"]["intervention"] == pytest.approx(172 / 240)
    assert c["verdict"]["label"] == "degraded" and c["verdict"]["matches_expected"] is True

    rows = {r["name"]: r for r in run.report["experiments"]}
    names = ["neutral_weight", "temperature", "threshold"]
    ps = [rows[n]["primary_p"] for n in names]
    assert [rows[n]["p_benjamini_hochberg"] for n in names] == pytest.approx(ref_bh(ps))
    assert rows["neutral_weight"]["significant_after_correction"] and not rows["temperature"]["significant_after_correction"]
    assert rows["threshold"]["family_label"] == "degraded"
    assert (run.directory / "family_summary.txt").is_file()

    again = run_family(store, "fam", [weights, temp, thr])
    assert [o.status for o in again.outcomes] == ["cached"] * 3
    victim = run.outcomes[2].experiment_id
    shutil.rmtree(store.root / "experiments" / "fam" / victim)
    resumed = run_family(store, "fam", [weights, temp, thr])
    assert [o.status for o in resumed.outcomes] == ["cached", "cached", "completed"]
    assert json.dumps(resumed.report, sort_keys=True) == json.dumps(run.report, sort_keys=True)

    assert experiment_id(weights) == run.outcomes[0].experiment_id
    assert experiment_id(spec("neutral_weight", WEIGHTS, "improve", seed=43)) != run.outcomes[0].experiment_id
    assert len({o.experiment_id for o in run.outcomes}) == 3

    other = make_store(tmp_path, "art_b")
    create_baseline(other, "fam", "base")
    run_family(other, "fam", [weights, temp, thr])
    for o in run.outcomes:
        a = (store.root / "experiments" / "fam" / o.experiment_id / "result.json").read_bytes()
        b = (other.root / "experiments" / "fam" / o.experiment_id / "result.json").read_bytes()
        assert a == b
    assert (store.root / "experiments/fam/family_report.json").read_bytes() == (
        other.root / "experiments/fam/family_report.json"
    ).read_bytes()


def test_slice_populations_baseline_immutability_and_incompatible_runs(tmp_path):
    store = make_store(tmp_path, "art")
    pred = fx.make_predictions()
    create_baseline(store, "fam", "base")
    by_samples = spec("mode_l_samples", WEIGHTS, population={"sample_ids": MODE_L_IDS})
    out = run_experiment(store, "fam", by_samples)
    r = out.result
    assert r["population"]["num_affected"] == 40
    assert r["statistics"]["affected"]["discordant_improved"] == 30
    aff = r["comparison"]["affected"]["differences"]["accuracy"]
    glob = r["comparison"]["global"]["differences"]["accuracy"]
    assert aff["difference"] == pytest.approx(0.75) and glob["difference"] == pytest.approx(0.125)
    assert glob["difference"] == pytest.approx(aff["difference"] * 40 / 240)
    assert r["statistics"]["global"]["n"] == 240 and r["statistics"]["affected"]["n"] == 40
    paired = __import__("pandas").read_parquet(out.directory / "paired.parquet")
    assert paired.affected.sum() == 40 and paired.changed.sum() == 30
    assert (paired[~paired.affected].baseline_pred == paired[~paired.affected].intervention_pred).all()

    analysis = analyze_failures(
        pred, fx.NAMES, FailureConfig(), audit_records=fx.make_audit_records(pred),
        store=store, analysis_id="an1", evaluation_id="base",
    )
    assert analysis.slice_sample_ids("mode=L") == MODE_L_IDS
    by_slice = spec("mode_l_slice", WEIGHTS, population={"analysis_id": "an1", "slice_ids": ["mode=L"]})
    sliced = run_experiment(store, "fam", by_slice).result
    assert sliced["statistics"]["affected"] == r["statistics"]["affected"]
    assert sliced["statistics"]["slices"]["mode=L"]["primary_p_adjusted"] <= 1.0
    assert sliced["population"]["kind"] == "slices"

    analyze_failures(pred, fx.NAMES, FailureConfig(), store=store, analysis_id="an2", evaluation_id="zzz")
    wrong = spec("wrong_analysis", WEIGHTS, population={"analysis_id": "an2", "slice_ids": ["true_class=NEUTRAL"]})
    with pytest.raises(ExperimentConfigError, match="different evaluation"):
        run_experiment(store, "fam", wrong)

    fx.write_evaluation(store.root, "other", pred.assign(confidence=0.5), fx.NAMES)
    with pytest.raises(ExperimentError, match="different baseline"):
        create_baseline(store, "fam", "other")

    tampered = make_store(tmp_path, "art_t")
    create_baseline(tampered, "fam", "base")
    target = tampered.root / "evaluations" / "base" / "predictions.parquet"
    target.write_bytes(target.read_bytes() + b"x")
    with pytest.raises(ExperimentError, match="modified"):
        load_baseline(tampered, "fam")
    with pytest.raises(ExperimentError, match="modified"):
        run_experiment(tampered, "fam", by_samples)


def test_invalid_configs_fail_before_anything_runs(tmp_path):
    base = {"name": "x", "hypothesis": {"claim": "c"}}
    bad = [
        ({"kind": "image_transform", "op": "nope"}, "unknown image op"),
        ({"kind": "image_transform", "op": "brightness", "params": {"factor": -1}}, "factor"),
        ({"kind": "image_transform", "op": "brightness", "params": {"factor": 1, "extra": 1}}, "extra"),
        ({"kind": "image_transform", "op": "hflip", "params": {"semantically_valid": False}}, "semantically valid"),
        ({"kind": "image_transform", "op": "channel_swap", "params": {"order": [0, 0, 1]}}, "permutation"),
        ({"kind": "inference", "temperature": 2.0, "class_weights": {"A": 2}}, "exactly one"),
        ({"kind": "inference", "confidence_threshold": 0.5}, "fallback_class"),
        ({"kind": "inference", "temperature": 2.0, "fallback_class": "A"}, "fallback_class"),
        ({"kind": "inference", "class_weights": {"A": -1}}, "positive"),
        ({"kind": "data", "op": "reweighting"}, "training interface"),
        ({"kind": "preprocessing", "changes": {}}, "at least 1"),
        ({"kind": "unknown"}, "kind"),
    ]
    for intervention, message in bad:
        with pytest.raises(ExperimentConfigError, match=message):
            parse_spec({**base, "intervention": intervention})
    with pytest.raises(ExperimentConfigError, match="repetitions"):
        parse_spec({**base, "intervention": TEMP, "repetitions": 3})
    with pytest.raises(ExperimentConfigError, match="not both"):
        parse_spec({**base, "intervention": TEMP, "population": {"slice_ids": ["a"], "sample_ids": ["b"], "analysis_id": "x"}})
    with pytest.raises(ExperimentConfigError, match="analysis_id"):
        parse_spec({**base, "intervention": TEMP, "population": {"slice_ids": ["a"]}})
    with pytest.raises(ExperimentConfigError):
        parse_spec({**base, "intervention": TEMP, "seed": -1})

    store = make_store(tmp_path, "art")
    create_baseline(store, "fam", "base")
    with pytest.raises(ExperimentConfigError, match="model and the dataset"):
        run_experiment(store, "fam", spec("img", {"kind": "image_transform", "op": "brightness", "params": {"factor": 1.5}}))
    with pytest.raises(ExperimentConfigError, match="unknown class"):
        run_experiment(store, "fam", spec("cls", {"kind": "inference", "class_weights": {"MISSING": 2.0}}))
    with pytest.raises(ExperimentConfigError, match="not in the baseline"):
        run_experiment(store, "fam", spec("ids", TEMP, population={"sample_ids": ["nope.png"]}))
    with pytest.raises(ExperimentError, match="no baseline"):
        run_experiment(store, "missing_family", spec("t", TEMP))

    good = spec("good", WEIGHTS)
    broken = spec("broken", {"kind": "inference", "class_weights": {"MISSING": 2.0}})
    with pytest.raises(ExperimentConfigError):
        run_family(store, "fam", [good, broken])
    assert not (store.root / "experiments" / "fam" / experiment_id(good)).exists()

    matrix = expand_matrix(
        {
            "name": "sweep", "hypothesis": {"claim": "weights help"},
            "intervention": {"kind": "inference", "class_weights": {"NEUTRAL": 1.0}},
            "grid": {"class_weights": [{"NEUTRAL": 2.0}, {"NEUTRAL": 3.0}]},
        }
    )
    assert [m.name for m in matrix] == ["sweep[class_weights={'NEUTRAL': 2.0}]", "sweep[class_weights={'NEUTRAL': 3.0}]"]
    family = run_family(store, "fam", matrix)
    assert family.report["num_completed"] == 2
    with pytest.raises(ExperimentConfigError):
        expand_matrix({"name": "x", "intervention": TEMP, "hypothesis": {"claim": "c"}, "grid": {}})
