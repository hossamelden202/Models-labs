import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("scipy")
pytest.importorskip("sklearn")
pytest.importorskip("pyarrow")

from modellab.analysis import (  # noqa: E402
    FailureConfig,
    analyze_failures,
    load_audit,
    load_evaluation,
    load_failure_analysis,
)
from modellab.analysis.records import MODEL_OUTPUT  # noqa: E402
from modellab.analysis.stats import benjamini_hochberg, holm  # noqa: E402
from modellab.cli.main import main  # noqa: E402
from modellab.core.errors import EvaluationError  # noqa: E402
from modellab.core.results import ArtifactStore  # noqa: E402

FIXTURES = Path(__file__).resolve().parents[1] / "analysis_fixtures.py"
_spec = importlib.util.spec_from_file_location("analysis_fixtures_for_tests", FIXTURES)
fx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fx)


def ref_bh(p):
    m = len(p)
    order = sorted(range(m), key=lambda i: p[i])
    q = [0.0] * m
    running = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        running = min(running, p[i] * m / rank)
        q[i] = running
    return q


def test_multiple_testing_helpers():
    p = [0.01, 0.04, 0.03, 0.005]
    assert benjamini_hochberg(p).tolist() == pytest.approx([0.02, 0.04, 0.04, 0.02])
    assert holm(p).tolist() == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert benjamini_hochberg([]).tolist() == []


def test_known_confusion_patterns_and_hand_checked_metrics(tmp_path):
    pred = fx.make_predictions()
    audit = fx.make_audit_records(pred)
    before = pred.drop(columns="class_probabilities").copy()
    store = ArtifactStore(tmp_path / "art")
    analysis = analyze_failures(
        pred, fx.NAMES, FailureConfig(clusters=True, n_clusters=3), audit_records=audit,
        store=store, analysis_id="a1", evaluation_id="ev1",
    )
    pd.testing.assert_frame_equal(pred.drop(columns="class_probabilities"), before)

    s, report = analysis.samples, analysis.report
    assert len(s) == 240 and report["overall"]["accuracy"] == pytest.approx(0.8)
    assert s.failure_type.value_counts().to_dict() == {
        "correct": 172, "error": 30, "low_confidence_correct": 20,
        "ambiguous_error": 10, "high_confidence_error": 8,
    }
    assert report["failure_categories"]["flags"] == {
        "errors": 48, "high_confidence_errors": 8, "low_confidence_correct": 20, "ambiguous": 30,
    }
    assert s.false_negative_for.value_counts().to_dict() == {"NEUTRAL": 30, "BLOOD": 10, "GORE": 8}
    assert s.false_positive_for.value_counts().to_dict() == {"GORE": 30, "NEUTRAL": 10, "BLOOD": 8}
    assert s.margin.notna().all() and s.entropy.notna().all()

    confusion = report["confusion"]
    assert confusion["matrix"] == [[72, 8, 0], [0, 70, 10], [30, 0, 50]]
    per = {c["class"]: c for c in confusion["per_class"]}
    assert per["GORE"]["recall"] == pytest.approx(0.9) and per["GORE"]["precision"] == pytest.approx(72 / 102)
    assert per["BLOOD"]["recall"] == pytest.approx(0.875) and per["BLOOD"]["precision"] == pytest.approx(70 / 78)
    assert per["NEUTRAL"]["recall"] == pytest.approx(0.625) and per["NEUTRAL"]["precision"] == pytest.approx(50 / 60)
    assert per["NEUTRAL"]["false_negatives"] == 30 and per["GORE"]["false_positives"] == 30
    top = [(t["true_class"], t["predicted_class"], t["count"]) for t in confusion["top_confusions"]]
    assert top == [("NEUTRAL", "GORE", 30), ("BLOOD", "NEUTRAL", 10), ("GORE", "BLOOD", 8)]
    flagged = [c["class"] for c in confusion["class_failures"] if c["evidence"] == "statistically_supported"]
    assert flagged == ["NEUTRAL"]

    sl = analysis.slices.set_index("slice_id")
    row = sl.loc["mode=L"]
    assert (row.support, row.errors) == (40, 30)
    assert row.error_rate == pytest.approx(0.75) and row.accuracy == pytest.approx(0.25)
    assert row.macro_precision == pytest.approx(1.0) and row.macro_recall == pytest.approx(0.25)
    assert row.macro_f1 == pytest.approx(0.4) and row.mean_confidence == pytest.approx(0.7375)
    assert row.complement_error_rate == pytest.approx(0.09) and row.risk_difference == pytest.approx(0.66)
    assert row.risk_ratio == pytest.approx(0.75 / 0.09, rel=1e-4)
    h = 2 * math.asin(math.sqrt(0.75)) - 2 * math.asin(math.sqrt(0.09))
    assert row.cohens_h == pytest.approx(h, abs=1e-5)
    assert row.p_value < 1e-10 and row.q_value < 0.05
    assert row.evidence == "statistically_supported" and not row.uses_model_output
    assert row.slice_type == "single_feature"
    assert "mode=L & true_class=NEUTRAL" in json.loads(row.equivalent_slices)
    assert sl.loc["true_class=NEUTRAL"].errors == 30 and sl.loc["true_class=NEUTRAL"].support == 80
    assert sl.loc["predicted_class=GORE"].support == 102 and sl.loc["predicted_class=GORE"].errors == 30
    assert sl.loc["confusion:NEUTRAL->GORE"].support == 30
    assert sl.loc["high_confidence_errors"].support == 8
    assert sl.loc["mode=RGB"].evidence == "supported_lower_error"

    tested = analysis.slices[analysis.slices.tested]
    assert len(tested) > 20 and len(tested) == report["slices"]["info"]["num_tested"]
    assert np.allclose(tested.q_value.to_numpy(), ref_bh(tested.p_value.tolist()))
    assert (tested.q_value >= tested.p_value - 1e-15).all()
    target_columns = {"correct", "is_error", "failure_type", "true_label", "predicted_label"}
    for conditions, flag in zip(tested.conditions, tested.uses_model_output, strict=True):
        features = {f for f, _ in json.loads(conditions)}
        assert not features & target_columns
        assert not {"true_class", "predicted_class"} <= features
        assert flag == bool(features & MODEL_OUTPUT)
    masks = {tuple(analysis.slice_mask(sid)) for sid in tested.slice_id}
    assert len(masks) == len(tested)
    data_only = tested[~tested.uses_model_output].sort_values("rank")
    assert data_only.iloc[0].slice_id == "mode=L"
    skipped = report["slices"]["skipped_features"]
    assert {"width", "height", "contrast", "status"} <= set(skipped)

    clusters = report["clusters"]["clusters"]
    assert [c["size"] for c in clusters] == [30, 10, 8]
    assert (clusters[0]["dominant_true_class"], clusters[0]["dominant_predicted_class"]) == ("NEUTRAL", "GORE")
    error_ids = set(s.sample_id[s.is_error])
    assert all(set(c["representative_sample_ids"]) <= error_ids for c in clusters)

    files = sorted(p.name for p in analysis.directory.iterdir())
    assert files == [
        "clusters.json", "confusion.json", "metadata.json", "report.json",
        "samples.parquet", "slices.parquet", "summary.txt",
    ]
    loaded = load_failure_analysis(store, "a1")
    expected_ids = [f"NEUTRAL/{k:03d}.png" for k in range(40)]
    assert loaded.slice_sample_ids("mode=L") == expected_ids == analysis.slice_sample_ids("mode=L")
    assert "not causes" in (analysis.directory / "summary.txt").read_text()
    with pytest.raises(KeyError):
        loaded.slice_sample_ids("nope=1")


def test_min_support_ordering_determinism_depth_and_exclusions():
    pred = fx.make_predictions()
    audit = fx.make_audit_records(pred)
    strict = analyze_failures(pred, fx.NAMES, FailureConfig(min_support=50), audit_records=audit)
    tested = strict.slices[strict.slices.tested]
    assert (tested.support >= 50).all() and "mode=L" not in set(strict.slices.slice_id)
    assert strict.report["slices"]["info"]["num_pruned_min_support"] > 0

    first = analyze_failures(pred, fx.NAMES, FailureConfig(), audit_records=audit)
    shuffled = pred.sample(frac=1.0, random_state=3).reset_index(drop=True)
    second = analyze_failures(shuffled, fx.NAMES, FailureConfig(), audit_records=audit.sample(frac=1.0, random_state=5))
    pd.testing.assert_frame_equal(first.slices, second.slices)
    assert json.dumps(first.report, sort_keys=True) == json.dumps(second.report, sort_keys=True)

    deep = analyze_failures(pred, fx.NAMES, FailureConfig(max_depth=3, beam_width=5), audit_records=audit)
    assert deep.slices.n_conditions.max() <= 3 and not deep.slices.slice_id.duplicated().any()
    assert (deep.slices[deep.slices.tested].support >= 20).all()

    no_model = analyze_failures(
        pred, fx.NAMES, FailureConfig(exclude_model_output_features=True), audit_records=audit
    )
    assert not no_model.slices[no_model.slices.tested].uses_model_output.any()
    assert "confidence" in no_model.report["slices"]["skipped_features"]

    by_q = analyze_failures(pred, fx.NAMES, FailureConfig(rank_by="q_value"), audit_records=audit)
    q = by_q.slices[by_q.slices.tested].sort_values("rank").q_value.to_numpy()
    tiers = by_q.slices[by_q.slices.tested].sort_values("rank").evidence.tolist()
    assert tiers[0] == "statistically_supported" and len(q) > 0


def test_degenerate_and_missing_inputs():
    pred = fx.make_predictions()
    audit = fx.make_audit_records(pred)

    empty = analyze_failures(pred.iloc[0:0], fx.NAMES)
    assert len(empty.samples) == 0 and len(empty.slices) == 0
    assert "no samples" in empty.report["warnings"]

    single = pd.DataFrame(
        {
            "sample_id": [f"a/{i}.png" for i in range(30)], "true_label": 0, "predicted_label": 0,
            "confidence": 0.9, "correct": True, "class_probabilities": [np.array([1.0])] * 30,
        }
    )
    one = analyze_failures(single, ["ONLY"])
    assert one.report["overall"]["accuracy"] == 1.0 and len(one.slices) == 0

    right = analyze_failures(pred.assign(predicted_label=pred.true_label.to_numpy()), fx.NAMES, audit_records=audit)
    assert right.report["overall"]["n_errors"] == 0
    assert not (right.slices.evidence == "statistically_supported").any()

    wrong = analyze_failures(pred.assign(predicted_label=(pred.true_label.to_numpy() + 1) % 3), fx.NAMES, audit_records=audit)
    assert wrong.report["overall"]["accuracy"] == 0.0
    assert not (wrong.slices.evidence == "statistically_supported").any()

    partial = analyze_failures(pred, fx.NAMES, audit_records=audit.iloc[:120])
    assert partial.report["inputs"]["join_coverage"] == pytest.approx(0.5)
    assert "mode=<missing>" in set(partial.slices.slice_id)
    assert any("50.0%" in w for w in partial.report["warnings"])

    bare = analyze_failures(pred.drop(columns="class_probabilities"), fx.NAMES, FailureConfig(clusters=True))
    assert bare.samples.margin.isna().all() and bare.samples.entropy.isna().all()
    assert bare.report["clusters"]["available"] is False
    assert not bare.samples.is_ambiguous.any()
    assert any("probabilities unavailable" in w for w in bare.report["warnings"])

    with pytest.raises(EvaluationError, match="outside"):
        analyze_failures(pred.assign(predicted_label=7), fx.NAMES)
    with pytest.raises(EvaluationError, match="duplicate"):
        analyze_failures(pd.concat([pred.iloc[:5], pred.iloc[:5]]), fx.NAMES)
    with pytest.raises(EvaluationError, match="missing columns"):
        analyze_failures(pred.drop(columns="confidence"), fx.NAMES)
    with pytest.raises(EvaluationError, match="confidence"):
        analyze_failures(pred.assign(confidence=1.7), fx.NAMES)


def test_cli_analyze_reads_stored_artifacts_without_touching_them(tmp_path):
    pred = fx.make_predictions()
    audit = fx.make_audit_records(pred)
    root = tmp_path / "art"
    root.mkdir()
    ev_dir = fx.write_evaluation(root, "ev1", pred, fx.NAMES)
    fx.write_audit(root, "au1", audit)
    digest = lambda p: hashlib.sha256((p / "predictions.parquet").read_bytes()).hexdigest()  # noqa: E731
    before = digest(ev_dir)

    code = main(["analyze", "--evaluation", "ev1", "--audit", "au1", "--output", str(root), "--analysis-id", "cli"])
    assert code == 0 and digest(ev_dir) == before

    stored = ArtifactStore(root)
    predictions, names, _ = load_evaluation(root, "ev1")
    records, report = load_audit(root, "au1")
    api = analyze_failures(
        predictions, names, FailureConfig(), audit_records=records, audit_report=report,
        evaluation_id="ev1", audit_id="au1",
    )
    cli = load_failure_analysis(stored, "cli")
    assert json.dumps(cli.report, sort_keys=True) == json.dumps(api.report, sort_keys=True)
    assert cli.report["inputs"]["evaluation_id"] == "ev1"
    assert main(["analyze", "--evaluation", "missing", "--output", str(root)]) == 1
