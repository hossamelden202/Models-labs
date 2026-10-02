import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("scipy")
pytest.importorskip("sklearn")
pytest.importorskip("pyarrow")

from modellab.analysis import FailureConfig, analyze_failures, load_audit, load_evaluation  # noqa: E402
from modellab.core.errors import ArtifactError  # noqa: E402
from modellab.core.results import ArtifactStore  # noqa: E402
from modellab.repair import RepairConfig, RepairError, comparable, run_repair  # noqa: E402

FIXTURES = Path(__file__).resolve().parents[1] / "analysis_fixtures.py"
_spec = importlib.util.spec_from_file_location("analysis_fixtures_for_repair", FIXTURES)
fx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fx)


def build(tmp_path):
    store = ArtifactStore(tmp_path / "art")
    pred = fx.make_predictions()
    fx.write_evaluation(store.root, "base", pred, fx.NAMES)
    fx.write_audit(store.root, "au1", fx.make_audit_records(pred))
    predictions, names, _ = load_evaluation(store.root, "base")
    records, report = load_audit(store.root, "au1")
    analyze_failures(predictions, names, FailureConfig(), audit_records=records, audit_report=report,
                     store=store, analysis_id="an1", evaluation_id="base", audit_id="au1")
    return store


def write_investigation(store, investigation_id="inv_manual", supported=True):
    hyps = [
        {"hypothesis_id": "hyp_bias", "failure_id": "fail_class", "category": "class_bias",
         "params": {"class": "NEUTRAL", "dominant_wrong_class": "GORE"}},
        {"hypothesis_id": "hyp_detail", "failure_id": "fail_slice", "category": "detail_loss", "params": {"feature": "width", "direction": "low"}},
        {"hypothesis_id": "hyp_noise", "failure_id": "fail_noise", "category": "label_noise_or_shortcut", "params": {}},
    ]
    failures = [
        {"failure_id": "fail_class", "kind": "class", "conditions": [], "population_ref": {"kind": "class", "class": "NEUTRAL"}},
        {"failure_id": "fail_slice", "kind": "slice", "conditions": [["mode", "L"]], "population_ref": {"kind": "slice", "slice_id": "mode=L", "analysis_id": "an1"}},
        {"failure_id": "fail_noise", "kind": "confusion", "conditions": [], "population_ref": {"kind": "confusion", "true_class": "GORE", "predicted_class": "BLOOD"}},
    ]
    status = "supported_mechanism" if supported else "inconclusive"
    conclusions = [{"failure_id": h["failure_id"], "status": status, "hypothesis_id": h["hypothesis_id"]} for h in hyps]
    record = {"schema_version": 1, "investigation_id": investigation_id,
              "inputs": {"evaluation_id": "base", "analysis_id": "an1", "audit_id": "au1"},
              "failures": failures, "hypotheses": hyps, "conclusions": conclusions}
    target = store.root / "investigations" / investigation_id
    target.mkdir(parents=True)
    (target / "investigation.json").write_text(json.dumps(record))


def files(directory):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.iterdir())}


def test_candidates_acceptance_rejection_and_regression_detection(tmp_path):
    store = build(tmp_path)
    write_investigation(store)
    base_before = files(store.root / "evaluations" / "base")
    run = run_repair(store, "rep1", "inv_manual", RepairConfig())
    res = run.results
    assert files(store.root / "evaluations" / "base") == base_before and res["baseline"]["unchanged"] is True

    results = {(r["category"], r["grid_index"]): r for r in res["results"]}
    weights = {k[1]: r for k, r in results.items() if k[0] == "class_weighting"}
    thresholds = {k[1]: r for k, r in results.items() if k[0] == "threshold"}
    assert len(weights) == 3 and len(thresholds) == 3
    for index in (0, 1):
        assert weights[index]["decision"] == "rejected" and "target_improvement" in weights[index]["violations"]
        assert weights[index]["global"]["fixed_errors"] == 0
    best = weights[2]
    assert best["decision"] == "accepted" and best["violations"] == [] and best["intervention"]["class_weights"] == {"NEUTRAL": 3.0}
    assert best["target"]["n"] == 80 and best["target"]["accuracy_before"] == pytest.approx(0.625) and best["target"]["accuracy_after"] == pytest.approx(1.0)
    assert best["target"]["difference"] == pytest.approx(0.375) and best["target"]["q"] < 0.05
    assert best["global"]["fixed_errors"] == 30 and best["global"]["new_errors"] == 0
    assert best["global"]["difference"] == pytest.approx(30 / 240)
    recalls = {c["class"]: c for c in best["classes"]}
    assert recalls["NEUTRAL"]["recall_after"] == pytest.approx(1.0) and recalls["GORE"]["drop"] == pytest.approx(0.0)

    for index in (1, 2):
        t = thresholds[index]
        assert t["decision"] == "rejected" and {"new_error_rate", "class_recall"} <= set(t["violations"])
        assert t["global"]["fixed_errors"] == 30 and t["global"]["new_errors"] == 20
        assert t["global"]["new_error_rate"] == pytest.approx(20 / 192) and t["global"]["difference"] > 0
        assert {c["class"]: c for c in t["classes"]}["GORE"]["drop"] == pytest.approx(0.25)
        assert t["target"]["difference"] == pytest.approx(0.375)
    assert thresholds[0]["decision"] == "rejected" and thresholds[0]["global"]["new_errors"] == 20 and thresholds[0]["global"]["fixed_errors"] == 0
    assert res["decision"] == "accepted" and res["accepted_candidate_id"] == best["candidate_id"]
    assert res["summary"]["accepted"] == 1 and res["summary"]["rejected"] == 5

    recs = {r["category"]: r for r in res["recommendations"]}
    assert {"targeted_data_selection", "augmentation_change", "class_balancing", "suspicious_sample_review", "limited_fine_tuning"} <= set(recs)
    assert all(not r["executable"] and "no training interface" in r["reason"] or "not supported" in r["reason"] for r in res["recommendations"])
    assert set(recs["targeted_data_selection"]["sample_ids"]) <= {f"NEUTRAL/{k:03d}.png" for k in range(40)}

    version = json.loads((run.directory / "repair_version.json").read_text())
    assert version["rule"] == {"type": "class_weights", "weights": {"NEUTRAL": 3.0}} and version["base_evaluation_id"] == "base"
    assert version["status"] == "accepted" and version["version_id"] == "rep1-v1"
    text = (run.directory / "repair_report.txt").read_text()
    assert "check new_error_rate: fail" in text and "RECOMMENDATIONS" in text and "not modified" in text

    again = run_repair(store, "rep2", "inv_manual", RepairConfig())
    assert comparable(again.results) == comparable(res)
    assert files(store.root / "evaluations" / "base") == base_before

    relaxed = RepairConfig(criteria={"max_new_error_rate": 1.0, "max_class_recall_drop": 1.0, "max_slice_regression": 1.0})
    loose = run_repair(store, "rep3", "inv_manual", relaxed).results
    accepted = {(r["category"], r["grid_index"]) for r in loose["results"] if r["decision"] == "accepted"}
    assert ("threshold", 1) in accepted and ("class_weighting", 2) in accepted
    assert loose["accepted_candidate_id"] == best["candidate_id"]

    strict = run_repair(store, "rep4", "inv_manual", RepairConfig(criteria={"min_target_improvement": 0.9})).results
    assert strict["decision"] == "rejected" and strict["accepted_candidate_id"] is None
    assert not (store.root / "repairs" / "rep4" / "repair_version.json").exists()

    write_investigation(store, "inv_none", supported=False)
    empty = run_repair(store, "rep5", "inv_none", RepairConfig()).results
    assert empty["decision"] == "no_repair_candidates" and empty["results"] == [] and empty["recommendations"] == []

    with pytest.raises(RepairError, match="not found"):
        run_repair(store, "rep6", "missing", RepairConfig())
    broken = store.root / "investigations" / "inv_broken"
    broken.mkdir(parents=True)
    (broken / "investigation.json").write_text(json.dumps({"schema_version": 1}))
    with pytest.raises(RepairError, match="missing"):
        run_repair(store, "rep7", "inv_broken", RepairConfig())
    with pytest.raises(ArtifactError, match="already exists"):
        run_repair(store, "rep1", "inv_manual", RepairConfig())
    assert files(store.root / "evaluations" / "base") == base_before
