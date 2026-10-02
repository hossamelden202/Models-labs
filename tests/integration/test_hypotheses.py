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
from modellab.experiments import create_baseline, parse_spec, run_family  # noqa: E402
from modellab.experiments.spec import canonical_json  # noqa: E402
from modellab.hypotheses import (  # noqa: E402
    DISCLAIMER,
    Evidence,
    ExperimentTest,
    Hypothesis,
    HypothesisConfig,
    build_plan,
    hypotheses_document,
    update_hypotheses,
)
from modellab.investigation import (  # noqa: E402
    InvestigationConfig,
    conclude,
    render_report,
    reproduce_investigation,
    run_investigation,
)

FIXTURES = Path(__file__).resolve().parents[1] / "analysis_fixtures.py"
_spec = importlib.util.spec_from_file_location("analysis_fixtures_for_hypotheses", FIXTURES)
fx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fx)


def build(tmp_path, name="art"):
    store = ArtifactStore(tmp_path / name)
    pred = fx.make_predictions()
    fx.write_evaluation(store.root, "base", pred, fx.NAMES)
    fx.write_audit(store.root, "au1", fx.make_audit_records(pred))
    predictions, names, meta = load_evaluation(store.root, "base")
    records, report = load_audit(store.root, "au1")
    analysis = analyze_failures(
        predictions, names, FailureConfig(), audit_records=records, audit_report=report,
        store=store, analysis_id="an1", evaluation_id="base", audit_id="au1",
    )
    return store, analysis, meta, report


def by_rule(hypotheses, failure_id):
    return {h.rule: h for h in hypotheses if h.failure_id == failure_id}


def test_generation_competition_selection_and_evidence_updates(tmp_path):
    store, analysis, meta, report = build(tmp_path)
    cfg = HypothesisConfig()
    failures, hyps, plan = build_plan(analysis, meta, report, cfg, allow_images=True)
    first = failures[0]
    assert first.target.conditions == [["mode", "L"]] and first.target.support == 40 and first.target.errors == 30
    assert first.target.slice_id == "mode=L" and first.target.tier == "statistically_supported"
    rules = by_rule(hyps, first.target.failure_id)
    assert {"input_mode", "overprediction"} <= set(rules)
    over, mode = rules["overprediction"], rules["input_mode"]
    assert over.params["class"] == "GORE" and over.params["share"] == 1.0 and over.params["dominant_true_class"] == "NEUTRAL"
    assert over.hypothesis_id in mode.competes_with and mode.hypothesis_id in over.competes_with
    assert mode.required_tests[0].intervention["op"] == "grayscale" and mode.required_tests[0].population == "control"
    assert over.required_tests[0].intervention == {"kind": "inference", "class_weights": {"GORE": 0.5}}
    assert all(h.status in ("untested", "untestable") for h in hyps)
    assert all(h.supporting_evidence and h.supporting_evidence[0].kind == "observation" for h in hyps)

    document = hypotheses_document(failures, hyps)
    json.loads(canonical_json(document))
    assert document["disclaimer"] == DISCLAIMER and document["competing_groups"]
    assert "confirmed" not in json.dumps(document).lower()

    selected = {e["name"]: e for e in plan["selected"]}
    assert len(selected) >= 2 and plan["separated_pairs"] >= 1 and plan["competing_pairs"] >= 1
    gray = next(e for e in plan["selected"] if e["spec"]["intervention"].get("op") == "grayscale")
    control = gray["spec"]["population"]["sample_ids"]
    correct_outside = set(analysis.samples.loc[analysis.samples.correct & ~analysis.samples.sample_id.isin(first.ids), "sample_id"])
    assert 0 < len(control) <= cfg.control_size and set(control) <= correct_outside and not set(control) & set(first.ids)
    weights = next(
        e for e in plan["selected"]
        if e["spec"]["intervention"].get("class_weights", {}).get("GORE") == 0.5
)
    assert weights["spec"]["population"] == {"slice_ids": ["mode=L"], "analysis_id": "an1"}
    for entry in plan["selected"]:
        parse_spec(entry["spec"])
        assert entry["gain"] > 0 and entry["tests"]
    assert canonical_json(plan) == canonical_json(build_plan(analysis, meta, report, cfg, True)[2])
    assert len(build_plan(analysis, meta, report, HypothesisConfig(max_experiments=1), True)[2]["selected"]) == 1

    _, offline, offline_plan = build_plan(analysis, meta, report, cfg, allow_images=False)
    assert all(e["spec"]["intervention"]["kind"] == "inference" for e in offline_plan["selected"])
    assert {h.rule: h.status for h in offline if h.failure_id == first.target.failure_id}["input_mode"] == "untestable"
    assert any("model and a dataset" in u["reason"] for u in offline_plan["untestable_hypotheses"])

    create_baseline(store, "fam", "base")
    specs = [parse_spec(e["spec"]) for e in offline_plan["selected"]]
    run = run_family(store, "fam", specs, analysis=analysis)
    results = [o.result for o in run.outcomes]
    assert all(r["status"] == "completed" for r in results)
    updated = update_hypotheses(offline, offline_plan, results, run.report, cfg)
    state = {h.rule: h for h in updated if h.failure_id == first.target.failure_id}
    assert state["overprediction"].status == "refuted" and state["overprediction"].evidence_strength < 0.3
    assert state["overprediction"].contradicting_evidence[0].data["label"] == "no_practical_difference"
    assert state["input_mode"].status == "untestable"
    assert [s["status"] for s in state["overprediction"].status_history] == ["untested", "refuted"]
    assert all(
        h.status != "supported"
        for h in updated
        if h.rule == "overprediction" and h.failure_id == first.target.failure_id
    )
    again = update_hypotheses(updated, offline_plan, results, run.report, cfg)
    assert [h.model_dump() for h in again] == [h.model_dump() for h in updated]


def hypothesis(strength=0.0, **kwargs):
    base = {"hypothesis_id": "hyp_a", "failure_id": "fail_x", "rule": "r", "category": "r", "mechanism": "m",
            "required_tests": [ExperimentTest(test_id="t1", description="d", intervention={"kind": "inference", "temperature": 2.0},
                                              population="failure", expected_effect="improve")]}
    return Hypothesis(**{**base, **kwargs})


def fake_result(name, label, diff=0.1, p=0.001):
    return {
        "status": "completed", "experiment_id": f"exp_{name}", "spec": {"name": name},
        "statistics": {"affected": {"accuracy_difference": diff, "ci_low": diff - 0.05, "ci_high": diff + 0.05, "primary_p": p}},
        "verdict": {"label": label, "primary_p": p}, "population": {"num_affected": 40},
        "comparison": {"global": {"differences": {"accuracy": {"difference": diff / 4}}}},
    }


def plan_for(*pairs):
    return {"selected": [{"name": name, "tests": [{"hypothesis_id": "hyp_a", "test_id": f"t{i}", "expected_effect": effect}]}
                         for i, (name, effect) in enumerate(pairs)], "untestable_hypotheses": []}


def test_status_transitions_never_reach_confirmed():
    cfg = HypothesisConfig()
    cases = [
        ([("e1", "improve")], [fake_result("e1", "improved")], "supported", 0.5),
        ([("e1", "improve")], [fake_result("e1", "degraded", -0.1)], "weakened", 0.5),
        ([("e1", "improve")], [fake_result("e1", "no_practical_difference", 0.0, 1.0)], "refuted", 0.75),
        ([("e1", "improve")], [fake_result("e1", "inconclusive", 0.0, 0.5)], "inconclusive", 0.0),
        ([("e1", "improve")], [fake_result("e1", "detectable_below_practical_threshold", 0.005)], "inconclusive", 0.25),
        ([("e1", "degrade")], [fake_result("e1", "degraded", -0.1)], "supported", 0.5),
        ([("e1", "improve"), ("e2", "improve")], [fake_result("e1", "improved"), fake_result("e2", "degraded", -0.1)], "conflicting", None),
    ]
    for pairs, results, status, strength in cases:
        h = update_hypotheses([hypothesis()], plan_for(*pairs), results, None, cfg)[0]
        assert h.status == status, (pairs, h.status)
        if strength is not None:
            experiment = [e for g in (h.supporting_evidence, h.contradicting_evidence, h.inconclusive_evidence) for e in g if e.kind == "experiment"][0]
            assert experiment.strength == strength and experiment.data["corrected"] is False
    family = {"experiments": [{"experiment_id": "exp_e1", "family_label": "improved", "primary_p": 1e-6, "p_benjamini_hochberg": 2e-6}]}
    corrected = update_hypotheses([hypothesis()], plan_for(("e1", "improve")), [fake_result("e1", "improved")], family, cfg)[0]
    assert corrected.supporting_evidence[0].strength == 1.0 and corrected.supporting_evidence[0].data["p_adjusted"] == 2e-6
    assert corrected.evidence_strength > update_hypotheses([hypothesis()], plan_for(("e1", "improve")), [fake_result("e1", "improved")], None, cfg)[0].evidence_strength
    assert update_hypotheses([hypothesis()], {"selected": [], "untestable_hypotheses": []}, [], None, cfg)[0].status == "untested"
    assert update_hypotheses([hypothesis(required_tests=[])], {"selected": []}, [], None, cfg)[0].status == "untestable"
    failed = {"status": "failed", "experiment_id": "exp_e1", "spec": {"name": "e1"}, "error": {"type": "Boom"}}
    assert update_hypotheses([hypothesis()], plan_for(("e1", "improve")), [failed], None, cfg)[0].status == "inconclusive"


def evidence(stance, strength, corrected=True, kind="experiment"):
    return Evidence(kind=kind, stance=stance, summary="s", strength=strength, data={"corrected": corrected})


def make(hid, strength, status, supports=(), contradicts=(), competes=()):
    return hypothesis(hypothesis_id=hid, evidence_strength=strength, status=status, competes_with=list(competes),
                      supporting_evidence=list(supports), contradicting_evidence=list(contradicts))


def test_conclusion_rules_refuse_to_force_a_mechanism():
    cfg = InvestigationConfig()
    top = make("hyp_a", 0.7, "supported", [evidence("supports", 1.0)], competes=["hyp_b"])
    other = make("hyp_b", 0.1, "refuted", contradicts=[evidence("contradicts", 0.75)], competes=["hyp_a"])
    won = conclude("fail_x", [top, other], cfg)
    assert won["status"] == "supported_mechanism" and won["hypothesis_id"] == "hyp_a" and "not a proven cause" in won["statement"]
    assert all(c["passed"] for c in won["rule_checks"]) and won["alternatives"][0]["status"] == "refuted"

    rival = make("hyp_b", 0.65, "supported", [evidence("supports", 1.0)], competes=["hyp_a"])
    tie = conclude("fail_x", [top, rival], cfg)
    assert tie["status"] == "inconclusive" and tie["mechanism"] is None
    assert {c["rule"] for c in tie["rule_checks"] if not c["passed"]} >= {"margin_over_runner_up", "no_competing_supported_hypothesis"}
    weak = conclude("fail_x", [make("hyp_a", 0.4, "supported", [evidence("supports", 0.5, False)], competes=["hyp_b"]), other], cfg)
    assert weak["status"] == "inconclusive" and weak["leading_hypothesis_id"] == "hyp_a"
    untested = conclude("fail_x", [make("hyp_a", 0.2, "untested"), make("hyp_b", 0.1, "untestable")], cfg)
    assert untested["status"] == "insufficient_evidence" and untested["hypothesis_id"] is None
    assert conclude("fail_x", [], cfg)["status"] == "insufficient_evidence"
    untried = conclude("fail_x", [top, make("hyp_b", 0.1, "untested", competes=["hyp_a"])], cfg)
    assert untried["status"] == "inconclusive"
    assert any(c["rule"] == "alternative_tested" and not c["passed"] for c in untried["rule_checks"])


def test_inference_only_investigation_records_links_and_reproduces(tmp_path):
    store, analysis, meta, report = build(tmp_path)
    run = run_investigation(store, "inv_syn", InvestigationConfig(), evaluation_id="base", audit_id="au1", analysis_id="an1")
    rec = run.record
    assert rec["inputs"]["evaluation_id"] == "base" and rec["inputs"]["analysis_id"] == "an1"
    assert rec["baseline"]["num_samples"] == 240 and rec["baseline"]["accuracy"] == pytest.approx(0.8)
    failure = rec["failures"][0]
    assert failure["slice_id"] == "mode=L" and failure["hypothesis_ids"] and failure["experiment_ids"]
    statuses = {h["rule"]: h["status"] for h in rec["hypotheses"] if h["failure_id"] == failure["failure_id"]}
    assert statuses["overprediction"] == "refuted" and statuses["input_mode"] == "untestable"
    conclusion = rec["conclusions"][0]
    assert conclusion["status"] == "inconclusive" and conclusion["mechanism"] is None
    assert conclusion["alternatives"] and all(link["failure_id"] and link["hypothesis_id"] and link["experiment_id"] for link in rec["links"])
    mine = next(
        e for e in rec["experiments"]
        if any(
            test["failure_id"] == failure["failure_id"]
            for test in e["tests"]
        )
    )
    assert mine["status"] == "completed" and mine["effect"]["n_affected"] == 40
    text = (run.directory / "evidence_report.txt").read_text()
    for heading in ("1. OBSERVED RESULTS", "2. INFERRED ASSOCIATIONS", "3. CONCLUSIONS", "LIMITATIONS"):
        assert heading in text
    assert text == render_report(rec) and "confirmed" not in text.lower()
    assert sorted(p.name for p in run.directory.iterdir()) == [
        "config.json", "evidence_report.txt", "experiment_plan.json", "hypotheses.json", "investigation.json", "metadata.json",
    ]
    with pytest.raises(ArtifactError, match="already exists"):
        run_investigation(store, "inv_syn", evaluation_id="base", audit_id="au1", analysis_id="an1")
    repro = reproduce_investigation(store, "inv_syn", "inv_syn2")
    assert repro["reproduced"] is True and repro["original_fingerprint"] == repro["new_fingerprint"]
    with pytest.raises(Exception, match="no baseline evaluation"):
        run_investigation(store, "inv_none", InvestigationConfig(), evaluation_id="nope")
