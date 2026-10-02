import hashlib
import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("scipy")
pytest.importorskip("sklearn")
pytest.importorskip("pyarrow")


from modellab.analysis import FailureConfig  # noqa: E402
from modellab.audit import scan_folder  # noqa: E402
from modellab.core.results import ArtifactStore  # noqa: E402
from modellab.evaluation.image_dataset import ImageClassificationDataset  # noqa: E402
from modellab.evaluation.preprocessing import PreprocessConfig  # noqa: E402
from modellab.evaluation.torch_model import TorchImageClassifier, TorchModelSpec  # noqa: E402
from modellab.hypotheses import HypothesisConfig  # noqa: E402
from modellab.investigation import InvestigationConfig, render_report, reproduce_investigation, run_investigation  # noqa: E402
from modellab.repair import RepairConfig, comparable, run_repair  # noqa: E402

TESTS = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("pipeline_fixtures_for_images", TESTS / "pipeline_fixtures.py")
pf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pf)


def setup(tmp_path):
    root = pf.build_dark_dataset(tmp_path / "imgs")
    spec = TorchModelSpec(model_id="dark-toy", source="factory", num_classes=3,
                          factory=f"{TESTS / 'dummy_models.py'}:dark_sensitive_classifier", class_names=pf.NAMES)
    model = TorchImageClassifier(spec, device="cpu")
    dataset = ImageClassificationDataset.from_folder(root, class_names=pf.NAMES, preprocess=PreprocessConfig())
    return ArtifactStore(tmp_path / "art"), model, dataset, root


def snapshot(root):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*")) if p.is_file()}


def weights(model):
    return {k: v.clone() for k, v in model.model.state_dict().items()}


def test_investigation_finds_the_illumination_mechanism_and_a_repair_is_accepted(tmp_path):
    store, model, dataset, root = setup(tmp_path)
    cfg = InvestigationConfig(analysis=FailureConfig(min_support=10, max_depth=1), hypotheses=HypothesisConfig(max_failures=1))

    def scan(audit_cfg):
        return scan_folder(root, audit_cfg, dataset_id="dark-toy")

    data_before, weights_before = snapshot(root), weights(model)
    run = run_investigation(store, "inv1", cfg, model, dataset, scan_fn=scan)
    rec = run.record
    assert rec["baseline"]["num_samples"] == 72 and rec["baseline"]["accuracy"] == pytest.approx(48 / 72)
    failure = rec["failures"][0]
    assert failure["conditions"][0][0] == "brightness" and failure["support"] == 24 and failure["errors"] == 24

    hyps = {h["rule"]: h for h in rec["hypotheses"]}
    assert set(hyps) == {"illumination", "overprediction"}
    ill, over = hyps["illumination"], hyps["overprediction"]
    assert ill["params"]["direction"] == "low" and ill["status"] == "supported" and ill["evidence_strength"] > 0.6
    assert over["status"] == "refuted" and over["params"]["class"] == "BLUE"
    assert over["hypothesis_id"] in ill["competes_with"]
    assert ill["supporting_evidence"][0]["kind"] == "observation" and any(e["kind"] == "experiment" for e in ill["supporting_evidence"])
    assert over["contradicting_evidence"][0]["data"]["label"] == "no_practical_difference"

    experiments = {e["tests"][0]["hypothesis_id"]: e for e in rec["experiments"]}
    fixed = experiments[ill["hypothesis_id"]]["effect"]
    assert fixed["accuracy_difference"] == pytest.approx(1.0) and fixed["global_accuracy_difference"] == pytest.approx(24 / 72)
    assert fixed["primary_p"] == pytest.approx(2 * 0.5**24, rel=1e-6) and fixed["n_affected"] == 24
    assert experiments[over["hypothesis_id"]]["effect"]["accuracy_difference"] == 0.0

    conclusion = rec["conclusions"][0]
    assert conclusion["status"] == "supported_mechanism" and conclusion["category"] == "illumination"
    assert conclusion["hypothesis_id"] == ill["hypothesis_id"] and "not a proven cause" in conclusion["statement"]
    assert conclusion["alternatives"][0]["status"] == "refuted" and all(c["passed"] for c in conclusion["rule_checks"])
    assert {link["hypothesis_id"] for link in rec["links"]} == {ill["hypothesis_id"], over["hypothesis_id"]}
    assert all(link["experiment_id"] and link["failure_id"] == failure["failure_id"] for link in rec["links"])
    text = (run.directory / "evidence_report.txt").read_text()
    assert text == render_report(rec) and "1. OBSERVED RESULTS" in text and "supported_mechanism" in text
    assert snapshot(root) == data_before and all(torch.equal(weights_before[k], v) for k, v in weights(model).items())

    repro = reproduce_investigation(store, "inv1", "inv1_again", model, dataset, scan)
    assert repro["reproduced"] is True and repro["original_fingerprint"] == repro["new_fingerprint"]

    base_files = snapshot(store.root / "evaluations" / rec["inputs"]["evaluation_id"])
    repair = run_repair(store, "rep1", "inv1", RepairConfig(), model, dataset)
    res = repair.results
    assert res["decision"] == "accepted" and res["baseline"]["unchanged"] is True
    by_factor = {r["intervention"]["params"]["factor"]: r for r in res["results"]}
    assert set(by_factor) == {1.3, 1.5, 2.0} and all(r["category"] == "slice_input_transform" for r in by_factor.values())
    assert by_factor[1.3]["target"]["difference"] == pytest.approx(0.5) and by_factor[1.3]["decision"] == "accepted"
    for factor in (1.5, 2.0):
        assert by_factor[factor]["target"]["difference"] == pytest.approx(1.0)
        assert by_factor[factor]["global"]["difference"] == pytest.approx(24 / 72)
        assert by_factor[factor]["global"]["new_errors"] == 0 and by_factor[factor]["violations"] == []
    assert res["accepted_candidate_id"] == by_factor[1.5]["candidate_id"]
    assert all(c["passed"] for c in by_factor[1.5]["checks"]) and by_factor[1.5]["classes"] and by_factor[1.5]["slices"]
    assert {r["category"] for r in res["recommendations"]} == set() or all(not r["executable"] for r in res["recommendations"])
    version = (repair.directory / "repair_version.json").read_text()
    assert "slice_conditioned_input_transform" in version and "not modified" in version
    assert sorted(p.name for p in repair.directory.iterdir()) == [
        "accepted_repair.json", "metadata.json", "repair_candidates.json", "repair_report.txt", "repair_results.json", "repair_version.json",
    ]
    assert snapshot(store.root / "evaluations" / rec["inputs"]["evaluation_id"]) == base_files
    assert snapshot(root) == data_before and all(torch.equal(weights_before[k], v) for k, v in weights(model).items())
    again = run_repair(store, "rep2", "inv1", RepairConfig(), model, dataset)
    assert comparable(again.results) == comparable(res)

    strict = RepairConfig(criteria={"min_target_improvement": 0.9})
    rejected = run_repair(store, "rep3", "inv1", strict, model, dataset).results
    assert rejected["decision"] == "accepted" and rejected["accepted_candidate_id"] == by_factor[1.5]["candidate_id"]
    harsh = run_repair(store, "rep4", "inv1", RepairConfig(factors_low=(1.3,), criteria={"min_target_improvement": 0.9}), model, dataset).results
    assert harsh["decision"] == "rejected" and harsh["results"][0]["violations"] == ["target_improvement"]
    assert harsh["summary"]["accepted"] == 0 and "rejected" in (store.root / "repairs/rep4/repair_report.txt").read_text()
    no_images = run_repair(store, "rep5", "inv1", RepairConfig()).results
    assert no_images["decision"] == "no_repair_candidates" and all(r["status"] == "not_evaluated" for r in no_images["results"])
