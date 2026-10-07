import json

import pytest

from modellab.ai.analysis import analyze_detection
from modellab.ai.knowledge.ingestion import (
    collect_documents,
    config_changes,
    experiment_document,
    ingest_all,
    tried_changes,
)
from modellab.ai.knowledge.store import HashingEmbedder, KnowledgeStore
from modellab.experiments.training_config import TrainingConfig

PER_CLASS = {
    "0": {"class": "handgun", "class_id": 0, "f1": 0.0125, "false_negatives": 77, "false_positives": 81,
          "precision": 0.0122, "recall": 0.0128, "support": 78, "true_positives": 1},
    "1": {"class": "long_gun", "class_id": 1, "f1": 0.0, "false_negatives": 0, "false_positives": 4,
          "precision": 0.0, "recall": 0.0, "support": 0, "true_positives": 0},
    "2": {"class": "knife", "class_id": 2, "f1": 0.22, "false_negatives": 190, "false_positives": 145,
          "precision": 0.249, "recall": 0.2017, "support": 238, "true_positives": 48},
}
METRICS = {"f1": 0.16, "precision": 0.165, "recall": 0.155, "num_samples": 472}


def result(changes=None, status="completed"):
    cfg = TrainingConfig().model_dump(mode="python")
    for path, value in (changes or {}).items():
        node = cfg
        parts = path.split(".")
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = value
    if status != "completed":
        return {"status": status, "task": "detection", "error": {"type": "RuntimeError", "message": "boom"}}
    return {
        "status": "completed",
        "task": "detection",
        "training": {"config": cfg, "total_epochs": cfg["epochs"], "best_epoch": 1},
        "comparison": {
            "available": True,
            "baseline": {"f1": 0.16, "precision": 0.165, "recall": 0.155, "mean_iou": 0.756},
            "trained": {"f1": 0.27, "precision": 0.24, "recall": 0.31, "mean_iou": 0.915},
            "delta": {"f1": 0.11, "precision": 0.075, "recall": 0.155, "mean_iou": 0.159},
        },
        "evaluation": {"metrics": {"per_class": {"knife": {"class_id": 2, "f1": 0.3, "fn": 100, "fp": 90,
                                                           "precision": 0.3, "recall": 0.4, "tp": 60}}}},
    }


@pytest.fixture
def root(tmp_path):
    def put(rel, data):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data))

    put("experiments/fam/exp_aaa/result.json", result({"epochs": 1}))
    put("experiments/fam/exp_bbb/result.json", result({"data.input_width": 416, "data.input_height": 416}))
    put("experiments/fam/exp_ccc/result.json", result(status="failed"))
    put("evaluations/ev1/metrics.json", {**METRICS, "task": "detection", "per_class": PER_CLASS})
    put("evaluations/ev1/metadata.json", {"dataset_id": "weapons-small", "model_id": "m"})
    put("evaluations/exp_aaa-trained/metrics.json", {**METRICS, "task": "detection", "per_class": PER_CLASS})
    put("evaluations/cls/metrics.json", {"task": "classification"})
    put("failure_analyses/fa1/per_class.json", PER_CLASS)
    put("failure_analyses/fa1/metadata.json", {"task": "detection", "evaluation_id": "ev1"})
    (tmp_path / "experiments/fam/exp_bad").mkdir()
    (tmp_path / "experiments/fam/exp_bad/result.json").write_text("{not json")
    return tmp_path


def test_config_changes_only_reports_differences():
    cfg = TrainingConfig().model_dump(mode="python")
    assert config_changes(cfg) == {}
    cfg["epochs"] = 30
    cfg["data"]["input_width"] = 416
    assert config_changes(cfg) == {"epochs": 30, "data.input_width": 416}


def test_experiment_document_content():
    doc = experiment_document("fam", "exp_x", result({"epochs": 1}))
    assert doc.id == "experiment:fam:exp_x"
    assert "epochs=1" in doc.title
    assert "recall: baseline 0.155, trained 0.310" in doc.content
    assert "knife" in doc.content
    assert doc.metadata["changes"] == {"epochs": 1}


def test_failed_experiment_document():
    doc = experiment_document("fam", "exp_x", result(status="failed"))
    assert "RuntimeError: boom" in doc.content


def test_collect_skips_noise(root):
    docs, skipped = collect_documents(root)
    kinds = sorted(d.source_type for d in docs)
    assert kinds == ["evaluation", "experiment", "experiment", "experiment", "failure_analysis"]
    assert skipped == 3


def test_tried_changes(root):
    tried = tried_changes(root, "fam")
    assert len(tried) == 2
    assert {"epochs": 1} in tried
    assert {"data.input_width": 416, "data.input_height": 416} in tried


def test_store_search_filter_upsert_persist(root, tmp_path):
    store = KnowledgeStore(HashingEmbedder(), tmp_path / "kb")
    info = ingest_all(root, store)
    assert info["documents"] == 5 and info["added"] == 5
    assert ingest_all(root, store)["added"] == 0

    hits = store.search("data.input_width=416 data.input_height=416", top_k=2, source_type="experiment")
    assert hits[0].doc.source_id == "exp_bbb"
    only_fam = store.search("experiment", top_k=10, filters={"family_id": "fam"})
    assert {h.doc.source_type for h in only_fam} == {"experiment"}
    assert store.search("x", filters={"family_id": "nope"}) == []

    reloaded = KnowledgeStore(HashingEmbedder(), tmp_path / "kb")
    assert len(reloaded) == 5
    again = reloaded.search("data.input_width=416 data.input_height=416", top_k=1, source_type="experiment")
    assert again[0].doc.source_id == "exp_bbb"
    assert again[0].score == pytest.approx(hits[0].score)


def test_store_reembeds_when_embedder_changes(root, tmp_path):
    store = KnowledgeStore(HashingEmbedder(), tmp_path / "kb")
    ingest_all(root, store)
    other = KnowledgeStore(HashingEmbedder(dim=128), tmp_path / "kb")
    assert len(other) == 5
    assert other.vecs[next(iter(other.vecs))].shape == (128,)


def test_analysis_separates_absent_classes_from_weak_ones():
    a = analyze_detection(METRICS, PER_CLASS)
    weak = {(w.class_name, w.metric) for w in a.weaknesses}
    assert ("handgun", "recall") in weak
    assert ("knife", "recall") in weak
    assert all(w.class_name != "long_gun" for w in a.weaknesses)
    assert any("long_gun" in i and "no ground-truth" in i for i in a.data_issues)
    assert a.primary_target == "recall"
    assert a.weaknesses[0].class_name == "handgun"
