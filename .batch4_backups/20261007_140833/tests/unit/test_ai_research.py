import json

import pytest

from modellab.ai import tools
from modellab.ai.knowledge.store import HashingEmbedder, KnowledgeStore
from modellab.ai.llm.provider import ScriptedLLM
from modellab.ai.research import nodes
from modellab.ai.research.graph import run_research
from modellab.experiments.training_config import TrainingConfig

FAMILY = "weapons-final-model"
BASE_METRICS = {
    "task": "detection", "f1": 0.16, "precision": 0.165, "recall": 0.155, "num_samples": 472,
    "per_class": {
        "handgun": {"class_id": 0, "f1": 0.01, "fn": 77, "fp": 81, "precision": 0.01, "recall": 0.013, "tp": 1},
        "knife": {"class_id": 2, "f1": 0.22, "fn": 190, "fp": 145, "precision": 0.25, "recall": 0.2, "tp": 48},
        "sword": {"class_id": 3, "f1": 0.0, "fn": 0, "fp": 9, "precision": 0.0, "recall": 0.0, "tp": 0},
    },
}
GOOD = json.dumps({"candidate_id": "res_416", "claim": "Higher resolution should raise handgun recall",
                   "rationale": "small objects", "confidence": 0.7})
EXPECTED_TRACE = ["collect_context", "analyze_problem", "retrieve_knowledge",
                  "generate_hypothesis", "generate_experiment", "validate_experiment"]


def put(root, rel, data):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data))


def exp_result(changes):
    cfg = TrainingConfig().model_dump(mode="python")
    for path, value in changes.items():
        node = cfg
        parts = path.split(".")
        for part in parts[:-1]:
            node = node[part]
        node[parts[-1]] = value
    return {
        "status": "completed", "task": "detection",
        "training": {"config": cfg, "total_epochs": cfg["epochs"], "best_epoch": 1},
        "comparison": {"available": True,
                       "baseline": {"f1": 0.16, "precision": 0.165, "recall": 0.155, "mean_iou": 0.75},
                       "trained": {"f1": 0.27, "precision": 0.24, "recall": 0.31, "mean_iou": 0.91},
                       "delta": {"f1": 0.11, "precision": 0.075, "recall": 0.155, "mean_iou": 0.16}},
    }


@pytest.fixture
def root(tmp_path):
    put(tmp_path, f"experiments/{FAMILY}/baseline.json",
        {"family_id": FAMILY, "task": "detection", "evaluation_id": "ev1", "metrics": BASE_METRICS})
    put(tmp_path, f"experiments/{FAMILY}/exp_aaa/result.json", exp_result({"epochs": 1}))
    return tmp_path


def store(root):
    return KnowledgeStore(HashingEmbedder(), root / "knowledge")


def run(root, replies, question="Why is my weapon detector weak?"):
    llm = ScriptedLLM(replies)
    return run_research(root, FAMILY, question, llm, store(root)), llm


def test_full_flow(root):
    out, llm = run(root, [GOOD])
    assert out["trace"] == EXPECTED_TRACE
    assert out["llm_used"] is True
    assert out["validation"]["valid"] is True
    assert out["validation"]["experiment_id"].startswith("exp_")
    assert out["proposal"]["candidate_id"] == "res_416"
    assert {c["path"] for c in out["proposal"]["changes"]} == {"data.input_width", "data.input_height"}
    assert out["hypothesis"]["target_metric"] == "recall"
    assert out["retrieved"] and out["retrieved"][0]["id"] == f"experiment:{FAMILY}:exp_aaa"
    assert any("recall: baseline" in e for e in out["hypothesis"]["evidence"])
    prompt = llm.calls[0][1]["content"]
    assert "res_416" in prompt and "handgun" in prompt and "Why is my weapon detector weak?" in prompt


def test_analysis_flags_absent_class(root):
    out, _ = run(root, [GOOD])
    assert any("sword" in i for i in out["analysis"]["data_issues"])
    assert all(w["class_name"] != "sword" for w in out["analysis"]["weaknesses"])
    assert out["failure_source"] == "baseline_metrics"


def test_prefers_failure_analysis_artifact(root):
    put(root, "failure_analyses/20261005T1_x/metadata.json", {"evaluation_id": "ev1", "task": "detection"})
    put(root, "failure_analyses/20261005T1_x/per_class.json",
        {"0": {"class": "handgun", "class_id": 0, "support": 78, "true_positives": 1,
               "false_positives": 81, "false_negatives": 77, "precision": 0.01, "recall": 0.013, "f1": 0.01}})
    out, _ = run(root, [GOOD])
    assert out["failure_source"] == "failure_analysis:20261005T1_x"


def test_unknown_candidate_is_retried(root):
    bad = json.dumps({"candidate_id": "made_up", "claim": "c", "rationale": "r", "confidence": 0.5})
    out, llm = run(root, [bad, GOOD])
    assert len(llm.calls) == 2
    assert "made_up" not in out["proposal"]["candidate_id"]
    assert out["llm_used"] is True


def test_llm_failure_falls_back_to_rule(root):
    out, _ = run(root, ["junk", "junk", "junk"])
    assert out["llm_used"] is False
    assert out["trace"] == EXPECTED_TRACE
    assert out["validation"]["valid"] is True
    assert out["hypothesis"]["confidence"] == 0.3
    assert any("LLM output unusable" in n for n in out["notes"])
    assert out["proposal"]["candidate_id"] == "res_416"


def test_tried_experiment_is_not_proposed_again(root):
    put(root, f"experiments/{FAMILY}/exp_bbb/result.json",
        exp_result({"data.input_width": 416, "data.input_height": 416}))
    out, llm = run(root, [GOOD, json.dumps({"candidate_id": "res_512", "claim": "c", "rationale": "r", "confidence": 0.6})])
    assert out["proposal"]["candidate_id"] == "res_512"
    assert "res_416" not in llm.calls[0][1]["content"].split("Allowed experiments:")[1]


def test_empty_menu_stops_early(root, monkeypatch):
    monkeypatch.setattr(nodes, "build_menu", lambda tried: [])
    out, llm = run(root, [GOOD])
    assert out["trace"] == ["collect_context"]
    assert out["proposal"] is None and llm.calls == []
    assert any("already been run" in n for n in out["notes"])


def test_missing_and_non_detection_baselines(root, tmp_path):
    with pytest.raises(tools.ToolError):
        run_research(root, "nope", "q", ScriptedLLM([]), store(root))
    put(root, "experiments/clsfam/baseline.json", {"task": "classification"})
    with pytest.raises(tools.ToolError):
        run_research(root, "clsfam", "q", ScriptedLLM([]), store(root))


@pytest.mark.parametrize("bad", ["../etc", "a/b", "", ".hidden", "a b"])
def test_family_id_is_sanitised(root, bad):
    with pytest.raises(tools.ToolError):
        tools.get_baseline(root, bad)
    with pytest.raises(tools.ToolError):
        tools.list_experiments(root, bad)


def test_list_experiments_summary(root):
    rows = tools.list_experiments(root, FAMILY)
    assert rows == [{"experiment_id": "exp_aaa", "status": "completed", "changes": {"epochs": 1},
                     "delta": {"f1": 0.11, "precision": 0.075, "recall": 0.155}}]
