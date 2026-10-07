import json
import os
import time

import pytest

from ai_client import AiApiError
from modellab.ai.llm.provider import LLMError, NoLLM, ScriptedLLM
from tests.ai_fixtures import BASE_METRICS, FAMILY, GOOD, Env, make_env, put


@pytest.fixture
def env(tmp_path):
    return make_env(tmp_path)


def test_research_creates_and_persists_record(env):
    rec = env.client.research(FAMILY, "Why is my weapon detector weak?")
    assert rec["status"] == "proposed" and rec["job"] is None
    assert rec["control_available"] is False and rec["result"]["ranking"]
    assert rec["defaults"] == {"model_id": "weapons-final-model", "dataset_id": "weapons-small"}
    assert rec["result"]["validation"]["valid"] is True
    assert (env.root / "ai_research" / rec["id"] / "research.json").is_file()
    assert [r["id"] for r in env.client.list_research()] == [rec["id"]]
    assert env.client.get(rec["id"])["result"]["proposal"]["candidate_id"] == "res_416"


def test_research_errors(env):
    with pytest.raises(AiApiError) as e:
        env.client.research("unknown-family", "q")
    assert e.value.status == 422
    with pytest.raises(AiApiError) as e:
        env.client.research("../x", "q")
    assert e.value.status == 422
    put(env.root, "experiments/clsfam/baseline.json", {"task": "classification"})
    with pytest.raises(AiApiError) as e:
        env.client.research("clsfam", "q")
    assert e.value.status == 422 and "detection" in e.value.message
    assert env.http.post("/ai/research", json={"family_id": FAMILY, "question": ""}).status_code == 422
    assert env.http.post("/ai/research", json={"family_id": FAMILY, "extra": 1}).status_code == 422
    assert env.client.list_research() == []


def test_unknown_research_id(env):
    with pytest.raises(AiApiError) as e:
        env.client.get("res_nope")
    assert e.value.status == 404


def test_llm_backend_unavailable_is_503(tmp_path):
    put(tmp_path, f"experiments/{FAMILY}/baseline.json",
        {"task": "detection", "evaluation_id": "e", "metrics": BASE_METRICS, "metadata": {}})
    e = Env(tmp_path, LLMError("model file missing"))
    with pytest.raises(AiApiError) as err:
        e.client.research(FAMILY, "q")
    assert err.value.status == 503 and "model file missing" in err.value.message


def test_no_llm_backend_still_returns_a_valid_proposal(tmp_path):
    put(tmp_path, f"experiments/{FAMILY}/baseline.json",
        {"task": "detection", "evaluation_id": "e", "metrics": BASE_METRICS, "metadata": {}})
    rec = Env(tmp_path, NoLLM()).client.research(FAMILY, "q")
    assert rec["result"]["llm_used"] is False
    assert rec["result"]["validation"]["valid"] is True


def test_approve_runs_control_first_then_exact_validated_spec(env):
    rec = env.client.research(FAMILY, "q")
    out = env.client.approve(rec["id"])
    assert out["status"] == "approved" and out["job"]["job_id"] == "job1"
    assert out["control"]["runs"] is True

    params = env.services.calls[0]
    assert params["family_id"] == FAMILY
    assert len(params["specs"]) == 2
    assert params["specs"][0]["name"] == "advisor_control_default"
    assert params["specs"][1]["name"] == rec["result"]["proposal"]["name"]
    assert params["specs"][1] == rec["result"]["validation"]["spec"]
    assert params["specs"][0]["intervention"]["changes"] == [{"path": "epochs", "value": 10}]
    assert (params["model_id"], params["dataset_id"]) == ("weapons-final-model", "weapons-small")
    assert params["matrices"] == [] and params["baseline_evaluation_id"] is None

    done = env.client.get(rec["id"])
    assert done["status"] == "approved" and done["job"]["status"] == "completed"
    assert done["control_available"] is True
    assert done["finding"]["basis"] == "control" and done["finding"]["verdict"] == "improved"
    assert "0.300 to 0.400" in done["finding"]["summary"]
    assert env.client.list_research()[0]["verdict"] == "improved"


def test_approve_without_control_compares_against_baseline(env):
    rec = env.client.research(FAMILY, "q")
    out = env.client.approve(rec["id"], include_control=False)
    assert out["control"] == {"experiment_id": None, "runs": False}
    assert len(env.services.calls[0]["specs"]) == 1
    finding = env.client.get(rec["id"])["finding"]
    assert finding["basis"] == "baseline" and "0.155 to 0.400" in finding["summary"]
    assert len(finding["caveats"]) == 2


def test_existing_control_is_reused_not_rerun(env):
    first = env.client.research(FAMILY, "q")
    env.client.approve(first["id"])
    control_id = env.client.get(first["id"])["approval"]["control_experiment_id"]
    assert control_id

    second = env.client.research(FAMILY, "q")
    out = env.client.approve(second["id"])
    assert out["control"] == {"experiment_id": control_id, "runs": False}
    assert len(env.services.calls[1]["specs"]) == 1
    assert env.client.get(second["id"])["finding"]["basis"] == "control"


def test_failed_control_falls_back_to_baseline_comparison(env):
    env.services.control_mode = "fail"
    rec = env.client.research(FAMILY, "q")
    env.client.approve(rec["id"])
    finding = env.client.get(rec["id"])["finding"]
    assert finding["basis"] == "baseline" and finding["verdict"] == "improved"


def test_approve_twice_conflicts_unless_forced(env):
    rec = env.client.research(FAMILY, "q")
    env.client.approve(rec["id"])
    with pytest.raises(AiApiError) as e:
        env.client.approve(rec["id"])
    assert e.value.status == 409
    assert len(env.services.calls) == 1
    env.client.approve(rec["id"], force=True)
    assert len(env.services.calls) == 2 and env.services.calls[1]["force"] is True


def test_forced_rerun_ignores_stale_result(env):
    rec = env.client.research(FAMILY, "q")
    env.client.approve(rec["id"])
    exp_id = rec["result"]["validation"]["experiment_id"]
    path = env.root / "experiments" / FAMILY / exp_id / "result.json"
    env.services.mode = "raise"
    env.client.approve(rec["id"], force=True)
    old = time.time() - 3600
    os.utime(path, (old, old))
    assert env.client.get(rec["id"])["finding"] is None
    now = time.time()
    os.utime(path, (now, now))
    assert env.client.get(rec["id"])["finding"]["verdict"] == "improved"


def test_failed_experiment_finding(env):
    env.services.mode = "fail"
    rec = env.client.research(FAMILY, "q")
    env.client.approve(rec["id"])
    done = env.client.get(rec["id"])
    assert done["finding"]["verdict"] == "failed" and "out of memory" in done["finding"]["summary"]


def test_reject_blocks_approval(env):
    rec = env.client.research(FAMILY, "q")
    assert env.client.reject(rec["id"])["status"] == "rejected"
    with pytest.raises(AiApiError) as e:
        env.client.approve(rec["id"])
    assert e.value.status == 409
    with pytest.raises(AiApiError):
        env.client.reject(rec["id"])
    assert env.services.calls == []


def test_unknown_model_or_dataset_rejected_before_any_job(env):
    rec = env.client.research(FAMILY, "q")
    with pytest.raises(AiApiError) as e:
        env.client.approve(rec["id"], model_id="other-model")
    assert e.value.status == 404
    with pytest.raises(AiApiError) as e:
        env.client.approve(rec["id"], dataset_id="other-data")
    assert e.value.status == 404
    assert env.jobs.items == {} and env.client.get(rec["id"])["status"] == "proposed"


def test_tampered_record_is_not_executed(env):
    rec = env.client.research(FAMILY, "q")
    path = env.root / "ai_research" / rec["id"] / "research.json"
    doc = json.loads(path.read_text())
    doc["result"]["proposal"]["changes"] = [{"path": "freeze.mode", "value": "all_but_head"}]
    path.write_text(json.dumps(doc))
    with pytest.raises(AiApiError) as e:
        env.client.approve(rec["id"])
    assert e.value.status == 422
    assert env.services.calls == []

    doc["result"]["proposal"]["changes"] = [{"path": "epochs", "value": 99}]
    path.write_text(json.dumps(doc))
    with pytest.raises(AiApiError) as e:
        env.client.approve(rec["id"])
    assert "no longer validates" in e.value.message
    assert env.services.calls == []


def test_missing_ids_when_baseline_has_none(tmp_path):
    put(tmp_path, f"experiments/{FAMILY}/baseline.json",
        {"task": "detection", "evaluation_id": "e", "metrics": BASE_METRICS, "metadata": {}})
    e = Env(tmp_path, ScriptedLLM([GOOD]))
    rec = e.client.research(FAMILY, "q")
    with pytest.raises(AiApiError) as err:
        e.client.approve(rec["id"])
    assert err.value.status == 422 and "model_id" in err.value.message
    assert e.client.approve(rec["id"], "weapons-final-model", "weapons-small")["status"] == "approved"


def test_empty_menu_gives_no_proposal_record(env, monkeypatch):
    from modellab.ai.research import nodes
    monkeypatch.setattr(nodes, "build_menu", lambda tried: [])
    rec = env.client.research(FAMILY, "q")
    assert rec["status"] == "no_proposal" and rec["result"]["proposal"] is None
    with pytest.raises(AiApiError) as e:
        env.client.approve(rec["id"])
    assert e.value.status == 409


def test_client_sends_ngrok_header(env):
    assert env.http.headers["ngrok-skip-browser-warning"] == "true"
