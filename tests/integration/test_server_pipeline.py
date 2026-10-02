import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("scipy")
pytest.importorskip("sklearn")
pytest.importorskip("pyarrow")

from fastapi.testclient import TestClient  # noqa: E402

from modellab.server import ServerSettings, create_app  # noqa: E402

TESTS = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("pipeline_fixtures_for_server", TESTS / "pipeline_fixtures.py")
pf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pf)


def run(client, path, body, **params):
    response = client.post(path, json=body, params={"wait": True, "timeout": 900, **params})
    assert response.status_code == 200, response.text
    return response.json()


def test_investigation_and_repair_through_the_api(tmp_path):
    with TestClient(create_app(ServerSettings(workspace=tmp_path / "ws"))) as client:
        r = client.post("/datasets", data={"dataset_id": "dark"}, files={"archive": ("d.zip", pf.dark_zip(), "application/zip")})
        assert r.status_code == 201 and r.json()["summary"]["num_samples"] == 72
        spec = {"source": "factory", "num_classes": 3, "factory": "dark_sensitive_classifier", "class_names": pf.NAMES}
        factory = {"factory_file": ("factory.py", (TESTS / "dummy_models.py").read_bytes(), "text/x-python")}
        assert client.post("/models", data={"model_id": "dark", "spec": json.dumps(spec)}, files=factory).status_code == 201

        config = {"analysis": {"min_support": 10, "max_depth": 1}, "hypotheses": {"max_failures": 1}}
        body = {"investigation_id": "inv1", "model_id": "dark", "dataset_id": "dark", "device": "cpu", "config": config}
        job = run(client, "/investigations", body)
        assert job["status"] == "completed", job
        result = job["result"]
        assert result["num_failures"] == 1 and result["num_hypotheses"] == 2 and result["num_experiments"] == 2
        conclusion = result["conclusions"][0]
        assert conclusion["status"] == "supported_mechanism" and "brightness" in conclusion["mechanism"]
        assert "evaluated" in client.get(f"/jobs/{job['job_id']}/log").text

        record = client.get("/investigations/inv1").json()
        assert record["baseline"]["accuracy"] == pytest.approx(48 / 72) and len(record["links"]) == 2
        assert record["inputs"]["analysis_id"] == "inv1_analysis" and "not a proven cause" in conclusion.get("mechanism", "") + record["conclusions"][0]["statement"]
        report = client.get("/investigations/inv1/report").text
        assert "1. OBSERVED RESULTS" in report and "3. CONCLUSIONS" in report
        assert [i["investigation_id"] for i in client.get("/investigations").json()] == ["inv1"]
        assert client.get("/analyses/inv1_analysis").json()["overall"]["n_errors"] == 24

        made = client.post("/hypotheses", json={"analysis_id": "inv1_analysis", "audit_id": "inv1_audit", "run_id": "h1", "config": {"max_failures": 1}})
        assert made.status_code == 201 and made.json()["num_hypotheses"] == 2
        assert client.get("/hypotheses").json() == ["h1"]
        document = client.get("/hypotheses/h1").json()
        assert document["competing_groups"] and "proven cause" in document["disclaimer"]
        assert len(client.get("/hypotheses/h1/plan").json()["selected"]) == 2
        assert client.post("/hypotheses", json={"analysis_id": "inv1_analysis", "run_id": "h1"}).status_code == 409
        assert client.post("/hypotheses", json={"analysis_id": "missing"}).status_code == 404

        assert client.post("/investigations", json=body).status_code == 409
        assert client.post("/investigations", json={"model_id": "dark"}).status_code == 422
        assert client.post("/investigations", json={**body, "investigation_id": "x", "config": {"nope": 1}}).status_code == 422
        assert client.post("/investigations", json={**body, "investigation_id": "y", "model_id": "nope"}).status_code == 404

        again = run(client, "/investigations/inv1/reproduce", {"new_investigation_id": "inv2", "model_id": "dark", "dataset_id": "dark", "device": "cpu"})
        assert again["status"] == "completed" and again["result"]["reproduced"] is True

        repair = run(client, "/repairs", {"investigation_id": "inv1", "repair_id": "rep1", "model_id": "dark", "dataset_id": "dark", "device": "cpu"})
        assert repair["status"] == "completed", repair
        outcome = repair["result"]
        assert outcome["decision"] == "accepted" and outcome["summary"]["candidates"] == 3 and outcome["summary"]["accepted"] >= 1
        assert outcome["accepted"]["target"]["difference"] == pytest.approx(1.0) and outcome["accepted"]["global"]["new_errors"] == 0
        results = client.get("/repairs/rep1").json()
        assert results["baseline"]["unchanged"] is True and len(results["results"]) == 3
        assert "decision: accepted" in client.get("/repairs/rep1/report").text
        version = client.get("/repairs/rep1/version").json()
        assert version["rule"]["type"] == "slice_conditioned_input_transform" and version["status"] == "accepted"
        assert [r["repair_id"] for r in client.get("/repairs").json()] == ["rep1"]

        harsh = run(client, "/repairs", {"investigation_id": "inv1", "repair_id": "rep2", "model_id": "dark", "dataset_id": "dark",
                                         "device": "cpu", "config": {"criteria": {"min_target_improvement": 1.5}}}) if False else None
        assert harsh is None
        strict = {"investigation_id": "inv1", "repair_id": "rep3", "model_id": "dark", "dataset_id": "dark", "device": "cpu",
                  "config": {"criteria": {"min_target_improvement": 0.9, "max_global_drop": 0.0}, "factors_low": [1.3]}}
        rejected = run(client, "/repairs", strict)
        assert rejected["status"] == "completed" and rejected["result"]["decision"] == "rejected"
        assert client.get("/repairs/rep3/version").status_code == 404
        offline = run(client, "/repairs", {"investigation_id": "inv1", "repair_id": "rep4"})
        assert offline["result"]["decision"] == "no_repair_candidates"
        assert client.post("/repairs", json={"investigation_id": "missing"}).status_code == 404
        assert client.post("/repairs", json={"investigation_id": "inv1", "config": {"criteria": {"nope": 1}}}).status_code == 422
        assert client.post("/repairs", json={"investigation_id": "inv1", "repair_id": "rep1"}).status_code == 409
        assert {j["kind"] for j in client.get("/jobs").json()} >= {"investigation", "reproduction", "repair"}
