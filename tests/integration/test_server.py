import io
import json
import zipfile
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("scipy")
pytest.importorskip("sklearn")
pytest.importorskip("pyarrow")

from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from modellab.server import ServerSettings, create_app  # noqa: E402

DUMMY = Path(__file__).resolve().parents[1] / "dummy_models.py"
COLORS = {"RED": (200, 50, 40), "GREEN": (40, 200, 50), "BLUE": (40, 50, 200)}
NAMES = list(COLORS)


def png(color):
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buffer, format="PNG")
    return buffer.getvalue()


def make_zip(per_class=12, bad=False, extra=None):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for cls, color in COLORS.items():
            for i in range(per_class):
                archive.writestr(f"{cls}/{i:03d}.png", png(color))
        archive.writestr(".DS_Store", b"junk")
        archive.writestr("__MACOSX/._RED", b"junk")
        if bad:
            archive.writestr("RED/bad.png", b"not an image")
        for name, data in (extra or {}).items():
            archive.writestr(name, data)
    return buffer.getvalue()


def upload_dataset(client, dataset_id, data):
    return client.post("/datasets", data={"dataset_id": dataset_id}, files={"archive": ("d.zip", data, "application/zip")})


def run(client, path, body, **params):
    response = client.post(path, json=body, params={"wait": True, "timeout": 900, **params})
    assert response.status_code == 200, response.text
    return response.json()


@pytest.fixture
def client(tmp_path):
    app = create_app(ServerSettings(workspace=tmp_path / "ws", token="secret"))
    with TestClient(app, headers={"Authorization": "Bearer secret"}) as c:
        yield c


def test_full_pipeline_through_the_api(client, tmp_path):
    anonymous = TestClient(client.app)
    assert anonymous.get("/health").status_code == 200
    assert anonymous.get("/models").status_code == 401
    system = client.get("/system").json()
    assert system["token_required"] and system["torch"]["version"]

    r = upload_dataset(client, "toy", make_zip())
    assert r.status_code == 201, r.text
    dataset = r.json()
    assert dataset["kind"] == "folder" and dataset["source"] == "upload"
    assert dataset["summary"]["num_samples"] == 36
    assert dataset["summary"]["classes"] == {"BLUE": 12, "GREEN": 12, "RED": 12}
    assert client.get("/datasets/toy/summary").json()["num_excluded"] == 0
    assert upload_dataset(client, "toy", make_zip()).status_code == 409
    assert upload_dataset(client, "notzip", b"hello").status_code == 400
    assert upload_dataset(client, "bad id!", make_zip()).status_code == 422
    slip = upload_dataset(client, "slip", make_zip(extra={"../evil.txt": b"x"}))
    assert slip.status_code == 400 and "unsafe path" in slip.json()["detail"]
    assert client.get("/datasets/slip").status_code == 404 and not (tmp_path / "evil.txt").exists()
    assert upload_dataset(client, "empty", zipfile_without_images()).status_code == 422
    assert [d["dataset_id"] for d in client.get("/datasets").json()] == ["toy"]

    spec = {"source": "factory", "num_classes": 3, "factory": "identity_classifier", "class_names": NAMES}
    factory = {"factory_file": ("factory.py", DUMMY.read_bytes(), "text/x-python")}
    r = client.post("/models", data={"model_id": "toy", "spec": json.dumps(spec)}, files=factory)
    assert r.status_code == 201, r.text
    assert r.json()["check"]["loaded"] and r.json()["check"]["metadata"]["num_classes"] == 3
    assert r.json()["spec"]["factory"].endswith("factory.py:identity_classifier")
    for model_id, bad_spec, bad_files, message in [
        ("m1", {"source": "factory", "num_classes": 3}, None, "requires factory"),
        ("m2", {"source": "state_dict", "num_classes": 3, "factory": "identity_classifier"}, factory, "requires path"),
        ("m3", {"source": "factory", "num_classes": 3, "factory": "a:b"}, factory, "callable name only"),
        ("m4", {"source": "factory", "num_classes": 3, "factory": "nope"}, factory, "no attribute"),
    ]:
        response = client.post("/models", data={"model_id": model_id, "spec": json.dumps(bad_spec)}, files=bad_files)
        assert response.status_code == 422 and message in response.json()["detail"], (model_id, response.text)
        assert client.get(f"/models/{model_id}").status_code == 404
    response = client.post("/models", data={"model_id": "m5", "spec": json.dumps(spec), "preprocess": '{"mean": [0.5, 0.5, 0.5]}'}, files=factory)
    assert response.status_code == 422 and "std" in response.json()["detail"]
    assert client.post("/models", data={"model_id": "toy", "spec": json.dumps(spec)}, files=factory).status_code == 409
    assert client.post("/models", data={"spec": "not json"}).status_code == 422

    job = run(client, "/evaluations", {"model_id": "toy", "dataset_id": "toy", "batch_size": 4, "device": "cpu", "evaluation_id": "eval1"})
    assert job["status"] == "completed", job
    assert job["result"]["num_samples"] == 36 and job["result"]["accuracy"] == pytest.approx(1.0)
    assert "evaluated" in client.get(f"/jobs/{job['job_id']}/log").text
    evaluation = client.get("/evaluations/eval1").json()
    assert evaluation["metrics"]["accuracy"] == 1.0 and evaluation["metadata"]["batch_size"] == 4
    assert evaluation["confusion_matrix"]["labels"] == NAMES
    page = client.get("/evaluations/eval1/predictions", params={"limit": 5}).json()
    assert page["total"] == 36 and len(page["rows"]) == 5 and "sample_id" in page["columns"]
    assert client.get("/evaluations/eval1/predictions", params={"only_errors": True}).json()["total"] == 0
    assert [e["evaluation_id"] for e in client.get("/evaluations").json()] == ["eval1"]
    assert client.post("/evaluations", json={"model_id": "toy", "dataset_id": "toy", "evaluation_id": "eval1"}).status_code == 409
    assert client.post("/evaluations", json={"model_id": "nope", "dataset_id": "toy"}).status_code == 404
    assert client.post("/evaluations", json={"model_id": "toy", "dataset_id": "toy", "batch_size": 0}).status_code == 422
    assert client.post("/evaluations", json={"model_id": "toy", "dataset_id": "toy", "subpath": "../x"}).status_code == 422

    job = run(client, "/audits", {"dataset_id": "toy", "audit_id": "aud1", "config": {"min_side": 4}})
    assert job["status"] == "completed" and job["result"]["num_samples"] == 36
    audit = client.get("/audits/aud1").json()
    assert audit["inventory"]["total_samples"] == 36 and audit["dataset_id"] == "toy"
    assert "ModelLab dataset audit" in client.get("/audits/aud1/summary").text
    assert client.get("/audits/aud1/records", params={"status": "ok"}).json()["total"] == 36
    assert client.post("/audits", json={"dataset_id": "toy", "config": {"nope": 1}}).status_code == 422

    job = run(client, "/analyses", {"evaluation_id": "eval1", "audit_id": "aud1", "analysis_id": "an1", "config": {"min_support": 5}})
    assert job["status"] == "completed" and job["result"]["n_errors"] == 0
    analysis = client.get("/analyses/an1").json()
    assert analysis["inputs"]["join_coverage"] == pytest.approx(1.0) and analysis["overall"]["accuracy"] == 1.0
    slices = client.get("/analyses/an1/slices", params={"tested": True}).json()
    assert slices["total"] > 0
    twelve = next(row for row in slices["rows"] if row["support"] == 12)
    ids = client.get("/analyses/an1/slice_samples", params={"slice_id": twelve["slice_id"]}).json()
    assert ids["count"] == 12 and len(ids["sample_ids"]) == 12
    assert client.get("/analyses/an1/slice_samples", params={"slice_id": "nope=1"}).status_code == 404
    assert client.get("/analyses/an1/samples", params={"is_error": True}).json()["total"] == 0
    assert client.post("/analyses", json={"evaluation_id": "missing"}).status_code == 404

    drop = {
        "name": "drop_red", "hypothesis": {"claim": "the red channel carries the class", "expected_direction": "degrade"},
        "intervention": {"kind": "image_transform", "op": "channel_drop", "params": {"channel": 0}},
    }
    body = {"family_id": "fam", "baseline_evaluation_id": "eval1", "model_id": "toy", "dataset_id": "toy", "device": "cpu", "specs": [drop]}
    job = run(client, "/experiments", body)
    assert job["status"] == "completed", job
    result = job["result"]
    assert result["num_completed"] == 1 and result["outcomes"][0]["status"] == "completed"
    assert result["report"]["experiments"][0]["accuracy_difference"] == pytest.approx(-1 / 3)
    exp_id = result["outcomes"][0]["experiment_id"]
    detail = client.get(f"/experiments/fam/{exp_id}").json()
    assert detail["statistics"]["affected"]["discordant_worsened"] == 12
    assert detail["verdict"]["label"] == "degraded" and detail["verdict"]["matches_expected"] is True
    assert client.get(f"/experiments/fam/{exp_id}/paired", params={"limit": 50}).json()["total"] == 36
    assert client.get(f"/experiments/fam/{exp_id}/paired", params={"changed_only": True}).json()["total"] == 12
    family = client.get("/experiments/fam").json()
    assert family["baseline"]["evaluation_id"] == "eval1" and family["family_report"]["num_completed"] == 1
    assert family["experiments"][0]["verdict"] == "degraded"
    assert [f["family_id"] for f in client.get("/experiments").json()] == ["fam"]
    again = run(client, "/experiments", body)
    assert [o["status"] for o in again["result"]["outcomes"]] == ["cached"]

    matrix = {
        "name": "temp", "hypothesis": {"claim": "temperature does not change accuracy"},
        "intervention": {"kind": "inference", "temperature": 1.0}, "grid": {"temperature": [0.5, 2.0]},
    }
    job = run(client, "/experiments", {"family_id": "fam2", "baseline_evaluation_id": "eval1", "matrices": [matrix]})
    assert job["status"] == "completed" and job["result"]["num_completed"] == 2

    before = len(client.get("/jobs").json())
    bad_op = {**drop, "intervention": {"kind": "image_transform", "op": "nope"}}
    assert client.post("/experiments", json={**body, "specs": [bad_op]}).status_code == 422
    assert client.post("/experiments", json={**body, "model_id": None, "dataset_id": None}).status_code == 422
    assert client.post("/experiments", json={"family_id": "fam3", "specs": [drop]}).status_code == 422
    data_spec = {**drop, "intervention": {"kind": "data", "op": "reweighting"}}
    refused = client.post("/experiments", json={**body, "specs": [data_spec]})
    assert refused.status_code == 422 and "training interface" in refused.json()["detail"]
    assert len(client.get("/jobs").json()) == before

    assert upload_dataset(client, "broken", make_zip(per_class=2, bad=True)).status_code == 201
    failed = run(client, "/evaluations", {"model_id": "toy", "dataset_id": "broken", "device": "cpu", "evaluation_id": "bad_eval"})
    assert failed["status"] == "failed" and failed["error"]["type"] == "DatasetError"
    assert "bad.png" in failed["error"]["message"] and str(tmp_path / "ws") not in json.dumps(failed["error"])
    assert "Traceback" in client.get(f"/jobs/{failed['job_id']}/log").text
    assert client.get("/evaluations/bad_eval").status_code == 404
    audited = run(client, "/audits", {"dataset_id": "broken", "audit_id": "aud_bad", "config": {"min_side": 4}})
    assert audited["status"] == "completed"
    codes = {f["finding_id"] for f in client.get("/audits/aud_bad").json()["findings"]}
    assert "image_integrity.decode_failures" in codes

    listing = client.get("/jobs").json()
    assert {j["kind"] for j in listing} == {"evaluation", "audit", "analysis", "experiments"}
    assert client.get("/jobs", params={"kind": "audit"}).json()[0]["kind"] == "audit"
    assert client.post(f"/jobs/{listing[0]['job_id']}/cancel").status_code == 409
    assert client.get("/jobs/doesnotexist").status_code == 404

    metrics = client.get("/files", params={"path": "evaluations/eval1/metrics.json"}).json()
    assert metrics["accuracy"] == 1.0
    names = {e["name"] for e in client.get("/files", params={"path": "evaluations/eval1"}).json()["entries"]}
    assert {"predictions.parquet", "metrics.json", "metadata.json"} <= names
    download = client.get("/files", params={"path": "evaluations/eval1/predictions.parquet", "download": True})
    assert download.status_code == 200 and len(download.content) > 100
    assert client.get("/files", params={"path": "../models"}).status_code == 400
    assert client.get("/files", params={"path": "nope/x"}).status_code == 404
    assert "Traceback" not in client.get("/files", params={"path": "audits/aud1/summary.txt"}).text

    assert client.delete("/models/toy").status_code == 200
    assert client.post("/evaluations", json={"model_id": "toy", "dataset_id": "toy"}).status_code == 404
    assert client.delete("/datasets/toy").status_code == 200
    assert client.get("/datasets/toy").status_code == 404


def zipfile_without_images():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("notes/readme.txt", "no images here")
    return buffer.getvalue()


def test_limits_and_disabled_local_paths(tmp_path):
    settings = ServerSettings(workspace=tmp_path / "ws", allow_local_paths=False, max_upload_bytes=2000)
    with TestClient(create_app(settings)) as client:
        assert client.get("/models").status_code == 200
        big = upload_dataset(client, "big", make_zip())
        assert big.status_code == 413
        assert client.get("/datasets/big").status_code == 404
        assert client.post("/datasets/register", json={"path": str(tmp_path)}).status_code == 403
        assert client.post("/models/register", json={"spec": {"source": "factory"}}).status_code == 403
        spec = {"source": "state_dict", "num_classes": 3, "factory": "x", "path": "/some/where.pt"}
        assert client.post("/models", data={"spec": json.dumps(spec)}).status_code == 403


def test_local_registration_persistence_and_restart(tmp_path):
    images = tmp_path / "images"
    for cls, color in COLORS.items():
        for i in range(3):
            path = images / cls / f"{i}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(png(color))
    workspace = tmp_path / "ws"
    spec = {"source": "factory", "num_classes": 3, "factory": f"{DUMMY}:identity_classifier", "class_names": NAMES}
    with TestClient(create_app(ServerSettings(workspace=workspace))) as client:
        r = client.post("/datasets/register", json={"dataset_id": "local-toy", "path": str(images)})
        assert r.status_code == 201 and r.json()["source"] == "local" and r.json()["summary"]["num_samples"] == 9
        assert client.post("/datasets/register", json={"path": str(tmp_path / "missing")}).status_code == 422
        r = client.post("/models/register", json={"model_id": "toy-local", "spec": spec})
        assert r.status_code == 201 and r.json()["check"]["loaded"] is True
        assert client.post("/models/register", json={"model_id": "ghost", "spec": {**spec, "factory": f"{DUMMY}:missing"}}).status_code == 422
        job = run(client, "/evaluations", {"model_id": "toy-local", "dataset_id": "local-toy", "device": "cpu"})
        assert job["status"] == "completed" and job["result"]["accuracy"] == 1.0
        evaluation_id = job["result"]["evaluation_id"]
        assert client.delete("/datasets/local-toy").status_code == 200
        assert (images / "RED" / "0.png").is_file()
        assert client.post("/datasets/register", json={"dataset_id": "local-toy", "path": str(images)}).status_code == 201

    with TestClient(create_app(ServerSettings(workspace=workspace))) as client:
        assert [m["model_id"] for m in client.get("/models").json()] == ["toy-local"]
        assert [d["dataset_id"] for d in client.get("/datasets").json()] == ["local-toy"]
        assert client.get(f"/evaluations/{evaluation_id}").json()["metrics"]["accuracy"] == 1.0
        assert client.get(f"/jobs/{job['job_id']}").json()["status"] == "completed"
        again = run(client, "/evaluations", {"model_id": "toy-local", "dataset_id": "local-toy", "device": "cpu"})
        assert again["status"] == "completed"
